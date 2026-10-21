"""训练/验证/测试按时间隔离的 DeepAR 价格训练入口。

CSV: ts,price[,series_id]。price 为不含输配基金的电能量价；不隐式换汇、
插值或补点。每个 series_id 可表示市场、节点或站点，所有序列共享 LSTM。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.price_deepar import (  # noqa: E402
    FEATURE_NAMES, FORMAT, PriceDeepAR, calendar_features, gaussian_nll,
    require_torch, sample_paths, validate_series,
)

from app.domain.forecast.price_data import read_series, split_origins

require_torch()
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def make_dataset(series, origins, context, horizon):
    previous, features, targets, ids = [], [], [], []
    for name, index, origin in origins:
        values = series[name][origin - context:origin + horizon]
        y = torch.tensor([v for _, v in values], dtype=torch.float32)
        scale = 1 + y[:context].abs().mean()
        z = y / scale
        previous.append(torch.cat([z[:1], z[:-1]]))
        features.append([calendar_features(ts) for ts, _ in values])
        targets.append(z)
        ids.append(index)
    return TensorDataset(torch.stack(previous), torch.tensor(features), torch.stack(targets), torch.tensor(ids))


def evaluate(model, series, origins, context, horizon, num_samples, seed):
    predictions, actual, baseline_day, baseline_week, rows = [], [], [], [], []
    for name, index, origin in origins:
        values = series[name]
        target = values[origin:origin + horizon]
        paths = sample_paths(model, values[origin - context:origin], [ts for ts, _ in target],
                             series_index=index, num_samples=num_samples, seed=seed)
        q = torch.quantile(paths, torch.tensor([0.1, 0.5, 0.9]), dim=0)
        day_steps = 1440 // int((values[1][0].timestamp() - values[0][0].timestamp()) / 60)
        # 所有基线只引用原点前数据；多日时按最后一日/周循环。
        d = [values[origin - day_steps + j % day_steps][1] for j in range(horizon)]
        w = [values[origin - 7 * day_steps + j % (7 * day_steps)][1] for j in range(horizon)]
        truth = [v for _, v in target]
        predictions.append(q)
        actual.extend(truth)
        baseline_day.extend(d)
        baseline_week.extend(w)
        for j, (ts, value) in enumerate(target):
            rows.append({"series_id": name, "origin": values[origin][0].isoformat(),
                         "ts": ts.isoformat(), "actual": value,
                         "p10": float(q[0, j]), "p50": float(q[1, j]), "p90": float(q[2, j]),
                         "seasonal_day": d[j], "seasonal_week": w[j]})
    q = torch.cat(predictions, dim=1).double()
    y = torch.tensor(actual, dtype=torch.float64)

    def errors(pred):
        residual = torch.as_tensor(pred, dtype=torch.float64) - y
        return {"MAE": float(residual.abs().mean()), "RMSE": float(residual.square().mean().sqrt())}

    metrics = {"DeepAR": errors(q[1]), "seasonal_day": errors(baseline_day),
               "seasonal_week": errors(baseline_week), "points": len(actual), "origins": len(origins)}
    for i, alpha in enumerate((0.1, 0.5, 0.9)):
        e = y - q[i]
        metrics["DeepAR"][f"pinball_{alpha}"] = float(torch.maximum(alpha * e, (alpha - 1) * e).mean())
    metrics["DeepAR"]["coverage_80"] = float(((y >= q[0]) & (y <= q[2])).double().mean())
    metrics["DeepAR"]["mean_interval_width"] = float((q[2] - q[0]).mean())
    return metrics, rows


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.context_days < 7:
        raise ValueError("epochs/batch_size/lr must be positive and context_days >= 7")
    if not 1 <= args.horizon_days <= 3 or not 20 <= args.num_samples <= 2000:
        raise ValueError("horizon_days must be 1..3; num_samples must be 20..2000")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    series = read_series(args.csv, args.timezone, args.resolution_minutes, args.series_id, args.skip_gapped_windows)
    steps = 1440 // args.resolution_minutes
    context, horizon = args.context_days * steps, args.horizon_days * steps
    groups, train_cut, val_cut = split_origins(series, context, horizon, steps)
    train_data = make_dataset(series, groups["train"], context, horizon)
    val_data = make_dataset(series, groups["validation"], context, horizon)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    architecture = {"num_series": len(series), "hidden_size": args.hidden_size,
                    "num_layers": 2, "embedding_size": 4}
    model = PriceDeepAR(**architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_state, best_epoch = math.inf, None, 0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for previous, features, target, ids in train_loader:
            optimizer.zero_grad()
            mu, sigma, _ = model(previous, features, ids)
            loss = gaussian_nll(mu[:, context:], sigma[:, context:], target[:, context:])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for previous, features, target, ids in val_loader:
                mu, sigma, _ = model(previous, features, ids)
                val_total += float(gaussian_nll(mu[:, context:], sigma[:, context:], target[:, context:])) * len(ids)
        val_loss = val_total / len(val_data)
        log.append({"epoch": epoch, "train_scaled_nll": total / count, "validation_scaled_nll": val_loss})
        print(json.dumps(log[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selection_end = max(series[name][origin + horizon - 1][0]
                        for name, _, origin in groups["validation"])
    metadata = {"format": FORMAT, "paper": "https://arxiv.org/abs/1704.04110",
                "implementation": "independent_price_adaptation", "likelihood": "Gaussian",
                "architecture": architecture, "feature_names": FEATURE_NAMES,
                "context_length": context, "training_horizon": horizon,
                "resolution_minutes": args.resolution_minutes, "timezone": args.timezone,
                "unit": args.unit, "target": "energy_price_excluding_transmission_and_fund",
                "series_ids": list(series), "selection_end": selection_end.isoformat(),
                "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "seed": args.seed, "best_epoch": best_epoch,
                "gaps": {"policy": "skip_crossing_windows" if args.skip_gapped_windows else "reject",
                         "source_gap_count": sum(sum(b[0].timestamp() - a[0].timestamp() != args.resolution_minutes * 60
                                                     for a, b in zip(values, values[1:]))
                                                 for values in series.values())},
                "scaling": "1 + mean(abs(context)); no future targets",
                "split": {"train_before": train_cut.isoformat(), "validation_before": val_cut.isoformat(),
                          "target_windows": {key: len(val) for key, val in groups.items()}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, args.output)
    # 只有选定 checkpoint 后才读取测试目标做最终评估，测试分数不参与选择。
    metrics, rows = evaluate(model, series, groups["test"], context, horizon, args.num_samples, args.seed)
    report = {"metadata": metadata, "training_log": log, "test_metrics": metrics,
              "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "evaluation": "rolling origins, observed past only; native frequency and source unit; p50 point forecast",
              "warning": "Independent DeepAR adaptation; test results do not establish Chinese site price accuracy"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit", required=True, help="CNY/kWh for online; original source unit for offline benchmarks")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--resolution-minutes", type=int, choices=(15, 60), default=15)
    p.add_argument("--series-id", default="default")
    p.add_argument("--context-days", type=int, default=7)
    p.add_argument("--horizon-days", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--skip-gapped-windows", action="store_true", help="跳过跨缺口的窗口，保留原始时间和值，不插值")
    train(p.parse_args())


if __name__ == "__main__":
    main()
