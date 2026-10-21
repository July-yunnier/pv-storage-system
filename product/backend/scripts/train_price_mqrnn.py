"""MQ-RNN 电价训练：共享编码的 forking-sequences、边界标签遮罩、时间隔离。

CSV: ts,price[,series_id]。仅电能量价，原生15/60分钟，有符号价格。
当前只使用日历已知未来特征，不声称复现论文的完整 GEFCom 协变量实验。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.price_mqrnn import (
    FEATURE_NAMES, FORMAT, QUANTILES, TARGET, PriceMQRNN, calendar_features,
    direct_quantiles, observed_features, pinball_loss, require_torch,
)
from app.domain.forecast.price_data import read_series, split_origins

require_torch()
import torch
from torch.utils.data import Dataset, DataLoader


def contiguous_segments(values, resolution):
    start = 0
    for i in range(1, len(values)):
        if values[i][0].timestamp() - values[i - 1][0].timestamp() != resolution * 60:
            yield values[start:i]
            start = i
    yield values[start:]


class ForkDataset(Dataset):
    """一个样本是序列块；编码块内每个时刻都接同一解码器，而非切成独立原点。

    连续段预先截于训练边界，目标不会读取验证/测试数据。末端部分可用标签
    保留；填充的输入时刻和边界外标签不参与 loss。块间重置状态（截断 BPTT）。
    """
    def __init__(self, series, scales, cutoff, resolution, context, horizon):
        self.context, self.horizon = context, horizon
        self.segments, self.blocks = [], []
        lag = 7 * 1440 // resolution
        for index, (name, values) in enumerate(series.items()):
            past = [(ts, y) for ts, y in values if ts < cutoff]
            for segment in contiguous_segments(past, resolution):
                if len(segment) <= lag + 1:
                    continue
                x = torch.tensor(observed_features(segment, resolution, scales[name]), dtype=torch.float32)
                y = torch.tensor([v / scales[name] for _, v in segment[lag:]], dtype=torch.float32)
                times = [ts for ts, _ in segment[lag:]]
                times += [times[-1] + timedelta(minutes=resolution * (j + 1)) for j in range(horizon)]
                f = torch.tensor([calendar_features(ts) for ts in times], dtype=torch.float32)
                sid = len(self.segments)
                self.segments.append((x, y, f, index))
                self.blocks.extend((sid, start) for start in range(0, len(y) - 1, context))
        if not self.blocks:
            raise ValueError("not enough continuous training history for weekly lags and forks")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        sid, start = self.blocks[item]
        x, y, features, index = self.segments[sid]
        positions = torch.arange(start, start + self.context)
        valid_position = positions < len(y) - 1
        safe = positions.clamp_max(len(y) - 1)
        observed = x[safe]
        target_idx = positions[:, None] + torch.arange(1, self.horizon + 1)[None, :]
        mask = valid_position[:, None] & (target_idx < len(y))
        target = y[target_idx.clamp_max(len(y) - 1)]
        # 日历可事先计算；真实价格只经 mask 内的标签参与损失。
        future = features[target_idx.clamp_max(len(features) - 1)]
        return observed, future, target, mask, torch.tensor(index)


def evaluate(model, series, origins, context, horizon, resolution, scales):
    predictions, actual, baseline_day, baseline_week, rows = [], [], [], [], []
    day = 1440 // resolution
    for name, index, origin in origins:
        values = series[name]
        q = direct_quantiles(model, values[:origin], resolution=resolution, context=context,
                             scale=scales[name], series_index=index)
        truth = [v for _, v in values[origin:origin + horizon]]
        d = [values[origin - day + j % day][1] for j in range(horizon)]
        w = [values[origin - 7 * day + j % (7 * day)][1] for j in range(horizon)]
        predictions.append(q)
        actual.extend(truth)
        baseline_day.extend(d)
        baseline_week.extend(w)
        for j, (ts, value) in enumerate(values[origin:origin + horizon]):
            rows.append({"series_id": name, "origin": values[origin][0].isoformat(),
                         "ts": ts.isoformat(), "actual": value,
                         "p10": float(q[j, 0]), "p50": float(q[j, 1]), "p90": float(q[j, 2]),
                         "seasonal_day": d[j], "seasonal_week": w[j]})
    q = torch.cat(predictions).double()
    y = torch.tensor(actual, dtype=torch.float64)
    def errors(pred):
        residual = torch.as_tensor(pred, dtype=torch.float64) - y
        return {"MAE": float(residual.abs().mean()), "RMSE": float(residual.square().mean().sqrt())}
    metrics = {"MQ-RNN": errors(q[:, 1]), "seasonal_day": errors(baseline_day),
               "seasonal_week": errors(baseline_week), "points": len(actual), "origins": len(origins)}
    for j, alpha in enumerate(QUANTILES):
        e = y - q[:, j]
        metrics["MQ-RNN"][f"pinball_{alpha}"] = float(torch.maximum(alpha * e, (alpha - 1) * e).mean())
    metrics["MQ-RNN"]["mean_pinball"] = float(pinball_loss(q, y))
    metrics["MQ-RNN"]["coverage_80"] = float(((y >= q[:, 0]) & (y <= q[:, 2])).double().mean())
    metrics["MQ-RNN"]["mean_interval_width"] = float((q[:, 2] - q[:, 0]).mean())
    return metrics, rows


def validation_loss(model, series, origins, context, resolution, scales):
    total, count = 0.0, 0
    for name, index, origin in origins:
        q = direct_quantiles(model, series[name][:origin], resolution=resolution, context=context,
                             scale=scales[name], series_index=index)
        y = torch.tensor([v for _, v in series[name][origin:origin + model.horizon]])
        total += float(pinball_loss(q / scales[name], y / scales[name]))
        count += 1
    return total / count


def train(args):
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.context_days < 1
            or not 1 <= args.horizon_days <= 4):
        raise ValueError("positive epochs/batch_size/lr/context_days required; horizon_days must be 1..4")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    series = read_series(args.csv, args.timezone, args.resolution_minutes, args.series_id, args.skip_gapped_windows)
    steps = 1440 // args.resolution_minutes
    context, horizon = args.context_days * steps, args.horizon_days * steps
    required = context + 7 * steps
    groups, train_cut, val_cut = split_origins(series, required, horizon, steps)
    # 固定尺度只拟合训练段，无未来观测或验证/测试标签参与。
    scales = {name: 1 + sum(abs(v) for ts, v in values if ts < train_cut) /
              sum(ts < train_cut for ts, _ in values) for name, values in series.items()}
    dataset = ForkDataset(series, scales, train_cut, args.resolution_minutes, context, horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    architecture = {"horizon": horizon, "num_series": len(series), "hidden_size": args.hidden_size,
                    "context_size": 8, "decoder_size": 64, "embedding_size": 4}
    model = PriceMQRNN(**architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_state, best_epoch = math.inf, None, 0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, future, target, mask, ids in loader:
            optimizer.zero_grad()
            prediction = model(x, future, ids)
            loss = pinball_loss(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            n = int(mask.sum())
            total += float(loss.detach()) * n
            count += n
        model.eval()
        val_loss = validation_loss(model, series, groups["validation"], context, args.resolution_minutes, scales)
        log.append({"epoch": epoch, "train_scaled_pinball": total / count,
                    "validation_scaled_pinball": val_loss})
        print(json.dumps(log[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selection_end = max(series[n][o + horizon - 1][0] for n, _, o in groups["validation"])
    metadata = {"format": FORMAT, "paper": "https://arxiv.org/abs/1711.11053",
                "paper_title": "A Multi-Horizon Quantile Recurrent Forecaster",
                "implementation": "independent_MQ-RNN_price_adaptation",
                "architecture": architecture, "feature_names": FEATURE_NAMES, "quantiles": QUANTILES,
                "context_length": context, "lag_steps": [steps, 7 * steps], "scales": scales,
                "resolution_minutes": args.resolution_minutes, "timezone": args.timezone,
                "unit": args.unit, "target": TARGET, "series_ids": list(series),
                "selection_end": selection_end.isoformat(),
                "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "seed": args.seed, "best_epoch": best_epoch,
                "training_scheme": "forking_sequences; all encoder positions; shared global/local decoder; truncated BPTT",
                "boundary_policy": "mask unavailable targets after train cutoff or contiguous-segment end",
                "quantile_head": "median +/- softplus gaps (engineering extension)",
                "scaling": "per-series 1+mean(abs(training prices)); fixed at inference",
                "gaps": {"policy": "split_contiguous_segments" if args.skip_gapped_windows else "reject"},
                "split": {"train_before": train_cut.isoformat(), "validation_before": val_cut.isoformat(),
                          "training_sequence_blocks": len(dataset),
                          "evaluation_windows": {k: len(v) for k, v in groups.items() if k != "train"}}}
    paper = Path(__file__).resolve().parents[3] / "1711.11053v2.pdf"
    if paper.is_file():
        metadata["paper_pdf_sha256"] = hashlib.sha256(paper.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, args.output)
    # 测试目标仅在验证选型完成后评估一次，不参与调参。
    metrics, rows = evaluate(model, series, groups["test"], context, horizon, args.resolution_minutes, scales)
    report = {"metadata": metadata, "training_log": log, "test_metrics": metrics,
              "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "environment": {"python": sys.version, "torch": str(torch.__version__)},
              "evaluation": "chronological 70/15/15; disjoint complete target windows; observed past only",
              "warning": "Calendar-only adaptation, not reproduction of competition score; source unit retained; not Chinese-site validation"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit", required=True, help="CNY/kWh for online; source unit for offline benchmarks")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--resolution-minutes", type=int, choices=(15, 60), default=15)
    p.add_argument("--series-id", default="default")
    p.add_argument("--context-days", type=int, default=7)
    p.add_argument("--horizon-days", type=int, default=4, help="4 days covers tomorrow's 72h plus today's bridge")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--skip-gapped-windows", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
