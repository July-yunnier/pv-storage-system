"""训练 MES/天气增强负荷模型并导出 ``LOAD_MODEL_PATH`` artifact。

输入 CSV 必须是连续 15 分钟数据，至少包含：
``ts,total_load_kw,mes_plan``。可选列：``workshop_load_kw,temperature_c,humidity_pct``。
目标日天气在真实部署中应来自天气预报；用历史实况做回放时必须在报告中注明口径。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.domain.forecast.load_advanced import (  # noqa: E402
    EliteEnergyTaskNet,
    multi_task_loss,
    require_torch,
)

require_torch()
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

SLOTS_PER_DAY = 96


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="15分钟负荷/MES/天气训练数据")
    parser.add_argument("--output", required=True, help="输出 .pt 模型 artifact")
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _cyclic(slot: int) -> tuple[float, float]:
    angle = 2 * math.pi * slot / SLOTS_PER_DAY
    return math.sin(angle), math.cos(angle)


def _load_frame(path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path)
    required = {"ts", "total_load_kw", "mes_plan"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段：{sorted(missing)}")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Shanghai")
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("训练数据不足")
    gaps = frame["ts"].diff().dropna()
    if not (gaps == pd.Timedelta(minutes=15)).all():
        raise ValueError("训练数据必须是无缺口、无重复的连续15分钟序列")
    if len(frame) % SLOTS_PER_DAY:
        raise ValueError("训练数据必须按完整天提供，每天96点")
    if (frame["total_load_kw"] < 0).any():
        raise ValueError("total_load_kw 不能为负")

    if "temperature_c" not in frame:
        frame["temperature_c"] = 20.0
    if "humidity_pct" not in frame:
        frame["humidity_pct"] = 50.0
    frame["is_workday"] = (frame["ts"].dt.dayofweek < 5).astype(float)
    slots = frame["ts"].dt.hour * 4 + frame["ts"].dt.minute // 15
    cyclic = [_cyclic(int(slot)) for slot in slots]
    frame["slot_sin"] = [item[0] for item in cyclic]
    frame["slot_cos"] = [item[1] for item in cyclic]

    # 0 保留给新工厂或未知排产；其余类别按训练集稳定映射。
    raw_labels = sorted(str(value) for value in frame["mes_plan"].dropna().unique())
    mapping = {label: index + 1 for index, label in enumerate(raw_labels)}
    frame["mes_id"] = [mapping.get(str(value), 0) for value in frame["mes_plan"]]
    return frame, mapping


def _samples(frame: pd.DataFrame, history_days: int):
    history_slots = history_days * SLOTS_PER_DAY
    hist_columns = [
        "total_load_kw", "temperature_c", "humidity_pct",
        "slot_sin", "slot_cos", "is_workday",
    ]
    target_columns = [
        "temperature_c", "humidity_pct", "slot_sin", "slot_cos", "is_workday",
    ]
    output = []
    for start in range(history_slots, len(frame) - SLOTS_PER_DAY + 1, SLOTS_PER_DAY):
        history = frame.iloc[start - history_slots : start]
        target = frame.iloc[start : start + SLOTS_PER_DAY]
        workshop = (
            target["workshop_load_kw"].to_numpy(dtype="float32")
            if "workshop_load_kw" in target else None
        )
        output.append((
            history[hist_columns].to_numpy(dtype="float32"),
            history["mes_id"].to_numpy(dtype="int64"),
            target[target_columns].to_numpy(dtype="float32"),
            target["mes_id"].to_numpy(dtype="int64"),
            target["total_load_kw"].to_numpy(dtype="float32"),
            workshop,
        ))
    if len(output) < 5:
        raise ValueError("至少需要 history_days + 5 个完整日，才能做时间顺序训练/验证")
    return output, hist_columns, target_columns


def _mean_std(values: torch.Tensor, dimensions: tuple[int, ...]):
    mean = values.mean(dim=dimensions)
    std = values.std(dim=dimensions).clamp_min(1e-6)
    return mean, std


def main() -> None:
    args = _arguments()
    torch.manual_seed(args.seed)
    frame, mes_mapping = _load_frame(args.csv)
    samples, hist_columns, target_columns = _samples(frame, args.history_days)
    split = max(1, int(len(samples) * 0.8))
    if split == len(samples):
        split -= 1
    train_samples, validation_samples = samples[:split], samples[split:]

    hist = torch.tensor([sample[0] for sample in train_samples])
    hist_mes = torch.tensor([sample[1] for sample in train_samples], dtype=torch.long)
    future = torch.tensor([sample[2] for sample in train_samples])
    future_mes = torch.tensor([sample[3] for sample in train_samples], dtype=torch.long)
    total = torch.tensor([sample[4] for sample in train_samples])
    has_workshop = all(sample[5] is not None for sample in samples)
    workshop = torch.tensor([sample[5] for sample in train_samples]) if has_workshop else None

    hist_mean, hist_std = _mean_std(hist, (0, 1))
    future_mean, future_std = _mean_std(future, (0, 1))
    output_mean, output_std = total.mean(), total.std().clamp_min(1e-6)
    hist = (hist - hist_mean) / hist_std
    future = (future - future_mean) / future_std
    total = (total - output_mean) / output_std
    if workshop is not None:
        workshop = (workshop - output_mean) / output_std

    dataset_tensors = [hist, hist_mes, future, future_mes, total]
    if workshop is not None:
        dataset_tensors.append(workshop)
    loader = DataLoader(TensorDataset(*dataset_tensors), batch_size=args.batch_size, shuffle=True)

    model = EliteEnergyTaskNet(
        history_feature_dim=len(hist_columns),
        target_feature_dim=len(target_columns),
        num_mes_plans=len(mes_mapping) + 1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            hist_x, hist_m, future_x, future_m, y_total = batch[:5]
            y_workshop = batch[5] if has_workshop else None
            optimizer.zero_grad()
            pred_total, pred_workshop, _ = model(hist_x, hist_m, future_x, future_m)
            loss = multi_task_loss(pred_total, pred_workshop, y_total, y_workshop)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        print(f"epoch={epoch + 1}/{args.epochs} loss={total_loss / len(loader):.6f}")

    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for sample in validation_samples:
            h = (torch.tensor(sample[0]).unsqueeze(0) - hist_mean) / hist_std
            f = (torch.tensor(sample[2]).unsqueeze(0) - future_mean) / future_std
            pred, _, _ = model(
                h, torch.tensor(sample[1]).unsqueeze(0),
                f, torch.tensor(sample[3]).unsqueeze(0),
            )
            pred_kw = pred.squeeze(0) * output_std + output_mean
            errors.append(float(torch.mean(torch.abs(pred_kw - torch.tensor(sample[4])))))
    validation_mae = sum(errors) / len(errors)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            "history_feature_dim": len(hist_columns),
            "target_feature_dim": len(target_columns),
            "history_feature_names": hist_columns,
            "target_feature_names": target_columns,
            "history_feature_mean": hist_mean.tolist(),
            "history_feature_std": hist_std.tolist(),
            "target_feature_mean": future_mean.tolist(),
            "target_feature_std": future_std.tolist(),
            "output_mean_kw": float(output_mean),
            "output_std_kw": float(output_std),
            "num_mes_plans": len(mes_mapping) + 1,
            "mes_mapping": mes_mapping,
            "mes_embed_dim": 8,
            "hidden_dim": 128,
            "history_days": args.history_days,
            "has_workshop_target": has_workshop,
            "validation_mae_kw": validation_mae,
            "seed": args.seed,
        },
    }, output)
    manifest = {
        "model_file": output.name,
        "training_data": Path(args.csv).name,
        "model_name": "load_mes_multitask_attention",
        "model_version": "v1",
        "data_nature": "simulated_typical_two_shift_factory",
        "history_days": args.history_days,
        "training_epochs": args.epochs,
        "seed": args.seed,
        "has_workshop_target": has_workshop,
        "validation_split": "last_20_percent_by_time",
        "validation_mae_kw": validation_mae,
        "mes_mapping": mes_mapping,
        "history_feature_names": hist_columns,
        "target_feature_names": target_columns,
        "limitations": [
            "trained on simulated rather than enterprise field data",
            "weather values are simulated and are not operational day-ahead forecasts",
            "accuracy must not be reported as real-factory performance",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={output} validation_mae_kw={validation_mae:.3f}")


if __name__ == "__main__":
    main()
