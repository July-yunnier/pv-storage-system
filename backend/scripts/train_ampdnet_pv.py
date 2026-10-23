#!/usr/bin/env python3
"""Train a small AMPDNet-compatible CPU artifact and export metadata.

This is an offline, reproducible trainer for CSV data with columns
``ts,tsi,dni,ghi,temperature_c,power_kw``.  It deliberately writes a
TorchScript artifact plus JSON metadata under ``backend/data/training_artifacts``;
the production delivery loader remains read-only and can consume the same
metadata contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--station", default="Solar1")
    parser.add_argument("--task", choices=["15min", "1h", "4h"], default="15min")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/training_artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("training requires pandas, numpy, and torch") from exc
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    frame = pd.read_csv(args.csv)
    required = ["ts", "tsi", "dni", "ghi", "temperature_c", "power_kw"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise SystemExit(f"missing CSV columns: {', '.join(missing)}")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.sort_values("ts").dropna(subset=required).reset_index(drop=True)
    length, horizon = {"15min": (48, 1), "1h": (96, 4), "4h": (192, 16)}[args.task]
    x = frame[["tsi", "dni", "ghi", "temperature_c"]].to_numpy(np.float32)
    y = frame["power_kw"].to_numpy(np.float32)
    if len(frame) < length + horizon:
        raise SystemExit(f"need at least {length + horizon} rows")
    low, high = x.min(axis=0), x.max(axis=0)
    scale = np.maximum(high - low, 1e-6)
    output_max_mw = max(float(y.max()) / 1000.0, 1e-6)
    windows = np.stack([(x[i : i + length] - low) / scale for i in range(len(x) - length - horizon + 1)])
    targets = np.stack([y[i + length : i + length + horizon] for i in range(len(windows))]) / 1000.0
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Conv1d(4, 16, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, horizon))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx, ty = torch.from_numpy(windows).transpose(1, 2), torch.from_numpy(targets)
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = ((model(tx) - ty) ** 2).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.station}_ampdnet_cpu_s{args.seed}"
    artifact_path = args.output_dir / f"{tag}.torchscript.pt"
    torch.jit.script(model).save(str(artifact_path))
    metadata = {
        "station": args.station, "task": args.task, "tag": tag,
        "training_epochs_completed": args.epochs, "selected_epoch": args.epochs,
        "seed": args.seed, "input_size": length, "output_size": horizon,
        "features": ["TSI", "DNI", "GHI", "Temp"], "input_min": low.tolist(), "input_max": high.tolist(),
        "output_min_mw": 0.0, "output_max_mw": output_max_mw, "cap_kw": float(y.max()),
        "artifact_file": artifact_path.name, "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "training_data": str(args.csv), "timestamp_policy": "UTC-aware CSV timestamps",
    }
    (args.output_dir / f"{tag}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"artifact": str(artifact_path), "metadata": str(args.output_dir / f"{tag}.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
