#!/usr/bin/env python3
"""Rolling-origin AMPDNet validation against yesterday persistence.

Input CSV columns: ``ts,tsi,dni,ghi,temperature_c,power_kw``.  Timestamps must
carry a timezone (``pandas`` accepts an explicit offset or UTC ``Z``).  The
script evaluates only origins whose complete daylight target remains on the
same local date; it never uses target-day measurements as model inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.domain.forecast.pv_ampdnet import TASKS
from app.domain.models import AmpdnetInput
from app.infrastructure.model_store import load_ampdnet_artifact
from app.application.ampdnet_service import _denormalize, _infer, _normalize
from app.domain.forecast.pv_ampdnet import predict_pv_ampdnet


def _daylight(ts) -> bool:
    return 6 <= ts.hour < 19 and ts.minute % 15 == 0


def _make_input(frame: pd.DataFrame, start: int, length: int, station: str, task: str):
    rows = frame.iloc[start : start + length]
    observations = [
        {"ts": row.ts.isoformat(), "tsi": row.tsi, "dni": row.dni, "ghi": row.ghi, "temperature_c": row.temperature_c}
        for row in rows.itertuples(index=False)
    ]
    return AmpdnetInput.model_validate({"station": station, "task": task, "observations": observations})


def evaluate(csv_path: Path, station: str, task: str) -> dict:
    frame = pd.read_csv(csv_path)
    required = {"ts", "tsi", "dni", "ghi", "temperature_c", "power_kw"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Shanghai")
    frame = frame.sort_values("ts").dropna(subset=required).reset_index(drop=True)
    length, steps = TASKS[task]
    model, metadata = load_ampdnet_artifact(station, task)
    predictions, actuals, baselines = [], [], []
    ts_to_power = dict(zip(frame.ts, frame.power_kw))
    for start in range(0, len(frame) - length - steps + 1):
        origin = frame.iloc[start + length - 1].ts
        targets = [origin + timedelta(minutes=15 * index) for index in range(1, steps + 1)]
        if not _daylight(origin) or any(ts.date() != origin.date() or not _daylight(ts) for ts in targets):
            continue
        if any(ts not in ts_to_power for ts in targets):
            continue
        inputs = _make_input(frame, start, length, station, task)
        result = predict_pv_ampdnet(
            inputs, origin, float(metadata["cap_kw"]), "Asia/Shanghai", metadata=metadata,
            infer=lambda batch: _infer(model, batch), normalize=_normalize, denormalize=_denormalize,
        )
        target_values = [float(ts_to_power[ts]) for ts in targets]
        yesterday = [float(ts_to_power.get(ts - timedelta(days=1), np.nan)) for ts in targets]
        if any(not np.isfinite(yesterday)):
            continue
        predictions.extend(point.value for point in result.points)
        actuals.extend(target_values)
        baselines.extend(yesterday)
    if not actuals:
        raise ValueError("no complete same-day daylight rolling origins found")
    mae = float(np.mean(np.abs(np.asarray(predictions) - actuals)))
    baseline_mae = float(np.mean(np.abs(np.asarray(baselines) - actuals)))
    return {
        "station": station, "task": task, "origins": len(actuals) // steps,
        "points": len(actuals), "mae_kw": mae, "day_persistence_mae_kw": baseline_mae,
        "skill_vs_day_persistence": (1.0 - mae / baseline_mae) if baseline_mae else None,
        "caliber": {"baseline": "day_persistence", "target": "same-day daylight rolling origin", "unit": "kW"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--station", default="Solar1")
    parser.add_argument("--task", choices=sorted(TASKS), default="15min")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.csv, args.station, args.task)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
