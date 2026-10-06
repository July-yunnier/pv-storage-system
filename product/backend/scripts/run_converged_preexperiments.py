"""Converged, leakage-safe GEFCom pre-experiments.

This runner deliberately keeps GEFCom-L, GEFCom-P and GEFCom-S separate.
It does not train or alter the frozen PV model.  Price has no Cap metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.domain.forecast.features import build_windows, regression_metrics
from app.domain.forecast.models import TrainSettings
from app.domain.forecast.quantile import QuantileSettings, fit_quantile, interval_coverage, pinball_numpy, predict
from app.domain.forecast.training import fit_classical


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(base: Path, task: str):
    dataset = "GEFCom2014-L" if task == "load" else "GEFCom2014-P"
    key = "load" if task == "load" else "price"
    frames = [pd.read_csv(base/dataset/key/freq/"all.csv") for freq in ("1h", "4h")]
    raw = pd.concat(frames).sort_values("ts", kind="stable").drop_duplicates("ts")
    raw["ts"] = pd.to_datetime(raw.ts, utc=True)
    labels = pd.concat([frame[["target_ts", "target_value"]].rename(columns={"target_ts": "ts", "target_value": task}) for frame in frames])
    labels.ts = pd.to_datetime(labels.ts, utc=True)
    raw = pd.concat([raw, labels]).sort_values("ts", kind="stable").drop_duplicates("ts", keep="first")
    keep = ["ts", task] + ([f"w{i}" for i in range(1, 26)] if task == "load" else ["forecast_total_load", "forecast_zonal_load"])
    return raw[keep].reset_index(drop=True), {"dataset": dataset, "files": [str(base/dataset/key/freq/"all.csv") for freq in ("1h", "4h")], "cross_site_join": False}


def empirical_interval(y_val, p_val):
    residual = np.asarray(y_val) - np.asarray(p_val)
    residual = residual[len(residual)//2:]
    return float(np.quantile(residual, .10)), float(np.quantile(residual, .90))


def metric_row(task, model, horizon, variant, y, pred, p10, p90, cap, native):
    m = regression_metrics(y, pred, cap if task == "load" else None)
    m.update({"task": task, "model": model, "horizon": horizon, "variant": variant,
              "Pinball": pinball_numpy(y, np.column_stack([p10, pred, p90])),
              "interval_coverage_80": interval_coverage(y, p10, p90),
              "interval_type": "native_pinball_head" if native else "validation_residual_postcalibration",
              "normalization": "train_abs_max_proxy" if task == "load" else "zscore_no_cap"})
    return m


def run_one(raw, task, horizon, model, variant, args, out, source_meta):
    data = build_windows(raw, task, horizon, args.history_size, variant)
    exp = out/task/f"{horizon}h"/f"{model}__{variant}"; (exp/"weights").mkdir(parents=True, exist_ok=True)
    train, val, test = (data.indices(s) for s in ("train", "validation", "test"))
    cap = float(np.max(np.abs(data.y[train, -1]))) if task == "load" else None
    t0 = time.time()
    if model in {"Persistence", "ARIMA"}:
        name = "SmartPersistence" if model == "Persistence" else "ARIMA"
        artifact, val_pred, test_pred = fit_classical(name, data, raw, TrainSettings(seed=42), exp)
        lo, hi = empirical_interval(data.y[val, -1], val_pred)
        p10, p90 = test_pred + lo, test_pred + hi
        native = False
        meta = {"kind": artifact["kind"], "note": "point model plus validation residual interval; not a native probability head"}
    else:
        settings = QuantileSettings(seed=42, max_epochs=args.max_epochs, patience=args.patience,
                                    learning_rate=args.learning_rate, batch_size=args.batch_size,
                                    hidden=args.hidden, noise_std=args.noise_std if variant == "optimized" else 0.0)
        model_name = "TCN" if model == "TCN" else "LSTM"
        net, stats, meta = fit_quantile(model_name, data, settings, exp, "pinball")
        val_all = predict(net, data, stats, val, settings.batch_size)
        test_all = predict(net, data, stats, test, settings.batch_size)
        val_pred, test_pred = val_all[:, -1, 1], test_all[:, -1, 1]
        p10, p90 = test_all[:, -1, 0], test_all[:, -1, 2]
        artifact = {"kind": "native_quantile", "name": model_name, "variant": variant,
                    "normalization": stats, "settings": meta["settings"], "horizon": horizon,
                    "task": task, "quantiles": [0.1, 0.5, 0.9], "seed": 42,
                    "source": source_meta, "early_stopping": True}
        (exp/"artifact.json").write_text(json.dumps(artifact, indent=2))
        native = True
    y_test = data.y[test, -1]
    metrics = metric_row(task, model, horizon, variant, y_test, test_pred, p10, p90, cap, native)
    metrics["validation_pinball"] = pinball_numpy(data.y[val, -1], np.column_stack([val_pred + empirical_interval(data.y[val, -1], val_pred)[0], val_pred, val_pred + empirical_interval(data.y[val, -1], val_pred)[1]]))
    metrics["elapsed_seconds"] = time.time() - t0
    pred = data.rows.iloc[test][["forecast_origin_ts", "target_ts"]].copy()
    pred["y_true"], pred["p10"], pred["p50"], pred["p90"] = y_test, p10, test_pred, p90
    pred.to_csv(exp/"predictions.csv", index=False)
    (exp/"metrics.json").write_text(json.dumps(metrics, indent=2))
    (exp/"config.json").write_text(json.dumps({"task": task, "model": model, "variant": variant,
        "horizon": horizon, "seed": 42, "shuffle": False, "history_size": args.history_size,
        "split": "chronological 70:30 development:test; 80:20 train:validation within development",
        "cap": cap, "cap_is_physical": False, "price_cap": None, "source": source_meta}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["load", "price"], default=["load", "price"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=[1, 4], default=[1, 4])
    parser.add_argument("--history-size", type=int, default=48)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=.01)
    parser.add_argument("--models", nargs="+", choices=["Persistence", "ARIMA", "LSTM", "TCN"], default=["Persistence", "ARIMA", "LSTM", "TCN"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"preexperiment_scope.json").write_text(json.dumps({
        "status": "GEFCom pre-experiment only; formal main experiment not started",
        "seed": 42, "weather_groups": "not used; no clear sky/cloud/overcast grouping",
        "cross_station_year_join": False, "pv_training": "forbidden; frozen artifact inference only",
        "price_cap": "forbidden; zscore normalization only", "probability": "native pinball for LSTM/TCN; post-calibrated intervals explicitly labeled for point baselines",
        "training_mode": "converged early stopping; epoch=3 coarse screening is not run"}, indent=2))
    rows = []
    for task in args.tasks:
        raw, meta = source(args.data, task); (args.output/task).mkdir(exist_ok=True); raw.to_csv(args.output/task/"raw.csv", index=False)
        for horizon in args.horizons:
            # The optimized variant is TCN+rolling statistics+noise. Baseline LSTM is native pinball without noise.
            for model in args.models:
                variant = "optimized" if model == "TCN" else "baseline"
                existing = args.output/task/f"{horizon}h"/f"{model}__{variant}/metrics.json"
                if args.resume and existing.exists():
                    rows.append(json.loads(existing.read_text()))
                    continue
                print(f"RUN {task} {horizon}h {model} {variant}", flush=True)
                rows.append(run_one(raw, task, horizon, model, variant, args, args.output, meta))
                pd.DataFrame(rows).to_csv(args.output/"results.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output/"results.csv", index=False)
    (args.output/"unavailable_models.json").write_text("[]")
    (args.output/"README.md").write_text("# GEFCom完整收敛预实验\n\n本目录只包含GEFCom预实验，不代表正式主实验或工程性能。负荷Cap为训练历史最大值统计代理；电价无Cap。未按天气分组，未跨站/跨年拼接。\n")


if __name__ == "__main__": main()
