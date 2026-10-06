"""Run traditional models then deep models on independent GEFCom tracks."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import numpy as np
import pandas as pd

from app.domain.forecast.features import build_windows, regression_metrics
from app.domain.forecast.models import MODEL_NAMES, DEEP, TrainSettings
from app.domain.forecast.sources import citations
from app.domain.forecast.training import fit_classical, fit_neural


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_version() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BACKEND, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "."], cwd=BACKEND)
    except subprocess.CalledProcessError:
        commit, diff = "unknown", b""
    files = [*Path(BACKEND/"app/domain/forecast").glob("*.py"), Path(__file__)]
    return {"git_commit": commit, "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "files_sha256": {str(p.relative_to(BACKEND)): sha(p) for p in files}, "python": sys.version,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def license_inventory() -> list[dict]:
    return [{"package": d.metadata["Name"], "version": d.version,
             "license_expression": d.metadata.get("License-Expression"),
             "license": d.metadata.get("License"),
             "project_urls": d.metadata.get_all("Project-URL") or [],
             "license_files": d.metadata.get_all("License-File") or []}
            for d in importlib.metadata.distributions()]


def resolved_environment_lock() -> str:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending = [Requirement(line) for line in (BACKEND/"requirements.txt").read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    visited, pins = set(), {}
    while pending:
        requirement = pending.pop()
        key = (canonicalize_name(requirement.name), tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        distribution = importlib.metadata.distribution(requirement.name)
        pins[key[0]] = distribution.version
        for specification in distribution.requires or []:
            dependency = Requirement(specification)
            if dependency.marker is None or any(dependency.marker.evaluate({"extra": extra}) for extra in ({""}|requirement.extras)):
                pending.append(dependency)
    return "# Exact project dependency closure; same Python/platform recommended.\n"+"".join(f"{name}=={version}\n" for name, version in sorted(pins.items()))


def load_source(base: Path, task: str) -> tuple[pd.DataFrame, dict]:
    dataset = "GEFCom2014-L" if task == "load" else "GEFCom2014-P"
    paths = [base/dataset/task/freq/"all.csv" for freq in ("1h", "4h")]
    frames = [pd.read_csv(path) for path in paths]
    raw = pd.concat(frames).sort_values("ts", kind="stable").drop_duplicates("ts", keep="first")
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    # Recover published target observations at edges where origin+h was absent.
    # These are original labels, not interpolation. Exogenous fields remain NA.
    labels = pd.concat([f[["target_ts", "target_value"]].rename(columns={"target_ts": "ts", "target_value": task}) for f in frames])
    labels["ts"] = pd.to_datetime(labels["ts"], utc=True)
    raw = pd.concat([raw, labels]).sort_values("ts", kind="stable").drop_duplicates("ts", keep="first")
    keep = ["ts", task]+([f"w{i}" for i in range(1, 26)] if task == "load" else ["forecast_total_load", "forecast_zonal_load"])
    raw = raw[keep].reset_index(drop=True)
    return raw, {"dataset": dataset, "files_sha256": {str(p.relative_to(base)): sha(p) for p in paths},
                 "source_manifest": json.loads((base/dataset/task/"1h/dataset_manifest.json").read_text())}


def run_experiment(data, raw, source, task, horizon, model, variant, args, version) -> dict:
    exp = args.output/task/f"{horizon}h"/f"{model}__{variant}"
    (exp/"weights").mkdir(parents=True, exist_ok=True)
    train, val, test = (data.indices(s) for s in ("train", "validation", "test"))
    # Held-out test is not used for model selection. Calibration uses the
    # second half of validation; selection uses the first half.
    cap = args.load_cap if task == "load" else args.price_cap
    cap = cap or float(np.max(np.abs(data.y[train, -1])))
    cfg = {"task": task, "model": model, "variant": variant, "seed": 42,
           "settings": asdict(TrainSettings(epochs=args.epochs, batch_size=args.batch_size)),
           "horizon_hours": horizon, "resolution_minutes": 60, "input_size": data.history_size,
           "feature_names": data.feature_names, "exog_names": data.exog_names, "shuffle": False,
           "exogenous_availability": "origin and past only; no target-day observed weather",
           "image": {"used": False, "resolution": None, "frames": 0},
           "Cap": cap, "Cap_rule": "explicit user capacity" if (args.load_cap if task == "load" else args.price_cap) else "absolute training maximum proxy; NOT official contest capacity",
           "training_samples": len(train), "validation_samples": len(val), "test_samples": len(test),
           "unit": "source_load_units" if task == "load" else "USD/MWh",
           "source": source, "split": "fixed chronological target-day development:test=70:30; train:validation=80:20 within development, boundary origin embargo"}
    cfg["training_implementation_sha256"] = {name: sha(BACKEND/"app/domain/forecast"/name) for name in ("features.py", "models.py", "training.py")}
    save_json(exp/"config.json", cfg)
    save_json(exp/"code_version.json", version)
    save_json(exp/"citations.json", citations(model))
    t0 = time.monotonic()
    try:
        with (exp/"training.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"task={task} model={model} horizon={horizon} seed=42 shuffle=False")
            print(f"train={len(train)} validation={len(val)} test={len(test)}")
            settings = TrainSettings(epochs=args.epochs, batch_size=args.batch_size)
            if model in DEEP:
                artifact, vpred, pred = fit_neural(model, data, settings, exp)
            else:
                artifact, vpred, pred = fit_classical(model, data, raw, settings, exp)
        residual = data.y[val, -1]-vpred
        residual = residual[len(residual)//2:]
        q10, q90 = np.quantile(residual, [.1, .9])
        rank = min(len(residual), int(np.ceil((len(residual)+1)*.8)))
        conformal_q = float(np.sort(np.abs(residual))[rank-1])
        artifact.update({"task": task, "resolution_minutes": 60, "unit": cfg["unit"],
                         "feature_names": data.feature_names, "variant": variant,
                         "residual_quantiles": [float(q10), float(q90)], "conformal_q": conformal_q})
        save_json(exp/"weights/artifact.json", artifact)
        save_json(exp/"normalization.json", artifact["normalization"])
        hashes = {p.name: sha(p) for p in (exp/"weights").iterdir() if p.is_file() and p.name != "hashes.json"}
        save_json(exp/"weights/hashes.json", hashes)
        predictions = data.rows.iloc[test][["forecast_origin_ts", "target_ts"]].copy()
        predictions["y_true"], predictions["y_pred"] = data.y[test, -1].astype(float), np.asarray(pred, float)
        predictions["p10"], predictions["p50"], predictions["p90"] = pred+q10, pred, pred+q90
        predictions["conformal_low"], predictions["conformal_high"] = pred-conformal_q, pred+conformal_q
        predictions.to_csv(exp/"predictions.csv", index=False)
        pd.DataFrame({"target_ts": data.rows.iloc[val].target_ts, "y_true": data.y[val, -1].astype(float), "y_pred": np.asarray(vpred, float)}).to_csv(exp/"validation_predictions.csv", index=False)
        metric = regression_metrics(data.y[test, -1], pred, cap)
        metric["interval_coverage_80"] = float(np.mean((data.y[test, -1] >= pred-conformal_q)&(data.y[test, -1] <= pred+conformal_q)))
        metric["interval_mean_width"] = 2*conformal_q
        metric["validation_RMSE"] = regression_metrics(data.y[val, -1], vpred, cap)["RMSE"]
        times = pd.to_datetime(predictions.target_ts, utc=True)
        groups = np.where(times.dt.dayofweek < 5, "weekday", "weekend")
        metric["by_day_type"] = {g: regression_metrics(data.y[test, -1][groups == g], pred[groups == g], cap) for g in ("weekday", "weekend")}
        metric["weather_groups"] = {"status": "unavailable", "reason": "load and price archives contain no validated cloud-cover labels"}
        save_json(exp/"metrics.json", metric)
        status = {"status": "complete", "elapsed_seconds": time.monotonic()-t0}
        save_json(exp/"status.json", status)
        return {"task": task, "horizon": horizon, "model": model, "variant": variant,
                **status, **{k: v for k, v in metric.items() if not isinstance(v, dict)}}
    except Exception as exc:
        save_json(exp/"status.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        return {"task": task, "horizon": horizon, "model": model, "variant": variant,
                "status": "failed", "error": str(exc), "elapsed_seconds": time.monotonic()-t0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["load", "price"], default=["load", "price"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=[1, 4], default=[1, 4])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-size", type=int, default=48)
    parser.add_argument("--load-cap", type=float)
    parser.add_argument("--price-cap", type=float)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--isolated-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.history_size < 2:
        parser.error("epochs/batch-size must be positive and history-size >= 2")
    if any(c is not None and (not np.isfinite(c) or c <= 0) for c in (args.load_cap, args.price_cap)):
        parser.error("capacity denominators must be positive finite numbers")
    args.output.mkdir(parents=True, exist_ok=True)
    version = code_version()
    if not args.isolated_worker:
        save_json(args.output/"code_version.json", version)
        save_json(args.output/"open_source_inventory.json", license_inventory())
        (args.output/"environment.full.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        (args.output/"environment.lock.txt").write_text(resolved_environment_lock())
        save_json(args.output/"availability.json", {"15min": "unavailable: no native GEFCom 15-minute labels", "sky_images": "unavailable"})
    rows = []
    ordered_models = [name for name in MODEL_NAMES if name in args.models]
    # Entire traditional matrix is completed before the deep matrix starts.
    for model in ordered_models:
        for task in args.tasks:
            raw, source = load_source(args.data, task)
            if not (args.output/task/"raw.csv").exists():
                (args.output/task).mkdir(parents=True, exist_ok=True)
                raw.to_csv(args.output/task/"raw.csv", index=False)
                save_json(args.output/task/"source.json", source)
            for horizon in args.horizons:
                variants = ["full"]+["no_exogenous", "no_calendar"] if args.ablations and model == "LightGBM" else ["full"]
                for variant in variants:
                    data = build_windows(raw, task, horizon, args.history_size, variant)
                    if variant != "full":
                        full = build_windows(raw, task, horizon, args.history_size, "full")
                        mask = data.rows.source_position.isin(full.rows.source_position).to_numpy()
                        data.rows = data.rows.loc[mask].reset_index(drop=True)
                        data.x, data.seq, data.exog, data.calendar, data.y = (a[mask] for a in (data.x, data.seq, data.exog, data.calendar, data.y))
                    exp = args.output/task/f"{horizon}h"/f"{model}__{variant}"
                    sample_path = args.output/task/f"{horizon}h"/"samples.csv"
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    if variant == "full" and not sample_path.exists():
                        data.rows.to_csv(sample_path, index=False)
                        save_json(sample_path.parent/"split_manifest.json", {
                            "sample_sha256": sha(sample_path), "source": source, "seed": 42,
                            "counts": data.rows.split.value_counts().to_dict(),
                            "ranges": {s: {"origin_min": str(data.rows.iloc[data.indices(s)].forecast_origin_ts.min()),
                                           "target_max": str(data.rows.iloc[data.indices(s)].target_ts.max())} for s in ("train", "validation", "test")},
                            "image": {"used": False, "frames": 0}, "native_15min": False,
                        })
                    reusable = False
                    if args.resume and (exp/"metrics.json").exists() and (exp/"status.json").exists() and (exp/"config.json").exists():
                        old_cfg = json.loads((exp/"config.json").read_text())
                        reusable = (json.loads((exp/"status.json").read_text())["status"] == "complete"
                                    and old_cfg.get("settings") == asdict(TrainSettings(epochs=args.epochs, batch_size=args.batch_size))
                                    and old_cfg.get("source") == source and old_cfg.get("input_size") == args.history_size
                                    and old_cfg.get("Cap") == ((args.load_cap if task == "load" else args.price_cap) or float(np.max(np.abs(data.y[data.indices("train"), -1]))))
                                    and (exp/"weights/hashes.json").exists()
                                    and old_cfg.get("training_implementation_sha256") == {name: sha(BACKEND/"app/domain/forecast"/name) for name in ("features.py", "models.py", "training.py")})
                    if reusable:
                        metric = json.loads((exp/"metrics.json").read_text())
                        row = {"task": task, "horizon": horizon, "model": model, "variant": variant, "status": "complete", **{k: v for k, v in metric.items() if not isinstance(v, dict)}}
                    else:
                        print(f"RUN {task} h={horizon} {model} {variant}", flush=True)
                        if args.isolated_worker:
                            row = run_experiment(data, raw, source, task, horizon, model, variant, args, version)
                        else:
                            # Native OpenMP runtimes in tree/statistical and
                            # torch libraries can deadlock after mixed imports.
                            # Each fitted model gets a fresh process as well as
                            # an independent checkpoint and normalization.
                            command = [sys.executable, str(Path(__file__).resolve()),
                                       "--data", str(args.data), "--output", str(args.output),
                                       "--tasks", task, "--horizons", str(horizon), "--models", model,
                                       "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                                       "--history-size", str(args.history_size), "--isolated-worker"]
                            if args.ablations and model == "LightGBM":
                                command.extend(["--ablations", "--resume"])
                            for cap_task in ("load", "price"):
                                cap_arg = getattr(args, f"{cap_task}_cap")
                                if cap_arg is not None:
                                    command.extend([f"--{cap_task}-cap", str(cap_arg)])
                            child_env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
                            exp.mkdir(parents=True, exist_ok=True)
                            with (exp/"worker.log").open("w") as worker_log:
                                finished = subprocess.run(command, env=child_env, stdout=worker_log, stderr=subprocess.STDOUT)
                            status = json.loads((exp/"status.json").read_text()) if (exp/"status.json").exists() else {"status": "failed", "error": "worker exited without status"}
                            if finished.returncode:
                                status = {"status": "failed", "error": f"worker exit {finished.returncode}; see worker.log"}
                            metric = json.loads((exp/"metrics.json").read_text()) if status["status"] == "complete" else {}
                            row = {"task": task, "horizon": horizon, "model": model, "variant": variant,
                                   **status, **{k: v for k, v in metric.items() if not isinstance(v, dict)}}
                    rows.append(row)
                    print(f"  {row['status']} RMSE={row.get('RMSE', row.get('error'))}", flush=True)
                    if not args.isolated_worker:
                        pd.DataFrame(rows).to_csv(args.output/"results.csv", index=False)
                        save_json(args.output/"results.json", rows)
                        save_json(args.output/"unavailable_models.json", [r for r in rows if r["status"] != "complete"])
    if not args.isolated_worker:
        for task in args.tasks:
            pd.DataFrame([r for r in rows if r["task"] == task]).to_csv(args.output/f"{task}_results.csv", index=False)
        save_json(args.output/"code_version.json", version)


if __name__ == "__main__":
    main()
