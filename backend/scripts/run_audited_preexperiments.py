"""Validation-only selection, frozen selection manifest, then final evaluation.

All results are exploratory GEFCom pre-experiments. Historical test exposure
cannot be undone by this prospective protocol. No PV or dispatch is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.domain.forecast.features import build_windows
from app.domain.forecast.models import TrainSettings
from app.domain.forecast.quantile import QuantileSettings, QuantileNet, fit_quantile, pinball_numpy, predict
from app.domain.forecast.training import fit_classical
from scripts.run_converged_preexperiments import empirical_interval, metric_row, source, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def remove_exog(data):
    # Keep exactly the same windows/masks/sample counts as the full model.
    return replace(data, exog=data.exog[..., :0], exog_names=[])


def protocol(args):
    paths = [BACKEND / p for p in (
        "app/domain/forecast/quantile.py", "app/domain/forecast/features.py",
        "app/domain/forecast/training.py", "app/domain/forecast/models.py",
        "scripts/run_converged_preexperiments.py", "scripts/run_audited_preexperiments.py")]
    files = [args.data / dataset / task / freq / "all.csv"
             for dataset, task in (("GEFCom2014-L", "load"), ("GEFCom2014-P", "price")) for freq in ("1h", "4h")]
    return {
        "scope": "GEFCom exploratory pre-experiment; main experiment NOT started",
        "historical_test_exposure": True,
        "seed": 42, "shuffle": False, "tasks": args.tasks, "horizons_hours": args.horizons,
        "history_hours": 48, "native_resolution_hours": 1,
        "15min": "unsupported: no interpolation or invented targets",
        "split": "chronological development:test=70:30 days; train:validation=80:20 within development; boundary embargo",
        "standardization": "train-only zscore for BOTH tasks; load Cap is reporting proxy ONLY",
        "code_sha256": {str(p.relative_to(BACKEND)): sha(p) for p in paths},
        "data_sha256": {str(p): sha(p) for p in files},
        "settings": {"max_epochs": args.max_epochs, "patience": args.patience, "hidden": 32, "dropout": .1},
        "tcn_grid": [{"lr": .001, "batch": 1024}, {"lr": .0005, "batch": 512}],
        "selection_metric": "validation final-horizon Pinball; test not used",
        "variants": ["LSTM_baseline", "TCN_baseline", "TCN_optimized", "no_rolling", "no_noise", "no_exogenous"],
        "pv": "inference-only adapter; awaiting checkpoint + architecture + input contract",
        "dispatch": "not executed", "weather_groups": "not used", "cross_station_join": False,
    }


def fit_trial(data, kind, settings, exp, fingerprint, resume):
    marker = exp / "completed.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if not resume or info["fingerprint"] != fingerprint or info["settings"] != asdict(settings):
            raise ValueError(f"existing trial differs: {exp}; use a new output folder")
        if sha(exp / "model.pt") != info["weight_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        return info
    t0 = time.monotonic()
    _, _, meta = fit_quantile(kind, data, settings, exp)
    info = {**meta, "fingerprint": fingerprint, "weight_sha256": sha(exp / "model.pt"),
            "elapsed_seconds": time.monotonic() - t0}
    save(marker, info)
    return info


def load_trial(data, exp, info):
    import torch
    settings = QuantileSettings(**info["settings"])
    net = QuantileNet(info["name"], info["input_dim"], settings.hidden, data.horizon, settings.dropout)
    net.load_state_dict(torch.load(exp / "model.pt", map_location="cpu", weights_only=True))
    return net, info["normalization"], settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["load", "price"], choices=["load", "price"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4], choices=[1, 4])
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 10: parser.error("epoch=3 coarse screening is forbidden")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = protocol(args)
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    old = args.output / "protocol.json"
    if old.exists() and json.loads(old.read_text()) != plan:
        raise ValueError("protocol changed; do not silently resume")
    save(old, plan)
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    (args.output / "environment.lock").write_text(f"# {platform.platform()}\n# Python {sys.version}\n" + "\n".join(packages) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND, capture_output=True, text=True)
    save(args.output / "code_version.json", {"git_head": git.stdout.strip(), "dirty_tree": True,
         "identity": "code SHA256 is authoritative for this dirty tree", "code_sha256": plan["code_sha256"]})
    all_selected, tuning = [], []
    for task in args.tasks:
        raw, src = source(args.data, task)
        raw.to_csv(args.output / f"{task}_source.csv", index=False)
        for horizon in args.horizons:
            data = build_windows(raw, task, horizon, 48)
            root = args.output / task / f"{horizon}h"
            root.mkdir(parents=True, exist_ok=True)
            data.rows.to_csv(root / "split_manifest.csv", index=False)
            settings = QuantileSettings(max_epochs=args.max_epochs, patience=args.patience, batch_size=1024, hidden=32)
            entries = []
            for kind in ("LSTM", "TCN"):
                label = f"{kind}_baseline"
                print(f"FIT {task} {horizon}h {label} (validation ONLY)", flush=True)
                exp = root / "trials" / label
                info = fit_trial(data, kind, settings, exp, fingerprint, args.resume)
                entries.append({"label": label, "path": str(exp), "info": info, "no_exogenous": False})
            candidates = []
            for i, grid in enumerate(plan["tcn_grid"]):
                tuned = replace(settings, learning_rate=grid["lr"], batch_size=grid["batch"], use_rolling=True, noise_std=.01)
                exp = root / "trials" / f"TCN_optimized_trial{i}"
                print(f"TUNE {task} {horizon}h TCN {grid} (validation ONLY)", flush=True)
                info = fit_trial(data, "TCN", tuned, exp, fingerprint, args.resume)
                entry = {"label": "TCN_optimized", "path": str(exp), "info": info, "no_exogenous": False}
                candidates.append(entry)
                tuning.append({"task": task, "horizon": horizon, "trial": i, **grid,
                               "validation_pinball": info["best_validation_loss"], "stop_reason": info["stop_reason"],
                               "best_epoch": info["best_epoch"], "epochs": info["epochs_completed"]})
                save(args.output / "tuning_records.json", tuning)
            winner = min(candidates, key=lambda e: e["info"]["best_validation_loss"])
            entries.append(winner)
            chosen = QuantileSettings(**winner["info"]["settings"])
            for label, absettings, noex in (
                ("no_rolling", replace(chosen, use_rolling=False), False),
                ("no_noise", replace(chosen, noise_std=0.0), False),
                ("no_exogenous", chosen, True),
            ):
                exp = root / "trials" / label
                print(f"ABLATE {task} {horizon}h {label} (validation ONLY)", flush=True)
                info = fit_trial(remove_exog(data) if noex else data, "TCN", absettings, exp, fingerprint, args.resume)
                entries.append({"label": label, "path": str(exp), "info": info, "no_exogenous": noex})
            all_selected.append({"task": task, "horizon": horizon, "entries": entries, "source": src})
            save(args.output / "training_progress.json", all_selected)
    # Freeze all selections BEFORE any test metric/prediction is produced.
    save(args.output / "selection_frozen.json", {"fingerprint": fingerprint, "selected": all_selected,
         "basis": "validation only; no further model choices may depend on final test metrics"})
    rows = []
    for selected in all_selected:
        task, horizon = selected["task"], selected["horizon"]
        raw, _ = source(args.data, task)
        data = build_windows(raw, task, horizon, 48)
        tr, va, te = [data.indices(s) for s in ("train", "validation", "test")]
        # Reporting proxy from complete historical training timestamps only.
        cutoff = pd.to_datetime(data.rows.iloc[tr].target_ts.max())
        cap = float(raw.loc[raw.ts <= cutoff, task].max()) if task == "load" else None
        entries = selected["entries"] + [{"label": name} for name in ("Persistence", "ARIMA")]
        for entry in entries:
            label = entry["label"]
            exp = args.output / task / f"{horizon}h" / "evaluation" / label
            exp.mkdir(parents=True, exist_ok=True)
            print(f"FINAL_EVAL {task} {horizon}h {label}", flush=True)
            if label in {"Persistence", "ARIMA"}:
                (exp / "weights").mkdir(exist_ok=True)
                art, vp, tp = fit_classical("SmartPersistence" if label == "Persistence" else label, data, raw, TrainSettings(seed=42), exp)
                lo, hi = empirical_interval(data.y[va, -1], vp)
                tq = np.column_stack([tp + lo, tp, tp + hi])
                vq = np.column_stack([vp + lo, vp, vp + hi])
                info = {"stop_reason": "fixed_baseline" if label == "Persistence" else "MLE_converged" if art.get("converged") else "MLE_not_converged",
                        "best_epoch": None, "epochs_completed": None}
                save(exp / "artifact.json", {**art, "interval_method": "second_validation_half_residual_quantiles",
                     "interval_limitations": "not native head; calibration-set validation score optimistic; no exchangeability coverage guarantee"})
                native = False
            else:
                work = remove_exog(data) if entry["no_exogenous"] else data
                net, stats, settings = load_trial(work, Path(entry["path"]), entry["info"])
                vq = predict(net, work, stats, va, settings.batch_size)[:, -1]
                tq = predict(net, work, stats, te, settings.batch_size)[:, -1]
                info = entry["info"]
                native = True
            metrics = metric_row(task, label.split("_")[0], horizon, label, data.y[te, -1], tq[:, 1], tq[:, 0], tq[:, 2], cap, native)
            metrics.update({"normalization": "train_only_zscore", "cap_is_physical": False,
                "validation_pinball": pinball_numpy(data.y[va, -1], vq),
                "interval_mean_width": float(np.mean(tq[:, 2] - tq[:, 0])),
                "coverage_abs_error": abs(metrics["interval_coverage_80"] - .8),
                "quantile_crossing_rate": float(np.mean((tq[:, 0] > tq[:, 1]) | (tq[:, 1] > tq[:, 2]))),
                "stop_reason": info["stop_reason"], "best_epoch": info["best_epoch"], "epochs_completed": info["epochs_completed"],
                "scope": "pre-experiment ONLY", "historical_test_exposure": True})
            save(exp / "metrics.json", metrics)
            preds = data.rows.iloc[te][["forecast_origin_ts", "target_ts"]].copy()
            preds["y_true"], preds["p10"], preds["p50"], preds["p90"] = data.y[te, -1], tq[:, 0], tq[:, 1], tq[:, 2]
            preds.to_csv(exp / "predictions.csv", index=False)
            rows.append(metrics)
    for task in args.tasks:
        # Separate tables: no price Cap or misleading capacity-threshold columns.
        pd.DataFrame([r for r in rows if r["task"] == task]).to_csv(args.output / f"{task}_results.csv", index=False)
    save(args.output / "completion.json", {"evaluations": len(rows), "training_trials": sum(len(s["entries"]) + 1 for s in all_selected),
         "budget_limited": [{"task": s["task"], "horizon": s["horizon"], "variant": e["label"]}
                            for s in all_selected for e in s["entries"] if e["info"]["stop_reason"] != "validation_early_stop"],
         "pv": "awaiting artifacts; no inference metrics fabricated", "main_experiment": "not started"})
    print("DONE: exploratory pre-experiment only", flush=True)


if __name__ == "__main__": main()
