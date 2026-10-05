from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pvlib

from experiments.protocol import ROOT, dump_json, package_run, sample_indices, seed_everything


def clear_sky_power(frame, manifest):
    location = pvlib.location.Location(manifest["lat"], manifest["lon"], tz=manifest["timezone"])
    times = pd.DatetimeIndex(frame.ts)
    ghi = location.get_clearsky(times, model="ineichen").ghi
    # Explicit horizontal PVWatts proxy; actual tilt is not supplied. This
    # baseline must be named accordingly, not an archived NWP forecast.
    dc = pvlib.pvsystem.pvwatts_dc(ghi, 25., manifest["cap_kw"], -.004)
    return np.clip(np.asarray(dc), 0, manifest["cap_kw"])


def prediction_rows(frame, idx, future, prediction, split_name="test"):
    parts = []
    for j, horizon in enumerate((15, 60, 240)):
        targets = future[idx, j]
        parts.append(pd.DataFrame({"issued_at": frame.ts.iloc[idx].to_numpy(),
            "target_at": frame.ts.iloc[targets].to_numpy(), "horizon_minutes": horizon,
            "actual_kw": frame.power_kw.iloc[targets].to_numpy(), "predicted_kw": prediction[:, j],
            "weather": frame.weather.iloc[targets].to_numpy(),
            "daylight": frame.daylight.iloc[targets].to_numpy(), "split": split_name}))
    return pd.concat(parts, ignore_index=True)


def frozen_historical_mean(frame, train_end):
    # Fit exclusively before the train cutoff. Keep the state available as JSON.
    times = pd.DatetimeIndex(frame.ts)
    slot = times.hour * 60 + times.minute
    train = times < pd.Timestamp(train_end)
    means = frame.loc[train].groupby(slot[train]).power_kw.mean().to_dict()
    default = float(frame.loc[train].power_kw.mean())
    return np.array([means.get(int(s), default) for s in slot]), {str(k): float(v) for k, v in means.items()}


def run(directory):
    seed_everything()
    directory = ROOT / directory
    frame = pd.read_pickle(directory / "frame.pkl")
    manifest = json.loads((directory / "manifest.json").read_text())
    split = json.loads((directory / "split.json").read_text())
    sky = manifest["dataset"] == "SKIPPD"
    indices, future = sample_indices(frame, split, 30 if sky else 64, 1 if sky else 15)
    cap = manifest["cap_kw"]
    means, mean_state = frozen_historical_mean(frame, split["train_end"])
    p = frame.power_kw.to_numpy()
    models = {"persistence": (np.repeat(p[:, None], 3, axis=1), {"type": "no learned weights"}),
              "historical_mean": (means[future], {"slot_means_kw": mean_state, "fit_before": split["train_end"]})}
    if sky:
        clear = clear_sky_power(frame, manifest)
        ratio = np.divide(p, clear, out=np.zeros_like(p), where=clear > .01 * cap)
        models["smart_persistence"] = (np.clip(ratio[:, None] * clear[future], 0, cap),
                                      {"clear_sky": "pvlib Ineichen, horizontal PVWatts, temp 25C", "low_clear_sky": "prediction=0"})
        models["pvlib_clear_sky_horizontal"] = (clear[future], {"ghi": "pvlib Ineichen", "temperature": 25,
                "tilt_degrees": 0, "status": "clear sky proxy, actual array geometry unavailable"})
    else:
        # Physical baseline uses current observed irradiance and temperature,
        # held into future. It NEVER reads target-time observed weather.
        cell = pvlib.temperature.faiman(frame.tsi.clip(lower=0), frame.temperature, 1.)
        physics = np.clip(np.asarray(pvlib.pvsystem.pvwatts_dc(frame.tsi.clip(lower=0), cell, cap, -.004)), 0, cap)
        models["pvlib_irradiance_persistence"] = (np.repeat(physics[:, None], 3, axis=1),
                {"irradiance": "TSI at issue held constant", "wind_speed": 1., "temperature": "Faiman using issue temperature",
                 "status": "TSI assumed POA; geometry unavailable; observations only through issue"})
    config = {"dataset_manifest": manifest, "split": split,
              "indices": {k: len(v) for k, v in indices.items()}, "fit_on": "train only",
              "nighttime": manifest.get("nighttime", "included; daylight subset also reported")}
    test_idx, val_idx = indices["test"], indices["val"]
    for model, (pred, state) in models.items():
        rows = prediction_rows(frame, test_idx, future, pred[test_idx])
        out = ROOT / "runs" / directory.name / model
        result = package_run(out, dict(config, model=model), rows, state, cap)
        # Residual quantiles calibrated exclusively on validation predictions.
        residual = p[future[val_idx]] - pred[val_idx]
        q = np.quantile(residual, [.1, .5, .9], axis=0)
        qrows = rows.copy()
        for j, horizon in enumerate((15, 60, 240)):
            mask = qrows.horizon_minutes == horizon
            for k, name in enumerate(("p10_kw", "p50_kw", "p90_kw")):
                qrows.loc[mask, name] = np.clip(qrows.loc[mask, "predicted_kw"] + q[k, j], 0, cap)
        qrows.predicted_kw = qrows.p50_kw
        package_run(ROOT / "runs" / directory.name / (model + "_residual_quantiles"),
                    dict(config, model=model + "_residual_quantiles", calibration="validation only"), qrows,
                    dict(base=state, residual_quantiles_kw=q.tolist(), quantile_order=[.1, .5, .9]), cap)
        print(directory.name, model, {h: round(m["one_minus_rmse_over_cap"], 4) for h,m in result["by_horizon"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()
    paths = sorted((ROOT / "data/processed").glob("*/manifest.json")) if args.dataset == "all" else [ROOT / "data/processed" / args.dataset / "manifest.json"]
    for path in paths:
        run(path.parent.relative_to(ROOT))
