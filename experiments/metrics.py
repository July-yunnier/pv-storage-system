from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = .96


def errors(actual, predicted, cap_kw):
    if not np.isfinite(cap_kw) or cap_kw <= 0:
        raise ValueError("Cap must be the documented station capacity in kW")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual) or actual.shape != predicted.shape:
        raise ValueError("metrics require nonempty aligned arrays")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("nonfinite predictions/labels cannot silently be dropped")
    residual = actual - predicted
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.square(residual).mean()))
    return dict(n=len(actual), mae_kw=mae, rmse_kw=rmse,
                one_minus_mae_over_cap=1 - mae / cap_kw,
                one_minus_rmse_over_cap=1 - rmse / cap_kw,
                pass_mae=bool(mae / cap_kw <= 1 - THRESHOLD + 1e-12),
                pass_rmse=bool(rmse / cap_kw <= 1 - THRESHOLD + 1e-12),
                pass_both=bool(max(mae, rmse) / cap_kw <= 1 - THRESHOLD + 1e-12))


def evaluate(frame, cap_kw):
    required = {"issued_at", "target_at", "horizon_minutes", "actual_kw", "predicted_kw", "weather"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions missing {required - set(frame.columns)}")
    frame = frame.copy()
    if frame.duplicated(["issued_at", "horizon_minutes"]).any():
        raise ValueError("duplicate forecast cases")
    delta = (pd.to_datetime(frame.target_at, utc=True) - pd.to_datetime(frame.issued_at, utc=True)).dt.total_seconds() / 60
    if not np.array_equal(delta.to_numpy(), frame.horizon_minutes.to_numpy()):
        raise ValueError("horizon is elapsed time, not a row offset")
    result = {"threshold": THRESHOLD, "cap_kw": float(cap_kw), "units": "kW",
              "by_horizon": {}, "nighttime": "all available cases plus daylight subset",
              "weather_definition": "source labels where available; unknown is not inferred from errors"}
    for horizon, group in frame.groupby("horizon_minutes"):
        overall = errors(group.actual_kw, group.predicted_kw, cap_kw)
        months = pd.to_datetime(group.target_at, utc=True).dt.strftime("%Y-%m")
        monthly = {m: errors(g.actual_kw, g.predicted_kw, cap_kw) for m, g in group.groupby(months)}
        overall["monthly"] = monthly
        overall["monthly_mean_mae_accuracy"] = float(np.mean([m["one_minus_mae_over_cap"] for m in monthly.values()]))
        overall["monthly_mean_rmse_accuracy"] = float(np.mean([m["one_minus_rmse_over_cap"] for m in monthly.values()]))
        overall["monthly_pass_both_rate"] = float(np.mean([m["pass_both"] for m in monthly.values()]))
        overall["pass_monthly_mean_both"] = bool(min(overall["monthly_mean_mae_accuracy"], overall["monthly_mean_rmse_accuracy"]) >= THRESHOLD - 1e-12)
        overall["weather"] = {w: errors(g.actual_kw, g.predicted_kw, cap_kw) for w, g in group.groupby("weather")}
        for w in ("sunny", "partly_cloudy", "overcast"):
            overall["weather"].setdefault(w, {"n": 0, "status": "no_verified_weather_labels"})
        if "daylight" in group and group.daylight.any():
            g = group[group.daylight]
            overall["daylight"] = errors(g.actual_kw, g.predicted_kw, cap_kw)
        if {"p10_kw", "p50_kw", "p90_kw"}.issubset(group.columns):
            qs = group[["p10_kw", "p50_kw", "p90_kw"]].to_numpy()
            if not np.isfinite(qs).all() or (np.diff(qs, axis=1) < -1e-10).any():
                raise ValueError("invalid or crossing quantiles")
            actual = group.actual_kw.to_numpy()[:, None]
            diff = actual - qs
            q = np.array([.1, .5, .9])
            overall["pinball_kw"] = np.maximum(q * diff, (q - 1) * diff).mean(axis=0).tolist()
            overall["coverage_p10_p90"] = float(((actual[:, 0] >= qs[:, 0]) & (actual[:, 0] <= qs[:, 2])).mean())
            overall["mean_interval_width_kw"] = float((qs[:, 2] - qs[:, 0]).mean())
        result["by_horizon"][str(int(horizon))] = overall
    result["pass_all_horizons_both"] = bool(
        set(result["by_horizon"]) == {"15", "60", "240"} and
        all(m["pass_both"] and m["pass_monthly_mean_both"] for m in result["by_horizon"].values()))
    return result
