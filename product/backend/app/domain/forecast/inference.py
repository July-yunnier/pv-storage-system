"""Validate and reload a task-specific checkpoint for native rolling inference."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain.forecast.features import CALENDAR, LAGS, calendar_features
from app.domain.forecast.models import TrainSettings, make_neural, set_seed
from app.domain.forecast.result import ForecastModel, ForecastPoint, TargetForecast
from app.domain.models import MachineLearningForecastRequest


def predict_artifact(request: MachineLearningForecastRequest, artifact_dir: Path,
                     *, _model_worker: bool = False) -> TargetForecast:
    metadata = json.loads((artifact_dir/"artifact.json").read_text())
    if metadata["task"] != request.task:
        raise ValueError("artifact task mismatch; load and price weights are independent")
    if metadata["name"] != request.model or metadata["horizon"] != request.horizon_hours:
        raise ValueError("artifact model/horizon mismatch")
    if metadata["resolution_minutes"] != request.resolution_minutes or metadata["unit"] != request.unit:
        raise ValueError("artifact native frequency/unit mismatch; no implicit interpolation or currency conversion")
    hashes = json.loads((artifact_dir/"hashes.json").read_text())
    for filename, digest in hashes.items():
        if Path(filename).name != filename or hashlib.sha256((artifact_dir/filename).read_bytes()).hexdigest() != digest:
            raise ValueError("checkpoint hash verification failed")
    history = request.history
    timestamps = pd.DatetimeIndex([p.ts for p in history]).tz_convert("UTC")
    if timestamps[-1] != pd.Timestamp(request.forecast_origin).tz_convert("UTC"):
        raise ValueError("last observation must be at forecast origin")
    if not (timestamps[1:]-timestamps[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("history must be chronological, unique and contiguous hourly points")
    values = np.array([p.value for p in history], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("history must contain finite values")
    history_size, horizon = metadata["history_size"], metadata["horizon"]
    if len(values) < max(169, history_size):
        raise ValueError("insufficient history for model window and lag_168")
    exog_names = metadata["exog_names"]
    if any(c not in request.historical_exogenous or len(request.historical_exogenous[c]) != len(values) for c in exog_names):
        raise ValueError("historical exogenous features must match model names and history length")
    exog = np.column_stack([request.historical_exogenous[c] for c in exog_names]) if exog_names else np.empty((len(values), 0))
    if not np.isfinite(exog).all():
        raise ValueError("exogenous features must be finite")
    target_ts = request.forecast_origin + pd.Timedelta(hours=horizon)
    x = np.r_[[values[-1-lag] for lag in LAGS], exog[-1], calendar_features(pd.Series([target_ts]))[0] if metadata.get("variant") != "no_calendar" else []]
    stats = metadata["normalization"]
    kind = metadata["kind"]
    if kind not in {"persistence", "history_mean"} and not _model_worker:
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        try:
            worker = subprocess.run([sys.executable, "-m", "app.domain.forecast.inference", str(artifact_dir.resolve())],
                                    input=request.model_dump_json(), text=True, capture_output=True, env=env,
                                    cwd=Path(__file__).resolve().parents[3], timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("model prediction worker timed out") from exc
        if worker.returncode:
            raise RuntimeError(f"model prediction worker failed: {worker.stderr[-2000:]}")
        return TargetForecast.model_validate_json(worker.stdout)
    if kind == "persistence":
        value = values[-1]
    elif kind == "history_mean":
        value = metadata["means"].get(str(pd.Timestamp(target_ts).tz_convert("UTC").hour), metadata["fallback"])
    elif kind == "regressor":
        import joblib
        estimator = joblib.load(artifact_dir/"estimator.joblib")
        z = (x-np.asarray(stats["x_mean"]))/np.asarray(stats["x_std"])
        value = float(estimator.predict(z.reshape(1, -1))[0]*stats["target_std"]+stats["target_mean"])
    elif kind == "neural":
        import torch
        from app.domain.forecast.training import torch_batch
        settings = TrainSettings(**metadata["settings"])
        set_seed(settings.seed, torch_required=True)
        net = make_neural(request.model, history_size, horizon, exog_names, CALENDAR, settings)
        net.load_state_dict(torch.load(artifact_dir/"model.pt", map_location="cpu", weights_only=True))
        net.eval()
        calendar_times = pd.Series(pd.date_range(timestamps[-history_size], periods=history_size+horizon, freq="h"))
        calendar = calendar_features(calendar_times)
        if metadata.get("variant") == "no_calendar":
            calendar[:] = 0
        batch = torch_batch(values[-history_size:][None], exog[-history_size:][None], calendar[None], stats)
        with torch.no_grad():
            value = float(net(batch).reshape(1, horizon, -1)[0, -1, 0]*stats["target_std"]+stats["target_mean"])
    elif kind == "prophet":
        from prophet.serialize import model_from_json
        estimator = model_from_json((artifact_dir/"prophet.json").read_text())
        value = float(estimator.predict(pd.DataFrame({"ds": [pd.Timestamp(target_ts).tz_convert("UTC").tz_localize(None)]})).yhat.iloc[0])
    elif kind == "arima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        z = (values-stats["target_mean"])/stats["target_std"]
        estimator = SARIMAX(z, order=tuple(metadata["order"]), seasonal_order=tuple(metadata["seasonal_order"]), trend="c")
        fitted = estimator.filter(np.load(artifact_dir/"params.npy"))
        value = float(fitted.forecast(horizon)[-1]*stats["target_std"]+stats["target_mean"])
    else:
        raise ValueError("unknown checkpoint kind")
    if not np.isfinite(value):
        raise ValueError("model generated a non-finite prediction")
    q10, q90 = metadata["residual_quantiles"]
    return TargetForecast(target="load_power" if request.task == "load" else "price", unit=request.unit,
                          model=ForecastModel(name=request.model, version="gefcom_v3_seed42"),
                          points=[ForecastPoint(ts=target_ts, value=value)], p10=[value+q10], p90=[value+q90],
                          notes="Native hourly rolling forecast; independently trained task checkpoint; no currency/frequency conversion.")


if __name__ == "__main__":
    payload = MachineLearningForecastRequest.model_validate_json(sys.stdin.read())
    result = predict_artifact(payload, Path(sys.argv[1]), _model_worker=True)
    print(result.model_dump_json())
