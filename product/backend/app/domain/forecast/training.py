"""Independent fitting, checkpointing and causal evaluation for each task."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain.forecast.features import CALENDAR, ForecastWindows
from app.domain.forecast.models import DEEP, TrainSettings, make_neural, make_tree, set_seed


def scale_statistics(data: ForecastWindows) -> dict:
    ids = data.indices("train")
    if not len(ids):
        raise ValueError("empty train split")
    # Windows overlap; each published training-origin value is counted once.
    mu = float(data.rows.iloc[ids].origin_value.mean())
    std = max(float(data.rows.iloc[ids].origin_value.std(ddof=0)), 1e-6)
    x = data.x[ids]
    ex = data.exog[ids].reshape(-1, len(data.exog_names)) if data.exog_names else np.empty((0, 0))
    return {"target_mean": mu, "target_std": std,
            "x_mean": x.mean(axis=0).tolist(), "x_std": np.maximum(x.std(axis=0), 1e-6).tolist(),
            "exog_mean": ex.mean(axis=0).tolist() if data.exog_names else [],
            "exog_std": np.maximum(ex.std(axis=0), 1e-6).tolist() if data.exog_names else [],
            "fit_split": "train"}


def torch_batch(seq, exog, calendar, stats):
    import torch
    s = torch.as_tensor((seq-stats["target_mean"])/stats["target_std"], dtype=torch.float32).unsqueeze(-1)
    if exog.shape[-1]:
        exog = (exog-np.asarray(stats["exog_mean"])) / np.asarray(stats["exog_std"])
    e = torch.as_tensor(exog, dtype=torch.float32)
    c = torch.as_tensor(calendar, dtype=torch.float32)
    return {"insample_y": s, "insample_mask": torch.ones_like(s),
            "hist_exog": e, "futr_exog": c, "stat_exog": None}


def neural_predictions(net, data: ForecastWindows, ids: np.ndarray, stats: dict, batch_size: int) -> np.ndarray:
    import torch
    net.eval()
    output = []
    with torch.no_grad():
        for begin in range(0, len(ids), batch_size):
            selected = ids[begin:begin+batch_size]
            pred = net(torch_batch(data.seq[selected], data.exog[selected], data.calendar[selected], stats))
            pred = pred.reshape(len(selected), data.horizon, -1)[..., 0]
            output.append(pred[:, -1].cpu().numpy()*stats["target_std"]+stats["target_mean"])
    return np.concatenate(output)


def fit_neural(name: str, data: ForecastWindows, settings: TrainSettings, exp: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    import torch
    set_seed(settings.seed, torch_required=True)
    stats = scale_statistics(data)
    net = make_neural(name, data.history_size, data.horizon, data.exog_names, CALENDAR, settings)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.learning_rate)
    train_ids, val_ids = data.indices("train"), data.indices("validation")
    selection_ids = val_ids[:len(val_ids)//2]
    best_loss = float("inf")
    best_state = None
    epochs = []
    for epoch in range(settings.epochs):
        net.train()
        total = 0.0
        for begin in range(0, len(train_ids), settings.batch_size):
            ids = train_ids[begin:begin+settings.batch_size]
            optimizer.zero_grad()
            prediction = net(torch_batch(data.seq[ids], data.exog[ids], data.calendar[ids], stats))
            prediction = prediction.reshape(len(ids), data.horizon, -1)[..., 0]
            truth = torch.tensor((data.y[ids, -1]-stats["target_mean"])/stats["target_std"], dtype=torch.float32)
            loss = torch.mean((prediction[:, -1]-truth)**2)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(ids)
        val_pred = neural_predictions(net, data, selection_ids, stats, settings.batch_size)
        val_mse = float(np.mean((val_pred-data.y[selection_ids, -1])**2))
        epochs.append({"epoch": epoch+1, "train_MSE_standardized": total/len(train_ids), "validation_MSE": val_mse})
        if val_mse < best_loss:
            best_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        (exp/"training_log.json").write_text(json.dumps(epochs, indent=2))
    net.load_state_dict(best_state)
    torch.save(best_state, exp/"weights"/"model.pt")
    artifact = {"kind": "neural", "name": name, "settings": asdict(settings), "normalization": stats,
                "history_size": data.history_size, "horizon": data.horizon, "exog_names": data.exog_names,
                "calendar_names": CALENDAR, "best_validation_MSE": best_loss,
                "architecture": {"class": type(net).__name__,
                                 "historical_exogenous_used": name in {"LSTM", "GRU", "BiLSTM", "RLinear"} or bool(getattr(net, "EXOGENOUS_HIST", False)),
                                 "future_calendar_used": name in {"LSTM", "GRU", "BiLSTM", "RLinear"} or bool(getattr(net, "EXOGENOUS_FUTR", False)),
                                 "informer_factor": data.history_size if name == "Informer" else None}}
    return artifact, neural_predictions(net, data, val_ids, stats, settings.batch_size), neural_predictions(net, data, data.indices("test"), stats, settings.batch_size)


def fit_classical(name: str, data: ForecastWindows, raw: pd.DataFrame, settings: TrainSettings,
                  exp: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    set_seed(settings.seed)
    train, val, test = (data.indices(s) for s in ("train", "validation", "test"))
    y = data.y[:, -1]
    stats = scale_statistics(data)
    base = {"name": name, "normalization": stats, "history_size": data.history_size,
            "horizon": data.horizon, "settings": asdict(settings), "exog_names": data.exog_names}
    if name == "SmartPersistence":
        return {**base, "kind": "persistence"}, data.seq[val, -1], data.seq[test, -1]
    if name == "HistoryMean":
        hours = pd.to_datetime(data.rows.target_ts, utc=True).dt.hour
        means = pd.Series(y[train]).groupby(hours.iloc[train].reset_index(drop=True)).mean().to_dict()
        pred = hours.map(means).fillna(float(y[train].mean())).to_numpy()
        return {**base, "kind": "history_mean", "means": means, "fallback": float(y[train].mean())}, pred[val], pred[test]
    if name in {"SVR", "LightGBM", "XGBoost"}:
        x = (data.x-np.asarray(stats["x_mean"]))/np.asarray(stats["x_std"])
        model = make_tree(name, settings)
        model.fit(x[train], (y[train]-stats["target_mean"])/stats["target_std"])
        pred = model.predict(x)*stats["target_std"]+stats["target_mean"]
        import joblib
        joblib.dump(model, exp/"weights"/"estimator.joblib")
        return {**base, "kind": "regressor"}, pred[val], pred[test]
    if name == "Prophet":
        from prophet import Prophet
        from prophet.serialize import model_to_json
        model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True,
                        uncertainty_samples=0)
        ds = pd.to_datetime(data.rows.target_ts, utc=True).dt.tz_localize(None)
        model.fit(pd.DataFrame({"ds": ds.iloc[train], "y": y[train]}), seed=settings.seed)
        pred = model.predict(pd.DataFrame({"ds": ds}))["yhat"].to_numpy()
        (exp/"weights"/"prophet.json").write_text(model_to_json(model))
        return {**base, "kind": "prophet"}, pred[val], pred[test]
    if name in {"ARIMA", "SARIMA"}:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.statespace.kalman_filter import MEMORY_NO_PREDICTED_COV, MEMORY_NO_FILTERED_COV, MEMORY_NO_GAIN, MEMORY_NO_SMOOTHING
        raw = raw.copy().sort_values("ts").set_index("ts")
        source = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))[base["task"] if "task" in base else ("load" if "load" in raw else "price")]
        z = (source.to_numpy()-stats["target_mean"])/stats["target_std"]
        cutoff = pd.Timestamp(data.rows.iloc[train].target_ts.max())
        last_train = source.index.searchsorted(cutoff, side="right")
        seasonal = (1, 0, 0, 24) if name == "SARIMA" else (0, 0, 0, 0)
        # Fixed documented 90-day estimation budget. Evaluation filters the
        # entire past with fixed parameters, never estimates on validation/test.
        fit_model = SARIMAX(z[max(0, last_train-2160):last_train], order=(2, 0, 0), seasonal_order=seasonal, trend="c")
        fitted = fit_model.fit(disp=False, maxiter=100)
        filtering = SARIMAX(z, order=(2, 0, 0), seasonal_order=seasonal, trend="c")
        filtering.ssm.set_conserve_memory(MEMORY_NO_PREDICTED_COV | MEMORY_NO_FILTERED_COV | MEMORY_NO_GAIN | MEMORY_NO_SMOOTHING)
        filtered = filtering.filter(fitted.params)
        ids_all = source.index.get_indexer(pd.to_datetime(data.rows.forecast_origin_ts))
        state = filtered.filtered_state[:, ids_all]
        transition = np.asarray(filtering.ssm["transition"])
        intercept = np.asarray(filtering.ssm["state_intercept"])
        if intercept.ndim == 2:
            # Constant-trend state intercept is stored as a repeated column
            # per timestamp by statsmodels, unlike the transition matrix.
            intercept = intercept[:, 0]
        for _ in range(data.horizon):
            state = transition @ state + intercept[:, None]
        pred = (np.asarray(filtering.ssm["design"]) @ state).reshape(-1)*stats["target_std"]+stats["target_mean"]
        np.save(exp/"weights"/"params.npy", fitted.params)
        return {**base, "kind": "arima", "order": [2, 0, 0], "seasonal_order": list(seasonal), "fit_tail_hours": 2160,
                "converged": bool(fitted.mle_retvals.get("converged"))}, pred[val], pred[test]
    raise ValueError(f"unimplemented model: {name}")
