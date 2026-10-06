from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.domain.forecast.features import build_windows, regression_metrics
from app.domain.models import MachineLearningForecastRequest


def source_frame() -> pd.DataFrame:
    size = 24*240
    ts = pd.date_range("2010-01-01", periods=size, freq="h", tz="UTC")
    return pd.DataFrame({"ts": ts, "price": 20+np.sin(np.arange(size)/24),
                         "forecast_total_load": np.full(size, 100), "forecast_zonal_load": np.full(size, 50)})


def test_target_day_split_has_boundary_embargo_and_causal_features():
    data = build_windows(source_frame(), "price", 4)
    train, val, test = (data.rows.iloc[data.indices(s)] for s in ("train", "validation", "test"))
    assert train.target_ts.max() < val.forecast_origin_ts.min()
    assert val.target_ts.max() < test.forecast_origin_ts.min()
    assert (data.rows.target_ts-data.rows.forecast_origin_ts == pd.Timedelta(hours=4)).all()
    np.testing.assert_array_equal(data.x[:, 0], data.rows.origin_value)


def test_future_labels_do_not_change_current_features():
    raw = source_frame()
    before = build_windows(raw, "price", 4)
    position = 1000
    row = before.rows[before.rows.source_position == position].index[0]
    raw.loc[position+1:, "price"] += 1000
    raw.loc[position+1:, "forecast_total_load"] += 1000
    after = build_windows(raw, "price", 4)
    np.testing.assert_array_equal(before.x[row], after.x[row])
    np.testing.assert_array_equal(before.seq[row], after.seq[row])
    np.testing.assert_array_equal(before.exog[row], after.exog[row])


def test_missing_history_is_not_interpolated():
    raw = source_frame().drop(index=1000)
    windows = build_windows(raw, "price", 4)
    assert not windows.rows.source_position.between(1000, 1047).any()


def test_metrics_do_not_hide_nonfinite_predictions():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0]), np.array([np.nan]), 10)
    assert regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 10)["passes_RMSE_96"]


def test_model_request_requires_timezone_and_blocks_arbitrary_paths():
    history = [{"ts": datetime(2010, 1, 1, tzinfo=timezone.utc)+timedelta(hours=i), "value": 20} for i in range(169)]
    with pytest.raises(ValueError):
        MachineLearningForecastRequest(task="price", model="../../fake", unit="USD/MWh", forecast_origin=history[-1]["ts"], history=history)


def test_ml_route_is_registered_and_missing_model_is_422(site_client, monkeypatch, tmp_path):
    monkeypatch.setenv("FORECAST_ARTIFACT_ROOT", str(tmp_path))
    start = datetime(2010, 1, 1, tzinfo=timezone.utc)
    response = site_client.post("/api/v1/forecast/ml", json={
        "task": "price", "model": "LightGBM", "unit": "USD/MWh",
        "forecast_origin": (start+timedelta(hours=168)).isoformat(),
        "history": [{"ts": (start+timedelta(hours=i)).isoformat(), "value": 20} for i in range(169)],
    })
    assert response.status_code == 422


def test_task_checkpoints_are_independent_and_api_reloads(site_client, monkeypatch, tmp_path):
    from app.domain.forecast.inference import predict_artifact
    monkeypatch.setenv("FORECAST_ARTIFACT_ROOT", str(tmp_path))
    start = datetime(2010, 1, 1, tzinfo=timezone.utc)
    origin = start+timedelta(hours=168)
    for task, value, unit in (("load", 100, "source_load_units"), ("price", 25, "USD/MWh")):
        path = tmp_path/task/"1h/SmartPersistence__full/weights"
        path.mkdir(parents=True)
        metadata = {"task": task, "name": "SmartPersistence", "horizon": 1,
                    "resolution_minutes": 60, "unit": unit, "history_size": 48,
                    "exog_names": [], "normalization": {}, "kind": "persistence",
                    "residual_quantiles": [-1, 1]}
        content = json.dumps(metadata)
        (path/"artifact.json").write_text(content)
        (path/"hashes.json").write_text(json.dumps({"artifact.json": hashlib.sha256(content.encode()).hexdigest()}))
        request = {"task": task, "model": "SmartPersistence", "unit": unit,
                   "forecast_origin": origin.isoformat(),
                   "history": [{"ts": (start+timedelta(hours=i)).isoformat(), "value": value} for i in range(169)]}
        response = site_client.post("/api/v1/forecast/ml", json=request)
        assert response.status_code == 200
        assert response.json()["results"][0]["points"][0]["value"] == value
        wrong = "price" if task == "load" else "load"
        with pytest.raises(ValueError, match="task mismatch"):
            predict_artifact(MachineLearningForecastRequest(**{**request, "task": wrong}), path)
        (path/"artifact.json").write_text(content+" ")
        with pytest.raises(ValueError, match="hash verification"):
            predict_artifact(MachineLearningForecastRequest(**request), path)


@pytest.mark.parametrize("name", ["LSTM", "GRU", "BiLSTM", "TFT", "Informer", "Autoformer", "PatchTST", "DLinear", "RLinear", "NHiTS"])
@pytest.mark.parametrize("horizon", [1, 4])
def test_deep_architecture_forward_backward_and_deterministic_reload(name, horizon):
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralforecast")
    from app.domain.forecast.models import make_neural, TrainSettings, set_seed
    set_seed(42, torch_required=True)
    batch = {"insample_y": torch.randn(4, 48, 1), "insample_mask": torch.ones(4, 48, 1),
             "hist_exog": torch.randn(4, 48, 2), "futr_exog": torch.randn(4, 48+horizon, 6), "stat_exog": None}
    net = make_neural(name, 48, horizon, ["a", "b"], list("abcdef"), TrainSettings())
    net(batch).square().mean().backward()
    net.eval()
    with torch.no_grad():
        expected = net(batch)
        # A full-query Informer can reorder equal-scored queries; allow
        # float32 summation rounding, not stochastic prediction variation.
        torch.testing.assert_close(expected, net(batch), rtol=1e-6, atol=1e-6)
        single = net({k: v[:1] if v is not None else None for k, v in batch.items()})
        torch.testing.assert_close(expected[:1], single, rtol=1e-4, atol=1e-5)
        reloaded = make_neural(name, 48, horizon, ["a", "b"], list("abcdef"), TrainSettings())
        reloaded.load_state_dict(net.state_dict())
        reloaded.eval()
        torch.testing.assert_close(expected, reloaded(batch), rtol=1e-5, atol=1e-6)


def test_chronological_training_is_reproducible(tmp_path):
    pytest.importorskip("torch")
    from app.domain.forecast.models import TrainSettings
    from app.domain.forecast.training import fit_neural
    data = build_windows(source_frame().iloc[:24*40], "price", 1)
    outputs = []
    for attempt in range(2):
        path = tmp_path/str(attempt)
        (path/"weights").mkdir(parents=True)
        artifact, validation, test = fit_neural("LSTM", data, TrainSettings(epochs=1), path)
        outputs.append((artifact, validation, test))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
