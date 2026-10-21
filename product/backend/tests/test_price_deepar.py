from datetime import datetime, timedelta
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app.domain.forecast.price_deepar import (
    FEATURE_NAMES, FORMAT, PriceDeepAR, calendar_features, gaussian_nll,
    predict_price_deepar, sample_paths, validate_series,
)
from scripts.train_price_deepar import split_origins, read_series, train
from tests.conftest import CN_TZ

START = datetime(2026, 1, 2, tzinfo=CN_TZ)
ISSUED = START - timedelta(minutes=5)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    model = PriceDeepAR(hidden_size=4, num_layers=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.mu.bias.fill_(-0.4)
        model.sigma.bias.fill_(-5)
    metadata = {"format": FORMAT, "feature_names": FEATURE_NAMES,
                "architecture": {"num_series": 1, "hidden_size": 4, "num_layers": 2},
                "series_ids": ["S-API"], "unit": "CNY/kWh", "timezone": "Asia/Shanghai",
                "context_length": 8, "resolution_minutes": 15,
                "selection_end": "2020-01-01T00:00:00+08:00"}
    path = tmp_path / "price.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    monkeypatch.setenv("PRICE_MODEL_PATH", str(path))
    return path


def history(resolution=15, length=8):
    return [(START - timedelta(minutes=resolution * (length - i)), -0.1) for i in range(length)]


def predict(**kwargs):
    options = dict(timezone_name="Asia/Shanghai", series_id="S-API", num_samples=100)
    options.update(kwargs)
    return predict_price_deepar(history(), START, 96, ISSUED, **options)


def rewrite(path, **kwargs):
    data = torch.load(path, weights_only=True)
    data["metadata"].update(kwargs)
    torch.save(data, path)


def test_negative_prices_quantiles_seed_and_fee(artifact):
    a = predict()
    b = predict(transmission_and_fund=0.21)
    assert len(a.points) == len(a.p10) == len(a.p90) == 96
    assert a.points[-1].ts == START + timedelta(hours=23, minutes=45)
    assert max(p.value for p in a.points) < 0
    for lo, point, hi, with_fee in zip(a.p10, a.points, a.p90, b.points):
        assert lo < point.value < hi
        assert with_fee.value - point.value == pytest.approx(0.21)
    assert predict().model_dump() == a.model_dump()
    assert predict(include_uncertainty=False).p10 is None
    assert a.accuracy is None
    assert len(a.provenance.checkpoint_sha256) == 64


def test_never_conditions_on_future_observations(artifact):
    a = predict()
    b = predict_price_deepar(history() + [(START, 1e9)], START, 96, ISSUED,
                             timezone_name="Asia/Shanghai", series_id="S-API", num_samples=100)
    assert a.points == b.points


@pytest.mark.parametrize("changes,match", [
    ({"unit": "USD/MWh"}, "CNY/kWh"),
    ({"timezone": "UTC"}, "timezone"),
    ({"selection_end": "2026-01-02T00:00:00+08:00"}, "cutoff"),
    ({"series_ids": ["other"]}, "series_id"),
])
def test_rejects_incompatible_or_leaking_checkpoint(artifact, changes, match):
    rewrite(artifact, **changes)
    with pytest.raises(ValueError, match=match):
        predict()


def test_hourly_price_is_explicitly_expanded(artifact):
    rewrite(artifact, resolution_minutes=60)
    result = predict_price_deepar(history(60), START, 192, ISSUED,
                                  timezone_name="Asia/Shanghai", series_id="S-API")
    assert len(result.points) == 192
    assert len({p.value for p in result.points[:4]}) == 1
    assert "原生小时" in result.notes


def test_gaps_and_stale_history_rejected(artifact):
    with pytest.raises(ValueError, match="continuous"):
        validate_series(history()[:-2] + history()[-1:], 15)
    stale = [(ts - timedelta(hours=3), value) for ts, value in history()]
    with pytest.raises(ValueError, match="stale"):
        predict_price_deepar(stale, START, 96, ISSUED, timezone_name="Asia/Shanghai", series_id="S-API")


def test_sampling_propagates_uncertainty_and_preserves_rng():
    class RandomWalk:
        def __call__(self, previous, features, ids, state=None):
            if state is None:
                state = (torch.zeros(1, len(ids), 1), torch.zeros(1, len(ids), 1))
            return previous, torch.ones_like(previous), state
    future = [START + timedelta(minutes=15 * i) for i in range(8)]
    rng = torch.random.get_rng_state().clone()
    paths = sample_paths(RandomWalk(), history(), future, num_samples=2000)
    assert paths[:, -1].var() > 5 * paths[:, 0].var()
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_gaussian_likelihood_has_finite_gradients_for_signed_prices():
    model = PriceDeepAR(hidden_size=4)
    y = torch.tensor([[-1.0, 0, 0.5, 10.0]])
    features = torch.tensor([[calendar_features(ts) for ts, _ in history(length=4)]])
    mu, sigma, _ = model(y, features, torch.tensor([0]))
    loss = gaussian_nll(mu, sigma, -y)
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert sigma.min() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shared_time_split_and_targets_never_overlap():
    values = [(START + timedelta(hours=i), float(i)) for i in range(60 * 24)]
    series = {"A": values, "B": values[24:]}
    groups, train_cut, val_cut = split_origins(series, 168, 72, 24)
    target_sets = {}
    for split, origins in groups.items():
        target_sets[split] = {(name, ts) for name, _, origin in origins for ts, _ in series[name][origin:origin + 72]}
    assert target_sets["train"].isdisjoint(target_sets["validation"])
    assert target_sets["validation"].isdisjoint(target_sets["test"])
    assert max(ts for _, ts in target_sets["train"]) < train_cut
    assert min(ts for _, ts in target_sets["test"]) >= val_cut


def test_gapped_training_csv_requires_explicit_policy(tmp_path):
    path = tmp_path / "prices.csv"
    values = [(START + timedelta(hours=i), 1) for i in range(60 * 24) if i != 500]
    path.write_text("ts,price\n" + "\n".join(f"{ts.isoformat()},{v}" for ts, v in values))
    with pytest.raises(ValueError, match="continuous"):
        read_series(path, "Asia/Shanghai", 60, "A")
    series = read_series(path, "Asia/Shanghai", 60, "A", True)
    groups, _, _ = split_origins(series, 168, 24, 24)
    for origins in groups.values():
        for name, _, origin in origins:
            validate_series(series[name][origin - 168:origin + 24], 60)


def test_api_price_ingestion_and_forecast(site_client, artifact):
    response = site_client.post("/api/v1/sites/S-API/measurements", json={
        "data_type": "price", "unit": "CNY/kWh", "resolution_minutes": 15,
        "points": [{"ts": ts.isoformat(), "value": value} for ts, value in history()],
    })
    assert response.status_code == 200, response.text
    payload = {"site_id": "S-API", "targets": ["price"], "issued_at": ISSUED.isoformat(),
               "options": {"price_model": "deepar", "include_uncertainty": False}}
    response = site_client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["model"]["name"] == "price_deepar"
    assert result["p10"] is None and result["p90"] is None
    assert all(p["value"] < 0 for p in result["points"])
    # TOU 默认仍按政策展开；显式深度模型不能在缺失权重时静默伪装成政策价。
    del payload["options"]
    assert site_client.post("/api/v1/forecast", json=payload).json()["results"][0]["model"]["name"] == "tou_policy"
    artifact.unlink()
    payload["options"] = {"price_model": "deepar"}
    assert site_client.post("/api/v1/forecast", json=payload).status_code == 422


def test_price_csv_unit_and_finite_value_validation(site_client):
    r = site_client.post("/api/v1/sites/S-API/measurements/csv", data={"data_type": "price"},
                         files={"file": ("price.csv", f"ts,value\n{START.isoformat()},-0.2\n", "text/csv")})
    assert r.status_code == 200, r.text
    r = site_client.get("/api/v1/sites/S-API/measurements", params={"data_type": "price", "date": "2026-01-02"})
    assert r.json()[0]["unit"] == "CNY/kWh"
    for unit, value in (("USD/MWh", 1), ("CNY/kWh", "NaN")):
        r = site_client.post("/api/v1/sites/S-API/measurements", json={
            "data_type": "price", "unit": unit, "points": [{"ts": START.isoformat(), "value": value}]})
        assert r.status_code == 422


def test_native_15min_training_export_and_online_prediction(tmp_path):
    """小型有符号合成数据只验证训练→选型→导出→加载→预测，不作为精度证据。"""
    values = [(START + timedelta(minutes=15 * i), 0.1 + 0.3 * math.sin(i * math.pi / 48))
              for i in range(40 * 96)]
    csv_path = tmp_path / "training.csv"
    csv_path.write_text("ts,price\n" + "\n".join(f"{ts.isoformat()},{v}" for ts, v in values))
    output = tmp_path / "trained.pt"
    train(SimpleNamespace(csv=csv_path, output=output, unit="CNY/kWh", timezone="Asia/Shanghai",
                          resolution_minutes=15, series_id="S-API", context_days=7, horizon_days=1,
                          epochs=1, batch_size=8, hidden_size=4, lr=1e-3, num_samples=20,
                          seed=42, threads=2, skip_gapped_windows=False))
    report = json.loads(output.with_suffix(".manifest.json").read_text())
    assert report["metadata"]["context_length"] == 672
    assert math.isfinite(report["test_metrics"]["DeepAR"]["MAE"])
    assert output.with_suffix(".test.csv").is_file()
    start = START + timedelta(days=40)
    result = predict_price_deepar(values, start, 96, start - timedelta(minutes=5),
                                  timezone_name="Asia/Shanghai", series_id="S-API",
                                  num_samples=20, model_path=str(output))
    assert len(result.points) == 96
    assert all(math.isfinite(p.value) for p in result.points)


@pytest.mark.parametrize("mode", ["spot", "tou_plus_spot"])
def test_explicit_legacy_deepar_for_spot_and_does_not_add_tou_energy(site_client, artifact, mode):
    tariff = site_client.post("/api/v1/tariffs", json={"tariff_id": "T-SPOT", "name": "现货",
        "mode": mode, "transmission_and_fund": 0.21,
        "tiers": [{"name": "flat", "price": 100, "hours": ["00:00-12:00", "12:00-00:00"]}]} )
    assert tariff.status_code == 201, tariff.text
    assert site_client.patch("/api/v1/sites/S-API", json={"tariff_id": "T-SPOT"}).status_code == 200
    site_client.post("/api/v1/sites/S-API/measurements", json={"data_type": "price", "unit": "CNY/kWh",
        "points": [{"ts": ts.isoformat(), "value": v} for ts, v in history()]})
    payload = {"site_id": "S-API", "targets": ["price"], "issued_at": ISSUED.isoformat(),
               "options": {"price_model": "deepar"}}
    r = site_client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 200, r.text
    forecast = r.json()["results"][0]
    assert forecast["model"]["name"] == "price_deepar"
    assert forecast["points"][0]["value"] == pytest.approx(predict(transmission_and_fund=0.21).points[0].value, abs=0.01)
    payload["options"] = {"price_model": "tou"}
    assert site_client.post("/api/v1/forecast", json=payload).status_code == 422
