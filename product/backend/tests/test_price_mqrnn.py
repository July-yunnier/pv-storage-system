from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
from app.domain.forecast.price_mqrnn import (
    FEATURE_NAMES, FORMAT, QUANTILES, TARGET, PriceMQRNN,
    pinball_loss, predict_price_mqrnn,
)
from app.domain.forecast.price import forecast_price_tou
from app.domain.models import TariffRead
from scripts.train_price_mqrnn import ForkDataset, train
from tests.conftest import CN_TZ

START = datetime(2026, 1, 2, tzinfo=CN_TZ)
ISSUED = START - timedelta(minutes=5)


def history(resolution=15, context=8):
    n = context + 7 * 1440 // resolution
    return [(START - timedelta(minutes=resolution * (n - i)), -0.1) for i in range(n)]


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    architecture = {"horizon": 384, "num_series": 1, "hidden_size": 4,
                    "context_size": 2, "decoder_size": 8}
    model = PriceMQRNN(**architecture)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.local_mlp[-1].bias.copy_(torch.tensor([-4., -0.4, -4.]))
    metadata = {"format": FORMAT, "feature_names": FEATURE_NAMES, "quantiles": QUANTILES,
                "target": TARGET, "architecture": architecture, "series_ids": ["S-API"],
                "unit": "CNY/kWh", "timezone": "Asia/Shanghai", "context_length": 8,
                "scales": {"S-API": 1.1}, "resolution_minutes": 15,
                "selection_end": "2020-01-01T00:00:00+08:00"}
    path = tmp_path / "mq.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    monkeypatch.setenv("PRICE_MODEL_PATH", str(path))
    return path


def predict(**kwargs):
    options = dict(timezone_name="Asia/Shanghai", series_id="S-API")
    options.update(kwargs)
    return predict_price_mqrnn(history(), START, 96, ISSUED, **options)


def rewrite(path, **changes):
    a = torch.load(path, weights_only=True)
    a["metadata"].update(changes)
    torch.save(a, path)


def test_signed_quantiles_fee_and_future_isolation(artifact):
    a, b = predict(), predict(transmission_and_fund=.21)
    assert len(a.points) == len(a.p10) == len(a.p90) == 96
    assert a.points[0].value < 0
    for lo, p, hi, with_fee in zip(a.p10, a.points, a.p90, b.points):
        assert lo < p.value < hi
        assert with_fee.value - p.value == pytest.approx(.21)
    assert b.components["energy_price_p50"] == [p.value for p in a.points]
    future = predict_price_mqrnn(history() + [(START, 1e9)], START, 96, ISSUED,
                                timezone_name="Asia/Shanghai", series_id="S-API")
    assert a.model_dump() == future.model_dump()
    assert predict(include_uncertainty=False).p10 is None
    assert a.accuracy is None
    assert len(a.provenance.checkpoint_sha256) == 64


@pytest.mark.parametrize("changes,match", [
    ({"unit": "source_price_unit"}, "CNY/kWh"),
    ({"timezone": "UTC"}, "timezone"),
    ({"selection_end": START.isoformat()}, "cutoff"),
    ({"series_ids": ["elsewhere"]}, "series_id"),
    ({"target": "total_retail_price"}, "target"),
    ({"scales": {"S-API": 0}}, "scale"),
    ({"format": "price_deepar_v1"}, "format"),
])
def test_checkpoint_contract(artifact, changes, match):
    rewrite(artifact, **changes)
    with pytest.raises(ValueError, match=match):
        predict()


def test_hourly_expansion_and_direct_horizon_limit(artifact):
    rewrite(artifact, resolution_minutes=60)
    result = predict_price_mqrnn(history(60), START, 288, ISSUED,
                                 timezone_name="Asia/Shanghai", series_id="S-API")
    assert len(result.points) == 288
    assert len({p.value for p in result.points[:4]}) == 1
    assert "原生小时" in result.notes
    with pytest.raises(ValueError, match="horizon"):
        predict_price_mqrnn(history(60), START + timedelta(days=16), 96, ISSUED,
                             timezone_name="Asia/Shanghai", series_id="S-API")


def test_stale_gap_and_insufficient_history(artifact):
    for values, match in [(history()[:-2] + history()[-1:], "needs at least"),
                          ([(t - timedelta(hours=3), v) for t, v in history()], "stale")]:
        with pytest.raises(ValueError, match=match):
            predict_price_mqrnn(values, START, 96, ISSUED,
                                 timezone_name="Asia/Shanghai", series_id="S-API")
    values = history(context=9)
    values.pop(-10)
    with pytest.raises(ValueError, match="continuous"):
        predict_price_mqrnn(values, START, 96, ISSUED,
                             timezone_name="Asia/Shanghai", series_id="S-API")


def test_forking_is_causal_and_equals_shared_per_origin_decoder():
    torch.manual_seed(7)
    model = PriceMQRNN(horizon=4, hidden_size=5, context_size=3, decoder_size=8)
    x = torch.randn(2, 6, 10)
    future = torch.randn(2, 6, 4, 7)
    ids = torch.tensor([0, 0])
    forked = model(x, future, ids)
    for t in range(6):
        state = model.encode(x[:, :t + 1], ids)[:, -1]
        assert torch.allclose(forked[:, t], model.decode(state, future[:, t]), atol=1e-6)
    altered = x.clone()
    altered[:, 3:] += 1e6
    assert torch.allclose(model(altered, future, ids)[:, :3], forked[:, :3], atol=1e-6)
    altered_future = future.clone().requires_grad_()
    model(x, altered_future, ids)[0, 0, 0, 1].backward()
    assert altered_future.grad[0, 0, -1].abs().sum() > 0


def test_pinball_mask_and_gradients():
    prediction = torch.tensor([[[0., 1., 2.], [100., 101., 102.]]], requires_grad=True)
    target = torch.tensor([[1., float("nan")]])
    mask = torch.tensor([[True, False]])
    loss = pinball_loss(prediction, target, mask)
    assert float(loss.detach()) == pytest.approx(.2 / 3)
    loss.backward()
    assert prediction.grad[0, 1].abs().sum() == 0
    assert torch.isfinite(prediction.grad).all()
    model = PriceMQRNN(horizon=3, hidden_size=4, decoder_size=8)
    q = model(torch.randn(1, 4, 10), torch.randn(1, 4, 3, 7), torch.tensor([0]))
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
    pinball_loss(q, torch.randn(1, 4, 3)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_partial_labels_and_train_cutoff_do_not_read_future():
    values = [(START + timedelta(hours=i), float(i)) for i in range(240)]
    cut = START + timedelta(hours=220)
    data = ForkDataset({"A": values}, {"A": 1}, cut, 60, 24, 48)
    altered = [(t, v if t < cut else 1e20) for t, v in values]
    other = ForkDataset({"A": altered}, {"A": 1}, cut, 60, 24, 48)
    for i in range(len(data)):
        assert all(torch.equal(x, y) for x, y in zip(data[i], other[i]))
    x, f, y, mask, _ = data[len(data) - 1]
    assert mask.any() and not mask.all()
    assert mask[0].sum() == 3
    assert (y[mask] < 220).all()
    assert x[0, 0] == 216 and y[0, 0] == 217
    assert x[0, 1] == 192 and x[0, 2] == 48


@pytest.mark.parametrize("mode", ["spot", "tou_plus_spot"])
def test_auto_uses_mqrnn_spot_fee_once(site_client, artifact, mode):
    r = site_client.post("/api/v1/tariffs", json={"tariff_id": "T-MQ", "name": "现货",
        "mode": mode, "transmission_and_fund": .21,
        "tiers": [{"name": "flat", "price": 100, "hours": ["00:00-12:00", "12:00-00:00"]}]})
    assert r.status_code == 201
    assert site_client.patch("/api/v1/sites/S-API", json={"tariff_id": "T-MQ"}).status_code == 200
    r = site_client.post("/api/v1/sites/S-API/measurements", json={"data_type": "price", "unit": "CNY/kWh",
        "points": [{"ts": t.isoformat(), "value": v} for t, v in history()]})
    assert r.status_code == 200
    payload = {"site_id": "S-API", "targets": ["price"], "issued_at": ISSUED.isoformat()}
    r = site_client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["model"]["name"] == "price_mqrnn"
    assert result["points"][0]["value"] == pytest.approx(predict(transmission_and_fund=.21).points[0].value)
    artifact.unlink()
    assert site_client.post("/api/v1/forecast", json=payload).status_code == 422


@pytest.mark.parametrize("province,hours,totals", [
    ("jiangsu", [8, 9, 5, 2], [.56, .86, 1.16, 1.46]),
    ("guangdong", [8, 9, 4, 3], [.49, .86, 1.26, 1.46]),
])
def test_supplied_tou_configs_full_day(province, hours, totals):
    config = Path(__file__).resolve().parents[2] / "configs" / f"tariff.{province}.yaml"
    now = datetime.now(timezone.utc)
    tariff = TariffRead(**yaml.safe_load(config.read_text()), created_at=now, updated_at=now)
    r = forecast_price_tou(tariff, START.date(), 3, CN_TZ)
    assert r.model.name == "tou_policy" and len(r.points) == 288
    assert r.p10 == r.p90 == [p.value for p in r.points]
    from collections import Counter
    assert Counter(p.value for p in r.points[:96]) == {p: h * 4 for p, h in zip(totals, hours)}
    assert r.points[0].value == r.points[96].value == totals[0]
    for i in range(96):
        assert r.points[i].value == r.points[i + 96].value


def test_tou_auto_needs_no_model_or_price_measurements(site_client, monkeypatch):
    monkeypatch.setenv("PRICE_MODEL_PATH", "/does/not/exist")
    from app.domain.forecast import price_mqrnn
    monkeypatch.setattr(price_mqrnn, "torch", None)
    payload = {"site_id": "S-API", "targets": ["price"], "issued_at": ISSUED.isoformat()}
    r = site_client.post("/api/v1/forecast", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["model"]["name"] == "tou_policy"


def test_native_15min_training_export_and_predict(tmp_path):
    values = [(START + timedelta(minutes=15 * i), .1 + .3 * math.sin(i * math.pi / 48))
              for i in range(30 * 96)]
    source = tmp_path / "price.csv"
    source.write_text("ts,price\n" + "\n".join(f"{t.isoformat()},{v}" for t, v in values))
    output = tmp_path / "mq.pt"
    report = train(SimpleNamespace(csv=source, output=output, unit="CNY/kWh", timezone="Asia/Shanghai",
        resolution_minutes=15, series_id="S-API", context_days=1, horizon_days=1,
        epochs=1, batch_size=2, hidden_size=4, lr=1e-3, seed=42, threads=2, skip_gapped_windows=False))
    assert math.isfinite(report["test_metrics"]["MQ-RNN"]["MAE"])
    assert "forking_sequences" in report["metadata"]["training_scheme"]
    assert output.with_suffix(".test.csv").is_file()
    start = START + timedelta(days=30)
    result = predict_price_mqrnn(values, start, 96, start - timedelta(minutes=5),
                                 timezone_name="Asia/Shanghai", series_id="S-API", model_path=str(output))
    assert len(result.points) == 96


def test_bridge_selects_direct_horizons_without_recursive_inputs(artifact, monkeypatch):
    from app.domain.forecast import price_mqrnn as module
    def by_horizon(model, *args, **kwargs):
        y = torch.arange(model.horizon, dtype=torch.float32)
        return torch.stack([y - 1, y, y + 1], -1)
    monkeypatch.setattr(module, "direct_quantiles", by_horizon)
    earlier = [(t - timedelta(hours=12), y) for t, y in history()]
    r = predict_price_mqrnn(earlier, START, 288, ISSUED - timedelta(hours=12),
                            timezone_name="Asia/Shanghai", series_id="S-API")
    assert r.points[0].value == 48 and r.points[-1].value == 335
    assert r.p10[0] == 47 and r.p90[0] == 49


def test_gap_splits_training_and_masks_targets_at_gap():
    values = [(START + timedelta(hours=i), float(i)) for i in range(450) if i != 200]
    data = ForkDataset({"A": values}, {"A": 1}, START + timedelta(days=30), 60, 24, 48)
    assert len(data.segments) == 2
    x, _, y, mask, _ = data[0]
    assert y[mask].max() == 199  # 缺口前不读取缺口后的未来价格
    for i in range(len(data)):
        x, _, _, _, _ = data[i]
        assert torch.all(x[:, 0] - x[:, 1] == 24)
        assert torch.all(x[:, 0] - x[:, 2] == 168)
