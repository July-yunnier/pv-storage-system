"""Tests for native probabilities, chronology and frozen PV inference."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.domain.forecast.features import build_windows, regression_metrics
from app.domain.forecast.quantile import (
    QuantileNet, QuantileSettings, batch_from, fit_quantile,
    make_stats, pinball_loss, pinball_numpy, predict, rolling_stats, set_seed,
)
from app.domain.forecast.frozen_pv import infer_frozen_pv, load_frozen_pv_artifact, FrozenPVArtifactError


def windows():
    ts = pd.date_range("2013-01-01", periods=24 * 35, freq="h", tz="UTC")
    raw = pd.DataFrame({"ts": ts, "price": 20 + np.sin(np.arange(len(ts)) / 24),
                        "forecast_total_load": np.full(len(ts), 100.)})
    return build_windows(raw, "price", 4)


def test_rolling_stats_causal_and_correct():
    x = np.array([[1., 2., 3., 4.], [5., 4., 3., 2.]])
    got = rolling_stats(x, 2)
    np.testing.assert_allclose(got[0, :, 0], [1, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(got[0, :, 1], [0, .5, .5, .5])
    later = x.copy(); later[:, 2:] += 50
    np.testing.assert_array_equal(got[:, :2], rolling_stats(later, 2)[:, :2])


@pytest.mark.parametrize("kind", ["LSTM", "TCN"])
def test_native_head_order_gradients_reload(kind):
    torch = pytest.importorskip("torch")
    set_seed()
    net = QuantileNet(kind, 9, 8, 4, .1)
    batch = (torch.randn(3, 48, 1), torch.randn(3, 48, 2), torch.randn(3, 52, 6), torch.empty(3, 48, 0))
    q = net(batch)
    assert q.shape == (3, 4, 3)
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()
    loss = pinball_loss(q[:, -1:], torch.randn(3, 1)); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    net.eval(); copy = QuantileNet(kind, 9, 8, 4, .1); copy.load_state_dict(net.state_dict()); copy.eval()
    torch.testing.assert_close(net(batch), copy(batch))
    np.testing.assert_allclose(pinball_numpy(np.zeros(3), np.zeros((3, 3))), 0)
    with pytest.raises(ValueError): pinball_numpy([1], [[np.nan, 1, 2]])


def test_normalization_train_only_and_no_price_cap():
    data = windows(); ids = data.indices("train")
    before = make_stats(data, ids, False)
    data.seq[data.indices("test")] += 1e5
    data.y[data.indices("test")] += 1e5
    assert before == make_stats(data, ids, False)
    assert batch_from(data, before, ids[:2])[-1].shape[-1] == 0
    assert "Cap" not in regression_metrics(np.array([1.]), np.array([2.]))
    with pytest.raises(ValueError): QuantileSettings(max_epochs=3)


def test_early_stop_reproducible_and_never_uses_test(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import app.domain.forecast.quantile as module
    data = windows()
    data.y[data.indices("test")] = np.nan
    original = module.batch_from
    def guard(data_arg, stats, ids):
        assert not np.isin(ids, data_arg.indices("test")).any()
        return original(data_arg, stats, ids)
    monkeypatch.setattr(module, "batch_from", guard)
    settings = QuantileSettings(max_epochs=10, patience=2, min_delta=1e6, hidden=4, batch_size=256)
    results = []
    for label in ("a", "b"):
        net, stats, meta = fit_quantile("LSTM", data, settings, tmp_path / label)
        assert meta["stop_reason"] == "validation_early_stop"
        assert meta["epochs_completed"] == 3
        assert meta["best_epoch"] == 1
        results.append(predict(net, data, stats, data.indices("validation"), 256))
        assert (tmp_path / label / "model.pt").exists()
    np.testing.assert_array_equal(*results)


def test_pv_inference_does_not_change_weights_or_file(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "frozen.pt"; torch.save(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.json").write_text(json.dumps({"sha256": digest}))
    state, _ = load_frozen_pv_artifact(path)
    model.load_state_dict(state)
    output = infer_frozen_pv(model, torch.ones(1, 2))
    assert not output.requires_grad
    for key, value in model.state_dict().items(): torch.testing.assert_close(value, state[key])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FrozenPVArtifactError): load_frozen_pv_artifact(tmp_path / "missing.pt")


def test_tcn_encoder_is_causal_and_future_calendar_unused():
    torch = pytest.importorskip("torch")
    set_seed()
    net = QuantileNet("TCN", 7, 8, 4, .1); net.eval()
    x = torch.randn(2, 48, 7)
    changed = x.clone(); changed[:, 20:] += 100
    with torch.no_grad():
        torch.testing.assert_close(net.encoder(x.transpose(1, 2))[:, :, :20],
                                   net.encoder(changed.transpose(1, 2))[:, :, :20])
        batch = (torch.randn(2, 48, 1), torch.empty(2, 48, 0), torch.randn(2, 52, 6), torch.empty(2, 48, 0))
        cal = batch[2].clone(); cal[:, 48:] += 100
        torch.testing.assert_close(net(batch), net((batch[0], batch[1], cal, batch[3])))


def test_resume_must_match_settings_protocol_and_checkpoint(tmp_path):
    from dataclasses import asdict
    from scripts.run_audited_preexperiments import fit_trial
    settings = QuantileSettings()
    (tmp_path / "model.pt").write_bytes(b"test-checkpoint")
    digest = hashlib.sha256(b"test-checkpoint").hexdigest()
    info = {"fingerprint": "frozen", "settings": asdict(settings), "weight_sha256": digest}
    (tmp_path / "completed.json").write_text(json.dumps(info))
    assert fit_trial(None, "TCN", settings, tmp_path, "frozen", True) == info
    with pytest.raises(ValueError, match="differs"):
        fit_trial(None, "TCN", settings, tmp_path, "different", True)
    (tmp_path / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_trial(None, "TCN", settings, tmp_path, "frozen", True)
