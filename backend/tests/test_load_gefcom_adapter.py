import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from app.domain.forecast.features import build_windows
from app.domain.forecast.load_gefcom_adapter import (
    GEFComEliteEnergyTaskNet, adapter_batch, adapter_stats)
from app.domain.forecast.load_advanced import EliteEnergyTaskNet


def windows():
    n = 1000
    return build_windows(pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                                       "load": np.arange(n, dtype=float), "w1": np.arange(n, dtype=float)}), "load", 4, 48)


def test_original_modules_and_unknown_mes():
    model = GEFComEliteEnergyTaskNet(8, 6)
    assert model.history_lstm.hidden_size == 128
    assert model.history_lstm.num_layers == 2
    assert model.history_lstm.dropout == .3
    assert not model.mes_embedding.weight.requires_grad
    assert torch.count_nonzero(model.mes_embedding.weight) == 0
    assert all(not p.requires_grad for p in model.workshop_load_head.parameters())
    total, workshop, attention = model(torch.randn(2, 48, 8), torch.zeros(2, 48, dtype=torch.long),
                                      torch.randn(2, 4, 6), torch.zeros(2, 4, dtype=torch.long))
    assert total.shape == workshop.shape == (2, 4)
    assert attention.shape == (2, 48, 1)
    assert torch.allclose(attention.sum(1), torch.ones(2, 1))


def test_future_weather_and_test_values_not_used_for_normalization():
    data = windows(); before = adapter_stats(data)
    te = data.indices("test")
    data.y[te] = 1e9; data.seq[te] = 1e9; data.exog[te] = 1e9
    assert adapter_stats(data) == before
    batch = adapter_batch(data, before, data.indices("validation")[:2])
    assert batch[2].shape == (2, 4, 6)
    assert torch.count_nonzero(batch[1]) == torch.count_nonzero(batch[3]) == 0


def test_invalid_mes_rejected_and_no_auxiliary_gradient():
    model = GEFComEliteEnergyTaskNet(8, 6)
    args = (torch.randn(2, 48, 8), torch.zeros(2, 48, dtype=torch.long),
            torch.randn(2, 1, 6), torch.zeros(2, 1, dtype=torch.long))
    total, _, _ = model(*args)
    total.square().mean().backward()
    assert model.history_lstm.weight_ih_l0.grad is not None
    assert model.attention.query.weight.grad is not None
    assert model.workshop_load_head.weight.grad is None
    with pytest.raises(ValueError, match="MES labels"):
        model(args[0], args[1] + 1, args[2], args[3])


def test_same_original_contract_gives_identical_network_outputs():
    adapted = GEFComEliteEnergyTaskNet(8, 6).eval()
    original = EliteEnergyTaskNet(8, 6, num_mes_plans=1).eval()
    original.load_state_dict(adapted.state_dict(), strict=True)
    inputs = (torch.randn(2, 48, 8), torch.zeros(2, 48, dtype=torch.long),
              torch.randn(2, 96, 6), torch.zeros(2, 96, dtype=torch.long))
    with torch.no_grad():
        expected, actual = original(*inputs), adapted(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
