"""算法注册表与 /algorithms 发现接口的测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.application import algorithm_registry


def test_catalog_covers_three_targets() -> None:
    catalog = algorithm_registry.catalog()
    assert len(catalog) == 15
    assert {a.target for a in catalog} == {"pv_power", "load_power", "price"}


def test_catalog_filter_by_target() -> None:
    pv = algorithm_registry.catalog("pv_power")
    assert pv and all(a.target == "pv_power" for a in pv)


def test_product_options() -> None:
    assert algorithm_registry.product_options("pv_power") == ["statistical", "physics", "ampdnet"]
    assert algorithm_registry.product_options("load_power") == ["statistical", "advanced"]
    assert algorithm_registry.product_options("price") == ["tou", "statistical", "lgbm"]


def test_validate_product_choice_accepts_known() -> None:
    algorithm_registry.validate_product_choice("pv_power", "statistical")
    algorithm_registry.validate_product_choice("load_power", "advanced")


def test_validate_product_choice_rejects_unknown_with_options() -> None:
    with pytest.raises(ValueError, match="statistical"):
        algorithm_registry.validate_product_choice("pv_power", "transformer_x")


def test_replay_methods_match_legacy_matrix() -> None:
    """注册表派生的回放矩阵必须与旧硬编码清单一致（行为不变）。"""
    assert algorithm_registry.replay_methods("pv_power") == [
        "day_persistence",
        "slot_mean",
        "slot_mean_bias",
        "physics",
        "blend",
        "oracle",
        "ampdnet",
    ]


def test_catalog_view_fields_and_availability() -> None:
    view = algorithm_registry.catalog_view()
    assert len(view) == 15
    for entry in view:
        assert {"name", "target", "kind", "available", "unavailable_reason"} <= set(entry)
    advanced = next(e for e in view if e["name"] == "load_mes_multitask_attention")
    # 本机装有 torch 与工件：advanced 可用；缺环境的机器上则为 unavailable 且给出原因
    if advanced["available"]:
        assert advanced["unavailable_reason"] is None
    else:
        assert advanced["unavailable_reason"]
    ampdnet = next(e for e in view if e["name"] == "ampdnet")
    assert ampdnet["requires"] == {"history_days": 0, "weather": False, "torch": True, "artifact": True}


def test_algorithms_api(client: TestClient) -> None:
    response = client.get("/api/v1/algorithms")
    assert response.status_code == 200, response.text
    algorithms = response.json()["algorithms"]
    assert len(algorithms) == 15
    names = {a["name"] for a in algorithms}
    assert {"blend", "load_mes_multitask_attention", "tou_policy", "ampdnet"} <= names


def test_forecast_rejects_unknown_model_key(site_client: TestClient) -> None:
    response = site_client.post(
        "/api/v1/forecast",
        json={
            "site_id": "S-API",
            "targets": ["pv_power"],
            "horizon_hours": 24,
            "options": {"pv_model": "transformer_x"},
        },
    )
    assert response.status_code == 422, response.text
    assert "transformer_x" in response.text
