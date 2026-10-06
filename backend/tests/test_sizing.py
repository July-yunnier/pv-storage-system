from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points

SIZING_URL = "/api/v1/storage/sizing"
RUNS_URL = "/api/v1/sites/S-API/sizing-runs"
START = datetime(2026, 9, 17, tzinfo=CN_TZ)


def seed_days(client: TestClient, days: int = 3) -> None:
    for offset in range(days):
        day = START + timedelta(days=offset)
        for data_type, value in [("pv_power", 500), ("load_power", 600)]:
            response = client.post(
                "/api/v1/sites/S-API/measurements",
                json={
                    "data_type": data_type,
                    "unit": "kW",
                    "resolution_minutes": 15,
                    "points": day_points(day, value),
                },
            )
            assert response.status_code == 200, response.text


def post_sizing(client: TestClient, **overrides) -> object:
    payload = {
        "site_id": "S-API",
        "candidates": {"pcs_kw": [100, 200, 500], "duration_hours": [1, 2]},
        "representative_days": 3,
    }
    payload.update(overrides)
    return client.post(SIZING_URL, json=payload)


def test_sizing_grid_and_contract(site_client: TestClient) -> None:
    seed_days(site_client)
    response = post_sizing(site_client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"].startswith("Z")
    assert len(body["candidates"]) == 6  # 3 × 2
    capacities = {(c["pcs_kw"], c["capacity_kwh"], c["duration_h"]) for c in body["candidates"]}
    assert (500, 1000, 2) in capacities
    assert body["representative_dates"] == ["2026-09-17", "2026-09-18", "2026-09-19"]
    for candidate in body["candidates"]:
        for field in (
            "capex_yuan", "annual_benefit_yuan", "annual_opex_yuan",
            "payback_years", "npv_yuan", "irr", "equivalent_full_cycles_per_day",
        ):
            assert field in candidate
    # assumptions 口径齐全
    assumptions = body["assumptions"]
    for key in (
        "annualization", "representative_days", "initial_soc", "strategy",
        "battery_price_per_kwh", "pcs_price_per_kw", "install_ratio", "opex_ratio",
        "project_years", "battery_life_years", "replacement_cost_ratio",
        "discount_rate", "residual_ratio",
    ):
        assert key in assumptions
    assert "365" in assumptions["annualization"]


def test_sizing_recommended_is_max_positive_npv(site_client: TestClient) -> None:
    seed_days(site_client)
    body = post_sizing(site_client).json()
    profitable = [c for c in body["candidates"] if c["npv_yuan"] > 0]
    if profitable:
        best = max(profitable, key=lambda c: c["npv_yuan"])
        assert body["recommended"] == best
        # 敏感性三档且只针对推荐方案
        assert len(body["sensitivity"]["price_spread"]) == 3
        assert len(body["sensitivity"]["battery_price"]) == 3
        assert [e["delta"] for e in body["sensitivity"]["price_spread"]] == [-0.2, 0.0, 0.2]
    else:
        assert body["recommended"] is None
        assert body["notes"]
        assert body["sensitivity"]["price_spread"] == []


def test_sizing_deterministic(site_client: TestClient) -> None:
    seed_days(site_client)
    first = post_sizing(site_client).json()
    second = post_sizing(site_client).json()
    assert first["candidates"] == second["candidates"]
    assert first["recommended"] == second["recommended"]
    assert first["sensitivity"] == second["sensitivity"]
    assert first["run_id"] != second["run_id"]


def test_sizing_financial_overrides(site_client: TestClient) -> None:
    seed_days(site_client)
    body = post_sizing(
        site_client, financial={"battery_price_per_kwh": 600, "discount_rate": 0.05}
    ).json()
    assert body["assumptions"]["battery_price_per_kwh"] == 600
    assert body["assumptions"]["discount_rate"] == 0.05
    # 电池单价减半，CAPEX 应低于默认值情形
    default_body = post_sizing(site_client).json()
    cheap = {c["capacity_kwh"]: c["capex_yuan"] for c in body["candidates"]}
    normal = {c["capacity_kwh"]: c["capex_yuan"] for c in default_body["candidates"]}
    assert cheap[1000] < normal[1000]


def test_sizing_runs_list_and_detail(site_client: TestClient) -> None:
    seed_days(site_client)
    body = post_sizing(site_client).json()
    listing = site_client.get(RUNS_URL).json()
    assert [r["run_id"] for r in listing] == [body["run_id"]]
    summary = listing[0]
    if body["recommended"]:
        assert summary["recommended_pcs_kw"] == body["recommended"]["pcs_kw"]
        assert summary["recommended_capacity_kwh"] == body["recommended"]["capacity_kwh"]
        assert summary["npv_yuan"] == body["recommended"]["npv_yuan"]
    detail = site_client.get(f"{RUNS_URL}/{body['run_id']}").json()
    assert detail["run_id"] == body["run_id"]
    assert detail["input_digest"]
    assert len(detail["candidates"]) == 6
    assert detail["assumptions"] == body["assumptions"]


def test_sizing_same_input_same_digest(site_client: TestClient) -> None:
    seed_days(site_client)
    first = post_sizing(site_client).json()
    second = post_sizing(site_client).json()
    d1 = site_client.get(f"{RUNS_URL}/{first['run_id']}").json()
    d2 = site_client.get(f"{RUNS_URL}/{second['run_id']}").json()
    assert d1["input_digest"] == d2["input_digest"]


def test_sizing_no_representative_days_422(site_client: TestClient) -> None:
    response = post_sizing(site_client)  # 无任何测量数据
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "DATA_QUALITY_ERROR"
    assert "representative days insufficient" in error["message"]


def test_sizing_grid_too_large_422(site_client: TestClient) -> None:
    seed_days(site_client)
    response = post_sizing(
        site_client,
        candidates={"pcs_kw": [float(i) for i in range(1, 12)], "duration_hours": [float(i) for i in range(1, 11)]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_sizing_unknown_site_404(site_client: TestClient) -> None:
    response = post_sizing(site_client, site_id="S-NOPE")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
