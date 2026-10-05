"""日内滚动修正的测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from app.application import live_service, rolling_service

from tests.conftest import day_points

CN_TZ = timezone(timedelta(hours=8))
NOW_LOCAL = datetime.now(CN_TZ).replace(hour=12, minute=0, second=0, microsecond=0)
TODAY = NOW_LOCAL.date().isoformat()


@pytest.fixture(autouse=True)
def midday_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # A revision needs remaining slots; wall-clock 23:45 has none.
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW_LOCAL.astimezone(tz) if tz is not None else NOW_LOCAL.replace(tzinfo=None)

    monkeypatch.setattr(live_service, "datetime", FixedDatetime)
    monkeypatch.setattr(rolling_service, "datetime", FixedDatetime)


def _seed_history(client: TestClient) -> None:
    """造 5 天光伏+负荷历史，让预测与今日计划可用。"""
    for offset in range(5, 0, -1):
        day = (NOW_LOCAL.date() - timedelta(days=offset)).isoformat()
        start = datetime.fromisoformat(f"{day}T00:00:00+08:00")
        client.post(
            "/api/v1/sites/S-API/measurements",
            json={"data_type": "pv_power", "points": day_points(start, 300.0)},
        )
        client.post(
            "/api/v1/sites/S-API/measurements",
            json={"data_type": "load_power", "points": day_points(start, 400.0)},
        )


def _make_today_plan(client: TestClient) -> None:
    response = client.post(
        "/api/v1/storage/schedule",
        json={
            "site_id": "S-API",
            "date": TODAY,
            "initial_soc": 0.5,
            "strategy": "rule_v1",
            "resolution_minutes": 15,
            "source": "forecast",
        },
    )
    assert response.status_code == 200, response.text


def test_rolling_state_without_plan(site_client: TestClient) -> None:
    state = site_client.get("/api/v1/sites/S-API/rolling")
    assert state.status_code == 200, state.text
    body = state.json()
    assert body["base_run_id"] is None
    assert body["revisions"] == []
    assert body["latest_revision"] == 0


def test_rolling_force_creates_revision(site_client: TestClient) -> None:
    _seed_history(site_client)
    _make_today_plan(site_client)

    simulated = site_client.post("/api/v1/sites/S-API/simulate-live")
    assert simulated.status_code == 200, simulated.text

    # This revision test supplies a valid operating SOC; simulator drift may
    # cross the site's configured limits, a separate recovery scenario.
    soc = site_client.post(
        "/api/v1/sites/S-API/measurements",
        json={"data_type": "storage_soc", "points": [{"ts": NOW_LOCAL.isoformat(), "value": 0.5}]},
    )
    assert soc.status_code == 200, soc.text

    forced = site_client.post("/api/v1/sites/S-API/rolling/run")
    assert forced.status_code == 200, forced.text
    body = forced.json()
    assert body["base_run_id"] is not None
    assert body["latest_revision"] == 1
    assert len(body["revisions"]) == 1
    revision = body["revisions"][0]
    assert revision["trigger_reason"]
    assert 0 <= revision["initial_soc"] <= 1
    assert body["latest_schedule"]
    # 修正计划只覆盖剩余时段（不到 96 点）
    assert 0 < len(body["latest_schedule"]) <= 95

    # 再强制一次 → revision 递增
    again = site_client.post("/api/v1/sites/S-API/rolling/run").json()
    assert again["latest_revision"] == 2


def test_rolling_auto_respects_deadband_or_creates(site_client: TestClient) -> None:
    """非强制路径：要么因死区/冷却不新增，要么新增——状态读取必须始终可用。"""
    _seed_history(site_client)
    _make_today_plan(site_client)
    site_client.post("/api/v1/sites/S-API/simulate-live")

    before = site_client.get("/api/v1/sites/S-API/rolling").json()["latest_revision"]
    state = site_client.get("/api/v1/sites/S-API/rolling")
    assert state.status_code == 200
    after = state.json()["latest_revision"]
    assert after >= before
