"""计划自动化（日前调度 / 早间兜底）的测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.application.automation_service import ensure_daily_plans
from tests.conftest import day_points

CN_TZ = timezone(timedelta(hours=8))


def _seed_history(client: TestClient, site_id: str) -> None:
    now = datetime.now(CN_TZ)
    for offset in range(5, 0, -1):
        day = (now.date() - timedelta(days=offset)).isoformat()
        start = datetime.fromisoformat(f"{day}T00:00:00+08:00")
        client.post(
            f"/api/v1/sites/{site_id}/measurements",
            json={"data_type": "pv_power", "points": day_points(start, 300.0)},
        )
        client.post(
            f"/api/v1/sites/{site_id}/measurements",
            json={"data_type": "load_power", "points": day_points(start, 400.0)},
        )


def _runs_on(client: TestClient, site_id: str, day: str) -> list[dict]:
    runs = client.get(f"/api/v1/sites/{site_id}/dispatch-runs").json()
    return [r for r in runs if r["date"] == day]


def test_day_ahead_generation_at_23(site_client: TestClient) -> None:
    site_id = "S-API"
    _seed_history(site_client, site_id)
    now = datetime.now(CN_TZ)
    tomorrow = (now.date() + timedelta(days=1)).isoformat()

    fake_now = now.replace(hour=23, minute=5, second=0, microsecond=0)
    generated = ensure_daily_plans(site_client.app.state.db, now_override=fake_now)
    assert len(generated) == 1

    runs = _runs_on(site_client, site_id, tomorrow)
    assert len(runs) == 1
    assert runs[0]["auto_generated"] is True

    # 同一晚再跑：不重复生成
    again = ensure_daily_plans(site_client.app.state.db, now_override=fake_now)
    assert again == []
    assert len(_runs_on(site_client, site_id, tomorrow)) == 1


def test_backstop_generation_at_midnight(site_client: TestClient) -> None:
    site_id = "S-API"
    _seed_history(site_client, site_id)
    now = datetime.now(CN_TZ)
    today = now.date().isoformat()

    fake_now = now.replace(hour=0, minute=5, second=0, microsecond=0)
    generated = ensure_daily_plans(site_client.app.state.db, now_override=fake_now)
    assert len(generated) == 1
    runs = _runs_on(site_client, site_id, today)
    assert len(runs) == 1
    assert runs[0]["auto_generated"] is True


def test_no_generation_outside_windows(site_client: TestClient) -> None:
    site_id = "S-API"
    _seed_history(site_client, site_id)
    now = datetime.now(CN_TZ)
    # 非生成窗口（白天 14:00）不生成
    fake_now = now.replace(hour=14, minute=0, second=0, microsecond=0)
    generated = ensure_daily_plans(site_client.app.state.db, now_override=fake_now)
    assert generated == []


def test_manual_plan_not_marked_auto(site_client: TestClient) -> None:
    site_id = "S-API"
    _seed_history(site_client, site_id)
    now = datetime.now(CN_TZ)
    today = now.date().isoformat()
    response = site_client.post(
        "/api/v1/storage/schedule",
        json={
            "site_id": site_id,
            "date": today,
            "initial_soc": 0.5,
            "strategy": "rule_v1",
            "resolution_minutes": 15,
            "source": "forecast",
        },
    )
    assert response.status_code == 200, response.text
    runs = _runs_on(site_client, site_id, today)
    assert len(runs) == 1
    assert runs[0]["auto_generated"] is False

    # 早间兜底发现今日已有计划 → 不再生成
    fake_now = now.replace(hour=0, minute=5, second=0, microsecond=0)
    generated = ensure_daily_plans(site_client.app.state.db, now_override=fake_now)
    assert generated == []
