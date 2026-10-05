"""观测体系测试：预测封存 → 实测对账 → 精度查询闭环。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from fastapi.testclient import TestClient

from app.application import observability_service
from app.infrastructure.db import Database
from tests.conftest import CN_TZ, SITE_PAYLOAD
from tests.test_replay import _seed_history, HISTORY_DAYS, START_DATE, TEST_DAYS


def _setup(client: TestClient, db: Database) -> None:
    client.post("/api/v1/sites", json=SITE_PAYLOAD)
    _seed_history(db, SITE_PAYLOAD["site_id"])


def test_forecast_archived_and_reconciled(client: TestClient, tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))
    db.init_schema()
    _setup(client, db)
    site_id = SITE_PAYLOAD["site_id"]

    # 发一次日前预测（目标日 = 本地明天）：应被封存
    # 历史数据停在 2026-03，日前统计法拿不到近日历史，故用 physics（无需历史，走 mock 气象）
    response = client.post("/api/v1/forecast", json={
        "site_id": site_id, "targets": ["pv_power"],
        "horizon_hours": 24, "resolution_minutes": 15,
        "options": {"pv_model": "physics"},
    })
    assert response.status_code == 200, response.text
    pending = db.get_unreconciled_forecasts(site_id, date(2100, 1, 1))
    assert len(pending) == 1
    assert pending[0]["target"] == "pv_power"
    assert pending[0]["points"]  # 96 点已封存


def test_reconcile_computes_accuracy(client: TestClient, tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))
    db.init_schema()
    _setup(client, db)
    site_id = SITE_PAYLOAD["site_id"]

    # 手工封存一条"历史日期的预测"（值取真实光伏日形的 0.9 倍），随后对账
    day = START_DATE + timedelta(days=HISTORY_DAYS)  # 测试窗口第一天
    start = datetime.combine(day, time.min, tzinfo=CN_TZ)
    from tests.test_replay import _day_pv

    db.insert_forecast_archive(
        site_id=site_id, target="pv_power", method="manual_test",
        target_date=day, issued_at=datetime.now(),
        points=[
            {"ts": (start + timedelta(minutes=15 * i)).isoformat(),
             "value": _day_pv(i) * 0.9}
            for i in range(96)
        ],
        provenance=None,
    )
    done = observability_service.reconcile_site(db, site_id)
    assert done == 1
    rows = db.list_forecast_accuracy(site_id, target="pv_power")
    assert len(rows) == 1
    metrics = rows[0]["metrics"]
    assert metrics["bias"] < 0  # 预测低 10%
    assert metrics["skill_all_day"] > 0.9
    # 幂等：再对账不重复
    assert observability_service.reconcile_site(db, site_id) == 0


def test_accuracy_api(client: TestClient, tmp_path) -> None:
    db = Database(str(tmp_path / "test.db"))
    db.init_schema()
    _setup(client, db)
    site_id = SITE_PAYLOAD["site_id"]
    response = client.get(f"/api/v1/sites/{site_id}/accuracy", params={"days": 30})
    assert response.status_code == 200
    body = response.json()
    assert body["site_id"] == site_id
    assert "summary" in body and "rows" in body

    bad = client.get(f"/api/v1/sites/{site_id}/accuracy", params={"target": "grid_power"})
    assert bad.status_code == 422
