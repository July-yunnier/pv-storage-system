"""观测体系：预测封存 → 实测对账 → 精度累积。

没有观测的算法调优等于算命——每次发出的预测原样封存，
等实测数据到位后自动对账，沉淀为"方法 × 日期"的精度序列。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.domain.experiment.metrics import forecast_metrics
from app.infrastructure.db import Database, _zone

TARGET_CAPS = {"pv_power": "pv", "load_power": "load"}


def reconcile_site(db: Database, site_id: str) -> int:
    """对有实测覆盖的封存预测逐条对账，返回新增对账条数。"""
    site = db.get_site(site_id)
    if site is None:
        return 0
    tz = _zone(site.location.timezone)
    today = datetime.now(tz).date()
    pending = db.get_unreconciled_forecasts(site_id, today - timedelta(days=1))
    done = 0
    for row in pending:
        target = row["target"]
        target_date = date.fromisoformat(row["target_date"])
        rows = db.get_measurements_for_date(site_id, target, target_date, site.location.timezone)
        if len(rows) != 96 or any(r.value is None for r in rows):
            continue  # 实测尚未齐，等下一轮
        actual = [r.value for r in rows]
        points = sorted(row["points"], key=lambda p: p["ts"])
        if len(points) != 96:
            continue  # 日内修正等不完整窗口不参与日级对账
        predicted = [float(p["value"]) for p in points]
        cap = site.pv.capacity_kwp if target == "pv_power" else (site.load.peak_kw or 1.0)
        db.upsert_forecast_accuracy(
            site_id=site_id, target=target, method=row["method"],
            target_date=target_date, metrics=forecast_metrics(predicted, actual, cap),
        )
        done += 1
    return done


def accuracy_summary(db: Database, site_id: str, target: str | None, days: int) -> dict:
    since = datetime.now(timezone.utc).date() - timedelta(days=days)
    rows = db.list_forecast_accuracy(site_id, target=target, since=since)
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(f"{row['target']}:{row['method']}", []).append(row["metrics"])
    summary = {}
    for key, metrics_list in by_method.items():
        n = len(metrics_list)
        summary[key] = {
            "days": n,
            "skill_all_day": round(sum(m["skill_all_day"] for m in metrics_list) / n, 4),
            "skill_daylight": round(
                sum(m["skill_daylight"] or 0 for m in metrics_list) / n, 4
            ),
            "rmse": round(sum(m["rmse"] for m in metrics_list) / n, 2),
            "bias": round(sum(m["bias"] for m in metrics_list) / n, 2),
        }
    return {"site_id": site_id, "days": days, "rows": rows, "summary": summary}
