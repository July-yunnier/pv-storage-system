"""观测体系：预测封存 → 实测对账 → 精度累积。

没有观测的算法调优等于算命——每次发出的预测原样封存，
等实测数据到位后自动对账，沉淀为"方法 × 日期"的精度序列。

电价走单独口径：功率的 skill=1−RMSE/Cap 在电价上没有意义（不存在装机容量
做分母），因此价格用 MAE/RMSE + 相对"昨日同时刻"基准的 skill + PICP。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np

from app.domain.experiment.metrics import forecast_metrics
from app.domain.forecast.price_metrics import summarize as price_summarize
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
        if target == "price":
            if _reconcile_price(db, site_id, row, target_date, tz):
                done += 1
            continue
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


def _reconcile_price(db: Database, site_id: str, row: dict, target_date: date, tz) -> bool:
    """电价对账：按小时对齐封存曲线与现货实测，基准为昨日同时刻。"""
    points = sorted(row["points"], key=lambda p: p["ts"])
    if not points:
        return False
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
    day_end = day_start + timedelta(days=1)
    rows = db.get_spot_prices(site_id, day_start, day_end)
    if len(rows) < 24:
        return False
    hourly = {}
    for ts, value, _ in rows:
        if value is not None:
            hourly[ts.astimezone(tz).hour] = float(value)
    if len(hourly) < 24:
        return False

    # 15 分钟封存点按小时取均值，与小时级实测对齐
    buckets: dict[int, list[float]] = {}
    for point in points:
        ts = datetime.fromisoformat(point["ts"]).astimezone(tz)
        buckets.setdefault(ts.hour, []).append(float(point["value"]))
    predicted = np.asarray([np.mean(buckets[h]) for h in range(24) if h in buckets])
    hours = [h for h in range(24) if h in buckets]
    if len(hours) < 24:
        return False
    actual = np.asarray([hourly[h] for h in hours])

    prev = db.get_spot_prices(site_id, day_start - timedelta(days=1), day_start)
    prev_map = {ts.astimezone(tz).hour: value for ts, value, _ in prev if value is not None}
    baseline = np.asarray([prev_map.get(h, np.nan) for h in hours])
    if not np.isfinite(baseline).all():
        baseline = None

    metrics = price_summarize(actual, predicted, baseline=baseline)
    metrics["baseline"] = "naive_daily"
    db.upsert_forecast_accuracy(
        site_id=site_id, target="price", method=row["method"],
        target_date=target_date, metrics=metrics,
    )
    return True


def accuracy_summary(db: Database, site_id: str, target: str | None, days: int) -> dict:
    since = datetime.now(timezone.utc).date() - timedelta(days=days)
    rows = db.list_forecast_accuracy(site_id, target=target, since=since)
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(f"{row['target']}:{row['method']}", []).append(row["metrics"])
    summary = {}
    for key, metrics_list in by_method.items():
        n = len(metrics_list)
        if key.startswith("price:"):
            summary[key] = {
                "days": n,
                "mae": round(sum(m.get("mae", 0.0) for m in metrics_list) / n, 4),
                "rmse": round(sum(m.get("rmse", 0.0) for m in metrics_list) / n, 4),
                "skill_all_day": round(
                    sum(m.get("skill", 0.0) for m in metrics_list if m.get("skill") is not None) / n, 4
                ),
                "picp80": round(sum(m.get("picp80", 0.0) for m in metrics_list) / n, 4),
                "bias": round(sum(m.get("bias", 0.0) for m in metrics_list) / n, 4),
            }
            continue
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
