"""观测体系：预测封存 → 实测对账 → 精度累积。

没有观测的算法调优等于算命——每次发出的预测原样封存，
等实测数据到位后自动对账，沉淀为"方法 × 日期"的精度序列。

电价走单独口径：功率的 skill=1−RMSE/Cap 在电价上没有意义（不存在装机容量
做分母），因此价格用 MAE/RMSE + 相对"昨日同时刻"基准的 skill + PICP。
"""

from __future__ import annotations

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
    completed = set()
    for row in pending:
        target = row["target"]
        target_date = date.fromisoformat(row["target_date"])
        if target == "price":
            key = (row["method"], target_date)
            if key not in completed and _reconcile_price(db, site_id, row, target_date, tz):
                done += 1
                completed.add(key)
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


def _hourly_price_points(points: list[dict], target_date: date, tz) -> list[list[dict]] | None:
    """只接受本地目标日完整的 24/48/96 点网格；裸时间戳不参与对账。"""
    buckets: list[list[dict]] = [[] for _ in range(24)]
    seen = set()
    for point in points:
        try:
            ts = datetime.fromisoformat(point["ts"])
            value = float(point["value"])
        except (KeyError, TypeError, ValueError):
            return None
        if ts.utcoffset() is None:
            return None
        ts = ts.astimezone(tz)
        if ts.date() != target_date:
            continue
        if ts in seen or ts.second or ts.microsecond or not np.isfinite(value):
            return None
        seen.add(ts)
        buckets[ts.hour].append({**point, "minute": ts.minute, "value": value})
    count = len(buckets[0])
    if count not in (1, 2, 4):
        return None
    minutes = set(range(0, 60, 60 // count))
    if any(len(bucket) != count or {p["minute"] for p in bucket} != minutes for bucket in buckets):
        return None
    return buckets


def _reconcile_price(db: Database, site_id: str, row: dict, target_date: date, tz) -> bool:
    """日前封存的真实区间按小时均值对齐；缺失区间明确记为未知。"""
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
    issued_at = datetime.fromisoformat(row["issued_at"])
    if issued_at.utcoffset() is None or issued_at >= day_start:
        return False  # 日内或事后预测不能充当日前样本
    day_end = day_start + timedelta(days=1)
    actual_points = [
        {"ts": ts.isoformat(), "value": value}
        for ts, value, _ in db.get_spot_prices(site_id, day_start, day_end)
    ]
    actual_buckets = _hourly_price_points(actual_points, target_date, tz)
    buckets = _hourly_price_points(row["points"], target_date, tz)
    if actual_buckets is None or buckets is None:
        return False
    actual = np.asarray([np.mean([p["value"] for p in bucket]) for bucket in actual_buckets])
    predicted = np.asarray([np.mean([p["value"] for p in bucket]) for bucket in buckets])

    previous_date = target_date - timedelta(days=1)
    prev_buckets = _hourly_price_points(
        [{"ts": ts.isoformat(), "value": value}
         for ts, value, _ in db.get_spot_prices(site_id, day_start - timedelta(days=1), day_start)],
        previous_date, tz,
    )
    baseline = None if prev_buckets is None else np.asarray(
        [np.mean([p["value"] for p in bucket]) for bucket in prev_buckets]
    )
    quantiles = None
    try:
        valid_intervals = all(
            np.isfinite(float(p["p10"])) and np.isfinite(float(p["p90"]))
            and float(p["p10"]) <= p["value"] <= float(p["p90"])
            for bucket in buckets for p in bucket
        )
        if valid_intervals and row["method"] != "tou_policy":
            quantiles = np.asarray([
                [np.mean([float(p["p10"]) for p in bucket]), predicted[h],
                 np.mean([float(p["p90"]) for p in bucket])]
                for h, bucket in enumerate(buckets)
            ])
    except (KeyError, TypeError, ValueError):
        pass
    metrics = price_summarize(actual, predicted, baseline=baseline, quantile_pred=quantiles)
    # 点预测或政策表不能伪装成名义 80% 的概率区间。
    if quantiles is None:
        for name in ("picp80", "width80", "winkler80", "pinball"):
            metrics[name] = None
    metrics.update(
        baseline="naive_daily", interval_coverage_version=1,
        interval_coverage=metrics["picp80"],
        interval_n_points=24 if quantiles is not None else 0,
    )
    # 无可比基准（例如昨日完全恒价）产生的 NaN 不能进入 JSON 响应。
    metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    db.upsert_forecast_accuracy(
        site_id=site_id, target="price", method=row["method"],
        target_date=target_date, metrics=metrics,
    )
    return True


def price_interval_status(db: Database, site_id: str, method: str) -> dict:
    """同一方法最近日覆盖率与连续低覆盖告警；缺日、未知或过期均不拼接。"""
    rows = [r for r in db.list_forecast_accuracy(site_id, target="price") if r["method"] == method]
    site = db.get_site(site_id)
    yesterday = datetime.now(_zone(site.location.timezone)).date() - timedelta(days=1)
    rows = sorted((r for r in rows if date.fromisoformat(r["target_date"]) <= yesterday),
                  key=lambda r: r["target_date"], reverse=True)
    latest = rows[0] if rows else None
    coverage = latest["metrics"].get("interval_coverage") if latest else None
    low_days = 0
    expected = yesterday
    for row in rows:
        value = row["metrics"].get("interval_coverage")
        if (date.fromisoformat(row["target_date"]) != expected or value is None
                or not np.isfinite(value) or value >= 0.75):
            break
        low_days += 1
        expected -= timedelta(days=1)
    return {
        "interval_coverage": coverage,
        "evaluated_on": latest["target_date"] if latest else None,
        "consecutive_low_days": low_days,
        "warning": "price_interval_undercoverage" if low_days >= 3 else None,
    }


def accuracy_summary(db: Database, site_id: str, target: str | None, days: int) -> dict:
    site = db.get_site(site_id)
    since = datetime.now(_zone(site.location.timezone)).date() - timedelta(days=days)
    rows = db.list_forecast_accuracy(site_id, target=target, since=since)
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(f"{row['target']}:{row['method']}", []).append(row["metrics"])
    summary = {}
    for key, metrics_list in by_method.items():
        n = len(metrics_list)
        if key.startswith("price:"):
            valid_coverage = [m["interval_coverage"] for m in metrics_list
                              if m.get("interval_coverage") is not None]
            status = price_interval_status(db, site_id, key.split(":", 1)[1])
            summary[key] = {
                "days": n,
                "mae": round(sum(m.get("mae", 0.0) for m in metrics_list) / n, 4),
                "rmse": round(sum(m.get("rmse", 0.0) for m in metrics_list) / n, 4),
                "skill_all_day": round(
                    sum(m.get("skill", 0.0) for m in metrics_list if m.get("skill") is not None) / n, 4
                ),
                "picp80": (round(sum(valid_coverage) / len(valid_coverage), 4)
                           if valid_coverage else None),
                "interval_coverage": (round(sum(valid_coverage) / len(valid_coverage), 4)
                                      if valid_coverage else None),
                "interval_days": len(valid_coverage),
                "warning": status["warning"],
                "consecutive_low_days": status["consecutive_low_days"],
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
