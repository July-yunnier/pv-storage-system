"""30 天历史滚动回放：预测方法 × 调度策略的公平对比实验。

防泄漏铁律：目标日 D 的预测只能使用 D 之前的测量与气象代理数据。
气象口径：历史日期无法获得当年真实预报，physics/blend 使用 Open-Meteo
archive 再分析作为"天气预报代理"，导出结果中标注 weather_proxy=reanalysis。

策略：
- d0_no_storage      不配储能基准（与预测无关，只跑一次）
- d1_fixed_curve     固定计划曲线（谷段充电、尖峰放电，不看预测）
- d4_rule_predictive 规则引擎 + 预测输入（每个光伏预测方法各跑一次）
- d5_rule_oracle     规则引擎 + 完美预测（理论上界，不作为产品能力）

SOC 跨日连续传递：每个 (预测方法 × 策略) 组合是一条独立的 30 天连续仿真。
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from app.api import errors
from app.application.algorithm_registry import replay_methods
from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.experiment.execute import ExecutionResult, execute_plan, no_storage_baseline
from app.domain.experiment.metrics import forecast_metrics
from app.domain.forecast.load import predict_load_profile
from app.domain.forecast.pv import predict_pv_diurnal, predict_pv_physics
from app.domain.forecast.blend import blend_pv_forecast
from app.domain.forecast.stats import daily_series, history_dates, slot_means
from app.domain.models import MeasurementRead, PricePoint, SiteRead
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database, _zone
from app.infrastructure.weather.base import WeatherPoint

# 回放方法矩阵由算法注册表派生（登记顺序即矩阵顺序），新增算法在注册表登记 replay=True 即入矩阵
PV_METHODS = replay_methods("pv_power")
STRATEGIES = ["d0_no_storage", "d1_fixed_curve", "d4_rule_predictive", "d5_rule_oracle"]
N_DAYS = 12
MIN_HISTORY = 3
RESOLUTION_MINUTES = 15
SLOTS = 96


def _as_measurements(data_type: str, ts_list: list[datetime], values: list[float]) -> list[MeasurementRead]:
    return [
        MeasurementRead(
            data_type=data_type, ts=ts, value=v, unit="kW",
            quality="estimated", source="forecast",
        )  # type: ignore[arg-type]
        for ts, v in zip(ts_list, values)
    ]


def _day_slots(day: date, tz) -> list[datetime]:
    start = datetime.combine(day, time.min, tzinfo=tz)
    return [start + timedelta(minutes=RESOLUTION_MINUTES * i) for i in range(SLOTS)]


def _values_for_day(rows: list[MeasurementRead]) -> list[float]:
    return [row.value if row.value is not None else 0.0 for row in rows]


def _fixed_curve_actions(price_points: list[PricePoint], pcs_kw: float) -> list[tuple[str, float]]:
    """D1 固定计划曲线：谷段满充、尖峰满放，其余不动。"""
    actions: list[tuple[str, float]] = []
    for point in price_points:
        if point.tier == "valley":
            actions.append(("charge", pcs_kw))
        elif point.tier == "sharp":
            actions.append(("discharge", pcs_kw))
        else:
            actions.append(("idle", 0.0))
    return actions


class _ReplayContext:
    """一次回放共享的只读上下文：站点、电价、气象代理、日期清单。"""

    def __init__(self, db: Database, site: SiteRead, weather: list[WeatherPoint] | None):
        self.db = db
        self.site = site
        self.tz = _zone(site.location.timezone)
        tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
        if tariff is None:
            raise errors.validation_error(f"tariff not found: {site.tariff_id}")
        self.price_provider = TouProvider(tariff)
        self.weather = weather or []

    def history(self, data_type: str, before: date, n_days: int) -> list[tuple[datetime, float]]:
        history: list[tuple[datetime, float]] = []
        for offset in range(n_days, 0, -1):
            day = before - timedelta(days=offset)
            rows = self.db.get_measurements_for_date(
                self.site.site_id, data_type, day, self.site.location.timezone
            )
            history.extend(
                (row.ts, row.value) for row in rows
                if row.value is not None and row.quality in ("measured", "estimated")
            )
        return history

    def actuals(self, data_type: str, day: date) -> list[MeasurementRead]:
        rows = self.db.get_measurements_for_date(
            self.site.site_id, data_type, day, self.site.location.timezone
        )
        if len(rows) != SLOTS or any(r.value is None for r in rows):
            raise errors.data_quality_error(f"{day} 的 {data_type} 不足 {SLOTS} 点，无法回放")
        return rows

    def prices(self, day: date) -> list[PricePoint]:
        start = datetime.combine(day, time.min, tzinfo=self.tz)
        return self.price_provider.points(start, periods=SLOTS, resolution_minutes=RESOLUTION_MINUTES)

    def weather_for(self, start: date, days: int) -> list[WeatherPoint]:
        end = start + timedelta(days=days)
        return [p for p in self.weather if start <= p.ts.date() < end]


def _forecast_pv(ctx: _ReplayContext, method: str, day: date, cap_kw: float) -> list[float]:
    """单日 96 点光伏预测（仅使用 day 之前的数据）。"""
    if method == "oracle":
        return _values_for_day(ctx.actuals("pv_power", day))
    history = ctx.history("pv_power", day, N_DAYS + 1)
    if method == "day_persistence":
        rows = ctx.db.get_measurements_for_date(
            ctx.site.site_id, "pv_power", day - timedelta(days=1), ctx.site.location.timezone
        )
        if len(rows) != SLOTS:
            raise ValueError(f"day_persistence 需要昨日完整 96 点: {day}")
        return [min(max(r.value or 0.0, 0.0), cap_kw) for r in rows]
    if method == "physics":
        weather = ctx.weather_for(day, 1)
        if not weather:
            raise ValueError("physics 需要气象代理数据（archive）")
        forecast = predict_pv_physics(
            weather, day, ctx.site.pv, ctx.site.location.lat, ctx.site.location.lon,
            ctx.tz, "open_meteo_archive",
        )
        return [p.value for p in forecast.points]
    if method == "blend":
        weather_past = ctx.weather_for(day - timedelta(days=N_DAYS), N_DAYS)
        weather_future = ctx.weather_for(day, 1)
        if not weather_future:
            raise ValueError("blend 需要气象代理数据（archive）")
        forecast = blend_pv_forecast(
            weather_past, weather_future, history, ctx.site.pv,
            ctx.site.location.lat, ctx.site.location.lon, day, ctx.tz, N_DAYS,
        )
        return [p.value for p in forecast.points]
    days_map = daily_series(history, ctx.tz)
    dates = history_dates(days_map, day, N_DAYS, "pv_power")
    base = slot_means(days_map, dates)
    if method == "slot_mean_bias":
        # 队友超短期公式的日前适配：昨日同刻偏差按 alpha 延续
        alpha = 0.25
        yesterday = day - timedelta(days=1)
        if yesterday in days_map:
            base = [
                base[slot] + alpha * (days_map[yesterday].get(slot * RESOLUTION_MINUTES, base[slot]) - base[slot])
                for slot in range(SLOTS)
            ]
    return [min(max(v, 0.0), cap_kw) for v in base]


def _forecast_load(ctx: _ReplayContext, day: date, oracle: bool) -> list[float]:
    if oracle:
        return _values_for_day(ctx.actuals("load_power", day))
    history = ctx.history("load_power", day, N_DAYS + 1)
    return [p.value for p in predict_load_profile(history, day, ctx.tz, N_DAYS).points]


def _plan_actions(
    ctx: _ReplayContext,
    day: date,
    pv_values: list[float],
    load_values: list[float],
    prices: list[PricePoint],
    initial_soc: float,
) -> list[tuple[str, float]]:
    slots = _day_slots(day, ctx.tz)
    result = RuleDispatchEngine().run(
        DispatchInputs(
            site=ctx.site,
            target_date=day,
            initial_soc=initial_soc,
            pv_points=_as_measurements("pv_power", slots, pv_values),
            load_points=_as_measurements("load_power", slots, load_values),
            price_points=prices,
            resolution_minutes=RESOLUTION_MINUTES,
        )
    )
    return [(p.action, p.power_kw) for p in result.schedule]


def _exec_dict(day: date, ex: ExecutionResult, baseline_cost: float, extra: dict) -> dict:
    usable_self_use = ex.pv_self_use_kwh / ex.pv_total_kwh if ex.pv_total_kwh > 0 else None
    return {
        "date": day.isoformat(),
        "cost_yuan": ex.cost_yuan,
        "saving_vs_d0_yuan": round(baseline_cost - ex.cost_yuan, 4),
        "grid_import_kwh": ex.grid_import_kwh,
        "peak_import_kw": ex.peak_import_kw,
        "pv_self_use_ratio": round(usable_self_use, 4) if usable_self_use is not None else None,
        "curtailment_kwh": ex.curtailment_kwh,
        "discharge_kwh": ex.discharge_kwh,
        "grid_charge_kwh": ex.grid_charge_kwh,
        "end_soc": ex.end_soc,
        "soc_violations": ex.soc_violations,
        **extra,
    }


def run_replay(
    db: Database,
    site_id: str,
    test_days: int = 30,
    pv_methods: list[str] | None = None,
    strategies: list[str] | None = None,
    weather: list[WeatherPoint] | None = None,
    export_dir: Path | None = None,
) -> dict:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    pv_methods = pv_methods or PV_METHODS
    strategies = strategies or STRATEGIES
    unknown_pv = set(pv_methods) - set(PV_METHODS)
    unknown_st = set(strategies) - set(STRATEGIES)
    if unknown_pv or unknown_st:
        raise errors.validation_error(f"未知方法: {sorted(unknown_pv)} {sorted(unknown_st)}")

    last_date = db.get_measurement_last_date(site_id, "pv_power")
    first_date = db.get_measurement_first_date(site_id, "pv_power")
    if last_date is None or first_date is None:
        raise errors.data_quality_error("站点没有 pv_power 测量数据，无法回放")
    test_dates = [last_date - timedelta(days=i) for i in range(test_days - 1, -1, -1)]
    if (test_dates[0] - first_date).days < MIN_HISTORY:
        raise errors.data_quality_error("测试窗口之前的历史不足，无法回放")

    need_weather = any(m in ("physics", "blend") for m in pv_methods)
    if need_weather and weather is None:
        from app.infrastructure.weather.open_meteo_archive import OpenMeteoArchiveProvider

        weather = OpenMeteoArchiveProvider().irradiance_range(
            lat=site.location.lat, lon=site.location.lon,
            timezone_name=site.location.timezone,
            start=first_date, end=last_date + timedelta(days=1),
        )
    ctx = _ReplayContext(db, site, weather)

    run_id = f"RP{uuid4().hex[:8].upper()}"
    config = {
        "site_id": site_id, "test_days": test_days,
        "test_window": [test_dates[0].isoformat(), test_dates[-1].isoformat()],
        "pv_methods": pv_methods, "strategies": strategies,
        "weather_proxy": "reanalysis" if need_weather else None,
        "n_days": N_DAYS,
    }
    db.insert_replay_run(run_id, site_id, config)

    cap_kw = site.pv.capacity_kwp
    all_rows: list[dict] = []
    forecast_rows: list[dict] = []

    # 预测精度：每个光伏方法 × 每天（与策略无关，只评一次）
    for method in pv_methods:
        for day in test_dates:
            predicted = _forecast_pv(ctx, method, day, cap_kw)
            actual = _values_for_day(ctx.actuals("pv_power", day))
            metrics = forecast_metrics(predicted, actual, cap_kw)
            row = {"run_id": run_id, "date": day.isoformat(), "pv_method": method,
                   "strategy": "__forecast__", "metrics": metrics}
            forecast_rows.append(row)
            db.insert_replay_day_result(run_id, day, method, "__forecast__", metrics)

    # 运行与经济：逐策略 × 逐预测方法，SOC 跨日连续
    combos: list[tuple[str, str]] = []
    for strategy in strategies:
        if strategy in ("d0_no_storage", "d1_fixed_curve", "d5_rule_oracle"):
            combos.append(("__actual__", strategy))
        else:
            combos.extend((m, strategy) for m in pv_methods)

    for pv_method, strategy in combos:
        soc = 0.5
        for day in test_dates:
            prices = ctx.prices(day)
            actual_pv = _values_for_day(ctx.actuals("pv_power", day))
            actual_load = _values_for_day(ctx.actuals("load_power", day))
            if strategy == "d0_no_storage":
                ex = no_storage_baseline(actual_pv, actual_load, prices, RESOLUTION_MINUTES)
            elif strategy == "d1_fixed_curve":
                actions = _fixed_curve_actions(prices, site.storage.pcs_kw)
                ex = execute_plan(actions, actual_pv, actual_load, prices, site.storage, soc, RESOLUTION_MINUTES)
            elif strategy == "d5_rule_oracle":
                actions = _plan_actions(ctx, day, actual_pv, actual_load, prices, soc)
                ex = execute_plan(actions, actual_pv, actual_load, prices, site.storage, soc, RESOLUTION_MINUTES)
            else:  # d4_rule_predictive
                pv_fc = _forecast_pv(ctx, pv_method, day, cap_kw)
                load_fc = _forecast_load(ctx, day, oracle=False)
                actions = _plan_actions(ctx, day, pv_fc, load_fc, prices, soc)
                ex = execute_plan(actions, actual_pv, actual_load, prices, site.storage, soc, RESOLUTION_MINUTES)
            soc = ex.end_soc
            baseline = no_storage_baseline(actual_pv, actual_load, prices, RESOLUTION_MINUTES)
            metrics = _exec_dict(day, ex, baseline.cost_yuan, {})
            db.insert_replay_day_result(run_id, day, pv_method, strategy, metrics)
            all_rows.append({"run_id": run_id, "pv_method": pv_method, "strategy": strategy, **metrics})

    summary = _summarize(all_rows, forecast_rows)
    db.finish_replay_run(run_id, summary)
    if export_dir is not None:
        _export(export_dir, run_id, config, all_rows, forecast_rows, summary)
    return {"run_id": run_id, "config": config, "summary": summary}


def _summarize(rows: list[dict], forecast_rows: list[dict]) -> dict:
    def agg(key_rows: list[dict]) -> dict:
        total_cost = sum(r["cost_yuan"] for r in key_rows)
        total_saving = sum(r["saving_vs_d0_yuan"] for r in key_rows)
        return {
            "days": len(key_rows),
            "total_cost_yuan": round(total_cost, 2),
            "total_saving_yuan": round(total_saving, 2),
            "avg_daily_saving_yuan": round(total_saving / len(key_rows), 2) if key_rows else 0,
            "max_peak_import_kw": max((r["peak_import_kw"] for r in key_rows), default=0),
            "total_discharge_kwh": round(sum(r["discharge_kwh"] for r in key_rows), 2),
            "pv_self_use_ratio": round(
                sum(r["pv_self_use_ratio"] or 0 for r in key_rows) / len(key_rows), 4
            ) if key_rows else None,
            "soc_violations": sum(r["soc_violations"] for r in key_rows),
        }

    strategies: dict[str, dict] = {}
    for combo in {(r["pv_method"], r["strategy"]) for r in rows}:
        key = f"{combo[0]}+{combo[1]}"
        strategies[key] = agg([r for r in rows if (r["pv_method"], r["strategy"]) == combo])

    forecasts: dict[str, dict] = {}
    for method in {r["pv_method"] for r in forecast_rows}:
        frs = [r["metrics"] for r in forecast_rows if r["pv_method"] == method]
        forecasts[method] = {
            "days": len(frs),
            "skill_all_day": round(sum(f["skill_all_day"] for f in frs) / len(frs), 4),
            "skill_daylight": round(
                sum(f["skill_daylight"] or 0 for f in frs) / len(frs), 4
            ),
            "rmse": round(sum(f["rmse"] for f in frs) / len(frs), 2),
            "bias": round(sum(f["bias"] for f in frs) / len(frs), 2),
        }
    return {"strategies": strategies, "forecasts": forecasts}


def _export(
    export_dir: Path,
    run_id: str,
    config: dict,
    rows: list[dict],
    forecast_rows: list[dict],
    summary: dict,
) -> None:
    out = export_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "day_results.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    flat_forecast = [
        {"run_id": r["run_id"], "date": r["date"], "pv_method": r["pv_method"], **r["metrics"]}
        for r in forecast_rows
    ]
    with (out / "forecast_metrics.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(flat_forecast[0].keys()))
        writer.writeheader()
        writer.writerows(flat_forecast)
