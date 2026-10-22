"""区间调度消融：P50 决策 vs P10/P90 决策，同一份预测、同一套物理约束。

实验设计（要点：只有一个变量在变）
- 预测：取 `run_price_benchmark.py` 产出的**真实样本外** P10/P50/P90；
- 结算：用当天**实际电价**结算，预测错多少就承担多少，不做任何"上帝视角"修正；
- 物理：同一套 PV/负荷曲线、同一块电池、同一组 SOC/功率/循环约束；
- 变量：只有决策规则不同。

对比的三类指标，对应组长关心的三件事：
1. **成本**：实际结算电费，以及相对"不装储能"的节省；
2. **风险代理**：实际高价时段电池目标供能与实际放电的差额（不是实际停电量）；
3. **弃光率**：无法消纳被弃掉的光伏占比。

另外引入两个**决策质量**指标，直接刻画"是否按错价格做决策"：
- `mis_charge_ratio`：充电电量中，实际价格高于当日中位数的占比（越高越糟）；
- `mis_discharge_ratio`：放电电量中，实际价格低于当日中位数的占比（越高越糟）。

用法::

    python scripts/run_interval_dispatch_ablation.py \
        --predictions data/forecast_experiments/price_calibration/predictions.csv \
        --model lgbm_calibrated \
        --output data/forecast_experiments/interval_dispatch
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
import json
import zlib
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.dispatch_interval import IntervalDispatchEngine, IntervalInputs, PriceBand  # noqa: E402
from app.domain.models import (  # noqa: E402
    LoadConfig,
    MeasurementRead,
    PVConfig,
    SiteCreate,
    SiteRead,
    StorageConfig,
)
from scripts.run_price_benchmark import _peak_rss_mb  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
SLOTS_PER_HOUR = 4
PV_CAPACITY_KW = 1000.0
LOAD_PEAK_KW = 1500.0


def _site(*, pcs_kw: float = 250.0, capacity_kwh: float = 500.0) -> SiteRead:
    payload = SiteCreate(
        site_id="S-ABLATION",
        name="区间调度消融站点",
        location={"lat": 22.94, "lon": 113.75, "timezone": "Asia/Shanghai"},
        pv=PVConfig(capacity_kwp=PV_CAPACITY_KW),
        load=LoadConfig(peak_kw=LOAD_PEAK_KW),
        storage=StorageConfig(
            pcs_kw=pcs_kw, capacity_kwh=capacity_kwh, round_trip_eff=0.88,
            soc_min=0.10, soc_max=0.95, cycles_per_day_max=2,
        ),
        tariff_id="T-GUANGDONG-DEMO",
    )
    now = datetime(2026, 1, 1, tzinfo=TZ)
    return SiteRead(**payload.model_dump(), created_at=now, updated_at=now)


def _pv_kw(hour: float) -> float:
    """单峰光伏曲线，6:00 出功、12:00 峰值、18:00 归零。"""
    if hour <= 6 or hour >= 18:
        return 0.0
    shape = np.sin(np.pi * (hour - 6) / 12) ** 1.4
    return PV_CAPACITY_KW * float(shape)


def _load_kw(hour: float) -> float:
    """两班制工业负荷：夜间维持、白天高位、晚班次高峰。"""
    if 8 <= hour < 12 or 13 <= hour < 17:
        factor = 0.88
    elif 17 <= hour < 21:
        factor = 0.78
    elif 12 <= hour < 13:
        factor = 0.55
    else:
        factor = 0.42
    return LOAD_PEAK_KW * factor


def _series_for_day(day: date, start: datetime | None = None) -> tuple[list[MeasurementRead], list[MeasurementRead]]:
    start = start or datetime(day.year, day.month, day.day, tzinfo=TZ)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("series start must carry a timezone")
    pv: list[MeasurementRead] = []
    load: list[MeasurementRead] = []
    for i in range(24 * SLOTS_PER_HOUR):
        hour = i / SLOTS_PER_HOUR
        ts = start + timedelta(minutes=15 * i)
        pv.append(MeasurementRead(data_type="pv_power", ts=ts, value=_pv_kw(hour), unit="kW", source="sim"))
        load.append(
            MeasurementRead(data_type="load_power", ts=ts, value=_load_kw(hour), unit="kW", source="sim")
        )
    return pv, load


def _validate_predictions(frame: pd.DataFrame, model: str, min_days: int = 1) -> pd.DataFrame:
    required = {"day", "ts", "model", "actual", "p10", "p50", "p90"}
    if missing := required - set(frame.columns):
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    selected = frame.loc[frame["model"] == model].copy()
    if selected.empty:
        raise ValueError(f"no predictions for model={model}")
    values = selected[["actual", "p10", "p50", "p90"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prices and quantiles must be finite")
    if ((values[:, 1] > values[:, 2]) | (values[:, 2] > values[:, 3])).any():
        raise ValueError("quantiles must be ordered p10 <= p50 <= p90")
    for day, block in selected.groupby("day"):
        stamps = [pd.Timestamp(value) for value in block["ts"]]
        if any(t.tzinfo is None or t.utcoffset() is None for t in stamps):
            raise ValueError(f"timestamps must carry a timezone: {day}")
        stamps = sorted(stamps)
        if (len(stamps) != 24 or len(set(stamps)) != 24
                or [t.hour for t in stamps] != list(range(24))
                or any(t.minute or t.second or t.microsecond or t.nanosecond
                       or t.date().isoformat() != day for t in stamps)
                or any(b - a != pd.Timedelta(hours=1) for a, b in zip(stamps, stamps[1:]))):
            raise ValueError(f"expected 24 unique aligned hours for {day}")
    days = sorted(date.fromisoformat(day) for day in selected["day"].unique())
    if len(days) < min_days:
        raise ValueError(f"need at least {min_days} evaluation days, got {len(days)}")
    if any(b - a != timedelta(days=1) for a, b in zip(days, days[1:])):
        raise ValueError("evaluation days must be consecutive")
    return selected.sort_values(["day", "ts"])


def _settle(response, actual_96: np.ndarray, pv_96: np.ndarray, net_96: np.ndarray,
            site: SiteRead, penalty_coefficient: float) -> dict:
    """实际结算与风险代理分开；实际价格仅在事后评分时使用。"""
    dt = 0.25
    grid_import = np.asarray([p.grid_import_kw for p in response.schedule])
    grid_charge = np.asarray([p.charge_from_grid_kw for p in response.schedule])
    pv_charge = np.asarray([p.charge_from_pv_kw for p in response.schedule])
    discharge = np.asarray([p.power_kw if p.action == "discharge" else 0.0 for p in response.schedule])
    curtail = np.asarray([p.pv_curtailment_kw for p in response.schedule])
    soc = np.asarray([p.soc_end for p in response.schedule])
    if not (len(grid_import) == len(actual_96) == len(pv_96) == len(net_96)):
        raise ValueError("settlement arrays must align")
    cost = float(np.sum(grid_import * dt * actual_96))
    baseline = float(np.sum(np.maximum(net_96, 0.0) * dt * actual_96))
    median_price = float(np.median(actual_96))
    charge_kwh = grid_charge * dt
    discharge_kwh = discharge * dt
    mis_charge_kwh = float(np.sum(charge_kwh[actual_96 > median_price]))
    mis_discharge_kwh = float(np.sum(discharge_kwh[actual_96 < median_price]))
    high_price = actual_96 >= np.quantile(actual_96, 0.85)
    # 尖峰目标放电受 PCS 和正净负荷限制。电网仍可供电，差额仅是电池供能风险代理。
    target_kw = np.minimum(np.maximum(net_96, 0.0), site.storage.pcs_kw)
    shortfall = float(np.sum(np.maximum(target_kw - discharge, 0.0) * high_price) * dt)
    reserve = (site.storage.soc_min + 0.10) * site.storage.capacity_kwh
    deficit = np.maximum(reserve - soc * site.storage.capacity_kwh, 0.0)
    reserve_deficit_max = float(np.max(deficit[high_price]) if high_price.any() else 0.0)
    peak_price = max(0.0, float(np.mean(actual_96[high_price])))
    penalty = shortfall * peak_price * penalty_coefficient
    pv_energy = float(np.sum(pv_96) * dt)
    curtail_kwh = float(np.sum(curtail) * dt)
    return {
        "cost_yuan": cost,
        "shortfall_penalty_yuan": penalty,
        "total_cost_yuan": cost + penalty,
        "saving_vs_no_storage_yuan": baseline - cost,
        "supply_shortfall_kwh": shortfall,
        "reserve_deficit_max_kwh": reserve_deficit_max,
        "peak_price_yuan_per_kwh": peak_price,
        "curtailment_rate": curtail_kwh / max(pv_energy, 1e-9),
        "curtailment_kwh": curtail_kwh,
        "pv_kwh": pv_energy,
        "mis_charge_kwh": mis_charge_kwh,
        "mis_discharge_kwh": mis_discharge_kwh,
        "mis_charge_ratio": mis_charge_kwh / max(charge_kwh.sum(), 1e-9),
        "mis_discharge_ratio": mis_discharge_kwh / max(discharge_kwh.sum(), 1e-9),
        "cycles": response.equivalent_full_cycles,
        "charge_kwh": float(charge_kwh.sum()),
        "pv_charge_kwh": float(np.sum(pv_charge) * dt),
        "discharge_kwh": float(discharge_kwh.sum()),
        "grid_charge_cost_yuan": float(np.sum(charge_kwh * actual_96)),
        "discharge_value_yuan": float(np.sum(discharge_kwh * actual_96)),
        "peak_import_kwh": float(np.sum(grid_import * dt * high_price)),
    }


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    sums = ("cost_yuan", "shortfall_penalty_yuan", "total_cost_yuan", "supply_shortfall_kwh",
            "curtailment_kwh", "pv_kwh", "mis_charge_kwh", "mis_discharge_kwh", "charge_kwh",
            "discharge_kwh", "grid_charge_cost_yuan", "discharge_value_yuan", "peak_import_kwh")
    grouped = frame.groupby(["stress", "policy", "margin"])
    result = grouped.agg(**{name: (name, "sum") for name in sums},
                         days=("day", "count"), cycles=("cycles", "mean"),
                         reserve_deficit_max_kwh=("reserve_deficit_max_kwh", "max")).reset_index()
    for numerator, denominator, output in (
        ("mis_charge_kwh", "charge_kwh", "mis_charge_ratio"),
        ("mis_discharge_kwh", "discharge_kwh", "mis_discharge_ratio"),
        ("curtailment_kwh", "pv_kwh", "curtailment_rate"),
        ("grid_charge_cost_yuan", "charge_kwh", "avg_charge_price"),
        ("discharge_value_yuan", "discharge_kwh", "avg_discharge_price"),
    ):
        result[output] = result[numerator] / result[denominator].clip(lower=1e-9)
    for field, output in (("cost_yuan", "cost_delta_pct"),
                          ("total_cost_yuan", "total_cost_delta_pct")):
        base = result[result.policy == "p50"].set_index("stress")[field]
        denominator = result["stress"].map(base)
        result[output] = ((result[field] - denominator) / denominator.abs().replace(0, np.nan) * 100)
    return result


def _ar1_bias(n: int, stress: float, rng: np.random.Generator, rho: float = 0.85) -> np.ndarray:
    """生成逐时段预测偏差（一阶自回归，日内持续）。

    全局同乘一个系数不会改变时段排序，量化规则对它是**尺度不变**的，
    测不出任何差异。真实的预测误差是逐时段、且有持续性的：
    比如"低估晚高峰、高估午间"。这里用 AR(1) 模拟这种结构。
    """
    if stress <= 0:
        return np.ones(n)
    noise = np.zeros(n)
    prev = rng.normal(0, 1)
    for i in range(n):
        prev = rho * prev + np.sqrt(1 - rho**2) * rng.normal(0, 1)
        noise[i] = prev
    return np.clip(1.0 + stress * noise, 0.2, None)


def _p50_policy_bands(p50: np.ndarray, ts_list: list[datetime]) -> list[PriceBand]:
    """基线策略：假装预测就是真实的，区间退化成一个点。"""
    return [PriceBand(ts=t, p10=float(v), p50=float(v), p90=float(v)) for t, v in zip(ts_list, p50)]


def _bias_for(day_text: str, stress: float) -> np.ndarray:
    seed = int(zlib.crc32(f"{day_text}|{stress}".encode("utf-8")))
    return _ar1_bias(24 * SLOTS_PER_HOUR, stress, np.random.default_rng(seed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default="lgbm_calibrated")
    parser.add_argument("--output", default="data/forecast_experiments/interval_dispatch")
    parser.add_argument("--margins", default="0.02", help="鲁棒价差门槛，按当日均价的比例")
    parser.add_argument("--price-scale", type=float, default=0.008,
                        help="原始价格→元/kWh 的换算（GEFCom 是 $/MWh，取 0.008）")
    parser.add_argument("--stress", default="0.0,0.1,0.2,0.3",
                        help="预测系统性偏差压力档（相对电平的比例）；0 表示用原始预测")
    parser.add_argument("--pcs-kw", type=float, default=250.0,
                        help="压力实验电池 PCS 功率（默认 250 kW）")
    parser.add_argument("--capacity-kwh", type=float, default=500.0,
                        help="压力实验电池容量（默认 500 kWh）")
    parser.add_argument("--penalty-coefficient", type=float, default=2.0,
                        help="缺电风险惩罚系数：风险代理 kWh × 尖峰电价 × 系数")
    parser.add_argument("--min-days", type=int, default=1, help="正式 T3 至少 60 天；不足时拒绝运行")
    args = parser.parse_args()
    started = time.perf_counter()

    for value, name in ((args.pcs_kw, "pcs-kw"), (args.capacity_kwh, "capacity-kwh"),
                        (args.price_scale, "price-scale")):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be positive and finite")
    if not np.isfinite(args.penalty_coefficient) or args.penalty_coefficient < 0 or args.min_days < 1:
        parser.error("penalty coefficient must be finite and non-negative; min-days must be positive")
    margins = list(dict.fromkeys(float(m) for m in args.margins.split(",") if m.strip()))
    stress_levels = list(dict.fromkeys(float(s) for s in args.stress.split(",") if s.strip()))
    if not margins or not stress_levels or any(not np.isfinite(v) or v < 0 for v in margins + stress_levels):
        parser.error("margins and stress levels must be finite, non-negative and nonempty")
    pred = _validate_predictions(pd.read_csv(args.predictions), args.model, args.min_days)
    site = _site(pcs_kw=args.pcs_kw, capacity_kwh=args.capacity_kwh)
    engine = IntervalDispatchEngine()
    rows: list[dict] = []
    # 偏差种子只由 (日期, 档位) 决定：换 --stress 列表或换策略都不会改变同一天的偏差，
    # 这样不同批次的结果才可横向比较
    day_bias = {
        (str(day), stress): _bias_for(str(day), stress)
        for day in pred["day"].unique()
        for stress in stress_levels
    }

    for day_text, block in pred.groupby("day"):
        block = block.sort_values("ts")
        day = date.fromisoformat(day_text)
        p10_raw = block["p10"].to_numpy(float) * args.price_scale
        p50_raw = block["p50"].to_numpy(float) * args.price_scale
        p90_raw = block["p90"].to_numpy(float) * args.price_scale
        actual = block["actual"].to_numpy(float) * args.price_scale

        start = pd.Timestamp(block["ts"].iloc[0]).to_pydatetime()
        pv, load = _series_for_day(day, start)
        pv_96 = np.asarray([p.value for p in pv])
        load_96 = np.asarray([p.value for p in load])
        net_96 = load_96 - pv_96
        actual_96 = np.repeat(actual, SLOTS_PER_HOUR)
        ts_96 = [p.ts for p in pv]

        for stress in stress_levels:
            bias = day_bias[(str(day_text), stress)]
            p10_96 = np.repeat(p10_raw, SLOTS_PER_HOUR) * bias
            p50_96 = np.repeat(p50_raw, SLOTS_PER_HOUR) * bias
            p90_96 = np.repeat(p90_raw, SLOTS_PER_HOUR) * bias

            # 1) 基线：把 P50 当成真实价格（等价于"模型很准"的假设）
            base_inputs = IntervalInputs(
                site=site, target_date=day, initial_soc=0.5,
                pv_points=pv, load_points=load,
                bands=_p50_policy_bands(p50_96, ts_96),
                margin_ratio=0.0,
                reserve_ratio=0.10,
            )
            base_resp = engine.run(base_inputs)
            rows.append({"day": day_text, "policy": "p50", "stress": stress, "margin": 0.0,
                         **_settle(base_resp, actual_96, pv_96, net_96, site, args.penalty_coefficient)})

            # 2) 区间策略：P90 充电 / P10 放电 + 鲁棒门槛
            for margin in margins:
                bands = [
                    PriceBand(ts=t, p10=a, p50=b, p90=c)
                    for t, a, b, c in zip(ts_96, p10_96, p50_96, p90_96)
                ]
                resp = engine.run(replace(base_inputs, bands=bands, margin_ratio=margin))
                rows.append({"day": day_text, "policy": "interval", "stress": stress, "margin": margin,
                             **_settle(resp, actual_96, pv_96, net_96, site, args.penalty_coefficient)})

    frame = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "by_day.csv", index=False, float_format="%.8f")

    agg = _summarize(frame)
    agg.to_csv(out / "summary.csv", index=False, float_format="%.8f")

    (out / "summary.json").write_text(
        json.dumps(
            {
                "predictions": args.predictions,
                "model": args.model,
                "days": int(frame["day"].nunique()),
                "price_scale": args.price_scale,
                "price_unit": "CNY/kWh",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "date_start": str(pred["day"].min()),
                "date_end": str(pred["day"].max()),
                "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
                "stress": stress_levels,
                "margins": margins,
                "ar1_rho": 0.85,
                "seed_rule": "crc32(day|stress)",
                "initial_soc": 0.5,
                "daily_soc_reset": True,
                "risk_definition": "sum(max(min(positive_net_kw, pcs_kw)-discharge_kw,0)*0.25) in actual-price top15% slots; proxy, not outage",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_rss_mb": _peak_rss_mb(),
                "python_version": sys.version,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timestamp_note": "source local clock with explicit offset; synthetic PV/load use the same local hours",
                "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "pydantic")},
                "storage": {"pcs_kw": args.pcs_kw, "capacity_kwh": args.capacity_kwh},
                "penalty_coefficient": args.penalty_coefficient,
                "rows": agg.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
