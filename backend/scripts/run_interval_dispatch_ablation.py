"""区间调度消融：P50 决策 vs P10/P90 决策，同一份预测、同一套物理约束。

实验设计（要点：只有一个变量在变）
- 预测：取 `run_price_benchmark.py` 产出的**真实样本外** P10/P50/P90；
- 结算：用当天**实际电价**结算，预测错多少就承担多少，不做任何"上帝视角"修正；
- 物理：同一套 PV/负荷曲线、同一块电池、同一组 SOC/功率/循环约束；
- 变量：只有决策规则不同。

对比的三类指标，对应组长关心的三件事：
1. **成本**：实际结算电费，以及相对"不装储能"的节省；
2. **缺电风险**：保底 SOC 之下的电量（`reserve_shortfall_kwh`）+ 想放但没电的时段数；
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
import json
import zlib
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
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

TZ = ZoneInfo("Asia/Shanghai")
SLOTS_PER_HOUR = 4
PV_CAPACITY_KW = 1000.0
LOAD_PEAK_KW = 1500.0


def _site() -> SiteRead:
    payload = SiteCreate(
        site_id="S-ABLATION",
        name="区间调度消融站点",
        location={"lat": 22.94, "lon": 113.75, "timezone": "Asia/Shanghai"},
        pv=PVConfig(capacity_kwp=PV_CAPACITY_KW),
        load=LoadConfig(peak_kw=LOAD_PEAK_KW),
        storage=StorageConfig(
            pcs_kw=500, capacity_kwh=1000, round_trip_eff=0.88,
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


def _series_for_day(day: date) -> tuple[list[MeasurementRead], list[MeasurementRead]]:
    pv: list[MeasurementRead] = []
    load: list[MeasurementRead] = []
    for i in range(24 * SLOTS_PER_HOUR):
        hour = i / SLOTS_PER_HOUR
        ts = datetime(day.year, day.month, day.day, tzinfo=TZ) + timedelta(minutes=15 * i)
        pv.append(MeasurementRead(data_type="pv_power", ts=ts, value=_pv_kw(hour), unit="kW", source="sim"))
        load.append(
            MeasurementRead(data_type="load_power", ts=ts, value=_load_kw(hour), unit="kW", source="sim")
        )
    return pv, load


def _expand(hourly: pd.Series, day: date) -> list[float]:
    """小时值按 15 分钟阶跃展开。"""
    values = hourly.to_numpy(dtype="float64")
    return np.repeat(values, SLOTS_PER_HOUR).tolist()


def _settle(
    response, actual_96: np.ndarray, pv_96: np.ndarray, net_96: np.ndarray, site: SiteRead
) -> dict:
    """用**实际电价**结算，并计算风险/弃光/决策质量指标。"""
    dt = 0.25
    grid_import = np.asarray([p.grid_import_kw for p in response.schedule])
    grid_charge = np.asarray([p.charge_from_grid_kw for p in response.schedule])
    pv_charge = np.asarray([p.charge_from_pv_kw for p in response.schedule])
    discharge = np.asarray([p.power_kw if p.action == "discharge" else 0.0 for p in response.schedule])
    curtail = np.asarray([p.pv_curtailment_kw for p in response.schedule])
    soc = np.asarray([p.soc_end for p in response.schedule])

    cost = float(np.sum(grid_import * dt * actual_96))
    baseline = float(np.sum(np.maximum(net_96, 0.0) * dt * actual_96))
    median_price = float(np.median(actual_96))

    charge_kwh = grid_charge * dt
    discharge_kwh = discharge * dt
    # 只对"花钱买的电"评估决策质量：光伏余电是免费电，充在什么价位都不算失误
    mis_charge = float(np.sum(charge_kwh[actual_96 > median_price]) / max(charge_kwh.sum(), 1e-9))
    mis_discharge = float(np.sum(discharge_kwh[actual_96 < median_price]) / max(discharge_kwh.sum(), 1e-9))

    min_energy = site.storage.soc_min * site.storage.capacity_kwh
    reserve = min_energy + 0.10 * site.storage.capacity_kwh
    soc_energy = soc * site.storage.capacity_kwh
    high_price = actual_96 >= np.quantile(actual_96, 0.85)
    shortfall = float(np.sum(np.maximum(reserve - soc_energy, 0.0) * dt * (high_price & (net_96 > 0))))

    pv_energy = float(np.sum(pv_96) * dt)
    top = actual_96 >= np.quantile(actual_96, 0.90)
    peak_import = float(np.sum(grid_import * dt * top))
    avg_charge = float(np.sum(charge_kwh * actual_96) / max(charge_kwh.sum(), 1e-9))
    avg_discharge = float(np.sum(discharge_kwh * actual_96) / max(discharge_kwh.sum(), 1e-9))
    return {
        "cost_yuan": round(cost, 2),
        "saving_vs_no_storage_yuan": round(baseline - cost, 2),
        "curtailment_rate": round(float(np.sum(curtail) * dt / max(pv_energy, 1e-9)), 4),
        "reserve_shortfall_kwh": round(shortfall, 3),
        "mis_charge_ratio": round(mis_charge, 4),
        "mis_discharge_ratio": round(mis_discharge, 4),
        "cycles": round(response.equivalent_full_cycles, 3),
        "charge_kwh": round(float(charge_kwh.sum()), 2),
        "pv_charge_kwh": round(float(np.sum(pv_charge) * dt), 2),
        "discharge_kwh": round(float(discharge_kwh.sum()), 2),
        "avg_charge_price": round(avg_charge, 4),
        "avg_discharge_price": round(avg_discharge, 4),
        "peak_import_kwh": round(peak_import, 2),
    }


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
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    pred = pred[pred["model"] == args.model]
    if pred.empty:
        raise SystemExit(f"no rows for model={args.model} in {args.predictions}")

    site = _site()
    engine = IntervalDispatchEngine()
    margins = [float(m) for m in args.margins.split(",") if m.strip()]
    stress_levels = [float(s) for s in args.stress.split(",") if s.strip()]
    rows: list[dict] = []
    # 偏差种子只由 (日期, 档位) 决定：换 --stress 列表或换策略都不会改变同一天的偏差，
    # 这样不同批次的结果才可横向比较
    def _bias_for(day_text: str, stress: float) -> np.ndarray:
        seed = int(zlib.crc32(f"{day_text}|{stress}".encode("utf-8")))
        return _ar1_bias(24 * SLOTS_PER_HOUR, stress, np.random.default_rng(seed))

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

        pv, load = _series_for_day(day)
        pv_96 = np.asarray([p.value for p in pv])
        load_96 = np.asarray([p.value for p in load])
        net_96 = load_96 - pv_96
        actual_96 = np.repeat(actual, SLOTS_PER_HOUR)
        ts_96 = [p.ts for p in pv]

        for stress in stress_levels:
            bias = day_bias[(str(day_text), stress)]
            p10_96 = np.clip(np.repeat(p10_raw, SLOTS_PER_HOUR) * bias, 0.0, None)
            p50_96 = np.clip(np.repeat(p50_raw, SLOTS_PER_HOUR) * bias, 0.0, None)
            p90_96 = np.clip(np.repeat(p90_raw, SLOTS_PER_HOUR) * bias, 0.0, None)

            # 1) 基线：把 P50 当成真实价格（等价于"模型很准"的假设）
            base_inputs = IntervalInputs(
                site=site, target_date=day, initial_soc=0.5,
                pv_points=pv, load_points=load,
                bands=_p50_policy_bands(p50_96, ts_96),
                margin_ratio=-1e9,
                reserve_ratio=0.10,
            )
            base_resp = engine.run(base_inputs)
            rows.append({"day": day_text, "policy": "p50", "stress": stress, "margin": 0.0,
                         **_settle(base_resp, actual_96, pv_96, net_96, site)})

            # 2) 区间策略：P90 充电 / P10 放电 + 鲁棒门槛
            for margin in margins:
                bands = [
                    PriceBand(ts=t, p10=a, p50=b, p90=c)
                    for t, a, b, c in zip(ts_96, p10_96, p50_96, p90_96)
                ]
                resp = engine.run(replace(base_inputs, bands=bands, margin_ratio=margin))
                rows.append({"day": day_text, "policy": "interval", "stress": stress, "margin": margin,
                             **_settle(resp, actual_96, pv_96, net_96, site)})

    frame = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "by_day.csv", index=False, float_format="%.4f")

    agg = (
        frame.groupby(["stress", "policy", "margin"])
        .agg(
            days=("day", "count"),
            cost_yuan=("cost_yuan", "sum"),
            saving_yuan=("saving_vs_no_storage_yuan", "sum"),
            curtailment_rate=("curtailment_rate", "mean"),
            reserve_shortfall_kwh=("reserve_shortfall_kwh", "sum"),
            mis_charge_ratio=("mis_charge_ratio", "mean"),
            mis_discharge_ratio=("mis_discharge_ratio", "mean"),
            cycles=("cycles", "mean"),
            peak_import_kwh=("peak_import_kwh", "sum"),
            avg_charge_price=("avg_charge_price", "mean"),
            avg_discharge_price=("avg_discharge_price", "mean"),
        )
        .reset_index()
    )
    # 每个压力档以该档的 p50 策略为基准算成本变化
    base = agg[agg.policy == "p50"].set_index("stress")["cost_yuan"]
    agg["cost_delta_pct"] = (
        (agg["cost_yuan"] - agg["stress"].map(base)) / agg["stress"].map(base) * 100
    ).round(2)
    agg.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    (out / "summary.json").write_text(
        json.dumps(
            {
                "predictions": args.predictions,
                "model": args.model,
                "days": int(frame["day"].nunique()),
                "price_scale": args.price_scale,
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
