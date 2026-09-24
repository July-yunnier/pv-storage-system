#!/usr/bin/env python3
"""端到端演示：从配置文件跑到一天的调度方案。

这是「最小可用链路」：
    读配置 → 建电价 Provider → 生成日内曲线 → 规则式调度 → 输出方案

用合成数据跑，不需要任何外部数据源。正式流程里把 synthetic_day 换成
真实的预测结果即可。

用法
----
    python scripts/demo_dispatch.py
    python scripts/demo_dispatch.py --site config/site.yaml --date 2026-09-22
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models import site_from_config            # noqa: E402
from src.optimizer.dispatch import rule_based_dispatch, synthetic_day  # noqa: E402
from src.tariff import get_provider                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="config/site.yaml")
    ap.add_argument("--date", default="2026-09-22")
    ap.add_argument("--resolution", type=int, default=15)
    args = ap.parse_args()

    site_path = args.site if os.path.isabs(args.site) else os.path.join(ROOT, args.site)
    if not os.path.exists(site_path):
        print(f"找不到配置 {site_path}")
        print(f"先复制模板： cp config/site.example.yaml config/site.yaml")
        return 1

    with open(site_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    site = site_from_config(cfg["site"])
    if site.storage is None:
        print("配置里没有 site.storage，无法做调度计算")
        return 1

    tariff_cfg = cfg["tariff"]
    provider = get_provider(
        tariff_cfg.get("mode", "tou"),
        template_path=os.path.join(ROOT, "config", tariff_cfg["template"]),
    )

    # ---- 时间轴与电价 ----
    start = datetime.fromisoformat(f"{args.date}T00:00:00")
    prices = provider.price_series(start, start + timedelta(days=1), args.resolution)

    print("=" * 74)
    print("光储系统 · 日调度演示")
    print("=" * 74)
    print(site.describe())
    print()
    print(f"电价模式：{provider.describe()['note']}")
    print(f"时段数：{len(prices)}  @ {args.resolution} min")
    print()

    # ---- 日内曲线（演示用合成数据）----
    load, pv = synthetic_day(site, prices)

    print(f"日内曲线（合成数据，仅供演示）：")
    print(f"  负荷  峰值 {load.max():>8.1f} kW   电量 {load.sum() * args.resolution / 60:>10,.0f} kWh")
    print(f"  光伏  峰值 {pv.max():>8.1f} kW   电量 {pv.sum() * args.resolution / 60:>10,.0f} kWh")
    print()

    # ---- 调度：跑两版做对比 ----
    common = dict(
        resolution_minutes=args.resolution,
        allow_grid_charge=cfg.get("optimizer", {}).get("allow_grid_charge", True),
        allow_grid_export=cfg.get("optimizer", {}).get("allow_grid_export", False),
    )
    res_naive = rule_based_dispatch(load, pv, prices, site.storage,
                                    demand_aware=False, **common)
    res = rule_based_dispatch(load, pv, prices, site.storage,
                              demand_aware=True, **common)

    # 主动削峰：以「基线峰值的 80%」为目标
    # 削峰目标本身是 sizing（规划）的决策变量，这里固定 80% 只为演示效果
    net = load - pv
    base_peak = float(np.max(np.clip(net, 0.0, None)))
    shave_target = base_peak * 0.8
    res_shave = rule_based_dispatch(load, pv, prices, site.storage,
                                    demand_aware=True,
                                    peak_target_kw=shave_target, **common)

    dc = cfg["tariff"].get("demand_charge", {})
    dc_price = dc.get("price_per_kw_month", 0) if dc.get("enabled") else 0

    print("=" * 74)
    print("调度结果")
    print("=" * 74)
    print(res.summary())
    print(f"主动削峰  目标 {shave_target:,.1f} kW（基线峰值的 80%）"
          f" → 实际峰值 {res_shave.peak_grid_kw:,.1f} kW")
    print()

    # ---- 需量感知的价值（这组对比最能说明问题）----
    if dc_price > 0:
        print("=" * 74)
        print("三种策略的对比")
        print("=" * 74)
        print(f"{'方案':<24}{'并网峰值':>11}{'基本电费/月':>14}{'电度电费/日':>14}")
        print("-" * 74)

        rows = [
            ("① 不配储能（基线）", res.baseline_peak_grid_kw, res.baseline_cost_yuan),
            ("② 储能·忽略需量", res_naive.peak_grid_kw, res_naive.optimized_cost_yuan),
            ("③ 储能·仅需量感知", res.peak_grid_kw, res.optimized_cost_yuan),
            ("④ 储能·主动削峰 80%", res_shave.peak_grid_kw, res_shave.optimized_cost_yuan),
        ]
        for name, peak, cost in rows:
            print(f"{name:<24}{peak:>9.1f}kW{peak * dc_price:>12,.0f} 元"
                  f"{cost:>12,.1f} 元")

        print("-" * 74)
        print("说明：")
        print("  ② 未加约束，谷时满功率充电把峰值推到比不配储能还高 —— 反面教材")
        print("  ③ 加了『充电不超过基线峰值』的约束，只是『不让情况恶化』：")
        print(f"     充电功率被顶到 {res.peak_grid_kw:,.1f} kW，与基线峰值相同，")
        print("     所以削峰量为 0，需量电费一分钱没省。")
        print("  ④ 设定削峰目标后，充电不超过目标、放电优先削峰并预留电量，")
        print(f"     峰值被真正压到 {res_shave.peak_grid_kw:,.1f} kW。")
        print()

        print("⚠️ 需量电费按【当月最大需量】计费，**不是按天累加**：")
        d_peak = res.baseline_peak_grid_kw - res_shave.peak_grid_kw
        if d_peak > 0:
            print(f"   ④ 相比基线削峰 {d_peak:,.1f} kW → "
                  f"当月基本电费少缴 {d_peak * dc_price:,.0f} 元")
            print(f"   全年 12 个月合计约 {d_peak * dc_price * 12:,.0f} 元")
        else:
            print("   ④ 未能削峰：储能容量/功率不足以覆盖峰值时段，")
            print("      属于 sizing（规划）环节要解决的问题。")
        print()
        print("注：④ 的削峰目标取多少，是要靠 sizing 算出来的（存在权衡：")
        print("    目标越低，需量省得越多，但谷时充电余量越小、套利收益下降）。")
        print("    本 demo 固定 80% 只为演示，最优值见 demo_sizing.py。")
        print()

    # ---- 逐时段明细（按小时汇总，便于阅读）----
    print("=" * 74)
    print("逐时明细")
    print("=" * 74)
    print(f"{'时刻':<6}{'时段':<6}{'电价':>7}{'负荷':>9}{'光伏':>9}"
          f"{'净负荷':>9}{'动作':>7}{'功率':>8}{'SOC':>7}")

    step = max(1, 60 // args.resolution)
    for i in range(0, len(res.entries), step):
        e = res.entries[i]
        hh = e.ts.strftime("%H:%M")
        tier = {"sharp": "尖峰", "peak": "高峰",
                "flat": "平段", "valley": "低谷"}.get(e.tier or "", "—")
        act = {"charge": "充电", "discharge": "放电", "idle": "待机"}[e.action]
        net = load[i] - pv[i]
        print(f"{hh:<6}{tier:<6}{e.price_yuan_per_kwh:>7.3f}"
              f"{load[i]:>9.1f}{pv[i]:>9.1f}{net:>9.1f}"
              f"{act:>7}{e.power_kw:>8.1f}{e.soc_eod:>7.1%}")

    print()
    print("=" * 74)
    print("提示：以上用的是合成曲线，仅用于验证链路。")
    print("     接入真实预测后，同样的调度器可直接复用。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
