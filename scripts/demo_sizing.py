#!/usr/bin/env python3
"""端到端演示：从配置算出「该配多大储能、几年回本」。

链路：
    读配置 → 建电价 Provider → 生成多季节场景 → 扫描候选容量
    → 推荐 NPV 最优方案 → 敏感性分析 → 自洽性验证

不需要任何外部数据源（用合成曲线跑通）。

用法
----
    python scripts/demo_sizing.py
    python scripts/demo_sizing.py --site config/site.yaml --max-payback 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (str(ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth import seasonal_scenarios, season_table              # noqa: E402
from src.models import site_from_config                          # noqa: E402
from src.optimizer.dispatch import rule_based_dispatch           # noqa: E402
from src.optimizer.sizing import (                               # noqa: E402
    EconomicsConfig,
    plan_capacity,
)
from src.tariff import get_provider                              # noqa: E402


def _w(s: str) -> int:
    """字符串显示宽度：中日韩字符占 2 列，其余占 1 列。

    直接用 f-string 的 <20 对齐中文会歪——那是按字符数算的。
    """
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


def pad(s, width: int, align: str = "<") -> str:
    s = str(s)
    gap = max(0, width - _w(s))
    if align == ">":
        return " " * gap + s
    if align == "^":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="config/site.yaml")
    ap.add_argument("--date", default="2026-07-15",
                    help="基准日期（各季节在此年份内取代表日）")
    ap.add_argument("--resolution", type=int, default=15)
    ap.add_argument("--max-payback", type=float, default=8.0,
                    help="回收期上限（年），超过则视为不可行")
    ap.add_argument("--top", type=int, default=8,
                    help="扫描明细显示前 N 条")
    args = ap.parse_args()

    site_path = args.site if os.path.isabs(args.site) else os.path.join(ROOT, args.site)
    if not os.path.exists(site_path):
        print(f"找不到配置 {site_path}")
        print("先复制模板： cp config/site.example.yaml config/site.yaml")
        return 1

    with open(site_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    site = site_from_config(cfg["site"])
    if site.storage is None:
        print("配置里没有 site.storage —— 规划需要它作为参数模板")
        print("（效率、SOC 区间、化学体系等；容量会被候选值覆盖）")
        return 1

    tariff_cfg = cfg["tariff"]
    provider = get_provider(
        tariff_cfg.get("mode", "tou"),
        template_path=os.path.join(ROOT, "config", tariff_cfg["template"]),
    )

    econ = EconomicsConfig.from_dict(cfg.get("economics"))
    dc = tariff_cfg.get("demand_charge", {})
    dc_price = float(dc.get("price_per_kw_month", 0)) if dc.get("enabled") else 0.0
    opt = cfg.get("optimizer", {})

    # ================================================================
    print("=" * 78)
    print("光储系统 · 容量规划（Sizing）演示")
    print("=" * 78)
    print(site.describe())
    print(f"  参数模板  效率 {site.storage.round_trip_eff:.2f}  "
          f"SOC [{site.storage.soc_min:.0%}, {site.storage.soc_max:.0%}]  "
          f"{site.storage.chemistry}")
    print()
    print(f"电价模式：{provider.describe()['note']}")
    if dc_price > 0:
        print(f"需量电费：{dc_price:.0f} 元/kW·月（按当月最大需量计费）")
    else:
        print("需量电费：未启用")
    print()

    # ---- 场景 ----
    print("=" * 78)
    print("典型日场景（年收益按天数加权，不用单日 × 365）")
    print("=" * 78)
    print(season_table())
    print()

    scenarios = seasonal_scenarios(
        site, provider,
        base_year=int(args.date[:4]),
        resolution_minutes=args.resolution,
    )
    total_days = sum(sc.days_per_year for sc in scenarios)
    print(f"合计 {total_days:.0f} 天")
    for sc in scenarios:
        print(f"  {sc.name}  负荷峰 {sc.load_kw.max():>7.1f} kW  "
              f"光伏峰 {sc.pv_kw.max():>7.1f} kW  "
              f"日电量 {sc.pv_kw.sum() * args.resolution / 60:>8,.0f} kWh")
    print()

    # ---- 规划 ----
    print("=" * 78)
    print("扫描候选容量")
    print("=" * 78)
    result = plan_capacity(
        scenarios, site, econ,
        demand_charge_price=dc_price,
        resolution_minutes=args.resolution,
        allow_grid_charge=opt.get("allow_grid_charge", True),
        allow_grid_export=opt.get("allow_grid_export", False),
        max_payback_years=args.max_payback,
    )

    cands = sorted(result.candidates, key=lambda r: -r.npv_yuan)
    print(f"共评估 {len(cands)} 个候选，按 NPV 排序（前 {args.top} 条）")
    print("每个候选都已对 5 档削峰目标试算，取年收益最大者。")
    print("-" * 78)
    hdr = (pad("功率kW", 8, ">") + pad("容量kWh", 9, ">")
           + pad("削峰", 7, ">")
           + pad("年收益元", 11, ">") + pad("投资元", 11, ">")
           + pad("回收期", 8, ">") + pad("NPV元", 12, ">") + "  可行")
    print(hdr)
    print("-" * 78)
    for r in cands[:args.top]:
        shave = "—" if r.shaving_ratio >= 0.999 else f"{r.shaving_ratio:.0%}"
        print(pad(f"{r.power_kw:.0f}", 8, ">")
              + pad(f"{r.capacity_kwh:.0f}", 9, ">")
              + pad(shave, 7, ">")
              + pad(f"{r.annual_benefit_yuan:,.0f}", 11, ">")
              + pad(f"{r.capex_yuan:,.0f}", 11, ">")
              + pad(f"{r.payback_years:.2f}", 8, ">")
              + pad(f"{r.npv_yuan:,.0f}", 12, ">")
              + "  " + ("是" if r.feasible else "否"))
    print()
    print("  「削峰」列 = 该配置的最优削峰目标（相对基线峰值）。")
    print("  目标越低 → 需量电费省得越多，但谷时充电的余量越小、套利收益下降。")
    print("  两者存在权衡，所以每个容量都试算了 100/90/80/70/60% 五档。")
    print()

    bad = [r for r in cands if not r.feasible]
    if bad:
        print(f"另有 {len(bad)} 个候选不可行（回收期 > {args.max_payback:.0f} 年）")
        print("  典型原因：容量过大 → 投资上去了，但每天只充放一次，收益不增")
        print()

    # ---- 推荐方案 ----
    print("=" * 78)
    print("推荐配置")
    print("=" * 78)
    print(result.summary())
    rec0 = result.recommendation
    if rec0.peak_shaving_target_ratio < 0.999:
        print(f"削峰目标   {rec0.peak_shaving_target_ratio:.0%} 基线峰值"
              f"（放电优先保证把峰值削到该线以下，并为后续高峰预留电量）")
    else:
        print("削峰目标   不主动削峰（1.00）——本轮扫描中削峰收益不占优")
    print()

    print("收益构成")
    print("-" * 78)
    e = result.economics
    arb_pct = e.arbitrage_yuan / e.annual_benefit_yuan * 100 if e.annual_benefit_yuan else 0
    dem_pct = 100 - arb_pct
    print(f"  峰谷套利     {e.arbitrage_yuan:>12,.0f} 元/年   ({arb_pct:>5.1f}%)")
    print(f"  需量电费节省 {e.demand_saving_yuan:>12,.0f} 元/年   ({dem_pct:>5.1f}%)")
    print(f"  合计         {e.annual_benefit_yuan:>12,.0f} 元/年")
    print(f"  减运维       {e.opex_yuan_per_year:>12,.0f} 元/年")
    print(f"  年净现金流   {e.annual_net_yuan:>12,.0f} 元/年")
    print()
    print(f"  回收期 {e.simple_payback_years:.2f} 年   "
          f"IRR {e.irr:.1%}   NPV {e.npv_yuan:,.0f} 元 "
          f"（{econ.project_years} 年，折现率 {econ.discount_rate:.0%}）")
    print()

    # ---- 敏感性分析 ----
    print("=" * 78)
    print("敏感性分析")
    print("=" * 78)
    print(pad("变量", 10) + pad("变化", 10) + pad("回收期", 10, ">")
          + pad("NPV元", 16, ">") + "   vs 基准")
    print("-" * 78)
    base_npv, base_pb = e.npv_yuan, e.simple_payback_years
    print(pad("基准", 10) + pad("—", 10) + pad(f"{base_pb:.2f}", 10, ">")
          + pad(f"{base_npv:,.0f}", 16, ">") + "   —")
    for s in result.sensitivity:
        d_npv = s.npv_yuan - base_npv
        d_pb = s.payback_years - base_pb
        print(pad(s.variable, 10) + pad(s.change, 10)
              + pad(f"{s.payback_years:.2f}", 10, ">")
              + pad(f"{s.npv_yuan:,.0f}", 16, ">")
              + f"   NPV {d_npv:+,.0f} / 回收期 {d_pb:+.2f} 年")
    print()
    print("  注：价差变化只缩放「峰谷差」，整体价格水平保持不变——")
    print("      这样检验的才是「价差风险」，而不是「电价水平下降」。")
    print()

    # ---- 自洽性验证 ----
    print("=" * 78)
    print("自洽性验证（规划与运行必须用同一个内核）")
    print("=" * 78)
    rec = result.recommendation
    from dataclasses import replace as _replace
    st = _replace(site.storage, pcs_kw=rec.power_kw, capacity_kwh=rec.capacity_kwh)

    def _target(sc):
        """按推荐方案的削峰目标，算出该场景的绝对目标功率。"""
        net = sc.load_kw - sc.pv_kw
        return float(np.max(np.clip(net, 0.0, None))) * rec.peak_shaving_target_ratio

    arb_v = dem_v = 0.0
    for sc in scenarios:
        rr = rule_based_dispatch(
            sc.load_kw, sc.pv_kw, sc.prices, st,
            resolution_minutes=args.resolution,
            allow_grid_charge=opt.get("allow_grid_charge", True),
            allow_grid_export=opt.get("allow_grid_export", False),
            demand_aware=True,
            peak_target_kw=_target(sc),
        )
        arb_v += rr.savings_yuan * sc.days_per_year
        dem_v += rr.peak_reduction_kw * dc_price * 12.0 * (sc.days_per_year / 365.0)

    verified = arb_v + dem_v
    diff = abs(verified - e.annual_benefit_yuan) / max(e.annual_benefit_yuan, 1e-9)
    print(f"  规划报告的加权年收益   {e.annual_benefit_yuan:>12,.0f} 元")
    print(f"  重跑 dispatch 复核     {verified:>12,.0f} 元")
    print(f"  差异                   {diff:>12.2%}   "
          f"{'通过（同一内核，必然一致）' if diff < 0.001 else '⚠️ 不一致，需排查'}")
    print()

    # 单场景外推 vs 多场景加权 —— 说明为什么必须多场景
    print("  对比：若只用单一典型日 × 365 会偏多少")
    print("  " + "-" * 74)
    for sc in scenarios:
        rr = rule_based_dispatch(
            sc.load_kw, sc.pv_kw, sc.prices, st,
            resolution_minutes=args.resolution,
            allow_grid_charge=opt.get("allow_grid_charge", True),
            allow_grid_export=opt.get("allow_grid_export", False),
            demand_aware=True,
            peak_target_kw=_target(sc),
        )
        one = rr.savings_yuan * 365.0 + rr.peak_reduction_kw * dc_price * 12.0
        dev = (one - verified) / max(verified, 1e-9)
        print(f"    {pad(sc.name, 6)}外推 {one:>12,.0f} 元   偏差 {dev:>+7.1%}")
    print()
    print("  这正是 sizing 必须用多场景加权的原因：")
    print("  拿单一季节外推，年收益偏差可达 10% 且**方向不确定**——")
    print("  用夏季会高估（空调用电多、套利机会多），用冬季会低估。")
    print("  而 10 年现金流的折现会把这个偏差放大成「该配多大」的错误结论。")
    print()

    # ---- 提示与假设 ----
    if result.notes:
        print("=" * 78)
        print("提示")
        print("=" * 78)
        for n in result.notes:
            print(f"  · {n}")
        print()

    print("=" * 78)
    print("关键假设（给企业看方案时必须一并给出）")
    print("=" * 78)
    a = result.assumptions
    print(f"  电池单价      {a['battery_price_yuan_per_kwh']:,.0f} 元/kWh"
          f"（{a['battery_price_note']}）")
    print(f"  电池寿命      {a['battery_life_years']} 年 / "
          f"{a['battery_cycle_life']:,.0f} 次循环")
    print(f"  折现率        {a['discount_rate']:.0%}   "
          f"评估期 {a['project_years']} 年")
    print(f"  年运维        {a['opex_ratio']:.1%} 初始投资")
    print(f"  期末残值      {a['residual_value_ratio']:.0%} 初始投资")
    print(f"  需量电费      {a['demand_charge_note']}")
    print(f"  场景加权      {a['scenario_note']}")
    print(f"  需量感知      {'已启用' if a['demand_aware'] else '未启用'}"
          f"（充电不得推高并网峰值）")
    print()
    print("=" * 78)
    print("⚠️ 以上使用的是合成曲线与行业参考单价，仅用于验证链路。")
    print("   接真实项目时，必须替换为：实测负荷曲线 / 预测光伏出力 /")
    print("   当地最新目录电价 / 厂商实际报价。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
