#!/usr/bin/env python3
"""配置自检：在跑任何计算之前，先确认配置文件是干净的。

为什么需要这个
--------------
电价时段配置漏一段，程序**不会报错**——它会静默地把该时段按平段计价，
于是收益测算就偏了，而且很难发现。

本脚本做四件事：
  1. 站点配置必填字段是否齐全
  2. 电价时段是否完整覆盖 24 小时（无空档、无重叠）
  3. 峰谷价差是否足够大（价差太小，储能套利模型不成立）
  4. 储能参数是否自洽（效率、SOC 区间、循环次数）

用法
----
    python scripts/check_config.py                      # 检查默认路径
    python scripts/check_config.py --site config/site.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tariff import load_tou_from_yaml  # noqa: E402

OK = "  [OK]  "
WARN = "  [警告]"
ERR = "  [错误]"


def check_site(cfg: dict) -> list[str]:
    errs: list[str] = []
    site = cfg.get("site", {})

    for path, label in [
        (("site", "id"), "站点 ID"),
        (("site", "location", "lat"), "纬度"),
        (("site", "location", "lon"), "经度"),
        (("site", "location", "timezone"), "时区"),
        (("site", "pv", "capacity_kwp"), "光伏装机 kWp"),
        (("site", "load", "annual_kwh"), "年用电量 kWh"),
    ]:
        node = cfg
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
        if node in (None, ""):
            errs.append(f"缺少必填项：{label}（{'/'.join(path)}）")

    loc = site.get("location", {})
    lat, lon = loc.get("lat"), loc.get("lon")
    if isinstance(lat, (int, float)) and not (-90 <= lat <= 90):
        errs.append(f"纬度越界：{lat}")
    if isinstance(lon, (int, float)) and not (-180 <= lon <= 180):
        errs.append(f"经度越界：{lon}")

    if lat is not None and lon is not None:
        print(f"{OK}站点位置：({lat}, {lon}) {loc.get('province', '')}")
    print(f"{OK}光伏装机：{site.get('pv', {}).get('capacity_kwp')} kWp")
    print(f"{OK}年用电量：{site.get('load', {}).get('annual_kwh'):,} kWh")

    return errs


def check_tariff(cfg: dict, root: str) -> list[str]:
    errs: list[str] = []
    tariff = cfg.get("tariff", {})
    mode = tariff.get("mode", "tou")
    print(f"{OK}电价模式：{mode}")

    if mode == "tou":
        print(f"{OK}分时电价由政策给定，无需价格预测")

    template = tariff.get("template")
    if not template:
        errs.append("tariff.template 未配置")
        return errs

    path = os.path.join(root, "config", template)
    if not os.path.exists(path):
        errs.append(f"电价模板不存在：{path}")
        return errs

    p = load_tou_from_yaml(path, voltage_level=tariff.get("voltage_level"))

    uncovered = p.validate_coverage(resolution_minutes=15)
    if uncovered:
        hh = ", ".join(f"{m // 60:02d}:{m % 60:02d}" for m in uncovered[:6])
        errs.append(f"电价时段存在空档（共 {len(uncovered)} 个 15 分钟点）：{hh} …")
    else:
        print(f"{OK}电价时段完整覆盖 24 小时（无空档）")

    prices = p.describe().get("prices", {})
    if prices:
        lo = min(prices.values())
        hi = max(prices.values())
        spread = hi - lo
        print(f"{OK}价差：{lo:.2f} ~ {hi:.2f} 元/kWh，峰谷差 {spread:.2f}")
        if spread < 0.5:
            errs.append(
                f"峰谷价差仅 {spread:.2f} 元/kWh，偏低。"
                "两充两放策略通常需要 0.6 元以上价差才能覆盖度电成本"
            )

    print(f"{OK}输配 + 基金加价：{p.transmission_and_fund:.3f} 元/kWh")
    return errs


def check_storage(cfg: dict) -> list[str]:
    errs: list[str] = []
    st = cfg.get("site", {}).get("storage", {})
    if not st:
        errs.append("site.storage 未配置")
        return errs

    rte = st.get("round_trip_eff")
    soc_min, soc_max = st.get("soc_min"), st.get("soc_max")
    cyc = st.get("cycles_per_day_max", 0)

    if rte is not None:
        if not (0.5 <= rte <= 1.0):
            errs.append(f"综合效率 {rte} 不合理（应在 0.5~1.0）")
        else:
            loss = (1 - rte) * 100
            print(f"{OK}综合效率 {rte:.2f}（单次循环损耗 {loss:.1f}%）")

    if soc_min is not None and soc_max is not None:
        if not (0 <= soc_min < soc_max <= 1):
            errs.append(f"SOC 区间不合理：[{soc_min}, {soc_max}]")
        else:
            usable = (soc_max - soc_min)
            cap = st.get("capacity_kwh", 0)
            print(f"{OK}SOC 区间 [{soc_min:.0%}, {soc_max:.0%}]，"
                  f"可用容量 {usable:.0%} = {cap * usable:.0f} kWh")

    if cyc and cyc > 3:
        errs.append(f"每日循环次数 {cyc} 偏高（常规不超过 2）")

    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="config/site.yaml")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(args.site):
        example = os.path.join(root, "config", "site.example.yaml")
        print(f"{ERR} 找不到 {args.site}")
        print(f"        先复制模板： cp {os.path.relpath(example, os.getcwd())} {args.site}")
        return 1

    with open(args.site, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 66)
    print("配置自检")
    print("=" * 66)

    print("\n[1/3] 站点档案")
    errs = check_site(cfg)

    print("\n[2/3] 电价配置")
    errs += check_tariff(cfg, root)

    print("\n[3/3] 储能参数")
    errs += check_storage(cfg)

    print()
    print("=" * 66)
    if errs:
        print(f"发现 {len(errs)} 个问题：")
        for i, e in enumerate(errs, 1):
            print(f"  {i}. {e}")
        print("=" * 66)
        return 1

    print("全部通过，配置可用于计算。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
