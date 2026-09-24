#!/usr/bin/env python3
"""演示用合成数据：季节典型日场景。

⚠️ 这是**合成数据**，用于在拿到实测数据前跑通链路与验证算法。
   量级参考了江苏地区的实际情况，但**不可用于任何对外结论**。

为什么需要「多场景」而不是「一个典型日 × 365」
--------------------------------------------
光伏和负荷都有强季节性，而且方向常常相反：

    夏季  光伏出力受高温降效影响，但空调负荷冲高
    冬季  光伏出力低（辐照弱 + 日照短），取暖负荷偏高
    春秋  光伏出力最好（温度适宜），负荷最平

用单一典型日外推全年，等于假设全年都是那个季节。
sizing 算的是 10 年现金流，这种偏差会直接改变「该配多大」的结论。
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer.dispatch import synthetic_day          # noqa: E402
from src.optimizer.sizing import DayScenario              # noqa: E402

#: (名称, 代表日 (月, 日), 一年代表天数, 光伏系数, 负荷系数)
#: 天数之和必须为 365 —— sizing 会用天数加权算年收益。
SEASONS: list[tuple[str, tuple[int, int], int, float, float]] = [
    ("夏季", (7, 15), 92, 1.00, 1.25),
    ("春秋", (4, 15), 183, 1.05, 1.00),
    ("冬季", (1, 15), 90, 0.72, 1.10),
]


def seasonal_scenarios(
    site,
    provider,
    *,
    base_year: int = 2026,
    resolution_minutes: int = 15,
    seed: int = 0,
) -> list[DayScenario]:
    """生成三个季节的典型日场景。

    参数
    ----
    site
        站点档案（用其中的光伏装机、负荷水平）。
    provider
        电价 Provider。**各场景分别取该季节的电价**，
        这样省级电价模板里的季节性调整会自动生效。
    """
    out: list[DayScenario] = []

    for name, (month, day), days, pv_factor, load_factor in SEASONS:
        start = datetime(base_year, month, day)
        prices = provider.price_series(
            start, start + timedelta(days=1), resolution_minutes
        )
        load, pv = synthetic_day(site, prices, seed=seed + month)

        out.append(DayScenario(
            name=name,
            load_kw=load * load_factor,
            pv_kw=pv * pv_factor,
            prices=prices,
            days_per_year=float(days),
        ))

    return out


def season_table() -> str:
    """季节配置的说明表，供 demo 打印。"""
    lines = []
    for name, (m, d), days, pv_f, ld_f in SEASONS:
        lines.append(f"  {name}  {days:>3} 天  "
                     f"（代表日 {m:02d}-{d:02d}）  "
                     f"光伏 ×{pv_f:.2f}   负荷 ×{ld_f:.2f}")
    return "\n".join(lines)
