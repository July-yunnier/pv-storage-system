"""生成可复现的典型工厂两班制负荷与 MES 排产仿真数据。

输出：
- factory_load_mes_90d.csv：用户要求的三列基础数据；
- factory_load_mes_90d_extended.csv：增加车间负荷、温度和湿度，可用于多任务训练。

这不是企业现场数据。假设来自公开的工商业典型日形：夜间基础负荷、
06:00-14:00 白班、14:00-22:00 晚班、周末降负荷，并加入设备启停和可复现扰动。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

SLOTS_PER_DAY = 96
CN_TZ = timezone(timedelta(hours=8))

HOURLY_SHAPE = [
    0.18, 0.16, 0.15, 0.15, 0.16, 0.20,
    0.32, 0.55, 0.78, 0.86, 0.88, 0.87,
    0.85, 0.84, 0.85, 0.86, 0.84, 0.80,
    0.72, 0.62, 0.52, 0.42, 0.32, 0.24,
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "training_artifacts"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-kw", type=float, default=1600.0)
    return parser.parse_args()


def _interpolated_shape(slot: int) -> float:
    hour = slot / 4.0
    left = min(int(hour), 23)
    right = min(left + 1, 23)
    fraction = hour - left
    return HOURLY_SHAPE[left] * (1.0 - fraction) + HOURLY_SHAPE[right] * fraction


def _mes_plan(day: date, slot: int, high_energy_day: bool) -> str:
    weekday = day.weekday()
    hour = slot // 4
    maintenance_day = weekday < 5 and (day - date(2026, 1, 1)).days % 14 == 13
    if maintenance_day and 8 <= hour < 18:
        return "maintenance"
    if weekday >= 5:
        return "weekend_base"
    if 6 <= hour < 8:
        return "startup"
    if 8 <= hour < 14:
        return "day_shift"
    if 14 <= hour < 18:
        return "high_energy" if high_energy_day else "day_shift"
    if 18 <= hour < 22:
        return "evening_shift"
    return "night_base"


def generate(start: date, days: int, peak_kw: float, seed: int) -> list[dict[str, object]]:
    if days < 35:
        raise ValueError("days 至少为35，才能使用默认30天历史窗口训练")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for day_offset in range(days):
        current = start + timedelta(days=day_offset)
        weekday_factor = {0: 1.0, 1: 1.02, 2: 0.98, 3: 1.04, 4: 1.01, 5: 0.72, 6: 0.50}[current.weekday()]
        day_factor = weekday_factor * (0.97 + 0.06 * rng.random())
        temperature = 8.0 + 10.0 * math.sin(2 * math.pi * day_offset / 90.0) + rng.gauss(0.0, 1.2)
        humidity = max(25.0, min(95.0, 62.0 - 14.0 * math.sin(2 * math.pi * day_offset / 90.0) + rng.gauss(0.0, 4.0)))
        high_energy_day = current.weekday() < 5 and rng.random() < 0.25

        for slot in range(SLOTS_PER_DAY):
            ts = datetime.combine(current, time.min, tzinfo=CN_TZ) + timedelta(minutes=15 * slot)
            mes_plan = _mes_plan(current, slot, high_energy_day)
            shape = _interpolated_shape(slot)
            if mes_plan == "weekend_base":
                profile_factor = 0.75
            elif mes_plan == "maintenance":
                profile_factor = 0.38
            elif mes_plan == "high_energy":
                profile_factor = 1.12
            elif mes_plan == "startup":
                profile_factor = 0.90
            else:
                profile_factor = 1.0
            noise = 1.0 + rng.gauss(0.0, 0.018)
            total_kw = max(80.0, peak_kw * shape * day_factor * profile_factor * noise)

            workshop_ratio = 0.30
            if mes_plan == "high_energy":
                workshop_ratio = 0.46
            elif mes_plan == "maintenance":
                workshop_ratio = 0.16
            workshop_kw = max(20.0, total_kw * workshop_ratio * (1.0 + rng.gauss(0.0, 0.025)))

            rows.append({
                "ts": ts.isoformat(),
                "total_load_kw": round(total_kw, 3),
                "mes_plan": mes_plan,
                "workshop_load_kw": round(min(workshop_kw, total_kw), 3),
                "temperature_c": round(temperature, 3),
                "humidity_pct": round(humidity, 3),
            })
    return rows


def _write(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def main() -> None:
    args = _arguments()
    rows = generate(date.fromisoformat(args.start), args.days, args.peak_kw, args.seed)
    output_dir = Path(args.output_dir)
    _write(output_dir / "factory_load_mes_90d.csv", rows, ["ts", "total_load_kw", "mes_plan"])
    _write(
        output_dir / "factory_load_mes_90d_extended.csv",
        rows,
        ["ts", "total_load_kw", "mes_plan", "workshop_load_kw", "temperature_c", "humidity_pct"],
    )
    print(f"generated_rows={len(rows)}")
    print(f"basic={output_dir / 'factory_load_mes_90d.csv'}")
    print(f"extended={output_dir / 'factory_load_mes_90d_extended.csv'}")


if __name__ == "__main__":
    main()
