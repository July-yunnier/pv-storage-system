"""国网公开光伏数据导入：50MW 场站功率标幺化 → 缩放到站点容量，配两班制仿真负荷。

对应交接文档《光储仿真数据接入与实验交接说明》P0 任务：
- 光伏：公开实测功率 ÷ 原场站容量 × 目标容量，裁剪到 [0, 容量]，
  source=public_replay、quality=measured（原始波动来自公开实测，经过容量缩放）
- 负荷：两班制日形 × 峰值，工作日/周末区分 + 确定性逐日扰动，
  source=simulator、quality=estimated（仿真输入，禁称真实工厂负荷）
- 质检：每天恰好 96 点、时间戳无重复、夜间功率≈0

用法（在 backend/ 目录下）：
    python scripts/import_guowang.py                    # 默认 site1，2019-01-01 起 730 天，平移到昨天结束
    python scripts/import_guowang.py --days 90 --start 2019-05-01 \
        --csv "../../dataset_guowang/solar_stations_xlsx/Solar station site 1 (Nominal capacity-50MW).csv"

日期平移（--shift-to，默认昨天）：把历史窗口映射为"最近 N 天历史"，
使预测（需最近 3 天）、自动计划（活站点守卫）、日内滚动等主链路对回放站点生效；
平移量写入 replay_config，live 实时推进据此把"今天"映射到源窗口外的下一天继续重播。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.domain.live_sim import LOAD_SHAPE, _hash01  # noqa: E402
from app.domain.models import MeasurementUpload  # noqa: E402
from app.infrastructure.db import Database, _zone  # noqa: E402

DB_PATH = BACKEND_DIR / "data" / "storage.db"
DEFAULT_CSV = (
    REPO_ROOT / "dataset_guowang" / "solar_stations_xlsx"
    / "Solar station site 1 (Nominal capacity-50MW).csv"
)
DEFAULT_SITE = "S-DONGGUAN-DEMO"
RESOLUTION_MINUTES = 15
POINTS_PER_DAY = 24 * 60 // RESOLUTION_MINUTES
WEEKEND_FACTOR = 0.35  # 周末保留值班/基础负荷


def parse_time(raw: str) -> datetime:
    text = raw.strip().strip('"')
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间戳: {raw!r}")


def read_pv_series(csv_path: Path) -> list[tuple[datetime, float]]:
    """读取国网站点 CSV，返回 [(naive_local_ts, power_mw)]，按时间升序。"""
    with csv_path.open(encoding="utf-8-sig", newline="") as fp:
        rows = [row for row in csv.reader(fp) if row and row[0].strip()]
    header = [h.strip().strip('"') for h in rows[0]]
    power_col = next(i for i, h in enumerate(header) if h.startswith("Power"))
    series: list[tuple[datetime, float]] = []
    for row in rows[1:]:
        series.append((parse_time(row[0]), float(row[power_col].strip().strip('"') or 0.0)))
    series.sort(key=lambda item: item[0])
    return series


def build_load_day(day: date, peak_kw: float) -> list[float]:
    """两班制负荷：LOAD_SHAPE × 峰值，工作日/周末区分 + 确定性扰动。"""
    day_key = day.isoformat()
    weekend = day.weekday() >= 5
    day_factor = 0.95 + 0.10 * _hash01("load-day", day_key)  # 逐日 ±5%
    values: list[float] = []
    for slot in range(POINTS_PER_DAY):
        hour = (slot * RESOLUTION_MINUTES) // 60
        minute = (slot * RESOLUTION_MINUTES) % 60
        frac = minute / 60
        next_hour = (hour + 1) % 24
        shape = LOAD_SHAPE[hour] * (1 - frac) + LOAD_SHAPE[next_hour] * frac
        if weekend:
            shape = min(shape * WEEKEND_FACTOR + 0.12, 1.0)
        wiggle = 0.94 + 0.12 * _hash01("load-slot", day_key, str(slot))
        values.append(round(peak_kw * shape * day_factor * wiggle, 3))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="国网公开光伏数据导入")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--site", default=DEFAULT_SITE)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--src-mw", type=float, default=50.0, help="原场站额定容量 MW")
    parser.add_argument(
        "--shift-to",
        type=date.fromisoformat,
        default=None,
        help="把窗口最后一天映射到该本地日期（默认昨天）：源 2019 数据按日平移为'最近历史'，"
        "预测/自动计划链路即可使用真实形态数据；同时写入 replay_config 供 live 回放驱动",
    )
    args = parser.parse_args()

    offset_days = 0
    if args.shift_to is None:
        # 默认平移：窗口最后一天 = 昨天（今天留给 live 回放推进）
        args.shift_to = date.today() - timedelta(days=1)
    offset_days = (args.shift_to - (args.start + timedelta(days=args.days - 1))).days

    db = Database(str(DB_PATH))
    site = db.get_site(args.site)
    if site is None:
        raise SystemExit(f"站点不存在，请先运行 seed 脚本: {args.site}")
    tz = _zone(site.location.timezone)
    cap_kw = site.pv.capacity_kwp
    peak_kw = site.load.peak_kw
    if peak_kw is None:
        raise SystemExit("站点未配置 load.peak_kw")

    print(f"读取: {args.csv}")
    series = read_pv_series(args.csv)
    print(f"源数据: {len(series)} 点，{series[0][0]} ~ {series[-1][0]}")

    end_date = args.start + timedelta(days=args.days)
    window = [(ts, mw) for ts, mw in series if args.start <= ts.date() < end_date]

    # ---- 质检 ----
    errors: list[str] = []
    days_seen: dict[date, int] = {}
    seen_ts: set[datetime] = set()
    for ts, _ in window:
        if ts in seen_ts:
            errors.append(f"重复时间戳: {ts}")
        seen_ts.add(ts)
        days_seen[ts.date()] = days_seen.get(ts.date(), 0) + 1
    incomplete = {d: n for d, n in days_seen.items() if n != POINTS_PER_DAY}
    if incomplete:
        errors.append(f"{len(incomplete)} 天不足 {POINTS_PER_DAY} 点: {list(incomplete)[:5]}...")
    missing_days = [
        args.start + timedelta(days=i) for i in range(args.days)
        if args.start + timedelta(days=i) not in days_seen
    ]
    if missing_days:
        errors.append(f"缺 {len(missing_days)} 天: {[d.isoformat() for d in missing_days[:5]]}...")
    if errors:
        print("质检未通过:")
        for err in errors:
            print(f"  ✗ {err}")
        raise SystemExit(1)
    print(f"质检通过: {args.days} 天 × {POINTS_PER_DAY} 点，无重复、无缺口")

    # ---- 写入（日期平移：src_day + offset_days → 本地日）----
    shift = timedelta(days=offset_days)
    counts = {"pv_power": 0, "load_power": 0}
    for offset in range(args.days):
        day = args.start + timedelta(days=offset)
        local_day = day + shift
        day_rows = [(ts, mw) for ts, mw in window if ts.date() == day]
        pv_points = [
            {
                "ts": (ts + shift).replace(tzinfo=tz).isoformat(),
                "value": round(min(max(mw / args.src_mw * cap_kw, 0.0), cap_kw), 3),
                "quality": "measured",
            }
            for ts, mw in day_rows
        ]
        load_values = build_load_day(local_day, peak_kw)
        start_dt = datetime.combine(local_day, time.min, tzinfo=tz)
        load_points = [
            {
                "ts": (start_dt + timedelta(minutes=RESOLUTION_MINUTES * i)).isoformat(),
                "value": load_values[i],
                "quality": "estimated",
            }
            for i in range(POINTS_PER_DAY)
        ]
        counts["pv_power"] += db.insert_measurements(
            args.site,
            MeasurementUpload(
                data_type="pv_power", unit="kW", resolution_minutes=RESOLUTION_MINUTES,
                source=f"public_replay_{args.csv.stem.split('(')[0].strip().replace(' ', '_').lower()}",
                points=pv_points,
            ),
        )
        counts["load_power"] += db.insert_measurements(
            args.site,
            MeasurementUpload(
                data_type="load_power", unit="kW", resolution_minutes=RESOLUTION_MINUTES,
                source="simulator",
                points=load_points,
            ),
        )

    db.set_replay_config(
        args.site,
        shift_days=offset_days,
        window_start=(args.start + shift).isoformat(),
        window_end=args.shift_to.isoformat(),
        cursor_date=None,  # live 首次推进时从 window_start 开始重播
        cursor_for=None,
    )

    pv_values = [mw / args.src_mw * cap_kw for _, mw in window]
    print(
        f"写入 {args.site} 源[{args.start} ~ {end_date - timedelta(days=1)}] → "
        f"本地[{args.start + shift} ~ {end_date - timedelta(days=1) + shift}]"
        f"（平移 {offset_days} 天）:"
    )
    for data_type, count in counts.items():
        print(f"  {data_type}: {count} 点")
    print(
        f"光伏缩放: {args.src_mw} MW → {cap_kw:.0f} kWp，"
        f"日发电量 {sum(pv_values) * 0.25 / args.days:,.0f} kWh/天（均值），"
        f"峰值 {max(pv_values):.0f} kW"
    )
    print(f"负荷: 峰值 {peak_kw:.0f} kW 两班制，周末系数 {WEEKEND_FACTOR}（按平移后日期）")
    print(f"replay_config: {args.site} shift_days={offset_days}（live 回放驱动依据）")


if __name__ == "__main__":
    main()
