"""演示数据种子脚本：写入江苏示例电价、南京示例站点和 3 天合成测量数据。

用法（在 backend/ 目录下）：
    python scripts/seed_demo.py

幂等：电价和站点已存在则跳过，测量点按 (site_id, data_type, ts_utc) 覆盖。
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.domain.models import MeasurementUpload, SiteCreate, TariffCreate  # noqa: E402
from app.infrastructure.db import Database, _zone  # noqa: E402

DB_PATH = BACKEND_DIR / "data" / "storage.db"
TARIFF_YAML = REPO_ROOT / "configs" / "tariff.jiangsu.yaml"
TARIFF_ID = "T-JIANGSU-DEMO"
SITE_ID = "S-NANJING-DEMO"
TIMEZONE = "Asia/Shanghai"
SEED = 20260921
SEED_DAYS = 3
RESOLUTION_MINUTES = 15


def ensure_tariff(db: Database) -> str:
    if db.get_tariff(TARIFF_ID) is not None:
        return "skipped"
    data = yaml.safe_load(TARIFF_YAML.read_text(encoding="utf-8"))
    db.create_tariff(TariffCreate.model_validate(data))
    return "created"


def ensure_site(db: Database) -> str:
    if db.get_site(SITE_ID) is not None:
        return "skipped"
    db.create_site(
        SiteCreate(
            site_id=SITE_ID,
            name="南京示例产业园",
            location={"lat": 32.06, "lon": 118.79, "timezone": TIMEZONE},
            pv={"capacity_kwp": 1000, "system_loss": 0.14},
            load={"peak_kw": 800, "profile_type": "factory_two_shift"},
            storage={
                "pcs_kw": 500,
                "capacity_kwh": 1000,
                "round_trip_eff": 0.88,
                "soc_min": 0.10,
                "soc_max": 0.95,
            },
            tariff_id=TARIFF_ID,
        )
    )
    return "created"


def pv_power_kw(hour_float: float, rng: random.Random, capacity_kwp: float, system_loss: float) -> float:
    if not 6.0 <= hour_float <= 18.0:
        return 0.0
    sine = math.sin(math.pi * (hour_float - 6.0) / 12.0)
    cloud = rng.uniform(0.75, 1.0)
    return round(capacity_kwp * (1 - system_loss) * sine * cloud, 3)


def load_power_kw(hour_float: float, rng: random.Random) -> float:
    if 8.0 <= hour_float < 20.0:
        base = 620 + 80 * math.sin(math.pi * (hour_float - 8.0) / 12.0)
    else:
        base = 160
    return round(max(base + rng.uniform(-40, 40), 0.0), 3)


def seed_measurements(db: Database, days: list[date], rng: random.Random) -> dict[str, int]:
    tz = _zone(TIMEZONE)
    site = db.get_site(SITE_ID)
    assert site is not None
    counts = {"pv_power": 0, "load_power": 0}
    points_per_day = 24 * 60 // RESOLUTION_MINUTES
    for day in days:
        start = datetime.combine(day, time.min, tzinfo=tz)
        pv_points = []
        load_points = []
        for index in range(points_per_day):
            ts = start + timedelta(minutes=RESOLUTION_MINUTES * index)
            hour_float = ts.hour + ts.minute / 60
            pv_points.append(
                {
                    "ts": ts.isoformat(),
                    "value": pv_power_kw(hour_float, rng, site.pv.capacity_kwp, site.pv.system_loss),
                    "quality": "estimated",
                }
            )
            load_points.append(
                {"ts": ts.isoformat(), "value": load_power_kw(hour_float, rng), "quality": "estimated"}
            )
        counts["pv_power"] += db.insert_measurements(
            SITE_ID,
            MeasurementUpload(
                data_type="pv_power", unit="kW", resolution_minutes=RESOLUTION_MINUTES,
                source="seed_demo", points=pv_points,
            ),
        )
        counts["load_power"] += db.insert_measurements(
            SITE_ID,
            MeasurementUpload(
                data_type="load_power", unit="kW", resolution_minutes=RESOLUTION_MINUTES,
                source="seed_demo", points=load_points,
            ),
        )
    return counts


def main() -> None:
    db = Database(str(DB_PATH))
    tz = _zone(TIMEZONE)
    today = datetime.now(tz).date()
    days = [today - timedelta(days=offset) for offset in range(SEED_DAYS, 0, -1)]
    rng = random.Random(SEED)

    tariff_status = ensure_tariff(db)
    site_status = ensure_site(db)
    counts = seed_measurements(db, days, rng)

    sample_request = {
        "site_id": SITE_ID,
        "date": days[-1].isoformat(),
        "initial_soc": 0.5,
        "strategy": "rule_v1",
        "resolution_minutes": 15,
    }
    print(f"数据库: {DB_PATH}")
    print(f"电价 {TARIFF_ID}: {tariff_status}")
    print(f"站点 {SITE_ID}: {site_status}")
    print(f"种子日期: {', '.join(day.isoformat() for day in days)} ({TIMEZONE})")
    for data_type, count in counts.items():
        print(f"写入 {data_type}: {count} 点")
    print("示例调度请求 POST /api/v1/storage/schedule:")
    print(json.dumps(sample_request, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
