"""东莞实验站点种子：写入广东珠三角示例电价 + 东莞站点（不写测量数据）。

测量数据由 scripts/import_guowang.py 灌入（国网公开光伏回放 + 两班制负荷仿真）。

用法（在 backend/ 目录下）：
    python scripts/seed_dongguan.py

幂等：电价和站点已存在则跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.domain.models import SiteCreate, TariffCreate  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402

DB_PATH = BACKEND_DIR / "data" / "storage.db"
TARIFF_YAML = REPO_ROOT / "configs" / "tariff.guangdong.yaml"
TARIFF_ID = "T-GUANGDONG-DEMO"
SITE_ID = "S-DONGGUAN-DEMO"
TIMEZONE = "Asia/Shanghai"


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
            name="东莞示例电子产业园",
            location={"lat": 22.94, "lon": 113.75, "timezone": TIMEZONE},
            pv={"capacity_kwp": 1000, "tilt_deg": 20, "azimuth_deg": 180, "system_loss": 0.14},
            load={"peak_kw": 1500, "profile_type": "factory_two_shift"},
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


def main() -> None:
    db = Database(str(DB_PATH))
    print(f"数据库: {DB_PATH}")
    print(f"电价 {TARIFF_ID}: {ensure_tariff(db)}")
    print(f"站点 {SITE_ID}: {ensure_site(db)}")


if __name__ == "__main__":
    main()
