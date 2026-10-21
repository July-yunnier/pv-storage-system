"""训练站点级电价预测工件（LightGBM 分位数），供在线 `/sites/{id}/price/forecast` 使用。

用法::

    python scripts/train_price_model.py --site S-XXX --db data/storage.db

产物写到 ``data/price_models/{site_id}/``（``q10/q50/q90.txt`` + ``manifest.json``），
可用环境变量 ``PRICE_MODEL_DIR`` 覆盖根目录。

数据要求：至少 ``--min-days`` 天的小时级现货电价（默认 45 天）。样本不足时不要
硬训——直接返回非零退出码，让在线链路走统计画像，比训练一个过拟合模型更诚实。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.application.price_service import build_price_series, model_dir  # noqa: E402
from app.domain.forecast.price_features import PriceSeries  # noqa: E402
from app.domain.forecast.price_model import LgbmQuantileForecaster  # noqa: E402
from app.infrastructure.db import Database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--db", default="data/storage.db")
    parser.add_argument("--min-days", type=int, default=45)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--origin-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = Database(args.db)
    site = db.get_site(args.site)
    if site is None:
        print(f"site not found: {args.site}", file=sys.stderr)
        return 2
    tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
    if tariff is None:
        print("tariff not found; create the site with a valid tariff first", file=sys.stderr)
        return 2

    tz = ZoneInfo(site.location.timezone)
    bounds = db.spot_price_bounds(args.site)
    if bounds is None:
        print("no spot price history for this site", file=sys.stderr)
        return 2
    days_available = (bounds[1] - bounds[0]).total_seconds() / 86400
    if days_available < args.min_days:
        print(
            f"only {days_available:.1f} days of spot prices, need >= {args.min_days}; "
            "online forecasting will use the diurnal profile instead",
            file=sys.stderr,
        )
        return 3

    target_end = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    series, _, _ = build_price_series(db, site, tariff, target_end)

    origins = np.arange(0, len(series.ts), max(1, args.origin_stride))
    model = LgbmQuantileForecaster().fit(
        series,
        horizon=24,
        origins=origins,
        num_boost_round=args.rounds,
        seed=args.seed,
    )
    target = model_dir() / args.site
    model.save(target)

    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "site_id": args.site,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_hours": int(len(series.ts)),
            "train_rows_note": "origin × horizon 全量展开，origin 按 stride 抽样",
            "origin_stride": args.origin_stride,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved price model for {args.site} -> {target}")
    print(f"  hours={len(series.ts)} features={len(manifest['columns'])} stride={args.origin_stride}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
