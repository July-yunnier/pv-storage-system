"""把现货电价序列导入站点库，供电价预测与精度对账使用。

两种来源：

1. **真实市场序列（CSV）** —— 生产用法：
   ``python scripts/import_spot_price.py --site S-XXX --csv spot.csv --source 广东现货 --db data/storage.db``
   CSV 至少两列：``ts,price``（ts 必须带时区或用 ``--tz`` 指定本地时区）。

2. **GEFCom2014-P 公开数据** —— 演示/回归用法：
   ``python scripts/import_spot_price.py --site S-DONGGUAN-DEMO --gefcom ../data/power/price/Price --recent``
   这是 PJM 区域 1 的真实 LMP，**不是广东电价**；``--recent`` 只是把时间轴平移到
   最近 N 天，方便在页面上看到"过去 90 天 + 明天预测"。导入后 source 标记为
   ``gefcom2014p``，前端会显式提示数据来源。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.forecast.gefcom_price import load_gefcom_price  # noqa: E402
from app.domain.models import SpotPricePoint, SpotPriceUpload, generate_site_id  # noqa: E402,F401
from app.infrastructure.db import Database  # noqa: E402


def _push(db: Database, site_id: str, frame: pd.DataFrame, source: str, chunk: int = 2000) -> int:
    total = 0
    for start in range(0, len(frame), chunk):
        block = frame.iloc[start : start + chunk]
        points = [
            SpotPricePoint(
                ts=row.ts.to_pydatetime(),
                price_cny_per_kwh=float(row.price),
                quality="measured",
            )
            for row in block.itertuples()
        ]
        total += db.insert_spot_prices(site_id, SpotPriceUpload(source=source, points=points))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--db", default="data/storage.db")
    parser.add_argument("--csv", help="CSV 路径，需含 ts 与 price 列")
    parser.add_argument("--ts-col", default="ts")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--tz", default="Asia/Shanghai")
    parser.add_argument("--source", default="csv")
    parser.add_argument("--gefcom", help="GEFCom2014 Price/ 目录")
    parser.add_argument("--recent", action="store_true", help="把时间轴平移到最近 N 天")
    parser.add_argument("--keep-days", type=int, default=120, help="保留多少天历史")
    parser.add_argument("--scale", type=float, default=1.0, help="价格缩放（单位换算用，如 $/MWh→¥/kWh 需 0.0072）")
    parser.add_argument(
        "--profile",
        choices=["raw", "guangdong"],
        default="raw",
        help="raw=按给定列直接入库；guangdong=按交易所导出格式规范化（自动识别元/MWh、补时区、15 分钟聚合到小时）",
    )
    parser.add_argument("--unit", default="auto", choices=["auto", "CNY/MWh", "CNY/kWh"])
    parser.add_argument(
        "--keep-resolution",
        action="store_true",
        help="guangdong 模式下保留原始 15 分钟粒度（默认聚合到小时，与模型粒度一致）",
    )
    args = parser.parse_args()

    db = Database(args.db)
    if db.get_site(args.site) is None:
        print(f"site not found: {args.site}", file=sys.stderr)
        return 2

    tz = ZoneInfo(args.tz)
    if args.gefcom:
        gefcom = load_gefcom_price(args.gefcom)
        series = gefcom.series
        mask = pd.Series(series.price).notna().to_numpy()
        frame = pd.DataFrame({"ts": series.ts[mask], "price": series.price[mask]})
        frame = frame.tail(args.keep_days * 24).reset_index(drop=True)
        if args.recent:
            # 平移到"最近一个整点结束"，保持小时连续；只改时间轴，不动价格数值
            now_local = datetime.now(tz).replace(minute=0, second=0, microsecond=0, tzinfo=None)
            shift = pd.Timestamp(now_local) - pd.Timestamp(frame["ts"].iloc[-1])
            frame["ts"] = frame["ts"] + shift
        frame["ts"] = frame["ts"].dt.tz_localize(tz)
        source = "gefcom2014p"
    elif args.csv:
        raw = pd.read_csv(args.csv)
        if args.profile == "guangdong":
            from app.domain.forecast.price_ingest import aggregate_hourly, normalize_spot

            try:
                clean, report = normalize_spot(
                    raw,
                    ts_col=args.ts_col if args.ts_col in raw.columns else None,
                    price_col=args.price_col if args.price_col in raw.columns else None,
                    tz=args.tz,
                    unit=args.unit,
                )
            except ValueError as exc:
                print(f"normalize failed: {exc}", file=sys.stderr)
                return 2
            if not args.keep_resolution:
                clean = aggregate_hourly(clean)
            report.rows_out = int(len(clean))
            print("[ingest report] " + json.dumps(report.as_dict(), ensure_ascii=False))
            frame = clean
            source = args.source if args.source != "csv" else "guangdong"
        else:
            if args.ts_col not in raw.columns or args.price_col not in raw.columns:
                print(f"csv must contain columns {args.ts_col!r} and {args.price_col!r}", file=sys.stderr)
                return 2
            frame = pd.DataFrame({"ts": pd.to_datetime(raw[args.ts_col]), "price": raw[args.price_col]})
            frame = frame.dropna(subset=["price"]).sort_values("ts")
            if frame["ts"].dt.tz is None:
                frame["ts"] = frame["ts"].dt.tz_localize(tz)
            source = args.source
    else:
        print("either --csv or --gefcom is required", file=sys.stderr)
        return 2

    frame["price"] = frame["price"].astype(float) * args.scale
    accepted = _push(db, args.site, frame, source)
    bounds = db.spot_price_bounds(args.site)
    print(
        f"imported {accepted} spot price points for {args.site} "
        f"(source={source}, range={bounds[0].isoformat()} .. {bounds[1].isoformat()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
