"""欧洲电价预训练 -> 中国典型电价微调的离线实验。

这是域适配实验入口，不是广东真实效果验收：欧洲标签来自公开欧洲日前市场，
中国标签来自带 ``is_synthetic=True`` 的典型分时电价样本。

模型以小时为粒度。欧洲 EUR/MWh 先按 ``--eur-cny`` 换算为元/kWh，再用
LightGBM ``init_model`` 继续在中国典型数据上训练；中国最后一段日期只用于
留出评估，不能参与微调或保形校准。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.forecast.price_features import PriceSeries, build_samples, column_medians, select_active_columns, to_matrix
from app.domain.forecast.price_model import QUANTILES, _order


def _load_series(path: Path, *, ts_col: str, price_col: str, tz: str, scale: float) -> PriceSeries:
    raw = pd.read_csv(path, comment="#")
    if ts_col not in raw or price_col not in raw:
        raise ValueError(f"{path}: expected columns {ts_col!r}, {price_col!r}")
    ts = pd.to_datetime(raw[ts_col], errors="raise", utc=True)
    # UTC input is made explicit, then converted to the requested local zone. This
    # keeps all model timestamps timezone-aware while preserving the instant.
    ts = ts.dt.tz_convert(ZoneInfo(tz))
    frame = pd.DataFrame({"price": pd.to_numeric(raw[price_col], errors="raise").to_numpy() * scale}, index=ts)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    hourly = frame["price"].resample("h").mean()
    grid = pd.date_range(hourly.index.min().floor("h"), hourly.index.max().floor("h"), freq="h", tz=ZoneInfo(tz))
    hourly = hourly.reindex(grid)
    return PriceSeries(ts=grid, price=hourly.to_numpy(dtype="float64"))


def _samples(series: PriceSeries, origins: np.ndarray, horizon: int = 24):
    x, y, target_ts, _ = build_samples(series, origins=origins, horizon=horizon, min_origin=167, require_target=True)
    if len(x) < 500:
        raise ValueError(f"insufficient rows: {len(x)}")
    return x, y, target_ts


def _fit_quantiles(x, y, columns, medians, *, rounds, seed, init_models=None):
    from lightgbm import LGBMRegressor

    matrix = to_matrix(x, columns, medians)
    models = {}
    for level in QUANTILES:
        model = LGBMRegressor(
            objective="quantile", alpha=level, n_estimators=rounds,
            learning_rate=0.04, num_leaves=63, min_child_samples=40,
            colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
            random_state=seed, n_jobs=1, verbose=-1,
        )
        kwargs = {}
        if init_models is not None:
            kwargs["init_model"] = init_models[level].booster_
        model.fit(matrix, y, **kwargs)
        models[level] = model
    return models


def _predict(models, x, columns, medians):
    matrix = to_matrix(x, columns, medians)
    raw = {level: models[level].predict(matrix) for level in QUANTILES}
    return _order(raw[0.5], raw[0.5] - raw[0.1], raw[0.9] - raw[0.5])


def _metrics(y, pred):
    p10, p50, p90 = pred
    covered = (y >= p10) & (y <= p90)
    return {
        "rows": int(len(y)),
        "mae_p50_cny_kwh": float(np.mean(np.abs(y - p50))),
        "picp80": float(np.mean(covered)),
        "mean_interval_width_cny_kwh": float(np.mean(p90 - p10)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--europe-csv", type=Path, required=True)
    parser.add_argument("--china-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/price_forecast/transfer_eu_china"))
    parser.add_argument("--eur-cny", type=float, default=7.8, help="EUR/CNY conversion used only to align label units")
    parser.add_argument("--eu-val-days", type=int, default=14)
    parser.add_argument("--china-finetune-days", type=int, default=60)
    parser.add_argument("--china-val-days", type=int, default=30)
    parser.add_argument("--rounds-pretrain", type=int, default=300)
    parser.add_argument("--rounds-finetune", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.eur_cny <= 0 or min(args.eu_val_days, args.china_finetune_days, args.china_val_days) <= 0:
        parser.error("exchange rate and split days must be positive")

    # EUROPE: UTC source, converted to CNY/kWh (1 EUR/MWh = EUR/CNY / 1000 CNY/kWh).
    eu = _load_series(args.europe_csv, ts_col="timestamp_utc", price_col="price_eur_mwh", tz="UTC", scale=args.eur_cny / 1000.0)
    cn = _load_series(args.china_csv, ts_col="ts", price_col="price", tz="Asia/Shanghai", scale=1.0)
    eu_cut = len(eu.ts) - args.eu_val_days * 24
    cn_cut = args.china_finetune_days * 24
    cn_val_start = len(cn.ts) - args.china_val_days * 24
    if eu_cut <= 200 or cn_cut <= 200 or cn_val_start < cn_cut:
        raise ValueError("datasets are too short for requested causal splits")

    eu_train_origins = np.arange(167, eu_cut - 24, 3)
    eu_val_origins = np.arange(max(167, eu_cut), len(eu.ts) - 24, 1)
    cn_train_origins = np.arange(167, cn_cut - 24, 1)
    cn_val_origins = np.arange(max(167, cn_val_start), len(cn.ts) - 24, 1)
    eu_x, eu_y, _ = _samples(eu, eu_train_origins)
    eu_xv, eu_yv, _ = _samples(eu, eu_val_origins)
    cn_x, cn_y, _ = _samples(cn, cn_train_origins)
    cn_xv, cn_yv, _ = _samples(cn, cn_val_origins)

    columns = select_active_columns(eu_x)
    medians = column_medians(eu_x, columns)
    pretrain = _fit_quantiles(eu_x, eu_y, columns, medians, rounds=args.rounds_pretrain, seed=args.seed)
    pretrain_metrics = _metrics(eu_yv, _predict(pretrain, eu_xv, columns, medians))

    finetuned = _fit_quantiles(cn_x, cn_y, columns, medians, rounds=args.rounds_finetune, seed=args.seed, init_models=pretrain)
    finetune_metrics = _metrics(cn_yv, _predict(finetuned, cn_xv, columns, medians))

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "europe_pretrain_china_synthetic_finetune",
        "status": "domain_adaptation_only",
        "warning": "中国微调标签是合成典型分时电价，不是真实广东现货；结果不能作为广东真实效果验收。",
        "units": "CNY/kWh",
        "eur_cny": args.eur_cny,
        "europe": {"path": str(args.europe_csv), "timezone": "UTC", "rows": len(eu.ts), "train_hours": eu_cut, "validation_hours": len(eu.ts)-eu_cut},
        "china": {"path": str(args.china_csv), "timezone": "Asia/Shanghai", "rows": len(cn.ts), "finetune_hours": cn_cut, "validation_hours": len(cn.ts)-cn_val_start},
        "features": columns,
        "metrics": {"europe_pretrain_holdout": pretrain_metrics, "china_synthetic_holdout_after_finetune": finetune_metrics},
        "split": {"eu_val_days": args.eu_val_days, "china_finetune_days": args.china_finetune_days, "china_val_days": args.china_val_days},
    }
    (args.output / "transfer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
