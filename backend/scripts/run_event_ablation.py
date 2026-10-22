"""Paired rolling event/no-event experiment on auditable same-market spot prices.

--example-smoke is synthetic engineering validation, never market-effect evidence.
Real runs reject missing labels, unknown coverage and expired fuel observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.event_calendar import EVENT_FEATURES, EventCalendar  # noqa: E402
from app.domain.forecast.event_data import aware, digest  # noqa: E402
from app.domain.forecast.price_features import PriceSeries  # noqa: E402
from app.domain.forecast.price_metrics import summarize  # noqa: E402
from app.domain.forecast.price_model import LgbmQuantileForecaster  # noqa: E402
from scripts.run_price_benchmark import _forecast_window, _peak_rss_mb  # noqa: E402

METRICS = ("mae", "pinball", "picp80", "winkler80")


def paired_intervals(by_day, *, repetitions=2000, block_days=7, seed=42):
    """Circular moving day-block bootstrap of paired differences (with - without)."""
    if repetitions < 100 or block_days < 1:
        raise ValueError("bootstrap requires repetitions >= 100 and block_days >= 1")
    left = by_day[by_day["model"] == "with_events"].set_index("day").sort_index()
    right = by_day[by_day["model"] == "without_events"].set_index("day").sort_index()
    if (left.index.has_duplicates or right.index.has_duplicates or set(left.index) != set(right.index)
            or len(left) < 2 * block_days):
        raise ValueError("bootstrap requires enough exactly paired days")
    delta = left[list(METRICS)].to_numpy() - right[list(METRICS)].to_numpy()
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite daily metrics")
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty((repetitions, len(METRICS)))
    for iteration in range(repetitions):
        starts = rng.integers(0, n, size=(n + block_days - 1) // block_days)
        indices = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
        means[iteration] = delta[indices].mean(axis=0)
    return pd.DataFrame([
        {"metric": metric, "with_events": left[metric].mean(), "without_events": right[metric].mean(),
         "difference_with_minus_without": delta[:, pos].mean(),
         "ci95_low": np.quantile(means[:, pos], .025), "ci95_high": np.quantile(means[:, pos], .975),
         "n_days": n, "block_days": block_days, "bootstrap_repetitions": repetitions}
        for pos, metric in enumerate(METRICS)])


def load_prices(path, meta_path, calendar, example):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_kind = "example" if example else "real"
    if (meta.get("data_kind") != expected_kind or calendar.data is None
            or calendar.data.data_kind != expected_kind):
        raise ValueError("real data is required; fixtures require --example-smoke")
    if meta.get("market") != calendar.data.market or meta.get("timezone") != calendar.data.timezone:
        raise ValueError("price and event market/timezone must match")
    if meta.get("unit") != "CNY/kWh" or meta.get("price_kind") != "spot":
        raise ValueError("only spot prices in CNY/kWh are eligible; TOU demo is not spot data")
    if meta.get("price_sha256") != digest(path):
        raise ValueError("price CSV hash mismatch")
    url = urlparse(meta.get("source_url", ""))
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise ValueError("price source_url is required")
    rel = Path(meta.get("source_file", ""))
    source = (meta_path.parent / rel).resolve()
    if rel.is_absolute() or not source.is_relative_to(meta_path.parent.resolve()) or not source.is_file():
        raise ValueError("price source_file must exist inside metadata directory")
    if digest(source) != meta.get("source_sha256"):
        raise ValueError("price source hash mismatch")
    frame = pd.read_csv(path, comment="#")
    stamps = [aware(value) for value in frame["ts"]]
    ts = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_convert(meta["timezone"])
    prices = pd.to_numeric(frame["price"], errors="raise").to_numpy(dtype=float)
    if (not len(ts) or ts.has_duplicates or not ts.is_monotonic_increasing
            or not np.isfinite(prices).all() or not ts.equals(pd.date_range(ts[0], ts[-1], freq="h"))
            or any(stamp.minute or stamp.second or stamp.microsecond for stamp in ts)):
        raise ValueError("prices must be complete, ordered, finite hourly observations")
    return PriceSeries(ts=ts, price=prices, event_calendar=calendar), meta


def feature_audit(series, origins, horizon=24):
    missing = {key: 0 for key in EVENT_FEATURES}
    rows = 0
    for origin in origins:
        target = series.ts[origin + 1:origin + horizon + 1]
        values = series.event_calendar.features(target, issued_at=series.ts[origin])
        rows += len(target)
        for key in EVENT_FEATURES:
            missing[key] += int((~np.isfinite(values[key])).sum())
    return {"rows": rows, "missing": missing}


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty; preserve previous run evidence")
    if (args.days < (2 if args.example_smoke else 30) or args.train_days < 30 or args.cal_days < 0
            or args.rounds < 1 or args.stride < 1 or args.bootstrap < 100 or args.block_days < 1
            or 2 * args.block_days > args.days or args.train_days < 2 * args.cal_days):
        raise ValueError("invalid windows/training/bootstrap (real evaluation requires >=30 days)")
    calendar = EventCalendar.from_yaml(args.calendar)
    series, meta = load_prices(args.prices, args.price_meta, calendar, args.example_smoke)
    first = pd.Timestamp(args.start, tz=series.ts.tz)
    if first != first.normalize():
        raise ValueError("start must be a local date at midnight")
    days = pd.date_range(first, periods=args.days, freq="D")
    started = time.monotonic()
    predictions, daily, audits = [], [], []
    for day in days:
        grid = pd.date_range(day, day + pd.DateOffset(days=1), freq="h", inclusive="left")
        if len(grid) != 24 or len(set(grid.hour)) != 24 or (series.ts.get_indexer(grid) < 0).any():
            raise ValueError(f"evaluation requires complete 24-hour local day: {day}")
        origin = series.ts.get_indexer([day - pd.Timedelta(hours=1)])[0]
        if origin < args.train_days * 24 - 1:
            raise ValueError("insufficient training history before evaluation")
        window, issue = _forecast_window(series, origin, args.train_days)
        train = np.arange(0, issue - 24 + 1, args.stride)
        train = train[train >= 167]  # Exactly the origins accepted by fit(min_origin=167).
        audits.append({"day": day.date().isoformat(), "training": feature_audit(window, train),
                       "evaluation": feature_audit(window, np.asarray([issue]))})
        if any(audits[-1]["training"]["missing"].values()) or any(audits[-1]["evaluation"]["missing"].values()):
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "feature_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
            raise ValueError("event feature gaps; see feature_audit.json; no market-effect conclusion possible")
    for day in days:
        origin = series.ts.get_loc(day - pd.Timedelta(hours=1))
        actual = series.price[origin + 1:origin + 25]
        for name in ("without_events", "with_events"):
            window, issue = _forecast_window(series, origin, args.train_days)
            if name == "without_events":
                window.event_calendar, window.event = None, None
            origins = np.arange(0, issue - 24 + 1, args.stride)
            model = LgbmQuantileForecaster().fit(window, horizon=24, origins=origins,
                num_boost_round=args.rounds, cal_days=args.cal_days, seed=args.seed)
            output = model.predict(window, issue)
            q = np.column_stack([output.p10, output.p50, output.p90])
            daily.append({"day": day.date().isoformat(), "model": name,
                          **summarize(actual, output.p50, quantile_pred=q)})
            for i in range(24):
                predictions.append({"issued_at": series.ts[origin].isoformat(), "ts": output.ts[i].isoformat(),
                    "model": name, "actual": actual[i], "p10": output.p10[i],
                    "p50": output.p50[i], "p90": output.p90[i]})
        print(f"paired day {day.date()} complete", flush=True)
    by_day = pd.DataFrame(daily)
    summary = paired_intervals(by_day, repetitions=args.bootstrap, block_days=args.block_days, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("predictions", pd.DataFrame(predictions)), ("by_day", by_day), ("summary", summary)):
        with (args.output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            if args.example_smoke:
                handle.write("# 示例 / synthetic engineering only; not market-effect evidence\n")
            frame.to_csv(handle, index=False)
    (args.output / "feature_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__).resolve(), *(root / f"app/domain/forecast/{name}.py" for name in (
        "event_data", "event_calendar", "price_features", "price_model")), root / "scripts/run_price_benchmark.py"]
    result = {"status": "synthetic_engineering_only" if args.example_smoke else "real_paired_evaluation",
        "effect_verified": False, "note": "CI conditional on this window and model protocol; no causal/universal claim",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source": meta, "calendar": calendar.summary(), "price_sha256": digest(args.prices),
        "calendar_sha256": digest(args.calendar), "predictions_sha256": digest(args.output / "predictions.csv"),
        "code_sha256": {str(p.relative_to(root)): digest(p) for p in code},
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "lightgbm")},
        "runtime_seconds": time.monotonic() - started, "peak_rss_mib": _peak_rss_mb()}
    (args.output / "run_meta.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prices", "price-meta", "calendar", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--start", required=True)
    for option, default in (("days", 60), ("train-days", 120), ("cal-days", 45), ("rounds", 300),
                            ("stride", 3), ("seed", 42), ("bootstrap", 2000), ("block-days", 7)):
        parser.add_argument(f"--{option}", type=int, default=default)
    parser.add_argument("--example-smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"event benchmark rejected: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
