"""Synthetic fixtures: causality/engineering checks, never market-effect evidence."""
from __future__ import annotations
import csv
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from app.domain.forecast.event_calendar import EVENT_FEATURES, EventCalendar, load_calendar
from app.domain.forecast.event_data import digest
from app.domain.forecast.price_features import FEATURE_COLUMNS, PriceSeries, build_samples
from scripts.import_event_calendar import import_records
from scripts.run_event_ablation import load_prices, paired_intervals, run


def bundle(root, rows=()):
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "evidence.txt"
    raw.write_text("示例 / synthetic fixture, not a real publication\n", encoding="utf-8")
    payload = {"schema_version": 2, "data_kind": "example", "market": "guangdong",
               "timezone": "Asia/Shanghai", "records": []}
    for row in rows:
        payload["records"].append({"source_file": "evidence.txt", "source_url": "https://example.invalid/fixture",
                                   "source_sha256": digest(raw), **row})
    path = root / "calendar.yaml"
    path.write_text("# 示例 / synthetic fixture\n" + yaml.safe_dump(payload), encoding="utf-8")
    return path


def event(**kwargs):
    return {"record_type": "event", "record_id": "unit-1", "kind": "maintenance",
        "start": "2025-03-15T00:00:00+08:00", "end": "2025-03-17T00:00:00+08:00",
        "published_at": "2025-03-14T12:00:00+08:00", "severity": .6, **kwargs}


def coverage(kind="maintenance", **kwargs):
    return {"record_type": "coverage", "record_id": f"complete-{kind}", "kind": kind,
        "start": "2025-01-01T00:00:00+08:00", "end": "2026-01-01T00:00:00+08:00",
        "published_at": "2024-12-01T00:00:00+08:00", **kwargs}


def features(calendar, issue="2025-03-14T23:00:00+08:00"):
    return calendar.features(pd.date_range("2025-03-15", periods=24, freq="h", tz="Asia/Shanghai"), issued_at=issue)


def test_publication_boundary_cancellation_and_replay(tmp_path):
    calendar = EventCalendar.from_yaml(bundle(tmp_path, [coverage(), event(),
        event(published_at="2025-03-15T12:00:00+08:00", status="cancelled")]))
    assert features(calendar, "2025-03-14T11:59:59+08:00")["in_maintenance"][0] == 0
    assert features(calendar, "2025-03-14T12:00:00+08:00")["in_maintenance"][0] == 1
    assert features(calendar, "2025-03-15T12:00:00+08:00")["in_maintenance"][0] == 0
    assert features(calendar)["in_maintenance"][0] == 1


def test_missing_feed_is_unknown_not_zero(tmp_path):
    values = features(EventCalendar.from_yaml(bundle(tmp_path, [coverage("heat_wave")])))
    assert (values["heat_wave"] == 0).all()
    for key in ("cold_wave", "event_pressure", "days_to_maintenance"):
        assert np.isnan(values[key]).all()


def test_incomplete_maintenance_horizon_is_not_imputed(tmp_path):
    values = features(EventCalendar.from_yaml(bundle(tmp_path, [coverage(end="2025-03-16T00:00:00+08:00")])))
    assert values["in_maintenance"][0] == 0
    assert np.isnan(values["days_to_maintenance"][0])


def test_maintenance_distance_uses_market_local_date(tmp_path):
    path = bundle(tmp_path, [event(start="2025-03-15T16:00:00+00:00")])
    values = features(EventCalendar.from_yaml(path))
    assert values["days_to_maintenance"][0] == 1  # March 16 in Guangdong.


@pytest.mark.parametrize("field,value", [("published_at", "2025-03-14T12:00:00"),
    ("start", "2025-03-15"), ("severity", "nan"), ("severity", 1.1),
    ("end", "2025-03-14T00:00:00+08:00"), ("source_url", ""), ("source_sha256", "bad")])
def test_invalid_records_rejected(tmp_path, field, value):
    with pytest.raises(ValueError):
        EventCalendar.from_yaml(bundle(tmp_path, [event(**{field: value})]))


def test_source_hash_and_target_timezone_required(tmp_path):
    path = bundle(tmp_path, [event()])
    calendar = EventCalendar.from_yaml(path)
    with pytest.raises(ValueError, match="timezone"):
        calendar.features(pd.date_range("2025-03-15", periods=1), issued_at="2025-03-14T23:00:00+08:00")
    with pytest.raises(ValueError, match="timezone"):
        calendar.features(pd.date_range("2025-03-15", periods=1, tz="Asia/Shanghai"))
    (tmp_path / "evidence.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calendar(path)


def fuel_rows():
    return [{"record_type": "fuel", "record_id": f"fuel-{i}", "series_id": "synthetic-index",
        "observation_ts": d.isoformat(), "published_at": (d + pd.Timedelta(hours=12)).isoformat(),
        "valid_until": (d + pd.Timedelta(days=8)).isoformat(), "index": 100 + i}
        for i, d in enumerate(pd.date_range("2025-01-01", periods=130, tz="Asia/Shanghai"))]


def test_weekly_fuel_uses_seven_calendar_days_not_eighth_release(tmp_path):
    calendar = EventCalendar.from_yaml(bundle(tmp_path, fuel_rows()[::7]))
    values = calendar.features(pd.date_range("2025-03-06", periods=24, freq="h", tz="Asia/Shanghai"),
                               issued_at="2025-03-05T23:00:00+08:00")
    assert values["fuel_change_7d"][0] == pytest.approx(163 / 156 - 1)
    assert np.isnan(values["fuel_index_norm"]).all()
    expired = calendar.features(pd.date_range("2025-07-01", periods=1, tz="Asia/Shanghai"),
                                issued_at="2025-06-30T23:00:00+08:00")
    assert np.isnan(expired["fuel_change_7d"][0])


def test_future_release_revision_and_expiry_cannot_leak(tmp_path):
    rows = fuel_rows()[:65]
    revised = {**rows[-1], "published_at": "2025-03-07T12:00:00+08:00", "index": 500}
    calendar = EventCalendar.from_yaml(bundle(tmp_path, [*rows, revised]))
    values = calendar.features(pd.date_range("2025-03-07", periods=24, freq="h", tz="Asia/Shanghai"),
                               issued_at="2025-03-06T23:00:00+08:00")
    assert np.allclose(values["fuel_change_7d"], 164 / 157 - 1)
    assert (values["fuel_index_norm"] == 1).all()


def test_revision_identity_and_fuel_series_mixing_rejected(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        EventCalendar.from_yaml(bundle(tmp_path, [event(),
            event(kind="heat_wave", published_at="2025-03-15T12:00:00+08:00")]))
    rows = fuel_rows()[:2]
    rows[1]["series_id"] = "other-index"
    with pytest.raises(ValueError, match="series_id"):
        EventCalendar.from_yaml(bundle(tmp_path, rows))


def test_training_features_use_each_origin_and_keep_column_contract(tmp_path):
    calendar = EventCalendar.from_yaml(bundle(tmp_path, [coverage(), event()]))
    ts = pd.date_range("2025-03-14T11:00:00+08:00", periods=48, freq="h")
    series = PriceSeries(ts=ts, price=np.arange(len(ts), dtype=float), event_calendar=calendar)
    x, _, targets, origins = build_samples(series, origins=np.asarray([0, 1]), horizon=24, min_origin=0)
    assert targets.tz is not None and origins.tz is not None
    assert sorted(x.loc[targets == pd.Timestamp("2025-03-15T00:00:00+08:00"), "in_maintenance"]) == [0, 1]
    assert tuple(x.columns) == FEATURE_COLUMNS
    assert FEATURE_COLUMNS[-7:] == EVENT_FEATURES
    series.event_available_at = ts[0]
    capped, _, _, _ = build_samples(series, origins=np.asarray([1]), horizon=24, min_origin=0)
    assert (capped["in_maintenance"] == 0).all()


def test_multiday_service_preserves_calendar_and_original_cutoff(tmp_path):
    from app.application.price_service import _recursive_day_ahead
    from app.domain.forecast.price_model import PriceForecastOutput
    calendar = EventCalendar.from_yaml(bundle(tmp_path, [coverage(), event(published_at="2025-03-14T13:00:00+08:00")]))
    ts = pd.date_range("2025-03-14T12:00:00+08:00", periods=49, freq="h")
    series = PriceSeries(ts=ts, price=np.zeros(len(ts)), event_calendar=calendar)
    seen = []
    class Forecaster:
        def predict(self, working, origin, horizon):
            x, _, target, _ = build_samples(working, origins=np.asarray([origin]), horizon=24, min_origin=0)
            seen.append(x["in_maintenance"].to_numpy())
            return PriceForecastOutput(target, np.zeros(24), np.zeros(24), np.zeros(24), "test", "v1")
    output = _recursive_day_ahead(Forecaster(), series, 0, 2)
    assert output.ts.tz is not None
    assert len(seen) == 2
    assert np.all(np.concatenate(seen) == 0)


def test_demo_is_not_implicitly_enabled(monkeypatch):
    from app.application.price_service import event_calendar_path
    monkeypatch.delenv("PRICE_EVENT_CALENDAR", raising=False)
    assert event_calendar_path() is None


def write_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# 示例 / synthetic fixture\n")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_periodic_import_idempotency_refresh_and_conflict_rollback(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notice.txt").write_text("示例公告", encoding="utf-8")
    row = {**event(), "source_file": "notice.txt", "source_url": "https://example.invalid/fixture"}
    path, output = tmp_path / "input.csv", tmp_path / "output/calendar.yaml"
    write_csv(path, [row])
    kwargs = {"market": "guangdong", "timezone_name": "Asia/Shanghai", "data_kind": "example"}
    assert import_records(path, raw, output, **kwargs)["records"] == 1
    assert import_records(path, raw, output, **kwargs)["records"] == 1
    assert load_calendar(output).summary()["records"] == 1
    write_csv(path, [row, {**row, "published_at": "2025-03-15T12:00:00+08:00", "status": "cancelled"}])
    assert import_records(path, raw, output, **kwargs)["records"] == 2
    assert load_calendar(output).summary()["records"] == 2
    before = output.read_bytes()
    write_csv(path, [{**row, "severity": .9}])
    with pytest.raises(ValueError, match="immutable revision"):
        import_records(path, raw, output, **kwargs)
    assert output.read_bytes() == before


def test_paired_day_block_bootstrap_and_pair_validation():
    rows = [{"day": str(i), "model": m, **{k: i + off for k in ("mae", "pinball", "picp80", "winkler80")}}
            for i in range(20) for m, off in (("with_events", 1), ("without_events", 2))]
    result = paired_intervals(pd.DataFrame(rows), repetitions=100, block_days=3)
    assert (result[["difference_with_minus_without", "ci95_low", "ci95_high"]] == -1).all().all()
    pd.testing.assert_frame_equal(result, paired_intervals(pd.DataFrame(rows), repetitions=100, block_days=3))
    with pytest.raises(ValueError, match="paired"):
        paired_intervals(pd.DataFrame(rows[:-1]), repetitions=100, block_days=3)
    with pytest.raises(ValueError, match="paired"):
        paired_intervals(pd.DataFrame(rows), repetitions=100, block_days=20)


def test_variable_paired_bootstrap_has_uncertainty_and_is_order_independent():
    rows = [{"day": f"2025-03-{i + 1:02d}", "model": model,
             **{key: float(i % 5) if model == "with_events" else 2.0
                for key in ("mae", "pinball", "picp80", "winkler80")}}
            for i in range(20) for model in ("with_events", "without_events")]
    frame = pd.DataFrame(rows)
    result = paired_intervals(frame, repetitions=1000, block_days=3)
    assert (result.ci95_low < result.difference_with_minus_without).all()
    assert (result.ci95_high > result.difference_with_minus_without).all()
    pd.testing.assert_frame_equal(result, paired_intervals(
        frame.sample(frac=1, random_state=8), repetitions=1000, block_days=3))


def test_source_traversal_and_duplicate_fuel_observations_rejected(tmp_path):
    with pytest.raises(ValueError, match="inside the bundle"):
        EventCalendar.from_yaml(bundle(tmp_path, [event(source_file="../outside.txt")]))
    rows = fuel_rows()[:1]
    with pytest.raises(ValueError, match="one record_id"):
        EventCalendar.from_yaml(bundle(tmp_path, [*rows, {**rows[0], "record_id": "duplicate"}]))


def smoke_inputs(root):
    path = bundle(root, [*(coverage(k) for k in ("maintenance", "cold_wave", "heat_wave")), event(), *fuel_rows()])
    ts = pd.date_range("2025-01-01", periods=100 * 24, freq="h", tz="Asia/Shanghai")
    prices = root / "prices.csv"
    values = .5 + .1 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
    values[0] = -.1
    pd.DataFrame({"ts": ts, "price": values}).to_csv(prices, index=False)
    prices.write_text("# 示例 / synthetic fixture\n" + prices.read_text(), encoding="utf-8")
    meta = {"data_kind": "example", "market": "guangdong", "timezone": "Asia/Shanghai",
        "unit": "CNY/kWh", "price_kind": "spot", "price_sha256": digest(prices),
        "source_file": "evidence.txt", "source_url": "https://example.invalid/fixture",
        "source_sha256": digest(root / "evidence.txt")}
    meta_path = root / "price_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return SimpleNamespace(prices=prices, price_meta=meta_path, calendar=path, output=root / "results",
        start="2025-03-15", days=4, train_days=30, cal_days=0, rounds=3, stride=24, seed=42,
        bootstrap=100, block_days=2, example_smoke=True)


def test_benchmark_rejects_demo_tou_cross_market_naive_and_missing_prices(tmp_path):
    args = smoke_inputs(tmp_path)
    calendar = EventCalendar.from_yaml(args.calendar)
    with pytest.raises(ValueError, match="real data"):
        load_prices(args.prices, args.price_meta, calendar, False)
    original = json.loads(args.price_meta.read_text())
    for key, value, match in (("market", "pjm", "market"), ("price_kind", "tou", "TOU")):
        args.price_meta.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ValueError, match=match):
            load_prices(args.prices, args.price_meta, calendar, True)
    args.price_meta.write_text(json.dumps(original))
    frame = pd.read_csv(args.prices, comment="#")
    for bad in (frame.drop(index=[3]), frame.assign(ts=frame["ts"].str[:19])):
        bad.to_csv(args.prices, index=False)
        args.price_meta.write_text(json.dumps({**original, "price_sha256": digest(args.prices)}))
        with pytest.raises(ValueError):
            load_prices(args.prices, args.price_meta, calendar, True)


def test_feature_gap_audit_rejects_before_training(tmp_path, monkeypatch):
    args = smoke_inputs(tmp_path)
    args.calendar = bundle(tmp_path, [event()])
    def unexpected(*args, **kwargs):
        pytest.fail("missing data must be rejected before fitting")
    monkeypatch.setattr("scripts.run_event_ablation.LgbmQuantileForecaster.fit", unexpected)
    with pytest.raises(ValueError, match="feature gaps"):
        run(args)
    audit = json.loads((args.output / "feature_audit.json").read_text())
    assert audit[0]["evaluation"]["missing"]["fuel_index_norm"] == 24
    assert not (args.output / "summary.csv").exists()


def test_preflight_audits_exact_fit_origins_and_preserves_previous_outputs(tmp_path, monkeypatch):
    args = smoke_inputs(tmp_path)
    args.stride = 3
    seen = []
    def audit(series, origins, horizon=24):
        seen.append(origins)
        return {"rows": len(origins) * horizon, "missing": {"cold_wave": 1}}
    monkeypatch.setattr("scripts.run_event_ablation.feature_audit", audit)
    with pytest.raises(ValueError, match="feature gaps"):
        run(args)
    assert np.array_equal(seen[0], np.arange(168, 696, 3))
    before = (args.output / "feature_audit.json").read_bytes()
    with pytest.raises(ValueError, match="must be empty"):
        run(args)
    assert (args.output / "feature_audit.json").read_bytes() == before


def test_calendar_audit_keeps_unknowns_and_aware_times(tmp_path):
    from scripts.audit_event_calendar import audit
    path = bundle(tmp_path, [event()])
    result = audit(path, "2025-03-15", 2, tmp_path / "audit")
    assert result["status"] == "incomplete" and result["effect_verified"] is False
    assert result["rows"] == 48
    assert result["missing"]["in_maintenance"] == 0
    assert result["missing"]["heat_wave"] == 48
    assert pd.read_csv(tmp_path / "audit/hourly_features.csv", comment="#")["ts"].str.endswith("+08:00").all()


def test_synthetic_rolling_path_emits_aware_paired_predictions_and_ci(tmp_path):
    pytest.importorskip("lightgbm")
    args = smoke_inputs(tmp_path)
    result = run(args)
    assert result["status"] == "synthetic_engineering_only"
    assert result["effect_verified"] is False
    predictions = pd.read_csv(args.output / "predictions.csv", comment="#")
    assert len(predictions) == 4 * 24 * 2
    assert predictions["ts"].str.endswith("+08:00").all()
    assert predictions["issued_at"].str.endswith("+08:00").all()
    assert ((predictions.p10 <= predictions.p50) & (predictions.p50 <= predictions.p90)).all()
    assert len(pd.read_csv(args.output / "summary.csv", comment="#")) == 4
    assert (args.output / "predictions.csv").read_text().startswith("# 示例")
    assert all(not any(row["evaluation"]["missing"].values())
               for row in json.loads((args.output / "feature_audit.json").read_text()))
