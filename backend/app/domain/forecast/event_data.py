"""Versioned, point-in-time event data. No network or price inference here.

Intervals are [start, end); publication times and all other timestamps must be aware.
Coverage is a source assertion of completeness, never inferred from an empty feed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KINDS = ("maintenance", "cold_wave", "heat_wave")
WEIGHTS = {"maintenance": 0.6, "cold_wave": 0.9, "heat_wave": 0.8}


def aware(value, field: str = "timestamp") -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone: {value}")
    return stamp


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class EventData:
    def __init__(self, payload: dict, root: Path):
        if payload.get("schema_version") != 2:
            raise ValueError("event schema_version must be 2")
        self.data_kind = payload.get("data_kind")
        if self.data_kind not in {"real", "example"}:
            raise ValueError("data_kind must be real or example")
        self.market = str(payload.get("market", ""))
        if not self.market:
            raise ValueError("event market is required")
        self.timezone = str(payload.get("timezone", ""))
        ZoneInfo(self.timezone)
        self.name = str(payload.get("name", self.market))
        self.records = []
        versions = set()
        series_ids = set()
        identities = {}
        fuel_observations = {}
        self.sources = {}
        for raw in payload.get("records", []):
            row = dict(raw)
            record_type = row.get("record_type")
            if record_type not in {"event", "coverage", "fuel"}:
                raise ValueError("record_type must be event, coverage or fuel")
            if not row.get("record_id"):
                raise ValueError("record_id is required")
            row["published_at"] = aware(row.get("published_at"), "published_at")
            key = (record_type, row["record_id"], row["published_at"])
            if key in versions:
                raise ValueError(f"duplicate version: {key}")
            versions.add(key)
            url = urlparse(str(row.get("source_url", "")))
            if url.scheme not in {"https", "http"} or not url.netloc:
                raise ValueError("source_url must identify the original HTTP(S) publication")
            rel = Path(row.get("source_file", ""))
            path = (root / rel).resolve()
            if rel.is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("source_file must exist inside the bundle directory")
            if path not in self.sources:
                self.sources[path] = digest(path)
            if row.get("source_sha256") != self.sources[path]:
                raise ValueError(f"source hash mismatch: {rel}")
            if record_type in {"event", "coverage"}:
                if row.get("kind") not in KINDS:
                    raise ValueError("unsupported event kind")
                row["start"] = aware(row.get("start"), "start")
                row["end"] = aware(row.get("end"), "end")
                if row["end"] <= row["start"]:
                    raise ValueError("end must be after start")
                row["status"] = row.get("status") or "active"
                if row["status"] not in {"active", "cancelled"}:
                    raise ValueError("status must be active or cancelled")
                if record_type == "event":
                    row["severity"] = float(row.get("severity", 1))
                    if not math.isfinite(row["severity"]) or not 0 <= row["severity"] <= 1:
                        raise ValueError("severity must be finite and in [0, 1]")
            else:
                for field in ("observation_ts", "valid_until"):
                    row[field] = aware(row.get(field), field)
                if row["observation_ts"] > row["published_at"] or row["valid_until"] <= row["published_at"]:
                    raise ValueError("fuel requires observation_ts <= published_at < valid_until")
                row["index"] = float(row.get("index", np.nan))
                if not math.isfinite(row["index"]) or row["index"] <= 0:
                    raise ValueError("fuel index must be finite and positive")
                if not row.get("series_id"):
                    raise ValueError("fuel series_id is required")
                series_ids.add(row["series_id"])
                observation = (row["series_id"], row["observation_ts"])
                if observation in fuel_observations and fuel_observations[observation] != row["record_id"]:
                    raise ValueError("fuel observation must retain one record_id across revisions")
                fuel_observations[observation] = row["record_id"]
            identity = (row["record_type"], row["record_id"])
            semantics = row.get("kind") if record_type != "fuel" else (row["series_id"], row["observation_ts"])
            if identity in identities and identities[identity] != semantics:
                raise ValueError("revision cannot change the kind or fuel observation identity")
            identities[identity] = semantics
            self.records.append(row)
        # Mixing different heat values/currencies/indices destroys meaningful ratios.
        if len(series_ids) > 1:
            raise ValueError("one consistent fuel series_id is required per calendar")
        self.records.sort(key=lambda row: row["published_at"])

    def snapshot(self, issue: pd.Timestamp) -> list[dict]:
        """Latest known revision per logical id; future corrections do not rewrite history."""
        latest = {}
        for row in self.records:
            if row["published_at"] > issue:
                break
            latest[(row["record_type"], row["record_id"])] = row
        rows = list(latest.values())
        return [row for row in rows if row.get("status", "active") == "active"]

    def features(self, ts: pd.DatetimeIndex, issued_at) -> dict[str, np.ndarray]:
        if ts.tz is None:
            raise ValueError("event target timestamps must include a timezone")
        issue = aware(issued_at, "issued_at")
        rows = self.snapshot(issue)
        targets = ts.tz_convert(self.timezone)
        result = {key: np.full(len(ts), np.nan) for key in (
            "event_pressure", "in_maintenance", "days_to_maintenance", "cold_wave",
            "heat_wave", "fuel_index_norm", "fuel_change_7d",
        )}
        for i, target in enumerate(targets):
            for kind, feature in zip(KINDS, ("in_maintenance", "cold_wave", "heat_wave")):
                known = any(r["record_type"] == "coverage" and r["kind"] == kind
                            and r["start"] <= target < r["end"] for r in rows)
                active = [r for r in rows if r["record_type"] == "event" and r["kind"] == kind
                          and r["start"] <= target < r["end"]]
                # A known positive does not assert that an incomplete feed contains every event.
                if active:
                    result[feature][i] = 1 if kind == "maintenance" else max(r["severity"] for r in active)
                elif known:
                    result[feature][i] = 0
            components = [result[key][i] for key in ("in_maintenance", "cold_wave", "heat_wave")]
            if np.isfinite(components).all():
                result["event_pressure"][i] = min(1.0, sum(WEIGHTS[k] * v for k, v in zip(KINDS, components)))
            if result["in_maintenance"][i] == 1:
                result["days_to_maintenance"][i] = 0
            else:
                starts = [r["start"] for r in rows if r["record_type"] == "event"
                          and r["kind"] == "maintenance" and r["start"] > target]
                if starts:
                    start_day = min(starts).tz_convert(self.timezone).date()
                    result["days_to_maintenance"][i] = min(60, (start_day - target.date()).days)
                elif self._covered_to(rows, "maintenance", target, target + pd.Timedelta(days=60)):
                    result["days_to_maintenance"][i] = 60
        # Fuel is the last value available at the issue time, held only to explicit expiry.
        fuel = {r["observation_ts"]: r for r in sorted(rows, key=lambda r: r["published_at"])
                if r["record_type"] == "fuel" and r["observation_ts"] <= issue}
        fuel = sorted(fuel.values(), key=lambda r: r["observation_ts"])
        if fuel:
            latest = fuel[-1]
            past = [r for r in fuel if r["observation_ts"] >= issue - pd.Timedelta(days=365)]
            week = [r for r in fuel if r["observation_ts"] <= latest["observation_ts"] - pd.Timedelta(days=7)]
            # Weekly releases are permitted; prevent indefinite carry across a missing release.
            base = week[-1] if week else None
            base_ok = base is not None and base["valid_until"] > latest["observation_ts"] - pd.Timedelta(days=7)
            for i, target in enumerate(targets):
                if not issue < latest["valid_until"] or target >= latest["valid_until"]:
                    continue
                if len(past) >= 30:
                    result["fuel_index_norm"][i] = np.mean([r["index"] <= latest["index"] for r in past])
                if base_ok:
                    result["fuel_change_7d"][i] = latest["index"] / base["index"] - 1
        return result

    @staticmethod
    def _covered_to(rows, kind, start, end) -> bool:
        cursor = start
        intervals = sorted((r["start"], r["end"]) for r in rows
                           if r["record_type"] == "coverage" and r["kind"] == kind)
        for left, right in intervals:
            if left > cursor:
                break
            cursor = max(cursor, right)
            if cursor >= end:
                return True
        return False

    def summary(self) -> dict:
        return {"name": self.name, "schema_version": 2, "data_kind": self.data_kind,
                "market": self.market, "timezone": self.timezone,
                "records": len(self.records), "sources": len(self.sources),
                "source_sha256": sorted(set(self.sources.values())),
                "by_type": {kind: sum(r["record_type"] == kind for r in self.records)
                            for kind in ("event", "coverage", "fuel")}}
