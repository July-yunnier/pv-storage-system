"""Audit a real or example event calendar without inventing price/model results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.event_calendar import EVENT_FEATURES, EventCalendar  # noqa: E402
from app.domain.forecast.event_data import digest  # noqa: E402


def audit(path: Path, start: str, days: int, output: Path) -> dict:
    calendar = EventCalendar.from_yaml(path)
    if calendar.data is None or days < 1:
        raise ValueError("a V2 calendar and positive audit days are required")
    first = pd.Timestamp(start, tz=calendar.data.timezone)
    if first != first.normalize():
        raise ValueError("start must be a date at local midnight")
    rows = []
    for day in pd.date_range(first, periods=days, freq="D"):
        targets = pd.date_range(day, day + pd.DateOffset(days=1), freq="h", inclusive="left")
        issued_at = day - pd.Timedelta(hours=1)
        values = calendar.features(targets, issued_at=issued_at)
        for i, target in enumerate(targets):
            rows.append({"issued_at": issued_at.isoformat(), "ts": target.isoformat(),
                         **{key: values[key][i] for key in EVENT_FEATURES}})
    frame = pd.DataFrame(rows)
    missing = {key: int((~np.isfinite(frame[key])).sum()) for key in EVENT_FEATURES}
    report = {"status": "incomplete" if any(missing.values()) else "features_complete_only",
        "effect_verified": False, "calendar": calendar.summary(), "rows": len(frame),
        "missing": missing, "calendar_sha256": digest(path),
        "source_hashes": {p.name: sha for p, sha in calendar.data.sources.items()},
        "created_at": datetime.now(timezone.utc).isoformat(), "issue_protocol": "previous local day 23:00"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "hourly_features.csv").open("w", encoding="utf-8", newline="") as handle:
        if calendar.data.data_kind == "example":
            handle.write("# 示例 / synthetic fixture; not market-effect evidence\n")
        frame.to_csv(handle, index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.calendar, args.start, args.days, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"event audit rejected: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "features_complete_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
