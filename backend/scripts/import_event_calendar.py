"""Periodically import reviewed CSV exports and immutable source evidence.

No scraping and no inferred publication times. All input timestamps must include zones.
See docs/19-price-event-data.md for the CSV contract and scheduled invocation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.event_data import EventData, digest  # noqa: E402


def import_records(records: Path, source_root: Path, output: Path, *, market: str,
                   timezone_name: str, data_kind: str) -> dict:
    source_root = source_root.resolve()
    payload = {"schema_version": 2, "name": f"{market}-{data_kind}", "market": market,
               "timezone": timezone_name, "data_kind": data_kind, "records": []}
    if output.exists():
        payload = yaml.safe_load(output.read_text(encoding="utf-8"))
        EventData(payload, output.parent)  # Fail before changing an existing invalid bundle.
        if (payload["market"], payload["timezone"], payload["data_kind"]) != (market, timezone_name, data_kind):
            raise ValueError("cannot change the market/timezone/data_kind of an existing bundle")
    with records.open(encoding="utf-8-sig", newline="") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        additions = [{k: v.strip() for k, v in row.items() if k and v and v.strip()}
                     for row in csv.DictReader(lines)]
    if not additions:
        raise ValueError("empty event export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="event-import-", dir=output.parent) as temp:
        stage = Path(temp)
        # Copy previously verified evidence so validation is independent of the live bundle.
        for row in payload["records"]:
            dest = stage / row["source_file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output.parent / row["source_file"], dest)
        for row in additions:
            rel = Path(row.get("source_file", ""))
            source = (source_root / rel).resolve()
            if rel.is_absolute() or not source.is_relative_to(source_root) or not source.is_file():
                raise ValueError("source_file must exist inside --source-root")
            sha = digest(source)
            if row.get("source_sha256") and row["source_sha256"] != sha:
                raise ValueError("input source hash mismatch")
            row["source_sha256"] = sha
            row["source_file"] = f"sources/{sha}{source.suffix.lower()}"
            dest = stage / row["source_file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        # Idempotent append; changing the same published revision is forbidden.
        versions = {(r["record_type"], r["record_id"], r["published_at"]): r for r in payload["records"]}
        from app.domain.forecast.event_data import aware
        versions = {(a, b, aware(c)): r for (a, b, c), r in versions.items()}
        for row in additions:
            key = (row["record_type"], row["record_id"], aware(row["published_at"]))
            if key in versions:
                # Numeric input CSV and YAML must compare semantically, not by serialization.
                candidate = {**payload, "records": [row]}
                before = {**payload, "records": [versions[key]]}
                left = EventData(candidate, stage).records[0]
                right = EventData(before, stage).records[0]
                if left != right:
                    raise ValueError("conflicting immutable revision; publish a new revision instead")
            else:
                versions[key] = row
        payload["records"] = list(versions.values())
        payload["imported_at"] = datetime.now(timezone.utc).isoformat()
        data = EventData(payload, stage)
        # Publish immutable sources first, then atomically switch the calendar.
        for source in (stage / "sources").iterdir():
            dest = output.parent / "sources" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and digest(dest) != digest(source):
                raise ValueError("immutable evidence archive was modified")
            if not dest.exists():
                shutil.copyfile(source, dest)
        staged_yaml = stage / "calendar.yaml"
        text = "# 示例 / synthetic fixture; not market evidence\n" if data_kind == "example" else ""
        staged_yaml.write_text(text + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        os.replace(staged_yaml, output)
    return {**data.summary(), "calendar_sha256": digest(output),
            "imported_at": payload["imported_at"], "input_sha256": digest(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--data-kind", choices=("real", "example"), required=True)
    args = parser.parse_args()
    try:
        report = import_records(args.records, args.source_root, args.output,
                                market=args.market, timezone_name=args.timezone, data_kind=args.data_kind)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"event import rejected: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
