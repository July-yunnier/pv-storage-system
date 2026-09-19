from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.models import (
    MeasurementRead,
    MeasurementUpload,
    SiteCreate,
    SitePatch,
    SiteRead,
    TariffCreate,
    TariffRead,
    generate_site_id,
    generate_tariff_id,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _zone(name: str):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8))


def default_tariff() -> TariffCreate:
    return TariffCreate(
        tariff_id="T-DEFAULT",
        name="示例工商业峰谷电价",
        mode="tou",
        transmission_and_fund=0.21,
        tiers=[
            {"name": "sharp", "price": 1.25, "hours": ["17:00-19:00"]},
            {"name": "peak", "price": 0.95, "hours": ["13:00-17:00", "19:00-20:00"]},
            {"name": "flat", "price": 0.65, "hours": ["07:00-13:00", "20:00-23:00"]},
            {"name": "valley", "price": 0.35, "hours": ["23:00-07:00"]},
        ],
    )


class Database:
    def __init__(self, path: str = "data/storage.db") -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()
        if self.get_tariff("T-DEFAULT") is None:
            self.create_tariff(default_tariff())

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sites (
                    site_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tariffs (
                    tariff_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS measurements (
                    site_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    value REAL,
                    unit TEXT NOT NULL,
                    resolution_minutes INTEGER NOT NULL,
                    quality TEXT NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (site_id, data_type, ts),
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                """
            )

    def create_site(self, payload: SiteCreate) -> SiteRead:
        site_id = payload.site_id or generate_site_id()
        now = _now()
        data = payload.model_dump(mode="json")
        data["site_id"] = site_id
        with self.connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO sites(site_id, payload, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (site_id, json.dumps(data, ensure_ascii=False), now.isoformat(), now.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"site already exists: {site_id}") from exc
        return SiteRead.model_validate({**data, "created_at": now, "updated_at": now})

    def list_sites(self) -> list[SiteRead]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM sites ORDER BY created_at DESC").fetchall()
        return [self._site_from_row(row) for row in rows]

    def get_site(self, site_id: str) -> SiteRead | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,)).fetchone()
        return self._site_from_row(row) if row else None

    def update_site(self, site_id: str, patch: SitePatch) -> SiteRead | None:
        existing = self.get_site(site_id)
        if existing is None:
            return None
        data = existing.model_dump(mode="json")
        updates = patch.model_dump(exclude_unset=True)
        data.update(updates)
        updated_at = _now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE sites SET payload = ?, updated_at = ? WHERE site_id = ?",
                (json.dumps(data, ensure_ascii=False), updated_at.isoformat(), site_id),
            )
        return SiteRead.model_validate({**data, "updated_at": updated_at})

    def create_tariff(self, payload: TariffCreate) -> TariffRead:
        tariff_id = payload.tariff_id or generate_tariff_id()
        now = _now()
        data = payload.model_dump(mode="json")
        data["tariff_id"] = tariff_id
        with self.connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO tariffs(tariff_id, payload, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (tariff_id, json.dumps(data, ensure_ascii=False), now.isoformat(), now.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"tariff already exists: {tariff_id}") from exc
        return TariffRead.model_validate({**data, "created_at": now, "updated_at": now})

    def list_tariffs(self) -> list[TariffRead]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM tariffs ORDER BY created_at DESC").fetchall()
        return [self._tariff_from_row(row) for row in rows]

    def get_tariff(self, tariff_id: str) -> TariffRead | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM tariffs WHERE tariff_id = ?", (tariff_id,)).fetchone()
        return self._tariff_from_row(row) if row else None

    def insert_measurements(self, site_id: str, upload: MeasurementUpload) -> int:
        if self.get_site(site_id) is None:
            raise KeyError(f"site not found: {site_id}")
        with self.connection() as conn:
            for point in upload.points:
                value = point.value
                if upload.data_type == "storage_soc" and value is not None and not 0 <= value <= 1:
                    raise ValueError("storage_soc value must be between 0 and 1")
                if upload.data_type != "storage_soc" and value is not None and value < 0:
                    raise ValueError("power measurement cannot be negative")
                conn.execute(
                    """
                    INSERT INTO measurements(
                        site_id, data_type, ts, value, unit, resolution_minutes, quality, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(site_id, data_type, ts) DO UPDATE SET
                        value = excluded.value,
                        unit = excluded.unit,
                        resolution_minutes = excluded.resolution_minutes,
                        quality = excluded.quality,
                        source = excluded.source
                    """,
                    (
                        site_id,
                        upload.data_type,
                        point.ts.isoformat(),
                        value,
                        upload.unit,
                        upload.resolution_minutes,
                        point.quality,
                        upload.source,
                    ),
                )
        return len(upload.points)

    def get_measurements_for_date(
        self, site_id: str, data_type: str, target_date: date, timezone_name: str
    ) -> list[MeasurementRead]:
        tz = _zone(timezone_name)
        start = datetime.combine(target_date, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT data_type, ts, value, unit, quality, source
                FROM measurements
                WHERE site_id = ? AND data_type = ? AND ts >= ? AND ts < ?
                ORDER BY ts
                """,
                (site_id, data_type, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [
            MeasurementRead(
                data_type=row["data_type"],
                ts=_parse_dt(row["ts"]),
                value=row["value"],
                unit=row["unit"],
                quality=row["quality"],
                source=row["source"],
            )
            for row in rows
        ]

    @staticmethod
    def _site_from_row(row: sqlite3.Row) -> SiteRead:
        data = json.loads(row["payload"])
        return SiteRead.model_validate(
            {
                **data,
                "created_at": _parse_dt(row["created_at"]),
                "updated_at": _parse_dt(row["updated_at"]),
            }
        )

    @staticmethod
    def _tariff_from_row(row: sqlite3.Row) -> TariffRead:
        data = json.loads(row["payload"])
        return TariffRead.model_validate(
            {
                **data,
                "created_at": _parse_dt(row["created_at"]),
                "updated_at": _parse_dt(row["updated_at"]),
            }
        )
