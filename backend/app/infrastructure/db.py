from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domain.models import (
    DispatchRunDetail,
    DispatchRunSummary,
    MeasurementRead,
    MeasurementUpload,
    ReportJob,
    ScheduleResponse,
    SiteCreate,
    SitePatch,
    SiteRead,
    TariffCreate,
    TariffRead,
    generate_site_id,
    generate_tariff_id,
)
from app.domain.sizing.result import SizingResponse, SizingRunDetail, SizingRunSummary


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


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
        self._memory_conn: sqlite3.Connection | None = None
        if path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()
        if self.get_tariff("T-DEFAULT") is None:
            self.create_tariff(default_tariff())

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            self._memory_conn.commit()
            return
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
                    name TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    tariff_id TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tariffs (
                    tariff_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    transmission_and_fund REAL NOT NULL,
                    tiers_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS measurements (
                    site_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    value REAL,
                    unit TEXT NOT NULL,
                    resolution_minutes INTEGER NOT NULL,
                    quality TEXT NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (site_id, data_type, ts_utc),
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE TABLE IF NOT EXISTS rolling_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    base_run_id TEXT NOT NULL,
                    initial_soc REAL NOT NULL,
                    schedule_json TEXT NOT NULL,
                    expected_benefit_yuan REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auto_plan_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    plan_date TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dispatch_runs (
                    run_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    initial_soc REAL NOT NULL,
                    input_digest TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE TABLE IF NOT EXISTS sizing_runs (
                    run_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    input_digest TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE TABLE IF NOT EXISTS report_jobs (
                    report_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    sizing_run_id TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    file_path TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE TABLE IF NOT EXISTS replay_runs (
                    run_id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE TABLE IF NOT EXISTS replay_day_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    pv_method TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES replay_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_replay_day_results_run
                    ON replay_day_results(run_id);
                CREATE TABLE IF NOT EXISTS forecast_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    method TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    points_json TEXT NOT NULL,
                    provenance_json TEXT,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
                CREATE INDEX IF NOT EXISTS idx_forecast_archive_lookup
                    ON forecast_archive(site_id, target, target_date);
                CREATE TABLE IF NOT EXISTS forecast_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    method TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    UNIQUE(site_id, target, method, target_date)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_config (
                    site_id TEXT PRIMARY KEY,
                    shift_days INTEGER NOT NULL DEFAULT 0,
                    cursor_date TEXT,
                    cursor_for TEXT,
                    window_start TEXT,
                    window_end TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self._migrate_dispatch_runs()
        self._migrate_replay_config()

    def _migrate_replay_config(self) -> None:
        """回放配置补列：live 指针（cursor_date/cursor_for）与窗口边界。"""
        with self.connection() as conn:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(replay_config)").fetchall()
            }
            for col in ("cursor_date", "cursor_for", "window_start", "window_end"):
                if col not in columns:
                    conn.execute(f"ALTER TABLE replay_config ADD COLUMN {col} TEXT")

    def _migrate_dispatch_runs(self) -> None:
        """I5 起 dispatch_runs 增加 data_source 列；存量库原地补列。"""
        with self.connection() as conn:
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(dispatch_runs)").fetchall()
            }
            if "data_source" not in columns:
                conn.execute(
                    "ALTER TABLE dispatch_runs "
                    "ADD COLUMN data_source TEXT NOT NULL DEFAULT 'measurements'"
                )

    def create_site(self, payload: SiteCreate) -> SiteRead:
        site_id = payload.site_id or generate_site_id()
        now = _now()
        config = {
            "location": {
                "province": payload.location.province,
                "city": payload.location.city,
            },
            "pv": payload.pv.model_dump(mode="json"),
            "load": payload.load.model_dump(mode="json"),
            "storage": payload.storage.model_dump(mode="json"),
            "demand_limit_kw": payload.demand_limit_kw,
        }
        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO sites(site_id, name, timezone, lat, lon, tariff_id, config_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        site_id,
                        payload.name,
                        payload.location.timezone,
                        payload.location.lat,
                        payload.location.lon,
                        payload.tariff_id,
                        json.dumps(config, ensure_ascii=False),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"site already exists: {site_id}") from exc
        site = self.get_site(site_id)
        assert site is not None
        return site

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
        updates = patch.model_dump(exclude_unset=True)
        if not updates:
            return existing
        name = updates.get("name", existing.name)
        tariff_id = updates.get("tariff_id", existing.tariff_id)
        updated_at = _now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE sites SET name = ?, tariff_id = ?, updated_at = ? WHERE site_id = ?",
                (name, tariff_id, updated_at.isoformat(), site_id),
            )
        return self.get_site(site_id)

    def create_tariff(self, payload: TariffCreate) -> TariffRead:
        tariff_id = payload.tariff_id or generate_tariff_id()
        now = _now()
        tiers = [tier.model_dump(mode="json") for tier in payload.tiers]
        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tariffs(tariff_id, name, mode, transmission_and_fund, tiers_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tariff_id,
                        payload.name,
                        payload.mode,
                        payload.transmission_and_fund,
                        json.dumps(tiers, ensure_ascii=False),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"tariff already exists: {tariff_id}") from exc
        tariff = self.get_tariff(tariff_id)
        assert tariff is not None
        return tariff

    def list_tariffs(self) -> list[TariffRead]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM tariffs ORDER BY created_at DESC").fetchall()
        return [self._tariff_from_row(row) for row in rows]

    def get_tariff(self, tariff_id: str) -> TariffRead | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM tariffs WHERE tariff_id = ?", (tariff_id,)
            ).fetchone()
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
                        site_id, data_type, ts_utc, value, unit, resolution_minutes, quality, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(site_id, data_type, ts_utc) DO UPDATE SET
                        value = excluded.value,
                        unit = excluded.unit,
                        resolution_minutes = excluded.resolution_minutes,
                        quality = excluded.quality,
                        source = excluded.source
                    """,
                    (
                        site_id,
                        upload.data_type,
                        _to_utc(point.ts).isoformat(),
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
        start = _to_utc(datetime.combine(target_date, time.min, tzinfo=tz))
        end = start + timedelta(days=1)
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT data_type, ts_utc, value, unit, quality, source
                FROM measurements
                WHERE site_id = ? AND data_type = ? AND ts_utc >= ? AND ts_utc < ?
                ORDER BY ts_utc
                """,
                (site_id, data_type, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [
            MeasurementRead(
                data_type=row["data_type"],
                ts=_parse_dt(row["ts_utc"]).astimezone(tz),
                value=row["value"],
                unit=row["unit"],
                quality=row["quality"],
                source=row["source"],
            )
            for row in rows
        ]

    def get_measurement_rows_for_date(
        self, site_id: str, data_type: str, target_date: date, timezone_name: str
    ) -> list[dict]:
        """与 get_measurements_for_date 同时区口径，但额外返回 resolution_minutes。"""
        tz = _zone(timezone_name)
        start = _to_utc(datetime.combine(target_date, time.min, tzinfo=tz))
        end = start + timedelta(days=1)
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT ts_utc, value, quality, resolution_minutes
                FROM measurements
                WHERE site_id = ? AND data_type = ? AND ts_utc >= ? AND ts_utc < ?
                ORDER BY ts_utc
                """,
                (site_id, data_type, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [
            {
                "ts": _parse_dt(row["ts_utc"]).astimezone(tz),
                "value": row["value"],
                "quality": row["quality"],
                "resolution_minutes": row["resolution_minutes"],
            }
            for row in rows
        ]

    def get_latest_measurement_point(
        self, site_id: str, data_type: str, timezone_name: str
    ) -> tuple[datetime, float] | None:
        """该类型最新的一个非 None 点（ts 转站点时区）。"""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT ts_utc, value FROM measurements
                WHERE site_id = ? AND data_type = ? AND value IS NOT NULL
                ORDER BY ts_utc DESC LIMIT 1
                """,
                (site_id, data_type),
            ).fetchone()
        if row is None:
            return None
        return _parse_dt(row["ts_utc"]).astimezone(_zone(timezone_name)), row["value"]

    def get_latest_measurement_with_source(
        self, site_id: str, data_type: str
    ) -> tuple[datetime, float, str] | None:
        """最新非 None 点（UTC 时间戳、值、source），供模拟源自动续传判定。"""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT ts_utc, value, source FROM measurements
                WHERE site_id = ? AND data_type = ? AND value IS NOT NULL
                ORDER BY ts_utc DESC LIMIT 1
                """,
                (site_id, data_type),
            ).fetchone()
        if row is None:
            return None
        return _parse_dt(row["ts_utc"]), row["value"], row["source"]

    # ---- 回放映射：public_replay 站点的历史平移 + live 回放指针 ----

    def set_replay_config(
        self,
        site_id: str,
        shift_days: int,
        window_start: str,
        window_end: str,
        cursor_date: str | None = None,
        cursor_for: str | None = None,
    ) -> None:
        """写入回放配置。cursor_date：live 下一次重播的本地日；cursor_for：该指针服务过的本地日期。"""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO replay_config
                    (site_id, shift_days, window_start, window_end, cursor_date, cursor_for, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id) DO UPDATE SET
                    shift_days = excluded.shift_days,
                    window_start = excluded.window_start,
                    window_end = excluded.window_end,
                    cursor_date = excluded.cursor_date,
                    cursor_for = excluded.cursor_for,
                    updated_at = excluded.updated_at
                """,
                (site_id, shift_days, window_start, window_end,
                 cursor_date, cursor_for, _now()),
            )

    def get_replay_config(self, site_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM replay_config WHERE site_id = ?", (site_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def update_replay_cursor(self, site_id: str, cursor_date: str, cursor_for: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE replay_config SET cursor_date = ?, cursor_for = ?, updated_at = ? "
                "WHERE site_id = ?",
                (cursor_date, cursor_for, _now(), site_id),
            )

    def get_latest_measurement_ts(
        self, site_id: str, data_types: list[str], timezone_name: str
    ) -> datetime | None:
        """若干类型中最新的非 None 点时间戳（ts 转站点时区）。"""
        placeholders = ", ".join("?" for _ in data_types)
        with self.connection() as conn:
            row = conn.execute(
                f"""
                SELECT MAX(ts_utc) AS latest FROM measurements
                WHERE site_id = ? AND data_type IN ({placeholders}) AND value IS NOT NULL
                """,
                (site_id, *data_types),
            ).fetchone()
        if row is None or row["latest"] is None:
            return None
        return _parse_dt(row["latest"]).astimezone(_zone(timezone_name))

    def get_measurement_first_date(self, site_id: str, data_type: str) -> date | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT MIN(ts_utc) AS earliest FROM measurements
                WHERE site_id = ? AND data_type = ? AND value IS NOT NULL
                """,
                (site_id, data_type),
            ).fetchone()
        if row is None or row["earliest"] is None:
            return None
        return _parse_dt(row["earliest"]).astimezone(timezone.utc).date()

    def get_measurement_last_date(self, site_id: str, data_type: str) -> date | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(ts_utc) AS latest FROM measurements
                WHERE site_id = ? AND data_type = ? AND value IS NOT NULL
                """,
                (site_id, data_type),
            ).fetchone()
        if row is None or row["latest"] is None:
            return None
        return _parse_dt(row["latest"]).astimezone(timezone.utc).date()

    def insert_replay_run(self, run_id: str, site_id: str, config: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO replay_runs(run_id, site_id, config_json, status, created_at)
                VALUES (?, ?, ?, 'running', ?)
                """,
                (run_id, site_id, json.dumps(config, ensure_ascii=False), _now().isoformat()),
            )

    def finish_replay_run(self, run_id: str, summary: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE replay_runs SET status = 'finished', summary_json = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (json.dumps(summary, ensure_ascii=False), _now().isoformat(), run_id),
            )

    def fail_replay_run(self, run_id: str, error: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE replay_runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (f"failed: {error[:200]}", _now().isoformat(), run_id),
            )

    def insert_replay_day_result(
        self, run_id: str, day: date, pv_method: str, strategy: str, metrics: dict
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO replay_day_results(run_id, date, pv_method, strategy, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, day.isoformat(), pv_method, strategy,
                 json.dumps(metrics, ensure_ascii=False)),
            )

    def list_replay_runs(self, site_id: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT run_id, site_id, config_json, status, summary_json, created_at, finished_at
                FROM replay_runs WHERE site_id = ? ORDER BY created_at DESC
                """,
                (site_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "site_id": row["site_id"],
                "config": json.loads(row["config_json"]),
                "status": row["status"],
                "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
                "created_at": _parse_dt(row["created_at"]),
                "finished_at": _parse_dt(row["finished_at"]) if row["finished_at"] else None,
            }
            for row in rows
        ]

    def get_replay_run(self, run_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, site_id, config_json, status, summary_json, created_at, finished_at
                FROM replay_runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "site_id": row["site_id"],
            "config": json.loads(row["config_json"]),
            "status": row["status"],
            "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
            "created_at": _parse_dt(row["created_at"]),
            "finished_at": _parse_dt(row["finished_at"]) if row["finished_at"] else None,
        }

    def get_replay_day_results(self, run_id: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT date, pv_method, strategy, metrics_json
                FROM replay_day_results WHERE run_id = ? ORDER BY date, pv_method, strategy
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "date": row["date"],
                "pv_method": row["pv_method"],
                "strategy": row["strategy"],
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in rows
        ]

    def insert_forecast_archive(
        self,
        *,
        site_id: str,
        target: str,
        method: str,
        target_date: date,
        issued_at: datetime,
        points: list[dict],
        provenance: dict | None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO forecast_archive(
                    site_id, target, method, target_date, issued_at, points_json, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    site_id, target, method, target_date.isoformat(), issued_at.isoformat(),
                    json.dumps(points, ensure_ascii=False),
                    json.dumps(provenance, ensure_ascii=False) if provenance else None,
                ),
            )

    def get_unreconciled_forecasts(self, site_id: str, up_to_date: date) -> list[dict]:
        """已有实测覆盖、但尚未对账的封存预测。"""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT fa.id, fa.target, fa.method, fa.target_date, fa.points_json
                FROM forecast_archive fa
                WHERE fa.site_id = ? AND fa.target_date <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM forecast_accuracy ac
                      WHERE ac.site_id = fa.site_id AND ac.target = fa.target
                        AND ac.method = fa.method AND ac.target_date = fa.target_date
                  )
                ORDER BY fa.target_date
                """,
                (site_id, up_to_date.isoformat()),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "target": row["target"],
                "method": row["method"],
                "target_date": row["target_date"],
                "points": json.loads(row["points_json"]),
            }
            for row in rows
        ]

    def upsert_forecast_accuracy(
        self,
        *,
        site_id: str,
        target: str,
        method: str,
        target_date: date,
        metrics: dict,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO forecast_accuracy(site_id, target, method, target_date, metrics_json, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id, target, method, target_date) DO UPDATE SET
                    metrics_json = excluded.metrics_json,
                    computed_at = excluded.computed_at
                """,
                (site_id, target, method, target_date.isoformat(),
                 json.dumps(metrics, ensure_ascii=False), _now().isoformat()),
            )

    def list_forecast_accuracy(
        self, site_id: str, target: str | None = None, since: date | None = None
    ) -> list[dict]:
        sql = """
            SELECT target, method, target_date, metrics_json, computed_at
            FROM forecast_accuracy WHERE site_id = ?
        """
        params: list = [site_id]
        if target is not None:
            sql += " AND target = ?"
            params.append(target)
        if since is not None:
            sql += " AND target_date >= ?"
            params.append(since.isoformat())
        sql += " ORDER BY target_date DESC, method"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "target": row["target"],
                "method": row["method"],
                "target_date": row["target_date"],
                "metrics": json.loads(row["metrics_json"]),
                "computed_at": _parse_dt(row["computed_at"]),
            }
            for row in rows
        ]

    def insert_auto_plan_run(
        self, *, site_id: str, plan_date: date, run_id: str, kind: str
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO auto_plan_runs(site_id, plan_date, run_id, kind, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (site_id, plan_date.isoformat(), run_id, kind, _now().isoformat()),
            )

    def get_auto_plan_run(self, run_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT site_id, plan_date, run_id, kind, created_at
                FROM auto_plan_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "site_id": row["site_id"],
            "plan_date": row["plan_date"],
            "run_id": row["run_id"],
            "kind": row["kind"],
            "created_at": _parse_dt(row["created_at"]),
        }

    def insert_dispatch_run(
        self,
        *,
        run_id: str,
        site_id: str,
        date: date,
        strategy: str,
        initial_soc: float,
        input_digest: str,
        result: ScheduleResponse,
        data_source: str = "measurements",
    ) -> None:
        now = _now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO dispatch_runs(
                    run_id, site_id, date, strategy, initial_soc, input_digest, result_json,
                    data_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    site_id,
                    date.isoformat(),
                    strategy,
                    initial_soc,
                    input_digest,
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    data_source,
                    now.isoformat(),
                ),
            )

    def list_dispatch_runs(self, site_id: str) -> list[DispatchRunSummary]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT run_id, site_id, date, strategy, initial_soc, result_json, data_source, created_at
                FROM dispatch_runs
                WHERE site_id = ?
                ORDER BY created_at DESC
                """,
                (site_id,),
            ).fetchall()
            auto_rows = conn.execute(
                "SELECT run_id FROM auto_plan_runs WHERE site_id = ?",
                (site_id,),
            ).fetchall()
        auto_ids = {row["run_id"] for row in auto_rows}
        summaries: list[DispatchRunSummary] = []
        for row in rows:
            result = json.loads(row["result_json"])
            summaries.append(
                DispatchRunSummary(
                    run_id=row["run_id"],
                    site_id=row["site_id"],
                    date=row["date"],
                    strategy=row["strategy"],
                    initial_soc=row["initial_soc"],
                    expected_benefit_yuan=result["expected_benefit_yuan"],
                    baseline_cost_yuan=result["baseline_cost_yuan"],
                    optimized_cost_yuan=result["optimized_cost_yuan"],
                    equivalent_full_cycles=result["equivalent_full_cycles"],
                    data_source=row["data_source"],
                    auto_generated=row["run_id"] in auto_ids,
                    created_at=_parse_dt(row["created_at"]),
                )
            )
        return summaries

    def get_dispatch_run(self, run_id: str) -> DispatchRunDetail | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, site_id, input_digest, result_json, created_at
                FROM dispatch_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"])
        return DispatchRunDetail.model_validate(
            {
                **result,
                "run_id": row["run_id"],
                "input_digest": row["input_digest"],
                "created_at": _parse_dt(row["created_at"]),
            }
        )

    def insert_rolling_plan(
        self,
        *,
        site_id: str,
        date: str,
        revision: int,
        base_run_id: str,
        initial_soc: float,
        schedule_json: str,
        expected_benefit_yuan: float,
        trigger_reason: str,
    ) -> datetime:
        now = _now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO rolling_plans(
                    site_id, date, revision, base_run_id, initial_soc,
                    schedule_json, expected_benefit_yuan, trigger_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    site_id,
                    date,
                    revision,
                    base_run_id,
                    initial_soc,
                    schedule_json,
                    expected_benefit_yuan,
                    trigger_reason,
                    now.isoformat(),
                ),
            )
        return now

    def get_rolling_plans(self, site_id: str, target_date: date) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT revision, base_run_id, initial_soc, schedule_json,
                       expected_benefit_yuan, trigger_reason, created_at
                FROM rolling_plans
                WHERE site_id = ? AND date = ?
                ORDER BY revision ASC
                """,
                (site_id, target_date.isoformat()),
            ).fetchall()
        return [
            {
                "revision": row["revision"],
                "base_run_id": row["base_run_id"],
                "initial_soc": row["initial_soc"],
                "schedule_json": row["schedule_json"],
                "expected_benefit_yuan": row["expected_benefit_yuan"],
                "trigger_reason": row["trigger_reason"],
                "created_at": _parse_dt(row["created_at"]),
            }
            for row in rows
        ]

    def insert_sizing_run(
        self, *, run_id: str, site_id: str, input_digest: str, result: SizingResponse
    ) -> datetime:
        now = _now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO sizing_runs(run_id, site_id, input_digest, result_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    site_id,
                    input_digest,
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    now.isoformat(),
                ),
            )
        return now

    def list_sizing_runs(self, site_id: str) -> list[SizingRunSummary]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT run_id, site_id, result_json, created_at
                FROM sizing_runs
                WHERE site_id = ?
                ORDER BY created_at DESC
                """,
                (site_id,),
            ).fetchall()
        summaries: list[SizingRunSummary] = []
        for row in rows:
            result = json.loads(row["result_json"])
            recommended = result.get("recommended")
            summaries.append(
                SizingRunSummary(
                    run_id=row["run_id"],
                    site_id=row["site_id"],
                    created_at=_parse_dt(row["created_at"]),
                    recommended_pcs_kw=recommended["pcs_kw"] if recommended else None,
                    recommended_capacity_kwh=recommended["capacity_kwh"] if recommended else None,
                    npv_yuan=recommended["npv_yuan"] if recommended else None,
                    payback_years=recommended["payback_years"] if recommended else None,
                )
            )
        return summaries

    def get_sizing_run(self, run_id: str) -> SizingRunDetail | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, site_id, input_digest, result_json, created_at
                FROM sizing_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"])
        return SizingRunDetail.model_validate(
            {
                **result,
                "run_id": row["run_id"],
                "input_digest": row["input_digest"],
                "created_at": _parse_dt(row["created_at"]),
            }
        )

    def insert_report_job(
        self,
        *,
        report_id: str,
        site_id: str,
        report_type: str,
        sizing_run_id: str | None,
    ) -> ReportJob:
        now = _now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO report_jobs(
                    report_id, site_id, report_type, sizing_run_id, status,
                    error, file_path, created_at, finished_at
                ) VALUES (?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)
                """,
                (report_id, site_id, report_type, sizing_run_id, now.isoformat()),
            )
        job = self.get_report_job(report_id)
        assert job is not None
        return job

    def update_report_job(
        self,
        report_id: str,
        *,
        status: str,
        error: str | None = None,
        file_path: str | None = None,
        finished: bool = False,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE report_jobs
                SET status = ?, error = ?, file_path = ?,
                    finished_at = CASE WHEN ? THEN ? ELSE finished_at END
                WHERE report_id = ?
                """,
                (status, error, file_path, finished, _now().isoformat(), report_id),
            )

    def get_report_job(self, report_id: str) -> ReportJob | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM report_jobs WHERE report_id = ?", (report_id,)
            ).fetchone()
        if row is None:
            return None
        return ReportJob(
            report_id=row["report_id"],
            site_id=row["site_id"],
            report_type=row["report_type"],
            sizing_run_id=row["sizing_run_id"],
            status=row["status"],
            error=row["error"],
            file_path=row["file_path"],
            created_at=_parse_dt(row["created_at"]),
            finished_at=_parse_dt(row["finished_at"]) if row["finished_at"] else None,
        )

    @staticmethod
    def _site_from_row(row: sqlite3.Row) -> SiteRead:
        config = json.loads(row["config_json"])
        location_extra = config.get("location", {})
        return SiteRead.model_validate(
            {
                "site_id": row["site_id"],
                "name": row["name"],
                "location": {
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "timezone": row["timezone"],
                    "province": location_extra.get("province"),
                    "city": location_extra.get("city"),
                },
                "pv": config["pv"],
                "load": config["load"],
                "storage": config["storage"],
                "tariff_id": row["tariff_id"],
                # 存量 config_json 可能没有 demand_limit_kw，缺省为 None
                "demand_limit_kw": config.get("demand_limit_kw"),
                "created_at": _parse_dt(row["created_at"]),
                "updated_at": _parse_dt(row["updated_at"]),
            }
        )

    @staticmethod
    def _tariff_from_row(row: sqlite3.Row) -> TariffRead:
        return TariffRead.model_validate(
            {
                "tariff_id": row["tariff_id"],
                "name": row["name"],
                "mode": row["mode"],
                "transmission_and_fund": row["transmission_and_fund"],
                "tiers": json.loads(row["tiers_json"]),
                "created_at": _parse_dt(row["created_at"]),
                "updated_at": _parse_dt(row["updated_at"]),
            }
        )
