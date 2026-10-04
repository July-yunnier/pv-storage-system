"""CSV 测量数据入库：解析/校验 CSV 文本并写入 Database，不依赖 FastAPI。"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime

from app.domain.models import MeasurementPoint, MeasurementType, MeasurementUpload
from app.infrastructure.db import Database

EXPECTED_HEADER = ("ts", "value")


@dataclass(frozen=True)
class RejectedRow:
    row: int
    reason: str


@dataclass
class IngestionResult:
    site_id: str
    data_type: str
    total_rows: int
    accepted: int
    rejected: list[RejectedRow] = field(default_factory=list)


def _parse_ts(raw: str) -> datetime:
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO8601 ts: {raw!r}") from exc
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"ts must include a timezone: {raw!r}")
    return ts


def _validate_value(data_type: MeasurementType, point: MeasurementPoint) -> None:
    """与 Database.insert_measurements 相同的取值范围规则，提前到行级校验。"""
    value = point.value
    if value is None:
        return
    if data_type == "storage_soc" and not 0 <= value <= 1:
        raise ValueError("storage_soc value must be between 0 and 1")
    if data_type != "storage_soc" and value < 0:
        raise ValueError("power measurement cannot be negative")


def _parse_row(line_no: int, cells: list[str]) -> MeasurementPoint:
    if len(cells) < 2:
        raise ValueError("row must have at least ts and value columns")
    ts = _parse_ts(cells[0].strip())
    raw_value = cells[1].strip()
    if raw_value == "":
        return MeasurementPoint(ts=ts, value=None, quality="missing")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value: {raw_value!r}") from exc
    return MeasurementPoint(ts=ts, value=value)


def ingest_csv(
    db: Database,
    site_id: str,
    data_type: MeasurementType,
    content: str,
    *,
    unit: str = "kW",
    resolution_minutes: int = 15,
    source: str = "csv",
) -> IngestionResult:
    """解析 CSV 文本并入库。

    行级容错：单行失败记入 rejected。文件无表头/无数据行/全部行非法时抛 ValueError，
    由调用方（HTTP 层）转换为 422。
    """
    reader = csv.reader(io.StringIO(content))
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("csv file is empty")
    header = [cell.lower() for cell in rows[0][:2]]
    if tuple(header) != EXPECTED_HEADER:
        raise ValueError(f"csv header must be 'ts,value', got: {','.join(rows[0])!r}")

    data_rows = rows[1:]
    if not data_rows:
        raise ValueError("csv file has no data rows")

    points: list[MeasurementPoint] = []
    rejected: list[RejectedRow] = []
    seen_ts: set[datetime] = set()
    for offset, cells in enumerate(data_rows):
        line_no = offset + 2  # 表头占第 1 行
        try:
            point = _parse_row(line_no, cells)
            _validate_value(data_type, point)
        except ValueError as exc:
            rejected.append(RejectedRow(row=line_no, reason=str(exc)))
            continue
        if point.ts in seen_ts:
            rejected.append(RejectedRow(row=line_no, reason=f"duplicate ts: {point.ts.isoformat()}"))
            continue
        seen_ts.add(point.ts)
        points.append(point)

    if not points:
        raise ValueError(f"all {len(data_rows)} data rows are invalid")

    points.sort(key=lambda point: point.ts)
    upload = MeasurementUpload(
        data_type=data_type,
        unit=unit,
        resolution_minutes=resolution_minutes,
        source=source,
        points=points,
    )
    accepted = db.insert_measurements(site_id, upload)
    return IngestionResult(
        site_id=site_id,
        data_type=data_type,
        total_rows=len(data_rows),
        accepted=accepted,
        rejected=rejected,
    )
