"""调度编排：取站点/测量或预测/电价 → 跑引擎 → 落库 dispatch_runs。不依赖 FastAPI。"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from app.api import errors
from app.application import forecast_service
from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.forecast.result import ForecastPoint
from app.domain.models import MeasurementRead, PricePoint, ScheduleRequest, ScheduleResponse
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database


def _input_digest(
    payload: ScheduleRequest,
    pv_points: list[MeasurementRead],
    load_points: list[MeasurementRead],
    price_points: list[PricePoint],
) -> str:
    material = {
        "site_id": payload.site_id,
        "date": payload.date.isoformat(),
        "initial_soc": payload.initial_soc,
        "strategy": payload.strategy,
        "resolution_minutes": payload.resolution_minutes,
        "source": payload.source,
        "pv": [point.value for point in pv_points],
        "load": [point.value for point in load_points],
        "price": [point.price for point in price_points],
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _forecast_as_measurements(data_type: str, points: list[ForecastPoint]) -> list[MeasurementRead]:
    return [
        MeasurementRead(
            data_type=data_type,  # type: ignore[arg-type]
            ts=point.ts,
            value=point.value,
            unit="kW",
            quality="estimated",
            source="forecast",
        )
        for point in points
    ]


def run_schedule(db: Database, payload: ScheduleRequest) -> ScheduleResponse:
    site = db.get_site(payload.site_id)
    if site is None:
        raise errors.not_found(f"site not found: {payload.site_id}")
    if payload.resolution_minutes != 15:
        raise errors.validation_error("MVP only supports 15-minute resolution")

    expected_points = 24 * 60 // payload.resolution_minutes
    source = payload.source
    if source in ("auto", "measurements"):
        pv_points = db.get_measurements_for_date(
            payload.site_id, "pv_power", payload.date, site.location.timezone
        )
        load_points = db.get_measurements_for_date(
            payload.site_id, "load_power", payload.date, site.location.timezone
        )
        complete = (
            len(pv_points) == expected_points
            and len(load_points) == expected_points
            and all(point.value is not None for point in pv_points)
            and all(point.value is not None for point in load_points)
        )
        if source == "measurements":
            if not pv_points or not load_points:
                raise errors.validation_error(
                    "pv_power and load_power measurements are required for the requested date"
                )
            if len(pv_points) != len(load_points):
                raise errors.validation_error("pv/load measurement lengths do not match")
            if len(pv_points) != expected_points:
                raise errors.validation_error(
                    f"a full day requires exactly {expected_points} points at {payload.resolution_minutes}-minute resolution"
                )
        elif not complete:
            source = "forecast"
        else:
            source = "measurements"

    if source == "forecast":
        try:
            pv_points = _forecast_as_measurements(
                "pv_power",
                forecast_service.forecast_series(db, site, "pv_power", payload.date),
            )
            load_points = _forecast_as_measurements(
                "load_power",
                forecast_service.forecast_series(db, site, "load_power", payload.date),
            )
        except ValueError as exc:
            if payload.source == "auto":
                raise errors.data_quality_error(
                    f"实测数据不足且预测历史不足：{exc}"
                ) from exc
            raise errors.data_quality_error(str(exc)) from exc

    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {tariff_id}")
    try:
        provider = TouProvider(tariff)
        price_points = provider.points(
            pv_points[0].ts,
            periods=len(pv_points),
            resolution_minutes=payload.resolution_minutes,
        )
        result = RuleDispatchEngine().run(
            DispatchInputs(
                site=site,
                target_date=payload.date,
                initial_soc=payload.initial_soc,
                pv_points=pv_points,
                load_points=load_points,
                price_points=price_points,
                resolution_minutes=payload.resolution_minutes,
            )
        )
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc

    run_id = f"R{uuid4().hex[:8].upper()}"
    digest = _input_digest(payload, pv_points, load_points, price_points)
    result = result.model_copy(update={"run_id": run_id, "data_source": source})
    db.insert_dispatch_run(
        run_id=run_id,
        site_id=payload.site_id,
        date=payload.date,
        strategy=payload.strategy,
        initial_soc=payload.initial_soc,
        input_digest=digest,
        result=result,
        data_source=source,
    )
    return result
