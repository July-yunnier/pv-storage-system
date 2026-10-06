import os
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.dependencies import get_db
from app.application import forecast_service
from app.domain.forecast.result import ForecastResponse
from app.domain.models import ForecastRequest, MachineLearningForecastRequest
from app.api import errors
from app.infrastructure.db import Database

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("", response_model=ForecastResponse)
def create_forecast(
    payload: ForecastRequest,
    db: Database = Depends(get_db),
) -> ForecastResponse:
    return forecast_service.run_forecast(db, payload)


@router.post("/ml", response_model=ForecastResponse)
def create_ml_forecast(payload: MachineLearningForecastRequest) -> ForecastResponse:
    """Trusted server-side checkpoints; client cannot submit arbitrary paths."""
    from app.domain.forecast.models import MODEL_NAMES
    from app.domain.forecast.load_forecast import forecast_load_ml
    from app.domain.forecast.price_forecast_service import forecast_price_ml
    if payload.model not in MODEL_NAMES:
        raise errors.validation_error("unknown ML forecast model")
    root = Path(os.environ.get("FORECAST_ARTIFACT_ROOT", "data/forecast_experiments"))
    artifact = root/payload.task/f"{payload.horizon_hours}h"/f"{payload.model}__full"/"weights"
    try:
        service = forecast_load_ml if payload.task == "load" else forecast_price_ml
        result = service(payload, artifact)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise errors.data_quality_error(str(exc)) from exc
    return ForecastResponse(site_id="offline-benchmark", issued_at=payload.forecast_origin,
                            resolution_minutes=payload.resolution_minutes, results=[result])
