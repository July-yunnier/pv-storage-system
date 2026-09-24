from fastapi import APIRouter, Depends

from app.api.dependencies import get_db
from app.application import forecast_service
from app.domain.forecast.result import ForecastResponse
from app.domain.models import ForecastRequest
from app.infrastructure.db import Database

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("", response_model=ForecastResponse)
def create_forecast(
    payload: ForecastRequest,
    db: Database = Depends(get_db),
) -> ForecastResponse:
    return forecast_service.run_forecast(db, payload)
