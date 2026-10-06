"""Price inference uses the shared model code, never the load checkpoint."""
from __future__ import annotations

from pathlib import Path

from app.domain.forecast.inference import predict_artifact
from app.domain.forecast.result import TargetForecast
from app.domain.models import MachineLearningForecastRequest


def forecast_price_ml(request: MachineLearningForecastRequest, artifact_dir: Path) -> TargetForecast:
    if request.task != "price":
        raise ValueError("price service only accepts task=price")
    return predict_artifact(request, artifact_dir)
