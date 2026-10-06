"""Load inference uses the shared model code, never the price checkpoint."""
from __future__ import annotations

from pathlib import Path

from app.domain.forecast.inference import predict_artifact
from app.domain.forecast.result import TargetForecast
from app.domain.models import MachineLearningForecastRequest


def forecast_load_ml(request: MachineLearningForecastRequest, artifact_dir: Path) -> TargetForecast:
    if request.task != "load":
        raise ValueError("load service only accepts task=load")
    return predict_artifact(request, artifact_dir)
