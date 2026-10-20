"""Explicit routing for delivered AMPDNet; does not replace default day-ahead model."""
from app.api import errors
from app.domain.forecast.ampdnet import predict_ampdnet
from app.domain.forecast.result import ForecastResponse
from app.infrastructure.db import _zone


def run_ampdnet_forecast(db, site, payload):
    inputs = payload.options.ampdnet_input
    assert inputs is not None
    hours = {'15min': .25, '1h': 1, '4h': 4}[inputs.task]
    if payload.targets != ['pv_power'] or not payload.options.intraday or payload.horizon_hours != hours or payload.issued_at is None:
        raise errors.validation_error("AMPDNet requires explicit issued_at, targets=['pv_power'], intraday=true and horizon_hours matching task (0.25/1/4)")
    try:
        result = predict_ampdnet(inputs,payload.issued_at,site.pv.capacity_kwp,site.location.timezone)
    except (ValueError,RuntimeError,OSError) as exc:
        raise errors.data_quality_error(str(exc)) from exc
    # Retain provenance where the application provides forecast archival support.
    if hasattr(db,'insert_forecast_archive'):
        db.insert_forecast_archive(site_id=site.site_id,target=result.target,method=result.model.name,
            target_date=result.points[0].ts.astimezone(_zone(site.location.timezone)).date(),
            issued_at=payload.issued_at,points=[p.model_dump(mode='json') for p in result.points],
            provenance=result.provenance.model_dump(mode='json'))
    return ForecastResponse(site_id=site.site_id,issued_at=payload.issued_at,resolution_minutes=15,results=[result])
