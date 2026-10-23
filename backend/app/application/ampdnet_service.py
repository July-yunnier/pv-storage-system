"""Application adapter for the explicit, CPU-only AMPDNet release."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from app.api import errors
from app.domain.forecast.pv_ampdnet import TASKS, predict_pv_ampdnet
from app.domain.forecast.result import ForecastResponse
from app.domain.forecast.pv import predict_pv_intraday
from app.infrastructure.db import _zone
from app.infrastructure.model_store import load_ampdnet_artifact


def _normalize(raw, metadata):
    low = np.asarray(metadata["input_min"], dtype=np.float32)
    high = np.asarray(metadata["input_max"], dtype=np.float32)
    scale = high - low
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("AMPDNet normalization metadata is invalid")
    values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("AMPDNet input features must be finite")
    import torch
    return torch.from_numpy(((values - low) / scale)[None, :, :])


def _infer(model, batch):
    import torch
    with torch.inference_mode():
        output = model(batch).detach().cpu().numpy()[0]
    if not np.isfinite(output).all():
        raise RuntimeError("AMPDNet output is not finite")
    return output


def _denormalize(output, metadata):
    values = np.asarray(output, dtype=np.float32)
    raw_kw = (
        values * np.float32(metadata["output_max_mw"] - metadata["output_min_mw"])
        + np.float32(metadata["output_min_mw"])
    ) * np.float32(1000)
    return raw_kw.tolist()


def _fallback(db, site, payload, steps: int, reason: str):
    """Use the existing intraday statistical forecast when explicitly enabled."""
    tz = _zone(site.location.timezone)
    issued = payload.issued_at.astimezone(tz)
    rows = []
    for offset in range(12, 0, -1):
        day = issued.date() - timedelta(days=offset)
        rows.extend(
            (row.ts, row.value)
            for row in db.get_measurements_for_date(site.site_id, "pv_power", day, site.location.timezone)
            if row.value is not None and row.quality in ("measured", "estimated")
        )
    rows.extend(
        (row.ts, row.value)
        for row in db.get_measurements_for_date(site.site_id, "pv_power", issued.date(), site.location.timezone)
        if row.value is not None and row.quality in ("measured", "estimated") and row.ts <= payload.issued_at
    )
    fallback = predict_pv_intraday(rows, issued, site.pv.capacity_kwp, tz)
    fallback.points = fallback.points[:steps]
    fallback.model.name = "pv_intraday_deviation_fallback"
    fallback.notes = f"AMPDNet unavailable; fell back to existing intraday statistical forecast: {reason}"
    return fallback


def run_ampdnet_forecast(db, site, payload) -> ForecastResponse:
    inputs = payload.options.ampdnet_input
    if inputs is None:
        raise errors.validation_error("ampdnet requires explicit ampdnet_input")
    steps = TASKS[inputs.task][1]
    hours = steps * 0.25
    if payload.targets != ["pv_power"] or not payload.options.intraday or payload.issued_at is None:
        raise errors.validation_error(
            "AMPDNet requires explicit issued_at, targets=['pv_power'], and intraday=true"
        )
    if abs(payload.horizon_hours - hours) > 1e-9:
        raise errors.validation_error(f"AMPDNet horizon_hours must be {hours} for task {inputs.task}")
    try:
        model, metadata = load_ampdnet_artifact(inputs.station, inputs.task)
        result = predict_pv_ampdnet(
            inputs,
            payload.issued_at,
            site.pv.capacity_kwp,
            site.location.timezone,
            metadata=metadata,
            infer=lambda batch: _infer(model, batch),
            normalize=_normalize,
            denormalize=_denormalize,
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        if payload.options.ampdnet_fallback == "error":
            raise errors.data_quality_error(str(exc)) from exc
        try:
            result = _fallback(db, site, payload, steps, str(exc))
        except (ValueError, RuntimeError) as fallback_exc:
            raise errors.data_quality_error(
                f"AMPDNet unavailable ({exc}); fallback unavailable ({fallback_exc})"
            ) from fallback_exc
    try:
        db.insert_forecast_archive(
            site_id=site.site_id,
            target=result.target,
            method=result.model.name,
            target_date=result.points[0].ts.astimezone(_zone(site.location.timezone)).date(),
            issued_at=payload.issued_at,
            points=[{"ts": point.ts.isoformat(), "value": point.value} for point in result.points],
            provenance=result.provenance.model_dump(mode="json") if result.provenance else None,
        )
    except Exception:  # noqa: BLE001 - archive failure never blocks inference
        pass
    return ForecastResponse(
        site_id=site.site_id,
        issued_at=payload.issued_at,
        resolution_minutes=payload.resolution_minutes,
        results=[result],
    )
