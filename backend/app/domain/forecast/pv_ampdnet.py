"""Pure AMPDNet PV inference.

The application layer injects an explicit weather window and an artifact loader;
this module only validates the temporal contract and turns model output into a
``TargetForecast``.  It never reads the database, environment, or filesystem.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from app.domain.forecast.result import (
    ForecastModel,
    ForecastPoint,
    ForecastProvenance,
    TargetForecast,
)

TASKS: dict[str, tuple[int, int]] = {"15min": (48, 1), "1h": (96, 4), "4h": (192, 16)}
RELEASE = None  # compatibility marker; artifact paths belong to infrastructure
FEATURE_ATTRIBUTES = ("tsi", "dni", "ghi", "temperature_c")


def _daylight_slot(ts: datetime) -> bool:
    return 6 <= ts.hour < 19 and ts.minute % 15 == 0 and ts.second == 0 and ts.microsecond == 0


def _next_daylight(ts: datetime) -> datetime:
    if ts.hour == 18 and ts.minute == 45:
        return (ts + timedelta(days=1)).replace(hour=6, minute=0)
    return ts + timedelta(minutes=15)


def predict_pv_ampdnet(
    inputs: Any,
    issued_at: datetime,
    cap_kw: float,
    timezone_name: str,
    *,
    metadata: dict[str, Any],
    infer: Callable[[Any], Any],
    normalize: Callable[[Sequence[Sequence[float]], dict[str, Any]], Any],
    denormalize: Callable[[Any, dict[str, Any]], Any],
) -> TargetForecast:
    """Run one frozen AMPDNet artifact using explicit observations.

    ``infer`` and scaling callbacks are injected by the application layer so
    this function remains deterministic and free of I/O.
    """
    if timezone_name != "Asia/Shanghai":
        raise ValueError("AMPDNet requires the Asia/Shanghai station timezone")
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("AMPDNet issued_at must include a timezone")
    tz = ZoneInfo(timezone_name)
    issued_at = issued_at.astimezone(tz)
    if not _daylight_slot(issued_at):
        raise ValueError("AMPDNet only supports 06:00-18:45 quarter-hour origins")
    if inputs.task not in TASKS:
        raise ValueError(f"unsupported AMPDNet task: {inputs.task}")
    length, steps = TASKS[inputs.task]
    observations = inputs.observations
    if len(observations) != length:
        raise ValueError(f"{inputs.task} requires exactly {length} daylight observations")
    times = [row.ts.astimezone(tz) for row in observations]
    if times[-1] != issued_at or any(not _daylight_slot(ts) for ts in times):
        raise ValueError("observations must be daylight quarter-hour samples ending at issued_at")
    if any(_next_daylight(left) != right for left, right in zip(times, times[1:])):
        raise ValueError("observations must be contiguous daylight samples without gaps or reordering")
    targets = [issued_at + timedelta(minutes=15 * index) for index in range(1, steps + 1)]
    if any(ts.date() != issued_at.date() or not _daylight_slot(ts) for ts in targets):
        raise ValueError("AMPDNet refuses forecast paths crossing the daylight boundary")
    if abs(cap_kw - float(metadata["cap_kw"])) > 1e-6:
        raise ValueError("site capacity does not match the selected AMPDNet station artifact")
    raw = [[float(getattr(row, feature)) for feature in FEATURE_ATTRIBUTES] for row in observations]
    normalized = normalize(raw, metadata)
    output = infer(normalized)
    expected_steps = steps
    if len(output) != expected_steps:
        raise RuntimeError(f"AMPDNet output length {len(output)} does not match task horizon {expected_steps}")
    raw_kw = denormalize(output, metadata)
    values = [min(max(float(value), 0.0), cap_kw) for value in raw_kw]
    return TargetForecast(
        target="pv_power",
        unit="kW",
        model=ForecastModel(name=f"ampdnet_{inputs.station.lower()}_{inputs.task}", version="delivery20-full-s42"),
        points=[ForecastPoint(ts=ts, value=round(value, 6)) for ts, value in zip(targets, values)],
        provenance=ForecastProvenance(
            provider="provided_daylight_observations",
            window=f"{length} daylight samples; {steps} quarter-hour outputs",
            checkpoint_sha256=metadata.get("checkpoint_sha256"),
            artifact_sha256=metadata.get("artifact_sha256"),
            training_status=(
                f"{metadata.get('training_epochs_completed', 0)} epochs completed; "
                f"validation-selected epoch {metadata.get('selected_epoch', metadata.get('best_epoch'))}; "
                "historical daylight-compressed protocol"
            ),
        ),
        accuracy=None,
        p10=None,
        p90=None,
        components={"raw_prediction_kw": [float(value) for value in raw_kw]},
        notes=(
            "AMPDNet is an opt-in intraday short-horizon candidate. It uses the historical "
            "daylight-compressed protocol and is not a day-ahead, probabilistic, or cross-site result."
        ),
    )
