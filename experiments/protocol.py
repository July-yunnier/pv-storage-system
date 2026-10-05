from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_version():
    # New workspace has no HEAD yet. Hash source contents as well as the source
    # repository commit, so uncommitted edits remain reproducible.
    base = json.loads((ROOT / "provenance/source.json").read_text())
    digest = hashlib.sha256()
    files = sorted(set((ROOT / "experiments").rglob("*.py")) | set((ROOT / "product/backend/app").rglob("*.py")))
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    base["working_source_sha256"] = digest.hexdigest()
    return base


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def chronological_splits(times, history_minutes=30, horizon_minutes=240):
    """Last 30% of calendar time is test; first 60% train, next 10% val.

    Select boundaries at day starts to prevent a day straddling partitions.
    Use actual timestamp comparisons for all labels. Purging is applied by
    sample_indices; no labels cross the cutoffs, no random split is used.
    """
    import pandas as pd
    times = pd.DatetimeIndex(times)
    days = pd.DatetimeIndex(times.normalize().unique()).sort_values()
    if len(days) < 10:
        raise ValueError("At least 10 calendar days are needed")
    train_end = days[int(len(days) * .6)]
    val_end = days[int(len(days) * .7)]
    return {
        "train_end": train_end.isoformat(), "val_end": val_end.isoformat(),
        "test_end": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "method": "chronological_calendar_days_60_10_30",
        "history_minutes": history_minutes, "max_horizon_minutes": horizon_minutes,
        "test_fraction_requested": .3,
    }


def sample_indices(frame, split, lookback, step_minutes, horizons=(15, 60, 240), stride=1):
    """Require exact continuous input and exact future timestamp matches.

    Training/validation/test targets all stay inside their own partition.
    Past observations in the test period are legal at later issue times;
    weights and scalers never fit on them. No forward-filling labels or NWP.
    """
    import pandas as pd
    times = pd.DatetimeIndex(frame["ts"])
    tick = times.asi8 // (60 * 1_000_000_000)
    if len(np.unique(tick)) != len(tick) or np.any(np.diff(tick) <= 0):
        raise ValueError("timestamps must be unique and ordered")
    future = np.searchsorted(tick, tick[:, None] + np.array(horizons)[None, :])
    bounded = future < len(tick)
    safe = np.minimum(future, len(tick) - 1)
    exact = (tick[safe] == tick[:, None] + np.array(horizons)) & bounded
    # Every history step must be observed at the configured cadence.
    bad = np.r_[0, np.diff(tick) != step_minutes].astype(int).cumsum()
    idx = np.arange(lookback - 1, len(times), stride)
    valid = (bad[idx] - bad[idx - lookback + 1] == 0) & exact[idx].all(axis=1)
    idx = idx[valid]
    boundaries = [times[0], pd.Timestamp(split["train_end"]), pd.Timestamp(split["val_end"]), pd.Timestamp(split["test_end"])]
    out = {}
    for name, start, end in zip(("train", "val", "test"), boundaries, boundaries[1:]):
        mask = (times[idx] >= start) & (times[idx] < end)
        mask &= times[safe[idx, -1]] < end
        mask &= times[idx - lookback + 1] >= start
        out[name] = idx[mask]
    if any(len(out[k]) == 0 for k in out):
        raise ValueError(f"empty split after gap/target checks: { {k: len(v) for k,v in out.items()} }")
    return out, safe


def validate_nwp(frame):
    """Archived forecast vintage required, not target-time observations."""
    import pandas as pd
    required = {"issued_at", "available_at", "valid_at", "forecast_ghi", "forecast_temperature"}
    if not required.issubset(frame.columns):
        raise ValueError(f"NWP requires {sorted(required)}")
    if (pd.to_datetime(frame["available_at"], utc=True) < pd.to_datetime(frame["issued_at"], utc=True)).any():
        raise ValueError("NWP availability cannot precede forecast issuance")


def package_run(directory, config, prediction_frame, state, cap_kw):
    from experiments.metrics import evaluate
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(config, code=code_version(), seed=42, cap_kw=float(cap_kw))
    dump_json(directory / "config.json", config)
    dump_json(directory / "weights.json", state)
    dump_json(directory / "code_version.json", config["code"])
    prediction_frame.to_csv(directory / "predictions.csv", index=False)
    result = evaluate(prediction_frame, cap_kw)
    dump_json(directory / "metrics.json", result)
    return result
