"""Causal features for already fitted, frozen PV estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

HORIZONS = (15, 60, 240)
IMAGE_NAMES = ["image_mean_r", "image_mean_g", "image_mean_b", "image_std_r", "image_std_g", "image_std_b",
               "image_rb_mean", "image_rb_std", "image_bright_fraction", "image_rb_positive_fraction"]
IMAGE_NAMES += [f"image_tile_{i}" for i in range(16)]


def image_descriptors(images):
    """Chronological uint8 RGB bitmaps, shape (30, 64, 64, 3)."""
    images = np.asarray(images)
    if images.shape != (30, 64, 64, 3) or images.dtype != np.uint8:
        raise ValueError("images must be 30 chronological 64x64 RGB uint8 frames")
    images = images.astype(np.float32) / 255.
    rb = (images[:, :, :, 0]-images[:, :, :, 2])/(images[:, :, :, 0]+images[:, :, :, 2]+.01)
    extra = np.stack([rb.mean((1, 2)), rb.std((1, 2)), (images.mean(3) > .85).mean((1, 2)),
                      (rb > 0).mean((1, 2))], 1)
    tiles = images.mean(3).reshape(30, 4, 16, 4, 16).mean((2, 4)).reshape(30, 16)
    return np.concatenate([images.mean((1, 2)), images.std((1, 2)), extra, tiles], 1).astype(np.float32)


def _aware_timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError("timestamps must be non-null and timezone-aware")
    return ts


def build_features(observations, issued_at, model_manifest, frozen_state,
                   images=None, descriptors=None, image_timestamps=None):
    """Build three feature rows without target observations or fitting.

    Observations are oldest to newest and end exactly at issued_at. Timestamp
    semantics must match the original dataset's stated availability assumptions.
    Descriptors are an alternate upstream preprocessing interface, not a trained
    cloud classifier. Their order is IMAGE_NAMES and timestamps are mandatory.
    """
    sky = model_manifest["family"] == "skippd"
    length, step = (30, 1) if sky else (64, 15)
    frame = pd.DataFrame(observations)
    required = ["power_kw"] if sky else ["power_kw", "tsi", "dni", "ghi", "temperature"]
    if "ts" not in frame or any(k not in frame for k in required) or len(frame) != length:
        raise ValueError(f"requires exactly {length} observations with ts and {required}")
    clock = pd.DatetimeIndex([_aware_timestamp(t).tz_convert("UTC") for t in frame.ts])
    issue = _aware_timestamp(issued_at).tz_convert("UTC")
    if clock[-1] != issue or (np.diff(clock.asi8) != step*60*10**9).any():
        raise ValueError("history must be continuous, ascending and end exactly at issuance")
    if (clock.second != 0).any() or (clock.microsecond != 0).any() or (clock.nanosecond != 0).any():
        raise ValueError("observation timestamps must be aligned to whole minutes")
    if "available_at" in frame:
        available = pd.DatetimeIndex([_aware_timestamp(t).tz_convert("UTC") for t in frame.available_at])
        if (available > issue).any() or (available < clock).any():
            raise ValueError("observations were unavailable at issuance or precede their measurement timestamp")
    values = frame[required].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any():
        raise ValueError("observations must be finite; power must be nonnegative")
    cap = model_manifest["cap_kw"]
    power = values[:, 0] / cap
    history = power[::-1][None, :]
    clock = clock.tz_convert(model_manifest["timezone"])
    targets = pd.DatetimeIndex([issue+pd.Timedelta(minutes=h) for h in HORIZONS]).tz_convert(model_manifest["timezone"])
    names, columns = [], []

    def add(name, value):
        names.append(name)
        columns.append(np.asarray(value, np.float32).reshape(1))

    def slot_mean(time):
        return frozen_state["historical_slot_means_kw"].get(str(time.hour*60+time.minute),
                                                        frozen_state["historical_slot_default_kw"])

    for lag in ([0, 1, 2, 3, 5, 10, 15, 20, 29] if sky else [0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 63]):
        add(f"power_lag_{lag*step}min", history[:, lag])
    for window in ([3, 5, 10, 15, 30] if sky else [4, 8, 16, 32, 64]):
        a = history[:, :window]
        for stat, v in (("mean", a.mean(1)), ("std", a.std(1)), ("min", a.min(1)), ("max", a.max(1))):
            add(f"power_{window*step}min_{stat}", v)
    add("power_delta", history[:, 0]-history[:, 1])
    add("power_slope", history[:, 0]-history[:, -1])
    for label, phase in (("clock", (clock[-1].hour*60+clock[-1].minute)/1440),
                         ("year", (clock[-1].dayofyear-1)/365.25)):
        add(label+"_sin", np.sin(2*np.pi*phase))
        add(label+"_cos", np.cos(2*np.pi*phase))
    add("historical_slot_mean", slot_mean(clock[-1])/cap)
    if sky:
        solar_times = clock.append(targets)
        location = pvlib.location.Location(model_manifest["latitude"], model_manifest["longitude"],
                                          tz=model_manifest["timezone"])
        clear = np.clip(np.asarray(pvlib.pvsystem.pvwatts_dc(
            location.get_clearsky(solar_times, model="ineichen").ghi, 25., cap, -.004)), 0, cap)/cap
        ratio = np.divide(power, clear[:length], out=np.zeros_like(power), where=clear[:length] > .01)
        add("clear_now", clear[length-1])
        add("clear_index_now", np.clip(ratio[-1], 0, 2))
        for w in (5, 15, 30):
            a = np.clip(ratio[-w:][::-1][None, :], 0, 2)
            add(f"clear_index_{w}_mean", a.mean(1))
            add(f"clear_index_{w}_std", a.std(1))
        solar = pvlib.solarposition.get_solarposition(clock[-1:], model_manifest["latitude"], model_manifest["longitude"])
        add("solar_elevation", solar.apparent_elevation.to_numpy()/90)
        add("solar_azimuth_sin", np.sin(np.deg2rad(solar.azimuth.to_numpy())))
        if model_manifest["images"]:
            if images is not None and descriptors is not None:
                raise ValueError("provide images or descriptors, not both")
            if image_timestamps is None:
                raise ValueError("image timestamps are required")
            im_clock = pd.DatetimeIndex([_aware_timestamp(t).tz_convert("UTC") for t in image_timestamps])
            if not im_clock.equals(clock.tz_convert("UTC")):
                raise ValueError("image timestamps must align exactly with history")
            descriptor = image_descriptors(images) if images is not None else np.asarray(descriptors, np.float32)
            if descriptor.shape != (30, 26) or not np.isfinite(descriptor).all():
                raise ValueError("descriptors must be finite and shaped (30, 26)")
            for j, name in enumerate(IMAGE_NAMES):
                add(name, descriptor[-1, j])
                add(name+"_30_mean", descriptor[::-1, j][None, :].mean(1))
                add(name+"_delta_15", descriptor[-1, j]-descriptor[-16, j])
        elif any(x is not None for x in (images, descriptors, image_timestamps)):
            raise ValueError("this frozen model does not accept image inputs")
    else:
        if any(x is not None for x in (images, descriptors, image_timestamps)):
            raise ValueError("state-grid models do not accept images")
        for label in ("tsi", "dni", "ghi", "temperature"):
            a = frame[label].to_numpy(np.float32)/(40. if label == "temperature" else 1000.)
            for lag in (0, 1, 4, 16):
                add(f"{label}_lag_{lag*15}min", a[-lag-1])
            for w in (4, 16, 64):
                past = a[-w:][::-1][None, :]
                add(f"{label}_{w*15}_mean", past.mean(1))
                add(f"{label}_{w*15}_std", past.std(1))
            add(label+"_delta", a[-1]-a[-2])
    common = np.column_stack(columns).astype(np.float32)
    matrices, schemas, baselines = [], [], []
    for j, h in enumerate(HORIZONS):
        phase = (targets[j].hour*60+targets[j].minute)/1440
        extra = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), slot_mean(targets[j])/cap, h/240.]
        extra_names = ["target_clock_sin", "target_clock_cos", "target_historical_mean", "horizon_fraction"]
        base = power[-1]
        if sky:
            tc = clear[length+j]
            extra += [tc, tc-clear[length-1]]
            extra_names += ["target_clear_power", "clear_change"]
            base = np.clip(ratio[-1]*tc, 0, 1)
        matrices.append(np.column_stack([common]+[np.array([v]) for v in extra]).astype(np.float32))
        schemas.append(names+extra_names)
        baselines.append(base)
    return matrices, schemas, np.array([baselines], dtype=np.float32), targets
