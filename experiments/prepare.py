"""Read existing real data, freeze hashes and convert to time-aligned tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from experiments.protocol import ROOT, chronological_splits, dump_json, sha256

SKY_FILE = Path("/Users/july/Downloads/2017_2019_images_pv_processed.hdf5")
SG_DIR = Path("/Users/july/Documents/揭榜/数据与参考文献/Renewable-energy-generation-input-feature-variables-analysis-main/data_original/solar_stations")
CAPS_MW = [50, 130, 30, 130, 110, 35, 30, 30]


def prepare_sky():
    frames = []
    sources = []
    with h5py.File(SKY_FILE, "r") as f:
        for group in ("trainval", "test"):
            path = Path(f"/Users/july/Downloads/times_{group}.npy")
            raw = np.load(path, allow_pickle=True)  # locally supplied datetime arrays
            ts = pd.DatetimeIndex(raw).floor("min").tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
            # Source PV is a forward minute average; available only at end of minute.
            ts += pd.Timedelta(minutes=1)
            frame = pd.DataFrame({"ts": ts, "power_kw": f[group]["pv_log"][:],
                                  "image_group": group, "image_index": np.arange(len(raw)), "raw_image_time": raw})
            frame["weather"] = "unknown"
            # Official test dates label SUNNY or CLOUDY only. Cloudy does not
            # distinguish partly cloudy from overcast, so retain its original class.
            sunny = {"2017-09-15", "2017-10-06", "2017-10-22", "2018-02-16", "2018-06-12", "2018-06-23", "2019-01-25", "2019-06-23", "2019-07-14", "2019-10-14"}
            if group == "test":
                dates = frame.ts.dt.strftime("%Y-%m-%d")
                frame["weather"] = np.where(dates.isin(sunny), "sunny", "cloudy_unspecified")
            frames.append(frame)
            sources.append({"path": str(path), "sha256": sha256(path)})
    df = pd.concat(frames, ignore_index=True).sort_values(["ts", "raw_image_time"]).reset_index(drop=True)
    duplicates = int(df.ts.duplicated().sum())
    df = df.drop_duplicates("ts", keep="first").reset_index(drop=True)
    if df.ts.isna().any():
        raise ValueError("ambiguous sky timestamps require manual review")
    negatives = int((df.power_kw < 0).sum())
    df.power_kw = df.power_kw.clip(lower=0)  # remove meter standby offset only
    if not np.isfinite(df.power_kw).all():
        raise ValueError("nonfinite power labels")
    df["daylight"] = True  # source only contains image acquisition daylight cases
    directory = ROOT / "data/processed/skippd"
    directory.mkdir(parents=True, exist_ok=True)
    df.to_pickle(directory / "frame.pkl")
    split = chronological_splits(df.ts)
    dump_json(directory / "split.json", split)
    manifest = {"dataset": "SKIPPD", "cap_kw": 30.1,
                "capacity_basis": "SKIPPD paper section 3.2: rated 30.1 kW-DC; available AC capacity unprovided",
                "capacity_status": "nominal DC reference; contest available capacity still to verify",
                "lat": 37.427, "lon": -122.174, "timezone": "America/Los_Angeles",
                "sources": [{"path": str(SKY_FILE), "sha256": sha256(SKY_FILE)}] + sources,
                "rows": len(df), "negative_meter_offsets_clipped": negatives,
                "duplicate_minutes_removed": duplicates, "duplicate_policy": "earliest image per minute",
                "time_semantics": "images floor to minute, forward-average PV available at minute end (+1min)",
                "image_source": str(SKY_FILE), "source_resolution": [64, 64],
                "start": df.ts.iloc[0].isoformat(), "end": df.ts.iloc[-1].isoformat(),
                "split": split, "nighttime": "absent in source HDF5", "forecast_nwp": "not provided"}
    dump_json(directory / "manifest.json", manifest)
    print("prepared skippd", len(df), split, flush=True)


def prepare_state_grid():
    for site, cap in enumerate(CAPS_MW, 1):
        path = SG_DIR / f"Solar station site {site} (Nominal capacity-{cap}MW).xlsx"
        raw = pd.read_excel(path)
        # Read raw tables; processed versions may already use whole-data scaling.
        df = pd.DataFrame({"ts": pd.to_datetime(raw.iloc[:, 0]).dt.tz_localize("Asia/Shanghai"),
                           "power_kw": pd.to_numeric(raw.iloc[:, 7], errors="coerce") * 1000,
                           "tsi": pd.to_numeric(raw.iloc[:, 1], errors="coerce"), "dni": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
                           "ghi": pd.to_numeric(raw.iloc[:, 3], errors="coerce"), "temperature": pd.to_numeric(raw.iloc[:, 4], errors="coerce")})
        df = df.sort_values("ts").reset_index(drop=True)
        # Remove missing rows, retain gaps, and exclude any history/target
        # window crossing those gaps. Never interpolate missing power labels.
        missing = int(df[["power_kw", "tsi", "dni", "ghi", "temperature"]].isna().any(axis=1).sum())
        df = df.dropna().reset_index(drop=True)
        df["weather"] = "unknown"  # no cloud labels or site coordinates in these tables
        df["daylight"] = (df.ts.dt.hour >= 6) & (df.ts.dt.hour < 19)
        negatives = int((df.power_kw < 0).sum())
        df.power_kw = df.power_kw.clip(lower=0)
        if df.ts.duplicated().any() or not np.isfinite(df[["power_kw", "tsi", "dni", "ghi", "temperature"]]).all().all():
            raise ValueError(f"invalid state grid station {site}")
        directory = ROOT / f"data/processed/state_grid_{site}"
        directory.mkdir(parents=True, exist_ok=True)
        df.to_pickle(directory / "frame.pkl")
        split = chronological_splits(df.ts, history_minutes=960)
        dump_json(directory / "split.json", split)
        dump_json(directory / "manifest.json", {"dataset": f"state_grid_{site}", "cap_kw": cap * 1000,
                  "capacity_basis": "source filename nominal MW * 1000; contest available capacity unprovided",
                  "capacity_status": "nominal reference", "timezone": "Asia/Shanghai", "rows": len(df),
                  "negative_meter_offsets_clipped": negatives,
                  "missing_rows_excluded": missing, "missing_policy": "exclude rows and gap-crossing windows, no label interpolation",
                  "time_semantics": "assumed measurement available at timestamp; interval convention unprovided",
                  "source": {"path": str(path), "sha256": sha256(path)}, "split": split,
                  "forecast_nwp": "not provided; weather columns are observations", "geolocation": "unprovided",
                  "daylight_definition": "06:00-19:00 local clock proxy; solar elevation unavailable"})
        print("prepared state_grid", site, len(df), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "sky", "state_grid"], default="all")
    args = parser.parse_args()
    if args.dataset in ("all", "sky"):
        prepare_sky()
    if args.dataset in ("all", "state_grid"):
        prepare_state_grid()
