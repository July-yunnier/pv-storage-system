"""Read-only inference on supplied PV releases; NOT a GEFCom experiment.

Never calls fit/train, exports weights or modifies the external PV repository.
Uses fixed 1000-case audit subsets, not complete-dataset performance claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pv-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.pv_root))
    from experiments.predict_optimized import load_release, predict_feature_matrices
    from experiments.optimize import build_features
    from experiments.protocol import sample_indices
    from threadpoolctl import threadpool_limits
    releases = args.pv_root / "product/backend/data/model_releases/optimization_v2"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1):
        for checkpoint in sorted(releases.glob("*.joblib")):
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metadata = json.loads(checkpoint.with_suffix(".json").read_text())
            if before != metadata["sha256"]: raise ValueError("PV release hash mismatch")
            bundle = load_release(checkpoint)
            dataset = checkpoint.stem
            sky = dataset.startswith("skippd")
            frame = pd.read_pickle(args.pv_root / "data/processed" / ("skippd" if sky else dataset) / "frame.pkl")
            config = bundle["training_config"]
            indices, future = sample_indices(frame, config["split"], 30 if sky else 64, 1 if sky else 15)
            if config["images"]:
                # Do not allow the imported feature routine to write a cache.
                cache = args.pv_root / "data/processed/skippd/image_descriptors_v1.npz"
                if not cache.exists(): raise ValueError("existing sky-descriptor cache required for read-only inference")
            idx, partitions, matrices, names, baseline, _ = build_features(
                frame, indices, future, config["dataset_manifest"], config["split"], config["images"])
            test = partitions["test"]
            audit = test[np.unique(np.linspace(0, len(test)-1, min(1000, len(test))).astype(int))]
            point, q = predict_feature_matrices(bundle, [a[audit] for a in matrices], names, baseline[audit])
            origins = idx[audit]
            y = frame.power_kw.to_numpy()[future[origins]]
            training = frame.ts < pd.Timestamp(config["split"]["train_end"])
            proxy = float(frame.loc[training, "power_kw"].max())
            pred_rows = []
            for j, h in enumerate(bundle["horizons_minutes"]):
                err = point[:, j] - y[:, j]
                residual = y[:, j, None] - q[:, :, j]
                levels = np.array([.1, .5, .9])
                mae, rmse = float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))
                rows.append({"dataset": dataset, "horizon_minutes": h, "n": len(origins), "MAE": mae, "RMSE": rmse,
                    "Pinball": float(np.maximum(levels*residual, (levels-1)*residual).mean()),
                    "interval_coverage_80": float(((y[:, j] >= q[:, 0, j]) & (y[:, j] <= q[:, 2, j])).mean()),
                    "Cap_training_max_proxy": proxy, "1-MAE/Cap_proxy": 1-mae/proxy, "1-RMSE/Cap_proxy": 1-rmse/proxy,
                    "model_input_scale_cap_kw_unchanged": bundle["cap_kw"], "Cap_is_physical": False,
                    "interval_type": "existing_validation_residual_postprocessing_NOT_native",
                    "scope": "supplied PV original-data fixed-subset inference audit; NOT GEFCom or main experiment",
                    "weights_sha256": before, "training_called": False})
                pred_rows.append(pd.DataFrame({"origin_ts": frame.ts.iloc[origins].to_numpy(),
                    "target_ts": frame.ts.iloc[future[origins, j]].to_numpy(), "horizon_minutes": h,
                    "actual_kw": y[:, j], "point_kw": point[:, j], "p10_kw": q[:, 0, j],
                    "p50_kw": q[:, 1, j], "p90_kw": q[:, 2, j]}))
            pd.concat(pred_rows).to_csv(args.output / f"{dataset}_inference_audit.csv", index=False)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != before: raise ValueError("frozen checkpoint changed")
            print(f"FROZEN_PV_INFERENCE_OK {dataset} {len(origins)} cases; no training", flush=True)
    pd.DataFrame(rows).to_csv(args.output / "pv_original_data_audit_metrics.csv", index=False)
    (args.output / "scope.json").write_text(json.dumps({"source_repo": str(args.pv_root),
        "datasets_joined": False, "GEFCom_PV_metrics": "unavailable; not transfer-compatible",
        "main_experiment": "not started", "subset_selection": "fixed evenly spaced 1000 original test cases, no score selection",
        "Cap": "report train-max statistical proxy, model original scale unmodified",
        "probabilities": "existing validation residual postprocessing, not native head",
        "model_change_or_training": False}, indent=2))


if __name__ == "__main__": main()
