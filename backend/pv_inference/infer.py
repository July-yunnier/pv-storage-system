"""Hash-checked loading and inference; no training/search dependencies."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .features import build_features

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes release root")
    return path


def verify_file(path, expected_hash, expected_bytes):
    if Path(path).stat().st_size != expected_bytes or sha256(path) != expected_hash:
        raise ValueError(f"artifact hash/size mismatch: {Path(path).name}")


def check_versions(contract):
    if list(sys.version_info[:2]) != contract["python_major_minor"]:
        raise RuntimeError("Python version does not match frozen loading contract")
    for package, expected in contract["packages"].items():
        found = version(package)
        if found != expected:
            raise RuntimeError(f"package version mismatch: {package}: expected {expected}, got {found}")


class FrozenPVModel:
    def __init__(self, manifest, model, bundle, threads):
        self.manifest, self.model = manifest, model
        self._bundle, self.threads = bundle, threads

    def predict_matrices(self, matrices, feature_names, baseline_normalized):
        """Advanced interface for upstream producers of the exact frozen schema."""
        schemas = self.model["feature_schema"]["names_per_horizon"]
        if feature_names != schemas or len(matrices) != 3:
            raise ValueError("feature schema/order differs from frozen contract")
        base = np.asarray(baseline_normalized)
        if base.ndim != 2 or base.shape[1] != 3 or not 1 <= len(base) <= self.manifest["runtime"]["max_batch_cases"]:
            raise ValueError("baseline shape or batch size violates contract")
        if not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("baseline must be finite, nonnegative and normalized by frozen Cap")
        output = np.zeros_like(base, dtype=np.float64)
        with threadpool_limits(limits=self.threads):
            for j, matrix in enumerate(matrices):
                matrix = np.asarray(matrix)
                if matrix.dtype != np.float32 or matrix.shape != (len(base), len(schemas[j])) or not np.isfinite(matrix).all():
                    raise ValueError("feature matrices must be finite float32 arrays with the exact schema")
                for name, weight in self._bundle["weights_per_horizon"][j].items():
                    if name == "reference_baseline":
                        prediction = base[:, j]
                    else:
                        prediction = self._bundle["estimators"][name][j].predict(matrix)
                        if name == "lightgbm_residual":
                            prediction = prediction+base[:, j]
                        prediction = np.clip(prediction, 0, 1)
                    output[:, j] += weight*prediction
        cap = self.model["cap_kw"]
        output = np.clip(output, 0, 1)*cap
        quantiles = np.clip(output[:, None, :]+np.asarray(self._bundle["residual_quantiles_kw"])[None, :, :], 0, cap)
        if not np.isfinite(output).all() or not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
            raise RuntimeError("model output violates finite/quantile contract")
        return output, quantiles

    def predict(self, observations, issued_at, **image_inputs):
        matrices, schemas, baseline, targets = build_features(observations, issued_at, self.model, self._bundle, **image_inputs)
        point, q = self.predict_matrices(matrices, schemas, baseline)
        return {"model_id": self.model["id"], "model_sha256": self.model["artifact"]["sha256"],
                "issued_at": str(issued_at), "units": "kW", "forecasts": [
                    {"horizon_minutes": h, "target_at": t.isoformat(), "predicted_kw": float(point[0, j]),
                     "p10_kw": float(q[0, 0, j]), "p50_kw": float(q[0, 1, j]), "p90_kw": float(q[0, 2, j])}
                    for j, (h, t) in enumerate(zip(self.manifest["horizons_minutes"], targets))]}


def load_model(model_id, root=ROOT, threads=1):
    """Use only an independently trusted release root/manifest; joblib is pickle."""
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or manifest["state"] != "frozen" or manifest["training_enabled"]:
        raise ValueError("release is not a supported frozen inference contract")
    if not isinstance(threads, int) or not 1 <= threads <= manifest["runtime"]["max_threads"]:
        raise ValueError("invalid thread budget")
    check_versions(manifest["environment"])
    if model_id not in manifest["models"]:
        raise ValueError(f"unknown model id: {model_id}")
    model = manifest["models"][model_id]
    schema = model["feature_schema"]
    calculated_schema_hash = hashlib.sha256(json.dumps(schema["names_per_horizon"], separators=(",", ":")).encode()).hexdigest()
    if calculated_schema_hash != schema["sha256"] or schema["counts"] != [len(n) for n in schema["names_per_horizon"]]:
        raise ValueError("feature schema hash/count mismatch")
    artifact = model["artifact"]
    path = _local_file(root, artifact["path"])
    # Integrity is verified BEFORE deserialization can execute pickle code.
    verify_file(path, artifact["sha256"], artifact["bytes"])
    with threadpool_limits(limits=threads):
        bundle = joblib.load(path)
    if (bundle["dataset"] != model_id or bundle["cap_kw"] != model["cap_kw"]
            or bundle["horizons_minutes"] != manifest["horizons_minutes"]
            or bundle["feature_names"] != model["feature_schema"]["names_per_horizon"]):
        raise ValueError("checkpoint metadata disagrees with manifest")
    if bundle["weights_per_horizon"] != model["weights_per_horizon"]:
        raise ValueError("checkpoint ensemble disagrees with manifest")
    for j, weights in enumerate(bundle["weights_per_horizon"]):
        w = np.asarray(list(weights.values()))
        if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1., atol=1e-7):
            raise ValueError("invalid ensemble weights")
        for name in weights:
            if name != "reference_baseline":
                estimator = bundle["estimators"][name][j]
                if estimator.n_features_in_ != len(bundle["feature_names"][j]):
                    raise ValueError("estimator feature count differs")
                params = estimator.get_params()
                if "n_jobs" in params:
                    estimator.set_params(n_jobs=threads)
    residual = np.asarray(bundle["residual_quantiles_kw"])
    if residual.shape != (3, 3) or not np.isfinite(residual).all() or (np.diff(residual, axis=0) < 0).any():
        raise ValueError("invalid frozen residual quantiles")
    return FrozenPVModel(manifest, model, bundle, threads)
