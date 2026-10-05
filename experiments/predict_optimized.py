"""Load a released validation-selected ensemble and predict ordered features."""
from pathlib import Path

import joblib
import numpy as np


def predict_feature_matrices(bundle, matrices, feature_names, baseline_normalized):
    """Inputs follow the causal feature schema in the release manifest.

    This API accepts already computed features. It does not fetch observations
    or NWP and is not automatically registered as a backend forecasting model.
    """
    if feature_names != bundle["feature_names"] or len(matrices) != 3:
        raise ValueError("feature order differs from trained schema")
    base = np.asarray(baseline_normalized)
    if base.ndim != 2 or base.shape[1] != 3 or not np.isfinite(base).all():
        raise ValueError("baseline must be finite with shape (n, 3)")
    result = np.zeros_like(base, dtype=float)
    for j, matrix in enumerate(matrices):
        matrix = np.asarray(matrix)
        if matrix.shape != (len(base), len(feature_names[j])) or not np.isfinite(matrix).all():
            raise ValueError("invalid feature matrix")
        for name, weight in bundle["weights_per_horizon"][j].items():
            if name == "reference_baseline":
                prediction = base[:, j]
            else:
                prediction = bundle["estimators"][name][j].predict(matrix)
                if name == "lightgbm_residual":
                    prediction = prediction + base[:, j]
                prediction = np.clip(prediction, 0, 1)
            result[:, j] += weight * prediction
    cap = bundle["cap_kw"]
    result = np.clip(result, 0, 1) * cap
    quantiles = np.clip(result[:, None, :] + np.asarray(bundle["residual_quantiles_kw"])[None, :, :], 0, cap)
    return result, quantiles


def load_release(path):
    return joblib.load(Path(path))
