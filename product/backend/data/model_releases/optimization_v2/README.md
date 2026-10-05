# Validation-selected PV model releases

Ten bundles cover eight state-grid solar sites, SKIPPD power/solar features,
and the identical SKIPPD cases with causal sky descriptors. Each bundle
contains three horizon estimators/ensembles, ordered feature schemas,
train-only historical slot statistics and validation residual quantiles.

Read the Chinese report at `reports/optimization_v2/optimization_report.md`
and install its `requirements-lock.txt` for the tested dependency versions.
Verify each checkpoint against its adjacent JSON SHA256 manifest.

```python
from experiments.predict_optimized import load_release, predict_feature_matrices

bundle = load_release("product/backend/data/model_releases/optimization_v2/state_grid_1.joblib")
prediction_kw, quantiles_kw = predict_feature_matrices(
    bundle, matrices, feature_names, baseline_normalized
)
```

`matrices` contains three finite `(n, features)` arrays following the exact
per-horizon feature order in the manifest. `baseline_normalized` is `(n, 3)`.
Point predictions are `(n, 3)` in kW; quantiles are `(n, 3, 3)`, with axes
case, P10/P50/P90, and 15/60/240 minutes. The causal feature formulas are in
`experiments/optimize.py`; real-time feature generation must use stored
training slot statistics, documented capacity and observations available
at issuance. Recomputing slot statistics from production future data is invalid.

These bundles were verified after serialization on 1000 deterministic test
cases per dataset. They are offline releases, not automatically registered
with the backend model registry. Neither a global optimum nor universal
96% accuracy is claimed. See the report for all failures and daylight metrics.
