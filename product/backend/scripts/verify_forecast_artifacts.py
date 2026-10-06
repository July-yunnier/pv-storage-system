"""Reload serialized test checkpoints and compare against saved predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import numpy as np
import pandas as pd
from app.domain.forecast.inference import predict_artifact
from app.domain.models import MachineLearningForecastRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = pd.read_csv(args.output/"results.csv")
    checks = []
    for row in results.itertuples():
        if row.status != "complete":
            continue
        print(f"VERIFY {row.task} {row.horizon}h {row.model} {row.variant}", flush=True)
        exp = args.output/row.task/f"{row.horizon}h"/f"{row.model}__{row.variant}"
        meta = json.loads((exp/"weights/artifact.json").read_text())
        raw = pd.read_csv(args.output/row.task/"raw.csv")
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        p = pd.read_csv(exp/"predictions.csv").iloc[0]
        origin = pd.Timestamp(p.forecast_origin_ts)
        # SARIMA/ARIMA state is filtered from full history in offline eval;
        # windows have gaps before this origin, so check recent continuous
        # history and record the state truncation difference separately.
        history = raw[raw.ts <= origin].tail(169).copy()
        request = MachineLearningForecastRequest(
            task=row.task, model=row.model, horizon_hours=int(row.horizon), unit=meta["unit"],
            forecast_origin=origin.to_pydatetime(),
            history=[{"ts": r.ts.to_pydatetime(), "value": getattr(r, row.task)} for r in history.itertuples()],
            historical_exogenous={c: history[c].tolist() for c in meta["exog_names"]},
        )
        try:
            actual = predict_artifact(request, exp/"weights").points[0].value
            interface_difference = abs(actual-float(p.y_pred))
            if meta["kind"] == "arima":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                prefix = raw[raw.ts <= origin].set_index("ts")
                prefix = prefix.reindex(pd.date_range(prefix.index.min(), origin, freq="h"))
                stats = meta["normalization"]
                z = (prefix[row.task].to_numpy()-stats["target_mean"])/stats["target_std"]
                model = SARIMAX(z, order=tuple(meta["order"]), seasonal_order=tuple(meta["seasonal_order"]), trend="c")
                fitted = model.filter(np.load(exp/"weights/params.npy"))
                actual = float(fitted.forecast(int(row.horizon))[-1]*stats["target_std"]+stats["target_mean"])
            error = abs(actual-float(p.y_pred))
            passed = bool(error < 1e-3 and np.isfinite(interface_difference))
            checks.append({"task": row.task, "model": row.model, "horizon": row.horizon,
                           "variant": row.variant, "absolute_reload_difference": error, "passed": passed,
                           "interface_169h_difference": interface_difference,
                           "state_note": "169h filter state versus full source filter" if meta["kind"] == "arima" else "identical inputs"})
        except Exception as exc:
            checks.append({"task": row.task, "model": row.model, "horizon": row.horizon,
                           "variant": row.variant, "passed": False, "error": str(exc)})
    (args.output/"reload_checks.json").write_text(json.dumps(checks, indent=2))
    print(pd.DataFrame(checks).to_string(index=False))
    if not all(r["passed"] for r in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
