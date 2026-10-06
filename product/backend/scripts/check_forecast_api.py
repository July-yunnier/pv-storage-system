"""Exercise the running API using persisted, independently fitted artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    results = pd.read_csv(args.output/"results.csv")
    checks = []
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        for (task, horizon), frame in results[(results.status == "complete") & (results.variant == "full")].groupby(["task", "horizon"]):
            best = frame.sort_values("validation_RMSE").iloc[0]
            experiment = args.output/task/f"{horizon}h"/f"{best.model}__full"
            artifact = json.loads((experiment/"weights/artifact.json").read_text())
            prediction = pd.read_csv(experiment/"predictions.csv").iloc[0]
            source = pd.read_csv(args.output/task/"raw.csv")
            source["ts"] = pd.to_datetime(source.ts, utc=True)
            history = source[source.ts <= pd.Timestamp(prediction.forecast_origin_ts)].tail(169)
            request = {"task": task, "model": best.model, "horizon_hours": int(horizon),
                       "unit": artifact["unit"], "forecast_origin": prediction.forecast_origin_ts,
                       "history": [{"ts": r.ts.isoformat(), "value": getattr(r, task)} for r in history.itertuples()],
                       "historical_exogenous": {name: history[name].tolist() for name in artifact["exog_names"]}}
            response = client.post("/api/v1/forecast/ml", json=request)
            response.raise_for_status()
            value = response.json()["results"][0]["points"][0]["value"]
            difference = abs(value-prediction.y_pred)
            rejected = client.post("/api/v1/forecast/ml", json={**request, "unit": "incompatible_unit"})
            checks.append({"task": task, "horizon": int(horizon), "model": best.model,
                           "http_status": response.status_code, "difference": difference,
                           "wrong_unit_status": rejected.status_code,
                           "passed": bool(difference < 1e-3 and rejected.status_code == 422)})
    (args.output/"api_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
