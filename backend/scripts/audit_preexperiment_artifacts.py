"""Read-only audit of saved pre-experiment predictions and selection evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.domain.forecast.quantile import pinball_numpy


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    root = args.output
    plan = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection_frozen.json").read_text())
    assert plan["scope"].startswith("GEFCom exploratory")
    assert selection["fingerprint"] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    n = 0
    for task in plan["tasks"]:
        table = pd.read_csv(root / f"{task}_results.csv")
        if task == "price": assert not any("Cap" in c or "96" in c for c in table.columns)
        for horizon in plan["horizons_hours"]:
            split = pd.read_csv(root / task / f"{horizon}h/split_manifest.csv", parse_dates=["target_ts", "forecast_origin_ts"])
            tr, va, te = (split[split.split == s] for s in ["train", "validation", "test"])
            assert tr.target_ts.max() < va.forecast_origin_ts.min()
            assert va.target_ts.max() < te.forecast_origin_ts.min()
            for label in table.loc[table.horizon == horizon, "variant"]:
                exp = root / task / f"{horizon}h/evaluation" / label
                m = json.loads((exp / "metrics.json").read_text())
                pred = pd.read_csv(exp / "predictions.csv", parse_dates=["forecast_origin_ts", "target_ts"])
                assert len(pred) == len(te) == m["n"]
                assert np.array_equal(pred.target_ts.values, te.target_ts.values)
                assert np.isfinite(pred[["y_true", "p10", "p50", "p90"]]).all().all()
                e = pred.p50 - pred.y_true
                assert np.isclose(e.abs().mean(), m["MAE"])
                assert np.isclose(np.sqrt((e**2).mean()), m["RMSE"])
                assert np.isclose(pinball_numpy(pred.y_true, pred[["p10", "p50", "p90"]]), m["Pinball"])
                coverage = ((pred.y_true >= pred.p10) & (pred.y_true <= pred.p90)).mean()
                assert np.isclose(coverage, m["interval_coverage_80"])
                if m["interval_type"] == "native_pinball_head":
                    assert (pred.p10 <= pred.p50).all() and (pred.p50 <= pred.p90).all()
                if task == "price": assert not any("Cap" in c for c in m)
                assert m["historical_test_exposure"] and m["scope"] == "pre-experiment ONLY"
                n += 1
    for item in selection["selected"]:
        for entry in item["entries"]:
            exp = Path(entry["path"]); info = entry["info"]
            assert hashlib.sha256((exp / "model.pt").read_bytes()).hexdigest() == info["weight_sha256"]
            logs = json.loads((exp / "training_log.json").read_text())
            assert len(logs) == info["epochs_completed"]
            assert np.isclose(logs[info["best_epoch"] - 1]["validation_loss"], info["best_validation_loss"])
            assert info["stop_reason"] in {"validation_early_stop", "max_budget_not_converged"}
    summary = {"status": "passed", "saved_evaluations_checked": n,
        "checks": ["chronological split embargo", "saved prediction metrics recomputed",
            "matching target timestamps and sample counts", "native quantile ordering",
            "no price Cap metrics", "frozen protocol fingerprint",
            "selected weight SHA256 and best-epoch validation loss"],
        "scope": "GEFCom exploratory pre-experiments only; historical test exposure remains",
        "pv_audit": "separate original-data inference audit; not GEFCom PV evaluation",
        "main_experiment": "not started"}
    (root / "artifact_audit.json").write_text(json.dumps(summary, indent=2))
    print(f"AUDIT PASSED {n} saved evaluations; scope remains exploratory; PV audit kept separate")


if __name__ == "__main__": main()
