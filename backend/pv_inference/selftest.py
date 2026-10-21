"""Verification only: hash, deserialization and real held-out inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from .features import image_descriptors
from .infer import ROOT, check_versions, load_model, verify_file


def _must_reject(operation, exception=ValueError):
    try:
        operation()
    except exception:
        return True
    raise AssertionError("invalid input was accepted")


def test_one(model_id, root):
    started = time.perf_counter()
    model = load_model(model_id, root=root)
    load_seconds = time.perf_counter()-started
    contract = model.model
    fixture = contract["verification_fixture"]
    fixture_path = root/fixture["path"]
    verify_file(fixture_path, fixture["sha256"], fixture["bytes"])
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    differences, times = [], []
    for case in cases:
        started = time.perf_counter()
        output = model.predict(**case["request"])
        times.append(time.perf_counter()-started)
        values = np.array([[v[k] for k in ("predicted_kw", "p10_kw", "p50_kw", "p90_kw")]
                           for v in output["forecasts"]])
        expected = np.asarray(case["expected_kw"])
        np.testing.assert_allclose(values, expected, rtol=0, atol=contract["verification_tolerance_kw"])
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= contract["cap_kw"]).all()
        assert (np.diff(values[:, 1:], axis=1) >= 0).all()
        assert [v["horizon_minutes"] for v in output["forecasts"]] == [15, 60, 240]
        assert [v["target_at"] for v in output["forecasts"]] == case["expected_target_at"]
        differences.append(float(np.abs(values-expected).max()))
    first = cases[0]["request"]
    corrupted = json.loads(json.dumps(first))
    corrupted["observations"][-1]["power_kw"] = float("nan")
    assert _must_reject(lambda: model.predict(**corrupted))
    corrupted = json.loads(json.dumps(first))
    corrupted["observations"][-1]["ts"] = corrupted["observations"][-2]["ts"]
    assert _must_reject(lambda: model.predict(**corrupted))
    corrupted = json.loads(json.dumps(first))
    for observation in corrupted["observations"]:
        observation["available_at"] = observation["ts"]
    corrupted["observations"][-1]["available_at"] = "2100-01-01T00:00:00+00:00"
    assert _must_reject(lambda: model.predict(**corrupted))
    names = [list(v) for v in contract["feature_schema"]["names_per_horizon"]]
    names[0][0] = "wrong_schema"
    matrices = [np.zeros((1, len(n)), np.float32) for n in names]
    assert _must_reject(lambda: model.predict_matrices(matrices, names, np.zeros((1, 3))))
    artifact = root/contract["artifact"]["path"]
    assert _must_reject(lambda: verify_file(artifact, "0"*64, artifact.stat().st_size))
    # Exercise the actual loader; hash rejection precedes joblib deserialization.
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        contract_copy = json.loads(json.dumps(model.manifest))
        model_copy = contract_copy["models"][model_id]
        model_copy["artifact"]["path"] = "corrupted.joblib"
        model_copy["artifact"]["bytes"] = 1
        (temp_root/"corrupted.joblib").write_bytes(b"x")
        (temp_root/"manifest.json").write_text(json.dumps(contract_copy))
        assert _must_reject(lambda: load_model(model_id, temp_root))
    names = contract["feature_schema"]["names_per_horizon"]
    assert _must_reject(lambda: model.predict_matrices([np.zeros((1,len(n)), np.float64) for n in names],
                                                       names, np.zeros((1,3))))
    wrong_env = json.loads(json.dumps(model.manifest["environment"]))
    wrong_env["packages"]["numpy"] = "0.0.0"
    assert _must_reject(lambda: check_versions(wrong_env), RuntimeError)
    bitmap_result = None
    if contract["images"]:
        bitmaps = contract["bitmap_fixture"]
        file = root/bitmaps["path"]
        verify_file(file, bitmaps["sha256"], bitmaps["bytes"])
        with np.load(file, allow_pickle=False) as archive:
            images = archive["images"]
        descriptors = image_descriptors(images)
        np.testing.assert_allclose(descriptors, first["descriptors"], atol=5e-6, rtol=0)
        request = dict(first)
        request.pop("descriptors")
        output = model.predict(**request, images=images)
        point = [v["predicted_kw"] for v in output["forecasts"]]
        np.testing.assert_allclose(point, np.asarray(cases[0]["expected_kw"])[:, 0],
                                   atol=contract["verification_tolerance_kw"], rtol=0)
        bitmap_result = {"frames": 30, "resolution": [64, 64], "passed": True}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss/(1024**2 if sys.platform == "darwin" else 1024)
    return dict(model_id=model_id, passed=True, checked_cases=len(cases),
                hash_before_load_verified=True, model_sha256=contract["artifact"]["sha256"],
                max_abs_error_kw=max(differences), tolerance_kw=contract["verification_tolerance_kw"],
                load_seconds=load_seconds, inference_seconds=times, peak_process_rss_mb=rss_mb,
                invalid_input_checks=["nonfinite", "timestamps/gaps", "future availability", "feature schema", "hash", "package version"],
                bitmap_inference=bitmap_result)


def main():
    parser = argparse.ArgumentParser(description="Self-test frozen PV release, no fitting")
    parser.add_argument("--release-root", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model:
        result = test_one(args.model, args.release_root)
    else:
        manifest = json.loads((args.release_root/"manifest.json").read_text(encoding="utf-8"))
        results = []
        for model_id in manifest["models"]:
            process = subprocess.run([sys.executable, "-m", "pv_inference.selftest", "--model", model_id,
                                      "--release-root", str(args.release_root)], capture_output=True, text=True,
                                     timeout=180, check=True)
            results.append(json.loads(process.stdout))
            print(f"verified {model_id}", file=sys.stderr, flush=True)
        result = dict(passed=all(r["passed"] for r in results), model_count=len(results),
                      python=sys.version, platform=platform.platform(), threads=1,
                      models=results, training_performed=False)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
