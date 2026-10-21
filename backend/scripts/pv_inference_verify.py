"""冻结 PV 推理包的跨平台验证（Windows 兼容版 selftest）。

与 backend/pv_inference/selftest.py 的检查项一一对应（哈希先于反序列化、
持有样本推理对账、非法输入拒绝、schema/版本/哈希篡改拒绝），仅将 POSIX
专属的 resource.getrusage 换成可选项；冻结包本体不做任何修改。

运行（.venv-infer）：
    .venv-infer/Scripts/python.exe backend/scripts/pv_inference_verify.py
    .venv-infer/Scripts/python.exe backend/scripts/pv_inference_verify.py --model state_grid_1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np  # noqa: E402

from pv_inference import load_model  # noqa: E402
from pv_inference.features import image_descriptors  # noqa: E402
from pv_inference.infer import ROOT, check_versions, verify_file  # noqa: E402


def _must_reject(operation, exception=ValueError) -> bool:
    try:
        operation()
    except exception:
        return True
    raise AssertionError("invalid input was accepted")


def _peak_rss_mb() -> float | None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024**2 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None  # Windows：无 resource 模块，RSS 留空


def test_one(model_id: str, root: Path) -> dict:
    started = time.perf_counter()
    model = load_model(model_id, root=root)
    load_seconds = time.perf_counter() - started
    contract = model.model
    fixture = contract["verification_fixture"]
    fixture_path = root / fixture["path"]
    verify_file(fixture_path, fixture["sha256"], fixture["bytes"])
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    differences, times = [], []
    for case in cases:
        started = time.perf_counter()
        output = model.predict(**case["request"])
        times.append(time.perf_counter() - started)
        values = np.array([[v[k] for k in ("predicted_kw", "p10_kw", "p50_kw", "p90_kw")]
                           for v in output["forecasts"]])
        expected = np.asarray(case["expected_kw"])
        np.testing.assert_allclose(values, expected, rtol=0,
                                   atol=contract["verification_tolerance_kw"])
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= contract["cap_kw"]).all()
        assert (np.diff(values[:, 1:], axis=1) >= 0).all()
        assert [v["horizon_minutes"] for v in output["forecasts"]] == [15, 60, 240]
        assert [v["target_at"] for v in output["forecasts"]] == case["expected_target_at"]
        differences.append(float(np.abs(values - expected).max()))

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
    artifact = root / contract["artifact"]["path"]
    assert _must_reject(lambda: verify_file(artifact, "0" * 64, artifact.stat().st_size))
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        contract_copy = json.loads(json.dumps(model.manifest))
        model_copy = contract_copy["models"][model_id]
        model_copy["artifact"]["path"] = "corrupted.joblib"
        model_copy["artifact"]["bytes"] = 1
        (temp_root / "corrupted.joblib").write_bytes(b"x")
        (temp_root / "manifest.json").write_text(json.dumps(contract_copy))
        assert _must_reject(lambda: load_model(model_id, temp_root))
    names = contract["feature_schema"]["names_per_horizon"]
    assert _must_reject(lambda: model.predict_matrices(
        [np.zeros((1, len(n)), np.float64) for n in names], names, np.zeros((1, 3))))
    wrong_env = json.loads(json.dumps(model.manifest["environment"]))
    wrong_env["packages"]["numpy"] = "0.0.0"
    assert _must_reject(lambda: check_versions(wrong_env), RuntimeError)

    bitmap_result = None
    if contract["images"]:
        bitmaps = contract["bitmap_fixture"]
        file = root / bitmaps["path"]
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

    return dict(model_id=model_id, passed=True, checked_cases=len(cases),
                hash_before_load_verified=True,
                model_sha256=contract["artifact"]["sha256"],
                max_abs_error_kw=max(differences),
                tolerance_kw=contract["verification_tolerance_kw"],
                load_seconds=load_seconds, inference_seconds=times,
                peak_process_rss_mb=_peak_rss_mb(),
                invalid_input_checks=["nonfinite", "timestamps/gaps", "future availability",
                                      "feature schema", "hash", "package version"],
                bitmap_inference=bitmap_result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen PV release (Windows-compatible)")
    parser.add_argument("--release-root", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.release_root / "manifest.json").read_text(encoding="utf-8"))
    model_ids = [args.model] if args.model else list(manifest["models"])
    results = [test_one(model_id, args.release_root) for model_id in model_ids]
    summary = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "passed": all(r["passed"] for r in results),
        "models": results,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
