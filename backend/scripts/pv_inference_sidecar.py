"""PV 推理侧车：在冻结环境（.venv-infer，Python 3.12 + 锁定依赖）中托管 pv_inference。

平台侧（主后端 FastAPI）通过 HTTP 网关路由推理请求到本进程，实现
「模型注册表 + 网关路由」的部署形态：主进程不加载 lightgbm/sklearn，
版本由 pv_inference.manifest 的加载契约强校验（哈希/版本/schema 失败即拒绝）。

运行（仓库根目录）：

    .venv-infer/Scripts/python.exe backend/scripts/pv_inference_sidecar.py

环境变量：
    PV_INFERENCE_HOST   默认 127.0.0.1
    PV_INFERENCE_PORT   默认 8010
    PV_INFERENCE_MAX_MODELS  同时驻留模型数（LRU 驱逐），默认 2
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version as pkg_version
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from pv_inference import load_model  # noqa: E402
from pv_inference.infer import ROOT as RELEASE_ROOT  # noqa: E402

MODEL_ID_RE = re.compile(r"^[a-z0-9_]+$")
LOCK_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn",
    "lightgbm", "joblib", "threadpoolctl", "pvlib",
]

_MANIFEST = json.loads((RELEASE_ROOT / "manifest.json").read_text(encoding="utf-8"))


class ModelCache:
    """LRU 模型驻留：按需加载、超限驱逐、可显式卸载（动态加载/卸载验证用）。"""

    def __init__(self, max_models: int) -> None:
        self._max = max(1, max_models)
        self._lock = threading.Lock()
        self._models: OrderedDict[str, object] = OrderedDict()

    def get(self, model_id: str):
        with self._lock:
            model = self._models.get(model_id)
            if model is not None:
                self._models.move_to_end(model_id)
            return model

    def put(self, model_id: str, model) -> None:
        with self._lock:
            self._models[model_id] = model
            self._models.move_to_end(model_id)
            while len(self._models) > self._max:
                self._models.popitem(last=False)

    def unload(self, model_id: str) -> bool:
        with self._lock:
            return self._models.pop(model_id, None) is not None

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return list(self._models)


_CACHE = ModelCache(int(os.getenv("PV_INFERENCE_MAX_MODELS", "2")))


def _get_model(model_id: str):
    if not MODEL_ID_RE.match(model_id):
        raise ValueError(f"非法模型 ID: {model_id!r}")
    model = _CACHE.get(model_id)
    if model is None:
        model = load_model(model_id, root=RELEASE_ROOT, threads=1)  # 契约校验在内
        _CACHE.put(model_id, model)
    return model


def _registry_view() -> list[dict]:
    loaded = set(_CACHE.loaded_ids())
    view = []
    for model_id, m in _MANIFEST["models"].items():
        view.append({
            "model_id": model_id,
            "family": m.get("family"),
            "cap_kw": m["cap_kw"],
            "sha256": m["artifact"]["sha256"],
            "bytes": m["artifact"]["bytes"],
            "weights_unchanged": m.get("weights_unchanged"),
            "horizons_minutes": _MANIFEST["horizons_minutes"],
            "timezone": m.get("timezone"),
            "loaded": model_id in loaded,
            "validation": m.get("validation"),
        })
    return view


def _health() -> dict:
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": {p: pkg_version(p) for p in LOCK_PACKAGES},
        "manifest_state": _MANIFEST["state"],
        "format_version": _MANIFEST["format_version"],
        "models_registered": len(_MANIFEST["models"]),
        "models_loaded": _CACHE.loaded_ids(),
        "max_models": _CACHE._max,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音默认访问日志
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                self._send(200, _health())
            elif self.path == "/models":
                self._send(200, {"models": _registry_view()})
            else:
                self._send(404, {"error": f"unknown path: {self.path}"})
        except Exception as exc:  # 健康/注册表读取失败属平台故障
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/predict":
                request = self._read_json()
                model_id = request.pop("model_id", None)
                if not model_id:
                    raise ValueError("缺少 model_id")
                model = _get_model(model_id)
                result = model.predict(**request)
                self._send(200, result)
            elif self.path == "/admin/unload":
                request = self._read_json()
                model_id = request.get("model_id", "")
                unloaded = _CACHE.unload(model_id)
                self._send(200, {"model_id": model_id, "unloaded": unloaded,
                                 "models_loaded": _CACHE.loaded_ids()})
            else:
                self._send(404, {"error": f"unknown path: {self.path}"})
        except ValueError as exc:  # 输入/契约违规：调用方问题
            self._send(400, {"error": str(exc)})
        except RuntimeError as exc:  # 环境/版本/哈希拒绝：平台故障
            self._send(500, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    host = os.getenv("PV_INFERENCE_HOST", "127.0.0.1")
    port = int(os.getenv("PV_INFERENCE_PORT", "8010"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"pv-inference sidecar on http://{host}:{port} "
          f"(models_registered={len(_MANIFEST['models'])}, max_loaded={_CACHE._max})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
