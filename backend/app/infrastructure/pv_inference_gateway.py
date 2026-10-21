"""PV 推理侧车网关客户端：主后端 → 侧车（.venv-infer 冻结环境）的 HTTP 桥。

stdlib urllib 实现，零新增依赖。侧车未运行/不可达抛 InferenceGatewayUnavailable，
由 API 层翻译成 503 PROVIDER_ERROR；契约违规（输入/模型 ID）抛 ValueError → 422。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8010"
DEFAULT_TIMEOUT = 30.0


class InferenceGatewayUnavailable(RuntimeError):
    """侧车进程不可达或内部故障。"""


def base_url() -> str:
    return os.getenv("PV_INFERENCE_URL", DEFAULT_BASE_URL).rstrip("/")


def _request(method: str, path: str, payload: dict | None = None,
             timeout: float = DEFAULT_TIMEOUT) -> dict:
    url = f"{base_url()}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", str(exc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = str(exc)
        if exc.code == 400:
            raise ValueError(detail) from exc
        raise InferenceGatewayUnavailable(f"推理侧车 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise InferenceGatewayUnavailable(
            f"PV 推理侧车不可达（{base_url()}）：{exc}；"
            "请用 .venv-infer 启动 backend/scripts/pv_inference_sidecar.py"
        ) from exc


def health() -> dict:
    return _request("GET", "/health", timeout=5.0)


def list_models() -> list[dict]:
    return _request("GET", "/models")["models"]


def predict(model_id: str, request: dict) -> dict:
    payload = {"model_id": model_id, **request}
    return _request("POST", "/predict", payload)


def unload(model_id: str) -> dict:
    return _request("POST", "/admin/unload", {"model_id": model_id})
