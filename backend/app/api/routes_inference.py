"""推理网关路由：模型注册表视图 + PV 推理路由 + 动态卸载（平台运维面）。

侧车在 .venv-infer 冻结环境中托管 pv_inference（hash/版本/schema 强校验），
主后端仅做网关转发，不加载 sklearn/lightgbm；侧车未启动时统一 502。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.api import errors
from app.infrastructure import pv_inference_gateway as gateway

router = APIRouter(prefix="/inference", tags=["inference"])


class PVInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # images/image_timestamps 等透传给冻结契约

    issued_at: str = Field(min_length=1)
    observations: list[dict] = Field(min_length=1)


def _gateway_call(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    except gateway.InferenceGatewayUnavailable as exc:
        raise errors.provider_error(str(exc)) from exc


@router.get("/health")
def inference_health() -> dict:
    """侧车健康 + 冻结环境版本（运维核查：版本不匹配会在此暴露）。"""
    return _gateway_call(gateway.health)


@router.get("/models")
def inference_models() -> dict:
    """模型注册表：id/容量/SHA256/验证指标/当前是否驻留。"""
    return {"models": _gateway_call(gateway.list_models)}


@router.post("/pv/{model_id}")
def predict_pv(model_id: str, request: PVInferenceRequest) -> dict:
    """路由推理请求到冻结模型，返回 15/60/240 分钟 P10/P50/P90（kW）。"""
    return _gateway_call(gateway.predict, model_id, request.model_dump())


@router.post("/models/{model_id}/unload")
def unload_model(model_id: str) -> dict:
    """卸载驻留模型（动态加载/卸载验证用，平台内部操作）。"""
    return _gateway_call(gateway.unload, model_id)
