"""AI 模型工件存储：加载、缓存与可用性探测。

领域层不感知文件路径与环境变量；工件（.pt）加载后缓存复用，
避免每次预测重复读盘反序列化。
"""

from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelBundle:
    """已加载的模型工件：state_dict + 元数据（特征清单/标准化参数/标签映射）。"""

    path: str
    state_dict: Any
    metadata: dict[str, Any]


def default_load_model_path() -> str:
    """负荷高级模型工件路径（LOAD_MODEL_PATH 环境变量）。"""
    return os.getenv("LOAD_MODEL_PATH", "").strip()


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


@lru_cache(maxsize=4)
def load_bundle(path: str) -> ModelBundle:
    """按路径加载并缓存工件。torch 缺失抛 RuntimeError，路径无效抛 ValueError。"""
    torch = _torch()
    if torch is None:
        raise RuntimeError("高级负荷模型需要 PyTorch，请安装 requirements-ai.txt")
    p = Path(path)
    if not str(path) or not p.is_file():
        raise ValueError(f"模型工件不存在或未配置: {path!r}")
    artifact = torch.load(p, map_location="cpu", weights_only=True)
    return ModelBundle(
        path=str(p), state_dict=artifact["state_dict"], metadata=artifact["metadata"]
    )


def load_model_unavailable_reason(path: str | None = None, model: str = "load") -> str | None:
    """Return an availability reason for a known model family.

    ``model='ampdnet'`` checks the frozen AMPDNet release; the default keeps
    the legacy LOAD_MODEL_PATH behavior for advanced load forecasting.
    """
    if model == "ampdnet":
        return load_ampdnet_unavailable_reason()
    if _torch() is None:
        return "未安装 PyTorch（pip install -r requirements-ai.txt）"
    resolved = path or default_load_model_path()
    if not resolved:
        return "未配置 LOAD_MODEL_PATH"
    if not Path(resolved).is_file():
        return f"模型工件不存在: {resolved}"
    return None


AMPDNET_RELEASE = Path(__file__).resolve().parents[2] / "data/model_releases/ampdnet_delivery20"


def default_ampdnet_release_path() -> Path:
    return Path(os.getenv("AMPDNET_MODEL_DIR", str(AMPDNET_RELEASE)))


@lru_cache(maxsize=32)
def load_ampdnet_artifact(station: str, task: str, release_dir: str | None = None):
    """Load and verify one TorchScript AMPDNet release on CPU."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("AMPDNet requires PyTorch; install requirements-ai.txt")
    root = Path(release_dir) if release_dir else default_ampdnet_release_path()
    tag = f"{task}_{station}_processed_3_full_tr0.6_val0.2_s42"
    metadata_path = root / f"{tag}.json"
    if not metadata_path.is_file():
        raise ValueError(f"AMPDNet metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    artifact = root / metadata["artifact_file"]
    if not artifact.is_file():
        raise ValueError(f"AMPDNet artifact not found: {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != metadata.get("artifact_sha256"):
        raise RuntimeError("AMPDNet model SHA256 mismatch")
    model = torch.jit.load(str(artifact), map_location="cpu")
    model.eval()
    return model, metadata


def load_ampdnet_unavailable_reason() -> str | None:
    if _torch() is None:
        return "未安装 PyTorch（pip install -r requirements-ai.txt）"
    root = default_ampdnet_release_path()
    required = root / "15min_Solar1_processed_3_full_tr0.6_val0.2_s42.json"
    if not required.is_file() or not (root / "15min_Solar1_processed_3_full_tr0.6_val0.2_s42.torchscript.pt").is_file():
        return f"AMPDNet 工件目录不存在或未配置: {root}"
    return None
