"""AI 模型工件存储：加载、缓存与可用性探测。

领域层不感知文件路径与环境变量；工件（.pt）加载后缓存复用，
避免每次预测重复读盘反序列化。
"""

from __future__ import annotations

import os
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


def load_model_unavailable_reason(path: str | None = None) -> str | None:
    """负荷高级模型可用性探测：返回 None 表示可用，否则返回原因（供注册表/前端展示）。"""
    if _torch() is None:
        return "未安装 PyTorch（pip install -r requirements-ai.txt）"
    resolved = path or default_load_model_path()
    if not resolved:
        return "未配置 LOAD_MODEL_PATH"
    if not Path(resolved).is_file():
        return f"模型工件不存在: {resolved}"
    return None
