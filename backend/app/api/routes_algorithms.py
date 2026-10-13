"""算法发现接口：暴露算法注册表目录（名称/对象/类型/依赖/可用性）。

前端与协作方（如下游算法同学）用它查询系统当前可用的预测算法，
不必翻源码；pv_model/load_model 的合法取值也从这里的 product_key 读。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.application import algorithm_registry

router = APIRouter(prefix="/algorithms", tags=["algorithms"])


@router.get("")
def list_algorithms() -> dict:
    return {"algorithms": algorithm_registry.catalog_view()}
