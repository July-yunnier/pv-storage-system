"""FastAPI 接口层。

计划提供的接口（详见 docs/系统设计.md 第三节）

    POST/GET  /api/v1/sites                    站点管理
    POST      /api/v1/sites/{id}/measurements  上传实测数据
    GET       /api/v1/sites/{id}/weather       拉取气象预报（系统自动）
    POST      /api/v1/forecast                 统一预测入口（pv / load / price）
    POST      /api/v1/storage/sizing           储能容量规划
    POST      /api/v1/storage/schedule         日调度计划
    GET       /api/v1/reports/{id}             导出方案报告

启动
----
    uvicorn src.api.main:app --reload --port 8000
    接口文档自动生成： http://localhost:8000/docs
"""
