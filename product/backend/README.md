# Backend

当前后端是项目第一轮可运行闭环，使用 FastAPI + SQLite + Pydantic。

## 本地启动

```powershell
Set-Location backend
python -m unittest discover -s tests -p "test_*.py" -v
python run.py
```

接口文档：<http://127.0.0.1:8000/docs>

默认数据库：`backend/data/storage.db`。该文件不会提交到 Git。

## 当前接口

- `GET /api/v1/health`
- `POST /api/v1/sites`
- `GET /api/v1/sites`
- `GET /api/v1/sites/{site_id}`
- `PATCH /api/v1/sites/{site_id}`
- `POST /api/v1/sites/{site_id}/measurements`
- `GET /api/v1/sites/{site_id}/measurements?data_type=pv_power&date=2026-09-20`
- `POST /api/v1/tariffs`
- `GET /api/v1/tariffs`
- `POST /api/v1/storage/schedule`

## 当前调度规则

- 谷段充电。
- 光伏余电充电。
- 峰段和尖峰放电。
- SOC、PCS 功率、等效循环次数和禁止反送电约束生效。
- 当前只支持 15 分钟粒度和完整 24 小时序列。

## 目录

```text
app/
├── api/              # FastAPI 路由
├── domain/           # 数据模型、电价和调度领域逻辑
└── infrastructure/   # SQLite 数据库
tests/                # 单元测试和 API 流程测试
```
