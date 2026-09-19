# 技术架构

## 1. 技术基线

| 层 | 首选方案 | 说明 |
|---|---|---|
| 后端 | Python + FastAPI | 快速构建 REST 和 OpenAPI |
| 数据校验 | Pydantic | 与 API 和领域模型一致 |
| 数据库 | SQLite → PostgreSQL | MVP 零配置，后续可升级 |
| 优化 | 规则式 → scipy/PuLP LP | 按复杂度逐步升级 |
| 前端 | React + Vite + ECharts | 适合阶梯图、markArea 和 dataZoom |
| 部署 | Docker Compose | 单机演示和本地开发 |
| 测试 | pytest + 前端测试工具 | 领域、API、端到端分层测试 |

## 2. 分层架构

```text
Frontend
   ↓ HTTP/JSON
API Layer
   ↓
Application Services
   ↓
Domain Services
   ├── TariffProvider
   ├── ForecastProvider
   ├── DispatchEngine
   ├── SizingEngine
   └── EconomicsEngine
   ↓
Repositories / Adapters
   ├── SQLite/PostgreSQL
   ├── Weather API
   └── CSV/SCADA input
```

领域计算不能依赖 FastAPI Request、数据库 Session 或前端对象，便于测试和未来替换基础设施。

## 3. 后端目录建议

```text
backend/app/
├── api/                 # 路由和 HTTP DTO
├── domain/              # 领域模型和纯业务计算
│   ├── site/
│   ├── tariff/
│   ├── measurement/
│   ├── forecast/
│   ├── optimization/
│   └── economics/
├── application/         # 用例服务
├── infrastructure/      # 数据库、外部 API、文件适配器
├── config.py
└── main.py
```

## 4. Provider 设计

优化器依赖抽象接口，不直接依赖外部服务：

```text
TariffProvider
WeatherProvider
ForecastProvider
OptimizationStrategy
```

首版至少实现：

- `TouProvider`
- `MockWeatherProvider`
- `CsvMeasurementProvider`
- `RuleDispatchStrategy`

## 5. 数据流和任务

MVP 的站点查询、上传和调度可以同步执行。预测、Sizing 和 PDF 报告后续应设计为任务型接口，以便支持耗时计算和重试。

## 6. 错误处理

错误分为：

- `VALIDATION_ERROR`：请求字段或数据点不合法
- `NOT_FOUND`：站点、模板或任务不存在
- `DATA_QUALITY_ERROR`：数据不足、缺失或异常
- `OPTIMIZATION_ERROR`：无法找到可行解
- `PROVIDER_ERROR`：外部天气/价格源失败
- `INTERNAL_ERROR`：未预期错误

## 7. 安全和审计

首版可以不实现完整用户体系，但代码结构不能把认证和授权写死在业务逻辑中。外部 API Key 使用环境变量，日志不得输出密钥、完整上传数据或用户敏感信息。

## 8. 可观测性

每个请求和计算任务至少记录：

- request_id
- site_id
- provider/model/optimizer version
- start_time、end_time
- status
- failure code

## 9. 部署验收

```text
docker compose up
→ API health check 通过
→ OpenAPI 页面可访问
→ 前端可访问
→ 示例站点可创建
→ 示例调度可执行
```
