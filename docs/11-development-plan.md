# 从 0 开发计划

## 1. 当前状态

状态：未开始。当前只有产品设计稿和本仓库文档，没有代码、数据库或可运行服务。

## 2. 角色定义

以下先使用角色名，后续可替换为具体成员：

- `LEAD`：项目负责人，维护共享契约和最终验收。
- `BE`：后端开发，负责 API、数据库和应用服务。
- `OPT`：优化/算法开发，负责调度、Sizing 和经济模型。
- `FE`：前端开发，负责页面和图表。
- `QA`：测试与发布，负责测试和验收。

## 3. 任务清单

### P0-001 项目初始化

- Owner：LEAD
- 依赖：无
- 目录：根目录、`docs/`、`backend/`、`frontend/`、`configs/`
- 交付：README、环境说明、Docker Compose、代码目录约定。
- 验收：新环境可以安装依赖并启动健康检查。

### P0-002 数据契约冻结

- Owner：LEAD + BE
- 依赖：无
- 目录：`docs/04-domain-model.md`、`docs/05-api-contract.md`
- 交付：Pydantic DTO、统一错误结构、时间和单位规范。
- 验收：后端和前端可以只依赖契约开发，不依赖未公开字段。

### BE-001 站点管理

- Owner：BE
- 依赖：P0-002
- 目录：`backend/app/domain/site/`、`backend/app/api/sites.py`、对应测试。
- 交付：站点创建、查询、修改和持久化。
- 验收：接口返回符合契约，非法输入有明确错误。

### BE-002 测量数据

- Owner：BE
- 依赖：P0-002、BE-001
- 目录：`backend/app/domain/measurement/`、`backend/app/api/measurements.py`、对应测试。
- 交付：测量点上传、校验、查询和质量标记。
- 验收：支持四类数据，时间戳、粒度、单位和缺失点规则生效。

### BE-003 TOU Provider

- Owner：BE
- 依赖：P0-002
- 目录：`backend/app/domain/tariff/`、`backend/app/infrastructure/tariff/`、对应测试。
- 交付：TOU 模板和统一价格点。
- 验收：任意时间点匹配正确档位，边界无重叠或歧义。

### OPT-001 规则调度器

- Owner：OPT
- 依赖：P0-002、BE-002、BE-003
- 目录：`backend/app/domain/optimization/dispatch/`、对应测试。
- 交付：谷充峰放、SOC、PCS、禁止反送电和基础收益计算。
- 验收：全部硬约束通过，结果可复现，并有完整 15 分钟序列。

### FE-001 曲线看板

- Owner：FE
- 依赖：P0-002；可使用 Mock 数据并行开发。
- 目录：`frontend/src/features/dashboard/`、对应测试。
- 交付：总览和曲线页。
- 验收：正确展示阶梯电价、markArea、光伏、负荷、净负荷、储能和 SOC。

### P1-001 模拟数据和示例配置

- Owner：LEAD + QA
- 依赖：P0-002
- 目录：`configs/`、`data/fixtures/`
- 交付：示例站点、TOU 模板、24 小时和 72 小时测试数据。
- 验收：任何开发者可用示例数据重现调度结果。

### P2-001 预测基线

- Owner：OPT
- 依赖：BE-002、P1-001
- 目录：`backend/app/domain/forecast/`
- 交付：持久性/历史同期基线预测和统一预测结果格式。
- 验收：模型版本、时间范围、p10/p90 和 accuracy 字段符合契约。

### P2-002 天气 Provider

- Owner：BE
- 依赖：P0-002
- 目录：`backend/app/infrastructure/weather/`
- 交付：Mock Provider 和真实天气 API Adapter。
- 验收：外部服务失败时有可读错误，不影响核心领域单测。

### P3-001 Sizing 和经济性

- Owner：OPT
- 依赖：OPT-001、P2-001
- 目录：`backend/app/domain/optimization/sizing/`、`economics/`
- 交付：候选容量扫描、收益、NPV、IRR、回本周期、敏感性。
- 验收：结果附带完整 assumptions，和 Dispatch 使用同一口径。

### P3-002 方案页和报告

- Owner：FE + BE
- 依赖：P3-001
- 目录：`frontend/src/features/sizing/`、`backend/app/domain/report/`
- 交付：方案页、报告任务和 PDF 初版。
- 验收：报告和页面中的输入、结果、假设一致。

### QA-001 端到端验收

- Owner：QA
- 依赖：BE-001、BE-002、BE-003、OPT-001、FE-001
- 目录：`backend/tests/integration/`、`frontend/tests/`、`docs/09-test-acceptance.md`
- 交付：从建站到看板的自动化或可重复验收。
- 验收：干净环境、示例数据和 Docker 启动均通过。

## 4. 依赖关系

```text
P0-001 ─┬─ P0-002 ─┬─ BE-001 ─┬─ BE-002 ─┐
        │          └─ BE-003 ─┴─ OPT-001 ─┴─ FE-001
        └─ P1-001 ──────────────── P2-001 ── P3-001 ── P3-002
```

## 5. 文件归属规则

- `LEAD` 独占修改根目录配置、公共 DTO、数据库迁移规则和架构文档。
- `BE` 独占后端 API、应用服务和基础设施目录。
- `OPT` 独占优化和经济模型目录。
- `FE` 独占前端 feature 目录。
- `QA` 独占集成测试和验收脚本目录。
- 共享入口文件由 `LEAD` 负责合并，避免多人同时修改。

## 6. 进度状态

统一使用：`未开始`、`进行中`、`待评审`、`阻塞`、`已完成`。

任务只有在代码、测试、文档和交接说明都满足验收标准后，才能标记为“已完成”。

## 7. 首个开发迭代

首个迭代只追求：

```text
项目能启动
→ 能创建站点
→ 能上传示例数据
→ 能读取 TOU 电价
→ 能生成规则调度
→ 能在前端展示结果
```

在这条链路跑通前，不开始复杂预测、现货电价或随机优化。
