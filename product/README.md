# 光储一体 · 储能配置优化系统

面向工商业用户的光伏发电预测、负荷预测、电价分析、储能容量配置与日充放电策略系统。

> 当前状态：从 0 开始建设，现阶段以产品与技术文档为主，尚未进入代码实现。

## 项目目标

系统围绕同一个优化内核提供两类能力：

- **规划（Sizing）**：根据站点、光伏、负荷、电价和投资假设，推荐储能功率与容量，并测算收益、回本周期、IRR 和 NPV。
- **运行（Dispatch）**：根据次日预测、价格曲线和当前 SOC，生成 15 分钟粒度的充放电计划。

系统明确不负责储能电气设计、电力交易申报和现场 EMS 实时控制。

## 文档导航

- [文档索引](docs/00-document-index.md)
- [产品需求文档 PRD](docs/01-prd.md)
- [范围与路线图](docs/02-scope-roadmap.md)
- [用户故事与验收标准](docs/03-user-stories.md)
- [领域模型与数据字典](docs/04-domain-model.md)
- [API 契约](docs/05-api-contract.md)
- [技术架构](docs/06-technical-architecture.md)
- [优化模型](docs/07-optimization-model.md)
- [前端交互与可视化规范](docs/08-ui-spec.md)
- [测试与验收计划](docs/09-test-acceptance.md)
- [决策记录与待确认项](docs/10-open-decisions.md)
- [从 0 开发计划](docs/11-development-plan.md)

## 首版 MVP

首版聚焦单站点、峰谷分时电价、15 分钟粒度、规则式调度、基础曲线看板和可复现的示例数据。现货电价、机器学习预测、线性规划、多站点和生产级权限体系作为后续扩展。

## 当前开发状态

后端闭环已完成，当前处于 I6（容量规划 + 经济性 + PDF 报告）迭代：

- FastAPI 服务骨架、统一错误结构（含 request_id）和健康检查
- SQLite 列式存储（站点/价目/测量，时间戳统一 UTC 规范化）
- 站点创建、查询、修改
- 光伏/负荷/SOC/电网测量数据上传（JSON + CSV 文件 `POST /api/v1/sites/{site_id}/measurements/csv`，行级容错）
- 数据质量报告 `GET /api/v1/sites/{site_id}/data-quality`（完整度、缺失区间合并）
- TOU 电价 Provider、价目查询、站点日价格点 `GET /api/v1/sites/{site_id}/tariff-points`
- 规则式日调度和 SOC 计算（需量感知：站点可配 `demand_limit_kw` 限制谷段电网充电功率）
- 调度运行落库追溯 `GET /api/v1/sites/{site_id}/dispatch-runs[/{run_id}]`（含 input_digest，可复现性校验）
- 曲线聚合 `GET /api/v1/sites/{site_id}/curves`（多日 15 分钟网格，pv/load/net_load/price/tier 单一对齐数据源）
- 超短期预测 `POST /api/v1/forecast`（pv 三模型可选：`auto`（默认，有历史→融合/无历史→物理）/`statistical`（同刻均值基线）/`physics`（pvlib 物理链）；回测精度口径 + p10/p90；`intraday` 日内滚动修正；load 同刻均值；TOU 价格政策给定）
- 气象 Provider `GET /api/v1/sites/{site_id}/weather`（Open-Meteo 逐小时 GHI/DNI/DHI 辐射预报，Provider 可插拔，`WEATHER_PROVIDER=mock` 离线开发）
- 光伏物理链 `domain/forecast/pv_physics.py`（pvlib：太阳位置 → Hay-Davies 转置 → Faiman 温度 → PVWatts 直流，逐小时插值 15 分钟）与融合校准 `blend.py`（物理 × 实测校准系数）
- 调度支持 `source=auto/measurements/forecast`（auto 按当天数据完整性自动选择，dispatch_runs 记录实际 data_source；光伏预测可走物理链冷启动）
- 容量规划 `POST /api/v1/storage/sizing`（同一调度引擎跑代表日折算全年，NPV/IRR/回本 + 价差/电池单价敏感性，全量 assumptions；`GET /sites/{id}/sizing-runs[/{run_id}]` 追溯）
- Sizing PDF 报告 `POST/GET /api/v1/reports[/{id}[/download]]`（reportlab，中文 CID 字体）
- pytest 测试套件

本地运行：

```powershell
Set-Location backend
python -m pytest -q        # 运行全部测试
python scripts/seed_demo.py  # 写入示例电价、站点和 3 天合成数据（幂等，可重复执行）
python run.py
```

种子脚本会创建江苏示例 TOU 电价 `T-JIANGSU-DEMO`、南京示例站点 `S-NANJING-DEMO` 以及最近 3 天 15 分钟粒度的光伏/负荷合成数据（写入 `backend/data/storage.db`），脚本末尾会打印一条可直接调用的 `/api/v1/storage/schedule` 示例请求。

启动后访问 <http://127.0.0.1:8000/docs> 查看 OpenAPI 文档。

## 重要原则

1. 规划和运行共享统一的价格、SOC、功率和收益口径。
2. 电价通过 Provider 抽象，优化器不直接依赖具体电价来源。
3. 时间序列统一使用带时区的 ISO 8601 时间戳。
4. 字段名显式携带单位，缺失值使用 `null`，不使用 `0` 代替缺失。
5. 所有收益结果必须同时返回假设、基准和计算口径。
6. 首版先跑通完整业务链路，再增加模型复杂度。

## 文档状态说明

- `已确认`：来自现有设计稿或本项目当前明确决定。
- `默认假设`：为了推进 MVP 暂时采用，后续可被决策覆盖。
- `待确认`：会影响实现或商业口径，开发前必须关闭。

## 原始输入

本项目的初始输入是《储能配置优化系统设计》v0.1。本仓库文档对其进行了产品化整理，并补充了范围、契约、验收和风险说明。
