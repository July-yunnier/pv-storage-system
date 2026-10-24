# 光储一体 · 储能配置优化系统

面向工商业用户的光伏发电预测、负荷预测、电价分析、储能容量配置与日充放电策略系统。

> 当前状态：系统本体位于根目录 `backend/` 与 `frontend/`，已有实现与测试；正式主实验 P0、P0-A、P1、P2、P3、P4 已执行并完成产物封存。实验结果仍需按公开光伏代理、仿真负荷、政策电价和 GEFCom 现货演示口径解释，不能直接作为真实广东现场收益承诺。

## 项目目标

系统围绕同一个优化内核提供两类能力：

- **规划（Sizing）**：根据站点、光伏、负荷、电价和投资假设，推荐储能功率与容量，并测算收益、回本周期、IRR 和 NPV。
- **运行（Dispatch）**：根据次日预测、价格曲线和当前 SOC，生成 15 分钟粒度的充放电计划。

系统明确不负责储能电气设计、电力交易申报和现场 EMS 实时控制。

## 文档导航

- [分支内容整合与待确认项](docs/14-repository-integration.md)
- [独立负荷/电价离线预实验](backend/docs/OFFLINE_PREEXPERIMENTS.md)

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
- [正式主实验规划与验收协议](docs/17-experiment-plan.md)
- [正式主实验报告（P0-P4）](reports/official_experiment/official_experiment_report.md)
- [侧边聊天 LLM 试运行归档](reports/generated/llm_live_p0_20261024/README.md)

## 首版 MVP

首版聚焦单站点、峰谷分时电价、15 分钟粒度、规则式调度、基础曲线看板和可复现的示例数据。现货电价预测已落地（见 `docs/15-price-forecast.md`），线性规划、多站点和生产级权限体系仍作为后续扩展。

## 当前开发状态

后端闭环 + 前端看板已完成，正式主实验已执行完毕，当前进入结果审计、未通过项诊断和真实数据补强阶段：

- FastAPI 服务骨架、统一错误结构（含 request_id）和健康检查
- SQLite 列式存储（站点/价目/测量，时间戳统一 UTC 规范化）
- 站点创建、查询、修改
- 光伏/负荷/SOC/电网测量数据上传（JSON + CSV 文件 `POST /api/v1/sites/{site_id}/measurements/csv`，行级容错）
- 数据质量报告 `GET /api/v1/sites/{site_id}/data-quality`（完整度、缺失区间合并）
- TOU 电价 Provider、价目查询、站点日价格点 `GET /api/v1/sites/{site_id}/tariff-points`
- 电价预测 `GET /api/v1/sites/{site_id}/price/forecast?days=1..7&model=auto`：现货历史 + 负荷 + 天气 → 小时级 P10/P50/P90，按 15 分钟阶跃展开，并直接折算最优充放窗口与净价差；降级链 `LightGBM 工件 → 统计画像 → 政策表`，模型名如实标注（`tou_policy` 仍表示"政策查表，不是预测"）。现货序列导入见 `backend/scripts/import_spot_price.py`，站点级训练见 `backend/scripts/train_price_model.py`，GEFCom2014-P 基准实验见 `backend/scripts/run_price_benchmark.py`，方法与结果见 `docs/15-price-forecast.md`
- 规则式日调度和 SOC 计算（需量感知：站点可配 `demand_limit_kw` 限制谷段电网充电功率）
- 调度运行落库追溯 `GET /api/v1/sites/{site_id}/dispatch-runs[/{run_id}]`（含 input_digest 与分时段人话计划说明，可复现性校验）
- 曲线聚合 `GET /api/v1/sites/{site_id}/curves`（多日 15 分钟网格，pv/load/net_load/price/tier 单一对齐数据源）
- 超短期预测 `POST /api/v1/forecast`（pv 三模型可选：`auto`（默认，有历史→融合/无历史→物理）/`statistical`（同刻均值基线）/`physics`（pvlib 物理链）；负荷双模型可选：`statistical` 同刻均值 / `advanced` MES 排产+多任务注意力（可选 AI 依赖，服务端自动构造特征）；`intraday` 日内滚动修正（光伏偏差衰减 / 负荷残差卡尔曼）；回测精度口径 + p10/p90；TOU 价格政策给定）
- 观测体系：预测自动封存、实测到位自动对账 `GET /api/v1/sites/{site_id}/observability/*`
- 回放实验 `POST /api/v1/sites/{site_id}/replay`：预测方法 × 调度策略矩阵，SOC 跨日连续，导出 JSON/CSV（`scripts/run_replay.py`）
- 政策电价解析 `POST /api/v1/policy-tariffs/parse-text|parse-pdf|confirm`：LLM 产候选 + 严格校验 + 人工确认入库（可选 AI 依赖）
- 气象 Provider `GET /api/v1/sites/{site_id}/weather`（Open-Meteo 逐小时 GHI/DNI/DHI + 温湿度，Provider 可插拔，`WEATHER_PROVIDER=mock` 离线开发）
- 光伏物理链 `domain/forecast/pv_physics.py`（pvlib：太阳位置 → Hay-Davies 转置 → Faiman 温度 → PVWatts 直流，逐小时插值 15 分钟）与融合校准 `blend.py`（物理 × 实测校准系数）
- 调度支持 `source=auto/measurements/forecast`（auto 按当天数据完整性自动选择，dispatch_runs 记录实际 data_source；光伏预测可走物理链冷启动）
- 容量规划 `POST /api/v1/storage/sizing`（同一调度引擎跑代表日折算全年，NPV/IRR/回本 + 价差/电池单价敏感性，全量 assumptions；`GET /sites/{id}/sizing-runs[/{run_id}]` 追溯）
- Sizing PDF 报告 `POST/GET /api/v1/reports[/{id}[/download]]`（reportlab，中文 CID 字体）
- 前端（React + Vite + AntD + ECharts）：总览 / 算法观测（光伏、负荷、电价、调度、储能）/ 投资 / 数据管理 / 测评中心
- pytest 测试套件；正式实验报告记录 353 passed，侧边 LLM 归档发布分支验证 383 passed。依赖平台相关的测试结果以当前环境复核为准。

本地运行：

```powershell
Set-Location backend
python -m pytest -q            # 运行全部测试
python scripts/seed_dongguan.py  # 东莞示例站点 + 广东示例电价（幂等）
python scripts/import_guowang.py # 灌入国网公开光伏两年回放 + 两班制仿真负荷（日期平移到最近）
python run.py
```

前端（另开终端）：`Set-Location frontend; npm install; npm run dev`，默认 <http://localhost:5173>。

演示站点为东莞示例电子产业园 `S-DONGGUAN-DEMO`（广东珠三角 TOU 示例电价
`configs/tariff.guangdong.yaml`，午间平段+双高峰，支持两充两放）；光伏为国网公开
50MW 场站实测标幺化回放（`dataset_guowang/`，需另行获取），负荷为两班制仿真
（`source=simulator`，不得对外宣称为真实工厂负荷）。`scripts/seed_demo.py`
（南京/江苏 + 3 天合成数据）为早期冒烟演示保留，新环境请用上面的东莞链路。

AI 增强（可选）：`pip install -r requirements-ai.txt` 后，负荷 MES 注意力模型经
`.env.ai`（参照 `.env.ai.example`）配置 `LOAD_MODEL_PATH` 启用；政策电价 LLM 解析
（`/api/v1/policy-tariffs/*`）默认指向本地 Ollama，未配置时接口优雅报错、不影响主链路。

启动后访问 <http://127.0.0.1:8000/docs> 查看 OpenAPI 文档。

正式主实验状态：`reports/official_experiment/` 保存 P0、P0-A、P1、P2、P3、P4 的结果与 `run_meta.json`。本轮 C1 未通过、C2 通过、C3 部分未通过、C4 部分覆盖、C5 按站点/任务拆分；因此“实验已完成”表示协议执行和结果封存完成，不表示所有验收标准均通过。报告中的金额单位为元，预测功率单位为 kW，时间戳按对应实验时区带时区记录。

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
