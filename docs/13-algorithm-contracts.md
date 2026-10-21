# 算法接口契约（团队协作界面）

> 状态：已确认。本文是算法分工的**唯一界面**：做算法的同学只面对本文的输入/输出契约，
> 不需要理解前端、路由、数据库之外的任何系统细节。契约未变，算法实现可以随意替换。

## 1. 五条算法线总览

| # | 算法 | 输入 | 输出 | 当前实现 | 插拔点 |
|---|---|---|---|---|---|
| A | 光伏功率预测 | 历史实测 + 气象预报 + 站点光伏参数 | 96 点功率曲线 + 置信区间 + 口径 | statistical / physics / blend | `options.pv_model` |
| B | 负荷预测 | 历史实测 + 日历特征 | 96 点功率曲线 + 置信区间 + 口径 | 同刻均值+工作日模式 | `domain/forecast/load.py` |
| C | 电价 | 历史现货电价 + 负荷/天气 + 目录电价政策 | 96 点价格曲线 + P10/P90 + 套利窗口 | `tou_policy` 查表 / `price_diurnal_quantile` 统计画像 / `price_lgbm_quantile` 分位数 GBM | `model=auto\|lgbm\|statistical\|tou`，见 `15-price-forecast.md` |
| D | 调度决策 | A/B/C 的 96 点曲线 + 储能参数 + 初始 SOC | 96 点充放电计划 + SOC 轨迹 + 收益 | rule_v1 贪心 | 分层：D1 固定 / D2 规则 / D3 LP（待做） |
| E | 容量规划 | 负荷/光伏曲线 + 设备候选 + 电价 + 造价 | 推荐 PCS/容量 + NPV/IRR/回收期 | 枚举+财务模型 | `sizing_service.py` |

**纪律**：

- 电价查表结果不得包装成"预测"（沿袭 `12-forecast-roadmap.md` 口径纪律）。
  现实现区分两种模型名：`tou_policy`（政策表展开，确定量，区间退化）与
  `price_diurnal_quantile` / `price_lgbm_quantile`（数据驱动预测，带 P10/P90）。
  响应里的 `model` 字段必须如实反映用的是哪一种，不允许混称。
- 任何精度数字必须带 `caliber`（归一化基准 / 是否含夜间 / 空间尺度 / 对照基线）。
- LLM 只做意图层与解释层（"优化器管算，LLM 管说"），**不直接生成 96 点计划**。

## 2. 数据基座：measurements 表

所有算法共用同一张事实表 `measurements`（`site_id, ts, data_type, value, quality, source`）：

| data_type | 含义 | 单位 |
|---|---|---|
| `pv_power` | 光伏出力 | kW |
| `load_power` | 园区负荷 | kW |
| `storage_soc` | 储能 SOC | %（0-100） |
| `grid_power` | 并网功率（正=购电） | kW |

血统字段（合规口径，报告与演示必须保持一致）：

| 字段 | 取值 | 语义 |
|---|---|---|
| `quality` | `measured` | 真实测量（含公开数据集回放） |
| | `estimated` | 仿真/推算 |
| | `missing` / `invalid` | 缺测 / 质检剔除 |
| `source` | `public_replay` | 公开真实数据经容量缩放（如国网数据集） |
| | `simulator` | 仿真生成 |
| | `open_meteo` 等 | 外部 API |

- 时间粒度统一 **15 分钟，每天 96 点**，时区归一化到站点时区（UTC 存储）。
- 新数据源接入 = 写一个导入脚本把 CSV 规整成上述格式 POST `/api/v1/sites/{id}/measurements`，
  参考 `backend/scripts/import_guowang.py`（国网数据集 → S-DONGGUAN-DEMO 的完整示例：
  列数弹性解析 → 容量缩放 → 96 点质检 → 血统标记）。

## 3. 契约 A：光伏功率预测

**输出 DTO**（`app/domain/forecast/result.py`，算法必须填满）：

```json
{
  "target": "pv_power",
  "unit": "kW",
  "model": {"name": "你的算法名", "version": "语义化版本"},
  "points": [{"ts": "ISO8601 带时区", "value": 0.0}, "...96 点"],
  "p10": [96 点下界], "p90": [96 点上界],
  "accuracy": {
    "metric": "skill",
    "value": 0.89,
    "caliber": {"normalization": "cap", "cap_kw": 1000, "includes_nighttime": true,
                "spatial_scale": "site", "baseline": "day_persistence"},
    "evaluated_on": "2026-09-01..2026-09-30"
  },
  "provenance": {"provider": "open_meteo", "factor": 2.07, "window": "回测9天+预报1天"},
  "notes": "人类可读的口径说明"
}
```

**插拔方式**：在 `forecast_service.py` 的 pv 路由中注册新 `pv_model` 名；
在 `replay_service.py` 的 `PV_METHODS` 加同名条目即可进入回放对比矩阵。

**验收标准**（不是自评，是框架对账）：

1. 防泄漏：预测目标日 T 时，只允许使用 T 之前的数据（回放框架强制，oracle 除外）。
2. 指标：对 S-DONGGUAN-DEMO 的 30 天回放，报全天/白天双口径 skill（基准 day_persistence）。
   当前基线：slot_mean 全天 0.8935 / 白天 0.8578；physics 链存在 −46.7kW 系统性偏低（坐标失配，见 §6）。
3. 精度数字由 `forecast_accuracy` 表产出，不得手工填写。

**分尺度分层**（预测的"时间尺度地图"，算法按层接管）：

| 层 | 时效 | 当前基线算法 | 预留插拔位 |
|---|---|---|---|
| ① 超短期 | 0~15 分钟 | 持续性外推（persistence-decay 首槽权重 ≈0.68） | 天空相机（鱼眼 ASI）算法——队友 SUNSET 工作包 |
| ② 短临 | 15 分钟~4 小时 | 偏差修正随时效指数衰减（τ=90min）回气候基线 | 天空相机 / 卫星云图反演；成品卫星辐射 provider（Open-Meteo Satellite Radiation API 增强位） |
| ③ 日前 | 4 小时以上 | 气候基线 + NWP（Open-Meteo → pvlib → 校准） | 更优 NWP 或混合模型 |

约定：

- 层 ①② 由 `predict_pv_intraday`（日内滚动修正）输出，层 ③ 由日前 `forecast` 输出；
  前端今日图用 markArea 把三层画在同一张图上，方便肉眼对齐。
- 新算法接管某一层时，注册为新 `pv_model` 名并声明接管的时效区间（写入 `notes`），
  回放矩阵按同一测试窗口对比该层指标，不允许跨层混报。
- 自建卫星云图反演（FY-4A HDF 原始通道）已调研并放弃：免费渠道仅保留 24h、
  反演工程量大；卫星路线只走"成品辐射产品"或队友的相机算法。

## 4. 契约 B：负荷预测

与 A 同形，`target="load_power"`，`caliber.normalization="mean_power"`。
日历特征（工作日/周末/节假日）是必须项；当前基线为同刻均值 × 工作日模式。

### 4.1 学习类模型（已接入，`load_model=advanced`）

- 模型：MES 排产 + 多任务注意力（`load_mes_multitask_attention`），工件
  `backend/data/training_artifacts/load_mes_attention.pt` + 同名 `.manifest.json`，
  经 `LOAD_MODEL_PATH` 环境变量或 `backend/.env.ai` 配置；依赖 `requirements-ai.txt`
  （torch 延迟加载，未安装时统计链路与调度不受影响，advanced 路径报
  `DATA_QUALITY_ERROR`/`PROVIDER_ERROR` 优雅降级）。
- 特征契约（列序即 manifest `history_feature_names`/`target_feature_names`，
  标准化由工件内参数完成，调用方喂原始物理量）：
  - history（`history_days`×96 槽，默认 30 天）：`[total_load_kw, temperature_c,
    humidity_pct, slot_sin, slot_cos, is_workday]` + 每槽 MES 标签
  - target（96 槽）：`[temperature_c, humidity_pct, slot_sin, slot_cos, is_workday]`
    + 每槽 MES 标签
- 服务端自动构造（`application/load_features.py`）：负荷取 measurements
  （measured/estimated，缺口用同刻均值补）；温湿度取气象 provider 窗口
  （历史=再分析/模型分析值，目标日=预报，小时锚点线性插值到 15 分钟）；
  MES 标签来自班次日历（`domain/shift_calendar.py`，站点
  `load.mes_calendar` 可覆盖，默认两班制；标签 id 与 manifest `mes_mapping` 一致）。
- 调用方也可显式传 `advanced_load_input`（实验/回放后门）。
- 日内滚动：`predict_load_intraday`（同刻均值基线 + 实测残差卡尔曼修正）。
- **口径红线**：工件在仿真两班制数据上训练（manifest `limitations`），
  精度数字不得作为真实工厂表现对外宣称；真实数据到位后用
  `fine_tune_for_factory` 微调落地。

### 4.2 算法注册表：新增算法三步走（算法与业务解耦）

全部预测算法的元数据集中在 `app/application/algorithm_registry.py` 的
`_CATALOG`（单一事实源）。业务层不再各自维护算法清单：API 入参校验、
回放方法矩阵（`replay_service.PV_METHODS`）、算法发现接口
`GET /api/v1/algorithms` 全部由注册表派生。

新增一个算法的全部工作：

1. **domain 纯函数实现**：在 `app/domain/forecast/` 写预测函数——禁止 I/O、
   禁止读环境变量/文件；模型工件加载走 `infrastructure/model_store.py`
   （`load_bundle` 带缓存），由 application 层注入 `state_dict`/`metadata`。
2. **注册表登记一行**：`_CATALOG` 加一条 `ForecastAlgorithm`——
   要对产品开放就填 `product_key`（即 `pv_model`/`load_model` 的取值），
   要进回放矩阵就标 `replay=True`，仅日内模式标 `intraday=True`。
3. **forecast_service 分发表挂分支**：在 `_pv_day_forecast` / `_forecast_target`
   里把 `product_key` 映射到实现。

`ForecastOptions.pv_model/load_model` 是宽 `str`（不再是 Literal）：
新算法登记即可用，不必改请求 schema；非法取值在 `run_forecast` 入口
统一 422 并报可用清单。`GET /api/v1/algorithms` 返回完整目录
（含 `requires` 依赖声明与 `available/unavailable_reason` 环境探测），
前端与协作方据此发现能力，无需翻源码。

## 5. 契约 C：电价

TOU 查表：输入站点电价配置（`configs/tariff.*.yaml`），输出 96 点 CNY/kWh。
`accuracy` 必须为 `null`，`p10 == p90 == points`（确定量无不确定性）。
未来现货价格预测另立算法线，不得复用本接口伪装。

## 6. 契约 D：调度决策（核心）

**输入**：A/B/C 三条 96 点曲线 + 储能参数（`pcs_kw / capacity_kwh / soc_min / soc_max /
charge_eff / discharge_eff`）+ 初始 SOC + 防逆流约束。

**输出**：96 点计划，每点 `{ts, action: charge|discharge|idle, power_kw, soc_start, soc_end}`
+ 汇总（电费、节省、峰谷套利明细）。SOC 递推必须满足：

```
soc_energy(t+1) = clamp(soc_energy(t) + charge*Δt*η_c − discharge*Δt/η_d,
                        soc_min*E, soc_max*E)
```

**分层路线**（回放框架已验证的动机）：

| 层 | 策略 | 30 天回放节省（东莞站点） |
|---|---|---|
| D0 | 无储能基线 | 0 元 |
| D1 | 固定曲线（传统定时） | **17,080 元** |
| D2 | rule_v1 预测驱动贪心（现有） | 14,485 元 |
| D5 | oracle 规则（理论近似上界） | 18,663 元 |

关键事实：D2 跑输 D1 —— 规则引擎对预测误差敏感且非最优，这正是 **D3 LP/优化求解器
工作包**的存在理由（输入同上，目标是在 D5 上界之内稳定超过 D1）。

**插拔方式**：`replay_service.py` 的 `STRATEGIES` 注册；线上路径在 `dispatch_service.py`。

## 7. 契约 E：容量规划

输入：8760（或代表日×365）负荷/光伏曲线、PCS/容量候选集、电价、造价与贴现参数。
输出：推荐配置 + NPV / IRR / 静态回收期 + 敏感性。口径见 `07-optimization-model.md`。

## 8. 观测体系（算法必须有"记分牌"）

- `forecast_archive`：每次预测自动封存（points + provenance + issued_at），封存失败不阻断主流程。
- `forecast_accuracy`：实测满 96 点/天后自动对账（幂等，UNIQUE(site,target,method,date)）。
- API：`GET /api/v1/sites/{id}/accuracy?target=&days=`，`POST .../accuracy/reconcile`。
- **任何新算法接入即自动获得封存与对账**——这是算法对比的唯一证据来源。

## 9. 回放实验框架（验收跑道）

```bash
python scripts/run_replay.py --site S-DONGGUAN-DEMO --days 30
# 或 API: POST /api/v1/sites/{id}/replay
```

产出 P×D 矩阵（预测法 × 调度策略）逐日明细 + 汇总，导出 JSON/CSV 到
`backend/data/replay/<run_id>/`。SOC 跨日连续传递；气象代理为 Open-Meteo 再分析
（archive-api），physics 链与光伏实测存在坐标失配——**报告中必须声明这两项近似**。

## 10. 变更规则

契约字段变更 = 同步修改本文 + `04-domain-model.md` + `05-api-contract.md` + 契约测试
（`tests/test_forecast_api.py` 等）。算法内部实现变更只要契约不变，无需评审。
