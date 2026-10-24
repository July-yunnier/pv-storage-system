# 17 · 正式实验规划：系统省钱能力验证

> 目的：用统计上站得住的实验回答一个问题——**这套系统到底能省多少钱**。
> 本文是实验协议：先定口径和验收标准，再跑实验，不接受事后移动球门。
> 状态：P0 已可执行（现有回放框架）；AMPDNet 已完成工程接入，但其日内效果仍按独立 rolling-origin 协议验证（见 §7、§8）。

## 1. 实验要回答的问题

| 编号 | 问题 | 决策含义 |
|---|---|---|
| Q1 | 相对**无储能基线**，系统调度省多少？ | 储能装不装、值多少 |
| Q2 | 相对**固定两充两放**（传统定时 EMS），预测驱动调度多省多少？ | 我们软件的智能增量，向甲方收费的底气 |
| Q3 | 距**事后最优**（拿实测当预测的理论上界）还差多少？ | 规则引擎够不够，LP 优化器值不值得上 |
| Q4 | 光伏/负荷/电价三条预测线，各自精度提升**值多少钱**？ | 算法资源往哪投；评估队友模型的实际贡献 |

## 2. 实验对象与数据

### 2.1 站点档案（东莞示例电子产业园，S-DONGGUAN-DEMO）

| 项 | 值 |
|---|---|
| 光伏 | 1000 kWp |
| 负荷 | 峰值约 800 kW，两班制（含周末/节假日模式） |
| 储能 | PCS 500 kW / 1000 kWh，往返效率 0.88，SOC 10%–95% |
| 电价 | 广东工商业 TOU（configs/tariff.guangdong.yaml） |

### 2.2 数据集分层（如实标注来源）

| 数据 | 来源 | 口径 |
|---|---|---|
| 光伏功率 | 国网站点真实测量回放 | 真实 |
| 负荷 | 两班制工厂仿真 | 仿真（标注 A-004） |
| 政策电价 | 广东 TOU 政策表 | 真实政策，静态 |
| 现货电价 | GEFCom2014-P（PJM Zone1 真实 LMP） | 真实但非本地，仅演示/协议验证 |
| 气象 | Open-Meteo（历史=再分析，未来=预报） | 再分析代替当年预报是已知近似 |

### 2.3 环境锁定

主后端 Python 3.14；PV 推理侧车 `.venv-infer`（CPython 3.12.13 + 8 包锁定）；
每次实验记录 git commit hash；所有实验产物落盘封存（`backend/data/replay/{run_id}`）。

## 3. 实验设计

### 3.1 实验臂（调度策略）

| 臂 | 策略 | 角色 |
|---|---|---|
| E0 | 无储能 | 基线（Q1 分母） |
| E1 | 固定曲线（谷充峰放定时） | 传统做法（Q2 分母） |
| E2 | 预测驱动规则（现网 rule_v1） | **被试对象** |
| E3 | 事后最优（实测驱动，理论上界） | 天花板（Q3），仅回放，不是产品能力 |

### 3.2 预测源矩阵（注册表与调度选择口径）

- **光伏**：blend（默认融合）/ slot_mean / day_persistence / physics / **ampdnet** / oracle（oracle 仅用于上界实验，不进产品链路）
- **AMPDNet 调度选择边界**：`pv_model="ampdnet"` 是已接入的可选光伏预测源，定位为白昼压缩协议下的日内短临候选，支持 `15min`、`1h`、`4h` 三个任务。它需要请求中显式提供 `ampdnet_input`（站点、任务以及带时区的 TSI/DNI/GHI/温度观测窗口），使用 CPU TorchScript 工件推理；工件不可用时默认回退现有日内统计预测，也可显式设置为错误模式。
- **日前边界**：AMPDNet 不等同于日前 96 点全天模型，不进入默认日前回放的全天预测源矩阵。只有在调度请求具备显式日内观测窗口、且预测时段仍处于白昼协议范围内时，才允许把它作为局部日内调度候选；其效果必须与日前模型分开统计。
- **负荷**：diurnal_mean（统计）/ mes_attention（高级工件，需 torch + LOAD_MODEL_PATH）
- **电价**：tou（政策表）/ statistical（14 天中位数画像）/ lgbm（分位工件，需站点训练件）

### 3.3 滚动回放协议（核心防泄漏机制）

对回放窗口内每一天 D：
1. 只用 D 之前的数据发报 D 的预测（T-1 日发报，滚动 origin）；
2. 用预测+电价+储能约束生成 96 点充放电计划；
3. 用 D 的**实测**光伏/负荷执行计划，结算电费；
4. SOC 跨日连续继承（E2 的现状行为）；
5. 预测原样封存（forecast_archive），实测到齐后对账精度。

红线：任何臂不得使用 D 当天或之后的实测做决策（E3 除外，它定义上界）。

### 3.4 消融实验（回答 Q4）

在 E2 固定规则引擎上，逐条线换预测源：
- 光伏：blend vs slot_mean vs day_persistence（现有 PV_METHODS 矩阵直接覆盖）
- AMPDNet：单独进行日内 rolling-origin 验证，按 15min/1h/4h 分任务与 day persistence 对比；不与日前全天模型直接混合排名，也不据此直接宣称整体调度收益改善。
- 负荷：统计 vs mes_attention
- 电价：tou vs statistical vs lgbm（需现货序列场景，见 P2）
- 气象：有气象（物理链）vs 无气象（纯统计）——量化 Open-Meteo 接入的价值

## 4. 指标体系

### 4.1 财务主指标
- 总电费节省额（元）与节省率（%），分臂汇总；
- 日节省分布（p10/p50/p90），看稳定性而非只看均值；
- 年化外推必须写明口径（当前：典型日×365，§8 列了升级项）。

### 4.2 运行约束指标
- SOC 违规次数（必须为 0）、充放功率越限次数；
- 等效满循环数（放电总量/可用容量，关乎寿命与衰减成本）；
- 光伏自用率（%）。

### 4.3 预测精度（封存对账）
- 光伏/负荷：skill（基准=昨日延续）、nRMSE、bias；
- 电价：MAE/RMSE、pinball(0.1/0.5/0.9)、PICP80、峰值命中率
  （实际最贵 4 小时中预测也排进前 4 的比例——储能套利真正吃这个）。

### 4.4 上限距离
`上限达成率 = E2 节省 / E3 节省`；差距即 LP 优化器（D3）的潜在空间。

## 5. 统计严谨性

1. **样本外**：预测与特征只用发报时刻之前的数据；
   `tests/test_price_features.py::test_features_do_not_look_past_origin` 这类
   防泄漏测试是协议的组成部分，不是附加项。
2. **配对比较**：E2 vs E1 按天配对，bootstrap 日节省差值的 95% 置信区间；
   区间不含 0 才宣称"显著更省"。
3. **不调参重报**：若验收标准不通过，如实报告并触发诊断；
   调整规则/特征后重跑必须披露为新一轮实验，不得混入原轮。
4. **演示数据声明**：GEFCom 现货序列、仿真负荷在所有报告中显著标注，
   不作为广东真实收益承诺。

## 6. 执行矩阵与排期

| 阶段 | 内容 | 前置 | 产物 |
|---|---|---|---|
| P0 | 现链路 30 天回放：E0/E1/E2×全 PV 方法/E3 | 无（已可跑） | 基线报告 + 逐日明细 CSV |
| P0-A | AMPDNet 日内 rolling-origin 验证：15min/1h/4h 分任务，对比 day persistence | 显式天气观测窗口、对应站点 CPU 工件 | 分任务 MAE、skill、样本量、置信区间及可否进入调度候选的结论 |
| P1 | 需量电费接入后重跑：结算加需量项，E2 加放电侧削减 | 五议题①实现 | 需量收益拆分报告 |
| P2 | 现货场景：GEFCom 演示序列 + 电价三模型消融 | 现货序列导入 | 现货下收益结构对比 |
| P3 | LP 优化器接入：E2 vs LP 同协议对比 | D3 工作包 | LP 增量收益报告 |
| P4 | 一年期仿真：跨季节典型日加权 + 衰减因子，算 NPV/IRR | 五议题②③实现 | sizing 投资报告 |

P0 命令（backend/ 下）：

```bash
python scripts/run_replay.py --site S-DONGGUAN-DEMO --days 30
# 气象代理（physics/blend 需要）网络受限时：
set HTTPS_PROXY=http://127.0.0.1:7897
```

## 7. 验收标准（预先声明）

| 编号 | 标准 | 含义 |
|---|---|---|
| C1 | E2 的 30 天总节省 > E1，配对 bootstrap 95% CI 不含 0 | 预测驱动确实优于定时（Q2） |
| C2 | E2 上限达成率 ≥ 70% | 规则链路吃到大头；LP 上线后目标 ≥ 85% |
| C3 | SOC 违规 = 0 且光伏自用率 ≥ 95% | 调度物理可行、不浪费光伏 |
| C4 | 光伏 blend skill > 昨日延续；现货场景 lgbm skill > naive_daily | 日前光伏与现货电价预测线各自成立 |
| C5 | AMPDNet 完成 15min/1h/4h 独立 rolling-origin，对每个任务报告 MAE、skill vs day persistence、样本量与 95% CI；只有对应任务 skill > 0 且 CI 不含 0 时，才允许将该任务纳入调度收益对比 | 日内候选模型的效果边界可审计；未满足时仅报告“机制已接入/候选能力”，不得写成“效果已验证” |

任何一条不达标：报告如实呈现并进入诊断，不算实验失败，算发现。

## 8. 现状缺口对照（先决条件清单）

| 缺口 | 影响阶段 | 说明 |
|---|---|---|
| 需量电费字段/结算/放电侧削减 | P1 | 政策解析已能读出字段，转换时被丢弃，接回成本低 |
| 现货结算 provider | P2 | 现仅有 TOU；P2 先用演示序列，报告注明 |
| 电池衰减现金流因子 | P4 | cashflow 层线性衰减（~2%/年），十几行改动 |
| sizing 单日评估期末 SOC=期初 | P4 | 回放已跨日连续；sizing 需加循环边界保证可比 |
| AMPDNet 光伏深度模型 | P0-A | 工程接入已完成：注册表、forecast service、CPU 工件加载、fallback、API/集成测试均已具备；仍需按白昼压缩协议完成 15min/1h/4h rolling-origin，验证显式天气输入、站点规模适配及相对 day persistence 的 skill。它仍定位为日内短临候选，不作为默认日前全天模型，当前不能把机制就绪写成效果已验证 |
| July MQ-RNN 电价模型 | P2 | 工件在 feat/product-mvp-impl 的 product/ 旧树；对比结论与同协议对决设计见 docs/18 |

## 9. 可复现性

- 回放：`backend/scripts/run_replay.py`（CLI 见 §6），结果封存 `backend/data/replay/{run_id}/`
- 电价基准：`backend/scripts/run_price_benchmark.py`（GEFCom 协议，见 docs/15 §4）
- AMPDNet 训练：`backend/scripts/train_ampdnet_pv.py`；CPU 工件目录：`backend/data/model_releases/ampdnet_delivery20/`
- AMPDNet 验证：`cd backend && python scripts/validate_ampdnet_pv.py <csv> --station Solar1 --task 15min`；将 `--task` 分别设为 `15min`、`1h`、`4h`，按 rolling-origin 与 day-persistence 基线对账。输入观测的时间戳必须带时区，功率单位为 kW。
- 预测封存与对账：`forecast_archive` 表 + `/api/v1/sites/{id}/accuracy`
- 环境：主后端 `backend/requirements.txt`；侧车 `requirements-infer.lock.txt`；
  所有实验报告必须附 `run_meta.json`（commit、参数、数据窗口）
