# 15 · 电价预测（Electricity Price Forecasting）

> 范围：把「电价中心」从"展示当前政策表"升级为"预测未来 1–7 天电价分位数曲线，
> 并直接折算储能套利窗口"。本文档同时是算法契约，改动实现请同步改这里。

## 1. 问题定义

工商业光储系统的收益来自**价差**，不是来自"均价"。因此本模块的输出不是一个数，
而是一条带不确定性的曲线：

- 目标量：站点侧小时级电价 `P(t)`，单位 CNY/kWh；
- 预测形式：`P10 / P50 / P90` 三个分位数（分位数损失训练，非事后加区间）；
- 提前量：日前 24 小时为主，支持 3 天 / 7 天（按天递归重发，见 §5.3）；
- 下游消费者：调度器（决定充放电深度与时段）、容量规划（算年化收益分布）。

只给点预测会在两个地方直接出错：
1. 套利决策会**过拟合噪声**——P50 上看着有价差，实际是赌出来的；
2. 无法表达"今天该保守还是激进"——同样的期望价差，区间越宽越该收缩功率。

## 2. 数据

### 2.1 离线基准：GEFCom2014-P（真实市场，公开可复现）

`数据与参考文献/GEFCom2014 Data/GEFCom2014-P_V2.zip`，已解压到 `data/power/price/Price/`。

已逐文件核对（不要凭印象改）：

| 事实 | 取值 |
|---|---|
| 任务数 | 15 个，每任务是**单日 24 小时**日前预测 |
| 历史起点 | 2011-01-01，小时级 |
| 任务日 | 2013-06-16、06-17、06-24、07-04、07-09、07-13、07-16、07-18、07-19、07-20、07-24、07-25、12-07、12-08、12-17 |
| 目标 | Zone 1 的 LMP（$/MWh），最低 12.52、最高 363.8 |
| 协变量 | `Forecasted Total Load`、`Forecasted Zonal Load`（**当时已知**的日前负荷预报） |
| 官方基准 | `BenchmarkN_P.csv` 提供 99 个分位数，但 99 列**完全相同**——实为确定性季节持续法 |
| 校验 | 本仓库 `naive_weekly` 与官方基准逐点完全一致（见 §4.2），说明任务切分与对齐无误 |

该数据集**不含气象**。这不是缺陷：负荷预报已经把温度效应吸收进去了，
GEFCom2014 电价赛道的设定就是"给定负荷预报预测电价"。线上路径才显式接温度/辐照（§3.3）。

### 2.2 线上：站点现货电价

`spot_prices` 表（`site_id, ts_utc, price_cny_per_kwh, source, quality`）。
独立于 `measurements` 的原因：单位是 CNY/kWh 且**允许负值**（负电价是真实市场现象），
与功率测量的 kW/非负不是同一口径。

导入：`backend/scripts/import_spot_price.py`（CSV 或 GEFCom 演示序列）。

## 3. 方法

### 3.1 特征（`app/domain/forecast/price_features.py`）

样本形式 `(origin, horizon)`：origin 是最后一小时已知数据，horizon 是提前小时数。
**所有特征在 origin 时刻可得**，这条红线有专门的单元测试守着
（`tests/test_price_features.py::test_features_do_not_look_past_origin`：
把 origin 之后的价格全部改掉，特征必须逐列不变）。

| 组 | 特征 | 作用 |
|---|---|---|
| 日历 | hour / dow / 周末 / 法定节假日 / 节前 / 节后 / 调休上班 / 月 / 年周期 | 工商业负荷的骨架 |
| 季节压力 | 迎峰度夏(+1) / 度冬(−1) | 尖峰月份的加价先验 |
| 价格滞后（锚定 origin） | `P(-1)`、`P(-24)`、`P(-168)`、24h 均值/标准差/极值、168h 均值、168h 分位排名 | 均值回复 + 周周期 + 波动状态 |
| 目录电价 | 目标小时目录价、origin 目录价、二者之差、价差（市场−目录） | 政策底价与市场溢价分离 |
| 负荷 | 目标时刻负荷（日前预报）、origin 负荷、相对 168h 均值比 | 电价第一驱动量 |
| 天气 | 目标时刻温度、origin 温度、目标时刻 GHI | 温度→空调负荷→尖峰；辐照→净负荷下凹→午间电价塌陷 |
| 提前量 | horizon | 让一个模型同时服务 1..24 步 |

关于"天气"的边界要说清楚：温度**不直接进价格**，它通过负荷起作用；
把它显式放进特征，模型才能学到"高温 ∧ 高负荷 → 尖峰"的交互。GEFCom 基准没有天气列，
对应列在训练时会被 `select_active_columns` 自动剔除（全 NaN 列不参与建模）。

### 3.2 模型（`app/domain/forecast/price_model.py`、`price_mqcnn.py`）

1. **`naive_daily`** 昨日同时刻 —— 行业最常用基准；
2. **`naive_weekly`** 上周同时刻 —— 与 GEFCom2014 官方基准等价，用于校验对齐；
3. **`lear`** Lasso 自回归（Uniejewski/Weron 系）—— 电价预测的经典强基准，
   每小时一个 Lasso，特征为前 7 天同刻与邻刻价格 + 当日负荷；
4. **`lgbm_quantile`** —— 单一全局模型 × 三个分位（0.1/0.5/0.9），
   direct multi-horizon，一次输出 1..24 步；
5. **`mqcnn`** —— MQ-CNN（Wen et al., NIPS 2017, arXiv:1711.11053）：
   因果卷积编码过去 168 小时 → 与未来协变量拼接 → MLP 输出 24 步 × 3 分位，
   用 `median ± softplus(·)` 参数化保证分位单调，直接优化 pinball 损失。

`mqcnn` 与本论文的三处差异（须在报告中披露）：单序列、未使用 embedding、
未做 forking-sequences（直接多步已覆盖其主要收益）。

### 3.3 线上链路（`app/application/price_service.py`）

```
spot_prices ──┐
measurements ─┼─→ 小时网格对齐 ─→ PriceSeries ─→ 模型 ─→ 小时分位曲线 ─→ 15 分钟阶跃展开
天气预报 ─────┘        ↑
                  TOU 政策表（作为 tou_price 先验）
```

- **粒度**：模型在小时上训练/推理，对外按 15 分钟**阶跃**展开（同一小时内价格相同）。
  不插值——插值会造出现货市场里不存在的价格。
- **未来负荷**：用 `(日类型, 小时)` 的同刻均值画像，与负荷模块同源口径。
- **天气**：Open-Meteo 逐小时温度与 GHI；不可用时降级为 NaN（LightGBM 原生支持缺失），
  绝不阻断主链路。
- **降级链**（`model=auto`）：
  `LightGBM 工件存在 → 用工件` → `现货历史 ≥ 3 天 → 统计画像` → `无历史 → 政策表`。
  每一级都在响应的 `model` / `method_note` 字段里显式说明，不静默降级。
- **统计画像只用最近 14 天**：电价水平不平稳（燃料、来水、机组结构都会整体移动价格带），
  用半年样本的经验分位数会在 regime shift 后系统性偏低。演示序列上 8—9 月约 0.3 元、
  12 月约 0.9 元，用全窗口会让 12 月的预测塌到 0.32 元。
- **统计画像输出中位数而非均值**：电价分布右偏（尖峰拉高均值），MAE 意义下中位数才是
  最优中心。因此它会比"近 14 天同时刻均值"低 15–20%，这是性质不是缺陷；
  需要无偏收益估算时应改用均值或对 p50 做偏度修正。

## 4. 实验

### 4.1 协议

- **滚动 origin**：对第 k 个任务日 D，只用 D 之前的数据训练与取特征；
- 15 个任务日各重新拟合一次，不是"训一次跑 15 天"；
- 每个任务日预测 24 小时，一次前向输出，不递归；
- 评价集 = 15 × 24 = **360 个点**，逐点对齐；
- 指标：MAE / RMSE / pinball(0.1,0.5,0.9) / PICP80 / Winkler80 / 峰值命中率 / 相对昨日延续的 skill。

复现：

```bash
cd backend
python scripts/run_price_benchmark.py \
  --data ../data/power/price/Price \
  --output data/forecast_experiments/price_benchmark \
  --models naive_daily,naive_weekly,lear,lgbm_quantile,mqcnn,gefcom_benchmark
```

### 4.2 结果

评价集：15 个任务日 × 24 小时 = 360 点，全部为**样本外**（每个任务日重新拟合）。

| 模型 | MAE | RMSE | Pinball(0.1/0.5/0.9) | PICP80 | Winkler80 | 峰值命中 | skill vs 昨日延续 |
|---|---|---|---|---|---|---|---|
| 昨日同时刻 `naive_daily` | 13.03 | 27.68 | 6.51 | — | — | 0.50 | 0（基准） |
| 上周同时刻 `naive_weekly` | 32.60 | 57.30 | 16.30 | — | — | 0.00 | −150.2% |
| GEFCom 官方基准 | 32.74 | 57.33 | 16.37 | — | — | 0.00 | −151.3% |
| LEAR（Lasso AR） | 12.17 | 21.69 | 6.09 | — | — | 0.50 | +6.6% |
| ~~MQ-CNN（arXiv:1711.11053）~~ | ~~10.77~~ | ~~18.32~~ | ~~3.54~~ | ~~68.1%~~ | ~~52.34~~ | ~~0.50~~ | ~~+17.3%~~ **作废** |
| MQ-CNN（泄漏修正后，2026-10-22 重跑） | 29.37 | 53.22 | 13.19 | 22.5% | 248.84 | 0.00 | −125.5% |
| **LightGBM 分位数** | **9.24** | **17.58** | **3.02** | **68.6%** | **44.52** | **0.75** | **+29.1%** |

> **2026-10-22 更正（必须阅读）**：原 MQ-CNN 数字是在**训练泄漏**下取得的——
> 基准脚本把全序列（含任务日之后的数据）喂给 MQ-CNN 训练，违反 §4.1 自己的协议。
> 截断修正后重跑（`data/forecast_experiments/price_benchmark_20261022/`，
> 同目录含逐点存档与 run_meta.json），MQ-CNN 真实水平为 MAE 29.37 /
> skill −125.5%，不如昨日延续。上表其余模型当时就是截断训练的，
> 数字仍然有效（重跑复现：naive_daily 13.03 / naive_weekly 32.60 完全一致；
> lgbm 9.52 vs 9.24 的差来自 lightgbm 版本差异，结论不变）。
> 修正 commit 同步改了 MQ-CNN 推理端（原实现依赖未来数据才能前向）。
> 详见 docs/18 §5。

- Pinball 单位与价格同量纲；`—` 表示该方法只产出点预测，报 pinball 无意义。
- PICP80 名义值应为 0.80，实测 0.68–0.69 → 区间**略窄**，见 §4.3 第 3 条。
- 峰值命中：实际最贵 4 小时中，预测也排进最贵 4 小时的比例。储能套利真正吃这个指标。

图：`data/forecast_experiments/price_benchmark/fig_metrics.png`（三指标对比）、
`fig_curve_task15.png`（尖峰日曲线）、`fig_by_task.png`（逐日失效点）。

### 4.3 读法

1. **LightGBM 分位数赢在"结构性"而非"容量"**：MAE 9.24，比昨日延续低 29.1%，
   比经典 LEAR 低 24.1%。主要贡献来自三处——① 分位数目标函数直接优化
   pinball，天然抗尖峰拖尾；② 锚定 origin 的 168h 滚动统计（波动状态）；
   ③ 日前负荷预报作为未来协变量。
2. **MQ-CNN 在单序列小样本场景不成立（修正后结论）**。修正训练泄漏后，
   MQ-CNN（MAE 29.37 / skill −125.5%）跑输全部统计基准——原因不变：
   单序列、约 2.5 年样本，深度模型的样本效率劣势是决定性的。
   **工程结论维持：生产链路用 LightGBM；深度路线只看 MQ-RNN 系
   （forking-sequences + 全局/局部解码器，见 docs/18），且必须先补
   日前负荷协变量再重跑同协议对决。**
3. **区间偏窄（PICP80≈0.68 < 0.80）**。三个分位数独立训练会低估尾部离散度；
   任务日集中在 6–7 月与 12 月，恰是价格尾部最厚的时段。工程上的补救：
   对外展示与调度决策时把 P10/P90 按经验覆盖率做一次保形校准（conformal），
   或直接改用 P05/P95。当前版本未做，避免"看起来更准"的假象。
4. **官方基准 = 上周同时刻**。本仓库 `naive_weekly` 在 14/15 个任务日与
   `BenchmarkN_P.csv` **逐点完全一致**（任务 7 官方给的是修订版 `_new3`，差异 < 5.5）。
   这既校验了任务切分与时间对齐无误，也说明该"官方基准"是确定性季节持续法，
   99 个分位数列完全相同，不能当作概率预测对手。
5. **失效日**：`fig_by_task.png` 显示 12-17 尖峰日所有模型都偏保守——实际最高
   229，预测最高 163。这类由寒潮+机组事件驱动的尖峰，纯统计模型无解，
   必须挂事件日历或燃料价格外生变量。

## 5. 系统接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/sites/{id}/price/forecast?days=1&model=auto&uncertainty=true` | 未来电价分位数曲线 + 套利窗口 |
| GET | `/api/v1/sites/{id}/price/history?days=30` | 历史现货电价与统计 |
| POST | `/api/v1/sites/{id}/price/spot-prices` | 导入现货序列 |
| DELETE | `/api/v1/sites/{id}/price/spot-prices` | 清空（重灌前用） |
| GET | `/api/v1/sites/{id}/accuracy?target=price` | 封存预测 vs 实测对账 |

### 5.1 响应要点

- `points[]`：`ts / p10 / p50 / p90 / tou_price / tier`；
- `summary.arbitrage`：按 P50 曲线求最优 2 小时充/放窗口，给出毛价差与含效率的净价差；
- `summary.holidays`：预测区间内的法定节假日（提示"这天负荷和电价形态会变"）；
- `data_note`：数据来源，`gefcom*` 会触发前端显著提示"这是演示序列，不是本地电价"。

### 5.2 精度对账

`forecast_archive` 每次预测原样封存；`spot_prices` 到位后按小时对账，
口径为 MAE/RMSE + 相对"昨日同时刻"的 skill + PICP80。
电价**不用** `skill = 1 − RMSE/Cap`：电价没有装机容量做分母，那个公式在这里没有意义。

### 5.3 多日预测

工件只在 24 步上训练。`days > 1` 时按**天递归**：第 d 天的 P50 写回历史再发第 d+1 天，
每天仍只做 1..24 步的直接多步预测。理由：硬喂 horizon=72 的特征组合属于外推，会失真；
天级递归的误差累积远小于小时级递归，且与"每天重发"的生产节奏一致。此性质在
`method_note` 里显式披露，不假装是一次前向 72 步。

## 6. 局限

1. **演示序列不是本地电价**。仓库内置的演示链路导入的是 GEFCom2014-P（PJM 区域 1 真实 LMP），
   价格水平与波动形态不属于广东。上线前必须换成省电力交易中心/代理购电的真实序列。
2. **未建模机组检修、报价策略、网络阻塞**。LMP 的尖峰很大一部分来自阻塞与机组组合，
   纯统计模型对这些结构性事件无解；工程上应挂"事件日历"再叠加。
3. **节假日表按年维护**。`price_calendar.py` 里的法定节假日是硬编码，国务院每年 11 月发通知，
   新一年必须补表，否则节假日特征会整年错位。
4. **分位数是边际的**。P10–P90 是按点估计的边际区间，不是整条曲线的联合覆盖，
   连续多小时的区间联合覆盖率会低于 80%。
5. **负电价**。国内部分现货试点已出现负电价，本模块在数据层允许负值，
   但统计画像在极低样本量下会向 0 收敛，需用真实负电价样本重新标定。

## 7. 第二阶段扩展（校准 / 事件 / 区间调度 / 真实数据接入）

详细实验记录见 `reports/price_forecast/interval_dispatch.md`。四条线一句话结论如下。

### 7.1 区间校准（`price_calibration.py`）

保形预测（CQR）：训练窗口尾部留出校准集，按提前小时学外扩量，只外扩不收缩。
GEFCom2014-P 上 PICP80 由 **0.642 → 0.756**（严格滚动）、**0.803**（校准集同分布的 LODO 估计）。
严格滚动没到 0.80，是因为 12 月寒潮尖峰在校准窗口里没有先例——这是可交换性假设被打破，
不是实现问题。上线需配**覆盖率监控**（PICP80 < 0.75 触发人工复核），而不是假设永远 0.80。

站点训练入口默认启用 `--cal-days 45`，透传给 `LgbmQuantileForecaster.fit()`；
可显式指定其他天数，`--cal-days 0` 关闭校准。保护条件按**实际训练窗口内有限值价格的
小时数 / 24**计算有效历史（缺失小时不计入，不能只看数据库首尾日期跨度）：
不足 `2 × cal_days` 天时打印提示，并自动以 `cal_days=0` 训练。
例如默认校准需要至少 90 天有效小时价格；恰好 90 天时保留校准。
`--min-days` 仍是独立的最低训练历史门槛，`--cal-days` 不接受负数。

```bash
cd backend
python scripts/train_price_model.py --site S-XXX --db data/storage.db --cal-days 45
```

重新训练后的 `data/price_models/<site>/manifest.json` 在启用且完成校准时保存非空
`calibration`，并记录 `cal_days_requested`、实际 `cal_days` 和 `valid_price_hours`。
现有 `/api/v1/sites/{id}/price/forecast` 会从工件读取校准量，外扩 P10/P90；
P50 不因应用校准而改变，响应结构、元/kWh 单位及带时区的对外时间戳保持不变。
旧工件缺少 `calibration` 时仍按原方式加载；只有重训后才会启用新校准量。
本项是训练到在线推理的机制接通，不代表真实站点 PICP80 已达 0.80；覆盖率监控仍待 T2。

### 7.1.1 覆盖率监控（T2）

电价专用预测接口会按站点本地自然日封存完整的 P10/P50/P90 曲线；实测到位后按小时
聚合（24 个小时点）计算 PICP80，并写入 `forecast_accuracy`。15/30/60 分钟封存粒度
都只在小时级对账，不套用功率模块的 96 点日级判断。缺少原始区间的旧工件只提供点误差，
其 `interval_coverage` 为 `null`，不会被当成 0，也不会参与低覆盖连续天数。

`GET /api/v1/sites/{id}/price/forecast` 和
`GET /api/v1/sites/{id}/accuracy?target=price` 暴露：

- `interval_coverage`：最近已对账日的 PICP80；
- `warning`：同一模型连续 3 个自然日低于 0.75 时为
  `price_interval_undercoverage`，否则为 `null`；
- `consecutive_low_days`（精度摘要）：连续低覆盖天数；日期缺口、未知覆盖率或达到
  0.75 都会中断计数。

这里的告警是覆盖率异常的运营信号，不是“校准已失效”或“模型效果已验证”的结论；
没有真实区间对账数据时接口返回 `null`。

### 7.2 事件日历（`event_calendar.py` + `configs/events.guangdong.yaml`）

机组检修、寒潮/高温、燃料价格指数三类外生信息，共 7 个特征，追加在 `FEATURE_COLUMNS` 末尾。
无事件配置时整列 NaN 并自动剔除，不影响无事件市场。**GEFCom 无事件数据，故不参与离线基准**，
只能在线验证——这条必须在汇报里说明，不能声称"事件特征提升了精度"。

### 7.3 区间调度（`dispatch_interval.py`）

三条区间约束：充电按 P90 计价、放电按 P10 计价、鲁棒价差门槛（`P10_top × η − P90_low / η`）。
消融实验（同一份预测、同一套物理约束、用实际电价结算、叠加 AR(1) 预测偏差）：

- **成本不暴涨成立**：溢价 0.03%–1.5%；
- **鲁棒性提升不成立**：偏差 10% 时误充率 1.74%→0，但偏差 30% 时反而 12.0%→15.2%；
- 限制：15 天样本不足；“保底 SOC 从未击穿”的旧说法有误，30% 档已有储备线不足，且旧风险指标量纲不等于缺电电量。

结论：作为**可选保守档位**上线，但不得声称"鲁棒性显著提升"。

**T3 重实验（90 天，250 kW / 500 kWh）**：每天重训校准 LightGBM，2013-08-01 至
2013-10-29 连续评估，金额统一为元，价格为元/kWh，预测时间戳带时区。
10%／20%／30% 附加 AR(1) 扰动下，区间组误充率为 0.00%／5.45%／19.34%，
低于 P50 的 0.28%／9.72%／22.00%；但电池尖峰供能缺口更大，计入系数 2 的缺口惩罚后
总评分溢价为 +3.86%／+2.36%／+1.26%。**不把 `interval_robust` 提为默认，默认仍为 `rule_v1`。**
对照是同一引擎的 P50 消融，尚未直接比较 `rule_v1`。供能缺口为尖峰时目标放电与实际放电
之差的积分，是风险代理，不是停电量；PV/负荷为合成曲线、SOC 每日重置，不能外推真实广东
运行或全年收益。完整定义、保留的负面结果与复现见
[T3 报告](../reports/price_forecast/t3_reexperiment.md)，原始数据见
[summary.csv](../reports/price_forecast/t3_interval_dispatch/summary.csv)。

### 7.4 广东电价接入（`price_ingest.py`）

`import_spot_price.py --profile guangdong` 处理四件事：元/兆瓦时→元/千瓦时自动识别、
无时区时间戳按本地解释、15 分钟按均值聚合到小时、负电价保留。导入时打印体检报告
（单位、重复、缺失、负电价点数）。

## 8. 参考文献

1. Wen R., Torkkola K., Narayanaswamy B., Madeka D. *A Multi-Horizon Quantile Recurrent
   Forecaster*. NIPS 2017 Time Series Workshop. arXiv:1711.11053. —— MQ-CNN 原始论文。
2. Hong T., Pinson P., Fan S., Zareipour H., Troccoli A., Hyndman R.J. *Probabilistic energy
   forecasting: Global Energy Forecasting Competition 2014 and beyond*. International Journal
   of Forecasting, 2016. —— GEFCom2014 数据集与赛道设定。
3. Nowotarski J., Weron R. *Recent advances in electricity price forecasting: A review of
   probabilistic forecasting*. Renewable and Sustainable Energy Reviews, 2018. —— 概率电价预测综述。
4. Uniejewski B., Nowotarski J., Weron R. *Automated Variable Selection and Shrinkage for
   Day-Ahead Electricity Price Forecasting*. IEEE Transactions on Power Systems, 2016. —— LEAR。
5. Lago J., De Ridder F., De Schutter B. *Forecasting spot electricity prices: Deep learning
   approaches and empirical comparison of traditional algorithms*. Applied Energy, 2018.
6. Koenker R., Bassett G. *Regression Quantiles*. Econometrica, 1978. —— 分位数回归与 pinball 损失。
