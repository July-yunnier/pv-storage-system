# 任务说明：实现储能容量规划（sizing）

> 本文档用于**把任务交接给其他人或 AI 编码工具**。
> 读这份 + 仓库现有代码，就能动手，不需要额外询问背景。
>
> ⚠️ 仓库是**私有**的，外部工具无法直接访问。请连同 `docs/系统设计.md`
> 与 `src/` 下的现有代码一起提供。

---

## 一、任务目标

在 `src/optimizer/sizing.py` 中实现**储能容量规划**，回答企业最关心的两个问题：

1. **该配多大？** —— 电池容量（kWh）与功率（kW）
2. **几年回本？** —— 投资回收期、IRR、NPV

**这与 `dispatch.py` 的区别：**

| | `dispatch.py`（已完成） | `sizing.py`（本任务） |
|---|---|---|
| 决策变量 | 每时段的充放电功率 | **容量 + 功率** + 每时段功率 |
| 目标 | 当日电费最小 | **全生命周期 NPV 最大** |
| 容量 | 给定 | **待求** |
| 输出 | 调度表 | 推荐配置 + 经济性 |

---

## 二、必须遵守的四条约定

违反任何一条都会导致返工，请先看完再动手。

### ① 零第三方依赖

只用 **标准库 `dataclasses` + `numpy`**。

**不要引入** `pydantic`、`pulp`、`scipy`、`cvxpy`。

理由：`scripts/demo_dispatch.py` 目前克隆下来直接能跑，不需要 `pip install`
任何东西。加依赖会破坏这个特性——队友接手时的环境门槛越低越好。

> 如果需要 LP 求解，先用**穷举 + 规则调度**实现（容量候选 10~20 个档位，
> 每个档位调用一次 `rule_based_dispatch` 评估收益）。这在本场景下精度足够，
> 因为峰谷时段已知，规则式调度已接近最优。

### ② 复用 `dispatch.py` 算收益，不要另写公式

**这是最重要的一条。**

sizing 评估某个候选容量能赚多少钱时，**必须调用 `rule_based_dispatch()`**，
不允许自己再写一套简化的收益公式。

理由：如果规划和运行用两套算法，会出现

```
规划说：3 年回本
实际运行：赚不到这个数
```

企业客户一眼就能看出矛盾。**规划与运行必须共用同一个收益内核。**

### ③ 所有地域相关参数走配置

电价、设备单价、寿命、折现率等**一律从 YAML 读**，代码里不出现具体数值。

站点配置见 `config/site.example.yaml`，省级电价见 `config/tariff/*.yaml`。
换地点 = 改配置，代码一行不动。

### ④ 需量感知必须开启 ⚠️

调用 `rule_based_dispatch()` 时**必须显式传 `demand_aware=True`**。

该参数默认为 `True`，但请显式写出——避免以后有人改动时误关，而这类错误
**不会报错，只会让收益数字悄悄偏大**，极难发现。

**理由（实测过的坑）：**

某示例站点上，不考虑需量的调度把并网峰值从 **542 kW 推到 706 kW**，
因为谷时 500 kW 满功率充电叠加在夜间 200 kW 负荷上。

| 方案 | 并网峰值 | 基本电费/月 | 电度电费/日 |
|---|---|---|---|
| 不配储能 | 542.1 kW | 20,601 元 | 4,661.8 元 |
| 储能·忽略需量 | **706.2 kW** | **26,836 元** | 4,464.8 元 |
| 储能·需量感知 | 542.1 kW | 20,601 元 | **4,376.7 元** |

**忽略需量时，基本电费一个月倒涨 6,235 元，全年 74,816 元**，
把省下的电度电费全吃掉还倒亏。

> **只算峰谷价差、不算需量电费，是工商业储能收益测算最常见的错误。**
> 这个坑我们已经踩过一次，不要在 sizing 里再踩第二次。

---

## 三、现有代码（请先通读这五个文件）

| 文件 | 职责 | 你需要用到的接口 |
|---|---|---|
| `src/models/site.py` | 站点档案 | `Site`、`StorageConfig`、`TariffConfig` |
| `src/models/schedule.py` | 调度结果 | `DispatchResult`：`baseline_cost_yuan`、`optimized_cost_yuan`、`charged_kwh`、`discharged_kwh`、`cycles_used`、`peak_grid_kw`、`baseline_peak_grid_kw`；派生属性 `savings_yuan`、`savings_pct`、`peak_reduction_kw` |
| `src/optimizer/dispatch.py` | **规则式日调度** | `rule_based_dispatch(load_kw, pv_kw, prices, storage, *, resolution_minutes=15, allow_grid_charge=True, allow_grid_export=False, demand_aware=True, initial_soc=None)` |
| `src/tariff/` | 电价 Provider | `get_provider(mode, **kwargs)`、`price_series(start, end, minutes)`、`PricePoint` |
| `config/site.example.yaml` | 全参数配置 | **`economics` 段已备好**：`discount_rate`、`project_years`、`battery_price_yuan_per_kwh`、`battery_life_years`、`battery_cycle_life`、`opex_ratio`、`include_replacement`、`residual_value_ratio` |

**代码风格参照 `dispatch.py`：**
- 全部用中文 docstring，说明"为什么这么做"而不只是"做了什么"
- 关键约束处加 ⚠️ 注释，标明踩过的坑
- 函数保持纯函数（输入对象 → 输出对象），不要读写全局状态

---

## 四、输出契约

`SizingResult` 的字段**必须与 `docs/系统设计.md` 第 5.1 节的 JSON 一一对应**：

```
recommendation   power_kw / capacity_kwh / duration_hours / chemistry
                 / round_trip_efficiency / depth_of_discharge / cycles_per_day
economics        capex_yuan / opex_yuan_per_year
                 annual_benefit{ arbitrage_yuan, demand_charge_saving_yuan, total_yuan }
                 simple_payback_years / irr / npv_10y_yuan
sensitivity[]    variable / change / payback_years     ← 至少 2 个变量
assumptions{}    tariff_source / battery_price_yuan_per_kwh
                 / battery_life_years / discount_rate
```

用 `dataclass` 定义，并提供 `to_dict()` 便于 API 层直接序列化。

### 两个必须有

**`sensitivity`（敏感性分析）**：至少覆盖「价差 ±20%」和「循环寿命 4000/6000 次」。
企业决策者一定会问"如果价差降了怎么办"。

**`assumptions`（假设摊开）**：**把假设写出来比给一个精确数字更可信。**
所有金额的来源要能追溯。

---

## 五、验收标准

请确保以下四条全部满足：

**① 能独立运行**

请一并写 `scripts/demo_sizing.py`，用与 `demo_dispatch.py` 相同的合成数据
（负荷 + 光伏 + 电价），**不需要任何外部数据文件**。

```bash
python scripts/demo_sizing.py
```

> 合成数据生成逻辑目前内联在 `scripts/demo_dispatch.py` 里。
> **建议先提取到 `scripts/_synth.py`**，两个 demo 共用，
> 避免复制粘贴后两边数据不一致（那样对比结果就没意义了）。

**② 输出包含完整的推荐配置与经济性指标**，字段与第四节一致。

**③ 规划与运行自洽**（关键）

拿 sizing 推荐的最优容量，喂回 `rule_based_dispatch()` 评估实际收益，
**两者应接近（差异 < 10%）**。若差异大，说明收益算法有分歧，必须修正。

**④ 敏感性分析有效**

价差 −20% 时的回本周期应**明显长于**基准情形。若几乎不变，说明敏感性
分析没接上真实计算，只是占位数字。

---

## 六、其他注意事项

- **单位统一**：功率 kW，电量 kWh，金额元，时间 15 分钟粒度（可配）
- **效率换算**：储能效率按**单程**计算（`one_way_eff = sqrt(round_trip_eff)`），
  不要在充放电两侧各乘一次 `round_trip_eff`——那会让收益高估约 10%
- **SOC 区间**：默认 `[0.10, 0.95]`，可用容量 = 容量 × 0.85
- **不要写死 10 年**：寿命与折现率从配置读，便于做敏感性
- **电价分档**：尖峰 / 高峰 / 平段 / 低谷四档，不要假设只有"峰平谷"三档

---

## 七、完成后

1. 跑一遍 `scripts/demo_sizing.py`，确认输出可读
2. 跑一遍 `scripts/demo_dispatch.py`，确认没有破坏已有功能
3. 更新 `README.md` 的实现进度表（把 `sizing.py` 标为 ✅）
4. 提交时写清"改了什么、为什么"
