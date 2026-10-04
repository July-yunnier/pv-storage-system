# 优化模型与计算口径

## 1. 统一符号

对每个时间段 `t`：

- `load_t`：负荷功率，kW
- `pv_t`：光伏功率，kW
- `charge_t`：充电功率，kW，非负
- `discharge_t`：放电功率，kW，非负
- `soc_t`：时段开始 SOC，范围 `[soc_min, soc_max]`
- `price_t`：到户电价，CNY/kWh
- `grid_import_t`：电网购电功率，kW
- `grid_export_t`：上网功率，kW

净负荷：

```text
net_load_t = load_t - pv_t
```

## 2. 功率平衡

```text
grid_import_t - grid_export_t
  = load_t - pv_t + charge_t - discharge_t
```

禁止反送电时：

```text
grid_export_t = 0
grid_import_t >= 0
```

## 3. SOC 计算

首版建议将充电效率和放电效率拆开：

```text
soc_energy_{t+1}
  = soc_energy_t
  + charge_t × charge_eff × Δt
  - discharge_t / discharge_eff × Δt
```

如果输入只有 round-trip efficiency，应在配置层转换为明确的 charge/discharge efficiency，并记录转换假设。

## 4. 规则式 Dispatch（MVP）

基本规则：

1. 谷段优先充电。
2. 光伏余电优先充电。
3. 尖峰和高峰优先放电。
4. SOC 达到上限时停止充电。
5. SOC 达到下限时停止放电。
6. PCS 功率和最大循环次数始终生效。
7. 禁止反送电时，充电不能造成电网倒送。

规则策略必须返回实际应用的约束，不能只返回动作结果。

## 5. LP Dispatch（后续）

目标函数建议为：

```text
min Σ(price_t × grid_import_t × Δt)
  + degradation_cost
  + export_penalty
```

约束包括：

- 功率平衡
- SOC 上下限
- 充电和放电功率上限
- 充放电互斥
- 循环次数
- 禁止反送电
- 期初和期末 SOC
- 需量控制

## 6. 循环次数

必须使用明确的等效完整循环定义。建议：

```text
equivalent_full_cycles
  = Σ(discharge_energy_t) / usable_capacity_kwh
```

不能简单用“充电次数”代替完整循环。

## 7. Sizing

Sizing 不直接凭一个公式决定容量，而是：

1. 生成候选 PCS 功率和电池容量组合。
2. 对每个候选组合调用统一 Dispatch。
3. 汇总典型日或全年代理日收益。
4. 计算 CAPEX、OPEX、衰减、更换、残值和折现现金流。
5. 选择满足约束且 NPV 或综合目标最优的方案。

## 8. 经济性口径

至少记录：

- 电池单价
- PCS 单价
- 安装和并网成本
- 年运维成本
- 电池寿命
- 更换年份和成本
- 折现率
- 残值
- 价差情景
- 需量电费单价和计算基准

简单回本不能替代 NPV。所有经济性结果必须附带 assumptions。

## 9. 当前未闭合事项

- 需量电费按月最大需量的具体算法。
- 典型日如何映射全年。
- 电池衰减是否影响每日可用容量。
- 期末 SOC 是否必须回到期初 SOC。
- 现货模式下到户电价各组成部分的实际来源。
