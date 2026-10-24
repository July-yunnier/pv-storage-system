# 21 · LLM-as-formulator 与事件日接管实验

## 目的与边界

本实验把 LLM 从数值调度器改为高层策略建模者。LLM 只输出有限枚举的事件场景、SOC 备用下限、放电窗口和风险权重；96 个 15 分钟动作仍由既有规则函数生成，并通过未修改的 `execute` 安全执行器结算。LLM 不读取当天实测，不输出动作序列，也不能改变物理 SOC 或功率边界。

普通日沿用 E2 预测驱动规则；只有发报时刻已知的事件日调用 formulator。当前离线事件标签是 `synthetic_demo_fixture`，用于验证协议和接口，不能作为广东真实事件收益证据。

## 可复现实验

```bash
cd backend
python scripts/run_official_experiment.py \
  --agent llm-formulator --mock-offline --event-mode events-only \
  --output-dir ../reports/official_experiment/llm_formulator_mock
```

结果目录包含 `p0_replay/formulator_day_results.csv`、`p0_replay/summary.json`、`formulator_policy_archive.jsonl` 和 `run_meta.json`。策略归档在结算前写入；历史摘要只包含目标日前数据，时间戳带 `Asia/Shanghai` 时区。`--no-mock-offline` 从环境变量加载 OpenAI-compatible 的 `LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_API_KEY`。

## 指标口径

所有尾部指标按日成本（元）计算。`VaRα` 是日成本的 α 分位数，`CVaRα` 是成本不低于该 VaR 的日样本均值；本实验固定报告 α=0.90 和 0.95。最差日为日成本最大值。人工干预次数只统计 `manual_review_required`、LLM fallback 或安全层改写策略的记录，不把模型调用次数当作人工干预。

## 离线结果

2019-04-01 至 2019-04-30、30 天、4 个合成事件日的 mock 回放结果：

| 指标 | 结果 |
|---|---:|
| E_LLM_FORMULATOR 总电费 | 197,518.56 元 |
| 相对 E2 增量节省 | -596.30 元 |
| 相对 E2 日配对 bootstrap 95% CI | [-72.31, 22.65] 元 |
| CVaR90 | 9,221.93 元 |
| CVaR95 | 9,271.59 元 |
| 最差日成本 | 9,355.08 元 |
| 事件日数量 | 4 |
| 人工干预次数 | 0 |
| SOC 违规次数 | 0 |

该结果只说明新接口和事件选择逻辑在演示数据上可复现且安全；CI 包含 0，不能宣称相对 E2 显著更省，也不能外推为真实广东事件日收益。

## 后续正式验证

正式收益结论需要替换为带发布时间的广东检修、寒潮、高温、限电和负电价事件源，在独立窗口上运行同样的事件日选择规则，并继续报告 E2 配对差异、CVaR、最差日、人工干预次数、SOC/功率违规和事件日子集 CI。新结果不得改写或混入已封存的两参数 E_LLM 实验。
