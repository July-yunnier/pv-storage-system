# 欧洲预训练 → 中国典型电价微调

本实验把公开欧洲日前市场价格用于基础模型训练，再把中国典型分时电价样本作为域适配微调集。它只验证迁移训练链路，不能替代真实广东现货验收。

## 输入

- 欧洲预训练：`/Users/july/Documents/ChatGPT/光储协同优化/free_electricity_csv/european_power_prices_2026-04-26_to_2026-10-22/DE-LU_180d_15min_raw.csv`
  - UTC、15 分钟、EUR/MWh；由欧洲公开数据包整理。
  - 预处理时按 `7.8 CNY/EUR ÷ 1000` 转为元/kWh，并聚合到小时。
- 中国微调：`/Users/july/Documents/ChatGPT/光储协同优化/china_synthetic_csv/china_typical_price_90d_15min_rich.csv`
  - Asia/Shanghai、15 分钟、元/kWh。
  - `is_synthetic=True`，依据广东分时电价规则构造，不是官方历史成交或结算数据。

输入文件在仓库外保存；来源、生成种子和 SHA-256 见各自 `MANIFEST.json`。禁止把中国样本描述为广东真实现货。

## 切分与运行

欧洲最近 14 天作为预训练留出集；中国前 60 天用于微调，最后 30 天只用于微调后评估。模型按小时训练，微调通过 LightGBM `init_model` 继续训练三套 P10/P50/P90 分位数模型。

```bash
cd backend
../光储系统/backend/.venv/bin/python scripts/run_price_transfer.py \
  --europe-csv /Users/july/Documents/ChatGPT/光储协同优化/free_electricity_csv/european_power_prices_2026-04-26_to-2026-10-22/DE-LU_180d_15min_raw.csv \
  --china-csv /Users/july/Documents/ChatGPT/光储协同优化/china_synthetic_csv/china_typical_price_90d_15min_rich.csv \
  --output ../reports/price_forecast/transfer_eu_china
```

结果写入 `transfer_report.json`。当前一次运行的结果（元/kWh）为：

| 评估集 | P50 MAE | PICP80 | 平均区间宽度 |
|---|---:|---:|---:|
| 欧洲预训练留出 | 0.31384 | 0.4455 | 0.48896 |
| 中国典型样本留出（微调后） | 0.18274 | 0.4104 | 0.41070 |

这里的 PICP80 未做保形校准，不能与生产验收目标直接比较；中国指标只说明对该合成形状的适配情况，不说明广东真实预测能力。
