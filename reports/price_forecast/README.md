# 电价预测实验结果（GEFCom2014-P）

> 由 `backend/scripts/run_price_benchmark.py` 生成，`backend/scripts/plot_price_benchmark.py` 出图。
> 方法与解读见 `docs/15-price-forecast.md`。

## 数据集

- GEFCom2014 Price Track，Zone 1 小时级 LMP（$/MWh），2011-01-01 起；
- 15 个任务日，每任务为单日 24 小时日前预测；
- 滚动 origin：每个任务日只用该日之前的数据重新训练；
- 评价集：15 × 24 = 360 点。

## 指标

| model | MAE | RMSE | pinball | PICP80 | Winkler80 | peak_hit | skill vs naive_daily |
|---|---|---|---|---|---|---|---|
| price_lgbm_quantile | 9.241 | 17.578 | 3.024 | 0.686 | 44.52 | 0.75 | +29.1% |
| price_mqcnn | 10.769 | 18.317 | 3.539 | 0.681 | 52.34 | 0.50 | +17.3% |
| lear | 12.173 | 21.689 | 6.087 | — | 121.73 | 0.50 | +6.6% |
| naive_daily | 13.027 | 27.679 | 6.513 | — | 130.27 | 0.50 | 0（基准） |
| naive_weekly | 32.598 | 57.304 | 16.299 | — | 325.98 | 0.00 | −150.2% |
| gefcom_benchmark | 32.740 | 57.328 | 16.370 | — | 327.40 | 0.00 | −151.3% |

## 协议校验

`naive_weekly` 与官方 `BenchmarkN_P.csv` 在 14/15 个任务日**逐点完全一致**
（任务 7 官方给的是修订版 `_new3`，最大差 5.4）。这说明任务切分、时区与逐点对齐无误。
官方基准的 99 个分位数列数值完全相同，实为确定性季节持续法。

## 图

- `fig_metrics.png`：MAE / pinball / 峰值命中三视图
- `fig_curve_task15.png`：2013-12-17 尖峰日的分位数曲线 vs 实测（昨日延续严重高估）
- `fig_by_task.png`：逐任务日 MAE，定位失效日

## 运行环境

- Python 3.13，LightGBM 4.7，torch 2.14（CPU）
- `origin_stride=3`（LightGBM），`mqcnn_stride=6`、`mqcnn_epochs=25`
- `run_meta.json` 记录耗时与逐任务日志
