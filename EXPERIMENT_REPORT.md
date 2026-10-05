# 光伏预测与储能后端交付报告

## 已固定的规则

实验协议在 [`configs/experiment_protocol.json`](configs/experiment_protocol.json) 中冻结，随机种子为 42，预测时距为 15 min、1 h、4 h，测试集按日历时间最后 30% 切分。训练占前 60%，验证占接着的 10%，权重只能按验证集选择。输入、归一化、Cap、缺失值和 NWP 可用时间约束都写进配置。每轮实验保存 `config.json`、`weights.json` 或模型权重、`predictions.csv`、`metrics.json` 和 `code_version.json`。

96% 门槛按每个时距同时满足 `1 - MAE/Cap` 和 `1 - RMSE/Cap` 执行。天气标签缺失时保存为 `unknown`，不会从误差反推晴天或阴天。当前数据没有可验证的 NWP 预报 vintage，因此多模态 NWP 结果未冒充完成。

## 后端修复

- 日内光伏预测的槽位包含 `issued_local.minute`，并拒绝无时区时间。
- 日内历史截断到发布时间，后来的观测不能泄漏进预测；指数衰减按实际分钟数计算。
- 调度在超过需量上限时主动削峰，并在较早峰段按后续需量和更高价格峰预留 SOC 与日循环预算。
- 对输入时间戳、连续性、SOC、有限数值、缺失值和初始 SOC 增加校验。
- 已检查所有 `__init__.py`，包路径可以完整导入。仓库当前版本没有 `Continue` 大写语句，未发现该拼写错误。

## 实际基线结果

指标 JSON 在 `runs/` 下。已经完成并可复核的结果如下，百分比为归一化准确率，括号内为是否同时通过两项 96% 门槛：

| 数据集 / 模型 | 15 min | 1 h | 4 h |
|---|---:|---:|---:|
| SKIPP'D Smart Persistence | MAE 97.36%, RMSE 93.17%（否） | 94.53%, 90.35%（否） | 87.97%, 81.61%（否） |
| SKIPP'D 历史均值 | 94.56%, 91.35%（否） | 94.11%, 90.94%（否） | 93.65%, 90.75%（否） |
| SKIPP'D pvlib 晴空代理 | 84.39%, 81.65%（否） | 84.70%, 81.76%（否） | 88.51%, 84.95%（否） |
| 国家电网站点 1 持续性 | 97.77%, 95.41%（否） | 93.39%, 88.42%（否） | 78.71%, 67.85%（否） |
| 国家电网站点 1 历史均值 | 93.25%, 88.39%（否） | 93.25%, 88.39%（否） | 93.25%, 88.39%（否） |
| 国家电网站点 1 pvlib 辐照度持续性 | 93.67%, 86.72%（否） | 90.47%, 82.74%（否） | 75.18%, 63.01%（否） |

这些是统一时间切分下的真实回测，不能支持“96% 已达标”的结论。SKIPP'D 的额定 DC 容量按论文 30.1 kW 记录，但可用 AC Cap 未提供；国家电网站点的 Cap 使用文件名额定 MW 转 kW，比赛要求的可用容量仍需组长确认。容量口径变更后必须重新跑全套实验。

## 模型与数据状态

- 官方 SUNSET notebook 已保存于 `research/SUNSET_forecast.official.ipynb`，其原始协议为 16 帧、64×64、15 min 单点预测，与本轮统一协议不同，尚未宣称复现完成。
- 当前融合模型权重已复制到 `research/local_models/` 并记录 SHA256。原脚本把测试集用于选权重、各分片自行归一化、用观测最大值代替 Cap，不能直接作为本轮达标结果。
- 官方 AMPDNet 代码在 `research/official-ampdnet/`。站点 1 CPU 训练已跑两轮，最佳验证 MSE 从 0.01006 降到 0.00930，耗时约 68 分钟，已保存 `runs/state_grid_1/ampdnet_official/weights.pt` 和日志；测试评估未完成，不能宣称 AMPDNet 达标。其余站点未启动长训练。
- ConvNeXt、Swin 的可控骨干和 FusionSF 风格适配器已写入 `experiments/models.py`。VMamba、Spatial Mamba 需要官方可用适配器，当前没有擅自用近似网络替代。
- 概率预测已实现验证集残差 P10/P50/P90 归档，包含 pinball、区间覆盖率和区间宽度；这不是 SkyGPT。

## 验证命令

```bash
.venv/bin/python -m pytest -q product/backend
.venv/bin/python -m pytest -q product/backend/tests/test_pv_revision.py product/backend/tests/test_dispatch.py
.venv/bin/python -m experiments.prepare --dataset all
.venv/bin/python -m experiments.baselines --dataset skippd
.venv/bin/python -m experiments.baselines --dataset state_grid_1
.venv/bin/python -m experiments.plots
```

最后一次后端测试为 170 passed。结果图在 `reports/skippd/`。价格预测未纳入本轮。
