# MQ-RNN 电价预测适配与验证

2026-10-21。用户提供的 arXiv:1711.11053v2 是 Wen、Torkkola、Narayanaswamy、Madeka 的 **A Multi-Horizon Quantile Recurrent Forecaster**，提出 MQ-RNN/MQ-CNN。上一版将它识别为 DeepAR，归属错误，现已修正代码、默认路由、前端和文档。下载的 tar.gz 是 LaTeX 论文源文件及插图，并非作者模型代码或预训练权重。

## 论文结构与本系统实现

实现：`product/backend/app/domain/forecast/price_mqrnn.py`。
训练：`product/backend/scripts/train_price_mqrnn.py`。
数据校验及日历：`product/backend/app/domain/forecast/price_data.py`。

对应原文 §3.1–3.4、图1(b)、§4.2：

1. LSTM 编码当前及过去已观测价格与日历；输入还包括原生日、周滞后价格和序列嵌入。
2. 全局 MLP 同时接收编码状态和整段未来已知日历，产生每个预测步专用 context 及一个共享 context。
3. 同一个局部 MLP 在每个预测步组合上述两种 context 和该步日历，直接输出 p10/p50/p90。p50 是点预测，没有递归价格回灌或高斯路径采样。
4. 使用 pinball loss。训练时编码序列块一次，在块内每个时刻接共享解码器，所有 fork 的误差共同反向传播；这是真正的 forking-sequences，而非将每个原点各自编码的普通滑窗训练。块间重置状态，采用截断 BPTT。
5. 训练结束或缺口之前，仍保留可用的短期目标，只遮罩超出边界的标签，不把验证/测试价格作为训练标签。

工程适配与原文的差异：分位数头用 `p50 ± softplus(间隔)` 保证不交叉，允许整体为负；每序列用训练段的 `1+mean(abs(price))` 固定缩放；使用日/周两个滞后，而非原文电价实验中的168维滞后向量。默认状态维30、7天编码块、4天直接预测。包含7天周滞后的预热历史，因此默认在线要求连续14天原生价格。

当前没有未来负荷预报版本/发布时间的数据接口，所以只使用可提前确定的日历特征，没有把未来实测负荷或光伏当作已知输入。原文 GEFCom 电价实验还使用分区和全系统负荷预报、节假日特征，预测24小时，并输出/插值99分位数；本实现不是原论文竞赛成绩的完整复现。p10/p90 是逐时刻的边际分位数，不代表跨时刻联合场景。

## 两省数据配置

两张配置截图对应已有 `product/configs/tariff.jiangsu.yaml` 和 `tariff.guangdong.yaml`，均为 **TOU 示例**。时间段左闭右开，跨午夜区间保留原意，输配及基金为0.21元/kWh。

| 地区 | 尖：电能量/合计 | 峰：电能量/合计 | 平：电能量/合计 | 谷：电能量/合计 |
| --- | --- | --- | --- | --- |
| 江苏 | 1.25 / 1.46 | 0.95 / 1.16 | 0.65 / 0.86 | 0.35 / 0.56 |
| 广东 | 1.25 / 1.46 | 1.05 / 1.26 | 0.65 / 0.86 | 0.28 / 0.49 |

江苏每天谷8、平9、峰5、尖2小时；广东每天谷8、平9、峰4、尖3小时。现有 YAML 标注为演示占位，不能当成最新正式政策文件。

`price_model=auto`：

- `tou`：已知政策价格直接展开，无需模型或历史，p10=p50=p90。
- `spot` / `tou_plus_spot`：MQ-RNN 预测电能量价，加一次 `transmission_and_fund`。遵循系统约定：现货价替代 TOU 电能量部分，不叠加 TOU 电能量价。

缺少真实现货历史/匹配权重时返回422，不生成虚构现货曲线。固定 TOU 不用于合成训练集并宣称模型精度。此前 DeepAR 仍保留为显式 `price_model=deepar` 的兼容基线（真实参考为 arXiv:1704.04110），前端和自动现货路由均使用 MQ-RNN。两种权重格式不兼容，不能直接互换。

## 训练与接入

训练 CSV：`ts,price[,series_id]`，带时区、时间严格递增、价格有限，允许负数；价格仅包含电能量部分。在线必须为 CNY/kWh，原生15或60分钟。不能把总零售电价重复加附加费，也不能将外币/MWh模型标成人民币/kWh。

```bash
.venv/bin/python product/backend/scripts/train_price_mqrnn.py \
  --csv /absolute/path/site_price.csv \
  --output product/backend/data/training_artifacts/price_mqrnn_site.pt \
  --unit CNY/kWh --timezone Asia/Shanghai --resolution-minutes 15 \
  --series-id SITE_ID --context-days 7 --horizon-days 4 --epochs 20
```

所有序列共用绝对时间70%/15%/15%切分。训练尺度仅拟合训练段，验证 pinball 选择 checkpoint，测试仅用于最终报告。默认遇缺口拒绝；显式 `--skip-gapped-windows` 将训练拆成连续段，并跳过跨缺口的完整评估窗口，无插值补齐。完整验证/测试目标窗互不重叠，历史可引用此前已知实测。

产出 `.pt`、`.manifest.json`（参数、数据/论文哈希、训练日志、环境和测试指标）、`.test.csv`（逐点真值、分位数及基线）。至少应有足够数据构造独立验证/测试窗口，14天只是推理最低上下文要求。

在后端运行环境设置（仅修改示例配置，未改动现有实际环境）：

```dotenv
PRICE_MODEL_PATH=/absolute/path/price_mqrnn_site.pt
```

通过测量接口上传同一站点历史：`data_type=price, unit=CNY/kWh`，原生频率用 `resolution_minutes=15` 或 `60`。前端“数据管理”支持价格上传；“电价中心”选择“遵循站点电价方案”或“MQ-RNN 现货概率预测”，查看24/48/72小时曲线。

```json
{
  "site_id": "SITE_ID",
  "targets": ["price"],
  "horizon_hours": 24,
  "resolution_minutes": 15,
  "options": {
    "price_model": "mqrnn",
    "price_series_id": "SITE_ID",
    "include_uncertainty": true
  }
}
```

`POST /api/v1/forecast` 默认预测次日零点起。四天直接输出覆盖今日到次日零点的间隔加最长72小时请求，按实际预测原点切片；超出已训练步数拒绝，不能递归扩展。小时模型仅将同一小时结算价展开到四个15分钟点，不宣称原生15分钟精度。模型的单位、时区、序列、训练/验证截止时间必须匹配，输入严格早于 issued_at，缺口/陈旧历史拒绝。当前学习模型不支持 intraday。

返回 `price_mqrnn/v1`、p50点、可选p10/p90、权重哈希；components区分电能量p50与附加费。`price_num_samples` / `price_seed` 仅用于旧 DeepAR，MQ-RNN 不使用采样参数。

## 现有 GEFCom 数据独立实验

输入 `reports/reused_gefcom/price_source.csv`，原始25,943个小时点，UTC，保留 `source_price_unit`。显式跳过跨缺口窗口。仅训练新模型一次预设20轮，以验证集选择第17轮；没有用测试集选择超参数。复现：

```bash
.venv/bin/python product/backend/scripts/train_price_mqrnn.py \
  --csv reports/reused_gefcom/price_source.csv \
  --output product/backend/data/training_artifacts/price_mqrnn_gefcom_hourly.pt \
  --unit source_price_unit --timezone UTC --resolution-minutes 60 \
  --series-id GEFCom2014-P --epochs 20 --horizon-days 4 --skip-gapped-windows
```

测试39个不重叠的96小时窗口，共3,744点；每个原点只引用此前观测，同一原点下基线也仅重复最近已知一日或一周：

| 方法 | MAE | RMSE |
| --- | ---: | ---: |
| MQ-RNN p50 | 8.7395 | 20.3152 |
| 前一日重复 | 10.9486 | 24.7140 |
| 前一周重复 | 12.0545 | 26.9086 |

三个分位数的平均 pinball 为3.1556；p10/p90实际覆盖率81.36%，平均区间宽22.2454（源数据价格单位）。这是本次单一数据集和切分下的结果。此前 DeepAR 报告为24小时窗口，不能与本次96小时结果直接横向比较；两者均不等同论文的12个竞赛任务评估。

目前提供的站点数据是固定TOU配置，没有本站真实现货训练数据。本次GEFCom模型因单位/时区不匹配会被在线接口拒绝，未安装为生产站点模型。需使用真实本站数据训练和独立回测后，才能报告本站预测精度。价格概率模型已接入预测API与前端；现有规则调度/规划仍使用TOU，不会自动消费该模型的分位数，也未实现随机优化。

## 验证

`product/backend/tests/test_price_mqrnn.py` 验证共享fork与逐原点解码等价、编码因果性、整段未来日历通路、部分标签遮罩及缺口、分位数梯度和次序、真实15分钟合成训练链、预测间隔切片、费用/单位/未来隔离、两省全天及跨午夜曲线、现货自动路由和缺模型报错。合成数据只用于功能测试，不作精度证明。

原论文依据：本地 `1711.11053v2.pdf` 第3–4页结构/训练方案及第7页GEFCom实验，配合 tar.gz 中 `MQRCNN_final_version.tex` 核对；本次没有修改原论文或源文件。

本次完整检查：后端235项通过；前端 `npm run build` 通过。现有依赖弃用提示及前端大包提示未影响构建。运行环境、代码哈希与验证记录见 `price_mqrnn_gefcom_hourly.environment.json`。

## GitHub仓库同步验证

本次将电价相关实现逐项合入本文件所在的“光储系统”Git仓库，保留该仓库原有功能。同步后的后端241项通过、20项因可选 NeuralForecast 依赖未安装跳过，前端构建通过。补齐了原有离线预测请求类型、调度请求字段和前端精度查询导出，以恢复现有接口的兼容性。训练输入 `reports/reused_gefcom/price_source.csv` 是工作目录已有外部数据，未随本次上传；复现时需自行提供同一原始数据。
