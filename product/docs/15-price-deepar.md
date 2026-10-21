# 旧 DeepAR 基线与论文归属更正

原说明误将 arXiv:1711.11053 归为 DeepAR。用户提供的原文已证实该论文为 **A Multi-Horizon Quantile Recurrent Forecaster**，正式适配已替换为 MQ-RNN，见 [MQ-RNN电价说明](15-price-mqrnn.md)。

DeepAR 的正确参考为 [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)。此前代码、训练脚本和 GEFCom 实验保留为独立旧基线，只有显式 `price_model=deepar` 使用；默认现货与前端选择项已改为 MQ-RNN。

历史实验 `price_deepar_gefcom_hourly.manifest.json` 的模型参数与测试结果保留，论文归属字段及权重摘要已更正。其原生小时、24小时预测实验 MAE7.2107、RMSE15.5715，未优于同窗前一日基线MAE6.9726、RMSE14.5514。不要将其与新 MQ-RNN 的96小时窗口结果或原论文竞赛成绩直接比较。
