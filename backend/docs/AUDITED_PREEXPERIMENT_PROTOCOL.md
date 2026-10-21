# GEFCom收敛预实验执行与交接

适用范围：AIC第八届AI+能源赛道科技创新组光储协同优化系统的GEFCom预研究，不是正式主实验，不是现场工程性能。

## 执行顺序

1. 冻结数据、源代码SHA256、seed=42及超参候选协议，保存environment.lock。
2. LSTM与TCN固定配置基线分别训练至验证早停或120轮预算结束。
3. 优化TCN在两个预声明学习率/batch size候选中仅用验证Pinball选择。
4. 固定选中超参，逐项移除滚动统计、噪声及外生特征。全部消融使用相同样本时间戳。
5. 所有任务/时长完成训练与验证选择后，先落盘selection_frozen.json，再生成测试预测和指标。
6. Persistence/ARIMA仅作固定经典基线，参数只在训练段估计，区间是验证集残差后处理。
7. 从已保存指标生成报告，不挑选测试表现最好的模型或只保留有益优化。

## 关键代码位置

`app/domain/forecast/quantile.py`

```text
统计参数 <- 训练段历史（均值/标准差）
输入 <- 历史值 + 已发生天气/负荷外生变量 + 历史周期编码
if use_rolling: 拼接因果24h滚动均值/标准差
if training and noise_std: 仅向标准化历史值添加噪声
context <- 两层LSTM 或 dilation=1,2,4 的因果残差TCN
P50 <- head中位输出
P10 <- P50 - softplus(下间隔)
P90 <- P50 + softplus(上间隔)
loss <- 目标h时刻三个分位数等权Pinball
按原时间顺序迭代（不shuffle）
仅验证损失选最佳checkpoint和早停
```

时间编码仅含小时、星期、月份正余弦。未来日历块未输入网络，不使用目标日未来实况。h=4时只有最后目标的头参与损失及评估，中间头不用于预测产品、展示或决策。

`scripts/run_audited_preexperiments.py`

输出trial的settings.json、artifact.json、model.pt、training_log.json、completed.json；每轮日志实时保存，完成标记携带权重SHA256。resume只复用相同协议/相同设置/权重哈希吻合的已完成试验，未完成试验会重新拟合，不伪称精确断点续训。代码或数据改变必须使用新输出目录。

`app/domain/forecast/frozen_pv.py`

只读权重加载与eval/no_grad推理。必须由光伏队友提供：已有权重路径、模型定义/推理代码、输入字段/图像尺寸和帧数、训练归一化参数、目标时间与标签对齐规则。不得猜测模型或重训替代。本轮不修改光伏网络和储能调度代码。

## 合理解释范围

- 所有学习模型均采用训练段Z-score；负荷Cap只作为报告的训练历史最大值统计代理。电价没有Cap。
- 原生分位数头不等于自动校准。80%覆盖率应同时看离80%的距离、区间宽度和Pinball。
- 同一GEFCom测试段已在旧粗筛和20轮探索中查看，历史暴露无法撤销。新流程选择仅看验证集，但仍只能称探索性预实验。
- 120轮是资源上限，不是自动收敛。validation_early_stop与max_budget_not_converged分别保存。
- 移除消融只能解释固定条件下的贡献，不能证明独立因果关系或普遍增益。
- 原生数据为小时级，1h与4h有真实目标；15min无真实目标，不能通过插值“补出”独立评估。
- 单种子、单时序切分没有统计显著性证据，主实验需新数据和更完整鲁棒性验证。

## 来源

TCN：Bai et al. 2018 https://arxiv.org/abs/1803.01271 。

直接分位数预测：Wen et al. 2017 https://arxiv.org/abs/1711.11053 ，该文含电价与负荷概率预测应用。

代码自行实现，未复制外部完整实现。PyTorch及statsmodels为BSD-3-Clause；数据协议未从本地归档核实，不能因为公开可下载就声称任意商用许可。更多参考由最终报告列出。
