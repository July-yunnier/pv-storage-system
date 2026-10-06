"""Separate scientific plots and honest report, generated only from saved runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCES = """
1. Hong et al. (2016), Probabilistic energy forecasting: GEFCom2014 and beyond. DOI:10.1016/j.ijforecast.2016.02.001. 数据协议未从本地归档核实，不等同于代码开源许可。
2. Hochreiter & Schmidhuber (1997), Long Short-Term Memory. DOI:10.1162/neco.1997.9.8.1735.
3. Bai, Kolter & Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271 （已在线核对题名及摘要）。参考作者仓库 https://github.com/locuslab/TCN （已核对MIT协议）。本项目自行实现因果残差TCN，非作者完整复现，未复制作者完整代码。
4. Wen et al. (2017), A Multi-Horizon Quantile Recurrent Forecaster. https://arxiv.org/abs/1711.11053 （已在线核对题名）。借鉴直接分位数输出思想，非原论文架构完整复制。
5. Koenker & Bassett (1978), Regression Quantiles. DOI:10.2307/1913643. Pinball/分位数回归理论来源。
6. Loshchilov & Hutter (2019), Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101. AdamW。
7. Bishop (1995), Training with Noise is Equivalent to Tikhonov Regularization. DOI:10.1162/neco.1995.7.1.108. 轻量噪声正则化思想，非该论文结论在本任务的保证。
8. 实现依赖：PyTorch BSD-3-Clause https://github.com/pytorch/pytorch；statsmodels BSD-3-Clause https://github.com/statsmodels/statsmodels；NumPy BSD-3-Clause；pandas BSD-3-Clause；Matplotlib PSF-based license。自编适配代码未设定仓库许可证，分发前需由项目方确认。
"""


def markdown_table(frame, columns):
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        result.append("| " + " | ".join(f"{row[c]:.5f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns) + " |")
    return "\n".join(result)


def effect(delta):
    return "提升" if delta < -1e-6 else "负收益" if delta > 1e-6 else "无效果"


def plot(root, task, result):
    figs = root / task / "figures"; figs.mkdir(parents=True, exist_ok=True)
    for horizon, rows in result.groupby("horizon"):
        base = rows[rows.variant.isin(["Persistence", "ARIMA", "LSTM_baseline", "TCN_baseline", "TCN_optimized"])]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flat, ["MAE", "RMSE", "Pinball", "interval_coverage_80"]):
            ax.bar(base.variant, base[metric], color="#28618b")
            ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=30)
            if metric == "interval_coverage_80": ax.axhline(.8, ls="--", color="#b24735", label="nominal 80%"); ax.legend()
        fig.suptitle(f"GEFCom PRE-EXPERIMENT {task} {horizon}h (historically exposed test)")
        fig.tight_layout(); fig.savefig(figs / f"{horizon}h_comparison.png", dpi=150); plt.close(fig)
        ab = rows[rows.variant.isin(["TCN_optimized", "no_rolling", "no_noise", "no_exogenous"])]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ["RMSE", "Pinball"]):
            ax.bar(ab.variant, ab[metric], color="#807252"); ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=20)
        fig.suptitle(f"{task} {horizon}h conditional ablations, matched sample IDs")
        fig.tight_layout(); fig.savefig(figs / f"{horizon}h_ablation.png", dpi=150); plt.close(fig)
        # Always show the predeclared optimized candidate; not test-best selection.
        pred = pd.read_csv(root / task / f"{horizon}h/evaluation/TCN_optimized/predictions.csv", parse_dates=["target_ts"])
        for name, weekend in (("weekday", False), ("weekend", True)):
            eligible = pred[pred.target_ts.dt.dayofweek.ge(5) == weekend]
            if eligible.empty: continue
            day = eligible.target_ts.dt.floor("D").iloc[0]
            segment = pred[pred.target_ts.dt.floor("D") == day]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(segment.target_ts, segment.y_true, label="actual", color="#252525")
            ax.plot(segment.target_ts, segment.p50, label="native P50", color="#28618b")
            ax.fill_between(segment.target_ts, segment.p10, segment.p90, alpha=.2, label="native P10/P90")
            ax.set_title(f"GEFCom pre-experiment {task} {horizon}h {name}")
            ax.set_ylabel("original source units (no price Cap)"); ax.legend(); fig.autofmt_xdate()
            fig.tight_layout(); fig.savefig(figs / f"{horizon}h_{name}_fit.png", dpi=150); plt.close(fig)
        pred["absolute_error"] = (pred.p50 - pred.y_true).abs()
        daily = pred.set_index("target_ts").resample("D").absolute_error.mean()
        fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(daily.index, daily)
        ax.set_title(f"{task} {horizon}h PRE-EXPERIMENT daily MAE"); ax.set_ylabel("MAE")
        fig.tight_layout(); fig.savefig(figs / f"{horizon}h_error.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); root = args.output
    if not (root / "completion.json").exists(): raise RuntimeError("training/evaluation not complete; do not fabricate a final report")
    protocol = json.loads((root / "protocol.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    progress = json.loads((root / "selection_frozen.json").read_text())
    sections = ["# 光储协同优化系统：GEFCom验证早停与消融预实验报告\n",
    "本报告全部结果为GEFCom预实验。AIC第八届AI+能源赛道科技创新组正式主实验尚未开展。预实验指标不可代表正式主实验或工程落地性能。\n",
    "## 3.1 实验环境与设置\n",
    "GEFCom-L负荷与GEFCom-P电价独立训练、独立权重，不拼为同一场站。不做无可靠标签的晴/多云/阴天分组。保留各连续原始序列，不跨不同站点/年份来源拼接。原生小时数据，仅评估1h、4h；缺少真实15min观测，15min预实验不评估。",
    "seed=42；48h历史窗口；按源序列目标日期开发集:测试集=70:30，开发集内训练:验证=80:20，边界embargo。两任务均使用训练段Z-score，负荷Cap只用于报告归一化，取训练历史最大值统计代理，不是真实硬件物理上限。电价无物理容量，不使用Cap、1-MAE/Cap、1-RMSE/Cap或MAPE。",
    "学习率/batch size仅在预先声明的两个TCN候选(.001/1024、.0005/512)中用验证Pinball选取。LSTM及TCN基线使用固定默认配置，并非同等搜索预算的排行榜。其余逐项移除消融固定选中超参；不是大规模调参。",
    f"训练预算上限{protocol['settings']['max_epochs']}轮，patience={protocol['settings']['patience']}，min_delta=.001原单位Pinball，保存验证最佳权重。早停代表该条件下停止改善，并非数学最优保证。达到预算但未早停均标为未确认收敛。验证、训练与最终指标统一为目标h时刻的Pinball。",
    "LSTM/TCN直接输出有序P10/P50/P90并原生Pinball训练（中位数±softplus间隔），不是残差后处理。训练/评估只监督目标h时刻，4h模型的中间输出不作为已训练的产品预测。Persistence/ARIMA使用验证后半段残差分位数后校准，不是原生概率头，不保证时序覆盖。ARIMA固定(2,0,0)，参数估计只用训练末90天，滚动滤波只取已发生历史观测。点基线仍以点预测作为P50，未单独估计残差中位数，可能发生区间与点值顺序不一致，crossing指标如实保留。",
    "完整环境见environment.lock，源码与数据SHA256见protocol.json，脏工作树快照见code_version.json与code_snapshot；时间切分见split_manifest.csv；配置、权重及逐轮训练日志随每个trial保存。",
    "光伏全程仅推理。用户提供了独立光伏项目 /Users/july/Documents/ChatGPT/光储协同优化，该项目optimization_v2的10套既有joblib已在各自原数据固定1000样本上只读推理核验，权重哈希未变，抽样指标另存frozen_pv_inference_audit；绝不混入本GEFCom对比或宣称全测试性能。其区间是原验证残差后处理。GEFCom-S无对应图像/同站输入契约，GEFCom光伏指标不可得。历史fusion_model*.pth仍未验证协议兼容性，不重训替代；见PV_ARTIFACT_STATUS.md。储能代码保留，本阶段未运行系统闭环。\n",
    "## 3.3 消融实验结果分析\n"]
    effects = []
    for task in protocol["tasks"]:
        frame = pd.read_csv(root / f"{task}_results.csv")
        plot(root, task, frame)
        sections += [f"### {'负荷' if task == 'load' else '电价'}（原始数据单位）\n",
            markdown_table(frame, ["horizon", "variant", "MAE", "RMSE", "Pinball", "interval_coverage_80", "interval_mean_width", "stop_reason"])]
        for horizon, rows in frame.groupby("horizon"):
            full = rows[rows.variant == "TCN_optimized"].iloc[0]
            for ablation, label, reference in (
                ("no_rolling", "因果24h滚动均值/标准差", "项目工程特征，不宣称原创算法"),
                ("no_noise", "标准化历史功率/电价小噪声(.01)", "Bishop 1995"),
                ("no_exogenous", "历史外生特征融合", "Wen 2017；GEFCom 2014"),
                ("TCN_baseline", "TCN优化组合", "Bai 2018 + Wen 2017 + Bishop 1995"),
                ("LSTM_baseline", "TCN优化方案相对LSTM", "非单项因果证据，仅整体比较"),
            ):
                ab = rows[rows.variant == ablation].iloc[0]
                delta = float(full.Pinball - ab.Pinball)
                effects.append({"task": task, "horizon": horizon, "optimization": label,
                     "reference": reference, "comparison": ablation, "delta_Pinball_full_minus_without": delta,
                     "delta_RMSE_full_minus_without": float(full.RMSE - ab.RMSE),
                     "validation_delta_Pinball": float(full.validation_pinball - ab.validation_pinball),
                     "actual_effect_by_test_Pinball": effect(delta),
                     "actual_effect_by_test_RMSE": effect(float(full.RMSE - ab.RMSE))})
            lstm = rows[rows.variant == "LSTM_baseline"].iloc[0]
            sections.append(f"{task} {horizon}h：优化TCN相对LSTM RMSE变化{(full.RMSE/lstm.RMSE-1)*100:.2f}%，Pinball变化{(full.Pinball/lstm.Pinball-1)*100:.2f}%。80%区间覆盖为{full.interval_coverage_80:.2%}，平均宽度{full.interval_mean_width:.5f}；覆盖更高本身不证明更好，应与名义80%的偏离、区间宽度及Pinball共同判断。")
    effect_frame = pd.DataFrame(effects); effect_frame.to_csv(root / "optimization_effects.csv", index=False)
    guidance = pd.DataFrame([
        {"优化名称": "因果多尺度TCN", "参考来源": "Bai 2018", "代码改动": "app/domain/forecast/quantile.py: QuantileNet，因果膨胀残差卷积", "预期作用": "扩大历史感受野，捕获不同时间尺度", "实际效果": "TCN基线四组RMSE均低于LSTM基线；网络整体比较，非等预算或独立组件因果证明"},
        {"优化名称": "24h滚动统计", "参考来源": "项目工程特征，不宣称原创算法", "代码改动": "quantile.py: rolling_stats / batch_from；QuantileSettings.use_rolling", "预期作用": "补充近期水平与波动，仅用起点前历史", "实际效果": "四组测试Pinball改善；负荷4h RMSE反而增加0.01966，验证Pinball增加0.00981，属于指标混合效果"},
        {"优化名称": "历史输入噪声正则化", "参考来源": "Bishop 1995", "代码改动": "quantile.py: fit_quantile；noise_std=.01，仅训练历史值加噪声", "预期作用": "减少对训练细节的依赖，缓解过拟合", "实际效果": "负荷1h、电价1h/4h测试Pinball改善；负荷4h Pinball增加0.04986、RMSE增加0.12622，负收益"},
        {"优化名称": "历史外生特征融合", "参考来源": "Wen 2017；GEFCom 2014", "代码改动": "quantile.py: QuantileNet历史外生输入；run_audited_preexperiments.py: no_exogenous同样本消融", "预期作用": "借助已知历史天气或系统变量补充目标序列", "实际效果": "电价两时距和负荷4h改善；负荷1h Pinball增加0.29504、RMSE增加0.74933，负收益；部分比较模型预算结束未确认收敛"},
        {"优化名称": "有序原生分位数头", "参考来源": "Koenker & Bassett 1978；Wen 2017", "代码改动": "quantile.py: QuantileNet中位数±softplus、pinball_loss，训练P10/P50/P90", "预期作用": "直接训练分位数并避免交叉", "实际效果": "原生模型所有保存预测分位数有序；未单独隔离概率头消融，不能宣称独立精度增益；负荷1h/电价1h仍过覆盖"},
        {"优化名称": "验证集超参搜索与早停", "参考来源": "验证集模型选择；AdamW参考Loshchilov & Hutter", "代码改动": "run_audited_preexperiments.py两候选；quantile.py: fit_quantile，max120/patience10/min_delta.001", "预期作用": "只依据验证损失选择配置，防止无上限调参", "实际效果": "负荷两时距选lr=.001/batch1024；电价两时距选lr=.0005/batch512；28试验中24验证早停、4预算结束；学习率和batch一起变动，独立贡献不可分辨"},
    ])
    guidance.to_csv(root / "optimization_guidance.csv", index=False)
    sections += ["### 优化名称、来源、修改、预期作用及实际效果\n",
        markdown_table(guidance, list(guidance.columns)),
        "网络实现 app/domain/forecast/quantile.py：LSTM改为因果膨胀残差TCN，预期扩大多尺度历史感受野；滚动均值/标准差只使用历史，预期补充局部水平/波动；噪声仅用于标准化训练历史输入，预期降低过拟合；移除外生特征消融保持相同样本与超参。实际条件增益见下表，负收益不隐藏。",
        markdown_table(effect_frame, list(effect_frame.columns)),
        "特别注意：负荷1h优化组合相对TCN基线的RMSE增加4.89%、Pinball增加0.07928，并非全面提升。负荷4h滚动统计的测试Pinball改善而RMSE和验证Pinball略恶化，因此不能用单一指标将其概括为无条件提升。下表两个实际效果字段分别按测试Pinball与RMSE判断，均为探索性描述，不用于重新选型。",
        "原生Pinball头与点预测+后校准基线的比较同时改变网络和训练目标，不能单独归因于概率头。分位数有序参数化保证不交叉，不保证校准。没有做未经数据支持的节假日、异常价格删除，也没有堆叠注意力。",
        "学习率和batch size同时变化的两个候选，不能分别证明两者独立贡献；调参实际效果只按tuning_records.json的验证Pinball解释。\n",
        "## 局限性说明\n",
        "1. 本部分全部为GEFCom预实验，正式主实验未开展，不可将预实验指标宣称为最终参赛效果或工程落地性能。",
        "2. 旧epoch=3粗筛及20轮探索已经查看同一测试段。本轮所有选择仍仅使用验证集并先冻结再测试，但历史暴露无法撤销。该测试只能作探索性比较，后续主实验必须采用未参与选择的新数据。",
        "3. 固定单随机种子只保证可复现，不证明统计显著性。重叠滑窗、单一切分、有限超参候选及预算差异限制结论，未报告多种子置信区间。部分优化可能只在GEFCom生效，存在过拟合及跨域失效风险。",
        "4. 负荷和光伏Cap若使用历史最大值都是统计代理，不等于真实物理容量。电价无物理容量，无Cap达标声明。",
        "5. 本实验历史天气仅来自预测起点之前，不用目标日未来实况。时区/节假日信息未经核实，时间编码沿用源时间索引，不套用中国法定假日。历史源导出的重复时间戳按固定keep-first规则去重，DST物理时刻无法可靠还原，小时轴是源索引约定，不声明UTC真实现场时刻。",
        "6. 光伏为预训练仅推理。已发布joblib的原数据抽样推理单独归档，不等于GEFCom性能或系统闭环。GEFCom缺对应输入，不生成其光伏指标。历史天空融合脚本的测试段最大值缩放问题不直接沿用；不改动冻结网络。无天气分组，无跨站拼接，无15min插值性能。",
        "7. 所有移除实验衡量固定配置下的条件贡献，特征交互可能存在；整体模型比较不能当作每项创新的独立证据。",
        "8. 储能经济收益、SOC、安全约束与系统闭环效果都留到正式主实验。",
        f"未触发验证早停的选中/消融模型：{json.dumps(completion['budget_limited'], ensure_ascii=False)}。这些不能标为已收敛。\n",
        "## 关键修改位置与伪代码\n",
        "- quantile.py：训练段统计 → 历史特征 → LSTM/TCN → 有序分位数头 → Pinball → 验证早停 → 恢复最佳权重。",
        "- run_audited_preexperiments.py：固定协议/哈希 → 基线训练 → 两候选验证调参 → 固定超参逐项消融 → 保存selection_frozen → 测试一次汇总。",
        "- frozen_pv.py：校验原权重哈希 → 加载队友模型定义/输入契约 → eval+no_grad推理；没有优化器或训练入口。verify_frozen_pv_releases.py复用独立项目既有joblib推理入口作原数据只读核验，不训练不修改光伏网络。GEFCom输入不兼容则保持不可评估。",
        "- report_audited_preexperiments.py：从落盘指标生成任务独立图表和本报告，不根据测试成绩择图择模。\n",
        "## 参考文献与协议\n", REFERENCES]
    (root / "REPORT.md").write_text("\n\n".join(sections))
    (root / "REFERENCES.md").write_text(REFERENCES)
    print(f"REPORT saved: {root / 'REPORT.md'}")


if __name__ == "__main__": main()
