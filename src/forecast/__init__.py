"""预测引擎：光伏功率 / 负荷 / 电价。

三个预测目标共用统一的输出契约（见 models.ForecastResult）：
每条预测都要带

    accuracy.caliber = {
        normalization,       归一化基准
        cap_kw,              额定容量
        includes_nighttime,  是否含夜间样本
        spatial_scale,       空间尺度
        baseline,            基准对照
    }

**没有口径的精度数字等于没有信息。** 这是从三个公开数据集的实测中得到的教训：
同一份预测、同一组数据，仅改变是否包含夜间样本，指标就差 2 个百分点。

光伏功率预测
------------
两条路线，按预见期切换（Bacher 2009 + 实测结论）：

| 预见期 | 方法 | 输入 |
|---|---|---|
| < 1 h    | 全天空图像 SUNSET / 持续性 | 天空图像 + 历史功率 |
| 1 ~ 6 h  | 近 N 天同刻均值 + 偏差延续 | 仅历史功率 |
| > 19 h   | 日周期气候态 + 气象预报 | 气象预报（NWP） |

已有实现（**已复制进本仓库**，见 `scripts/reference/`）：

    scripts/reference/sunset_nowcast_train.py   SUNSET nowcast 训练与评估
                                                短临 15 min 级，10 折已跑通
                                                晴天 RMSE 0.4477 / 整体 2.2554 kW

    scripts/reference/ultra_short_term.py       超短期统计法（第 4 小时）
                                                无需训练，逐样本实时可算
                                                `1 − RMSE/Cap = 0.9506`（含夜间口径）

两者的用法与数据依赖见 `scripts/reference/README.md`。

**从脚本到模块的演进路径**：先在 reference 里跑通并锁定指标，再把核心算法
抽成本模块下的纯函数（去掉 argparse 和 print），由 API 层调用。
不要直接 import reference 里的脚本——它们是"给人看的"，不是"给程序调的"。

电价预测
--------
**仅在 spot / tou_plus_spot 模式下需要**。
tou 模式下电价由政策给定，直接查表，不训练模型。
现货模式建议用分位数回归（FQRA），输出 p10/p50/p90 供储能策略做风险边界。

待实现
------
- `pv.py`       光伏功率预测（按预见期路由到不同方法）
- `load.py`     负荷预测（先做 profile_based，即按用户类型套典型曲线）
- `price.py`    电价预测（仅现货模式需要，分位数回归）
"""
