"""回放实验：预测与调度的对比评估（预测方法 × 调度策略矩阵）。

- app/domain/experiment/metrics.py   误差与运行指标（纯函数）
- app/domain/experiment/execute.py   计划执行仿真：计划动作 × 实测光伏/负荷 → SOC 递推与费用
- app/application/replay_service.py  编排：逐日滚动，严格不使用目标日之后的数据
"""
