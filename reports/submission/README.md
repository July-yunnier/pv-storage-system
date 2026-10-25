# AI能源技术报告初稿

本目录包含匿名技术报告的 Markdown、Word 和 PDF 版本。

报告基于仓库中已有代码、授权数据和已保存实验结果整理。正式主实验 P0、P0-A、P1、P2、P3、P4 已经执行并封存；完整实验边界、复现命令、未通过验收项和仍待补充的真实数据见 [`reports/official_experiment/official_experiment_report.md`](../official_experiment/official_experiment_report.md) 与 [`docs/17-experiment-plan.md`](../../docs/17-experiment-plan.md)。

2026-10-25 提交前修订：两份 Word 已核对，电价阶段汇报的结果表统一为
[`docs/18 §5`](../../docs/18-price-model-comparison.md#51-总表360-点样本外)，
MQ-CNN 10.77 / +17.3% 只保留在明确标注“作废”的勘误中；含旧数的三张图已撤下。
AI 技术稿的 Markdown、Word 与 PDF 均增加本次口径说明，原历史预实验不混入同协议表。
PDF 的说明页置于原稿之前，原稿页码不变。

LightGBM 保形校准已完成，严格滚动 PICP80 0.642 → 0.756，仍不等于真实站点达到
名义 0.80。LLM 调度表述限于“离线消融”；政策文本解析 LLM 是另一项产品能力。
数据与权重的许可状态见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。
