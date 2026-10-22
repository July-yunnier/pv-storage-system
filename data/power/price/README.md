# GEFCom2014-P 电价赛道数据（重建版）

`Task 15/Task15_P.csv`：GEFCom2014 电价赛道 Task 15 全量历史
（2011-01-01 ~ 2013-12-17，小时级，Zone 1 LMP + 日前负荷预报）。

- **来源**：公开镜像 [unresponsive51/GEFCom2014Price](https://github.com/unresponsive51/GEFCom2014Price)
  的 `gefcom_dataset.csv`（原始发布为 Tao Hong 论文附录，官方 Dropbox 链接已失效；
  引用请标注 Hong et al., IJF 2016）。
- **完整性验证（2026-10-22）**：与 July 训练存档 `price_mqrnn_gefcom_hourly.test.csv`
  的 3744 个逐点真值比对，最大绝对偏差 0.0，即同源数据。
- **已知缺口**：2013-03-10 02:00（美东夏令时切换，该小时不存在），
  基准脚本内单点前向填充处理（见 `run_price_benchmark.py`）。
- **缺测**：官方 `BenchmarkN_P.csv` 分位数文件镜像未包含，`gefcom_benchmark` 臂缺测，
  以 `naive_weekly` 近似代表（docs/15 §4.3 第 4 条已证二者 14/15 任务日逐点一致）。
- 数据行尾 2013-12-17 的价格列即原 Solution 值（镜像已合并）。
