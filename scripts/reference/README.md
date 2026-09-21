# 参考实现

这里存放从竞赛方案中复用过来的**光伏预测算法实现**。

它们**不是本仓库 `src/` 下的库代码**，而是可独立运行的脚本——
先在脚本形态下跑通、锁定指标，再决定是否重构成 `src/forecast/` 下的模块。

---

## sunset_nowcast_train.py

**用途**：SKIPP'D 数据集上的 SUNSET nowcast 模型训练与评估（**短临**，15 分钟级）

**来源**：原 `~/全天空图像方案/scripts/05_train_sunset.py`

**已验证成果**（10 折交叉验证，Apple M5 MPS，耗时约 9.5 小时）：

| 分组 | 样本 | 复现 RMSE | 论文 RMSE | 改善 | 1−RMSE/Cap |
|---|---|---|---|---|---|
| 晴天 | 6,986 | **0.4477** | 0.80 | ↓44.0% | 0.9851 |
| 阴天 | 7,017 | **3.1547** | 3.34 | ↓5.5% | 0.8952 |
| 整体 | 14,003 | **2.2554** | 2.43 | ↓7.2% | 0.9251 |

⚠️ **口径**：官方基准数据集构建时过滤了 `PV > 0`，14,003 个样本**全部是白天**。
上表是**纯白天口径**，不可与含全天 24 小时的数字直接比较。

**依赖**：
- 官方基准数据 `2017_2019_images_pv_processed.hdf5`（4.16 GB）
- 同目录下的 `times_trainval.npy`、`times_test.npy`
- `torch`（训练）；`h5py`、`numpy`（读数据）

**运行**：

```bash
python scripts/reference/sunset_nowcast_train.py \
  --task nowcast \
  --hdf5 /path/to/2017_2019_images_pv_processed.hdf5 \
  --out_dir ./model_output --folds 10 --epochs 200 --device mps
```

> 服务器/CI 上先跑 `--folds 1 --epochs 1 --max_samples 3000` 验证链路（约 10 秒）。

---

## ultra_short_term.py

**用途**：超短期（**第 4 小时**）光伏功率预测，**纯统计法，不需要训练**

**来源**：原 `超短期方案/ultra_short_term.py`

**方法**：

```
pred(t+4h) = clip( 近 N 天同刻均值 + alpha × 当前偏差 , 0, Cap )
```

**已验证成果**（SKIPPD 2019 数据，7:3 时间顺序划分）：

| 指标 | 数值 |
|---|---|
| 测试段整体 `1 − RMSE/Cap` | **0.9506** ✅ |
| 逐月平均 | **0.9588** ✅ |
| `1 − MAE/Cap` | 0.9813 |
| RMSE / MAE | 1.4821 / 0.5617 kW |

⚠️ **口径**：评估**含夜间样本**（占 51.05%）。夜间"预测≈0/实际≈0"几乎零误差，
把整体抬高了。**若官方按白天口径评估，只有 0.9295，不达标。**
→ 报告里必须声明评估口径，并尽快向组委会确认。

**依赖**：`2019_pv_raw.csv`（SKIPPD 2019 原始功率数据，1 分钟采样）

**运行**：

```bash
python scripts/reference/ultra_short_term.py --csv /path/to/2019_pv_raw.csv

# 或用环境变量
export SKIPPD_CSV=/path/to/2019_pv_raw.csv
python scripts/reference/ultra_short_term.py

# 换参数试试（alpha 0.20~0.35、N 10~20 均在达标区间）
python scripts/reference/ultra_short_term.py --alpha 0.30 --N 14
```

---

## 与 `src/` 的关系

| | `scripts/reference/` | `src/forecast/` |
|---|---|---|
| 形态 | 可独立运行的脚本 | 库模块，供 API 调用 |
| 输入 | 命令行参数 + 数据文件 | 数据对象 |
| 目的 | **验证算法指标** | **生产化调用** |

**演进路径**：在 reference 里跑通并锁定指标 → 把核心算法抽成 `src/forecast/`
下的纯函数 → 由 API 层调用。

> 不要直接把这两个脚本 import 进 `src/`——它们带 argparse 和大量 print，
> 是"给人看的"，不是"给程序调的"。

---

## 数据来源

两个数据集都是公开的、可复现的：

| 数据集 | 获取地址 | 许可 |
|---|---|---|
| SKIPP'D 基准数据（图像+功率） | `https://purl.stanford.edu/dj417rh1007` | CC-BY-4.0 |
| SKIPP'D 官方代码仓库 | `https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset` | **MIT** |
| 中国国家电网风光数据 | `https://doi.org/10.6084/m9.figshare.17304221.v4` | CC BY 4.0 |

> ⚠️ **竞赛规则要求"版权清晰"**，凡引用第三方数据或代码，必须在报告中注明出处。
> 主动注明来源反而是加分项——说明用的是公开可复现的基准，评审更容易采信。
