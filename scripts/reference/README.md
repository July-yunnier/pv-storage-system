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

## 模型权重：在哪里、要不要搬运

**核心结论：本系统运行时不需要任何训练好的权重文件。**

### 三个模型产物的现状

| 模型 | 权重文件 | 体积 | 位置 | 系统是否使用 |
|---|---|---|---|---|
| 老师的融合模型 | `fusion_model.pth`<br>`fusion_model30-15.pth` | 11 + 29 MB | `~/Documents/揭榜/方案三/` | ❌ **不使用** |
| SUNSET nowcast（本仓库复现） | `fold_1.pt` … `fold_10.pt` | 每折 54.6 MB<br>**合计 546 MB** | `~/全天空图像方案/model_output/SUNSET_nowcast/` | ⚠️ **仅演示短临时使用** |
| 超短期统计法 | **无** | — | — | ✅ **生产路线** |

### 为什么系统不需要权重

两个深度学习模型都**需要天空图像作输入**，而企业站点不会有全天空相机。
所以系统里的光伏预测走的是：

```
光伏功率预测 = 统计法（超短期，参数写进 config，无权重）
             + 公开气象 API（长时域）
```

**这是"手上有什么数据，就用什么方法"的必然结果**，不是妥协——三个公开数据集的
实测已经证明，在只有功率序列的场景下，统计法（0.9506）稳定优于各种复杂模型。

### SUNSET 权重怎么处理（如果队友要做短临演示）

三个选项，按推荐度排序：

**① 实际演示时不接图像模型**（推荐）

系统演示走统计法 + 典型曲线即可，短临那一层用报告里的复现结果引用，
**不需要在系统里跑图像模型**。

**② 加载已有权重直接推理**（如果一定要演示，且权重已在手）

`--resume_dir` 会跳过训练，直接加载权重推理并集成：

```bash
python scripts/reference/sunset_nowcast_train.py \
  --task nowcast \
  --hdf5 /path/to/2017_2019_images_pv_processed.hdf5 \
  --out_dir ./model_output --folds 10 \
  --resume_dir /path/to/model_output/SUNSET_nowcast
```

**实测**（Apple M5 MPS）：**15 秒**完成 10 折推理，输出与训练时**完全一致**——

```
RMSE = 2.2554 kW    MAE = 1.2179 kW    1 − RMSE/Cap(30.1) = 0.9251
单折 RMSE 均值 = 2.3512
```

⚠️ 仍需要 `--hdf5`（要读测试集），所以 4.16 GB 的数据文件躲不掉。
**但如果只是要指标，直接用上面的数字即可，连数据都不用下。**

**③ 完整重训**（没必要，除非要改结构）

10 折约 9.5 小时（Apple M5 MPS）。**新机器上的耗时会长得多**，
而且需要先下载 4.16 GB 的 hdf5。

> ⚠️ **权重不进 Git 仓库**（`.gitignore` 已排除 `*.pt` / `*.pth` / `*.npy`）。
> 546 MB 会让仓库变得无法克隆；而且二进制文件在 Git 里无法做有意义的版本对比。
> 真要共享，走百度网盘或 GitHub Release，别进版本历史。

### 一句话分工

| 角色 | 需要什么 |
|---|---|
| **队友（开发系统）** | **什么都不需要**，克隆仓库 → 跑 `demo_dispatch.py` 就能看到完整链路 |
| 你（本地复现） | 权重已在本机，需要时可直接推理 |
| 报告 / 答辩 | 引用本文件里的指标表即可，不需要附权重 |

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
