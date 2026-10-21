# 光储协同优化系统

面向**工商业用户**的光伏 + 储能协同优化系统：融合光伏功率预测与电价预测，
输出储能容量配置方案与日充放电调度策略。

---

## 仓库定位

| 项目 | 说明 |
|---|---|
| **服务对象** | 工商业用户（有本地负荷、有需量电费） |
| **核心能力** | ①储能容量规划（配多大、几年回本）　②日调度策略（今天怎么充放） |
| **电价机制** | 峰谷分时为主，现货市场可切换/叠加（Provider 可插拔） |
| **部署形态** | Web 看板 + REST API + 方案报告导出 |

设计文档见 [`docs/系统设计.md`](docs/系统设计.md)。

---

## 实现进度

| 模块 | 内容 | 状态 |
|---|---|---|
| `src/tariff/` | 电价 Provider（tou 已实现，spot / tou_plus_spot 预留接口） | ✅ 完成 |
| `src/models/` | 站点档案、预测结果、调度结果 | ✅ 完成 |
| `src/optimizer/dispatch.py` | 规则式日调度（**含需量感知约束**） | ✅ 完成 |
| `src/optimizer/sizing.py` | 容量规划（决策变量含容量） | ⏳ 待做 |
| `src/forecast/` | 预测引擎（光伏 / 负荷 / 电价） | ⏳ 待做 |
| `src/api/` | FastAPI 接口层 | ⏳ 待做 |

**现在就能跑**：

```bash
python scripts/check_config.py     # 配置自检（电价时段空档检测）
python scripts/demo_dispatch.py    # 端到端调度演示（合成数据，无需外部数据源）
```

> `demo_dispatch.py` 会同时输出「有/无 需量感知」两版结果做对比：
> 忽略需量约束时，谷时满功率充电会把并网峰值从 542 kW 推到 **706 kW**，
> 省下的电度电费会从基本电费里加倍还回去（月增 6,235 元）。

## 目录结构

```
光储系统/
├── config/                     配置文件（换地点只改这里）
│   ├── site.example.yaml       示例站点：江苏某产业园，1MWp 光伏
│   └── tariff/                 省级电价模板
│       └── jiangsu.yaml
├── src/
│   ├── models/                 数据模型（Site / ForecastResult / DispatchResult）
│   ├── tariff/                 电价 Provider（tou / spot / tou_plus_spot）
│   ├── forecast/               预测引擎（光伏 / 负荷 / 电价）
│   ├── optimizer/              优化内核
│   │   ├── core.py             目标函数 + 约束（规划与调度共用）  ⏳
│   │   ├── dispatch.py         日调度  ✅
│   │   └── sizing.py           容量规划  ⏳
│   └── api/                    FastAPI 接口层
├── scripts/                    配置自检 + 端到端演示
├── tests/
├── data/                       运行时数据（gitignore）
├── models/                     模型权重（gitignore）
└── docs/                       设计文档
```

---

## 快速开始

```bash
# 1. 建环境（复用已有的 skippd 环境即可）
conda activate skippd
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 2. 复制配置模板
cp config/site.example.yaml config/site.yaml

# 3. 启动 API
uvicorn src.api.main:app --reload --port 8000
# 接口文档自动生成： http://localhost:8000/docs

# 4. 跑一次端到端 demo
python scripts/demo.py
```

---

## 核心设计约定

### 1. 统一时间序列格式

```json
{
  "points": [
    {"ts": "2026-09-19T08:00:00+08:00", "value": 623.4}
  ]
}
```

- 时间戳一律 **ISO 8601 带时区**
- **单位写进字段名**（`capacity_kwp` / `power_kw` / `price_yuan_per_kwh`）
- 缺失值用 `null`，**绝不用 0**——夜间功率 0 是有效值

### 2. 精度必须带口径

任何预测结果都要随附口径，没有口径的精度数字等于没有信息：

```json
{
  "accuracy": {
    "metric": "1 - RMSE/Cap",
    "value": 0.9251,
    "caliber": {
      "normalization": "rated_capacity",
      "cap_kw": 1000,
      "includes_nighttime": false,
      "spatial_scale": "single_site",
      "baseline": "persistence"
    }
  }
}
```

### 3. 电价 Provider 可插拔

三种模式对下游完全透明：

| 模式 | 电能量价格来源 | 是否需要电价预测 |
|---|---|---|
| `tou` | 省级目录电价表 | ❌ 不需要 |
| `spot` | 市场出清价 | ✅ 需要（含 p10/p90） |
| `tou_plus_spot` | 现货价替代电能量部分 | ✅ 需要 |

> ⚠️ 中国的"叠加"**不是价格相加**。到户电价 = 电能量价格 + 输配电价 + 政府性基金，
> 参与现货后**只有电能量部分被替代**，其余照旧。这个差异在 Provider 内部消化。

### 4. 优化：先调度后规划

```
optimizer/core.py      目标函数 + 约束（共用）
optimizer/dispatch.py  容量固定，决策每时段功率
optimizer/sizing.py    决策变量含容量
```

顺序不能反：**调度跑通了，规划的收益模型才有可信的验证基础。**

---

## ⚠️ 提交纪律

- **匿名要求**：仓库内**不得出现学校名称、指导教师姓名、班级信息**
- 提交材料里的作者信息统一用抽取码
- 第三方代码（官方 SKIPP'D 仓库等）要注明来源与许可
- 大数据与模型权重**不上传 git**，走网盘

---

## 依赖说明

- Python 3.11（建议复用 `skippd` 环境）
- 优化：`PuLP` 或 `scipy.optimize.linprog`
- 接口：`FastAPI` + `uvicorn`
- 前端图表：**ECharts**（原生支持阶梯图 `step`、时段色块 `markArea`）

---

## 相关资产位置

本仓库只放**产品代码**。

### 已并入仓库

核心算法实现已复制进 `scripts/reference/`（详见该目录的 README）：

| 文件 | 来源 | 说明 |
|---|---|---|
| `scripts/reference/sunset_nowcast_train.py` | 原 `~/全天空图像方案/scripts/05_train_sunset.py` | SUNSET nowcast 训练，10 折已跑通 |
| `scripts/reference/ultra_short_term.py` | 原 `超短期方案/ultra_short_term.py` | 超短期第 4 小时统计法，1−RMSE/Cap = 0.9506 |

### 未并入（本机参考资产）

⚠️ 下面这些路径是**开发机上的位置**，队友克隆仓库后**不存在**。
需要哪个就单独拷贝，或找项目负责人要。

| 资产 | 本机位置 | 说明 |
|---|---|---|
| 老师的原始方案 | `~/Documents/揭榜/方案三/` | `fusion_model.py` + 模型权重 40 MB，参考用 |
| SKIPP'D 完整复现工程 | `~/全天空图像方案/` | 含 10 折模型权重（546 MB）与完整文档 |
| 三个数据集评估流程 | `../超短期方案/` `../国网方案/` `../gefcom方案/` | 各自的评估脚本与报告 |
| 论文翻译 | `../翻译文献/` | 6 篇中文全文翻译（Word） |

> ✅ **这些资产对系统开发不是必需的。**
> 系统里的光伏预测走统计法（无权重）+ 公开气象 API，**两个深度学习模型都需要天空图像作输入，
> 企业站点不会有全天空相机**，因此不在系统技术路线上。
> 队友克隆仓库后，跑 `scripts/demo_dispatch.py` 就能看到完整链路，无需任何权重文件。
> 详见 `scripts/reference/README.md` 的「模型权重」一节。

## MQ-RNN 电价模型（2026-10-21）

`product/` 中已接入 arXiv:1711.11053v2 的 MQ-RNN 独立适配：LSTM、全局/局部MLP、直接多步分位数和 forking-sequences 训练。前端“电价中心”提供模型选择与 p10/p50/p90 曲线。

实现、训练命令、江苏/广东TOU配置、GEFCom独立评估及本站上线条件见 [模型说明](product/docs/15-price-mqrnn.md)。固定TOU确定性展开；现货模型需真实本站电能量价格训练，不能把外部小时模型精度当成本站效果。储能规则调度仍使用TOU。
