# 第三方来源与许可状态

核对日期：2026-10-25。本文件记录已知来源和许可边界，不为第三方代码、数据或权重授予新许可，也不代替上游完整许可证。仓库尚未为自有代码选定统一 LICENSE；不得据此推定整仓为 MIT 或允许任意再分发。

## 数据、代码与模型工件

| 资产 / 本仓库位置 | 来源与署名 | 当前许可状态 |
|---|---|---|
| AMPDNet TorchScript 及训练元数据：`backend/data/model_releases/ampdnet_delivery20/` | [USTC-AI4EEE/AMPDNet](https://github.com/USTC-AI4EEE/AMPDNet)，commit `751bf2931cf1d369a6b998710daf6162aab8b6ab`；Liu et al. (2026), *Solar Energy* 312, 114563。精确 commit 以工件 JSON 为准 | 工件元数据记载 pinned checkout 无 LICENSE，许可未核实；不声明可自由再分发。模型架构并非本项目原创。 |
| GEFCom2014-P 重建数据：`data/power/price/Price/Task 15/Task15_P.csv`；相关衍生实验 CSV | [unresponsive51/GEFCom2014Price](https://github.com/unresponsive51/GEFCom2014Price)；Hong et al. (2016), *Probabilistic energy forecasting: GEFCom2014 and beyond*, [DOI](https://doi.org/10.1016/j.ijforecast.2016.02.001)；重建过程见 `data/power/price/README.md` | 数据再分发许可尚未核实。公开镜像可访问不等于原权利人授权；不对数据赋予新许可。 |
| SKIPP’D 样例与冻结推理验证资料：`backend/pv_inference/verification/`、`examples/` | Nie et al. (2023), [论文](https://doi.org/10.1016/j.solener.2023.03.043)，[Stanford-solar-forecasting-dataset](https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset) | 按已有交付记录：数据 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，基准代码 MIT。保留署名、来源、许可链接和修改说明；本仓库样例经过窗口抽取、预处理。详见 `backend/pv_inference/docs/RESEARCH.md`。 |
| 国网八站样例、场景代理数据及衍生预测：`backend/pv_inference/`、`reports/official_experiment/` | 既有交付标记为国家电网公开光伏数据；报告记录公开能源数据镜像 [Bob05757/Renewable-energy-generation-input-feature-variables-analysis](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis) | 原始站点权利人及许可链尚未完整核实，不能因“公开数据”推定再分发权。容量映射与负荷仿真属于本项目处理，不改变底层数据权利。 |
| 冻结 PV 推理代码、manifest 与外部部署权重：`backend/pv_inference/` | 交付记录原提交 `f680b05df795f1e181a7ed05582bddff30bb8ec5`；详见该目录 `docs/RESEARCH.md` | 代码、模型与底层训练数据分别核对；`models/` 不入 git，由发布包部署，不能用哈希验证代替许可验证。 |
| MES 负荷训练工件：`backend/data/training_artifacts/` | 本项目仿真负荷与模型实现；借鉴 LSTM、Bahdanau attention 方法 | 底层依赖仍适用其原许可；不把通用架构称为项目原创算法。 |

## 运行依赖

依赖由包管理器安装，没有在仓库内复制这些项目的完整源码。随二进制、容器或离线包发布时，应保留实际安装版本附带的 LICENSE / NOTICE；传递依赖也需随分发物核对。

| 项目 | 已知上游许可 | 来源 |
|---|---|---|
| FastAPI、Pydantic、Uvicorn、HTTPX、PyYAML、python-multipart、python-dotenv | MIT / BSD 系（按各包文件核对） | 各包 PyPI 元数据与发行包 LICENSE |
| OpenAI Python SDK | Apache-2.0 | [openai-python](https://github.com/openai/openai-python) |
| LightGBM、React、Vite、TypeScript、Ant Design、Day.js、React Router | MIT | 各包发行文件与上游仓库 |
| ECharts | Apache-2.0 | [apache/echarts](https://github.com/apache/echarts) |
| PyTorch、NumPy、pandas、SciPy、scikit-learn、pvlib、ReportLab | BSD 系（具体条款以发行包为准） | 实际安装版本 LICENSE |
| Matplotlib | Matplotlib / PSF-based license | [Matplotlib license](https://matplotlib.org/stable/project/license.html) |
| pypdf | BSD-3-Clause | [py-pdf/pypdf](https://github.com/py-pdf/pypdf) |

可选研究依赖见 `backend/requirements-forecast.txt`，冻结推理依赖见 `backend/pv_inference/requirements-infer.lock.txt`。后续实验在 `run_meta.json` 中记录完整 `dependency_versions`，用于定位对应许可文本和复现环境；既有封存结果不回填当前环境版本。

许可未核实的原始数据、权重或代码进入新的对外分发包之前，需由项目方确认相应授权；本声明只记录现状，不宣称已补齐授权。
