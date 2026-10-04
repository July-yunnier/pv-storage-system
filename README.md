# 光储协同优化系统

本仓库的正式可运行系统位于 [`product/`](product/)。根目录旧版 `src/` 原型已完成迁移和清理，后续开发、测试与比赛交付均以 `product/` 为准。

## 正式源码位置

| 能力 | 代码位置 |
|---|---|
| 后端入口与 API | `product/backend/app/main.py`、`product/backend/app/api/` |
| 储能调度 | `product/backend/app/domain/dispatch.py` |
| 储能容量规划 | `product/backend/app/domain/sizing/` |
| 光伏、负荷、电价预测 | `product/backend/app/domain/forecast/` |
| MES 增强负荷模型 | `product/backend/app/domain/forecast/load_advanced.py` |
| 电价与政策解析 | `product/backend/app/domain/tariff.py`、`policy_tariff.py` |
| 数据模型 | `product/backend/app/domain/models.py` |
| 前端可视化 | `product/frontend/src/` |
| 系统与接口文档 | `product/docs/` |

更完整的功能说明见 [`product/README.md`](product/README.md)。

## 快速验证

### 后端

```bash
cd product/backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run.py
```

接口文档：<http://127.0.0.1:8000/docs>

高级负荷预测、政策 PDF 解析等可选能力另需安装：

```bash
python -m pip install -r requirements-ai.txt
```

复制 `.env.ai.example` 为 `.env.ai`，并将 `LOAD_MODEL_PATH` 指向：

```text
product/backend/data/training_artifacts/load_mes_attention.pt
```

### 前端

```bash
cd product/frontend
npm install
npm run build
npm run dev
```

## 训练与复现

- 工厂负荷仿真数据生成：`product/backend/scripts/generate_factory_load_mes.py`
- MES 增强负荷模型训练：`product/backend/scripts/train_advanced_load.py`
- 已训练模型：`product/backend/data/training_artifacts/load_mes_attention.pt`
- 模型元数据：`product/backend/data/training_artifacts/load_mes_attention.manifest.json`

训练数据为可解释的典型工厂仿真数据，不应表述为真实企业运行数据。

## 参考算法

不属于生产系统运行路径的天空图像和超短期算法参考实现保留在 [`scripts/reference/`](scripts/reference/)，便于比赛实验和技术溯源。

## 安全要求

- 不提交 `.env.ai`、API 密钥、客户信息或真实企业数据；
- `.env.ai.example`只保存无敏感信息的配置模板；
- 仓库中的模型由仿真数据训练，真实站点应用前需重新验证或微调。
