# Backend

当前后端是项目第一轮可运行闭环，使用 FastAPI + SQLite + Pydantic。

## 本地启动

```powershell
Set-Location backend
python -m unittest discover -s tests -p "test_*.py" -v
python run.py
```

接口文档：<http://127.0.0.1:8000/docs>

默认数据库：`backend/data/storage.db`。该文件不会提交到 Git。

## 当前接口

- `GET /api/v1/health`
- `POST /api/v1/sites`
- `GET /api/v1/sites`
- `GET /api/v1/sites/{site_id}`
- `PATCH /api/v1/sites/{site_id}`
- `POST /api/v1/sites/{site_id}/measurements`
- `GET /api/v1/sites/{site_id}/measurements?data_type=pv_power&date=2026-09-20`
- `POST /api/v1/tariffs`
- `GET /api/v1/tariffs`
- `POST /api/v1/policy-tariffs/parse-text`（只生成候选，不入库）
- `POST /api/v1/policy-tariffs/parse-pdf`（文本型 PDF；扫描件先 OCR）
- `POST /api/v1/policy-tariffs/confirm`（人工确认并严格校验后入库）
- `POST /api/v1/storage/schedule`

## 可选 AI 能力

基础系统不强制安装 PyTorch、PDF 与 LLM SDK。需要 MES 多任务负荷预测或政策 PDF
解析时，再安装：

```bash
python -m pip install -r requirements.txt -r requirements-ai.txt
```

复制 `.env.ai.example` 为 `.env.ai`。当前模板已预设为本地 Ollama＋Qwen：

- `LLM_API_KEY=ollama-local`：兼容 SDK 的占位值，本地 Ollama 不校验。
- `LLM_BASE_URL=http://127.0.0.1:11434/v1`：本地 OpenAI 兼容接口。
- `LLM_MODEL=qwen2.5:7b`：默认中文政策解析模型。
- `LOAD_MODEL_PATH`：训练完成后的负荷模型 artifact 路径。

首次使用：

```bash
cp .env.ai.example .env.ai
ollama pull qwen2.5:7b
ollama serve
```

文本型 PDF 默认使用 pypdf。扫描版 PDF 在 `POLICY_PDF_EXTRACTOR=auto` 时会自动尝试
PaddleOCR；需要额外安装：

```bash
python -m pip install -r requirements-ocr.txt
```

电价解析接口只返回候选规则和原文证据，不会自动写入调度价目。必须由人员核对后调用
`/policy-tariffs/confirm`。高级负荷模型必须同时提供历史特征、目标日 96 点 MES 计划和
目标日天气/日历特征；未选择 `load_model=advanced` 时继续使用同刻均值基线。

高级负荷模型训练入口：

```bash
python scripts/train_advanced_load.py \
  --csv data/advanced_load_training.csv \
  --output data/models/load_mes_attention.pt
```

CSV 至少需要 `ts,total_load_kw,mes_plan`；有真实 `workshop_load_kw` 才启用多任务辅助损失。
交接文档中的仿真数据没有车间标签，因此脚本会退化为总厂单任务，不会伪造辅助标签。
目标日天气在生产环境应来自天气预报；如果回放实验使用历史实况代替，报告必须注明这
不是严格的日前天气输入。

## 当前调度规则

### 独立负荷与电价机器学习实验

新增18种模型的独立GEFCom负荷/电价实验、原生小时级 `/api/v1/forecast/ml`、预测曲线驱动的主动削峰与未来高峰电量预留。国内政策TOU保留为兼容路径；市场电价不隐式转换为国内分时电价。

当前新增的验证早停训练及独立消融只用于 GEFCom 预研究：不分天气组、不跨站拼接，电价不计算 Cap，光伏只读既有权重。运行 `scripts/run_audited_preexperiments.py` 后用 `scripts/report_audited_preexperiments.py` 生成报告。达到训练预算不等于收敛；旧20轮结果与同一测试段的历史暴露必须披露。参赛自查清单见 `docs/PREEXPERIMENT_SELF_CHECKLIST.md`。正式主实验尚未开展。

训练、模型权重接入、单位要求、复现与验收命令见 [实验说明](docs/FORECAST_EXPERIMENTS.md)。设置 `FORECAST_ARTIFACT_ROOT` 后即可加载独立任务权重。原96点调度可显式传入 `predicted_loads` / `predicted_pv` / `predicted_prices`，必须满足当地目标日15分钟网格与kW/CNY每kWh单位。

### 默认兼容模式

- 谷段充电。
- 光伏余电充电。
- 峰段和尖峰放电。
- SOC、PCS 功率、等效循环次数和禁止反送电约束生效。
- 当前只支持 15 分钟粒度和完整 24 小时序列。

## 目录

```text
app/
├── api/              # FastAPI 路由
├── domain/           # 数据模型、电价和调度领域逻辑
└── infrastructure/   # SQLite 数据库
tests/                # 单元测试和 API 流程测试
```

## MQ-RNN 电价预测

训练、配置和API见 [MQ-RNN电价说明](../docs/15-price-mqrnn.md)。前端“电价中心”支持日前24/48/72小时曲线；固定TOU直接展开，现货使用匹配的本站训练权重。
