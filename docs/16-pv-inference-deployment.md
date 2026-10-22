# PV 推理模型平台化部署（冻结包接入）

对应协作方（算法同学）发布的 `pv_inference_frozen_*.zip` 五步部署指令，
本文记录在本平台的落地方式与运维口径。目标是：**模型以冻结工件托管、
平台只路由不训练、默认光伏链路零影响**。

## 1. 架构形态

```
浏览器/调用方
   → 主后端 FastAPI :8000（/api/v1/inference/*，网关）
   → 推理侧车 :8010（.venv-infer，Python 3.12 + 锁定依赖）
      → pv_inference.load_model（manifest 契约强校验后反序列化）
```

- **侧车**（`backend/scripts/pv_inference_sidecar.py`）：stdlib-only HTTP 服务，
  独占冻结环境；模型按需加载、LRU 驻留（默认 2 个，环境变量可调）、可显式卸载。
- **主后端不装 sklearn/lightgbm**，只通过 `app/infrastructure/pv_inference_gateway.py`
  （stdlib urllib）转发。侧车未启动 → `/api/v1/inference/*` 统一 502 PROVIDER_ERROR，
  其余业务（含默认光伏预测）不受影响。
- **为什么独立环境**：冻结契约要求 Python 3.12 + numpy 2.3.5/pandas 2.2.3/
  scikit-learn 1.9.1/lightgbm 4.7.0 等精确版本（manifest `environment` 块，
  `check_versions` 加载前强制比对，不匹配即拒绝加载）；主后端环境
  （Python 3.14/pandas 3）不满足，故采用"单独容器"方案，以本机独立 venv 落地。

## 2. 部署步骤（对应五步指令）

### 2.1 迁入（指令第 1 步）

发布 zip 解压到仓库根目录，保持 `backend/pv_inference/` 目录结构（权重字节不变）。
权重目录 `backend/pv_inference/models/`（约 296MB）不入 git，由发布 zip 部署；
完整性由 `manifest.json` 中的 SHA256/字节数保障（`load_model` 在反序列化前校验）。

### 2.2 模型注册与路由（指令第 2、3 步）

- 模型注册表 = `backend/pv_inference/manifest.json`（发布方维护的加载契约）；
  平台视图经 `GET /api/v1/inference/models` 暴露：模型 ID、容量(kW)、SHA256、
  字节数、验证指标、当前是否驻留。
- 网关路由：`POST /api/v1/inference/pv/{model_id}` 把推理请求转发给侧车对应模型；
  侧车内 `pv_inference` 的加载器执行哈希/版本/schema 三重校验，失败即拒绝。
- 平台内部操作（注册表录入、配额、网关绑定）在本实现中即上述两步，无独立后台。

### 2.3 依赖策略（指令第 4 步）

```bash
# 仓库根目录；uv 提供 CPython 3.12.13
uv venv --python 3.12.13 .venv-infer
uv pip install --python ./.venv-infer/Scripts/python.exe \
  -r backend/pv_inference/requirements-infer.lock.txt
```

- 版本锁定：`requirements-infer.lock.txt` 与 manifest `environment.packages`
  一致；加载时 `check_versions` 精确比对，**版本不匹配直接拒绝加载**（RuntimeError）。
- 资源配额（manifest `runtime`）：默认线程 1、上限 4；单模型建议 4 GiB、
  全量 10 模型建议 8 GiB；实测单进程峰值 RSS ≈ 529 MB。
- OpenMP：LightGBM 运行所需；Windows 轮子自带，Linux 需 libgomp。

### 2.4 上线验证（指令第 5 步）

```bash
# 1) 冻结验证（哈希先于反序列化 + 持有样本对账 + 非法输入拒绝）
.venv-infer/Scripts/python.exe backend/scripts/pv_inference_verify.py
#    （POSIX 机器可直接跑官方版：python -m pv_inference.selftest）

# 2) 启动侧车
.venv-infer/Scripts/python.exe backend/scripts/pv_inference_sidecar.py

# 3) 平台冒烟
curl :8000/api/v1/inference/health
curl :8000/api/v1/inference/models
curl -X POST :8000/api/v1/inference/pv/state_grid_1 \
  -H "Content-Type: application/json" \
  -d @backend/pv_inference/examples/state_grid_1_request.json
curl -X POST :8000/api/v1/inference/models/state_grid_1/unload   # 动态卸载
```

验收口径：推理返回 15/60/240 分钟 P10/P50/P90（kW），与
`verification/state_grid_1.json` 参考值误差 ≤ 容差；卸载后再次推理自动重载；
**默认光伏预测（`pv_model=auto` 融合/统计链）不经过本网关，行为不变**。

## 3. 运维接口

| 接口 | 说明 |
|---|---|
| `GET /api/v1/inference/health` | 侧车健康 + 冻结环境版本（版本不匹配在此暴露） |
| `GET /api/v1/inference/models` | 模型注册表（10 个：state_grid_1..8、skippd、skippd_images） |
| `POST /api/v1/inference/pv/{model_id}` | 推理路由；输入契约见 manifest（时区时间戳、连续升序、末条 ts=issued_at） |
| `POST /api/v1/inference/models/{model_id}/unload` | 卸载驻留模型（平台内部操作） |

环境变量：`PV_INFERENCE_URL`（默认 `http://127.0.0.1:8010`）、
侧车侧 `PV_INFERENCE_HOST/PORT/MAX_MODELS`。

## 4. 边界与红线

- 预测时域为 **15/60/240 分钟短临**；与日前 96 点链路的融合（接入
  `pv_model` 选项）是后续议题，当前仅提供网关推理能力。
- skippd 系列为天空相机图像模型（30.1 kW 基准）：**按 D-011（2026-10-21）不启用**，
  权重仅作发布包完整性的一部分保留，注册表不向其开放产品选项；
  state_grid 系列为国网站点模型（30–130 MW），用于平台验证，不代表任意站点可直接套用。
- 权重不训练、不重序列化；替换权重或包版本须走新发布号并重验哈希，
  不得静默回退其他模型（发布方冻结规则，平台层原样执行）。
