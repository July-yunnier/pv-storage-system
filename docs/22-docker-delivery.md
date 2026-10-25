# Docker 部署与 AI+能源算法精英大赛离线交付

本次只调整容器、CI 和交付配置；业务实现、既有回归用例、勘误、封存实验与数值均不变。
本机无 Docker 运行时，以下容器命令待在有 Docker 的联网构建机及目标机验证；远端
Actions 需人工提交并推送后才会运行，当前未声称构建或部署成功。

## 1. 构建与运行结构

所有构建命令在仓库根目录执行。要求 Docker Engine / Docker Desktop（Linux 容器）、
Buildx 和 Compose v2.20+（支持 `up --wait` 与 `--wait-timeout`）。

| 服务 | 构建文件 | 生产运行内容 | 本机默认入口 |
|---|---|---|---|
| backend | `backend/Dockerfile` | Python 3.12、生产虚拟环境、app、agents、scripts、pv_inference 代码与 manifest、YAML configs | `http://localhost:8000/docs` |
| frontend | `frontend/Dockerfile` | 无特权 Nginx、Vite dist、API 反向代理 | `http://localhost:8080` |

后端依赖构建阶段与运行阶段分离；不安装 pytest、实验绘图库或完整研究框架。
生产依赖单列于 `backend/requirements-runtime.txt`，不改既有开发/测试依赖。
前端 Node 与开发依赖仅留在构建阶段，运行镜像不含 Node、源码和 node_modules。
两个运行镜像均使用非 root 用户；后端 UID/GID 为 10001，单 Uvicorn worker；Compose
启用 init、健康检查依赖及后端 30 秒退出宽限，沿用已实现的后台巡检取消逻辑。

`INSTALL_FORECAST=false` 为默认值；`true` 安装
`backend/requirements-docker-forecast.txt` 的 LightGBM、scikit-learn、joblib 等生产依赖，
另装 PyTorch 官方 CPU wheel 和 libgomp1。此开关不装 neuralforecast、Prophet 或论文完整
实验环境，也不提供数据和权重。基础服务仍按现有逻辑降级，不能把构建开关视为模型已加载。
冻结 PV 模型有独立的严格版本契约，通用预测依赖不保证匹配；实际部署前仍须按
`docs/16-pv-inference-deployment.md` 和冻结 manifest/lock 核对环境并运行其自检。
一键启动的是产品前后端；冻结 PV HTTP 侧车需按上述文档在独立冻结环境中部署，
不会因复制 pv_inference 目录而自动启用。未部署时相应推理网关按既有逻辑报不可用。
政策文本/普通 PDF 解析依赖已包含，扫描件 PaddleOCR 与本地 LLM 服务不随镜像安装。

`.dockerignore` 使用默认拒绝、明确允许构建输入的清单；不发送本地数据目录、
verification/examples、模型文件、数据库、报告、.env 或密钥。`pv_inference` 只复制
Python 代码及运行时 manifest；configs 只复制仓库中的 YAML 配置，不复制样例 CSV。
不要把密钥写入获准的源码或 YAML；LLM 凭据通过运行时环境或只读挂载提供。
agents 在镜像中供离线脚本使用，不表示 LLM 调度已接入产品。

## 2. 一键启动与日常命令

```bash
# 在仓库根目录：默认构建基础后端、前端，并等待健康状态
bash scripts/start-docker.sh

# 或启用生产预测依赖；会增加镜像体积与构建时间
INSTALL_FORECAST=true bash scripts/start-docker.sh

# 自定义本机端口（端口冲突时使用）
FRONTEND_PORT=18080 BACKEND_PORT=18000 bash scripts/start-docker.sh

# 等价的 Compose 启动；脚本本身不安装 Docker
docker compose up -d --build --wait --wait-timeout 180
docker compose ps
docker compose logs -f
curl -fsS http://localhost:8000/api/v1/health
curl -fsS http://localhost:8080/api/v1/health

# 可选初始化：仅生成示例站点与示例政策电价，不导入外部测量数据
docker compose exec backend python scripts/seed_dongguan.py

# 停止并保留命名卷；不要加 -v，以免删除数据
docker compose down
```

前端通过 Nginx 将 `/api/` 转发到 Compose 内部 `backend:8000`，React Router 深层路径
回退到 index.html；API 请求与现有浏览器代码同源。默认只绑定本机回环地址，供赛事本机
演示；需要局域网访问时可显式设置 `FRONTEND_BIND=0.0.0.0`，服务权限与访问策略按交付
环境配置。天气默认 `mock`，适合离线演示；要联网天气需设置 `WEATHER_PROVIDER=open_meteo`。

启动脚本 `--offline` 模式不构建、不拉取镜像，仅启动已加载的本地镜像。脚本默认从环境
变量读取 IMAGE_TAG / 端口等参数；若使用 `.env`，不要把含密钥的文件加入交付包。
PowerShell 环境可直接运行对应的 `docker compose` 命令，或在 WSL / Git Bash 中运行脚本。

如需提供政策解析的 LLM 环境、外部模型或独立侧车地址，可在仓库外保存私有 Compose
覆盖文件，例如 `/secure/runtime.yml`，并手动用两份配置启动：

```yaml
services:
  backend:
    env_file:
      - /secure/backend.env
    # 可选：仅挂载已获授权并核对契约的权重目录
    volumes:
      - /secure/models:/external-models:ro
```

`backend.env` 按需设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL、LOAD_MODEL_PATH 等既有
运行参数；地址需在容器网络内可达，容器的 127.0.0.1 不是宿主机。独立冻结侧车通过
PV_INFERENCE_URL 配置；不要把通用后端预测环境当成冻结侧车环境。
启动命令为 `docker compose -f docker-compose.yml -f /secure/runtime.yml up -d --no-build --pull never --wait`。
私有覆盖文件、凭据和外部权重均不加入构建上下文或公开交付包。

## 3. SQLite WAL、锁等待与旧数据迁移

既有应用代码已对文件数据库设置 `journal_mode=WAL`，每个连接使用 5 秒锁等待。
Docker 仅设置 `STORAGE_DB_PATH=/app/data/storage.db`，不另改数据库逻辑或引入新的 PRAGMA。
Compose 的 `storage-data` 命名卷挂载整个 `/app/data`，同时承载 storage.db、storage.db-wal、
storage.db-shm；首次创建空卷时由镜像目录初始化权限。绑定目录或导入文件需可由 UID/GID
10001 写入。使用本机磁盘，避免把 SQLite WAL 目录放在 NFS 等不支持共享内存语义的文件系统。

此版从原来的 `./backend/data:/app/data` 改为命名卷，旧数据库不会自动迁移。
切换前停止旧后端及其他写入进程；以下示例使用 SQLite backup API 生成一致快照，
不改原库，也不把数据库放进镜像或赛事 tar：

```bash
# 在旧服务停止后，用可用的 Python 3 创建快照（Windows 可用 python）
python3 - <<'PY'
import sqlite3
from pathlib import Path
source = Path('backend/data/storage.db').resolve()
if not source.is_file():
    raise SystemExit('旧数据库不存在；首次部署请跳过迁移')
destination = Path('storage-migration.db')
if destination.exists():
    raise SystemExit('快照已存在，请先人工确认目标文件')
with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as src:
    with sqlite3.connect(destination) as dst:
        src.backup(dst)
PY

# 使用已构建/加载的后端镜像创建卷并导入；首次迁移适用，不覆盖已有卷数据
docker compose run --rm --no-deps --pull never --user 0 \
  -v "$PWD/storage-migration.db:/migration.db:ro" backend \
  sh -c 'test ! -e /app/data/storage.db && cp /migration.db /app/data/storage.db && chown 10001:10001 /app/data/storage.db'
bash scripts/start-docker.sh --offline

# 验证实际 SQLite 配置，预期输出 wal 5000
docker compose exec -T backend python - <<'PY'
import os
from app.infrastructure.db import Database
with Database(os.environ['STORAGE_DB_PATH']).connection() as connection:
    print(connection.execute('PRAGMA journal_mode').fetchone()[0],
          connection.execute('PRAGMA busy_timeout').fetchone()[0])
PY
```

有写入时不要只拷贝 storage.db；备份使用 SQLite backup API，或完全停止所有写入者后
备份整个卷。这里没有改变历史封存实验工件，也不要求将旧数据导入新演示环境。

## 4. GitHub Actions 的顺序与诊断

`.github/workflows/ci.yml` 在 push 到 `codex/repo-health-fixes` 和 PR 上触发，使用单个
ubuntu-latest job 顺序执行：checkout → Node 22 / Python 3.12 → 全量后端 pytest →
前端 npm ci、tsc -b、build → Buildx → Compose 配置校验 → 基础后端、预测后端、前端构建。
不安装 Docker、不登录镜像仓库，三个构建均为 `push: false`。Buildx 验证 Linux amd64，
不能据此声称其他架构或目标宿主机运行已通过。

pytest 跑全量发现的用例，当前已验证基线为 396 项；未硬编码截取 396 项或跳过后续新增
用例。流水线不使用 continue-on-error；tee 配合 pipefail，测试和构建失败会阻断后续验证。
仅诊断上传/摘要步骤使用 always，便于失败后保留日志。

`verification-reports` 工件保留 14 天，包含 pytest.xml、测试与安装日志、前端类型检查和
构建日志；Buildx 的步骤日志、构建摘要与构建记录保留在 Actions 运行中。GHA cache
只保存构建缓存，不是推送镜像，也不产生可直接 docker load 的交付 tar。

## 5. 赛事交付：联网构建、导出 tar、离线加载

先确认目标机架构。以下采用常见 Linux amd64；ARM64 目标机需改为 linux/arm64 并独立
验证，不能把本机 Apple Silicon 的默认架构镜像直接交给 amd64 赛场机器。
CI 仅验证构建，不自动发布交付包；以下命令由交付人员在有 Docker 的联网机器手动执行。

```bash
# 仓库根目录，两个生产镜像一起交付。无需 docker login / push。
docker buildx build --platform linux/amd64 --load \
  -f backend/Dockerfile --build-arg INSTALL_FORECAST=true \
  -t pv-storage-backend:contest-20261025 .
docker buildx build --platform linux/amd64 --load \
  -f frontend/Dockerfile -t pv-storage-frontend:contest-20261025 .

IMAGE_TAG=contest-20261025 bash scripts/start-docker.sh --offline
# 目标环境验收与演示后导出（Docker save 不包含命名卷）
mkdir -p delivery/scripts delivery/docs
docker save -o delivery/pv-storage-images-linux-amd64.tar \
  pv-storage-backend:contest-20261025 pv-storage-frontend:contest-20261025
cp docker-compose.yml delivery/
cp scripts/start-docker.sh delivery/scripts/
cp docs/22-docker-delivery.md delivery/docs/
cp THIRD_PARTY_NOTICES.md delivery/

# Linux；macOS 可改用 shasum -a 256
(cd delivery && sha256sum pv-storage-images-linux-amd64.tar > SHA256SUMS)
```

将 delivery 目录整体交付；其中没有数据卷、外部权重或 API 密钥。若赛事确需这些资产，
由项目方先核对来源与授权，再按模型/数据部署文档另行只读挂载，不能混入镜像。
运行镜像中的 dist、Python 代码和依赖是本次工程产物，不包含论文封存实验结果。

赛场离线加载并一键启动：

```bash
cd delivery
sha256sum -c SHA256SUMS           # macOS: shasum -a 256 -c SHA256SUMS
docker load -i pv-storage-images-linux-amd64.tar
IMAGE_TAG=contest-20261025 bash scripts/start-docker.sh --offline

# 无 Bash 的 PowerShell：先 $env:IMAGE_TAG='contest-20261025'，再执行这一条
docker compose up -d --no-build --pull never --wait --wait-timeout 180
docker compose ps
docker compose logs --tail=100
```

Compose 在 `--no-build --pull never` 下只使用本地标签，不读取 Dockerfile 来构建镜像；
交付目录无需带上完整源码和 node_modules。目标机仍须预装可用的 Docker 与 Compose。
自检 UI、API health、示例站点、数据与模型降级状态，再按赛事要求验收；没有现场数据
或外部权重时不宣称真实现场收益或模型已加载。

## 6. 第三方来源与许可清单

完整已知清单见仓库根目录 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，两类镜像
均复制该声明。主要项目包括：FastAPI/Pydantic/Uvicorn、OpenAI SDK、pvlib/pandas/NumPy、
LightGBM/scikit-learn/PyTorch，以及 React/Ant Design/ECharts；本次新增基础镜像还涉及
Python（PSF 等）、Node.js（MIT 等）、Debian/Alpine 系统包及 Nginx（BSD-2-Clause）。
各资产以实际发行版本附带的 LICENSE/NOTICE 为准，不把集合名称当作统一许可证。
后端构建会记录 `/opt/venv/requirements-installed.txt`，可用
`docker compose exec backend cat /opt/venv/requirements-installed.txt` 核对实际安装版本。

数据与模型来源包括 GEFCom2014、SKIPP’D、国网公开样例、AMPDNet 和冻结 PV 交付。
这些数据、验证样本与权重不打进本次镜像；原声明中的待核实状态仍然保留。
导出二进制镜像或离线包前，交付人员需核对实际依赖及系统包许可、保留必要的声明与署名。
来源记录不等于已获授权，本次工程配置不作合规保证，也不替项目选择整仓许可证。
