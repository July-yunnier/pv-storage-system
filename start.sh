#!/usr/bin/env bash
# 光储一体 - 一键启动（Git Bash / Linux / macOS）
set -e
cd "$(dirname "$0")"

command -v python >/dev/null || { echo "[错误] 未找到 python（需 3.11+）"; exit 1; }
command -v npm >/dev/null || { echo "[错误] 未找到 npm（需 Node.js 18+）"; exit 1; }

if [ ! -f backend/data/storage.db ]; then
  echo "[初始化] 安装后端依赖..."
  python -m pip install -q -r backend/requirements.txt
  echo "[初始化] 生成演示站点数据..."
  (cd backend && python scripts/seed_demo.py)
fi
(cd backend && python scripts/seed_dongguan.py >/dev/null 2>&1 || true)

if [ ! -d frontend/node_modules ]; then
  echo "[初始化] 安装前端依赖..."
  (cd frontend && npm install)
fi

echo "[启动] 后端 http://127.0.0.1:8000"
(cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000) &
BACK_PID=$!
sleep 3
echo "[启动] 前端 http://localhost:5174"
(cd frontend && npm run dev -- --port 5174 --strictPort) &
FRONT_PID=$!
sleep 4
(command -v open >/dev/null && open http://localhost:5174) || \
(command -v xdg-open >/dev/null && xdg-open http://localhost:5174) || true

echo "Ctrl+C 停止全部服务"
trap "kill $BACK_PID $FRONT_PID 2>/dev/null" EXIT
wait
