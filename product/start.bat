@echo off
chcp 65001 >nul
title 光储一体 - 一键启动
cd /d %~dp0

REM ---- 依赖检查 ----
where python >nul 2>nul || (echo [错误] 未找到 python，请先安装 Python 3.11+ && pause && exit /b 1)
where npm >nul 2>nul || (echo [错误] 未找到 npm，请先安装 Node.js 18+ && pause && exit /b 1)

REM ---- 首次运行：后端依赖 + 种子数据 ----
if not exist backend\data\storage.db (
  echo [初始化] 安装后端依赖...
  python -m pip install -q -r backend\requirements.txt
  echo [初始化] 生成演示站点数据...
  cd backend && python scripts\seed_demo.py && cd ..
)
REM 东莞站点种子幂等，每次确保存在
cd backend && python scripts\seed_dongguan.py >nul 2>&1 && cd ..

REM ---- 前端依赖 ----
if not exist frontend\node_modules (
  echo [初始化] 安装前端依赖...
  cd frontend && call npm install && cd ..
)

REM ---- 启动服务（各自独立窗口，关闭窗口即停止）----
echo [启动] 后端 http://127.0.0.1:8000
start "光储-后端" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
timeout /t 3 >nul
echo [启动] 前端 http://localhost:5174
start "光储-前端" cmd /k "cd /d %~dp0frontend && npm run dev -- --port 5174 --strictPort"
timeout /t 5 >nul
start http://localhost:5174
echo.
echo 完成。两个服务窗口可独立关闭；本窗口可直接关闭。
pause >nul
