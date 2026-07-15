:: 中医 Agentic RAG — 启动脚本
@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HOME=%cd%\.cache\huggingface
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe project\app.py
) else if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe project\app.py
) else (
  echo [ERROR] 未找到 .venv 或 venv，请先运行 setup.bat。
  exit /b 1
)
