:: 中医 RAG 完整重建环境并启动（会删除 .venv）
@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo === 1. 重建虚拟环境 ===
if exist .venv rmdir /s /q .venv
python -m venv .venv
call .\.venv\Scripts\activate.bat

echo === 2. 安装依赖 ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo === 3. 预下载 Embedding 模型 ===
set HF_HOME=%cd%\.cache\huggingface
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5'); print('Model ready')"

echo === 4. 启动 ===
echo API Key 仅从 project\.env 或系统环境变量读取。
python project\app.py
pause
