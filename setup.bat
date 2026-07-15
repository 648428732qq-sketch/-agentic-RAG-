:: 中医 Agentic RAG — 一键环境安装 (在项目根目录运行)
@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo === 中医 Agentic RAG 环境安装 ===

:: 1. 创建虚拟环境
python -m venv .venv
call .\.venv\Scripts\activate.bat

:: 2. 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 3. 预下载当前配置使用的 BGE Embedding 模型
set HF_HOME=%cd%\.cache\huggingface
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5'); print('Model ready')"

echo === 安装完成 ===
echo 启动命令:
echo   set HF_HOME=%cd%\.cache\huggingface
echo   请将 API Key 写入 project\.env 或系统环境变量，不要写进脚本。
echo   .\.venv\Scripts\python.exe project\app.py
