#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_CUDA_TORCH:-0}" == "1" ]]; then
  if [[ -z "${PYTORCH_INDEX_URL:-}" ]]; then
    echo "[ERROR] INSTALL_CUDA_TORCH=1 时必须显式设置 PYTORCH_INDEX_URL。" >&2
    echo "请从 PyTorch 官方安装页选择与服务器驱动匹配的 CUDA wheel 地址。" >&2
    exit 1
  fi
  python -m pip install torch --index-url "$PYTORCH_INDEX_URL"
fi

python -m pip install -r requirements.txt

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5'); print('Embedding model ready')"
fi
if [[ "${DOWNLOAD_RERANK_MODEL:-0}" == "1" ]]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3'); print('Rerank model ready')"
fi

CHECK_ARGS=()
if [[ "${INSTALL_CUDA_TORCH:-0}" == "1" ]]; then
  CHECK_ARGS+=(--require-cuda)
fi
python scripts/check_runtime_environment.py "${CHECK_ARGS[@]}"

echo "安装完成。将 project/.env.example 复制为 project/.env 并填写密钥后，运行 ./start.sh。"
