#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "$ROOT_DIR/.venv-linux/bin/python" \
    "$ROOT_DIR/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] 未找到 Linux Python 环境，请先运行 ./setup_ubuntu.sh。" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/check_runtime_environment.py --require-models
exec "$PYTHON_BIN" project/app.py
