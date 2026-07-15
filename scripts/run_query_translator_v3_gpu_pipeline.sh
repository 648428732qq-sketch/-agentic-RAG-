#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv-linux/bin/python"
BASE_MODEL="$ROOT/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
RUN_DIR="$ROOT/artifacts/query_translator/20260630_biencoder_v3"
AUGMENTATION="$ROOT/datasets/external/query_translator_augmented_v3/supervision/query_term_pairs.jsonl"
FROZEN="$ROOT/tests/evals/query_translator_v3_nondialect_100/private/gold_keys.jsonl"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project virtualenv: $PYTHON" >&2
  exit 1
fi
for required in "$BASE_MODEL/model.safetensors" "$AUGMENTATION" "$FROZEN"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required project file: $required" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m pytest \
  tests/test_extract_query_translator_training_seeds.py \
  tests/test_generate_local_asr_query_augmentation.py \
  tests/test_generate_query_translator_mixed_blindset.py \
  tests/test_query_translator_biencoder.py -q

"$PYTHON" scripts/train_query_translator_biencoder.py \
  --output "$RUN_DIR" \
  --device cuda \
  --loss-mode mnrl_distill \
  --rounds 80 \
  --batch-size 96 \
  --learning-rate 3e-6 \
  --temperature 0.05 \
  --preservation-weight 1.0 \
  --augmentation-query-pairs "$AUGMENTATION"

"$PYTHON" scripts/evaluate_query_translator_biencoder.py \
  --model "$RUN_DIR/model" \
  --catalog "$RUN_DIR/term_catalog.jsonl" \
  --cases datasets/external/splits/dev/query_term_pairs.jsonl \
  --kind supervision \
  --output "$RUN_DIR/dev_trained.json" \
  --device cuda --batch-size 256 --max-top-k 20

"$PYTHON" scripts/evaluate_query_translator_biencoder.py \
  --model "$BASE_MODEL" \
  --cases "$FROZEN" \
  --kind frozen \
  --output "$RUN_DIR/frozen_baseline.json" \
  --device cuda --batch-size 256 --max-top-k 20

"$PYTHON" scripts/evaluate_query_translator_biencoder.py \
  --model "$RUN_DIR/model" \
  --catalog "$RUN_DIR/term_catalog.jsonl" \
  --cases "$FROZEN" \
  --kind frozen \
  --output "$RUN_DIR/frozen_trained.json" \
  --device cuda --batch-size 256 --max-top-k 20

"$PYTHON" scripts/compare_query_translator_biencoder.py \
  --input "$RUN_DIR/frozen_baseline.json" \
  --input "$RUN_DIR/frozen_trained.json" \
  --output "$RUN_DIR/frozen_fused.json" \
  --limit 20 --rank-constant 60

echo "V3 Translator pipeline complete: $RUN_DIR"
