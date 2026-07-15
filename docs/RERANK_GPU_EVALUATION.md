# GPU Rerank Evaluation

## Scope

This experiment evaluates a safety-preserving second-stage reranker for the structured
TCM syndrome retrieval path. It is an engineering benchmark, not a clinical validation
or a production deployment claim.

The evaluated stack is:

- bi-encoder: `BAAI/bge-small-zh-v1.5`;
- hybrid candidate retrieval: dense retrieval, local BM25, and payload overlap;
- local evidence gates: required symptom groups, forbidden terms, and ambiguity checks;
- cross-encoder: `BAAI/bge-reranker-v2-m3`;
- GPU: NVIDIA A100-SXM4-80GB;
- index: 1,419 structured entries in an in-memory Qdrant instance.

The in-memory index is intentional. A copied Windows Qdrant local database is not a
portable Linux artifact, and the shared project filesystem does not provide reliable
SQLite persistence semantics for Qdrant Local.

## Current benchmark

The broad synthetic hard-negative set contains 538 cases over 150 formulas:

| Case style | Count | Baseline | Rerank |
|---|---:|---:|---:|
| Full signature | 150 | 96.00% | 96.00% |
| Missing required symptom | 150 | 99.33% | 100.00% |
| Forbidden conflict | 88 | 96.59% | 96.59% |
| Shared symptoms requiring clarification | 150 | 84.67% | 93.33% |
| **Overall** | **538** | **93.87%** | **96.47%** |

Observed A/B changes:

- 14 previously failed cases passed after reranking;
- no previously passed case failed;
- 103 queries changed their top-ranked item;
- rerank was used in 507/538 cases (94.24%) with no model errors;
- full-signature Recall@1, Recall@5, MRR@8, and nDCG@8 were already 100% before rerank.

The last point matters: this benchmark shows the clearest gain in ambiguity handling
and safety-gate behavior, not in already-saturated complete-signature retrieval.

## Latency

The measured rerank run had an average per-query latency of 595.77 ms and P95 of
844.73 ms. The earlier baseline run averaged 696.17 ms with P95 1,089.59 ms, but those
runs had different cold/warm cache conditions. Do not claim that reranking made the
system faster from this pair of runs.

`scripts/run_gpu_rerank_ab.py` improves the methodology by:

1. preloading both embedding and rerank models;
2. warming both retrieval paths before measurement;
3. alternating baseline-first and rerank-first execution order per case;
4. using one shared in-memory index and emitting one combined report.

## Reproduce the A/B run

```bash
export HF_HOME="$PWD/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=2

.venv-linux/bin/python scripts/run_gpu_rerank_ab.py \
  --device cuda \
  --dictionary datasets/structured/syndrome_dictionary.jsonl \
  --rerank-model models/bge-reranker-v2-m3 \
  --max-formulas 150 \
  --top-k 8 \
  --rerank-candidates 8 \
  --rerank-max-length 256 \
  --rerank-mode evidence_first \
  --output-dir datasets/structured/rerank_gpu_ab
```

Outputs:

- `cases.jsonl`;
- `baseline_predictions.jsonl`;
- `rerank_predictions.jsonl`;
- `report.json`.

## Locked regression set

Generate a deterministic question/gold split once, review it, then lock it:

```bash
.venv-linux/bin/python scripts/prepare_rerank_locked_eval.py \
  --formula-count 60 \
  --output-dir tests/evals/rerank_locked_v1
```

Run without exposing the gold labels to the query file:

```bash
CUDA_VISIBLE_DEVICES=2 .venv-linux/bin/python scripts/run_gpu_rerank_ab.py \
  --questions tests/evals/rerank_locked_v1/questions.jsonl \
  --gold tests/evals/rerank_locked_v1/private/gold_keys.jsonl \
  --device cuda \
  --rerank-model models/bge-reranker-v2-m3 \
  --output-dir datasets/structured/rerank_locked_v1_results
```

This locked set is synthetic and dictionary-derived. It is useful for regression
testing, but it is not an external clinical blind set. A stronger follow-up is a
human-reviewed query set sourced independently from the indexed dictionary.

### Locked v1 black-box result

The locked v1 set was also executed through the Gradio API, testing the actual demo
boundary rather than calling retrieval functions directly:

| Case style | Count | Baseline | Rerank |
|---|---:|---:|---:|
| Full signature | 60 | 100.00% | 100.00% |
| Missing required symptom | 60 | 98.33% | 100.00% |
| Forbidden conflict | 41 | 95.12% | 95.12% |
| Shared symptoms requiring clarification | 60 | 71.67% | 85.00% |
| **Overall** | **221** | **90.95%** | **95.02%** |

Reranking recovered 10 failed cases, introduced one regression, and changed the top-1
result for 46 queries. Average latency increased from 582.00 ms to 725.51 ms; P95
increased from 843.10 ms to 1,088.90 ms. This controlled black-box measurement is the
preferred latency comparison.

Reproduce the black-box test while the private demo is running:

```bash
python scripts/evaluate_rerank_demo_api.py \
  --url http://127.0.0.1:17860/ \
  --questions tests/evals/rerank_locked_v1/questions.jsonl \
  --gold tests/evals/rerank_locked_v1/private/gold_keys.jsonl \
  --output-dir datasets/structured/rerank_locked_v1_api_results
```

The API evaluator checkpoints every completed case and retries transient tunnel or HTTP
timeouts, so interrupted runs can resume without discarding prior results.

## Local test page

The dedicated test page compares baseline and reranked results, shows gate decisions,
matched terms, missing required groups, conflicts, and query latency.

```bash
CUDA_VISIBLE_DEVICES=2 \
RERANK_DEMO_SERVER_NAME=127.0.0.1 \
RERANK_DEMO_SERVER_PORT=17860 \
.venv-linux/bin/python scripts/rerank_gradio_demo.py
```

Access it through an SSH tunnel instead of exposing it publicly:

```bash
ssh -N -L 17860:127.0.0.1:17860 -p 8080 hejun@202.127.200.31
```

Then open `http://127.0.0.1:17860`.

## Resume-safe wording

> Built a safety-preserving two-stage TCM retrieval pipeline using BGE embeddings and
> bge-reranker-v2-m3 on A100; created a fixed-seed 60-formula/221-case hard-negative
> regression set and black-box Gradio API evaluation, improving pass rate from 90.95%
> to 95.02% (+4.07 pp) and shared-symptom clarification from 71.67% to 85.00%, with a
> measured P95 latency cost of 245.8 ms.

Avoid claiming clinical accuracy, external blind-test performance, production SLA, or
latency improvement until independently reviewed data and controlled repeated timing
runs are available.
