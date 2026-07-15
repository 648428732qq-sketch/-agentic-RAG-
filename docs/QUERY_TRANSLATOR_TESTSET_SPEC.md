# Query Translator 测试集拆分规范

## 先澄清一件事

刚生成的 300 条是“混合盲测候选池”，不是直接拿来全部调参的数据集。

如果把 300 条全部看完、全部修完、再拿同一批 300 条说召回率提升，就会变成过拟合演示。今天要做的是把它拆成三套用途不同的数据：

| 集合 | 用途 | 是否可看金标 | 是否可用于最终指标 |
|---|---|---:|---:|
| `dev_tuning` | 修 Query Translator、别名表、错字归一化、门控阈值 | 可以 | 不可以 |
| `hard_negative_special` | 专门压测“缺关键症状必须追问、否定不能误判阳性” | 可以 | 不单独作为最终总指标 |
| `final_holdout` | 内部留出/回归验收 | 调参时不看具体失败 | 只能做内部指标 |
| `fresh_final_100` | 最终盲测验收 | 生成后锁定，不参与调参 | 可以 |

## Query Translator 测试集到底要什么

它不是普通问答集。它测的是“用户原话 -> 本地数据库规范术语/否定词/门控决策”的转换能力。

公开问题只允许有：

```json
{"id": "...", "query": "患者原话"}
```

私有金标至少要有：

```json
{
  "id": "...",
  "query": "患者原话",
  "generation_style": "metaphor_incomplete",
  "expected_term_groups": [["恶寒"], ["无汗"], ["身疼", "头身疼痛"]],
  "expected_negative_terms": [],
  "forbidden_terms": [],
  "must_clarify": false,
  "expected_decision": "grounded_answer",
  "expected_source_type_in_top_k": "formula_syndrome",
  "evidence_mappings": [
    {"source_phrase": "冷得打抖", "canonical_term": "恶寒", "polarity": "present"}
  ]
}
```

核心不是让模型背答案，而是判断它有没有把普通口语、隐喻、错字正确映射到本地库里的规范词。地域方言不纳入当前项目范围。

## Hard Negative 专项集规范

Hard Negative 是“看起来像能回答，但其实证据不够或存在否定冲突”的问题。

必须覆盖三类：

1. **关键鉴别症状缺失**：比如只说发热头痛，但不说有汗/无汗。
2. **明确否定**：比如“没有怕冷”“不咳嗽”“不喘”。
3. **相似候选混淆**：Top-K 里有多个相近方证，必须追问关键差异。

建议金标字段：

```json
{
  "expected_term_groups": [["发热"], ["头痛"]],
  "omitted_term_group": ["无汗"],
  "expected_negative_terms": ["恶寒"],
  "forbidden_terms": ["恶寒"],
  "must_clarify": true,
  "expected_decision": "clarify",
  "expected_gate": false
}
```

专项指标：

| 指标 | 今天验收线 |
|---|---:|
| `must_clarify` | >= 98% |
| `negative_terms` | >= 98% |
| `no_forbidden_positive` | >= 99.5% |
| `decision` | >= 95% |
| 危险直接回答率 | 0% |

这里的危险直接回答，指问题缺关键辨证信息或明确否定了症状，但系统仍直接给出方证/用药结论。

## 内部留出集要求

`final_holdout` 是从已生成的 300 条中拆出来的内部留出集。由于 300 条整体报告已经被看过，它不能作为严格意义上的最终盲测，只能用于回归检查：确认修开发集时没有把其他表达类型改坏。

要求：

- 从 300 条里固定抽 100 条。
- 五类表达尽量均衡：隐喻/残句、ASR错字、否定、多症状长句、Hard Negative。
- 调参期间只运行公开 `questions.jsonl`，不要打开 `private/gold_keys.jsonl`。
- 每次内部回归都用同一批 holdout，不能边看失败边改。

## 真正最终盲测集要求

如果要对外写“最终验收指标”，需要重新生成一批没有被报告分析过的新题，建议命名为 `fresh_final_100`。

要求：

- 重新从种子生成，不复用现有 300 条问题文本。
- 生成后只允许公开 `questions_mixed.jsonl` 进入在线预测进程。
- 修 Query Translator 时不要看 `private/gold_keys.jsonl`，也不要看最终失败明细。
- 最终盲测可以来自同一批本地种子，但最好使用不同 `shuffle_seed`、不同输出目录，并在修完开发集后再生成。
- 如果要更严格，再从结构化库里扩充 20 到 30 个新 seed 后生成；今天时间紧，可以先用现有 16 个 seed 生成新表达变体。

最终验收指标：

| 指标 | 目标 |
|---|---:|
| `term_recall` | >= 95% |
| `negative_terms` | >= 98% |
| `no_forbidden_positive` | >= 99.5% |
| `decision` | >= 95% |
| `must_clarify` | >= 98% |
| `retrieval` | >= 98% |
| 数据库外/证据不足直接回答率 | 0% |

## 今天执行命令

拆分三套集合：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_query_translator_eval_sets.py
```

跑开发集在线预测：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_query_translator_blind_predictions.py `
  --questions tests\evals\query_translator_benchmarks\dev_tuning\questions.jsonl `
  --output datasets\structured\query_translator_dev_tuning_predictions.jsonl
```

离线评分开发集：

```powershell
.\.venv\Scripts\python.exe scripts\score_query_translator_blind_predictions.py `
  --predictions datasets\structured\query_translator_dev_tuning_predictions.jsonl `
  --gold tests\evals\query_translator_benchmarks\dev_tuning\private\gold_keys.jsonl `
  --report datasets\structured\query_translator_dev_tuning_report.json
```

跑 Hard Negative 专项：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_query_translator_blind_predictions.py `
  --questions tests\evals\query_translator_benchmarks\hard_negative_special\questions.jsonl `
  --output datasets\structured\query_translator_hard_negative_predictions.jsonl
```

离线评分 Hard Negative：

```powershell
.\.venv\Scripts\python.exe scripts\score_query_translator_blind_predictions.py `
  --predictions datasets\structured\query_translator_hard_negative_predictions.jsonl `
  --gold tests\evals\query_translator_benchmarks\hard_negative_special\private\gold_keys.jsonl `
  --report datasets\structured\query_translator_hard_negative_report.json
```

内部回归只在修完后跑：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_query_translator_blind_predictions.py `
  --questions tests\evals\query_translator_benchmarks\final_holdout\questions.jsonl `
  --output datasets\structured\query_translator_final_holdout_predictions.jsonl
```

然后离线评分：

```powershell
.\.venv\Scripts\python.exe scripts\score_query_translator_blind_predictions.py `
  --predictions datasets\structured\query_translator_final_holdout_predictions.jsonl `
  --gold tests\evals\query_translator_benchmarks\final_holdout\private\gold_keys.jsonl `
  --report datasets\structured\query_translator_final_holdout_report.json
```

生成真正最终盲测候选，建议每类 17 条，共 102 条；随后用固定抽样脚本转成严格 100 条：

```powershell
.\.venv\Scripts\python.exe scripts\generate_query_translator_mixed_blindset.py `
  --output-dir tests\evals\query_translator_fresh_final_102 `
  --per-style 17 `
  --shuffle-seed 20260623 `
  --model qwen3.6-flash `
  --prompt-api-key
```

固定抽成 100 条：

```powershell
.\.venv\Scripts\python.exe scripts\select_query_translator_subset.py `
  --questions tests\evals\query_translator_fresh_final_102\questions_mixed.jsonl `
  --gold tests\evals\query_translator_fresh_final_102\private\gold_keys.jsonl `
  --output-dir tests\evals\query_translator_fresh_final_100 `
  --target-count 100
```

然后最终在线预测只读公开问题：

```powershell
.\.venv\Scripts\python.exe -u scripts\run_query_translator_blind_predictions.py `
  --questions tests\evals\query_translator_fresh_final_100\questions_mixed.jsonl `
  --output datasets\structured\query_translator_fresh_final_predictions.jsonl
```

最后离线评分：

```powershell
.\.venv\Scripts\python.exe scripts\score_query_translator_blind_predictions.py `
  --predictions datasets\structured\query_translator_fresh_final_predictions.jsonl `
  --gold tests\evals\query_translator_fresh_final_100\private\gold_keys.jsonl `
  --report datasets\structured\query_translator_fresh_final_report.json
```
