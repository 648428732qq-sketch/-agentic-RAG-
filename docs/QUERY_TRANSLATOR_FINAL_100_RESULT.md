# Query Translator Final 100 Evaluation

> **完整性状态（2026-06-29 复核）**：该数据集已在同一轮开发中连续用于原始版和 fix1-fix5 调参，不能再视为最终盲测或用于对外宣称泛化指标。以下结果仅作为历史回归记录。3 条 rerank probe 也是该 100 条的原样子集，不是独立金标准。

## Scope

本次历史评测使用了当时固定的 100 条数据：

- Questions: `tests/evals/query_translator_fresh_final_100/questions_mixed.jsonl`
- Gold: `tests/evals/query_translator_fresh_final_100/private/gold_keys.jsonl`
- Final predictions: `datasets/structured/query_translator_fresh_final_100_predictions_after_fix5.jsonl`
- Final report: `datasets/structured/query_translator_fresh_final_100_report_after_fix5.json`

该测试集由 `query_translator_fresh_final_102` 固定抽样 100 条得到，覆盖 ASR 同音错字、Hard Negative、隐喻不完整、多症状一句话、否定不确定、方言口语。

## Main Changes

- 补齐 `project/config.py` 中结构化检索缺失配置：
  - `EMBEDDING_LOCAL_FILES_ONLY`
  - `SYNDROME_COLLECTION`
  - `ENABLE_LLM_SYMPTOM_TRANSLATOR`
  - `SYNDROME_*` 检索和 Query Translator 参数
- 修正 `SyndromeRetriever`：
  - `decision=clarify` 时同步 `needs_more_info=true`
  - direct clinical signature 按证据组覆盖判断，不被 primary term 数量误伤
  - 复用 Qdrant client 并保持关闭逻辑
- 增强 Query Translator 泛化：
  - 同音错字：`发扫/吐玩/经慢/瓶躺/谈像清西`
  - 方言和繁体：`發燒/風吹/過來/難受/無/冇出汗`
  - 口语隐喻：头部紧绷、鼻孔堵住、平躺憋醒、胃痛腹胀
  - 后置否定：`发热都没有/压根没有发热`
  - 避免 `气儿顺` 误判为针刺 `得气`
  - 限制 `咳逆上气` 必须有喘、上气、气促等原文证据
- 调整盲测预测脚本的 `gate` 字段：
  - `gate` 表示是否有可用结构化候选
  - 不再等同于 WebUI 是否直接生成最终答案

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m py_compile project\config.py project\core\symptom_query_translator.py project\core\syndrome_retriever.py scripts\run_query_translator_blind_predictions.py
.\.venv\Scripts\python.exe -m unittest tests.test_strict_local_retrieval -v
.\.venv\Scripts\python.exe -u scripts\run_query_translator_blind_predictions.py --questions tests\evals\query_translator_fresh_final_100\questions_mixed.jsonl --output datasets\structured\query_translator_fresh_final_100_predictions_after_fix5.jsonl
.\.venv\Scripts\python.exe scripts\score_query_translator_blind_predictions.py --predictions datasets\structured\query_translator_fresh_final_100_predictions_after_fix5.jsonl --gold tests\evals\query_translator_fresh_final_100\private\gold_keys.jsonl --report datasets\structured\query_translator_fresh_final_100_report_after_fix5.json
```

## Historical Metrics

```json
{
  "case_count": 100,
  "passed": 93,
  "candidate_coverage": 1.0,
  "term_recall": 1.0,
  "gate": 1.0,
  "decision": 0.99,
  "must_clarify": 1.0,
  "needs_more_info": 1.0,
  "negative_terms": 1.0,
  "no_forbidden_positive": 1.0,
  "retrieval": 0.94,
  "average_latency_ms": 2623.2
}
```

## Remaining Boundary

剩余失败主要是 `retrieval` 排序问题，不是 Query Translator 理解失败。典型情况是单症状或少量症状本身不足以唯一指向某个方剂/来源类型，例如只给出“汗出”“恶风”“气上冲”等局部信息时，多个古籍条文和方剂都能命中。继续为了分数强行把这些单症状排序到某个固定方剂，会降低系统泛化能力。

该 100 条后续只能用于回归检查。新的最终验收集必须在代码和阈值冻结后一次性运行，运行前不可查看私有标签，运行后不得继续据其失败样本修改规则并重复宣称为盲测结果。
