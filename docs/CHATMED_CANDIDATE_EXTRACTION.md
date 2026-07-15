# ChatMed 候选字段抽取说明

## 目标

`ChatMed_TCM-v0.2.json` 只能作为问题侧语料和弱候选来源，不能作为临床事实库。  
本流程只读取 `query` 字段，把其中能和本地 `syndrome_dictionary.jsonl` 对齐的词抽出来，供后续人工或规则复核。

禁止事项：

- 不读取、不写出 ChatMed 的 `response` 作为医学证据。
- 不把候选直接注入 Qdrant。
- 不把候选直接合并进正式 `diagnostic_keys`、`pathogenesis`、`required_symptom_groups`。
- 不用 ChatMed 未审核答案修正本地知识库。

## 执行命令

```powershell
.\.venv\Scripts\python.exe scripts\extract_chatmed_field_candidates.py --max-rows-per-file 5000
```

输出目录：

```text
datasets/structured/chatmed_candidates/
├── diagnostic_key_candidates.jsonl
├── pathogenesis_candidates.jsonl
├── source_type_candidates.jsonl
├── negation_forbidden_candidates.jsonl
├── unmatched_or_unusable.jsonl
└── chatmed_candidate_summary.json
```

## 当前抽取结果

本次全量处理结果：

- ChatMed 问题数：112,565
- 本地结构化条目数：1,419
- 本地诊断词：4,287
- 本地病机词：559
- 本地来源提示词：1,759
- 命中诊断候选的问题：70,537
- 命中来源类型提示的问题：98,098
- 命中病机候选的问题：304
- 命中否定候选的问题：307
- 未命中本地词表的问题：14,466
- 跳过空值或疑似隐私：1

`diagnostic_key_candidates.jsonl`、`source_type_candidates.jsonl`、`unmatched_or_unusable.jsonl` 默认各最多写出 5,000 行样例；summary 里保留真实 `seen` 数和 `truncated` 标志。

## 字段含义

所有候选记录都带有：

- `review_status: candidate_unverified`
- `evidence_origin: chatmed_query_only`
- `source_dataset: ChatMed_TCM-v0.2.json`
- `source_line`
- `query_hash`
- `query`

候选类型：

- `diagnostic_key`：ChatMed 问题中的症状、体征、口语症状与本地结构化词表发生字符串命中。
- `pathogenesis`：问题本身直接出现本地病机词，例如“阴虚火旺证”“肝郁化火证”。
- `source_type`：通过本地命中和关键词推断问题更像方证、本草、古籍理论或针灸问题。
- `negation_forbidden`：问题中明确否定的本地症状候选，例如“没有出汗”命中“出汗”。
- `unmatched_or_unusable`：问题没有命中本地结构化词表，可用于发现覆盖缺口或噪声。

## 后续使用方式

推荐流程：

1. 先看 `negation_forbidden_candidates.jsonl`，把高频、明确的否定表达补进 Query Translator 的否定测试集。
2. 再看 `unmatched_or_unusable.jsonl`，判断是本地知识库缺覆盖，还是 ChatMed 问题本身不可用。
3. 对 `diagnostic_key_candidates.jsonl` 做频次聚合，只把能追溯到本地 payload 的表达加入同义词候选。
4. 对 `pathogenesis_candidates.jsonl` 只做“用户会怎么问病机”的测试，不直接改病机事实。
5. 用这些候选生成新的盲测或 dev tuning 样本，验证 Query Translator，而不是直接改正式知识库。

## 二次筛选

第一层抽取会保留大量宽泛命中，不能直接使用。第二层筛选脚本会重新扫描 ChatMed `query`，按 `term + entry_id + source_type` 聚合，并输出二次 review queue：

```powershell
.\.venv\Scripts\python.exe scripts\filter_chatmed_candidate_terms.py `
  --min-query-count 3 `
  --max-local-entries 8 `
  --max-rows-per-file 5000
```

输出目录：

```text
datasets/structured/chatmed_candidates/filtered/
├── diagnostic_term_review_queue.jsonl
├── pathogenesis_query_review_queue.jsonl
├── source_hint_review_queue.jsonl
├── negation_forbidden_review_queue.jsonl
└── chatmed_candidate_filter_summary.json
```

当前筛选结果：

- ChatMed 问题数：112,565
- 跳过空值或疑似隐私：1
- 诊断词条聚合总数：3,702
- 病机问法聚合总数：62
- 来源提示聚合总数：551
- 否定词聚合总数：27
- 写出诊断 review queue：2,824 条
  - high：711
  - medium：1,013
  - broad_review_only：1,100
- 写出病机 review queue：50 条
  - high：20
  - medium：17
  - low：13
- 写出来源提示 review queue：331 条
  - high：194
  - medium：95
  - broad_review_only：42
- 写出否定词 review queue：27 条
  - high：21
  - medium：6

筛选规则：

- `high`：出现频次达到阈值，且本地命中条目数量不高，适合优先人工查看。
- `medium`：有一定频次，但存在轻度歧义或泛化风险。
- `broad_review_only`：高频但太宽泛，只能用于覆盖/压力测试，不应作为精确规则。
- `low`：证据不足，暂不优先。

脚本会对“疼痛、头晕、头痛、腹痛、咳嗽、发热、失眠”等泛症状自动加 `generic_symptom` 风险标记，避免它们因为高频被误认为高精度方证映射。

所有二次筛选结果仍然是：

- `review_status: filtered_candidate_unverified`
- `evidence_origin: chatmed_query_only+local_syndrome_dictionary`

这些文件只能作为 Query Translator 调优和盲测设计材料，不能自动合并入正式知识库。

## 验证要求

每次改脚本后至少运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_extract_chatmed_field_candidates -v
.\.venv\Scripts\python.exe -m unittest tests.test_filter_chatmed_candidate_terms -v
.\.venv\Scripts\python.exe -m py_compile scripts\extract_chatmed_field_candidates.py
rg -n 'reference_answer|source_answer|"response"|二陈汉药|甘淡平|灵兰秘典|推理过程' datasets\structured\chatmed_candidates -S
rg -n 'reference_answer|source_answer|"response"|二陈汉药|甘淡平|灵兰秘典|推理过程' datasets\structured\chatmed_candidates\filtered -S
```

最后一条命令只应命中 summary 里的安全说明，不应在候选 JSONL 中发现 ChatMed 答案内容。
