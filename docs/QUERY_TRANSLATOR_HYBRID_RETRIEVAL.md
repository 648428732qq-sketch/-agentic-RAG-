# Query Translator 与混合检索说明

## 目标

这条链路把用户口语映射到本地结构化中医数据，再从 Qdrant 返回可核验的 payload。模型只负责受约束的语义对齐，不负责诊断、定方、补充病因或生成剂量。

核心原则：

1. 新口语不依赖逐条加入别名字典。
2. 规范词必须来自当前 Qdrant 候选 payload。
3. 每个模型映射必须绑定用户原文字符区间。
4. 方名、剂量、加减和“主之/可与”等干预语句不能进入 Query Translator 白名单。
5. 临床查询的所有已确认表征必须由同一条 payload 覆盖；多个候选无法区分时只追问，不自动定方。

## 运行流程

```text
用户口语
  -> 本地意图识别、已知别名和否定解析
  -> 原始问题向量召回 + 本地 BM25 召回
  -> 从 Top-N payload 动态收集规范词白名单
  -> DeepSeek 结构化映射（仅在本地信息不足时调用）
  -> 原文区间、白名单、极性和干预词校验
  -> 规范词向量召回
  -> Raw Dense + Canonical Dense + BM25 的 RRF 融合
  -> 否定冲突排除、唯一规范词覆盖度和候选分歧门控
  -> payload 直接格式化回答，或生成确定性追问
```

## 泛化机制

`project/core/symptom_query_translator.py` 不包含测试问句示例。DeepSeek 每次只能从当前检索结果生成的候选规范词中选择，因此新古籍入库后，新的症状、理论词和针法词会自动成为候选，不要求先修改 Python 字典。

`SYMPTOM_ALIASES` 只承担低延迟快速路径和确定性否定解析。它不是完整知识边界，也不应为了单条测试不断追加表达。未登录口语由受约束模型映射；模型不可输出候选集合之外的术语。

模型输出中的 `source_start`、`source_end` 是用户原话的左闭右开字符区间。服务端重新切片得到 `source_phrase`，不信任模型自由填写的证据文本。部位、时间和程度字段同样必须能在用户原话中定位。

## 混合检索与门控

`project/core/syndrome_retriever.py` 使用三路检索：

- `raw_dense`：保留未见口语的语义召回。
- `lexical`：本地 BM25，使用中文 2/3 字符 gram 和精确字段匹配。
- `canonical_dense`：使用通过校验的规范词再次向量召回。

三路结果通过 RRF 融合。排序优先使用唯一规范词命中数、诊断字段命中数、词语具体程度和查询意图对应的来源类型，不再用重复别名累计分数。

临床查询满足以下条件才直接展示最接近的 payload：

- 达到来源类型所需的最少规范词数。
- Top-1 覆盖全部已确认正向表征，`SYNDROME_MIN_QUERY_COVERAGE=1.0`。
- 不与用户明确否认的表征冲突。
- 相近候选之间存在足够的区分证据。
- 一般症状命中针刺操作或单味药时仍先追问；只有明确的针法/本草知识问句可直接展示知识条目。

不满足时，`format_syndrome_clarification()` 从候选 payload 的缺失辨证要素生成追问。`ChatInterface` 会在这里终止，不再回落到旧 LangGraph 自由生成答案。理论、方剂组成和针法概念等知识查询仍可在精确命中时直接回答。

## 稳定性和负载

默认配置位于 `project/config.py`：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `ENABLE_HYBRID_SYNDROME_RETRIEVAL` | `True` | 关闭时仅保留原始问题向量召回，便于故障隔离 |
| `LLM_SYMPTOM_TRANSLATOR_MIN_LOCAL_TERMS` | 3 | 本地少于 3 个规范词时才请求模型 |
| `LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS` | 12 | 单次模型硬超时 |
| `LLM_SYMPTOM_TRANSLATOR_CACHE_SIZE` | 128 | 相同查询与候选集合的翻译缓存 |
| `LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD` | 2 | 连续失败后打开熔断器 |
| `LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS` | 120 | 熔断冷却时间 |
| `SYNDROME_PAYLOAD_CACHE_TTL_SECONDS` | 300 | Qdrant payload/BM25 内存缓存 |
| `SYNDROME_TRANSLATOR_CONTEXT_CANDIDATES` | 16 | 用于形成动态白名单的候选 payload 数 |
| `SYNDROME_TRANSLATOR_MAX_TERMS` | 120 | 白名单上限 |

Query Translator 使用独立、可关闭的 DeepSeek HTTP 客户端，避免 LangChain 超时后残留连接。失败、超时或熔断时自动退回本地解析；本地链路召回会降低，但不会凭空补词。

Embedding 默认使用本地缓存，`EMBEDDING_LOCAL_FILES_ONLY=1`，防止每次启动向 Hugging Face 发探测请求。如首次部署机器没有模型缓存，可临时设为 `0` 完成下载，再恢复为 `1`。

## 验证

未见表达评测集：`tests/evals/query_translator_cases.jsonl`。它覆盖未登录口语、近义规范词、错别字、否定冲突、理论、针法、本草、方剂知识、信息不足和域外问题。

```powershell
.\.venv\Scripts\python.exe scripts\validate_query_translator_safety.py
.\.venv\Scripts\python.exe scripts\evaluate_query_translator.py --with-llm --fail-on-error
.\.venv\Scripts\python.exe scripts\finalize_query_translator_validation.py
.\.venv\Scripts\python.exe scripts\validate_syndrome_dictionary.py
.\.venv\Scripts\python.exe scripts\validate_structured_chat_runtime.py
```

主要报告写入 `datasets/structured/`：

- `query_translator_safety_validation.json`
- `query_translator_eval_llm_final.json`
- `query_translator_release_validation.json`
- `syndrome_dictionary_validation.json`
- `structured_chat_runtime_validation.json`

评测必须同时检查候选覆盖、规范词召回、否定词、禁止正向幻觉、Top-K 检索、门控和最终决策。不要只用“目标方剂是否 Top-1”作为通过条件；症状信息不足时，安全的正确结果是候选召回成功并要求补充信息。

## 新数据维护

新增或重建古籍结构化 payload 后：

1. 确保 `ancient_symptoms`、`modern_symptoms`、`diagnostic_keys`、理论词或针法词是独立字段，不把整段“某方主之”混入症状字段。
2. 重建 `tcm_syndrome_entries`。动态白名单会自动读取新 payload，不需要同步写死提示词。
3. 在评测集中加入未见表达和否定表达，使用同义规范词组与 Top-K 来源验收。
4. 先跑离线安全套件，再跑 DeepSeek 全量评测和聊天运行时验证。
5. 只有稳定、高频且无歧义的表达才考虑加入 `SYMPTOM_ALIASES` 快速路径。

## 2026-06-24 结构化检索补强

本次补强把结构化方证 payload 从“可召回”推进到“可过滤、可排序、可追问”：

- `required_symptom_groups`：每组代表一个必须覆盖的辨证要点，同组内可以是同义表达。例如麻黄汤要求恶寒、无汗、喘；小青龙汤要求恶寒、无汗、喘咳、痰多或痰涎清稀。
- `forbidden_terms`：用于排除正向误命中，例如麻黄汤排除汗出、有汗、自汗；桂枝汤排除无汗。该字段不参与用户否定症状的冲突判断，避免“用户否认汗出”反而冲突到“禁忌汗出”的候选。
- `differential_keys`：用于本地排序的鉴别要点，例如小青龙汤的外寒里饮、痰涎清稀、不得平卧；苏子降气汤的上实下虚、胸膈满闷、呼多吸少。
- `must_clarify_fields`：当命中不完整时生成追问，例如是否出汗、痰是清稀还是黄稠、能否平卧。

排序策略也同步调整：

1. 明确意图先按 `source_type` 对齐，例如针法问题优先 `classical_acupuncture_principle`，不让普通理论条文抢占。
2. 临床症状再看 `differential_keys` 命中数和覆盖率，防止只靠“咳嗽、喘、痰多”等泛症状误排。
3. 再看 `required_symptom_groups` 完整度，信息缺失时允许召回但必须追问。
4. 最后再比较 `diagnostic_keys`、查询覆盖率、规范词命中数和 BM25/dense 融合分。

数据构建和验证新增硬约束：`required_symptom_groups` 与 `forbidden_terms` 不允许有交集；`formula_syndrome` 和 `classical_clause` 必须有 `differential_keys`。`validate_syndrome_dictionary.py` 会输出各来源类型的新字段覆盖率，并校验 Qdrant payload 与 JSONL 一致。

本次实测结果：

- 结构化字典：1419 条，Qdrant `tcm_syndrome_entries` 1419 点一致。
- 字段覆盖：方剂 180 条全部有 `differential_keys`，150 条有 `required_symptom_groups`，83 条有 `forbidden_terms`，97 条有 `must_clarify_fields`。
- Query Translator 过滤评估：42/42，通过率 100%。
- 全量单测：75/75 通过。

仍不建议立刻上 rerank。当前更关键的收益来自可解释 payload 和本地硬排序；rerank 适合在 hard negative 测试集稳定后，作为同证据层候选的二级重排。

## 已知边界

- 当前结构化库为 1419 条，检索质量仍受原始 payload 的症状拆分和 `diagnostic_keys` 质量限制。
- 系统返回的是文献匹配，不是个人诊断或处方。
- DeepSeek 增加数秒延迟；模型不可用时系统会保守追问。
- 本地 BM25 当前在进程内构建，数据规模显著增长后应改为持久化稀疏索引或 Qdrant sparse vector。

## 2026-06-20 验证快照

- 离线安全套件：通过，包含硬超时、原文区间、干预词屏蔽、双重否定、否定复合词收窄、针刺临床门控和熔断测试。
- 结构化字典：1419 条 JSONL 与 Qdrant 1419 点一致，6 类 payload 数量一致，无解析或 schema 错误。
- 聊天运行时：通过；临床歧义和检索故障均未回落旧 Agent。
- 最近一次 16 条 DeepSeek 完整运行：规范词召回、候选覆盖、否定识别、禁止正向幻觉和 Top-K 检索均为 100%；15 条端到端通过。唯一未通过项是“胃胀痛、吃不下”在模型输出更具体时被针刺条文直接放行。
- 上述针刺放行已改为确定性“临床针刺/单味药必须追问”并通过离线回归。由于在线额度已满，`finalize_query_translator_validation.py` 保留原在线报告及 SHA-256，只重放唯一失败的确定性门控，生成独立发布验收报告；它不会把离线重放伪装成第二次在线调用。
