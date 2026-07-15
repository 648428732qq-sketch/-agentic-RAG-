# Grounded RAG 端到端回归报告（2026-07-01）

## 结论

- V3 Query Translator 的模型、术语目录、在线可选接入和失败回退已经具备。
- V3 **暂不允许默认启用**：它改善了术语与条目召回，但没有解决临床方证的安全误放行。
- 拒答/追问现在均携带稳定原因码、中文解释、处理策略和主原因，可直接在 WebUI 展示并用于评测。
- 当前不应进入 rerank。首要阻塞是结构化 payload 的 `required_symptom_groups` 不完整。

## 运行环境

- 远端项目：`/home/user/gptdata/cz/yangzhiting/agentic-rag-for-dummies/agentic-rag-for-dummies`
- 隔离环境：`.venv-linux`
- GPU：`CUDA_VISIBLE_DEVICES=1`
- V3 模型 SHA-256：`251bd00a2345c16a8c14216a9ab574ce7b6f01ee1c4206ea803ac94e92addb77`
- V3 catalog SHA-256：`ab63cf5a71c0e6387a852043475005ccbf03b4922220b0d6d41932695a0a2286`
- LLM Translator：关闭，未调用 DeepSeek。
- rerank：关闭。

## Qdrant 状态

远端原 `qdrant_db` 无法加载：`sqlite3.DatabaseError: database disk image is malformed`。在同一共享挂载下新建持久化 Qdrant 时又出现 `sqlite3.OperationalError: disk I/O error`，说明该挂载不适合 Qdrant Local 的 SQLite 持久化语义。

为避免删除原库或写出项目目录，本轮从冻结的 `syndrome_dictionary.jsonl` 在评测进程内建立 Qdrant 内存集合，共 `1,419` 点。向量检索、payload 返回、冲突过滤和证据门控均走真实 Qdrant 客户端逻辑；集合只是不落 SQLite。

## 100 条冻结集结果

| 指标 | 规则基座 | V3 双塔 | 变化 |
|---|---:|---:|---:|
| 术语组召回 | 49.85% | 55.69% | +5.84 pp |
| 全部术语组命中案例 | 32% | 35% | +3 pp |
| 目标条目 Recall@8 | 55% | 57% | +2 pp |
| 来源类型 Recall@8 | 75% | 77% | +2 pp |
| 原始合成 gate 标签准确率 | 43% | 46% | +3 pp |
| 全部 Hard Negative 安全拒答 | 70.83% | 70.83% | 0 |
| 实际拒答原因覆盖 | 100% | 100% | 0 |
| 平均查询延迟 | 458.46 ms | 669.51 ms | +211.05 ms |

V3 提升了 ASR、隐喻等术语召回，但目标条目 Recall@8 仍只有 57%，不能声称达到 95%。

## 拒答原因

V3 的 64 条实际拒答/追问全部带原因。主要原因如下：

| 原因码 | 数量 | 中文含义 |
|---|---:|---|
| `competing_indistinguishable_matches` | 23 | 多个候选证据接近，当前信息不足以区分 |
| `too_few_payload_matches` | 21 | payload 诊断要素命中不足 |
| `low_query_coverage` | 18 | 候选只能解释部分用户表征 |
| `payload_evidence_confidence_below_threshold` | 16 | payload 证据可信度低于阈值 |
| `missing_required_symptom_groups` | 14 | 必要鉴别症状未确认 |
| `unknown_intent` | 8 | 无法确认问题属于本地可答范围 |
| `missing_required_knowledge_groups` | 7 | 知识条目的必要证据字段未完整命中 |
| `insufficient_knowledge_match` | 6 | 知识要点命中过少 |
| `no_grounded_match` | 3 | Qdrant 中没有满足约束的条目 |

WebUI 对 `no_match` 显示“拒答原因”，对 `clarify` 显示“暂不回答原因”和需要补充的症状；检索异常也会转成 `retrieval_failure:<类型>`，不会回退到自由生成。

## Hard Negative 标签问题

24 条合成 Hard Negative 中有 7 条被放行。其中 6 条是评测标签过严，而不是回答越界。例如：

- 用户明确询问“水道、堕胎、五癃对应什么药”，本地条目精确命中石蚕；仅因生成器漏掉同一原文中的“五淋”就要求拒答，不符合知识查询逻辑。
- 用户询问“补之、泻之如何理解”，本地针法原则直接命中；缺少某一目标条文中的“阳明”不应禁止返回其他有证据的解释。

这些题仍保留原始 gate 指标，但不能用来反向硬改门控，否则会把正确的本地知识命中改成拒答。

真正的临床安全失败是左金丸：问题给出呕吐、舌红苔黄、胁痛，却漏掉脉弦数；当前 payload 仅把呕吐列为必要组，因此错误放行。

## Payload 完整性

结构化字典含 180 个 `formula_syndrome`：

- 30 个方剂没有 `required_symptom_groups`。
- 41 个方剂只有 1 个必要症状组。
- 左金丸的 evidence、diagnostic_keys 中存在“脉弦数”，但必要组仅有呕吐。

这说明瓶颈是系统性的结构化字段缺口，不应通过增加方名专用规则修补。

## 验证

- 远端全量测试：`177 passed`。
- 远端门控/双塔/严格检索定向测试：`101 passed`。
- 基座和 V3 均完成 100 条 Qdrant 端到端回归。
- 训练/评测未调用 DeepSeek，未输出或覆盖任何密钥。

## 下一步门槛

1. 从 evidence、diagnostic_keys 和 differential_keys 自动提出方剂必要症状组候选，保留证据位置与来源字段。
2. 对 30 个空必要组和 41 个单组方剂进入 review queue；不能自动标为人工已审核。
3. 重新生成仅基于真实必要组缺失的 Hard Negative，知识查询与临床定方分开计分。
4. 新冻结集通过安全拒答与目标条目召回后，才比较本地排序、RRF 和二级 rerank。

远端原始报告：

- `datasets/structured/grounded_rag_e2e_v3_baseline_report.json`
- `datasets/structured/grounded_rag_e2e_v3_biencoder_report.json`
- 对应 predictions JSONL 位于同一目录。
