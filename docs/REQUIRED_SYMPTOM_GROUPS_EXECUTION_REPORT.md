# 必要症状组补强与 Hard Negative 执行报告

日期：2026-07-01

## 执行范围

- 处理 30 个 `required_symptom_groups` 为空的方剂。
- 处理 41 个只有一个必要组的方剂。
- 从本地 `raw_text` 中的“辨证要点/证治要点”抽取主症、寒热、舌象、脉象和关键鉴别项。
- 每个候选保存原文、字段名、字符起止位置、上下文摘录、来源文件、来源 URL 和内容哈希。
- 候选先进入队列，再经规则校验合并到 reviewed replacements；未覆盖基础字典。
- 由真实必要组生成逐组删除的 Hard Negative，并在真实检索链路验证追问/拒答。

## 产物

- 候选队列：`datasets/structured/syndrome_required_groups_candidates.jsonl`
- 候选报告：`datasets/structured/syndrome_required_groups_candidates_report.json`
- 审核替换：`datasets/structured/syndrome_dictionary_reviewed_replacements.jsonl`
- 有效字典：`datasets/structured/syndrome_dictionary_effective.jsonl`
- Hard Negative：`tests/evals/required_group_hard_negative_v1/cases.jsonl`
- 验收摘要：`datasets/structured/required_group_hard_negative_acceptance.json`

## 数据结果

| 项目 | 结果 |
|---|---:|
| 待补强方剂 | 71 |
| 原空必要组 | 30 |
| 原单必要组 | 41 |
| 自动规则校验通过 | 71 |
| reviewed replacements 总数 | 84 |
| 有效字典条目 | 1419 |
| 方剂条目 | 180 |
| 补强后空必要组 | 0 |
| 补强后单必要组 | 0 |

所有新增组都能回溯到本地原文。两个仅靠辨证短语不足的条目使用同一条目内的明确证据补足，并保留 fallback 轨迹；没有调用 DeepSeek、Qwen 或其他外部模型生成医学字段。

## 检索修正

回归首次暴露出“完整症状也被拒绝”的问题：目标条目虽全部排第 1，但 Query Translator 将一个原始症状扩成多个派生别名，门控误把派生词当作独立主诉计算覆盖率。

已完成通用修正：

- 从全部本地结构化证据字段建立不受 800 词上限影响的精确术语索引。
- 将原问题中直接命中的 payload 词记录为 `literal_payload_terms`。
- 精确命中词用于主查询覆盖率；派生别名只用于召回，不再制造虚假缺失。
- 明确否定词不会被精确索引重新注入为阳性症状。
- “患者目前有/表现为/症状为/已知症状”等通用叙述统一识别为临床主诉。
- 验收增加 `full_grounded_rate >= 0.95`，防止“全部拒答”获得通过。

## Hard Negative 验收

测试共 361 条：71 条完整必要组，290 条每次只删除一个必要组的反例。

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| 完整方证目标 Top-8 召回 | 100% | 80% |
| 完整方证门控放行 | 95.77% | 95% |
| 缺一组安全拒答/追问 | 100% | 100% |
| 缺组拒答原因覆盖 | 100% | 100% |
| 检索路由成功 | 100% | 100% |
| Top payload 无冲突 | 100% | 100% |

290 个缺组样本全部包含 `missing_required_symptom_groups`。剩余 3 个完整样本被保守门控拦下，目标条目仍排第 1：两个属于近似疝证鉴别，一个属于相近血热出血方竞争。未为追求 100% 而降低安全门控。

评测关闭了 LLM、rerank 和 Query Translator 双塔模型，只使用本地规则、payload、混合检索和 GPU embedding。因此结果不依赖外部模型记忆答案。

## 测试

- 第二轮定向回归：103 passed。
- 全量回归：191 passed。
- Qdrant 重建脚本新增测试：2 passed。
- Python 语法检查通过。

## Qdrant 状态

1419 个有效条目已重建到进程内真实 Qdrant，并用于上述 361 条端到端验收。

远端正式配置使用项目共享目录上的本地 Qdrant。为避免破坏正式库，先在项目 `cache/` 创建独立持久化探针；首次 upsert 稳定复现：

```text
sqlite3.OperationalError: disk I/O error
```

这属于共享文件系统与 Qdrant Local/SQLite 锁及 fsync 的兼容问题。正式 Qdrant 数据库未移动、删除或覆盖。生产持久化需满足以下任一条件：

1. 在服务器节点本地 SSD 上运行 Qdrant Server，并通过 `QDRANT_URL` 连接。
2. 将 `QDRANT_DB_PATH` 指向支持 SQLite 锁和 fsync 的节点本地文件系统。

`scripts/rebuild_effective_syndrome_qdrant.py --apply` 已准备好；切换到合适后端后可直接从有效字典重建并校验 1419 点。

## 是否需要额外数据

当前 71 个方剂的必要组补强不需要额外数据，本地“辨证要点”和主治原文已覆盖。

若目标升级为临床产品级，仍建议补充两类资源，但不阻塞当前 RAG 验收：

- 有授权的权威方剂教材或标准，用于人工/专家复核规则抽取结果。
- 近似方证的明确鉴别条文，尤其疝证与血热出血方，用于减少目前 3 个保守追问。

当前真正阻塞持久化上线的是 Qdrant 存储后端，不是医学数据缺失。
