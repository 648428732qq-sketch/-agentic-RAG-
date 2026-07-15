# 中医 RAG 金标准测试集执行清单

> 用途：提醒、对照和验收金标准数据集建设工作。  
> 目标：在不污染知识库和提示词的前提下，建立可复现的 600 条分层评测集，验证问题识别、规范词转换、召回、否定识别、追问门控和答案溯源。

## 一、基本原则

- [ ] 金标准问题及答案标签在运行 RAG 前确定。
- [ ] RAG 自己生成的回答只作为待评结果，不能直接作为正确答案。
- [ ] 测试问题不得加入 `markdown_docs/`、Qdrant、提示词或同义词表。
- [ ] LLM 只用于生成语言变体和辅助检查，不能单独决定临床正确答案。
- [ ] 每个标签必须能够追溯到本地 payload、古籍原文或可靠考试答案。
- [ ] 临床信息不足时允许多个候选，或把“必须追问”标记为正确结果。
- [ ] 不使用自动换算后的古代剂量作为金标准。
- [ ] 测试集冻结后，不针对测试原句增加固定别名。

## 二、目录准备

- [ ] 创建 `tests/evals/gold/`。
- [ ] 创建 `tests/evals/gold/candidates/`，保存尚未审核的问题。
- [ ] 创建 `tests/evals/gold/dev.jsonl`，用于调试和优化。
- [ ] 创建 `tests/evals/gold/test_locked.jsonl`，作为锁定盲测集。
- [ ] 创建 `tests/evals/gold/review_queue.jsonl`，保存有争议项目。
- [ ] 创建 `tests/evals/gold/manifests/`，记录数据来源、许可和版本。
- [ ] 在发布和灌库脚本中明确排除 `tests/evals/gold/`。

## 三、目标数量与分层

首版目标 600 条：

| 类型 | 目标数量 | 已收集 | 已审核 | 状态 |
|---|---:|---:|---:|---|
| 单一明确症状 | 80 | 80条种子 | 0 | 待生成、待审核 |
| 多症状组合 | 140 | 140条种子 | 0 | 待生成、待审核 |
| 模糊口语、隐喻表达 | 80 | 80条种子 | 0 | 待生成、待审核 |
| 否定、矛盾、反问 | 60 | 60条种子 | 0 | 待生成、待审核 |
| 错别字、语音转写、方言 | 40 | 40条种子 | 0 | 待生成、待审核 |
| 方剂、本草知识 | 70 | 70条种子 | 0 | 待生成、待审核 |
| 古籍理论、针法知识 | 50 | 50条种子 | 0 | 待生成、待审核 |
| 多轮上下文 | 40 | 40条种子 | 0 | 待生成、待审核 |
| 信息不足、域外及危险问题 | 40 | 40条种子 | 0 | 待生成、待审核 |
| **总计** | **600** | **600条种子** | **0** | **待生成、待审核** |

来源建议：

- [ ] 从本地 1419 条结构化 payload 生成 250 条来源约束问题。
- [ ] 从中医考试和公开数据提取 250 条候选问法。
- [ ] 从真实 WebUI 失败记录和安全问题中收集 100 条。
- [ ] 每个主要来源类型至少保留 40 条测试。
- [ ] 单一古籍或方剂数据不得占测试集 25% 以上。

## 四、本地 payload 候选生成

- [ ] 按 `source_type`、`source_book` 和难度分层抽取条目。
- [ ] 每个抽中条目生成直接、口语、信息缺失、否定和错字等变体。
- [ ] 临床问句中移除方剂名和条目标题，避免答案泄漏。
- [ ] 正确标签直接继承原 payload 的 `entry_id` 和来源字段。
- [ ] 生成问题不能增加 payload 中没有出现的症状或病机。
- [ ] 为相似方证构造 hard negative，测试是否会错误定方。
- [ ] 同一原始条目的语言变体设置相同 `paraphrase_family_id`。
- [ ] 对自动生成结果执行术语、否定、重复和证据校验。

## 五、公开数据处理

候选来源：

- [CMMLU 中医题](https://huggingface.co/datasets/shuyuej/CMMLU-Traditional-Chinese-Medicine-Benchmark)
- [Traditional Chinese Medicine Exam](https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Exam)
- [ShenNong TCM](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset)
- [TCMChat 600k](https://huggingface.co/datasets/ZJUFanLab/TCMChat-dataset-600k)
- [CMtMedQA](https://huggingface.co/datasets/Suprit/CMtMedQA)

执行检查：

- [ ] 记录仓库 ID、版本、许可、下载日期和原始 split。
- [ ] CMMLU/中医考试题主要用于理论与知识测试。
- [ ] CMtMedQA 必须筛选中医内容，并优先使用患者问法而非答案。
- [ ] ShenNong 和 TCMChat 只作为候选问法来源，答案必须重新核验。
- [ ] 开源问句必须重新关联本地 `entry_id` 或标记为无可靠命中。
- [ ] 无法关联本地证据的问题不得强行指定方剂。
- [ ] 不把公开数据全量灌入活动 RAG 知识库。

## 六、真实问题收集

- [ ] 只收集已脱敏的用户问题和必要上下文。
- [ ] 删除姓名、电话、住址、身份证、病历号等个人信息。
- [ ] 记录原始错别字和口语，不提前修正原问题。
- [ ] 同时保存 Query Translator、Top-K、门控和延迟结果。
- [ ] 记录用户反馈，但不自动修改知识库或同义词表。
- [ ] 将“未理解”“错误命中”“应命中其他条目”写入 review queue。
- [ ] 对处方、孕妇、儿童、老人和急症问题单独标记安全等级。

## 七、金标准字段

每条记录至少包含：

```json
{
  "id": "clinical_0001",
  "query": "盖两床被子还是冷，身上酸疼又不出汗",
  "category": "clinical_colloquial",
  "intent": "clinical_symptom",
  "expected_term_groups": [["恶寒"], ["身疼", "肢体酸楚疼痛"], ["无汗"]],
  "negative_terms": [],
  "acceptable_entry_ids": [],
  "acceptable_formulas": ["麻黄汤", "九味羌活汤"],
  "expected_source_types": ["formula_syndrome"],
  "must_clarify": true,
  "forbidden_terms": ["汗出"],
  "evidence_source": "伤寒论/方剂大全",
  "difficulty": "hard",
  "paraphrase_family_id": "family_0001",
  "split": "test",
  "review_status": "double_reviewed"
}
```

- [ ] `expected_term_groups` 支持本地同义规范词，不强制唯一措辞。
- [ ] `acceptable_entry_ids` 支持多个合理候选。
- [ ] `must_clarify` 明确哪些问题不允许自动定方。
- [ ] `forbidden_terms` 覆盖否定、禁忌和不能补充的症状。
- [ ] `evidence_source` 能追溯到古籍、方剂、本草或考试来源。
- [ ] `review_status` 明确自动生成、单审、双审或争议状态。

## 八、审核流程

- [ ] 自动检查字段、JSON schema、重复和来源有效性。
- [ ] 自动检查 `entry_id` 是否存在于当前结构化库。
- [ ] 自动检查预期术语是否真实存在于对应 payload。
- [ ] 自动检查正向术语与否定术语是否冲突。
- [ ] 使用第二模型辅助发现分歧，但不直接覆盖标签。
- [ ] 简单知识题抽查 10%-20%。
- [ ] 多方证、处方、安全和禁忌项目全部进入人工 review queue。
- [ ] 涉及临床诊疗正确性的项目由中医专业人员最终确认。

## 九、数据切分和防泄漏

- [ ] 开发集 200 条，用于调试提示词、阈值和检索策略。
- [ ] 锁定测试集 400 条，只用于正式回归。
- [ ] 同一 `paraphrase_family_id` 不得跨越 dev/test。
- [ ] 测试问题不得出现在 `SYMPTOM_ALIASES` 或 Prompt 示例中。
- [ ] 测试文件不得进入 `markdown_docs/`、`datasets/unified/` 或 Qdrant。
- [ ] 冻结后记录文件 SHA-256 和数据版本。

## 十、验收指标

| 指标 | 目标 | 当前结果 | 是否达标 |
|---|---:|---:|---|
| 意图识别 Macro-F1 | ≥95% | 待测 | 否 |
| 规范词召回率 | ≥95% | 待测 | 否 |
| 否定识别 F1 | ≥98% | 待测 | 否 |
| 候选 Recall@40 | ≥98% | 待测 | 否 |
| 最终 Recall@5 | ≥95% | 待测 | 否 |
| 应追问问题识别率 | ≥95% | 待测 | 否 |
| 无依据正向结论 | 0 | 待测 | 否 |

- [ ] 输出整体指标。
- [ ] 输出每个来源类型和难度分层指标。
- [ ] 输出失败问题、候选排名和错误分类。
- [ ] 不使用整体平均值掩盖《温病条辨》等低覆盖来源。

## 十一、错误驱动的优化顺序

```text
Recall@40 < 95%
  -> 补数据、Query Translator、候选词和初始混合检索
  -> 不增加 reranker

Recall@40 >= 98%，但 Recall@5 < 95%
  -> 候选存在但排序不准
  -> 再评估 reranker

Recall@5 >= 95%
  -> 暂不增加 reranker
```

- [ ] 每次改动前保存基线报告。
- [ ] 每次只处理一种错误类别。
- [ ] 优化后同时跑 dev 和锁定回归测试。
- [ ] 禁止通过添加测试原句别名刷分。

## 十二、完成定义

- [ ] 600 条记录字段完整且无重复泄漏。
- [ ] 400 条锁定测试集已冻结并记录 SHA-256。
- [ ] 所有标签可追溯到本地 payload 或可靠来源。
- [ ] 临床高风险项目完成专业审核或明确标记不作临床结论。
- [ ] 分层 Recall@5 达到目标，且无依据正向结论为 0。
- [ ] 失败案例已进入下一轮数据或算法优化队列。
