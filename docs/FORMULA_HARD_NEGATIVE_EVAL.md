# 相似方证 Hard Negative 评测

## 目标

本评测用于验证“本地结构化 payload + 硬排序”在相似方证场景下是否足够稳定，再决定 rerank 是否只作为同证据层候选的二级重排。

它不是只测“麻黄汤 / 桂枝汤 / 小青龙汤 / 苏子降气汤”。脚本会从 `datasets/structured/syndrome_dictionary.jsonl` 的 `formula_syndrome` 条目中自动选择多方剂样本，并强制纳入上述易混方作为代表样本。

## 运行命令

```powershell
$env:PYTHONIOENCODING='utf-8'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
.\.venv\Scripts\python.exe scripts\evaluate_formula_hard_negatives.py --max-formulas 60 --top-k 8 --quiet
```

全量方剂候选评测使用：

```powershell
$env:PYTHONIOENCODING='utf-8'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
.\.venv\Scripts\python.exe scripts\evaluate_formula_hard_negatives.py --max-formulas 150 --top-k 8 --quiet
```

输出文件：

- `tests/evals/formula_hard_negative_broad/cases.jsonl`
- `datasets/structured/formula_hard_negative_broad_predictions.jsonl`
- `datasets/structured/formula_hard_negative_broad_report.json`

## Case 类型

- `full_signature_topk`：完整方证信息是否能把目标方剂召回到 top-k。
- `missing_required_clarify`：故意漏掉一个必要症状组，系统必须追问或拒绝确定回答。
- `forbidden_conflict`：加入反向/禁忌症状，系统不能把原方作为确定答案。
- `shared_terms_clarify`：只给相似方共有症状，系统必须识别信息不足，不能强行锁定一个方。

## 当前结果

最近一次全量候选 broad run：

- 方剂库总数：150
- 可测方证数：109
- case 总数：400
- 总通过：400 / 400
- 总通过率：100.00%
- 安全阈值：通过

核心指标：

- 完整方证 top-k 召回：100.00%
- 漏关键症状不强答：100.00%
- 漏关键症状目标方保留 top-k：100.00%
- 漏关键症状 missing group 识别：100.00%
- 禁忌/反向症状不错误确定原方：100.00%
- 禁忌/反向症状过滤或冲突识别目标方：100.00%
- 共有症状不强答：100.00%
- 平均延迟：349.29 ms
- P95 延迟：443.67 ms

2026-06-27 最终复测结果：400 / 400，通过率 100%；平均延迟 339.67 ms，P95 440.37 ms。

## Rerank A/B

评测脚本支持真实 CrossEncoder 二级重排：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_formula_hard_negatives.py `
  --max-formulas 8 --top-k 8 --with-rerank `
  --rerank-candidates 8 --rerank-device cpu `
  --rerank-max-length 256 --rerank-mode evidence_first --quiet
```

本机 `BAAI/bge-reranker-v2-m3` 已完整缓存。8 个方剂、32 个场景的 A/B 结果：

- 基线：32 / 32；平均 890.18 ms，P95 420.56 ms。
- rerank：32 / 32；平均 12256.43 ms，P95 17167.77 ms。
- 安全门控无退化，但准确率无增益，平均额外延迟约 11.37 秒。

当前 PyTorch 为 CPU 构建，因此该结果不代表 GPU 延迟；但在没有独立排名金标签证明收益前，不应仅为加技术组件而默认启用 rerank。

## 结论

当前本地 payload + 硬排序已经满足“先安全、再召回”的基础要求：完整方证信息可以 100% 召回目标方，禁忌/反向症状可以 100% 阻止目标方被确定命中，相似共有症状和缺关键症状时不会贸然给确定方剂。

系统已新增本地 `payload_overlap` 召回路线：Query Translator 抽出的规范症状会直接命中结构化 payload 的证据字段，再交给硬排序和 gate。这样在用户漏掉关键症状时，目标方仍能保留在 top-k，便于后续追问补齐后继续命中，而不是追问之后丢失候选。

本轮额外补强了五个点：

1. `missing_required_clarify` 生成器增加泄漏校验：遗漏组不能通过原词、子串、Query Translator 规范词再次进入问题；禁忌词的等价映射也不能进入问题，例如“口干”被本地映射为“口渴”时，不能用于需要避开“口渴”的 hard negative。
2. 本地硬排序调整为优先保留高 query coverage 的缺项候选，再由决策层依据 `required_symptom_groups` 追问，避免用户已说症状覆盖很高的目标方被一批低覆盖但 required group 完整的候选挤出 top-k；当前保留阈值为 0.8，相当于 5 个规范症状里命中 4 个也能保留为追问候选，但不会直接落地为确定答案。
3. 同一方名存在多条证据时，优先保留 `formula_syndrome` 结构化 payload，而不是让同名古籍条文抢占 top-1，保证问诊检索先走结构化方证字段。
4. 麻黄汤类 direct clinical signature 收紧：`恶寒 + 身疼 + 无汗` 不再足以直接定方，必须同时有 `喘 / 喘咳 / 咳喘` 等呼吸证据；缺喘时进入追问。
5. direct clinical signature 拆成严格型和高区分型：风寒表实类签名仍要求必要症状组完整；`痰涎清稀 + 不得平卧 + 喘咳/咳嗽` 这类高区分组合允许在缺少外感项时进入排序和落地判断，用于保护小青龙汤等“外寒里饮”场景不被泛咳喘方剂压掉。

下一步仍不建议让 rerank 直接接管总排序。rerank 只适合作为同证据层候选的二级重排，不越过必要症状组、禁忌症状和本地证据 gate。更高优先级是继续扩大盲测集，尤其是非经典高频方、口语化症状、多症状长句、互斥寒热/虚实/汗出/口渴组合。

## 已暴露问题

- 旧版 hard negative 生成器会把“故意遗漏的症状”通过重叠词再次写回问题，例如遗漏“喘”却保留“咳喘”，遗漏“发热/壮热”却保留“壮热面赤”。已通过生成器泄漏校验修复。
- 结构化字典已清理“口干/口渴”和“不渴”在 required group 中互相打架的问题；Query Translator 已修复“多汗/汗出”被候选“无汗”污染的问题。后续还要继续审计寒热、虚实等类似互斥组。
- `validate_syndrome_dictionary.py` 已确认 Qdrant `tcm_syndrome_entries` 与 JSONL 一致：1419 条 / 1419 点。
