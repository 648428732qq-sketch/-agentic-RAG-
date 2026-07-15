# Query Translator 召回补强状态

更新时间：2026-06-27

## 本轮目标

提升口语、方言、隐喻、错别字、否定表达进入本地方证库的稳定召回能力，同时不放松相似方证 hard negative 的安全门槛。

本轮没有把规则绑定到具体方剂，而是补“表达模式 -> 规范词”的通用归一化：

- 风一吹难受、遭不住风吹、过堂风躲着 -> 恶风
- 冻得打颤、透骨地冷、厚棉被仍冷 -> 恶寒
- 周身酸痛、像灌了铅 -> 身疼，并扩展到肢体酸楚疼痛/头身疼痛
- 汗星子挤不出来 -> 无汗
- 渗汗、往外渗汗、直淌汗 -> 汗出
- 那股气从小腹窜进心窝 -> 气上冲/欲作奔豚
- 眼前雾水散尽、视线分明 -> 明目
- 把脉找手腕寸半截 -> 寸口
- 连咳嗽都没有 -> 咳嗽为否定词
- “哪些法子能明目”这类知识问法不再被误判为反问否定
- “跟针扎似的揪痛”按疼痛比喻处理，不再误判成针灸操作问题
- “鼻子憋得死紧连不上风、脑袋针扎样揪痛”可归一化为鼻塞、头痛

## 当前验证结果

### 单元测试

命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

结果：122/122 通过。

### 相似方证 hard negative

命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_formula_hard_negatives.py --max-formulas 150 --top-k 8 --quiet
```

结果文件：

```text
datasets/structured/formula_hard_negative_broad_report.json
```

核心结果：

- 400/400 通过
- pass_rate = 1.0
- tested_formula_count = 109
- full_signature_topk.target_in_topk = 1.0
- missing_required_clarify.not_grounded = 1.0
- forbidden_conflict.no_unsafe_target_grounded = 1.0
- shared_terms_clarify.not_grounded = 1.0

这说明本轮口语规则补宽后，没有把易混方证误放行。

### final_holdout 本地降级链

命令：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_query_translator.py `
  --cases tests\evals\query_translator_benchmarks\final_holdout\private\gold_keys.jsonl `
  --report datasets\structured\query_translator_final_holdout_local_after_norm4.json `
  --quiet
```

结果：

- 100 条中 49 条全项通过
- candidate_coverage = 1.0
- term_recall = 0.77
- negative_terms = 0.94
- retrieval = 1.0
- decision = 0.64

本地降级链不等于线上最终效果；它主要证明无 LLM 时也能覆盖更多口语变体。

### final_holdout 离线 DeepSeek 回放估算

使用旧 DeepSeek 预测结果叠加当前本地归一化，生成：

```text
datasets/structured/query_translator_final_holdout_predictions_current_replay_norm4.jsonl
datasets/structured/query_translator_final_holdout_report_current_replay_norm4.json
```

核心结果：

- 100 条中 93 条全项通过
- candidate_coverage = 1.0
- term_recall = 1.0
- negative_terms = 1.0
- no_forbidden_positive = 1.0
- retrieval = 1.0
- decision = 0.93

注意：这是离线估算，不是重新调用 DeepSeek 的真实线上重跑。真实重跑需要等 DeepSeek/平台额度恢复后执行。

## 离线回放剩余标签/决策冲突

剩余失败全部是 decision，不是术语理解失败。主要集中在：

- 恶寒 + 身疼 + 无汗，但缺少喘、脉浮紧等关键鉴别项，当前安全策略要求追问。
- 鼻塞 + 头痛 + 恶风曾被“针扎样疼痛”误分到针灸意图；当前代码已修复并定点复测为 `clarify`。
- 饭吃不下 + 胃胀痛当前代码已识别为临床症状并转为 `clarify`。

这类失败不是 rerank 能解决的。rerank 只能重排候选，不能决定“缺关键鉴别项时是否允许直接回答”。要提高这些 decision，需要调整安全策略或补更完整的结构化 payload。

## 当前 rerank 判断

rerank 已完成真实本地模型烟测和小规模多方剂 A/B，但暂不应默认启用。

原因：

- 本地硬排序 + payload 过滤在 broad hard negative 上已达到 400/400。
- final_holdout 的 term_recall 离线估算已到 1.0。
- 剩余瓶颈是 decision safety gate，不是 top-k 排名。
- `BAAI/bge-reranker-v2-m3` 已完整缓存并可离线加载；当前虚拟环境是 CPU 版 PyTorch。
- 8 个方剂、32 个场景的 rerank A/B 为 32/32，通过率与基线相同。
- 基线平均延迟约 0.89 秒，rerank 平均约 12.26 秒；没有可测收益，额外增加约 11.37 秒。

结果文件：

```text
datasets/structured/formula_hard_negative_8_baseline_report.json
datasets/structured/formula_hard_negative_broad_rerank_report.json
```

更重要的是，现有 100 条 final_holdout 没有 `expected_formula`、`expected_entry_id` 或相关性等级。因此其中的 `retrieval = 1.0` 不能证明排名正确，只代表该批样本没有配置可执行的排名断言。400 条 broad hard negative 有排名标签，但完整方证基线 Top-1 已经是 100%，无法衡量 rerank 的增益。

rerank 更适合作为“同证据层候选”的二级排序，而不是用来弥补 Query Translator 或缺失 payload。

## 下一步

1. 创建独立的 rerank 排名盲测：每条必须有 `expected_entry_id` 或可接受方剂集合，并标注 hard negatives；生成时覆盖至少 50 个方剂，测试集与调参集隔离。
2. 在该盲测上记录 pre-rerank / post-rerank 的 Recall@5、MRR@10、nDCG@10、安全门控回归和 P95 延迟；只有排名显著提升且门控零退化才启用。
3. 等 DeepSeek 额度恢复后，重新跑 final_holdout live 预测，确认真实线上 term_recall 是否达到 0.95 以上。
4. 保留“恶寒 + 身疼 + 无汗但缺喘/脉象时追问”的安全 gate；当前自动生成金标签中把这类样本标成 `grounded_answer`，应人工或规则审裁后另存修订版，不能直接降门槛迎合标签。
5. 后续数据补强重点是给结构化方证 payload 补 required_symptom_groups、forbidden_terms、diagnostic_keys、pathogenesis，尤其是易混方之间的鉴别项。
