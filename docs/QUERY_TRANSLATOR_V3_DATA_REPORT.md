# Query Translator V3 数据报告

## 当前状态

- 地域方言：已排除。
- 新冻结种子：100 个未使用本地条目。
- 新训练种子：300 个未使用本地条目。
- Qwen 隐喻/残句训练问题：480 条。
- 本地历史 ASR 混淆对生成问题：600 条。
- 最终训练监督：1,080 条问题、3,851 个术语映射。
- 训练与冻结 `entry_id` 重叠：0。
- 训练与冻结规范化 query 重叠：0。

## 冻结集

路径：`tests/evals/query_translator_v3_nondialect_100/`

| 风格 | 数量 |
|---|---:|
| 隐喻/残句 | 24 |
| ASR 同音错字 | 24 |
| 多症状长句 | 24 |
| Hard Negative | 24 |
| 否定/不确定 | 4 |

剩余未使用本地条目中只有 4 个入选条目带可靠 `forbidden_terms`，因此没有人为伪造 20 条否定金标。

冻结金标 SHA-256：`fd51b502701b76097dc2b9ed74d6e1102806a1de85e60f2e94d013b0d3928578`

## 训练监督

路径：`datasets/external/query_translator_augmented_v3/supervision/query_term_pairs.jsonl`

- 480 条隐喻由 `qwen3.6-flash` 生成，并通过 source phrase 逐字存在、规范词白名单和症状组完整性校验。
- Qwen 在第 480 条后因账户 `Arrearage` 停止；失败记录未进入训练监督。
- 600 条 ASR 错字由历史测试中的 952 个已验证 typo pair 构建，去重后包含 325 个词组混淆和 215 个可替换汉字来源。
- 本地 ASR 生成器只改变表面文字，规范术语和证据条目继续来自本地结构化方证库。

训练监督 SHA-256：`8aa37ce98c4e5bb94c2dc7576d07a9df15f9c1f0b1e3cfdf3c484ccc68b0d419`

## GPU 流程

`scripts/run_query_translator_v3_gpu_pipeline.sh` 只在项目根目录和 `.venv-linux` 中运行，依次执行：

1. 数据与生成器测试。
2. V3 双塔训练，加载 1,080 条扩增监督。
3. 未见术语开发集评估。
4. 新冻结集基座评估。
5. 新冻结集 V3 评估。
6. 基座与 V3 固定 RRF 比较。

冻结集只在训练完成后进入评估命令，不参与训练或阈值调节。
