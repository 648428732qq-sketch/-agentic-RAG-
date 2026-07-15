# ChatMed 盲测 RAG 评测

脚本：`scripts/run_chatmed_gold_eval.py`

## 输入

- RAG可见问题：`tests/evals/gold/candidates/chatmed/questions/all_questions.jsonl`
- 私有答案键：`tests/evals/gold/candidates/chatmed/private/answer_keys/`

当前共有592条可测问题和592条答案键。其余8条在问题生成阶段失败，不进入本轮。

## 检查

不初始化RAG：

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py --dry-run
```

## 小批量运行

先运行5条：

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py --limit 5
```

指定类别：

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py `
  --category fuzzy_colloquial `
  --limit 10
```

脚本使用项目当前配置的 `RAGSystem + ChatInterface`，因此LLM、Qdrant和环境变量与WebUI一致。

## 全量运行

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py
```

每完成一题立即追加保存。中断后重复运行会跳过已有ID。

## Qwen语义裁判

运行RAG并同时判分：

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py `
  --judge `
  --prompt-judge-api-key
```

只为已有RAG回答补做判分：

```powershell
.\.venv\Scripts\python.exe scripts\run_chatmed_gold_eval.py `
  --judge-only `
  --prompt-judge-api-key
```

裁判分别处理三种期望行为：回答、追问无汗、信息不足或危险信号转诊。裁判结果会缓存，可断点续跑。

## 输出

```text
tests/evals/gold/results/chatmed/
├── rag_answers.jsonl                 RAG原始回答和耗时
├── answers_by_category/              按类别拆分的RAG回答
├── manifest.json                     数量、错误和平均指标
└── private/
    ├── judge_cache.jsonl             Qwen裁判缓存
    └── gold_results.jsonl            问题、RAG回答、参考答案和判分
```

## 注意

- RAG只能读取公开问题文件，参考答案在回答完成后才按ID连接。
- 当前 `ChatInterface.chat(message, history)` 接收 `history` 参数，但实现中未使用该参数。脚本仍按WebUI入口原样传入，多轮类别因此会真实暴露当前系统的上下文缺陷。
- `lexical_bigram_f1`只用于辅助观察，不能替代语义判断。
- ChatMed参考答案标记为 `imported_unverified`，判分表示与数据集答案的一致性，不等于临床正确性认证。
