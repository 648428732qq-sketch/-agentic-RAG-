# ChatMed TCM 盲测问题工作流

数据源：`ChatMed_TCM-v0.2.json`

该流程分成两个互相独立的脚本：

1. `scripts/prepare_chatmed_gold_seeds.py`：去重、脱敏、分类并保存600条种子。
2. `scripts/generate_chatmed_blind_questions.py`：使用Qwen按不同类型的专用提示词生成盲测问题。

两个脚本都不初始化RAG、不运行检索、不判断RAG答案。

## 第一步：分层种子

```powershell
.\.venv\Scripts\python.exe scripts\prepare_chatmed_gold_seeds.py
```

种子保存在：

```text
tests/evals/gold/seeds/chatmed/private/
├── single_symptom.jsonl             80
├── multi_symptom.jsonl             140
├── fuzzy_colloquial.jsonl           80
├── negation_contradiction.jsonl     60
├── typo_asr.jsonl                   40
├── formula_herb.jsonl               70
├── classics_theory.jsonl            50
├── multi_turn.jsonl                 40
└── insufficient_safety.jsonl        40
```

共600条，来源问题互不重复。种子文件包含参考答案，RAG不得读取该目录。

## 第二步：检查生成计划

```powershell
.\.venv\Scripts\python.exe scripts\generate_chatmed_blind_questions.py --dry-run
```

小批量检查某个类别：

```powershell
.\.venv\Scripts\python.exe scripts\generate_chatmed_blind_questions.py `
  --dry-run `
  --category fuzzy_colloquial `
  --limit-per-category 5
```

## 第三步：生成盲测问题

直接安全输入密钥并运行，输入内容不会显示，也不会写入文件或PowerShell历史：

```powershell
.\.venv\Scripts\python.exe scripts\generate_chatmed_blind_questions.py --prompt-api-key
```

也可以预先通过环境变量 `DASHSCOPE_API_KEY` 提供密钥，此时不需要 `--prompt-api-key`。

建议先生成一个类别：

```powershell
.\.venv\Scripts\python.exe scripts\generate_chatmed_blind_questions.py `
  --category fuzzy_colloquial
```

相同种子、风格和样本序号生成稳定ID，重复运行自动跳过已完成ID。

## 独立提示词

生成脚本内置12套不同提示词，而不是一个模板替换风格名称：

- 单一症状口语化
- 多症状完整保留
- 方言和极度口语
- 一般模糊表达
- 头痛隐喻
- 否定反问
- 输入法错别字
- 方剂本草知识改写
- 古籍针法理论改写
- 多轮上下文拆分
- 故意漏掉无汗
- 信息不足和危险信号

每套提示词分别规定必须保留内容、禁止项、`history`格式和期望系统行为。

## 生成结果隔离

```text
tests/evals/gold/candidates/chatmed/
├── questions/
│   ├── all_questions.jsonl          只含RAG可见输入
│   └── <category>.jsonl             九类独立测试文件
└── private/
    ├── chatmed_blind_master.jsonl   完整生成审计
    ├── chatmed_blind_failures.jsonl 失败记录
    └── answer_keys/
        └── <category>.jsonl         隐藏参考答案
```

RAG测试阶段只能读取 `questions/`。答案比较程序通过相同 `id` 连接 `private/answer_keys/`。

## 质量边界

- 原始112,565条记录先执行模板清理、近重复去重和个人信息过滤。
- 科研论文、文章生成、联系方式和明显模板污染不会进入种子。
- 600条种子已经满足数量和分类，但尚未人工审核。
- ChatMed原始 `response` 标记为 `imported_unverified`，不是已经确认的临床金答案。
- 自动生成问题标记为 `auto_generated_unreviewed`，审核通过后才能进入锁定测试集。

生成完成后的RAG执行和答案对照见 `docs/CHATMED_GOLD_RAG_EVAL.md`。
