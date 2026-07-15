# 中医医院 Agentic RAG — 项目工程总纲

---

## 一、当前项目结构

```
agentic-rag-for-dummies/
│
├── project/                          ← 核心系统（LangGraph RAG）
│   ├── config.py                     ✅ 已改：中文Embedding、多LLM提供商、分块适配
│   ├── app.py                        ✅ 已改：启动信息中文化
│   ├── rag_agent/
│   │   ├── graph.py                  🔒 不改：LangGraph 图结构
│   │   ├── graph_state.py            🔒 不改：共享状态定义
│   │   ├── nodes.py                  🔒 不改：节点实现
│   │   ├── edges.py                  🔒 不改：条件路由
│   │   ├── tools.py                  🔒 不改：检索工具（search_child_chunks / retrieve_parent_chunks）
│   │   ├── schemas.py                🔒 不改：QueryAnalysis 数据模型
│   │   └── prompts.py                ✅ 已改：全部中文化 + 中医问诊领域适配
│   ├── core/
│   │   ├── rag_system.py             ✅ 已改：多LLM提供商工厂函数
│   │   ├── chat_interface.py         ✅ 已改：系统节点标题中文化
│   │   ├── document_manager.py       🔒 不改：文档摄入管道
│   │   └── observability.py          🔒 不改：Langfuse 可观测
│   ├── db/
│   │   ├── vector_db_manager.py      🔒 不改：Qdrant 管理
│   │   └── parent_store_manager.py   🔒 不改：Parent store
│   ├── ui/
│   │   ├── gradio_app.py             ✅ 已改：中医医院品牌界面
│   │   └── css.py                    ✅ 已改：中国红+金色 主题
│   ├── document_chunker.py           🔒 不改：分块逻辑
│   ├── utils.py                      🔒 不改：PDF→Markdown + token估算
│   ├── .env.example                  ✅ 已改：多LLM密钥模板
│   └── assets/                       (聊天头像)
│
├── scripts/                          ← 数据预处理工具
│   ├── crawl_qihuang.py              ✅ 新建：岐黄古籍爬虫（1200+本）
│   ├── convert_datasets.py           ✅ 新建：通用 JSON/CSV/HF → Markdown 转换
│   ├── convert_dialogue_csv.py       ✅ 新建：医患对话 CSV → Markdown（仅内/妇/儿）
│   ├── check_cmtmedqa.py             ✅ 新建：检查 CMtMedQA 数据集内容
│   └── check_tcm_dialogs.py          ✅ 新建：筛查 CMtMedQA 中医科内容
│
├── notebooks/                        ← 学习用（不改）
├── requirements.txt                  ✅ 已加：可选的 langchain-openai 等注释
├── Chinese-medical-dialogue-data/    ✗ 待删除：西医数据，无用
├── Huatuo-26M/                       ✗ 待删除：西医QA，无用
├── Zhongjing/                        ⚠️ 仅用 HuggingFace 的 CMtMedQA 中医科
│   └── data/                         (test files only — 不起作用)
└── markdown_docs/                    ← 运行时自动生成
```

> 🔒 = LangGraph 骨骼，不能动 | ✅ = 已改造 | ✗ = 删除

---

## 二、LangGraph Agent 工作流（核心骨架）

```
START
  │
  ▼
summarize_history        ← 对话摘要（提取上下文）
  │
  ▼
rewrite_query            ← 查询改写 + 意图澄清
  │
  ├── 不清楚 ──→ request_clarification → 等用户补充 → 回到 rewrite_query
  │
  └── 清楚 ──→ 多 Agent 并行（Send API）
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   agent[0]     agent[1]     agent[n]      ← 子图实例（共享同一 Qdrant 集合）
    │              │             │
    ├── orchestrator     ← LLM调用工具（search/retrieve）
    ├── tools            ← 执行检索
    ├── should_compress  ← 判断是否压缩上下文
    ├── compress_context ← LLM 压缩
    ├── fallback_response← 检索超限时兜底
    └── collect_answer   ← 提取最终答案
       │              │             │
       └──────────────┼─────────────┘
                      ▼
              aggregate_answers    ← LLM 合并多答案
                      │
                      ▼
                     END
```

---

## 三、数据方案（三个来源，一个集合）

```
数据来源                        转换脚本                     Qdrant 集合
─────────────────────────────────────────────────────────────
1. 岐黄古籍 (qihuang.vip)  →  crawl_qihuang.py     ──┐
2. ShenNong-TCM QA (GitHub)→  convert_datasets.py   ──┤
3. CMtMedQA 中医科 (HF)    →  新写导出脚本          ──┤
                                                        ├──→  tcm_child_chunks
古籍 PDF (Anna's Archive)  →  pymupdf4llm 自动      ──┘    (统一集合)
```

| 层 | 内容 | 规模 | 用途 |
|----|------|------|------|
| 古籍全文 | 伤寒论、金匮要略、黄帝内经、本草纲目等 | ~500篇 Markdown | RAG 核心知识库 |
| 中药/方剂 QA | 性味归经、功效主治、方剂配伍、辨证论治 | ~21,000条 QA | 精确知识检索 |
| 中医问诊对话 | 真实医患多轮辨证对话 | 4,307条 | 问诊模式学习 |

---

## 四、运行时架构

```
┌────────────────────────────────────────────────┐
│             Gradio 界面 (localhost:7860)         │
│     📚 知识库管理  │  💬 智能问答                │
├────────────────────────────────────────────────┤
│                                                │
│   LangGraph Agentic RAG                        │
│   ┌──────────────────────────────────────┐    │
│   │  Qdrant 向量检索 (Hybrid: 稠密+稀疏)  │    │
│   │  Parent Store (JSON 文件)            │    │
│   │  InMemorySaver (对话记忆)            │    │
│   └──────────────────────────────────────┘    │
│                                                │
│   模型层                                       │
│   ├── Embedding: BAAI/bge-large-zh-v1.5       │
│   ├── LLM:      Ollama qwen3  / DeepSeek API  │
│   └── 稀疏检索: Qdrant/bm25                   │
│                                                │
│   可观测性 (可选)                               │
│   └── Langfuse 追踪                            │
└────────────────────────────────────────────────┘
```

---

## 五、短期行动计划

### 第一阶段：数据就绪（1-2天）

```
□ 1. 删除无用数据
     rm -rf Chinese-medical-dialogue-data  Huatuo-26M

□ 2. 爬取岐黄古籍
     pip install requests beautifulsoup4
     python scripts/crawl_qihuang.py

□ 3. 下载 ShenNong-TCM
     git clone https://github.com/michael-wzhu/ShenNong-TCM-LLM.git
     python scripts/convert_datasets.py json ./ShenNong-TCM-LLM/data/xxx.json

□ 4. 导出 CMtMedQA 中医科
     python -c "
     from datasets import load_dataset
     ds = load_dataset('Suprit/CMtMedQA', split='train')
     tcm = [d for d in ds if d['cate1']=='中医科']
     # 导出为 markdown_docs/cmtmedqa_tcm.md
     "

□ 5. PDF 古籍补充（可选）
     Anna's Archive 搜"伤寒论""金匮要略" 下载 PDF
```

### 第二阶段：启动验证（半天）

```
□ 6. 安装 Ollama + 拉模型
     ollama pull qwen3:4b-instruct-2507-q4_K_M
     
     首次运行会自动下载 bge-large-zh-v1.5 (~1.3GB)

□ 7. 启动系统
     python project/app.py

□ 8. 测试
     → 知识库管理：上传 markdown_docs/ 文件
     → 智能问答：问"麻黄汤的组成和功效？""阴虚火旺有什么症状？"
     → 验证回答是否引用原文+辨证逻辑
```

### 第三阶段：优化调优（按需）

```
□ 9. 分块调优
     根据检索效果，在 config.py 调整 CHILD_CHUNK_SIZE / OVERLAP

□ 10. LLM 切换
     如果 Qwen3 4B 效果不够，config.py 改 ACTIVE_LLM_CONFIG="deepseek"

□ 11. 对话记忆持久化
     将 InMemorySaver 改为 SqliteSaver（graph.py 改2行）

□ 12. 新增 Agent 工具（可选）
     tools.py 新增 search_formula_db（查询结构化方剂数据）
```

---

## 六、关键决策速查表

| 问题 | 答案 |
|------|------|
| 需要训练/微调吗？ | 不需要，纯 RAG |
| 需要显卡吗？ | 不需要，CPU 可跑 |
| 数据集几个来源？ | 3个：古籍全文 + 中药方剂QA + 中医问诊对话 |
| Agent 分数据集吗？ | 不分，一个 Qdrant 集合，Agent 只做后处理 |
| Embedding 用什么？ | BAAI/bge-large-zh-v1.5（中文优化） |
| LLM 用什么？ | 默认 Ollama Qwen3 4B，也可切 DeepSeek API |
| 向量库是什么？ | 本地 Qdrant（Hybrid Search） |
| 核心代码能改吗？ | graph.py / nodes.py / edges.py / tools.py / schemas.py / graph_state.py 不改 |
