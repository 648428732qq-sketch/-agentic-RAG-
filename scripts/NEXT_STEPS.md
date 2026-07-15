# 下一步执行指导 — Agentic RAG 中医系统

> 依据: `project_skeleton.md` 框架 + `CLAUDE.md` 编码原则  
> 前提: 数据预处理完成，7本古籍+药材+方剂+对话已分块，待灌入 Qdrant

---

## 一、数据预处理 — 验收确认

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 古籍完整性 | 7/8 OK | 医宗金鉴因 qihuang 截断暂跳过 |
| 噪声清理 | 零残留 | 登录/截断/音频/HTML 全清 |
| 古文转义 | 186 处标注 | 脉浮→（脉象轻取即得，多主表证）等 |
| 文本格式 | 已修复 | jicheng 碎片换行已合并为连贯段落 |
| 分块验证 | 10,384 父块 / 61,919 子块 | 零失败 |
| 来源追溯 | OK | source_url + chapter + doc_name 已注入 metadata |
| 数据集目录 | `datasets/unified/` | classics/herbs/formulas/dialogues |

---

## 二、灌入 Qdrant — 执行步骤

### 2.1 前置依赖

```powershell
# 确认 Ollama 已安装并拉取模型
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 确认 Embedding 模型会自动下载 (BAAI/bge-large-zh-v1.5, ~1.3GB)
```

### 2.2 灌入命令

```powershell
cd D:\BaiduNetdiskDownload\agentic-rag-for-dummies

# 方式一: 通过 Gradio 界面逐文件上传 (适合调试)
python project/app.py

# 方式二: 通过 document_manager 批量灌入 (适合量产)
python -c "
from project.core.rag_system import RAGSystem
from project.core.document_manager import DocumentManager
import config

rag = RAGSystem()
rag.initialize()
dm = DocumentManager(rag)
dm.add_documents(config.MARKDOWN_DIR)
"
```

> **注意**: `config.MARKDOWN_DIR` 当前指向 `markdown_docs/`。需要改为 `datasets/unified/` 下各子目录，或先将文件拷贝到 `markdown_docs/`。

### 2.3 验证灌入结果

```powershell
python -c "
from project.db.vector_db_manager import VectorDbManager
vdb = VectorDbManager()
print(vdb.get_collection('tcm_child_chunks').count())
"
```

---

## 三、API Key 方式测试 LLM

### 3.1 配置修改 ([config.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/project/config.py#L50))

```python
# 第 50 行, 从 Ollama 切到 API:
ACTIVE_LLM_CONFIG = "deepseek"  # 或 "openai" / "anthropic" / "google"
```

### 3.2 设置 API Key

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxxx"
# 或
$env:OPENAI_API_KEY = "sk-xxxx"
```

### 3.3 测试查询

启动后问以下问题验证 RAG 流水线:

```
1. "麻黄汤的组成和功效？"          # 验证方剂检索
2. "太阳病的主要症状是什么？"       # 验证古籍检索 + 古文转义
3. "高血压患者能吃党参吗？"         # 验证对话数据检索
4. "伤寒论中的桂枝汤和金匮要略中的有何不同？"  # 跨文档检索
```

---

## 四、Agentic RAG 工作流 — 中医问诊作用

```
用户: "麻黄汤怎么用？有什么禁忌？"
 │
 ├─ summarize_history    → 提取之前聊过的症状/方剂上下文
 ├─ rewrite_query        → "麻黄汤" → 自包含查询: "麻黄汤组成功效主治禁忌"
 │
 ├─ Multi-Agent 并行检索
 │   ├─ Agent[0] search_child_chunks    → 找到方剂条目子块
 │   ├─ Agent[1] retrieve_parent_chunks → 取回完整方剂上下文  
 │   └─ Agent[2] search_child_chunks    → 搜索"麻黄汤禁忌"片段
 │
 ├─ compress_context     → 结果过多时 LLM 压缩去重
 ├─ fallback_response    → 检索超限时兜底回答
 │
 └─ aggregate_answers    → 合并:
     "麻黄汤组成: 麻黄三两(约45g)、桂枝二两... 
      功用: 发汗解表，宣肺平喘。
      禁忌: 表虚自汗者禁用。参考: 伤寒论.md, 方剂大全.md"
```

**6 个 Prompt 各司其职** ([prompts.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/project/rag_agent/prompts.py)):

| Prompt | 作用 | 中医特殊规则 |
|--------|------|------------|
| summary | 对话摘要 ≤50字 | 保留方剂名/药材名 |
| rewrite | 查询改写+消歧 | 中医专有名词不替换, 症状模糊时标记 unclear |
| orchestrator | Agent 核心指令 | 每个论断必须有文献依据, 古文理解60+术语映射 |
| compress | 上下文压缩 | 按文件组织, 标注信息缺口 |
| fallback | 检索超限兜底 | 仅用已有信息, 急重症提示就医 |
| aggregate | 答案合并 | 自然流畅, 矛盾处如实陈述 |

---

## 五、优化建议 (根据效果逐步实施)

### 5.1 高优先级 — 工具扩展

当前 Agent 只有 2 个工具 (search/retrieve)。建议新增:

| 工具 | 函数签名 | 用途 |
|------|---------|------|
| `search_herb_by_name` | `(name: str) → str` | 精确查药材性味归经 |
| `search_formula_by_name` | `(name: str) → str` | 精确查方剂组成/功效 |
| `convert_classical_dosage` | `(text: str) → str` | 一两→15g 自动换算 |

**实现位置**: [tools.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/project/rag_agent/tools.py) — `ToolFactory.create_tools()`

### 5.2 中优先级

- **分块大小调优**: 当前 `CHILD_CHUNK_SIZE=400`, 中文1字≈1.5 token, 如检索精度不足可调到 500-600
- **相似度阈值**: `score_threshold=0.7` 对中医古文可能偏高
- **记忆持久化**: `InMemorySaver` → `SqliteSaver` (graph.py 改 2 行)

### 5.3 低优先级

- 古籍分块优化: 给长段落加 `##` 断点, 减少 3500 字巨块
- 新增 `check_herb_interaction` 工具: 检查十八反十九畏配伍禁忌

---

## 六、执行顺序

```
1. 安装模型                        ← 一次性
   ollama pull qwen3:4b-instruct-2507-q4_K_M

2. 灌入数据                        ← 需将 datasets/unified/ 下的 .md 放入 markdown_docs/
   python project/app.py

3. 启动 + API Key 测试             ← 设置环境变量
   python project/app.py

4. 验证 4 条测试查询               ← 确认 pipeline 通畅

5. 根据效果调整分块参数/阈值       ← 按需

6. 扩展 Agent 工具                 ← 提高检索精度
```

---

## 七、风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| config.MARKDOWN_DIR 指向旧路径 | 灌入空数据 | 第2.2步确认路径 |
| qwen3:4b 显存不足 | 推理慢/失败 | 切 DeepSeek API |
| 古籍分块过大 (3500字) | 上下文超 token 限制 | 配合 compress_context |
| jicheng 转简体丢失字形 | 罕见字错误 | 人工抽查前20条检索结果 |
