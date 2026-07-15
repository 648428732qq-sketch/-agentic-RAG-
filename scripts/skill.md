# 中医数据集切块流水线 (Skill)

> **SCOPE**: 仅限 `scripts/` 目录下的数据集预处理任务 (古文转义、格式转换、文档分块)  
> **CONFLICT**: 此 skill 仅当工作上下文在 `scripts/` 或数据集预处理/切块阶段时激活，不得在其他场景使用  
> **适用**: 中医古籍/药材/方剂/医患对话数据集 → 统一 Markdown → Parent-Child Chunking  
> **工具**: CodeGraph MCP (代码导航) + 本地脚本 (转换/分块)

---

## 流水线总览

```
原始数据集 (50+ 文件)
    │
    ▼
Step 1: 古文转义
    ├── 识别文本是否为古籍文言文
    ├── 对古籍文本执行 CLASSICAL_ANNOTATIONS 术语标注
    └── 不修改原文，仅追加现代汉语注释
    │
    ▼
Step 2: 格式分类
    ├── 类型A: 古籍经典 (纯文言，## 章节标题)
    ├── 类型B: 药材知识 (结构化字段，## 药名)
    ├── 类型C: 方剂知识 (结构化字段，## 方名)
    ├── 类型D: 医患问答 Q&A (## Q: + **答**:)
    ├── 类型E: 华佗问答 JSON (instruction/output 列表)
    ├── 类型F: CMtMedQA 多轮对话 JSON (含 history)
    └── 类型G: RLHF 偏好对 JSON (output 候选数组)
    │
    ▼
Step 3: 格式统一转换 (→ Markdown)
    ├── .json → convert_datasets.py / convert_dialogues.py
    ├── .csv  → convert_datasets.py / convert_dialogue_csv.py
    ├── .txt  → 直接重命名为 .md
    └── .md   → 跳过转换，直接进入分块
    │
    ▼
Step 4: 智能分块 (DocumentChuncker)
    ├── 有 ## 标题的结构化文档 → MarkdownHeaderTextSplitter
    ├── 古籍无 ## 标题 → RecursiveCharacterTextSplitter 按段落
    ├── 过大父块 → __split_large_parents (MAX_PARENT_SIZE=3500)
    ├── 过小父块 → __merge_small_parents (MIN_PARENT_SIZE=1500)
    └── 子块生成 → RecursiveCharacterTextSplitter (400字, 80重叠)
    │
    ▼
灌入 Qdrant 向量数据库
```

---

## 数据集清单 (50 个文件)

### 1. 古籍经典 — `datasets/tcm_knowledge/classics/`

| # | 文件路径 | 类型 | 格式 |
|---|---------|------|------|
| 1 | `datasets/tcm_knowledge/classics/伤寒论.md` | A-古籍 | Markdown |
| 2 | `datasets/tcm_knowledge/classics/金匮要略.md` | A-古籍 | Markdown |
| 3 | `datasets/tcm_knowledge/classics/神农本草经.md` | A-古籍 | Markdown |
| 4 | `datasets/tcm_knowledge/classics/黄帝内经_素问.md` | A-古籍 | Markdown |
| 5 | `datasets/tcm_knowledge/classics/黄帝内经_灵枢.md` | A-古籍 | Markdown |
| 6 | `datasets/tcm_knowledge/classics/难经.md` | A-古籍 | Markdown |
| 7 | `datasets/tcm_knowledge/classics/温病条辨.md` | A-古籍 | Markdown |
| 8 | `datasets/tcm_knowledge/classics/医宗金鉴.md` | A-古籍 | Markdown |

### 2. 药材 & 方剂 — `datasets/tcm_knowledge/`

| # | 文件路径 | 类型 | 条数 |
|---|---------|------|------|
| 9 | `datasets/tcm_knowledge/herbs/中药百科.md` | B-药材 | 474味 |
| 10 | `datasets/tcm_knowledge/formulas/方剂大全.md` | C-方剂 | 180首 |

### 3. 医患问答对话 — `datasets/dialogues/medical_qa/`

| # | 文件路径 | 类型 | 条数 |
|---|---------|------|------|
| 11 | `datasets/dialogues/medical_qa/内科5000-33000.md` | D-问答 | ~5000 |
| 12 | `datasets/dialogues/medical_qa/外科5-14000.md` | D-问答 | ~5000 |
| 13 | `datasets/dialogues/medical_qa/儿科5-14000.md` | D-问答 | ~5000 |
| 14 | `datasets/dialogues/medical_qa/妇产科6-28000.md` | D-问答 | ~6000 |
| 15 | `datasets/dialogues/medical_qa/男科5-13000.md` | D-问答 | ~5000 |
| 16 | `datasets/dialogues/medical_qa/肿瘤科5-10000.md` | D-问答 | ~5000 |
| 17 | `datasets/dialogues/medical_qa/样例_内科5000-6000.md` | D-问答 | 样例 |

### 4. 仲景数据集 — `datasets/dialogues/zhongjing/`

| # | 文件路径 | 类型 | 条数 |
|---|---------|------|------|
| 18 | `datasets/dialogues/zhongjing/CMtMedQA.md` | D-问答 | 4307 |
| 19 | `datasets/dialogues/zhongjing/huatuo26M_test.md` | D-问答 | 6000 |

### 5. 原始 CSV 源文件 — `Chinese-medical-dialogue-data/`

| # | 文件路径 | 原始格式 | 需转换 |
|---|---------|---------|--------|
| 20 | `Chinese-medical-dialogue-data/样例_内科5000-6000.csv` | CSV | → .md |
| 21 | `Chinese-medical-dialogue-data/Data_数据/IM_内科/内科5000-33000.csv` | CSV | → .md |
| 22 | `Chinese-medical-dialogue-data/Data_数据/IM_内科/内科.txt` | TXT(乱码) | → .md |
| 23 | `Chinese-medical-dialogue-data/Data_数据/Surgical_外科/外科5-14000.csv` | CSV | → .md |
| 24 | `Chinese-medical-dialogue-data/Data_数据/Pediatric_儿科/儿科5-14000.csv` | CSV | → .md |
| 25 | `Chinese-medical-dialogue-data/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv` | CSV | → .md |
| 26 | `Chinese-medical-dialogue-data/Data_数据/OAGD_妇产科/妇产科6-28000.csv` | CSV | → .md |
| 27 | `Chinese-medical-dialogue-data/Data_数据/Andriatria_男科/男科5-13000.csv` | CSV | → .md |

### 6. 原始 JSON 源文件 — `Zhongjing/data/`

| # | 文件路径 | 原始格式 | 需转换 |
|---|---------|---------|--------|
| 28 | `Zhongjing/data/CMtMedQA_test.json` | F-多轮对话 | → .md |
| 29 | `Zhongjing/data/huatuo26M_test.json` | E-问答JSON | → .md |
| 30 | `Zhongjing/data/rlhf2000_7_31.json` | G-RLHF | → .md(可选) |

### 7. 注册表 & 配置

| # | 文件路径 | 说明 |
|---|---------|------|
| 31 | `datasets/registry.json` | 数据集注册表 |

### 8. 待扩展 (markdown_docs 目录中的额外文件)

`markdown_docs/` 为分块输出目录，含 qihuang.vip 1170本古籍的标题索引。其中已有部分章节正文的 .md 文件可作为补充数据集。

---

## Step 1: 古文转义

### 触发条件

仅对**古籍类型**（A-古籍）和包含文言文的**方剂**（C-方剂）文本执行。对话类数据（D/E/F/G）不执行。

### 转义脚本

脚本位置：[scripts/annotate_classical.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/scripts/annotate_classical.py)

```python
from scripts.annotate_classical import annotate_classical_text, CLASSICAL_ANNOTATIONS

# 标注字典含 60+ 术语，覆盖六经/症状/舌象/治法/计量/省略六大类
text = "太阳病，头痛发热，脉浮。"
annotated = annotate_classical_text(text)
# → "太阳病（即太阳经表证，病在体表，多属外感初起），头痛发热（发烧），脉浮（脉象轻取即得，多主表证）。"
```

### 关键原则

- **只追加不修改**：原文保留完整，现代注释以括号形式追加
- **首次出现标注**：每术语仅标注第一次出现，避免冗余
- **可扩展**：`CLASSICAL_ANNOTATIONS` 字典可随时扩充新术语

### 使用 CodeGraph 导航代码

在 IDE 中使用 CodeGraph MCP 可以快速定位转义逻辑的调用链：

```
# 查找 annotate_classical_text 的所有调用者
mcp_codegraph_codegraph_callers("annotate_classical_text")
```

```
# 探索相关符号之间的关系
mcp_codegraph_codegraph_explore("annotate_classical_text CLASSICAL_ANNOTATIONS document_chunker")
```

---

## Step 2: 格式分类

### 分类决策树

```
输入文件是否为 .md?
  ├── 是 → 检查 H1 标题
  │   ├── "方剂大全" → 类型C (方剂)
  │   ├── "中药百科" → 类型B (药材)
  │   ├── 含 "## Q:" 或 "## Q{n}:" → 类型D (问答)
  │   └── 其他 → 类型A (古籍)
  └── 否 → 检查扩展名
      ├── .csv → 医患对话 CSV (待转换)
      ├── .json → 进一步检查结构
      │   ├── 含 "history" 字段 → 类型F (多轮对话)
      │   ├── output 为数组 → 类型E 或 G
      │   └── 含 "department"/"cate1" → 类型F (CMtMedQA)
      └── .txt → 直接重命名
```

### 七种格式对照

| 类型 | 格式 | Markdown结构 | 示例文件 | 条数 |
|------|------|-------------|---------|------|
| A | 古籍 | `# 书名` `## 章节` 正文 | `伤寒论.md` | 8本 |
| B | 药材 | `# 中药百科` `## 麻黄` 结构化字段 | `中药百科.md` | 474 |
| C | 方剂 | `# 方剂大全` `## 麻黄汤` 结构化字段 | `方剂大全.md` | 180 |
| D | 医患问答 | `## Q:` `**答**:` `---` | `内科5000-33000.md` | ~3万 |
| E | 华佗JSON | `instruction`/`output` 数组 | `huatuo26M_test.json` | 6000 |
| F | 多轮对话JSON | `history` 多轮 + `cate1` 科室 | `CMtMedQA_test.json` | 4307 |
| G | RLHF偏好对 | `output` 候选数组(4个) | `rlhf2000_7_31.json` | 2000 |

---

## Step 3: 格式统一转换 (→ Markdown)

### 转换脚本矩阵

| 源格式 | 转换脚本 | 命令 | 目标目录 |
|--------|---------|------|---------|
| JSON → MD | `scripts/convert_dialogues.py` | `python scripts/convert_dialogues.py` | `datasets/dialogues/` |
| CSV → MD | `scripts/convert_dialogue_csv.py` | `python scripts/convert_dialogue_csv.py <data_dir>` | `markdown_docs/` |
| JSON → MD | `scripts/convert_datasets.py` | `python scripts/convert_datasets.py json <path>` | `markdown_docs/` |
| CSV → MD | `scripts/convert_datasets.py` | `python scripts/convert_datasets.py csv <path>` | `markdown_docs/` |
| HF → MD | `scripts/convert_datasets.py` | `python scripts/convert_datasets.py hf <name>` | `markdown_docs/` |
| TXT → MD | `scripts/convert_datasets.py` | `python scripts/convert_datasets.py txt <dir>` | `markdown_docs/` |

### 统一输出格式规范

所有格式最终转换为统一 Markdown 结构：

**古籍/药材/方剂**：
```markdown
# 书名/分类名

> 来源: <URL>

## 章节名/药名/方名

正文内容...
```

**对话/问答**：
```markdown
# 数据集名

> 共 N 条

## Q: 患者问题

**答**: 医生回答...

---

## Q: 下一个问题...
```

### 原则：保留原始表达

- **古籍**：原文不修改，仅追加以括号包裹的现代注释
- **药材/方剂**：结构化字段保持原样（`药性`、`功效`、`组成`、`用法` 等）
- **对话**：保留 `**患者**：` / `**医生**：` 角色标记
- **多轮对话**：`history` 全部展开，保留对话上下文

---

## Step 4: 智能分块 (Chunking)

### 分块器架构

核心类：[project/document_chunker.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/project/document_chunker.py) — `DocumentChuncker`

```
DocumentChuncker
├── __parent_splitter   → MarkdownHeaderTextSplitter (按 H1/H2/H3 切分)
├── __child_splitter    → RecursiveCharacterTextSplitter (400字/80重叠)
├── __min_parent_size   = 1500 (过小合并)
├── __max_parent_size   = 3500 (过大再切)
│
└── create_chunks_single(md_path)
    ├── 1. MarkdownHeaderTextSplitter.split_text()      # H1/H2/H3 标题切分
    ├── 2. __merge_small_parents()                       # 合并 < 1500 字的碎片
    ├── 3. __split_large_parents()                       # 切分 > 3500 字的大块
    ├── 4. __clean_small_chunks()                        # 二次清理残余小片
    └── 5. __create_child_chunks()                       # 生成 400 字子块
```

### 配置参数 ([config.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/project/config.py#L59-L69))

| 参数 | 值 | 说明 |
|------|-----|------|
| `CHILD_CHUNK_SIZE` | 400 | 子块大小 (检索用) |
| `CHILD_CHUNK_OVERLAP` | 80 | 子块重叠 (避免截断) |
| `MIN_PARENT_SIZE` | 1500 | 最小父块(小于此值合并) |
| `MAX_PARENT_SIZE` | 3500 | 最大父块(大于此值切分) |
| `HEADERS_TO_SPLIT_ON` | H1/H2/H3 | 标题层级分块 |

### 各类型分块策略

#### 类型A (古籍) — 特殊处理

古籍通常**只有 H1 标题**，没有 H2/H3 子标题。MarkdownHeaderTextSplitter 只能切出一个巨块。

**策略**：`__split_large_parents()` 自动用 RecursiveCharacterTextSplitter 按 3500 字切分。

**已知风险** (来自 [test_chunk.py](file:///D:/BaiduNetdiskDownload/agentic-rag-for-dummies/scripts/test_chunk.py#L51-L65))：
```
- 古籍会被截断在句中，语义不完整
- 建议: 用 RecursiveCharacterTextSplitter 按段落/句号分块
        或先通过 LLM 给古籍加 ## 标题再分块
```

**优化方案** (实施中)：
1. 执行 `annotate_classical_text()` 转义后再分块
2. 用 `separators=["\n\n", "\n", "。", "；", "，"]` 优先在句号处断句
3. 可通过 CodeGraph explore 导航到分块代码快速定位修改点

#### 类型B/C (药材/方剂) — 直接分块

每个 `## 药材名` / `## 方剂名` 正好是一个独立知识单元，直接按标题分块即可。结构完美，无需额外处理。

#### 类型D/E/F (对话/问答) — 按条分块

每条 `## Q: ... **答**: ... ---` 区间已是通过 `---` 分隔的独立单元。

**策略**：转换为 Markdown 后，`##` 标题被 splitter 识别，每条 Q&A 为一个父块。适合问答检索场景。

#### 类型G (RLHF) — 可选转换

RLHF 偏好对（output 含 4 个候选回复）主要用于评估，非标准知识检索。可选择：
- 不灌入 Qdrant（仅用于评估）
- 或取最佳回答转为 D 类格式灌入

### 使用 CodeGraph 导航分块代码

```bash
# 查看 document_chunker 的所有依赖关系
# 在 Trae 中使用: mcp_codegraph_codegraph_explore("document_chunker")

# 查看 create_chunks_single 被谁调用
# mcp_codegraph_codegraph_callers("create_chunks_single")

# 查看 document_manager 如何使用 chunker
# mcp_codegraph_codegraph_explore("document_manager add_documents chunker create_chunks")
```

---

## 完整执行流程

### 一次性执行全部流水线

```
1. 古文转义 (仅古籍)
   └── 为所有 classics/*.md 执行 annotate_classical_text()

2. 格式转换
   ├── python scripts/convert_dialogues.py          # JSON → datasets/dialogues/
   ├── python scripts/convert_dialogue_csv.py <dir>  # CSV → markdown_docs/
   └── python scripts/convert_datasets.py json <path> # 额外JSON

3. 收集所有 .md 文件到 markdown_docs/
   └── 确认 config.MARKDOWN_DIR 路径正确

4. 执行分块
   └── python scripts/test_chunk.py                 # 先测试
   └── python project/app.py                        # 全量分块+灌入

5. 验证
   └── 检查 Qdrant collection: tcm_child_chunks
```

### 测试文档标题结构化 (CodeGraph MCP)

使用 `mcp_codegraph_codegraph_files` 检查项目结构：
```
# 按路径查看 datasets 目录树
mcp_codegraph_codegraph_files(path="datasets", format="tree")
```

---

## 依赖关系图

```
config.py (配置中心)
    │
    ├──→ document_chunker.py (分块器)
    │       ├── MarkdownHeaderTextSplitter (标题分块)
    │       └── RecursiveCharacterTextSplitter (递归分块)
    │
    ├──→ rag_system.py (RAG系统)
    │       ├── VectorDbManager (Qdrant)
    │       ├── ParentStoreManager (父块存储)
    │       └── DocumentChuncker (分块)
    │
    └──→ document_manager.py (文档管理器)
            └── RAGSystem.add_documents()

convert_datasets.py ──→ config.MARKDOWN_DIR
convert_dialogues.py ──→ datasets/dialogues/
convert_dialogue_csv.py ──→ config.MARKDOWN_DIR
annotate_classical.py (独立工具, 可选在分块前调用)
```

---

## 注意事项

1. **古籍分块精度**：当前 `MAX_PARENT_SIZE=3500` 可能在句中截断。如果检索质量不达预期，优先调整古籍的 `separators` 为 `["\n\n", "。", "\n"]`
2. **CSV 编码问题**：`convert_dialogues.py` 已处理多编码，gkb/gb2312 自动回退
3. **转换产物不覆盖**：所有转换脚本输出到 `datasets/dialogues/` 或 `markdown_docs/`，不会覆盖原始源文件
4. **CodeGraph MCP 集成**：本项目的 `.codegraph/` 索引使得在修改分块逻辑时可以用 `codegraph_impact("create_chunks_single")` 分析影响范围
