# Query Translator 数据流水线与 Rerank 前置实施说明

## 1. 当前结论

本轮已完成数据清单、隔离清洗、证据审计、监督对构建、防泄漏切分、通用 Gate 和一键执行器的代码实现。所有新增代码已在本地项目虚拟环境通过 171 个测试。

远端服务器已完成原始数据清单和首轮清洗。修正 CBLUE 配对保留逻辑后的第二次清洗因服务器 SSH/网络不可达尚未完成验收，因此当前不能把 rerank 设为默认开启。

## 2. 数据使用边界

| 数据源 | 用途 | 明确禁止 |
|---|---|---|
| CBLUE 社区镜像 | 中文医疗问句相似度、相关性语言监督 | 不作为中医方证事实；不随项目重新分发 |
| Huatuo26M-Lite | 只抽取患者问题、科室和疾病标签 | 不读取或导出回答作为中医事实 |
| MTCMB | 独立验证 | 不进入训练或调参 |
| TCM-Ladder 文本表 | 独立验证 | 不进入训练；不处理图片、视频和音频 |
| 本地 syndrome dictionary | 规范词、方证和证据候选 | 现代别名不能用模型生成内容自证正确 |
| V2 盲测集 | 冻结回归 | 不按单题失败添加词典特例 |

CBLUE 的 CHIP-STS 是二分类语义相似任务；KUAKE-QQR 和 KUAKE-QTR 是分级相关性任务。流水线只把人工标签用于语言相似度训练，不把它们解释为中医事实标签。

## 3. 新增实现

### 3.1 原始数据审计

- `scripts/audit_external_datasets.py`
- 输出文件级 SHA-256、格式、记录数、字段结构、许可与使用边界。
- 原始文件不原地修改，跳过 `.git` 内容。

### 3.2 清洗和隔离

- `scripts/prepare_external_query_data.py`
- 保留中文全角标点，统一 UTF-8 和换行。
- 清理 HTML、控制字符和 URL。
- 掩码电话、邮箱、身份证、账号、显式姓名和地址。
- 全局精确去重及 SimHash 候选加 Jaccard 复核的近重复去重。
- 验证集先于开发集去重，防止相同问题进入训练侧。
- CBLUE 配对独立去重，不再因单条问题重复而丢失配对监督。
- Huatuo 回答完全不进入输出 schema。
- MTCMB/TCM-Ladder 的答案只写入隔离的 `validation_labels/`。

### 3.3 独立数据校验

- `scripts/validate_external_processed_data.py`
- 检查 JSONL 可解析、记录 ID 唯一、来源哈希有效、跨分区无精确重复。
- 检查问题文件和 CBLUE 配对没有答案字段。
- 检查验证标签只属于验证数据且能够回指问题 ID。
- 扫描残留电话、邮箱和身份证格式。

### 3.4 Payload 证据审计

- `scripts/audit_syndrome_payload_evidence.py`
- 分层：`verified`、`high`、`medium`、`quarantine`。
- `verified` 只允许人工或专家审核状态。
- `high` 要求可靠方剂来源、完整必需症状组和足够诊断键证据。
- `medium` 只参与候选召回，不作为自动金标签。
- `quarantine` 排除证据缺失、必需/禁忌冲突、已知网页噪声和弱证据条目。
- 单字不是一刀切：`喘、咳、呕、渴、痛`等可独立成义；`脉、浮`等宽泛单字需上下文。

本地实际审计结果：

| 指标 | 数量 |
|---|---:|
| Payload 总数 | 1,419 |
| high | 140 |
| medium | 983 |
| quarantine | 296 |
| 有原文证据的术语 | 7,154 |
| 可作为高可信标签的术语记录 | 1,662 |

高可信标签只包含阳性症状、诊断键、必需症状和证候名。方剂名、药名及仅用于鉴别的术语即使出现在证据中，也不会作为 Query Translator 的正向金标签。

### 3.5 监督对和 Hard Negative

- `scripts/build_query_translator_supervision.py`
- 高置信监督只来自“真实问题中的精确字符跨度 + high 证据术语”。
- 保存 `source_start/source_end/polarity/evidence_entry_ids`。
- medium 命中和未知问题写入 `rejected_candidates.jsonl`，不强行贴标签。
- CBLUE 配对写入独立语言监督文件。
- 从所有同来源类型、共享证据术语但干预不同的本地条目构造 hard negative，不限定少数方剂。

### 3.6 防泄漏切分

- `scripts/split_query_translator_supervision.py`
- 规范术语与关联 evidence entry 建连通组，整组进入 train 或 dev。
- CBLUE 以重复问句连通组切分。
- 报告术语、entry 和文本重合；不通过随机逐行切分制造虚高成绩。
- MTCMB、TCM-Ladder 和 V2 不被此脚本读取。

### 3.7 通用 Gate 和本地硬排序

- `project/core/evidence_gate.py`
- Gate 读取 payload 的置信度、必需组、禁忌冲突、查询覆盖率和候选可区分性。
- 临床查询缺必需组、低证据置信度、候选不可区分时必须追问。
- 检索排序允许“覆盖高但缺信息”的候选进入 Top-K，供系统追问；Gate 不允许直接回答。
- 不再用少数固定方证签名决定是否放行。
- `SYNDROME_MIN_EVIDENCE_CONFIDENCE=0.60` 可通过环境变量调整。

### 3.8 一键执行器

- `scripts/run_query_translator_data_pipeline.py`
- 强制使用项目目录中的 `.venv` 或 `.venv-linux`。
- 使用项目内锁文件防止双实例。
- 固定执行顺序：清单、清洗、校验、证据审计、监督构建、防泄漏切分、测试。
- 不调用 DeepSeek/Qwen，不读取 API Key，不写项目目录外文件。

### 3.9 Rerank 前结构化基线

- `scripts/evaluate_local_hard_negative_ranking.py`
- 从全库证据角色构造似方证候选，不把方剂名、药名泄漏到 query。
- 每组候选分别生成带特异证据的排序案例，以及证据覆盖相近时的追问案例。
- 本地结构化 preflight：6,208 例、659 个干预项、6 类 source type；3,293 条排序案例与 2,915 条追问案例通过。
- 该结果只验证 payload 规则内部一致性，不作为独立盲测或泛化成绩。

服务器恢复后执行：

```bash
cd /home/user/gptdata/cz/yangzhiting/agentic-rag-for-dummies/agentic-rag-for-dummies
.venv-linux/bin/python scripts/run_query_translator_data_pipeline.py
```

## 4. 已得到的远端结果

### 原始清单

- 文件数：24
- CBLUE：6
- Huatuo26M-Lite：4
- MTCMB：12
- TCM-Ladder：2
- 解析错误：0

### 首轮清洗

- 原始记录：260,437
- 候选问题：315,611
- 保留问题：265,756
- 精确重复删除：47,994
- 近重复删除：101
- 过短删除：1,759
- 验证标签：24,039

首轮 CBLUE 配对只保留 15,240 条，已定位为配对写出依赖单条问题去重的实现缺陷。代码已修复并增加回归测试，但远端修正版尚待服务器恢复后重跑。

## 5. Rerank 启用门槛

以下条件全部满足前，`ENABLE_SYNDROME_RERANK` 保持 `false`：

1. 修正版清洗和独立校验通过。
2. 监督对及防泄漏报告生成，term/entry/text overlap 均为 0。
3. 通用 Gate 回归通过。
4. 广覆盖 local hard negative 的 Top-K 基线完成。
5. 同一批固定 dev hard negative 上完成 rerank off/on A/B。
6. Rerank 只能调整同证据层候选顺序，不能覆盖必需组、禁忌和证据置信度。
7. 报告 Top-1、MRR、NDCG、退化数量、平均/P95 延迟和显存。

若高置信 query-term 训练对少于 20,000 条，不训练 Query Translator 小模型，继续使用本地混合候选加受约束裁决。这个门槛由数据量决定，不为追求“用了训练”而降低。

## 6. 当前阻塞与恢复点

- 远端 `pyarrow 17.0.0` 已仅安装在项目 `.venv-linux`。
- 第二次全量清洗期间 SSH 连接失去响应；本机到服务器 8080 端口的后续探测也未返回。
- 未执行全局安装、未修改 CUDA、未操作其他目录或其他用户进程。
- 恢复后先检查是否存在 `prepare_external_query_data.py` 残留进程和项目锁，再运行一键执行器。
