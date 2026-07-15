# Query Translator 数据与训练执行大纲

> 状态：待用户审核。审核通过前不执行训练、不修改线上检索规则、不解封 V2 逐题调参。

## 一、目标与验收口径

- [ ] 明确“问题识别率 95%”定义为：未见问题集上的**症状组级 micro recall >= 95%**。
- [ ] 否定/不确定极性识别准确率 >= 98%。
- [ ] 无本地证据时拒答或追问准确率 >= 95%。
- [ ] 禁止术语、方剂和病因凭空补充率 <= 1%。
- [ ] Top-5 语义正确干预召回率 >= 90%，再逐步提升至 95%。
- [ ] 分别报告普通口语、错别字、隐喻、否定、残句和多症状长句，不只报告总体平均值。
- [ ] “整题全部症状命中率”和“症状组级召回率”分开报告。

## 二、数据使用边界

### 2.1 可用于 Translator 开发或训练

- [ ] `datasets/external/raw/cblue_mirror_unverified/`
  - KUAKE-QQR：相似问句。
  - KUAKE-QTR：问句与标题相关性。
  - CHIP-STS：医疗句子语义相似。
  - 仅做语言增强；社区镜像许可证未完全核验，不随项目发布。
- [ ] `datasets/external/raw/huatuo26m_lite/`
  - 只使用 `question`、科室和疾病标签。
  - 回答不得作为中医方证、药物或剂量事实。
- [ ] `datasets/external/raw/chatmed_tcm/ChatMed_TCM-v0.2.json`
  - 只使用患者问题和对话表达。
  - 回答不得直接作为金标准。
- [ ] `datasets/structured/syndrome_dictionary.jsonl`
  - 作为唯一可选本地规范词、方证和证据来源之一。
  - 低置信度 payload 必须先经过证据一致性检查。

### 2.2 只用于验证，不参与训练

- [ ] `datasets/external/raw/mtcmb/`
  - 优先使用 TCMeEE、TCM-CHGD、TCM-MSDD、TCM-Diagnosis、TCM-PR、TCM-FRD。
- [ ] `datasets/external/raw/tcm_ladder/`
  - 仅使用 `fillInTheBlank.parquet` 和 `multiChoice.parquet`。
- [ ] `tests/evals/query_translator_v2_unseen_seed_pool_100/`
  - 已完成首次评分并冻结。
  - 不得根据单题失败增加词典特例或调阈值。

### 2.3 不需要上传或处理

- [ ] Common Voice 音频。
- [ ] TCM-Ladder 的 `herbImage.zip`、视频和 `visual.parquet`。
- [ ] 未明确许可证的原始网站抓取数据。
- [ ] 医疗问答中的姓名、电话、地址、身份证号等个人信息。

## 三、本地与服务器目录

### 3.1 本地项目根目录

```text
D:\BaiduNetdiskDownload\agentic-rag-for-dummies
```

### 3.2 服务器项目根目录

```text
/home/user/gptdata/cz/yangzhiting/agentic-rag-for-dummies/agentic-rag-for-dummies
```

### 3.3 服务器原始数据目录

```text
datasets/external/raw/
├── cblue_mirror_unverified/
├── huatuo26m_lite/
├── chatmed_tcm/
├── mtcmb/
└── tcm_ladder/
```

## 四、阶段 1：数据清点与传输

- [ ] 列出每个来源的文件名、大小、格式、记录数和字段。
- [ ] 记录来源 URL、许可证、下载日期和版本。
- [ ] 本地计算 SHA-256。
- [ ] 上传到服务器对应 `datasets/external/raw/` 子目录。
- [ ] 服务器重新计算 SHA-256，与本地核对。
- [ ] 不覆盖服务器已有文件；同名不同哈希进入版本化目录。
- [ ] 不上传 API Key、`.env`、模型密钥或代理池鉴权信息。

**阶段产物：**

```text
datasets/external/manifests/raw_dataset_manifest.json
```

## 五、阶段 2：原始数据隔离清洗

- [ ] 原始文件保持只读，不原地修改。
- [ ] 建立 `datasets/external/processed/`。
- [ ] 统一 UTF-8、换行符、JSON/JSONL/Parquet 字段格式。
- [ ] 清除 HTML、控制字符、网页导航和重复模板。
- [ ] 对问题文本做精确去重和近重复去重。
- [ ] 使用规则删除或掩码姓名、电话、地址、账号等隐私字段。
- [ ] 每条记录保留 `source_dataset/source_id/source_hash/license`。
- [ ] Huatuo 和 ChatMed 只抽取问题，不导入回答事实。
- [ ] 生成清洗统计和被丢弃原因。

**阶段产物：**

```text
datasets/external/processed/questions/
datasets/external/reports/data_cleaning_report.json
```

## 六、阶段 3：本地方证证据审计

- [ ] 对 `syndrome_dictionary.jsonl` 的每条规范词检查其是否出现在 evidence、原文或明确别名中。
- [ ] 识别错字、OCR 噪声、过宽单字词和无证据诊断键。
- [ ] 按 `verified/high/medium/quarantine` 分层。
- [ ] 置信度不足或证据冲突的条目退出监督标签池，但可保留在隔离报告中。
- [ ] 不使用 LLM 自己生成的解释反向证明自己正确。

**阶段产物：**

```text
datasets/structured/syndrome_dictionary_verified.jsonl
datasets/quarantine/syndrome_payload_rejected.jsonl
datasets/external/reports/payload_evidence_audit.json
```

## 七、阶段 4：构建“口语问句 → 本地规范词”监督对

- [ ] 从真实问题中抽取带字符位置的症状原话。
- [ ] 先用本地别名、BM25 和向量检索产生 Top-N 规范词候选。
- [ ] LLM 只能从候选白名单选择，不得创造术语。
- [ ] 每个映射保存 `source_phrase/start/end/canonical/polarity/confidence/evidence_entry_ids`。
- [ ] 无法映射的片段写入 `unknown_phrases`，不强行贴标签。
- [ ] 构造否定、矛盾、缺关键症状和易混方 hard negative。
- [ ] 不采集或训练地域方言；普通口语增强没有来源证据时不得标为人工金标准。
- [ ] 多模型一致只提高候选置信度，不替代本地 evidence。

**目标格式：**

```json
{
  "query": "患者原话",
  "mappings": [
    {
      "source_phrase": "肚子里翻江倒海",
      "source_start": 0,
      "source_end": 8,
      "canonical_term": "腹痛",
      "polarity": "present",
      "confidence": 0.91,
      "evidence_entry_ids": ["..."]
    }
  ],
  "unknown_phrases": [],
  "source_dataset": "..."
}
```

**阶段产物：**

```text
datasets/external/supervision/query_term_pairs.jsonl
datasets/external/supervision/hard_negatives.jsonl
datasets/external/supervision/rejected_candidates.jsonl
```

## 八、阶段 5：防泄漏划分

- [ ] 按方剂、条目、证型和来源分组切分，禁止同一概念跨 train/dev/test。
- [ ] 训练集只包含高置信监督对。
- [ ] 开发集用于 Translator、Gate 和阈值选择。
- [ ] MTCMB、TCM-Ladder 与 V2 不进入训练。
- [ ] 调参完成后，从从未使用的条目和问题来源生成 V3 最终盲测。
- [ ] 输出精确 ID 重合、症状签名近重复和 n-gram 泄漏报告。

## 九、阶段 6：Translator 技术路线

### 9.1 首选低风险路线

- [ ] 静态别名匹配。
- [ ] 本地规范词和可信口语短语建立向量索引。
- [ ] 口语查询检索 Top-N 规范词候选。
- [ ] DeepSeek/Qwen 只负责证据跨度、极性和候选裁决。
- [ ] 低置信度映射进入 unknown，不直接降低阈值放行。

### 9.2 是否训练小模型的决策门

- [ ] 高置信监督对少于 20,000 条：先不微调，使用向量候选 + LLM 裁决。
- [ ] 高置信监督对达到 20,000 条以上：评估微调中文 bi-encoder 或小型 cross-encoder。
- [ ] 训练模型只解决“原话与规范词匹配”，不学习诊断和处方生成。
- [ ] 训练前必须保留不训练的开发集和最终集。

## 十、阶段 7：Gate 去硬编码改造

- [ ] 删除仅针对少数方剂、寸口或固定来源类型的放行特例。
- [ ] Gate 改为读取 payload：required groups、forbidden terms、polarity、source type、evidence confidence。
- [ ] 信息完整且证据无冲突才允许结构化回答。
- [ ] 缺关键组、存在否定冲突或候选接近时必须追问。
- [ ] 理论、针灸、本草和古籍条文使用统一证据框架，不强制只能命中方剂来源。
- [ ] 增加“正确候选已检索但被 Gate 错拒”的独立指标。

## 十一、阶段 8：服务器隔离训练

- [ ] 全程只在项目目录操作。
- [ ] 使用项目内独立虚拟环境，不安装全局包。
- [ ] 不卸载或覆盖服务器 CUDA、驱动和他人环境。
- [ ] 选择一张空闲 GPU，通过 `CUDA_VISIBLE_DEVICES` 限定。
- [ ] 模型、缓存、日志和 checkpoint 全部写入项目目录。
- [ ] 训练前记录 GPU、CUDA、PyTorch、依赖锁文件和随机种子。
- [ ] 支持断点续训和最佳 checkpoint 回滚。

**服务器产物：**

```text
artifacts/query_translator/<run_id>/
├── config.json
├── metrics.json
├── checkpoints/
├── tokenizer/
└── training_log.jsonl
```

## 十二、阶段 9：测试先验与验收顺序

- [ ] 单元测试：字符跨度、否定、矛盾、白名单约束、UTF-8。
- [ ] 数据测试：许可证、哈希、去重、泄漏、隐私、证据一致性。
- [ ] 开发集评测：症状组召回、极性、unknown、Gate、Top-K。
- [ ] 来源分层评测：方剂、古籍、本草、针灸、理论。
- [ ] 风格分层评测：普通口语、错字、隐喻、否定、残句、多症状。
- [ ] 只有 Translator 和 Gate 达标后才进行 rerank A/B。
- [ ] rerank 只允许调整同证据层候选顺序，不覆盖证据和禁忌规则。
- [ ] 最终使用 V3 未见条目集一次性验收。

## 十三、阶段 10：报告与回滚

- [ ] 输出 baseline、改造后、rerank on/off 对比。
- [ ] 报告增益、退化、平均延迟、P95 延迟和显存。
- [ ] 所有失败按“数据、Translator、Gate、检索、排序”归因。
- [ ] 不以调整金标准掩盖系统错误。
- [ ] 保留旧索引和模型清单，验证成功后再切换活动版本。
- [ ] 任何阶段失败均可回滚到当前 V2 baseline。

## 十四、用户审核点

请重点确认：

1. 是否同意 CBLUE 社区镜像仅用于开发、不发布。
2. 是否同意 Huatuo/ChatMed 只取问题、不采用回答。
3. 是否同意 MTCMB、TCM-Ladder、V2 永不进入训练。
4. 是否接受先做“向量候选 + LLM 裁决”，数据达到 20,000 条高置信监督对后再决定微调。
5. 是否同意症状组级 recall >= 95% 作为主要识别率，整题全命中率单独报告。
