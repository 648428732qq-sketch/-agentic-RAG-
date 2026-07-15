"""检查 CMtMedQA 数据集是否包含中医问诊内容"""
from datasets import load_dataset

ds = load_dataset("Suprit/CMtMedQA", split="train")
print(f"总条数: {len(ds)}")
print(f"字段: {list(ds[0].keys())}")

# 统计科室分布
from collections import Counter
cates = Counter(ds["cate1"])
print(f"\n科室分布:")
for k, v in cates.most_common():
    print(f"  {k}: {v}")

# 随机抽5条看实际内容
import random
indices = random.sample(range(len(ds)), min(5, len(ds)))
for i, idx in enumerate(indices):
    item = ds[idx]
    hist = item["history"]
    print(f"\n{'='*60}")
    print(f"样本{i+1} | 科室: {item['cate1']} | ID: {item['id']}")
    if hist:
        print(f"患者: {hist[0][0][:200]}")
        print(f"医生: {hist[0][1][:200]}")
    if len(hist) > 1:
        print(f"  ...(共{len(hist)}轮对话)")
