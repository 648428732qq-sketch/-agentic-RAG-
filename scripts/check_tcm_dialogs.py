"""检查 CMtMedQA 中医科内容"""
from datasets import load_dataset
import random

ds = load_dataset("Suprit/CMtMedQA", split="train")

# 筛选中医科数据
tcm = [item for item in ds if item["cate1"] == "中医科"]
print(f"中医科条目: {len(tcm)} / {len(ds)} 条 ({len(tcm)/len(ds)*100:.1f}%)")

# 随机抽5条看内容
indices = random.sample(range(len(tcm)), min(5, len(tcm)))
for i, idx in enumerate(indices):
    item = tcm[idx]
    hist = item["history"]
    print(f"\n{'='*60}")
    print(f"样本{i+1} | 子类: {item['cate2']} | ID: {item['id']}")
    print(f"患者首问: {hist[0][0][:200]}")
    print(f"医生首答: {hist[0][1][:200]}")
    if len(hist) > 1:
        print(f"  ...({len(hist)}轮对话)")
