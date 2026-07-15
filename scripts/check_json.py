"""检查 JSON 数据结构"""
import json
for name in ["CMtMedQA_test", "huatuo26M_test"]:
    path = f"Zhongjing/data/{name}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n{name}: type={type(data).__name__}, len={len(data)}")
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if isinstance(item, dict):
            print(f"  keys: {list(item.keys())}")
            for k, v in item.items():
                sv = str(v)[:100]
                print(f"  {k}: {sv}")
        else:
            print(f"  item type: {type(item).__name__}")
            print(f"  sample: {str(item)[:200]}")
    elif isinstance(data, dict):
        print(f"  keys: {list(data.keys())[:5]}")
