"""修复 jicheng.tw 数据的碎片化换行问题 — 5本古籍"""
import re
from pathlib import Path

d = Path(__file__).resolve().parent.parent / "datasets" / "unified" / "classics"
# 只需要修复 jicheng.tw 来源的（gushiwen.cn 的伤寒论/金匮要略格式正常）
jicheng_files = ["黄帝内经_素问.md", "黄帝内经_灵枢.md", "神农本草经.md", "温病条辨.md", "难经.md"]

for fname in jicheng_files:
    f = d / fname
    if not f.exists():
        continue
    text = f.read_text(encoding="utf-8")
    lines = text.split("\n")
    merged = []
    for line in lines:
        stripped = line.strip()
        # 保留空行、标题行、元数据行
        if not stripped or stripped.startswith("#") or stripped.startswith(">"):
            merged.append(line)
        elif len(stripped) <= 5 and not any("\u4e00" <= c <= "\u9fff" for c in stripped):
            # 纯短标点/数字碎片 → 跳过
            continue
        else:
            merged.append(line)

    # 合并: 连续非标题/非空行拼接
    result = []
    buf = []
    for line in merged:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(">"):
            if buf:
                result.append("".join(b for b in buf if b.strip()))
                buf = []
            result.append(line)
        else:
            buf.append(stripped)
    if buf:
        result.append("".join(b for b in buf if b.strip()))

    new_text = "\n".join(result)
    # 压缩连续空行
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    f.write_text(new_text, encoding="utf-8")
    old_size = len(text)
    new_size = len(new_text)
    print(f"  [FIX] {fname}: {old_size:,} -> {new_size:,} chars")
print("Done")
