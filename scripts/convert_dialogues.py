"""
对话数据集转换 → datasets/dialogues/ 子目录
来源: Chinese-medical-dialogue (CSV) + Zhongjing (JSON)
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════
#  1. CMtMedQA — 中医问答测试集 (4307条)
# ═══════════════════════════════════════
def convert_cmtmedqa():
    src = ROOT / "Zhongjing" / "data" / "CMtMedQA_test.json"
    dst_dir = ROOT / "datasets" / "dialogues" / "zhongjing"
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["# CMtMedQA 中医问答测试集\n", f"> 共 {len(data)} 条\n"]
    for i, item in enumerate(data):
        q = item.get("instruction", item.get("question", ""))
        a = item.get("output", item.get("answer", item.get("response", "")))
        if q and a:
            lines.append(f"## Q{i+1}: {q}\n\n**答**: {a}\n\n---\n")

    out = dst_dir / "CMtMedQA.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] CMtMedQA: {len(data)} 条 → {out.name}")

# ═══════════════════════════════════════
#  2. Chinese-medical-dialogue — CSV 医患对话
# ═══════════════════════════════════════
def convert_csv_dialogs():
    src_dir = ROOT / "Chinese-medical-dialogue-data"
    dst_dir = ROOT / "datasets" / "dialogues" / "medical_qa"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in src_dir.glob("**/*.csv"):
        try:
            # 尝试多种编码
            for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]:
                try:
                    df = pd.read_csv(csv_file, nrows=5000, encoding=enc)
                    break
                except:
                    continue
            name = csv_file.stem
            out_path = dst_dir / f"{name}.md"

            # 尝试识别问答列
            q_col = next((c for c in df.columns if c in ("ask","question","query","input","title")), df.columns[0])
            a_col = next((c for c in df.columns if c in ("answer","response","output","reply")), df.columns[-1])

            lines = [f"# {name}\n", f"> 共 {len(df)} 条\n"]
            for i, row in df.iterrows():
                q = str(row[q_col])[:500]
                a = str(row[a_col])[:1000]
                if q and a and q != 'nan' and a != 'nan':
                    lines.append(f"## Q: {q}\n\n**答**: {a}\n\n---\n")

            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[OK] CSV {name}: {len(df)} 行 → {out_path.name}")
        except Exception as e:
            print(f"[WARN] {csv_file.name}: {e}")

# ═══════════════════════════════════════
#  3. huatuo26M_test — 华佗测试集
# ═══════════════════════════════════════
def convert_huatuo_test():
    src = ROOT / "Zhongjing" / "data" / "huatuo26M_test.json"
    dst_dir = ROOT / "datasets" / "dialogues" / "zhongjing"
    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["# Huatuo-26M 测试子集\n", f"> 共 {len(data)} 条\n"]
    for i, item in enumerate(data):
        q = item.get("instruction", item.get("question", ""))
        a = item.get("output", item.get("answer", ""))
        # huatuo26M 字段可能是 list, 取第一个元素
        if isinstance(q, list): q = q[0] if q else ""
        if isinstance(a, list): a = a[0] if a else ""
        if q and a:
            lines.append(f"## Q{i+1}: {q}\n\n**答**: {a}\n\n---\n")

    out = dst_dir / "huatuo26M_test.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Huatuo-26M test: {len(data)} 条 → {out.name}")


if __name__ == "__main__":
    print("=" * 50)
    print("  对话数据集 → datasets/dialogues/")
    print("=" * 50)
    convert_cmtmedqa()
    convert_huatuo_test()
    convert_csv_dialogs()
    print("\nDone.")
