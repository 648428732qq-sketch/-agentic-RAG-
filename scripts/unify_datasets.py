"""
统一数据集预处理工具
将所有来源数据转换为 Markdown ==> datasets/unified/ 按内容分类存放
"""
import json
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "datasets" / "unified"


def write_md(subdir: str, filename: str, content: str):
    """写入统一目录"""
    dst = UNIFIED / subdir
    dst.mkdir(parents=True, exist_ok=True)
    (dst / filename).write_text(content, encoding="utf-8")
    print(f"  [OK] {subdir}/{filename}")


# ═══════════════════════════════════════
#  1. CMtMedQA JSON → dialogues/
# ═══════════════════════════════════════
def convert_cmtmedqa():
    src = ROOT / "Zhongjing" / "data" / "CMtMedQA_test.json"
    if not src.exists():
        print(f"  [WARN] 未找到: {src}")
        return

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["# CMtMedQA 中医问答测试集\n", f"> 共 {len(data)} 条\n"]
    for i, item in enumerate(data):
        q = item.get("instruction", item.get("question", ""))
        a = item.get("output", item.get("answer", item.get("response", "")))
        if q and a:
            lines.append(f"## Q{i+1}: {q}\n\n**答**: {a}\n\n---\n")

    write_md("dialogues", "CMtMedQA.md", "\n".join(lines))
    print(f"  CMtMedQA: {len(data)} 条")


# ═══════════════════════════════════════
#  2. Huatuo-26M test JSON → dialogues/
# ═══════════════════════════════════════
def convert_huatuo():
    src = ROOT / "Zhongjing" / "data" / "huatuo26M_test.json"
    if not src.exists():
        print(f"  ⚠ 未找到: {src}")
        return

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["# Huatuo-26M 测试子集\n", f"> 共 {len(data)} 条\n"]
    for i, item in enumerate(data):
        q = item.get("instruction", item.get("question", ""))
        a = item.get("output", item.get("answer", ""))
        if isinstance(q, list):
            q = q[0] if q else ""
        if isinstance(a, list):
            a = a[0] if a else ""
        if q and a:
            lines.append(f"## Q{i+1}: {q}\n\n**答**: {a}\n\n---\n")

    write_md("dialogues", "huatuo26M_test.md", "\n".join(lines))
    print(f"  Huatuo-26M: {len(data)} 条")


# ═══════════════════════════════════════
#  3. RLHF 偏好对 JSON → dialogues/ (可选评估用)
# ═══════════════════════════════════════
def convert_rlhf():
    src = ROOT / "Zhongjing" / "data" / "rlhf2000_7_31.json"
    if not src.exists():
        print(f"  ⚠ 未找到: {src}")
        return

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["# RLHF 偏好对数据\n", f"> 共 {len(data)} 条\n"]
    for i, item in enumerate(data):
        q = item.get("instruction", "")
        outputs = item.get("output", [])
        lines.append(f"## Q{i+1}: {q}\n\n")
        for j, ans in enumerate(outputs[:2]):  # 只取前2个候选
            lines.append(f"**候选{j+1}**: {ans}\n\n")
        lines.append("---\n")

    write_md("dialogues", "rlhf_preferences.md", "\n".join(lines))
    print(f"  RLHF: {len(data)} 条")


# ═══════════════════════════════════════
#  4. CSV 医患对话 → dialogues/
# ═══════════════════════════════════════
def convert_csv_dialogs():
    csv_dir = ROOT / "Chinese-medical-dialogue-data"
    if not csv_dir.exists():
        print(f"  ⚠ 未找到: {csv_dir}")
        return

    dept_map = {
        "Andriatria": "男科",
        "IM": "内科",
        "OAGD": "妇产科",
        "Surgical": "外科",
        "Pediatric": "儿科",
        "Oncology": "肿瘤科",
    }

    for csv_file in csv_dir.glob("**/*.csv"):
        try:
            for enc in ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]:
                try:
                    df = pd.read_csv(csv_file, nrows=5000, encoding=enc)
                    break
                except Exception:
                    continue

            # 识别科室
            parent_dir = csv_file.parent.name
            dept = "未知"
            for key, name in dept_map.items():
                if key in parent_dir or key in csv_file.stem:
                    dept = name
                    break

            q_col = next((c for c in df.columns if c in ("ask", "question", "query", "input", "title")), df.columns[0])
            a_col = next((c for c in df.columns if c in ("answer", "response", "output", "reply")), df.columns[-1])

            lines = [f"# {dept} 医患问诊对话\n", f"> 共 {len(df)} 条\n"]
            for _, row in df.iterrows():
                q = str(row[q_col])[:500]
                a = str(row[a_col])[:1000]
                if q and a and q != "nan" and a != "nan":
                    lines.append(f"## Q: {q}\n\n**答**: {a}\n\n---\n")

            write_md("dialogues", f"{dept}_from_csv.md", "\n".join(lines))
            print(f"  CSV {csv_file.stem}: {len(df)} 行 → {dept}")

        except Exception as e:
            print(f"  ⚠ {csv_file.name}: {e}")


# ═══════════════════════════════════════
#  Main
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  统一数据集预处理 → datasets/unified/")
    print("=" * 55)

    print("\n[JSON 转换]")
    convert_cmtmedqa()
    convert_huatuo()
    convert_rlhf()

    print("\n[CSV 转换]")
    convert_csv_dialogs()

    # 汇总
    print("\n" + "=" * 55)
    print("  目录结构:")
    for sub in ["classics", "herbs", "formulas", "dialogues"]:
        p = UNIFIED / sub
        files = list(p.glob("*.md")) if p.exists() else []
        print(f"  {sub}/: {len(files)} 个文件")
    print("\n[DONE] 完成")
