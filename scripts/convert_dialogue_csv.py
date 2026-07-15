"""
医患对话 CSV → Markdown 转换（只保留内/妇/儿科）
来源: https://github.com/Toyhom/Chinese-medical-dialogue-data
"""
import sys
from pathlib import Path

# 复用 config
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
import config


def convert_csv(csv_path: str, dept_name: str):
    """单个科室 CSV → Markdown"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    # 列名: department,title,ask,answer (按实际情况调整)
    lines = [f"# {dept_name} 医患问诊对话\n"]
    for _, row in df.iterrows():
        q = row.get("ask", row.get("question", ""))
        a = row.get("answer", "")
        title = row.get("title", "")
        if title:
            lines.append(f"## {title}\n")
        lines.append(f"**患者**：{q}\n\n**医生**：{a}\n\n---\n")

    out = Path(config.MARKDOWN_DIR) / f"临床对话_{dept_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {dept_name}: {len(df)} 条 → {out.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Chinese-medical-dialogue-data 目录路径")
    args = parser.parse_args()

    # 只取这三个科室
    KEEP = {
        "内科": "Andriatria_内科.csv",
        "妇科": "Gynecology_妇科.csv",
        "儿科": "Paediatrics_儿科.csv",
    }

    for dept, fname in KEEP.items():
        p = Path(args.data_dir) / fname
        if p.exists():
            convert_csv(str(p), dept)
        else:
            # 尝试匹配
            matches = list(Path(args.data_dir).glob(f"*{dept}*"))
            if matches:
                convert_csv(str(matches[0]), dept)
            else:
                print(f"  ⚠ 未找到{dept}数据: {fname}")

    print("\n✅ 转换完成。运行 python project/app.py 即可索引。")
