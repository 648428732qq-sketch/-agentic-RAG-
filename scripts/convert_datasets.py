"""
中医数据集预处理脚本
将所有来源的数据统一转为 Markdown 格式，存入 markdown_docs/

支持格式:
- HuggingFace datasets (Huatuo-26M 等)
- JSON/JSONL (CMtMedQA, MedDialog, BianCang)
- CSV (Chinese-medical-dialogue-data)
- 纯文本 HTML (中医世家等爬下来的)
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# 把 project/ 加入路径以复用 config
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
import config


def write_markdown(filename: str, content: str):
    """写入并打印进度"""
    path = Path(config.MARKDOWN_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ 已生成: {path.name}")


# ============================================================
#  转换器 1: HuggingFace datasets (Huatuo-26M 等)
# ============================================================
def convert_huggingface(dataset_path: str, max_per_file: int = 10000):
    """从 HuggingFace 加载数据集并转 Markdown"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ 需要安装 datasets: pip install datasets")
        return

    print(f"📦 加载 HuggingFace 数据集: {dataset_path}")
    ds = load_dataset(dataset_path, split="train")
    print(f"  {len(ds)} 条记录")

    buffer = []
    file_idx = 0
    for i, item in enumerate(ds):
        q = item.get("question", item.get("query", item.get("input", "")))
        a = item.get("answer", item.get("response", item.get("output", "")))
        dept = item.get("department", item.get("科室", "通用"))
        disease = item.get("disease", item.get("疾病", ""))

        buffer.append(f"## 问：{q}\n\n## 答：{a}\n\n---\n")

        if len(buffer) >= max_per_file:
            title = disease or dept or "问答"
            safe_name = str(title).replace("/", "_").replace("\\", "_")[:50]
            write_markdown(f"huatuo_{safe_name}_{file_idx}.md",
                           f"# {title} 问答集\n\n" + "\n".join(buffer))
            buffer = []
            file_idx += 1

    if buffer:
        write_markdown(f"huatuo_余_第{file_idx}批.md",
                       f"# 问答补充集\n\n" + "\n".join(buffer))


# ============================================================
#  转换器 2: JSON/JSONL 对话数据 (CMtMedQA, MedDialog 等)
# ============================================================
def convert_json_dialogs(json_path: str, max_per_file: int = 5000):
    """
    支持两种格式:
    A. 多轮对话: [{"dialog": [{"role":"patient","content":"..."}, ...]}, ...]
    B. 问答对: [{"question":"...", "answer":"..."}, ...]
    """
    print(f"📦 加载 JSON: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f) if json_path.endswith(".json") else [json.loads(l) for l in f]

    print(f"  {len(data)} 条记录")

    buffer = []
    file_idx = 0
    for item in data:
        if "dialog" in item:
            # 多轮对话格式
            lines = []
            for turn in item["dialog"]:
                role = "患者" if "patient" in str(turn.get("role", "")).lower() else "医生"
                lines.append(f"**{role}**：{turn.get('content', '')}")
            buffer.append("\n\n".join(lines) + "\n\n---\n")
        else:
            # 简单问答格式
            q = item.get("question", item.get("query", item.get("input", "")))
            a = item.get("answer", item.get("response", item.get("output", "")))
            buffer.append(f"## 问：{q}\n\n## 答：{a}\n\n---\n")

        if len(buffer) >= max_per_file:
            write_markdown(f"dialog_{Path(json_path).stem}_{file_idx}.md",
                           f"# 医患对话集\n\n" + "\n".join(buffer))
            buffer = []
            file_idx += 1

    if buffer:
        write_markdown(f"dialog_{Path(json_path).stem}_余.md",
                       f"# 对话补充集\n\n" + "\n".join(buffer))


# ============================================================
#  转换器 3: CSV 对话数据 (Chinese-medical-dialogue)
# ============================================================
def convert_csv_dialogs(csv_path: str, max_per_file: int = 5000):
    """转换 CSV：按科室分组输出 Markdown"""
    try:
        import pandas as pd
    except ImportError:
        print("⚠ 需要安装 pandas: pip install pandas")
        return

    print(f"📦 加载 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} 行, 列: {list(df.columns)}")

    # 自动检测列名
    q_col = next((c for c in df.columns if c in ("ask", "question", "query", "input", "title")), df.columns[0])
    a_col = next((c for c in df.columns if c in ("answer", "response", "output", "reply")), df.columns[-1])
    dept_col = next((c for c in df.columns if c in ("department", "科室", "dept")), None)

    if dept_col and dept_col in df.columns:
        for dept, group in df.groupby(dept_col):
            lines = []
            for _, row in group.iterrows():
                lines.append(f"## 问：{row[q_col]}\n\n## 答：{row[a_col]}\n\n---\n")
            safe_dept = str(dept).replace("/", "_")[:30]
            write_markdown(f"{safe_dept}_qa.md",
                           f"# {dept} 问答集\n\n" + "\n".join(lines))
    else:
        # 无科室列，整体转一个文件
        lines = []
        for _, row in df.iterrows():
            lines.append(f"## 问：{row[q_col]}\n\n## 答：{row[a_col]}\n\n---\n")
        write_markdown(f"{Path(csv_path).stem}_qa.md",
                       f"# 问答集\n\n" + "\n".join(lines))


# ============================================================
#  转换器 4: 纯文本文件直接拷贝
# ============================================================
def convert_txt_as_markdown(txt_dir: str):
    """将 .txt 文件重命名为 .md 拷贝到 markdown_docs"""
    txt_dir = Path(txt_dir)
    for txt_file in txt_dir.glob("*.txt"):
        dest = Path(config.MARKDOWN_DIR) / txt_file.name.replace(".txt", ".md")
        shutil.copy(txt_file, dest)
        print(f"  ✓ 拷贝: {txt_file.name} → {dest.name}")


# ============================================================
#  CLI 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中医数据集 → Markdown 预处理工具")
    parser.add_argument("command", choices=["hf", "json", "csv", "txt"],
                        help="hf=HuggingFace | json=JSON对话 | csv=CSV | txt=纯文本")
    parser.add_argument("path", help="HuggingFace数据集名 或 文件路径")
    parser.add_argument("--max", type=int, default=None, help="每个 Markdown 文件最大条数")
    args = parser.parse_args()

    kwargs = {"max_per_file": args.max} if args.max else {}

    if args.command == "hf":
        convert_huggingface(args.path, **kwargs)
    elif args.command == "json":
        convert_json_dialogs(args.path, **kwargs)
    elif args.command == "csv":
        convert_csv_dialogs(args.path, **kwargs)
    elif args.command == "txt":
        convert_txt_as_markdown(args.path)

    print("\n✅ 转换完成。运行 python project/app.py 即可索引新数据。")
