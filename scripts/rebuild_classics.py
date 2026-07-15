"""清理并隔离当前未校勘古籍，触发知识库重建。

默认只预览。使用 --apply 才会写入：

1. 完整备份 datasets/unified/classics 与 markdown_docs 中的七部古籍。
2. 从 unified 版本移除旧版自动括号注释和明确爬虫噪声。
3. 给 unified 版本添加“未经校勘”状态标记。
4. 将七部古籍从 markdown_docs 移出，避免继续参与问诊回答。
5. 写入 .reindex_required，应用下次启动时强制重建 Qdrant。
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from annotate_classical import strip_legacy_annotations, validate_classical_text


ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = ROOT / "datasets" / "unified" / "classics"
MARKDOWN_DIR = ROOT / "markdown_docs"
BACKUP_ROOT = ROOT / "datasets" / "quarantine"
REINDEX_MARKER = ROOT / ".reindex_required"

CLASSIC_NAMES = (
    "伤寒论.md",
    "金匮要略.md",
    "黄帝内经_素问.md",
    "黄帝内经_灵枢.md",
    "难经.md",
    "神农本草经.md",
    "温病条辨.md",
)

STATUS_LINES = (
    "> 数据状态: 未经权威底本逐条校勘，仅供检索实验",
    "> 临床限制: 不可单独作为诊断、处方、剂量换算或治疗依据",
    "> 自动转译: 已移除；现代释义必须来自独立的人工审核数据",
)


def clean_text(text: str) -> str:
    cleaned = strip_legacy_annotations(text)
    cleaned = "\n".join(
        line for line in cleaned.splitlines() if line.strip() != "元素。"
    )
    cleaned = cleaned.rstrip() + "\n"

    if not any(line in cleaned for line in STATUS_LINES):
        lines = cleaned.splitlines()
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].startswith(">"):
            insert_at += 1
        lines[insert_at:insert_at] = list(STATUS_LINES)
        cleaned = "\n".join(lines).rstrip() + "\n"

    return cleaned


def backup_file(source: Path, backup_dir: Path) -> None:
    relative = source.relative_to(ROOT)
    destination = backup_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def run(apply: bool) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"classics_before_cleanup_{timestamp}"
    changed = 0

    for name in CLASSIC_NAMES:
        unified_path = UNIFIED_DIR / name
        active_path = MARKDOWN_DIR / name

        if unified_path.exists():
            original = unified_path.read_text(encoding="utf-8")
            cleaned = clean_text(original)
            issues = validate_classical_text(cleaned)
            print(
                f"[unified] {name}: {len(original):,} -> {len(cleaned):,} chars; "
                f"remaining issues={sum(issue.count for issue in issues)}"
            )
            if apply:
                backup_file(unified_path, backup_dir)
                unified_path.write_text(cleaned, encoding="utf-8")
            changed += original != cleaned

        if active_path.exists():
            print(f"[active]  {name}: quarantine from markdown_docs")
            if apply:
                backup_file(active_path, backup_dir)
                active_path.unlink()
            changed += 1

    if apply:
        REINDEX_MARKER.write_text(
            "Classics were quarantined; rebuild vector and parent stores.\n",
            encoding="utf-8",
        )
        print(f"Backup: {backup_dir}")
        print(f"Reindex marker: {REINDEX_MARKER}")
    else:
        print("DRY RUN: no files changed. Re-run with --apply to execute.")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changed = run(args.apply)
    print(f"Planned/changed items: {changed}")


if __name__ == "__main__":
    main()
