"""发布经过筛选的 Markdown 文档到 markdown_docs。

发布内容：
1. 七部已重抓古籍简体原文。
2. 七部古籍术语通俗释义伴随稿。
3. 药材、方剂知识。
4. 从 CMtMedQA 中保守筛选的中医问诊语料。

不会发布 Huatuo、RLHF、CSV 全量问诊和旧 qihuang 古籍。
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_DIR = ROOT / "markdown_docs"
QUARANTINE_DIR = ROOT / "datasets" / "quarantine"
TCM_CLASSICS = ROOT / "datasets" / "tcm_knowledge" / "classics"
UNIFIED = ROOT / "datasets" / "unified"
REINDEX_MARKER = ROOT / ".reindex_required"

CLASSIC_SOURCES = {
    "黄帝内经_素问": TCM_CLASSICS / "jicheng" / "simplified" / "黄帝内经_素问.md",
    "黄帝内经_灵枢": TCM_CLASSICS / "jicheng" / "simplified" / "黄帝内经_灵枢.md",
    "难经": TCM_CLASSICS / "jicheng" / "simplified" / "难经.md",
    "神农本草经": TCM_CLASSICS / "jicheng" / "simplified" / "神农本草经.md",
    "温病条辨": TCM_CLASSICS / "jicheng" / "simplified" / "温病条辨.md",
    "伤寒论": TCM_CLASSICS / "gushiwen" / "伤寒论" / "伤寒论.md",
    "金匮要略": TCM_CLASSICS / "gushiwen" / "金匮要略" / "金匮要略.md",
}

COMPANION_DIR = TCM_CLASSICS / "plain_language"

KNOWLEDGE_SOURCES = {
    "中药百科": UNIFIED / "herbs" / "中药百科.md",
    "方剂大全": UNIFIED / "formulas" / "方剂大全.md",
}

CMTMEDQA_SOURCE = UNIFIED / "dialogues" / "CMtMedQA.md"

TCM_DIALOGUE_KEYWORDS = (
    "中医",
    "中药",
    "中成药",
    "方剂",
    "药材",
    "草药",
    "针灸",
    "艾灸",
    "推拿",
    "拔罐",
    "刮痧",
    "舌苔",
    "舌质",
    "脉象",
    "阴虚",
    "阳虚",
    "气虚",
    "血虚",
    "肾虚",
    "脾虚",
    "肝郁",
    "湿热",
    "痰湿",
    "寒湿",
    "上火",
    "辨证",
    "调理",
    "经络",
    "穴位",
    "桂枝",
    "麻黄",
    "柴胡",
    "黄芪",
    "当归",
    "党参",
    "甘草",
    "枸杞",
    "决明子",
    "金银花",
    "蒲公英",
)

QUESTION_MARKERS = (
    "？",
    "?",
    "吗",
    "么",
    "哪些",
    "怎么",
    "如何",
    "请问",
    "应该",
    "能不能",
    "可以",
)

LOW_VALUE_QUESTION_PREFIXES = (
    "好的",
    "谢谢",
    "没有了",
    "非常感谢",
    "我知道了",
    "我了解了",
    "不客气",
)

FORBIDDEN_PATTERNS = (
    "约15克",
    "约200毫升",
    "约5-10克",
    "不可与（不可以用）（可以用）",
    "上主之（主治此证）",
    "元素。",
)


@dataclass(frozen=True)
class PublishedFile:
    source: Path
    destination: Path
    category: str
    chars: int


def _assert_inside_workspace(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to operate outside workspace: {resolved}")
    return resolved


def _clear_directory(path: Path) -> None:
    path = _assert_inside_workspace(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _backup_markdown_docs() -> Path | None:
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(MARKDOWN_DIR.iterdir())
    if not existing:
        return None
    backup_dir = QUARANTINE_DIR / (
        "markdown_docs_before_publish_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for source in existing:
        destination = backup_dir / source.relative_to(MARKDOWN_DIR)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return backup_dir


def _first_metadata_url(text: str) -> str | None:
    patterns = (
        r"^>\s*来源页[：:]\s*(https?://\S+)",
        r"^>\s*章节来源[：:]\s*(https?://\S+)",
        r"^>\s*目录来源[：:]\s*(https?://\S+)",
        r"^>\s*来源[：:]\s*(https?://\S+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)
    return None


def _add_source_aliases(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        match = re.match(r"^>\s*(?:来源页|章节来源|目录来源)[：:]\s*(https?://\S+)", line)
        if match:
            alias = f"> 来源: {match.group(1)}"
            if alias not in out:
                out.append(alias)
    if not any(re.match(r"^>\s*来源[：:]\s*https?://", line) for line in out):
        url = _first_metadata_url(text)
        if url:
            insert_at = 1 if out and out[0].startswith("# ") else 0
            out.insert(insert_at, f"> 来源: {url}")
    return "\n".join(out).rstrip() + "\n"


def _with_publish_notice(text: str, category: str) -> str:
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].startswith("# ") else 0
    notices = [
        f"> 发布分类: {category}",
        "> 发布状态: 已筛选进入活动知识库",
    ]
    for notice in reversed(notices):
        if notice not in lines:
            lines.insert(insert_at, notice)
    return "\n".join(lines).rstrip() + "\n"


def _read_clean_source(path: Path, category: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in text:
            raise ValueError(f"{path} contains forbidden pattern: {pattern}")
    return _with_publish_notice(_add_source_aliases(text), category)


def _is_useful_tcm_dialogue(question: str, answer: str) -> bool:
    question = question.strip()
    answer = answer.strip()
    if len(question) < 8 or len(answer) < 30:
        return False
    if any(question.startswith(prefix) for prefix in LOW_VALUE_QUESTION_PREFIXES):
        return False
    body = question + answer
    if not any(keyword in body for keyword in TCM_DIALOGUE_KEYWORDS):
        return False
    return any(marker in question for marker in QUESTION_MARKERS)


def build_filtered_cmtmedqa(source: Path = CMTMEDQA_SOURCE) -> tuple[str, int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    pattern = re.compile(
        r"## Q(?P<num>\d+): (?P<question>.*?)\n\n"
        r"\*\*答\*\*: (?P<answer>.*?)(?=\n---\n|\Z)",
        re.DOTALL,
    )
    selected: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in pattern.finditer(text):
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        key = (question, answer)
        if key in seen or not _is_useful_tcm_dialogue(question, answer):
            continue
        seen.add(key)
        selected.append((match.group("num"), question, answer))

    lines = [
        "# 问诊_CMtMedQA_中医筛选",
        "> 来源: datasets/unified/dialogues/CMtMedQA.md",
        "> 发布分类: 低信任问诊语料",
        "> 发布状态: 已按中医关键词、问句形态、非寒暄规则筛选",
        "> 使用限制: 仅用于理解问诊表达和对话流程，不得作为诊断、处方、剂量或禁忌事实源",
        f"> 筛选数量: {len(selected)}",
        "",
    ]
    for index, (original_id, question, answer) in enumerate(selected, start=1):
        lines.extend(
            [
                f"## Q{index}: {question}",
                "",
                f"> 原始ID: CMtMedQA Q{original_id}",
                "",
                f"**答**: {answer}",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n", len(selected)


def _write_document(destination: Path, text: str) -> PublishedFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return PublishedFile(
        source=destination,
        destination=destination,
        category="generated",
        chars=len(text),
    )


def publish(apply: bool) -> list[PublishedFile]:
    planned: list[tuple[Path, str, str]] = []

    for title, source in CLASSIC_SOURCES.items():
        planned.append((source, f"古籍_{title}.md", "古籍原文"))
        planned.append((COMPANION_DIR / f"{title}.md", f"古籍_{title}_术语通俗释义.md", "古籍术语伴随稿"))

    for title, source in KNOWLEDGE_SOURCES.items():
        planned.append((source, f"{title}.md", "药材方剂知识"))

    published: list[PublishedFile] = []
    for source, filename, category in planned:
        text = _read_clean_source(source, category)
        print(f"[plan] {category}: {filename} <- {source.relative_to(ROOT)} ({len(text):,} chars)")
        if apply:
            destination = MARKDOWN_DIR / filename
            destination.write_text(text, encoding="utf-8")
            published.append(PublishedFile(source, destination, category, len(text)))

    dialogue_text, dialogue_count = build_filtered_cmtmedqa()
    if dialogue_count == 0:
        raise ValueError("CMtMedQA 中未筛出合格中医问诊数据")
    print(
        "[plan] 低信任问诊语料: 问诊_CMtMedQA_中医筛选.md "
        f"({dialogue_count} entries, {len(dialogue_text):,} chars)"
    )
    if apply:
        published.append(
            _write_document(MARKDOWN_DIR / "问诊_CMtMedQA_中医筛选.md", dialogue_text)
        )
        REINDEX_MARKER.write_text(
            "markdown_docs was republished; rebuild vector and parent stores.\n",
            encoding="utf-8",
        )
    return published


def validate_published() -> None:
    files = sorted(MARKDOWN_DIR.glob("*.md"))
    expected_count = 7 + 7 + 2 + 1
    if len(files) != expected_count:
        raise AssertionError(f"Expected {expected_count} markdown files, got {len(files)}")
    forbidden_names = ("huatuo", "rlhf", "_from_csv", "5000-", "5-14000", "6-28000")
    for file in files:
        lower_name = file.name.lower()
        if any(pattern in lower_name for pattern in forbidden_names):
            raise AssertionError(f"Forbidden published file: {file.name}")
        text = file.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                raise AssertionError(f"{file.name} contains forbidden pattern: {pattern}")
    print(f"Published validation OK: {len(files)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="写入 markdown_docs；默认仅预览")
    parser.add_argument("--validate", action="store_true", help="验证当前 markdown_docs")
    args = parser.parse_args()

    if args.validate:
        validate_published()
        return

    if args.apply:
        backup_dir = _backup_markdown_docs()
        if backup_dir:
            print(f"Backup markdown_docs: {backup_dir}")
        _clear_directory(MARKDOWN_DIR)
    published = publish(args.apply)
    if args.apply:
        validate_published()
        print(f"Published files: {len(published)}")
    else:
        print("DRY RUN: no files written. Re-run with --apply to publish.")


if __name__ == "__main__":
    main()
