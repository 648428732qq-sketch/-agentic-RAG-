"""验证重新抓取的七部古籍原始层，不执行清洗或发布。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opencc import OpenCC

from annotate_classical import validate_classical_text


ROOT = Path(__file__).resolve().parent.parent
JICHENG_DIR = ROOT / "datasets" / "tcm_knowledge" / "classics" / "jicheng"
JICHENG_SIMPLIFIED_DIR = JICHENG_DIR / "simplified"
GUSHIWEN_DIR = ROOT / "datasets" / "tcm_knowledge" / "classics" / "gushiwen"
MARKDOWN_DIR = ROOT / "markdown_docs"
REINDEX_MARKER = ROOT / ".reindex_required"

JICHENG_BOOKS = {
    "黄帝内经_素问",
    "黄帝内经_灵枢",
    "难经",
    "神农本草经",
    "温病条辨",
}

GUSHIWEN_BOOKS = {
    "伤寒论": 24,
    "金匮要略": 25,
}

CONVERTER = OpenCC("t2s")


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_body(text: str) -> str:
    return text.split("\n\n", 1)[1].rstrip() if "\n\n" in text else ""


def validate_jicheng() -> None:
    manifest = json.loads(
        (JICHENG_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    records = manifest["books"]
    assert {record["book"] for record in records} == JICHENG_BOOKS
    assert manifest["directory_url"] == "https://jicheng.tw/tcm/book/index.html"
    assert manifest["conversion"] == "OpenCC t2s"
    assert (
        sha256_bytes(JICHENG_DIR / "raw_html" / "catalog.html")
        == manifest["catalog_html_sha256"]
    )

    for record in records:
        name = record["book"]
        html_path = JICHENG_DIR / "raw_html" / f"{name}.html"
        traditional_path = JICHENG_DIR / f"{name}.md"
        simplified_path = JICHENG_SIMPLIFIED_DIR / f"{name}.md"
        assert html_path.is_file(), html_path
        assert traditional_path.is_file(), traditional_path
        assert simplified_path.is_file(), simplified_path
        assert sha256_bytes(html_path) == record["html_sha256"], html_path
        traditional = traditional_path.read_text(encoding="utf-8")
        simplified = simplified_path.read_text(encoding="utf-8")
        traditional_body = markdown_body(traditional)
        simplified_body = markdown_body(simplified)
        assert (
            hashlib.sha256(traditional_body.encode("utf-8")).hexdigest()
            == record["traditional_text_sha256"]
        ), traditional_path
        assert (
            hashlib.sha256(simplified_body.encode("utf-8")).hexdigest()
            == record["simplified_text_sha256"]
        ), simplified_path
        assert CONVERTER.convert(traditional_body) == simplified_body, name
        assert record["traditional_text_sha256"] in simplified, simplified_path
        assert "派生方式: OpenCC t2s" in simplified, simplified_path
        assert traditional != simplified, name
        assert record["review_status"] == "unreviewed", name
        assert not validate_classical_text(traditional), traditional_path
        assert not validate_classical_text(simplified), simplified_path


def validate_gushiwen() -> None:
    for name, expected_chapters in GUSHIWEN_BOOKS.items():
        book_dir = GUSHIWEN_DIR / name
        manifest = json.loads(
            (book_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["expected_chapters"] == expected_chapters, name
        assert manifest["retrieved_chapters"] == expected_chapters, name
        assert manifest["review_status"] == "unreviewed", name
        assert len(manifest["chapters"]) == expected_chapters, name

        markdown = (book_dir / f"{name}.md").read_text(encoding="utf-8")
        assert markdown.count("\n## ") == expected_chapters, name
        assert not validate_classical_text(markdown), name

        for chapter in manifest["chapters"]:
            html_path = ROOT / chapter["html_file"]
            assert html_path.is_file(), html_path
            assert sha256_bytes(html_path) == chapter["html_sha256"], html_path


def validate_quarantine_boundary() -> None:
    active_names = {path.name for path in MARKDOWN_DIR.glob("*.md")}
    assert not (active_names & {f"{name}.md" for name in GUSHIWEN_BOOKS})
    assert not (active_names & {f"{name}.md" for name in JICHENG_BOOKS})

    if REINDEX_MARKER.is_file():
        return

    expected_published = {
        *(f"古籍_{name}.md" for name in JICHENG_BOOKS),
        *(f"古籍_{name}_术语通俗释义.md" for name in JICHENG_BOOKS),
        *(f"古籍_{name}.md" for name in GUSHIWEN_BOOKS),
        *(f"古籍_{name}_术语通俗释义.md" for name in GUSHIWEN_BOOKS),
    }
    missing = expected_published - active_names
    assert not missing, missing


def main() -> None:
    validate_jicheng()
    validate_gushiwen()
    validate_quarantine_boundary()
    print("七部古籍原始层验证通过: jicheng=5, gushiwen=2, HTML=54")
    if REINDEX_MARKER.is_file():
        print("发布状态: HOLD（未进入 active index）")
    else:
        print("发布状态: PUBLISHED（已进入 markdown_docs，等待或已完成索引验证）")


if __name__ == "__main__":
    main()
