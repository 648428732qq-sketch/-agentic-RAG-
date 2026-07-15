"""抓取古诗文网的《伤寒论》和《金匮要略》到原始数据层。

整书只有在章节数与预期完全一致、所有章节均成功获取时才会生成。
原始章节 HTML 会保留，提取文本不直接进入 unified 或 markdown_docs。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "datasets" / "tcm_knowledge" / "classics" / "gushiwen"

BOOKS = {
    "伤寒论": {
        "index_url": "https://www.gushiwen.cn/guwen/book.aspx?id=37",
        "expected_chapters": 24,
    },
    "金匮要略": {
        "index_url": "https://www.gushiwen.cn/guwen/book.aspx?id=193",
        "expected_chapters": 25,
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

NOISE_LINES = {
    "元素。",
    "上一章",
    "下一章",
    "目录",
    "古文岛",
    "原古诗文网",
    "播放列表",
    "列表循环",
    "随机播放",
    "单曲循环",
    "单曲播放",
    "您的浏览器不支持",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def get_chapter_links(index_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(fetch(index_url), "lxml")
    chapters = []
    seen = set()
    for link in soup.select("a[href]"):
        href = link.get("href", "").strip()
        if "bookv_" not in href:
            continue
        chapter_url = urljoin(index_url, href)
        title = link.get_text(strip=True)
        if title and chapter_url not in seen:
            seen.add(chapter_url)
            chapters.append((chapter_url, title))
    return chapters


def extract_chapter_text(html: str) -> str:
    """提取章节正文；原始 HTML 会完整保留。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "img"]):
        tag.decompose()

    content = None
    for selector in (".contson", ".sons", "article", "main", "body"):
        candidate = soup.select_one(selector)
        if candidate and len(candidate.get_text(strip=True)) > 100:
            content = candidate
            break
    if content is None:
        raise RuntimeError("未找到章节正文容器")

    lines = []
    for raw_line in content.get_text(separator="\n", strip=True).splitlines():
        line = raw_line.strip()
        if not line or line in NOISE_LINES:
            continue
        if any(noise in line for noise in ("播放列表", "您的浏览器不支持")):
            continue
        if re.fullmatch(r"\d{1,2}(?::\d{2})?", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def crawl_book(name: str, index_url: str, expected_chapters: int) -> dict[str, object]:
    chapters = get_chapter_links(index_url)
    if len(chapters) != expected_chapters:
        raise RuntimeError(
            f"{name} 章节数异常: expected={expected_chapters}, actual={len(chapters)}"
        )

    book_dir = OUTPUT_DIR / name
    raw_html_dir = book_dir / "raw_html"
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    chapter_records = []

    for index, (chapter_url, title) in enumerate(chapters, 1):
        if index > 1:
            time.sleep(random.uniform(0.8, 1.5))
        print(f"[{name} {index}/{expected_chapters}] {title}")
        html = fetch(chapter_url)
        text = extract_chapter_text(html)
        if len(text) < 100:
            raise RuntimeError(f"{name}/{title} 正文不足 100 字符")

        html_file = raw_html_dir / f"{index:03d}.html"
        html_file.write_bytes(html.encode("utf-8"))
        sections.append(f"## {title}\n\n> 章节来源: {chapter_url}\n\n{text}\n")
        chapter_records.append(
            {
                "index": index,
                "title": title,
                "source_url": chapter_url,
                "html_file": str(html_file.relative_to(ROOT)),
                "html_sha256": sha256_text(html),
                "text_sha256": sha256_text(text),
                "text_chars": len(text),
            }
        )

    fetched_at = datetime.now(timezone.utc).isoformat()
    markdown = "\n".join(
        [
            f"# {name}",
            f"> 目录来源: {index_url}",
            "> 来源站点: 古诗文网 gushiwen.cn",
            f"> 抓取时间: {fetched_at}",
            f"> 章节验收: {len(chapters)}/{expected_chapters}",
            "> 底本状态: 来源页未明确提供具体刊本，需另行校勘",
            "> 文本状态: 网页正文提取；未自动翻译、注释或剂量换算",
            "",
            "\n".join(sections),
        ]
    )
    (book_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    (book_dir / "manifest.json").write_text(
        json.dumps(
            {
                "book": name,
                "index_url": index_url,
                "fetched_at": fetched_at,
                "expected_chapters": expected_chapters,
                "retrieved_chapters": len(chapters),
                "review_status": "unreviewed",
                "chapters": chapter_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "book": name,
        "chapters": len(chapters),
        "output": str(book_dir.relative_to(ROOT)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, config in BOOKS.items():
        results.append(crawl_book(name, **config))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("[HOLD] 未写入 unified/markdown_docs，需完成底本和正文抽样验收")


if __name__ == "__main__":
    main()
