"""从 jicheng.tw 目录抓取五部古籍并生成简体派生文本。

原始 HTML 和繁体提取文本保存在来源层；简体文本由 OpenCC ``t2s``
转换生成，并记录繁体源文本哈希。脚本不会写入 markdown_docs 或 Qdrant。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from opencc import OpenCC


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "datasets" / "tcm_knowledge" / "classics" / "jicheng"
RAW_HTML_DIR = OUTPUT_DIR / "raw_html"
SIMPLIFIED_DIR = OUTPUT_DIR / "simplified"
DIRECTORY_URL = "https://jicheng.tw/tcm/book/index.html"

TARGET_BOOKS = {
    "黃帝內經素問": "黄帝内经_素问",
    "靈樞": "黄帝内经_灵枢",
    "八十一難經": "难经",
    "神農本草經": "神农本草经",
    "溫病條辨": "温病条辨",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9",
}

CONVERTER = OpenCC("t2s")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(url: str, timeout: int = 60) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def discover_books(directory_html: str) -> dict[str, str]:
    """从总目录精确发现目标书目，避免静态维护编码 URL。"""
    soup = BeautifulSoup(directory_html, "lxml")
    discovered = {}
    for link in soup.select("a[href]"):
        traditional_title = link.get_text(" ", strip=True)
        if traditional_title not in TARGET_BOOKS:
            continue
        discovered[TARGET_BOOKS[traditional_title]] = urljoin(
            DIRECTORY_URL,
            link.get("href"),
        )

    missing = set(TARGET_BOOKS.values()) - set(discovered)
    if missing:
        raise RuntimeError(f"目录页缺少目标书目: {sorted(missing)}")
    return discovered


def extract_text(html: str) -> str:
    """提取可读繁体文本；原始 HTML 会另行完整保存。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    for selector in ("article", ".content", "#content", "main", "body"):
        element = soup.select_one(selector)
        if element:
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 500:
                return text
    return soup.get_text(separator="\n", strip=True)


def build_traditional_markdown(
    name: str,
    url: str,
    fetched_at: str,
    html_hash: str,
    text: str,
) -> str:
    return "\n".join(
        [
            f"# {name}",
            f"> 目录来源: {DIRECTORY_URL}",
            f"> 来源页: {url}",
            "> 来源站点: 中醫笈成 jicheng.tw",
            f"> 抓取时间: {fetched_at}",
            f"> 原始HTML SHA256: {html_hash}",
            "> 文本状态: 原始繁体页面提取；未做自动释义、剂量换算或段落合并",
            "> 底本状态: 来源页信息尚待人工核验，不可直接作为临床依据",
            "",
            text,
            "",
        ]
    )


def build_simplified_markdown(
    name: str,
    url: str,
    fetched_at: str,
    traditional_hash: str,
    simplified_text: str,
) -> str:
    return "\n".join(
        [
            f"# {name}",
            f"> 目录来源: {DIRECTORY_URL}",
            f"> 来源页: {url}",
            "> 来源站点: 中醫笈成 jicheng.tw",
            f"> 抓取时间: {fetched_at}",
            "> 派生方式: OpenCC t2s",
            f"> 繁体源文本 SHA256: {traditional_hash}",
            "> 文本状态: 简体检索派生文本；繁体原文保留在上级目录",
            "> 底本状态: 来源页信息尚待人工核验，不可直接作为临床依据",
            "",
            simplified_text,
            "",
        ]
    )


def crawl_book(name: str, url: str) -> dict[str, object]:
    print(f"[FETCH] {name}: {url}")
    html = fetch(url)
    traditional_text = extract_text(html)
    if len(traditional_text) < 1000:
        raise RuntimeError(f"{name} 提取正文不足 1000 字符")

    simplified_text = CONVERTER.convert(traditional_text)
    if simplified_text == traditional_text:
        raise RuntimeError(f"{name} 繁简转换未产生任何变化")

    fetched_at = datetime.now(timezone.utc).isoformat()
    html_hash = sha256_text(html)
    traditional_hash = sha256_text(traditional_text)
    simplified_hash = sha256_text(simplified_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    SIMPLIFIED_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_HTML_DIR / f"{name}.html").write_bytes(html.encode("utf-8"))
    (OUTPUT_DIR / f"{name}.md").write_text(
        build_traditional_markdown(
            name,
            url,
            fetched_at,
            html_hash,
            traditional_text,
        ),
        encoding="utf-8",
    )
    (SIMPLIFIED_DIR / f"{name}.md").write_text(
        build_simplified_markdown(
            name,
            url,
            fetched_at,
            traditional_hash,
            simplified_text,
        ),
        encoding="utf-8",
    )

    return {
        "book": name,
        "directory_url": DIRECTORY_URL,
        "source_url": url,
        "fetched_at": fetched_at,
        "html_sha256": html_hash,
        "traditional_text_sha256": traditional_hash,
        "simplified_text_sha256": simplified_hash,
        "traditional_chars": len(traditional_text),
        "simplified_chars": len(simplified_text),
        "conversion": "OpenCC t2s",
        "review_status": "unreviewed",
    }


def main() -> None:
    directory_html = fetch(DIRECTORY_URL)
    books = discover_books(directory_html)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_HTML_DIR / "catalog.html").write_bytes(directory_html.encode("utf-8"))

    records = []
    for index, (name, url) in enumerate(books.items()):
        if index:
            time.sleep(random.uniform(1.0, 2.0))
        records.append(crawl_book(name, url))

    manifest = {
        "directory_url": DIRECTORY_URL,
        "catalog_html_sha256": sha256_text(directory_html),
        "conversion": "OpenCC t2s",
        "review_status": "unreviewed",
        "books": records,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 五部繁体原文与简体派生文本已保存: {OUTPUT_DIR}")
    print("[HOLD] 未写入 unified/markdown_docs，需完成底本和章节验收")


if __name__ == "__main__":
    main()
