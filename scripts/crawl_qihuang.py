"""抓取 qihuang.vip 的药材和方剂到原始数据层。

古籍抓取已永久停用。代理池为可选功能，只从环境变量
JULIANG_PROXY_API_URL 读取，不允许在代码或日志中保存认证参数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://qihuang.vip"
PROXY_API_URL = os.environ.get("JULIANG_PROXY_API_URL", "").strip()

DATASETS = {
    "herbs": {
        "label": "中药百科",
        "list_url": f"{BASE_URL}/yao/page/{{page}}.html",
        "output_dir": ROOT / "datasets" / "tcm_knowledge" / "herbs",
        "output_file": "中药百科.md",
        "minimum_items": 400,
    },
    "formulas": {
        "label": "方剂大全",
        "list_url": f"{BASE_URL}/fang/page/{{page}}.html",
        "output_dir": ROOT / "datasets" / "tcm_knowledge" / "formulas",
        "output_file": "方剂大全.md",
        "minimum_items": 150,
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

CONTENT_SELECTORS = (
    ".mn",
    ".ct2",
    ".bm_c",
    ".pct",
    ".t_f",
    "article",
    ".content",
    "main",
    "#content",
)

NOISE_SELECTORS = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "noscript",
    "iframe",
    "#hd",
    "#toptb",
    "#qmenu_menu",
    "#append_parent",
    ".p_pop",
    ".blk",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_dynamic_proxy() -> str | None:
    if not PROXY_API_URL:
        return None

    response = requests.get(PROXY_API_URL, timeout=15)
    response.raise_for_status()
    first_line = response.text.strip().splitlines()[0]
    parts = first_line.split(":")
    if len(parts) < 2:
        raise RuntimeError("代理 API 返回格式异常")

    host, port = parts[0].strip(), parts[1].strip()
    if len(parts) >= 4:
        username = quote(parts[2].strip(), safe="")
        password = quote(parts[3].strip(), safe="")
        return f"http://{username}:{password}@{host}:{port}"
    return f"http://{host}:{port}"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    proxy_url = fetch_dynamic_proxy()
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
        print("[PROXY] 已启用动态代理，认证信息不会写入日志")
    else:
        print("[PROXY] 未配置 JULIANG_PROXY_API_URL，使用直连")
    return session


def fetch(session: requests.Session, url: str) -> str:
    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def collect_detail_links(
    session: requests.Session,
    list_url: str,
    max_pages: int = 20,
) -> list[str]:
    links = []
    seen = set()
    stale_pages = 0

    for page in range(1, max_pages + 1):
        url = list_url.replace("/page/{page}", "") if page == 1 else list_url.format(page=page)
        soup = BeautifulSoup(fetch(session, url), "lxml")
        new_count = 0
        for element in soup.select(".yao a[href], .yao[href]"):
            detail_url = urljoin(BASE_URL, element.get("href", "").strip())
            if detail_url and detail_url not in seen:
                seen.add(detail_url)
                links.append(detail_url)
                new_count += 1

        stale_pages = stale_pages + 1 if new_count == 0 else 0
        print(f"[LIST {page}] +{new_count}, total={len(links)}")
        if stale_pages >= 3:
            break
        time.sleep(random.uniform(0.8, 1.5))

    return links


def extract_detail(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for selector in NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    title_element = soup.select_one("h1, h2")
    title = title_element.get_text(strip=True) if title_element else fallback_title
    for selector in CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content:
            text = content.get_text(separator="\n", strip=True)
            if len(text) > 50:
                return title, text
    raise RuntimeError("未找到详情正文")


def crawl_dataset(kind: str) -> dict[str, object]:
    config = DATASETS[kind]
    output_dir = config["output_dir"]
    pages_dir = output_dir / "raw_html"
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = create_session()
    detail_urls = collect_detail_links(session, config["list_url"])
    if len(detail_urls) < config["minimum_items"]:
        raise RuntimeError(
            f"{config['label']}详情数不足: minimum={config['minimum_items']}, "
            f"actual={len(detail_urls)}"
        )

    sections = []
    records = []
    failures = []
    for index, detail_url in enumerate(detail_urls, 1):
        try:
            html = fetch(session, detail_url)
            fallback_title = detail_url.rstrip("/").split("/")[-1]
            title, text = extract_detail(html, fallback_title)
            html_file = pages_dir / f"{index:04d}.html"
            html_file.write_bytes(html.encode("utf-8"))
            sections.append(f"## {title}\n\n> 来源: {detail_url}\n\n{text}\n")
            records.append(
                {
                    "index": index,
                    "title": title,
                    "source_url": detail_url,
                    "html_file": str(html_file.relative_to(ROOT)),
                    "html_sha256": sha256_text(html),
                    "text_sha256": sha256_text(text),
                }
            )
        except Exception as exc:
            failures.append({"url": detail_url, "error": str(exc)})

        if index % 20 == 0:
            print(f"[{config['label']}] {index}/{len(detail_urls)}")
        time.sleep(random.uniform(0.5, 1.2))

    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset": kind,
        "source_site": BASE_URL,
        "fetched_at": fetched_at,
        "discovered": len(detail_urls),
        "retrieved": len(records),
        "failures": failures,
        "review_status": "unreviewed",
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if failures or len(records) != len(detail_urls):
        raise RuntimeError(
            f"{config['label']}存在未完成页面: "
            f"retrieved={len(records)}, discovered={len(detail_urls)}"
        )

    markdown = "\n".join(
        [
            f"# {config['label']}",
            f"> 来源站点: {BASE_URL}",
            f"> 抓取时间: {fetched_at}",
            f"> 条目数: {len(records)}",
            "> 数据状态: 原始网页提取，尚未完成医学审核",
            "",
            "\n---\n\n".join(sections),
        ]
    )
    (output_dir / config["output_file"]).write_text(markdown, encoding="utf-8")
    return {"dataset": kind, "items": len(records), "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", "herbs", "formulas"),
        default="all",
    )
    args = parser.parse_args()

    kinds = ("formulas", "herbs") if args.dataset == "all" else (args.dataset,)
    results = [crawl_dataset(kind) for kind in kinds]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("[HOLD] qihuang 古籍模式已禁用；原始数据未写入 unified/markdown_docs")


if __name__ == "__main__":
    main()
