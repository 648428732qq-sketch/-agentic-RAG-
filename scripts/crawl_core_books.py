"""
核心中医古籍精爬 — 爬取章节正文（非仅目录）
目标: 8本核心经典, 带 content-N.html 子页面
"""
import sys, os, re, time, requests
from pathlib import Path
from bs4 import BeautifulSoup

# ── 配路径 ──
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "datasets" / "tcm_knowledge" / "classics"
OUTPUT.mkdir(exist_ok=True, parents=True)

# ── 核心古籍URL (书名 → book页面) ──
CORE_BOOKS = {
    "伤寒论": "https://qihuang.vip/book-109.html",
    "金匮要略": "https://qihuang.vip/book-111.html",
    "神农本草经": "https://qihuang.vip/book-31.html",
    "温病条辨": "https://qihuang.vip/book-262.html",
    "黄帝内经_素问": "https://qihuang.vip/book-113.html",
    "黄帝内经_灵枢": "https://qihuang.vip/book-114.html",
    "难经": "https://qihuang.vip/book-107.html",
    "医宗金鉴": "https://qihuang.vip/book-250.html",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
DELAY = 1.5  # 礼貌间隔

def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "lxml")

def extract_text(soup):
    """从多种页面结构提取正文"""
    for t in soup(["script", "style", "nav", "header", "footer",
                   "#hd", "#toptb", "#qmenu_menu", "#append_parent",
                   ".p_pop", ".blk"]):
        t.decompose()
    # 优先级: .mn (列表页) > #view (content-N正文) > #content > #container
    for sel in [".mn", "#view", "#content", "#container", ".ct2", ".bm_c"]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 50:
                return text
    return ""

def crawl_book(name, book_url):
    """爬取一本古籍: 提取目录链接 → 爬每个 content-N.html 章节"""
    safe_name = name.replace("/", "_")[:60]
    out_path = OUTPUT / f"{safe_name}.md"
    
    if out_path.exists() and out_path.stat().st_size > 200:
        print(f"  [SKIP] {name} (已存在, {out_path.stat().st_size} bytes)")
        return

    print(f"\n📖 {name}: {book_url}")
    
    # Step 1: 获取 book 首页, 提取所有 content-N.html 链接
    soup = fetch(book_url)
    time.sleep(DELAY)
    
    chapter_urls = []
    seen = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if re.match(r"content-\d+\.html", href):
            full = f"https://qihuang.vip/{href}"
            if full not in seen:
                seen.add(full)
                # 获取章节标题
                title = a.get_text(strip=True)
                chapter_urls.append((full, title))
    
    if not chapter_urls:
        # 无章节链接 → 直接提取首页正文
        print(f"  (单页, 无子章节)")
        text = extract_text(soup)
        if text:
            markdown = f"# {name}\n\n> 来源: {book_url}\n\n{text}"
            tmp_path = out_path.with_suffix(".tmp")
            tmp_path.write_text(markdown, encoding="utf-8")
            tmp_path.replace(out_path)
            print(f"  [OK] (单页) {len(markdown)} 字符")
        return

    print(f"  共 {len(chapter_urls)} 个章节")
    
    # Step 2: 爬取每个章节
    sections = [f"# {name}\n\n> 来源: {book_url}\n"]
    for i, (ch_url, ch_title) in enumerate(chapter_urls):
        try:
            ch_soup = fetch(ch_url)
            time.sleep(DELAY * 0.5)
            text = extract_text(ch_soup)
            if text:
                title_line = f"## {ch_title}" if ch_title else f"## 章节 {i+1}"
                sections.append(f"{title_line}\n\n> 来源: {ch_url}\n\n{text}\n")
        except Exception as e:
            print(f"    [ERR] {ch_url}: {e}")
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(chapter_urls)}] ...")

    full_md = "\n\n---\n\n".join(sections)
    # 原子写入: 先写临时文件再改名, 防止中途崩溃留下空文件
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(full_md, encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"  [OK] {len(chapter_urls)} 章, {len(full_md):,} 字符 → {out_path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("  核心中医古籍精爬 (带章节正文)")
    print("=" * 60)
    
    for name, url in CORE_BOOKS.items():
        try:
            crawl_book(name, url)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    
    print(f"\n✅ 完成! 输出: {OUTPUT}")
