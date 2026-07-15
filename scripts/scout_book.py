"""侦察古籍详情页结构"""
import requests, sys, io
from bs4 import BeautifulSoup as BS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

r = requests.get('https://qihuang.vip/book-1.html', headers={'User-Agent':'Mozilla/5.0'}, timeout=15)
r.encoding = 'utf-8'
soup = BS(r.text, 'lxml')

print("=== Structure ===")
for sel in ['.mn', '.ct2', '.bm_c', 'h2', 'h3']:
    els = soup.select(sel)
    if els:
        print(f'{sel}: {len(els)} elements')

h2 = soup.select_one('h2')
print(f'Title: {h2.get_text(strip=True)[:80] if h2 else "?"}')

mn = soup.select_one('.mn')
if mn:
    text = mn.get_text(separator='\n', strip=True)
    print(f'\nContent length: {len(text)} chars')
    for line in text.split('\n')[:15]:
        if line.strip():
            print(f'  {line.strip()[:120]}')

links = []
for a in soup.select('.mn a[href], .ct2 a[href]'):
    h = a.get('href','').strip()
    t = a.get_text(strip=True)
    if h and t and len(t) > 1:
        links.append((t, h))
print(f'\nChapter links: {len(links)}')
for t, h in links[:8]:
    print(f'  {t[:40]} -> {h[:60]}')

# Also check for pagination/book-page links
page_links = []
for a in soup.select('a[href]'):
    h = a.get('href','').strip()
    if 'book-1-' in h or 'book_1_' in h:
        page_links.append((a.get_text(strip=True), h))
print(f'\nPagination links: {len(page_links)}')
for t, h in page_links[:5]:
    print(f'  {t[:30]} -> {h}')
