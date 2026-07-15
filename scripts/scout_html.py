"""侦察 qihuang.vip 方剂/药材页 HTML 结构"""
import requests, sys
from bs4 import BeautifulSoup

io_wrapper = None
try:
    import io
    io_wrapper = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except: pass

def p(*args, **kw):
    msg = ' '.join(str(a) for a in args)
    if io_wrapper: io_wrapper.write(msg + '\n')
    else: print(msg)

URLS = {
    "方剂列表": "https://qihuang.vip/fang.html",
    "方剂第2页": "https://qihuang.vip/fang/page/2.html",
    "药材列表": "https://qihuang.vip/yao.html",
    "古籍目录": "https://qihuang.vip/book.html",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

for name, url in URLS.items():
    p(f"\n{'='*60}")
    p(f"  {name}: {url}")
    p('='*60)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')
        
        # 移除脚本/样式
        for t in soup(['script', 'style']): t.decompose()
        
        # 探测各种class
        probes = ['article', '.content', 'main', '.main', '.container', 
                   '.yao', '.fang', '.item', '.card', '.list',
                   '[class*=content]', '[class*=main]', '[class*=item]',
                   '[class*=yao]', '[class*=fang]', '[class*=book]',
                   '.bm_c', '.tl', '.xw1', '.xs2', '.c', '.ct2', '.mn']
        
        for sel in probes:
            els = soup.select(sel)
            if els:
                sample = els[0].get_text(strip=True)[:120]
                p(f"  {sel:25s} x{len(els):4d}  -> {sample}")
    except Exception as e:
        p(f"  ERROR: {e}")

p("\n\n=== 方剂列表页 body 直接子元素 ===")
try:
    resp = requests.get("https://qihuang.vip/fang.html", headers=HEADERS, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'lxml')
    body = soup.find('body')
    if body:
        for child in body.find_all(recursive=False):
            p(f"  <{child.name}> class={child.get('class')} id={child.get('id')}")
        # 更深层:
        wp = soup.select_one('.wp, #wp, [class*=wrapper], [class*=wrap]')
        if wp:
            p(f"\n  .wp 直接子元素:")
            for c in wp.find_all(recursive=False):
                p(f"    <{c.name}> class={c.get('class')} id={c.get('id')}")
except Exception as e:
    p(f"  ERROR: {e}")
