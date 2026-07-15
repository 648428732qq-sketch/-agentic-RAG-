"""清理 qihuang.vip 爬取内容中的登录/注册提示噪声和 …… 截断标记"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "datasets" / "unified"

# 匹配从 "\n提示:" 到 "】" 为止的完整噪声块
NOISE_PATTERN = re.compile(
    r'\n提示:\s*\n'
    r'.*?(?:注册登录后，享有更多权限|当前显示\d+张图片，您需要登录).*?【'
    r'\s*\n登录\s*\n】\s*\n'
    r'(?:…… …… ……\s*)?',
    re.MULTILINE
)

# 匹配 qihuang 内容截断标记: "……" 在行尾，后跟 "---" 或空行
# 例如: "夫仲景殚心 ……---" 或 独立行 "……---"
TRUNCATION_PATTERN = re.compile(
    r'\s*……[-…\s]*$',
    re.MULTILINE
)

cleaned_noise = 0
cleaned_trunc = 0

for subdir in ["classics", "herbs", "formulas"]:
    d = UNIFIED / subdir
    if not d.exists():
        continue
    for f in d.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        
        # 清理登录提示
        new_text, n = NOISE_PATTERN.subn("", text)
        if n > 0:
            cleaned_noise += n
        
        # 清理 …… 截断标记（只删正文末尾的，保留中文正文中间的可能有效省略号）
        new_text2, m = TRUNCATION_PATTERN.subn("", new_text)
        cleaned_trunc += m
        
        if n > 0 or m > 0:
            f.write_text(new_text2, encoding="utf-8")
            print(f"  [OK] {subdir}/{f.name}: noise={n}, trunc={m}")

print(f"\n[DONE] noise blocks: {cleaned_noise}, trunc marks: {cleaned_trunc}")
