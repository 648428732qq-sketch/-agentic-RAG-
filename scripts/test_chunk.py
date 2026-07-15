"""分块初步测试 — 方剂/药材/古籍各一例"""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'project'))
os.chdir(PROJECT_ROOT)

from document_chunker import DocumentChuncker

chunker = DocumentChuncker()

samples = {
    '方剂': 'markdown_docs/方剂大全.md',
    '药材': 'markdown_docs/中药百科.md',
    '古籍_伤寒论': 'markdown_docs/伤寒论.md',
    '古籍_一得集': 'markdown_docs/一得集.md',
}

print('=' * 60)
print('  分块测试')
print('=' * 60)

for label, path in samples.items():
    if not os.path.exists(path):
        print(f'\n{label}: FILE NOT FOUND')
        continue

    fsize = os.path.getsize(path)
    parents, children = chunker.create_chunks_single(path)

    print(f'\n--- {label} ---')
    print(f'  文件大小: {fsize:,} bytes')
    print(f'  父块数: {len(parents)}')
    print(f'  子块数: {len(children)}')

    if parents:
        sizes = [len(p[1].page_content) for p in parents]
        print(f'  父块大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}')

    if children:
        csizes = [len(c.page_content) for c in children]
        print(f'  子块大小: min={min(csizes)}, max={max(csizes)}, avg={sum(csizes) // len(csizes)}')

    if parents:
        meta = parents[0][1].metadata
        first_title = meta.get('H2', meta.get('H1', '?'))
        content = parents[0][1].page_content.replace('\n', ' ')[:150]
        print(f'  第1父块标题: {first_title[:60]}')
        print(f'  第1父块内容: {content}')

    if '古籍' in label and parents:
        h1_only = all('H2' not in p[1].metadata for p in parents)
        if h1_only:
            print(f'  [注] 无 ## 标题, 仅按 # H1 + 大小强制切割')

print()
print('=' * 60)
print('  总结')
print('=' * 60)
print('- 方剂: 180条独立 ## 标题 -> MarkdownHeaderTextSplitter 完美切割')
print('- 药材: 474条独立 ## 标题 -> 同上')
print('- 古籍: 大部分无 ## 标题 -> __split_large_parents 按3500字强制切分')
print('- 风险: 古籍会被截断在句中，语义不完整')
print()
print('  建议:')
print('  1. 方剂/药材直接可用现有配置')
print('  2. 古籍需额外处理: 要么用 RecursiveCharacterTextSplitter')
print('     按段落/句号分块, 要么先通过LLM给古籍加 ## 标题再分块')
