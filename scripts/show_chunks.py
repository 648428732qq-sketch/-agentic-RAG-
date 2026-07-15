"""展示分块结果样本 + 验证来源，供人工审查"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from document_chunker import DocumentChuncker

chunker = DocumentChuncker()
unified = Path(__file__).parent.parent / "datasets" / "unified"
out_file = Path(__file__).parent / "chunk_samples.txt"

with open(out_file, "w", encoding="utf-8") as out:
    def p(*args, **kw):
        print(*args, file=out, **kw)

    for sub, sample_file in [
        ("classics",    "伤寒论.md"),
        ("herbs",       "中药百科.md"),
        ("formulas",    "方剂大全.md"),
        ("dialogues",   "内科5000-33000.md"),
    ]:
        f = unified / sub / sample_file
        if not f.exists():
            continue
        parents, children = chunker.create_chunks_single(str(f))

        p(f"===== [{sub}] {sample_file} =====")
        p(f"父块总数: {len(parents)}, 子块总数: {len(children)}")

        for i in range(min(3, len(parents))):
            pid, p_chunk = parents[i]
            meta = p_chunk.metadata
            content = p_chunk.page_content[:500]

            p(f"\n--- 父块 #{i} (id={pid}) ---")
            p(f"  doc_name   : {meta.get('doc_name','?')}")
            p(f"  chapter    : {meta.get('chapter','?')}")
            p(f"  source_url : {meta.get('source_url','?')}")
            p(f"  H1/H2      : {meta.get('H1','?')} | {meta.get('H2','?')}")
            p(f"  size       : {len(p_chunk.page_content)} chars")
            p(f"  content    :")
            p(f"    {content}")

            child_count = sum(1 for c in children if c.metadata.get("parent_id") == pid)
            p(f"  child_chunks: {child_count}")

            shown = 0
            for c in children:
                if c.metadata.get("parent_id") == pid and shown < 1:
                    c_content = c.page_content[:400]
                    p(f"  child #{shown}:")
                    p(f"    size      : {len(c.page_content)}")
                    p(f"    source_url: {c.metadata.get('source_url','?')}")
                    p(f"    content   : {c_content}")
                    shown += 1

        p()

print(f"[OK] Written {out_file.stat().st_size} bytes to {out_file}")
