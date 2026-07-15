"""从 markdown_docs 重建 Qdrant 集合和 parent_store。"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from core.rag_system import RAGSystem  # noqa: E402


REINDEX_MARKER = ROOT / ".reindex_required"
QUARANTINE_DIR = ROOT / "datasets" / "quarantine"


def _assert_inside_workspace(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to operate outside workspace: {resolved}")
    return resolved


def _backup_and_remove_directory(path: Path, label: str) -> Path | None:
    path = _assert_inside_workspace(path)
    if not path.exists():
        return None
    backup_dir = QUARANTINE_DIR / (
        f"{label}_before_rebuild_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(backup_dir))
    return backup_dir


def rebuild(apply: bool, batch_size: int, wipe_storage: bool) -> tuple[int, int, int]:
    markdown_dir = _assert_inside_workspace(Path(config.MARKDOWN_DIR))
    files = sorted(markdown_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files found in {markdown_dir}")

    if apply and wipe_storage:
        qdrant_backup = _backup_and_remove_directory(Path(config.QDRANT_DB_PATH), "qdrant_db")
        parent_backup = _backup_and_remove_directory(Path(config.PARENT_STORE_PATH), "parent_store")
        if qdrant_backup:
            print(f"Backup qdrant_db: {qdrant_backup}")
        if parent_backup:
            print(f"Backup parent_store: {parent_backup}")

    if apply:
        REINDEX_MARKER.write_text(
            "Vector rebuild in progress; do not trust current stores until this file is removed.\n",
            encoding="utf-8",
        )

    rag = RAGSystem()
    if apply:
        rag.parent_store.clear_store()
        rag.vector_db.delete_collection(rag.collection_name)
        rag.vector_db.create_collection(rag.collection_name)

    total_parents = 0
    total_children = 0
    processed = 0
    collection = rag.vector_db.get_collection(rag.collection_name) if apply else None

    for file in files:
        parent_chunks, child_chunks = rag.chunker.create_chunks_single(file)
        print(
            f"[chunk] {file.name}: parents={len(parent_chunks):,}, "
            f"children={len(child_chunks):,}"
        )
        if not child_chunks:
            raise ValueError(f"No child chunks generated for {file}")
        total_parents += len(parent_chunks)
        total_children += len(child_chunks)
        processed += 1

        if apply:
            rag.parent_store.save_many(parent_chunks)
            for start in range(0, len(child_chunks), batch_size):
                collection.add_documents(child_chunks[start : start + batch_size])

    if apply:
        actual_count = collection.client.count(rag.collection_name, exact=True).count
        if actual_count != total_children:
            raise AssertionError(
                f"Qdrant count mismatch: expected {total_children}, got {actual_count}"
            )
        if REINDEX_MARKER.exists():
            REINDEX_MARKER.unlink()
        print("Reindex marker removed.")
    else:
        print("DRY RUN: no vector store or parent store changes written.")

    print(
        f"Rebuild summary: files={processed}, parents={total_parents:,}, "
        f"children={total_children:,}"
    )
    return processed, total_parents, total_children


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="实际重建；默认仅分块预览")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--wipe-storage",
        action="store_true",
        help="先备份并移除整个 qdrant_db/parent_store，再从空存储重建",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    rebuild(args.apply, args.batch_size, args.wipe_storage)


if __name__ == "__main__":
    main()
