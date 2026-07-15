from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

import config  # noqa: E402
from scripts.build_syndrome_dictionary import SyndromeEntry, write_qdrant  # noqa: E402


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary_effective.jsonl"
DEFAULT_MARKER = ROOT / ".syndrome_reindex_required"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "syndrome_qdrant_rebuild_report.json"


def read_entries(path: Path) -> list[SyndromeEntry]:
    entries = [
        SyndromeEntry.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entry_ids = [entry.entry_id for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("effective dictionary contains duplicate entry_id values")
    if not entries:
        raise ValueError("effective dictionary is empty")
    return entries


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the syndrome Qdrant collection from the effective dictionary")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    entries = read_entries(args.dictionary)
    dictionary_sha256 = hashlib.sha256(args.dictionary.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "ok": True,
        "applied": bool(args.apply),
        "backend": "remote" if config.QDRANT_URL else "local",
        "collection": config.SYNDROME_COLLECTION,
        "dictionary": str(args.dictionary.resolve()),
        "dictionary_sha256": dictionary_sha256,
        "expected_points": len(entries),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if not args.apply:
        report["status"] = "dry_run_validated"
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(
        "Syndrome Qdrant rebuild in progress or failed; do not serve structured syndrome answers.\n",
        encoding="utf-8",
    )
    try:
        count = write_qdrant(entries, recreate=True)
        if count != len(entries):
            raise RuntimeError(f"Qdrant count mismatch: expected {len(entries)}, got {count}")
        report.update({"status": "rebuilt", "actual_points": count})
        args.marker.unlink(missing_ok=True)
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_report(args.report, report)
        raise
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
