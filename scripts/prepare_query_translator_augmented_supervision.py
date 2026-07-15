from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN = (
    ROOT / "datasets" / "external" / "query_translator_augmented_v3" / "generated" / "private" / "gold_keys.jsonl"
)
DEFAULT_LOCAL_ASR = (
    ROOT / "datasets" / "external" / "query_translator_augmented_v3" / "generated" / "local_asr_gold.jsonl"
)
DEFAULT_FROZEN = ROOT / "tests" / "evals" / "query_translator_v3_nondialect_100" / "private" / "gold_keys.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "external" / "query_translator_augmented_v3" / "supervision" / "query_term_pairs.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def normalized(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum() or "\u3400" <= char <= "\u9fff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and convert augmented questions into Translator supervision")
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--local-asr", type=Path, default=DEFAULT_LOCAL_ASR)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frozen = list(iter_jsonl(args.frozen))
    frozen_queries = {normalized(str(record.get("query", ""))) for record in frozen}
    frozen_entries = {str(record.get("expected_entry_id_in_top_k", "")) for record in frozen}
    output: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    styles: Counter[str] = Counter()
    mapping_count = 0
    for path in (args.qwen, args.local_asr):
        for record in iter_jsonl(path):
            query = str(record.get("query", "")).strip()
            query_key = normalized(query)
            entry_id = str(record.get("expected_entry_id_in_top_k", ""))
            if not query or not query_key or query_key in seen_queries:
                continue
            if query_key in frozen_queries or (entry_id and entry_id in frozen_entries):
                raise AssertionError(f"augmentation/frozen leakage: {record.get('id')}")
            mappings = []
            for mapping in record.get("evidence_mappings") or []:
                term = str(mapping.get("canonical_term", "")).strip()
                phrase = str(mapping.get("source_phrase", "")).strip()
                polarity = str(mapping.get("polarity", "present"))
                if not term or not phrase or phrase not in query or polarity != "present":
                    continue
                mappings.append(
                    {
                        "source_phrase": phrase,
                        "canonical_term": term,
                        "polarity": "present",
                        "confidence": 1.0,
                        "evidence_entry_ids": [entry_id] if entry_id else [],
                        "evidence_tier": "augmented_local_evidence",
                    }
                )
            if not mappings:
                continue
            seen_queries.add(query_key)
            style = str(record.get("generation_style", "unknown"))
            styles[style] += 1
            mapping_count += len(mappings)
            output.append(
                {
                    "query_id": str(record.get("id") or hashlib.sha256(query.encode()).hexdigest()[:24]),
                    "query": query,
                    "source_dataset": "query_translator_augmented_v3",
                    "source_id": str(record.get("source_seed_id", "")),
                    "generation_style": style,
                    "mappings": mappings,
                    "unknown_phrases": [],
                    "supervision_confidence": "synthetic_structurally_validated_local_evidence",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "record_count": len(output),
        "mapping_count": mapping_count,
        "unique_queries": len(seen_queries),
        "style_counts": dict(sorted(styles.items())),
        "frozen_query_overlap": 0,
        "frozen_entry_id_overlap": 0,
        "output": str(args.output),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
