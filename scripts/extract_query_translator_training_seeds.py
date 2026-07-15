from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_unseen_query_translator_seeds import (  # noqa: E402
    DEFAULT_DICTIONARY,
    DEFAULT_EVAL_ROOT,
    build_seed_query,
    candidate_from_entry,
    collect_old_usage,
    read_jsonl,
    stable_hash,
    write_jsonl,
)


DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "query_translator_augmented_v3"
SOURCE_QUOTAS = {
    "formula_syndrome": 6,
    "classical_clause": 60,
    "herb_indication": 152,
    "classical_acupuncture": 45,
    "classical_theory": 26,
    "classical_acupuncture_principle": 11,
}


def exact_signature(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(item["signature"]))


def select_training_candidates(
    candidates: list[dict[str, Any]],
    quotas: dict[str, int],
    salt: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_concepts: set[str] = set()
    used_signatures: set[tuple[str, ...]] = set()
    for source_type, quota in quotas.items():
        source_candidates = [item for item in candidates if item["source_type"] == source_type]
        source_candidates.sort(
            key=lambda item: (-item["quality"], stable_hash(item["entry_id"], salt))
        )
        picked: list[dict[str, Any]] = []
        direction_counts: Counter[str] = Counter()
        while len(picked) < quota:
            eligible = [
                item
                for item in source_candidates
                if item["concept_key"] not in used_concepts
                and exact_signature(item) not in used_signatures
            ]
            if not eligible:
                raise RuntimeError(
                    f"training seed quota unavailable: {source_type}={len(picked)}/{quota}"
                )
            chosen = min(
                eligible,
                key=lambda item: (
                    direction_counts[item["direction"]],
                    -item["quality"],
                    stable_hash(item["entry_id"], salt),
                ),
            )
            picked.append(chosen)
            source_candidates.remove(chosen)
            used_concepts.add(chosen["concept_key"])
            used_signatures.add(exact_signature(chosen))
            direction_counts[chosen["direction"]] += 1
        selected.extend(picked)
    return selected


def export_training_seed(item: dict[str, Any], salt: str) -> dict[str, Any]:
    entry = item["entry"]
    payload_json = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result: dict[str, Any] = {
        "id": f"aug_seed_{stable_hash(item['entry_id'], salt)[:16]}",
        "source_seed_id": f"entry::{item['entry_id']}",
        "source_entry_id": item["entry_id"],
        "source_payload_hash": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "source_type": item["source_type"],
        "source_book": entry.get("source_book", ""),
        "source_file": entry.get("source_file", ""),
        "direction": item["direction"],
        "query": build_seed_query(entry, item["groups"]),
        "expected_term_groups": item["groups"],
        "expected_negative_terms": [],
        "forbidden_terms": item["forbidden_terms"],
        "expected_entry_id_in_top_k": item["entry_id"],
        "expected_source_type_in_top_k": item["source_type"],
        "expected_gate": True,
        "review_status": "auto_extracted_training_seed_from_local_payload",
        "source_evidence": str(entry.get("evidence", ""))[:1200],
    }
    formula = str(entry.get("formula", "")).strip()
    if formula:
        result["expected_formula_in_top_k"] = formula
    intervention = str(entry.get("herb_name") or entry.get("intervention_name") or "").strip()
    if intervention and intervention not in {"针刺/取穴相关条文", "脉诊/诊法理论"}:
        result["expected_intervention_text"] = intervention
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract evidence-grounded training seeds disjoint from all eval sets")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--salt", default="query-translator-augmented-v3-20260630")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_path = args.output_dir / "train_seeds.jsonl"
    if seed_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to rebuild: {seed_path}")
    if sum(SOURCE_QUOTAS.values()) != 300:
        raise AssertionError("training source quotas must sum to 300")
    old_usage = collect_old_usage(args.eval_root, args.output_dir)
    raw_entries = read_jsonl(args.dictionary)
    candidates = [candidate_from_entry(entry, old_usage) for entry in raw_entries]
    candidates = [candidate for candidate in candidates if candidate]
    selected = select_training_candidates(candidates, SOURCE_QUOTAS, args.salt)
    seeds = [export_training_seed(item, args.salt) for item in selected]
    seeds.sort(key=lambda seed: stable_hash(seed["id"], args.salt))
    entry_ids = [str(seed["source_entry_id"]) for seed in seeds]
    if len(seeds) != 300 or len(set(entry_ids)) != 300:
        raise AssertionError("training seed count or entry uniqueness failed")
    if set(entry_ids) & old_usage["entry_ids"]:
        raise AssertionError("training seeds overlap an evaluation entry_id")
    write_jsonl(seed_path, seeds)
    manifest = {
        "status": "training_seeds_ready_pending_generation",
        "count": len(seeds),
        "salt": args.salt,
        "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        "source_type_counts": dict(sorted(Counter(seed["source_type"] for seed in seeds).items())),
        "direction_counts": dict(sorted(Counter(seed["direction"] for seed in seeds).items())),
        "unique_entry_ids": len(set(entry_ids)),
        "eval_entry_id_overlap": 0,
        "old_eval_entry_ids_excluded": len(old_usage["entry_ids"]),
        "planned_generation_styles": ["metaphor_incomplete", "asr_homophone_typos"],
        "planned_questions_per_style": 600,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
