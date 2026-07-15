from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = ROOT / "datasets" / "external" / "query_translator_augmented_v3" / "train_seeds.jsonl"
DEFAULT_REFERENCE_ROOT = ROOT / "tests" / "evals"
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "external" / "query_translator_augmented_v3" / "generated" / "local_asr_gold.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def collect_confusions(reference_root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    phrase_map: dict[str, set[str]] = defaultdict(set)
    char_map: dict[str, set[str]] = defaultdict(set)
    pair_count = 0
    for path in reference_root.rglob("*.jsonl"):
        try:
            records = iter_jsonl(path)
            for record in records:
                for pair in record.get("typo_pairs") or []:
                    if not isinstance(pair, dict):
                        continue
                    correct = str(pair.get("correct", "")).strip()
                    typo = str(pair.get("typo", "")).strip()
                    if not correct or not typo or correct == typo:
                        continue
                    phrase_map[correct].add(typo)
                    pair_count += 1
                    if len(correct) == len(typo):
                        for left, right in zip(correct, typo):
                            if left != right:
                                char_map[left].add(right)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    return (
        {key: sorted(values) for key, values in phrase_map.items()},
        {key: sorted(values) for key, values in char_map.items()},
        pair_count,
    )


def mutate_surface(
    value: str,
    phrase_map: dict[str, list[str]],
    char_map: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, list[dict[str, str]]]:
    direct = phrase_map.get(value, [])
    if direct:
        typo = direct[rng.randrange(len(direct))]
        return typo, [{"correct": value, "typo": typo}]
    positions = [index for index, char in enumerate(value) if char in char_map]
    if not positions:
        return value, []
    index = positions[rng.randrange(len(positions))]
    chars = list(value)
    replacement = char_map[chars[index]][rng.randrange(len(char_map[chars[index]]))]
    correct = chars[index]
    chars[index] = replacement
    return "".join(chars), [{"correct": correct, "typo": replacement}]


def build_record(
    seed: dict[str, Any],
    variant: int,
    phrase_map: dict[str, list[str]],
    char_map: dict[str, list[str]],
    salt: str,
) -> dict[str, Any] | None:
    rng = random.Random(int(hashlib.sha256(f"{salt}|{seed['id']}|{variant}".encode()).hexdigest()[:16], 16))
    surfaces: list[str] = []
    mappings: list[dict[str, str]] = []
    typo_pairs: list[dict[str, str]] = []
    groups = seed.get("expected_term_groups") or []
    for group in groups:
        canonical = next((str(term).strip() for term in group if str(term).strip()), "")
        if not canonical:
            continue
        surface, mutations = mutate_surface(canonical, phrase_map, char_map, rng)
        surfaces.append(surface)
        mappings.append({"source_phrase": surface, "canonical_term": canonical, "polarity": "present"})
        typo_pairs.extend(mutations)
    if not surfaces or not typo_pairs:
        return None
    filler_variants = (
        "我最近{symptoms}这些情况是怎么回事",
        "这几天老是{symptoms}想问问可能对应什么情况",
        "身上出现{symptoms}这种表现该怎么看",
    )
    query = filler_variants[variant % len(filler_variants)].format(symptoms="还会".join(surfaces))
    identity = hashlib.sha256(f"{seed['id']}|asr|{variant}|{query}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"qt_local_asr_{identity}",
        "query": query,
        "source_seed_id": seed["id"],
        "generation_style": "asr_homophone_typos",
        "expected_term_groups": groups,
        "expected_negative_terms": [],
        "forbidden_terms": seed.get("forbidden_terms", []),
        "must_clarify": False,
        "expected_decision": None,
        "omitted_term_group": [],
        "typo_pairs": typo_pairs,
        "evidence_mappings": mappings,
        "review_status": "local_confusion_map_structurally_validated",
        "generation": {
            "provider": "local_existing_asr_confusion_pairs",
            "model": "deterministic",
            "variant_index": variant,
        },
        "expected_entry_id_in_top_k": seed.get("expected_entry_id_in_top_k", ""),
        "expected_formula_in_top_k": seed.get("expected_formula_in_top_k", ""),
        "expected_source_type_in_top_k": seed.get("expected_source_type_in_top_k", ""),
        "expected_gate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ASR typo augmentation from existing validated typo pairs")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--salt", default="local-asr-augmentation-v1")
    args = parser.parse_args()
    seeds = list(iter_jsonl(args.seeds))
    phrase_map, char_map, source_pair_count = collect_confusions(args.reference_root)
    if not phrase_map or not char_map:
        raise ValueError("no validated ASR confusion pairs found")
    records: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    variant = 0
    max_attempts = max(args.count * 20, 1000)
    while len(records) < args.count and variant < max_attempts:
        seed = seeds[variant % len(seeds)]
        record = build_record(seed, variant // len(seeds), phrase_map, char_map, args.salt)
        variant += 1
        if not record or record["query"] in seen_queries:
            continue
        seen_queries.add(record["query"])
        records.append(record)
    if len(records) != args.count:
        raise RuntimeError(f"generated only {len(records)}/{args.count} unique ASR cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "count": len(records),
        "unique_queries": len(seen_queries),
        "source_validated_pair_count": source_pair_count,
        "unique_phrase_confusions": sum(len(values) for values in phrase_map.values()),
        "unique_character_sources": len(char_map),
        "output": str(args.output),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
