from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = ROOT / "datasets" / "external" / "processed"
DEFAULT_LABEL_POOL = ROOT / "datasets" / "structured" / "query_translator_evidence_label_pool.jsonl"
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "supervision"
DEFAULT_REPORT = ROOT / "datasets" / "external" / "reports" / "supervision_build_report.json"

NEGATION_PATTERN = re.compile(r"(?:没有|并没有|没|不|无|未|否认|并非|不是).{0,2}$")
UNCERTAINTY_PATTERN = re.compile(r"(?:好像|可能|似乎|不确定|说不准|大概).{0,4}$")
HARD_NEGATIVE_TERM_ROLES = {
    "ancient_symptom",
    "diagnostic_key",
    "required_symptom",
    "syndrome_name",
    "theory_term",
    "acupuncture_term",
    "diagnostic_method",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


class ExactTermMatcher:
    def __init__(self, terms: Iterable[str]) -> None:
        by_first: dict[str, set[str]] = defaultdict(set)
        for term in terms:
            value = str(term).strip()
            if value:
                by_first[value[0]].add(value)
        self.by_first = {
            first: sorted(values, key=lambda value: (-len(value), value))
            for first, values in by_first.items()
        }

    def find(self, text: str) -> list[tuple[int, int, str]]:
        candidates: list[tuple[int, int, str]] = []
        for start, character in enumerate(text):
            for term in self.by_first.get(character, ()):
                if text.startswith(term, start):
                    candidates.append((start, start + len(term), term))
        selected: list[tuple[int, int, str]] = []
        occupied: set[int] = set()
        for start, end, term in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
            positions = set(range(start, end))
            if positions & occupied:
                continue
            occupied.update(positions)
            selected.append((start, end, term))
        return sorted(selected)


def detect_polarity(text: str, start: int) -> str:
    context = text[max(0, start - 10) : start]
    if NEGATION_PATTERN.search(context):
        return "absent"
    if UNCERTAINTY_PATTERN.search(context):
        return "uncertain"
    return "present"


def load_label_pool(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    high: dict[str, dict[str, Any]] = {}
    medium: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        term = str(record.get("canonical_term", "")).strip()
        if not term:
            continue
        target = high if record.get("eligible_as_gold_label") else medium
        item = target.setdefault(
            term,
            {
                "entry_ids": set(),
                "source_types": set(),
                "source_books": set(),
                "evidence_tiers": set(),
                "term_roles": set(),
            },
        )
        item["entry_ids"].add(str(record.get("entry_id", "")))
        item["source_types"].add(str(record.get("source_type", "")))
        item["source_books"].add(str(record.get("source_book", "")))
        item["evidence_tiers"].add(str(record.get("evidence_tier", "")))
        item["term_roles"].update(str(role) for role in record.get("term_roles") or [] if str(role))
    return high, medium


def serialize_label_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {key: sorted(value) if isinstance(value, set) else value for key, value in item.items()}


def build_query_mappings(
    processed_root: Path,
    high_labels: dict[str, dict[str, Any]],
    medium_labels: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    matcher = ExactTermMatcher(set(high_labels) | set(medium_labels))
    mapped: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    development_root = processed_root / "questions" / "development"
    for path in sorted(development_root.glob("*.jsonl")):
        for question in iter_jsonl(path):
            counts["development_questions"] += 1
            text = str(question.get("text", ""))
            high_mappings: list[dict[str, Any]] = []
            medium_candidates: list[dict[str, Any]] = []
            for start, end, term in matcher.find(text):
                metadata = high_labels.get(term)
                mapping = {
                    "source_phrase": text[start:end],
                    "source_start": start,
                    "source_end": end,
                    "canonical_term": term,
                    "polarity": detect_polarity(text, start),
                    "confidence": 1.0 if metadata else 0.6,
                    "evidence_entry_ids": sorted((metadata or medium_labels[term])["entry_ids"]),
                    "evidence_tier": "high" if metadata else "medium",
                }
                if metadata:
                    high_mappings.append(mapping)
                else:
                    medium_candidates.append(mapping)
            base = {
                "query_id": question.get("record_id"),
                "query": text,
                "source_dataset": question.get("source_dataset"),
                "source_id": question.get("source_id"),
                "source_hash": question.get("source_hash"),
            }
            if high_mappings:
                mapped.append(
                    {
                        **base,
                        "mappings": high_mappings,
                        "unknown_phrases": [],
                        "supervision_confidence": "high_exact_evidence_match",
                    }
                )
                counts["high_confidence_questions"] += 1
                counts["high_confidence_mappings"] += len(high_mappings)
            else:
                rejected.append(
                    {
                        **base,
                        "candidate_mappings": medium_candidates,
                        "unknown_phrases": [text] if not medium_candidates else [],
                        "reason": "medium_evidence_only" if medium_candidates else "no_evidence_grounded_exact_match",
                    }
                )
                counts["medium_candidate_questions" if medium_candidates else "unknown_questions"] += 1
    return mapped, rejected, counts


def classify_cblue_pair(record: dict[str, Any]) -> tuple[str, bool | None]:
    source_file = str(record.get("source_file", ""))
    label = str(record.get("label", ""))
    if "CHIP-STS" in source_file:
        return "binary_semantic_similarity", label == "1"
    if "KUAKE-QQR" in source_file:
        grade = int(label) if label.isdigit() else -1
        return "graded_query_similarity", True if grade == 2 else False if grade == 0 else None
    if "KUAKE-QTR" in source_file:
        grade = int(label) if label.isdigit() else -1
        return "graded_query_title_relevance", True if grade == 3 else False if grade == 0 else None
    return "unknown", None


def build_semantic_pairs(processed_root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    path = processed_root / "pairs" / "cblue_semantic_pairs.jsonl"
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    if not path.exists():
        return records, counts
    for pair in iter_jsonl(path):
        relation_type, positive = classify_cblue_pair(pair)
        record = {
            **pair,
            "relation_type": relation_type,
            "is_positive": positive,
            "training_policy": "language_similarity_only_not_tcm_fact",
        }
        records.append(record)
        counts["semantic_pairs"] += 1
        counts["semantic_positive" if positive is True else "semantic_negative" if positive is False else "semantic_ambiguous"] += 1
    return records, counts


def intervention_name(entry: dict[str, Any]) -> str:
    for field in ("intervention_name", "formula", "herb_name", "theory_topic", "title"):
        value = str(entry.get(field, "")).strip()
        if value:
            return value
    return str(entry.get("entry_id", ""))


def required_coverage_for_terms(entry: dict[str, Any], terms: set[str]) -> float:
    groups = []
    for raw_group in entry.get("required_symptom_groups") or []:
        group = {str(term) for term in (raw_group if isinstance(raw_group, list) else [raw_group]) if str(term)}
        if group:
            groups.append(group)
    if not groups:
        return 0.0
    return sum(bool(group & terms) for group in groups) / len(groups)


def confidence_value(entry: dict[str, Any]) -> float:
    try:
        return float(entry.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def should_generate_clarify_pair(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    shared_terms: set[str],
) -> bool:
    anchor_forbidden = {str(term) for term in anchor.get("forbidden_terms") or []}
    candidate_forbidden = {str(term) for term in candidate.get("forbidden_terms") or []}
    if shared_terms & (anchor_forbidden | candidate_forbidden):
        return False
    anchor_required = required_coverage_for_terms(anchor, shared_terms)
    candidate_required = required_coverage_for_terms(candidate, shared_terms)
    return (
        abs(anchor_required - candidate_required) <= 0.25
        and abs(confidence_value(anchor) - confidence_value(candidate)) <= 0.15
    )


def build_local_hard_negatives(
    dictionary_path: Path,
    high_labels: dict[str, dict[str, Any]],
    medium_labels: dict[str, dict[str, Any]],
    per_entry_limit: int = 5,
) -> list[dict[str, Any]]:
    entries = {str(entry.get("entry_id")): entry for entry in iter_jsonl(dictionary_path)}
    entry_terms: dict[str, set[str]] = defaultdict(set)
    entry_tier: dict[str, str] = {}
    for tier, labels in (("high", high_labels), ("medium", medium_labels)):
        for term, metadata in labels.items():
            roles = set(metadata.get("term_roles") or [])
            if roles and not roles.intersection(HARD_NEGATIVE_TERM_ROLES):
                continue
            for entry_id in metadata["entry_ids"]:
                entry_terms[entry_id].add(term)
                if tier == "high" or entry_id not in entry_tier:
                    entry_tier[entry_id] = tier
    inverted: dict[str, set[str]] = defaultdict(set)
    for entry_id, terms in entry_terms.items():
        for term in terms:
            inverted[term].add(entry_id)

    results: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor_id, anchor_terms in sorted(entry_terms.items()):
        anchor = entries.get(anchor_id)
        if not anchor:
            continue
        candidate_ids = set().union(*(inverted[term] for term in anchor_terms)) - {anchor_id}
        scored: list[tuple[float, str, set[str]]] = []
        for candidate_id in candidate_ids:
            candidate = entries.get(candidate_id)
            if not candidate or candidate.get("source_type") != anchor.get("source_type"):
                continue
            if intervention_name(candidate) == intervention_name(anchor):
                continue
            candidate_terms = entry_terms[candidate_id]
            shared = anchor_terms & candidate_terms
            union = anchor_terms | candidate_terms
            score = len(shared) / len(union) if union else 0.0
            if shared and score >= 0.08:
                scored.append((score, candidate_id, shared))
        for score, candidate_id, shared in sorted(scored, key=lambda item: (-item[0], item[1]))[:per_entry_limit]:
            pair_key = tuple(sorted((anchor_id, candidate_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            candidate = entries[candidate_id]
            anchor_only = anchor_terms - entry_terms[candidate_id]
            candidate_only = entry_terms[candidate_id] - anchor_terms
            query_terms = sorted(shared, key=lambda term: (-len(term), term))[:4]
            query_terms.extend(
                term
                for term in sorted(anchor_only, key=lambda term: (-len(term), term))
                if term not in query_terms
            )
            query_terms = query_terms[:8]
            base_record = {
                "type": "local_differential",
                "anchor_entry_id": anchor_id,
                "candidate_entry_id": candidate_id,
                "source_type": anchor.get("source_type"),
                "anchor_intervention": intervention_name(anchor),
                "candidate_intervention": intervention_name(candidate),
                "shared_terms": sorted(shared),
                "anchor_only_terms": sorted(anchor_only),
                "candidate_only_terms": sorted(candidate_only),
                "term_jaccard": round(score, 6),
                "anchor_evidence_tier": entry_tier.get(anchor_id, "medium"),
                "candidate_evidence_tier": entry_tier.get(candidate_id, "medium"),
                "policy": "ranking_only_never_overrides_required_or_forbidden_evidence",
            }
            if anchor_only:
                results.append(
                    {
                        **base_record,
                        "negative_id": f"local::{pair_key[0]}::{pair_key[1]}::rank",
                        "query_terms": query_terms,
                        "expected_decision": "rank_anchor",
                    }
                )
            if should_generate_clarify_pair(anchor, candidate, shared):
                results.append(
                    {
                        **base_record,
                        "negative_id": f"local::{pair_key[0]}::{pair_key[1]}::clarify",
                        "query_terms": sorted(shared, key=lambda term: (-len(term), term))[:8],
                        "expected_decision": "clarify",
                    }
                )
    return results


def run_build(
    processed_root: Path,
    label_pool_path: Path,
    dictionary_path: Path,
    output_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    high_labels, medium_labels = load_label_pool(label_pool_path)
    mappings, rejected, mapping_counts = build_query_mappings(processed_root, high_labels, medium_labels)
    semantic_pairs, pair_counts = build_semantic_pairs(processed_root)
    hard_negatives = build_local_hard_negatives(dictionary_path, high_labels, medium_labels)

    counts = {
        **mapping_counts,
        **pair_counts,
        "unique_high_terms": len(high_labels),
        "unique_medium_terms": len(medium_labels),
        "local_hard_negatives": len(hard_negatives),
    }
    output_counts = {
        "query_term_pairs": write_jsonl(output_root / "query_term_pairs.jsonl", mappings),
        "rejected_candidates": write_jsonl(output_root / "rejected_candidates.jsonl", rejected),
        "semantic_pair_supervision": write_jsonl(output_root / "semantic_pair_supervision.jsonl", semantic_pairs),
        "hard_negatives": write_jsonl(output_root / "hard_negatives.jsonl", hard_negatives),
    }
    report = {
        "report_version": 1,
        "counts": dict(sorted(counts.items())),
        "outputs": output_counts,
        "policies": {
            "query_term_gold": "exact span match to high evidence label only",
            "medium_terms": "candidate queue only",
            "cblue": "language similarity supervision only, never TCM factual supervision",
            "hard_negatives": "generated across every eligible same-source-type entry with shared evidence terms",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence-constrained Query Translator supervision")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--label-pool", type=Path, default=DEFAULT_LABEL_POOL)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_build(args.processed_root, args.label_pool, args.dictionary, args.output_root, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
