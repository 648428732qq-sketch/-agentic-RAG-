from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from extract_chatmed_field_candidates import (
        DEFAULT_CHATMED,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYNDROME_DICTIONARY,
        EVIDENCE_ORIGIN,
        ROOT,
        TermHit,
        display_path,
        extract_from_record,
        iter_jsonl,
        load_local_term_index,
        stable_hash,
        write_jsonl,
    )
except ModuleNotFoundError:
    from scripts.extract_chatmed_field_candidates import (
        DEFAULT_CHATMED,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYNDROME_DICTIONARY,
        EVIDENCE_ORIGIN,
        ROOT,
        TermHit,
        display_path,
        extract_from_record,
        iter_jsonl,
        load_local_term_index,
        stable_hash,
        write_jsonl,
    )


FILTERED_STATUS = "filtered_candidate_unverified"
FILTERED_ORIGIN = f"{EVIDENCE_ORIGIN}+local_syndrome_dictionary"
GENERIC_SYMPTOM_TERMS = {
    "疼痛",
    "头晕",
    "头痛",
    "头疼",
    "腹痛",
    "肚子疼",
    "心痛",
    "胸痛",
    "咳嗽",
    "发热",
    "发烧",
    "恶心",
    "呕吐",
    "失眠",
    "腹泻",
    "便秘",
    "口干",
    "口渴",
    "胸闷",
    "乏力",
    "无力",
    "身痛",
    "腰痛",
    "胃痛",
    "胃疼",
    "腹胀",
    "腹满",
    "水肿",
    "浮肿",
    "烦躁",
}


@dataclass
class Aggregate:
    term: str
    entry_id: str = ""
    title: str = ""
    source_type: str = ""
    formula: str = ""
    herb_name: str = ""
    query_hashes: set[str] = field(default_factory=set)
    source_lines: set[int] = field(default_factory=set)
    field_counts: Counter[str] = field(default_factory=Counter)
    source_type_counts: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def query_count(self) -> int:
        return len(self.query_hashes)

    def add(self, *, record: dict[str, Any], hit: TermHit | dict[str, Any], sample_limit: int) -> None:
        query_hash = str(record["query_hash"])
        self.query_hashes.add(query_hash)
        self.source_lines.add(int(record["source_line"]))
        if isinstance(hit, TermHit):
            field = hit.field
            source_type = hit.source_type
        else:
            field = str(hit.get("field", ""))
            source_type = str(hit.get("source_type", ""))
        if field:
            self.field_counts[field] += 1
        if source_type:
            self.source_type_counts[source_type] += 1
        if len(self.samples) < sample_limit and query_hash not in {str(item["query_hash"]) for item in self.samples}:
            self.samples.append(
                {
                    "source_line": record["source_line"],
                    "query_hash": query_hash,
                    "query": record["query"],
                }
            )


def term_local_stats(term_index: Any, term: str) -> dict[str, Any]:
    diagnostic_hits = term_index.diagnostic_terms.get(term, [])
    pathogenesis_hits = term_index.pathogenesis_terms.get(term, [])
    hint_hits = term_index.source_hint_terms.get(term, [])
    all_hits = diagnostic_hits + pathogenesis_hits + hint_hits
    entry_ids = {hit.entry_id for hit in all_hits if hit.entry_id}
    source_types = sorted({hit.source_type for hit in all_hits if hit.source_type})
    fields = sorted({hit.field for hit in all_hits if hit.field})
    return {
        "local_entry_count_for_term": len(entry_ids),
        "local_source_types_for_term": source_types,
        "local_fields_for_term": fields,
    }


def priority_for_term(
    *,
    term: str,
    query_count: int,
    local_entry_count: int,
    min_query_count: int,
    max_local_entries: int,
    candidate_type: str,
) -> tuple[str, list[str], str]:
    flags: list[str] = []
    if query_count < min_query_count:
        flags.append("sparse_query_support")
    if local_entry_count > max_local_entries:
        flags.append("broad_local_term")
    if term in GENERIC_SYMPTOM_TERMS:
        flags.append("generic_symptom")

    if candidate_type == "negation_forbidden":
        if query_count >= max(1, min_query_count) and local_entry_count <= max_local_entries * 4:
            return "high", flags, "review_for_negative_term_tests"
        return "medium", flags, "review_for_negative_term_tests_only"

    if "generic_symptom" not in flags and query_count >= min_query_count and local_entry_count <= max_local_entries:
        return "high", flags, "review_for_query_translator_term_candidate"
    if query_count >= min_query_count and local_entry_count <= max_local_entries * 3:
        return "medium", flags, "review_for_query_translator_term_candidate"
    if query_count >= min_query_count:
        return "broad_review_only", flags, "use_for_eval_coverage_not_auto_rules"
    return "low", flags, "hold_for_more_evidence"


def aggregate_hit(
    buckets: dict[tuple[str, str, str], Aggregate],
    *,
    record: dict[str, Any],
    hit: TermHit | dict[str, Any],
    sample_limit: int,
) -> None:
    term = str(hit.term if isinstance(hit, TermHit) else hit.get("term", ""))
    entry_id = str(hit.entry_id if isinstance(hit, TermHit) else hit.get("entry_id", ""))
    source_type = str(hit.source_type if isinstance(hit, TermHit) else hit.get("source_type", ""))
    key = (term, entry_id, source_type)
    aggregate = buckets.get(key)
    if aggregate is None:
        aggregate = Aggregate(
            term=term,
            entry_id=entry_id,
            title=str(hit.title if isinstance(hit, TermHit) else hit.get("title", "")),
            source_type=source_type,
            formula=str(hit.formula if isinstance(hit, TermHit) else hit.get("formula", "")),
            herb_name=str(hit.herb_name if isinstance(hit, TermHit) else hit.get("herb_name", "")),
        )
        buckets[key] = aggregate
    aggregate.add(record=record, hit=hit, sample_limit=sample_limit)


def aggregate_negation(
    buckets: dict[str, Aggregate],
    *,
    record: dict[str, Any],
    hit: TermHit | dict[str, Any],
    sample_limit: int,
) -> None:
    term = str(hit.term if isinstance(hit, TermHit) else hit.get("term", ""))
    aggregate = buckets.get(term)
    if aggregate is None:
        aggregate = Aggregate(term=term)
        buckets[term] = aggregate
    aggregate.add(record=record, hit=hit, sample_limit=sample_limit)


def build_entry_row(
    *,
    candidate_type: str,
    aggregate: Aggregate,
    term_index: Any,
    min_query_count: int,
    max_local_entries: int,
) -> dict[str, Any]:
    local_stats = term_local_stats(term_index, aggregate.term)
    priority, risk_flags, recommended_use = priority_for_term(
        term=aggregate.term,
        query_count=aggregate.query_count,
        local_entry_count=int(local_stats["local_entry_count_for_term"]),
        min_query_count=min_query_count,
        max_local_entries=max_local_entries,
        candidate_type=candidate_type,
    )
    return {
        "candidate_id": f"chatmed_filtered_{candidate_type}_{stable_hash('|'.join([aggregate.term, aggregate.entry_id, aggregate.source_type]), 20)}",
        "candidate_type": candidate_type,
        "term": aggregate.term,
        "entry_id": aggregate.entry_id,
        "title": aggregate.title,
        "source_type": aggregate.source_type,
        "formula": aggregate.formula,
        "herb_name": aggregate.herb_name,
        "query_count": aggregate.query_count,
        "source_line_count": len(aggregate.source_lines),
        "field_counts": dict(aggregate.field_counts),
        "source_type_counts": dict(aggregate.source_type_counts),
        **local_stats,
        "priority": priority,
        "risk_flags": risk_flags,
        "recommended_use": recommended_use,
        "review_status": FILTERED_STATUS,
        "evidence_origin": FILTERED_ORIGIN,
        "sample_queries": aggregate.samples,
    }


def build_negation_row(
    *,
    aggregate: Aggregate,
    term_index: Any,
    min_query_count: int,
    max_local_entries: int,
) -> dict[str, Any]:
    local_stats = term_local_stats(term_index, aggregate.term)
    priority, risk_flags, recommended_use = priority_for_term(
        term=aggregate.term,
        query_count=aggregate.query_count,
        local_entry_count=int(local_stats["local_entry_count_for_term"]),
        min_query_count=min_query_count,
        max_local_entries=max_local_entries,
        candidate_type="negation_forbidden",
    )
    return {
        "candidate_id": f"chatmed_filtered_negation_forbidden_{stable_hash(aggregate.term, 20)}",
        "candidate_type": "negation_forbidden",
        "term": aggregate.term,
        "query_count": aggregate.query_count,
        "source_line_count": len(aggregate.source_lines),
        "field_counts": dict(aggregate.field_counts),
        "source_type_counts": dict(aggregate.source_type_counts),
        **local_stats,
        "priority": priority,
        "risk_flags": risk_flags,
        "recommended_use": recommended_use,
        "review_status": FILTERED_STATUS,
        "evidence_origin": FILTERED_ORIGIN,
        "sample_queries": aggregate.samples,
    }


def sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_rank = {"high": 0, "medium": 1, "broad_review_only": 2, "low": 3}
    return sorted(
        rows,
        key=lambda row: (
            priority_rank.get(str(row.get("priority", "low")), 9),
            -int(row.get("query_count", 0)),
            int(row.get("local_entry_count_for_term", 0)),
            str(row.get("term", "")),
        ),
    )


def cap_rows(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if cap and len(rows) > cap:
        return rows[:cap]
    return rows


def filter_candidates(
    *,
    chatmed_path: Path,
    syndrome_dictionary_path: Path,
    output_dir: Path,
    limit: int = 0,
    min_query_count: int = 3,
    max_local_entries: int = 8,
    max_rows_per_file: int = 5000,
    sample_limit: int = 5,
    max_terms_per_query: int = 80,
) -> dict[str, Any]:
    term_index = load_local_term_index(syndrome_dictionary_path)
    diagnostic: dict[tuple[str, str, str], Aggregate] = {}
    pathogenesis: dict[tuple[str, str, str], Aggregate] = {}
    source_hints: dict[tuple[str, str, str], Aggregate] = {}
    negation: dict[str, Aggregate] = {}
    counters: Counter[str] = Counter()

    for source_line, raw_record in iter_jsonl(chatmed_path, limit=limit):
        counters["records_seen"] += 1
        record = extract_from_record(
            source_line=source_line,
            raw_record=raw_record,
            term_index=term_index,
            max_terms_per_query=max_terms_per_query,
        )
        if record is None:
            counters["skipped_empty_or_pii"] += 1
            continue

        query_seen: set[tuple[str, str, str, str]] = set()
        for hit in record["diagnostic_matches"]:
            key = (hit["term"], hit["entry_id"], hit["source_type"], "diagnostic")
            if key in query_seen:
                continue
            query_seen.add(key)
            aggregate_hit(diagnostic, record=record, hit=hit, sample_limit=sample_limit)
        for hit in record["pathogenesis_matches"]:
            key = (hit["term"], hit["entry_id"], hit["source_type"], "pathogenesis")
            if key in query_seen:
                continue
            query_seen.add(key)
            aggregate_hit(pathogenesis, record=record, hit=hit, sample_limit=sample_limit)
        for hit in record["source_hint_matches"]:
            key = (hit["term"], hit["entry_id"], hit["source_type"], "source_hint")
            if key in query_seen:
                continue
            query_seen.add(key)
            aggregate_hit(source_hints, record=record, hit=hit, sample_limit=sample_limit)
        for hit in record["negated_matches"]:
            aggregate_negation(negation, record=record, hit=hit, sample_limit=sample_limit)

    diagnostic_rows = sorted_rows(
        build_entry_row(
            candidate_type="diagnostic_term_entry",
            aggregate=aggregate,
            term_index=term_index,
            min_query_count=min_query_count,
            max_local_entries=max_local_entries,
        )
        for aggregate in diagnostic.values()
        if aggregate.query_count >= min_query_count
    )
    pathogenesis_rows = sorted_rows(
        build_entry_row(
            candidate_type="pathogenesis_query_entry",
            aggregate=aggregate,
            term_index=term_index,
            min_query_count=max(1, min_query_count),
            max_local_entries=max_local_entries,
        )
        for aggregate in pathogenesis.values()
        if aggregate.query_count >= max(1, min_query_count - 1)
    )
    source_hint_rows = sorted_rows(
        build_entry_row(
            candidate_type="source_hint_entry",
            aggregate=aggregate,
            term_index=term_index,
            min_query_count=min_query_count,
            max_local_entries=max_local_entries,
        )
        for aggregate in source_hints.values()
        if aggregate.query_count >= min_query_count
    )
    negation_rows = sorted_rows(
        build_negation_row(
            aggregate=aggregate,
            term_index=term_index,
            min_query_count=1,
            max_local_entries=max_local_entries,
        )
        for aggregate in negation.values()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "diagnostic_term_review_queue": output_dir / "diagnostic_term_review_queue.jsonl",
        "pathogenesis_query_review_queue": output_dir / "pathogenesis_query_review_queue.jsonl",
        "source_hint_review_queue": output_dir / "source_hint_review_queue.jsonl",
        "negation_forbidden_review_queue": output_dir / "negation_forbidden_review_queue.jsonl",
    }
    row_sets = {
        "diagnostic_term_review_queue": diagnostic_rows,
        "pathogenesis_query_review_queue": pathogenesis_rows,
        "source_hint_review_queue": source_hint_rows,
        "negation_forbidden_review_queue": negation_rows,
    }

    output_files: dict[str, Any] = {}
    for name, rows in row_sets.items():
        written_rows = cap_rows(rows, max_rows_per_file)
        written = write_jsonl(files[name], written_rows)
        output_files[name] = {
            "path": display_path(files[name]),
            "written": written,
            "seen": len(rows),
            "truncated": bool(max_rows_per_file and len(rows) > max_rows_per_file),
            "priority_counts": dict(Counter(str(row["priority"]) for row in rows)),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": display_path(chatmed_path),
        "syndrome_dictionary": display_path(syndrome_dictionary_path),
        "records_seen": counters["records_seen"],
        "skipped_empty_or_pii": counters["skipped_empty_or_pii"],
        "review_status": FILTERED_STATUS,
        "evidence_origin": FILTERED_ORIGIN,
        "thresholds": {
            "min_query_count": min_query_count,
            "max_local_entries": max_local_entries,
            "max_rows_per_file": max_rows_per_file,
            "sample_limit": sample_limit,
        },
        "aggregate_counts": {
            "diagnostic_term_entry_total": len(diagnostic),
            "pathogenesis_query_entry_total": len(pathogenesis),
            "source_hint_entry_total": len(source_hints),
            "negation_forbidden_term_total": len(negation),
        },
        "output_files": output_files,
        "safety_note": (
            "Rows are filtered candidates only. They are not approved knowledge, not Qdrant payloads, "
            "and must not be auto-merged into syndrome_dictionary without separate review."
        ),
    }
    summary_path = output_dir / "chatmed_candidate_filter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="聚合筛选 ChatMed query 侧候选，生成二次 review queue。")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHATMED)
    parser.add_argument("--syndrome-dictionary", type=Path, default=DEFAULT_SYNDROME_DICTIONARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "filtered")
    parser.add_argument("--limit", type=int, default=0, help="最多读取多少行；0 表示全量")
    parser.add_argument("--min-query-count", type=int, default=3)
    parser.add_argument("--max-local-entries", type=int, default=8)
    parser.add_argument("--max-rows-per-file", type=int, default=5000)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--max-terms-per-query", type=int, default=80)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = filter_candidates(
        chatmed_path=args.input,
        syndrome_dictionary_path=args.syndrome_dictionary,
        output_dir=args.output_dir,
        limit=args.limit,
        min_query_count=args.min_query_count,
        max_local_entries=args.max_local_entries,
        max_rows_per_file=args.max_rows_per_file,
        sample_limit=args.sample_limit,
        max_terms_per_query=args.max_terms_per_query,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
