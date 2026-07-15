from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHATMED = ROOT / "ChatMed_TCM-v0.2.json"
DEFAULT_SYNDROME_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "structured" / "chatmed_candidates"

QUERY_CANDIDATE_STATUS = "candidate_unverified"
EVIDENCE_ORIGIN = "chatmed_query_only"

DIAGNOSTIC_FIELDS = (
    "diagnostic_keys",
    "ancient_symptoms",
    "modern_symptoms",
    "symptom_aliases",
)
PATHOGENESIS_FIELDS = ("pathogenesis",)
SOURCE_HINT_FIELDS = (
    "formula",
    "herb_name",
    "herb_aliases",
    "theory_terms",
    "acupoints_or_channels",
    "treatment_method",
    "treatment_principle",
)

NEGATION_PREFIX_RE = re.compile(r"(?:没有|没|无|未|不见|否认|并无|并没有|不是|不)")
META_SUFFIX_RE = re.compile(
    r"(?:[。；;\s]*)要求\s*[:：].*$|"
    r"(?:[。；;\s]*(?:\d+[.、]\s*)?请[^。；;]*推理过程.*$)|"
    r"(?:[。；;\s]*)(?:请)?(?:根据输出)?(?:一步步地?)?输出(?:详细的?)?推理过程.*$|"
    r"(?:[。；;\s]*)请考虑所有症状.*$",
    flags=re.DOTALL,
)
PII_RE = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?!\d)|"
    r"(?<!\d)\d{17}[0-9Xx](?!\w)|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"(?:姓名|身份证|手机号|电话|住址)\s*[:：]"
)

STOP_TERMS = {
    "中药",
    "方剂",
    "药方",
    "治疗",
    "症状",
    "患者",
    "推荐",
    "如何",
    "什么",
    "怎么",
    "可以",
    "需要",
    "使用",
    "主治",
    "功效",
    "用法",
    "禁忌",
    "组成",
    "疾病",
    "医学",
    "医生",
    "医院",
    "脉",
    "浮",
    "数",
    "迟",
}

SOURCE_TYPE_KEYWORDS = {
    "formula_knowledge": ("方剂", "方子", "药方", "组成", "配伍", "君臣佐使"),
    "herb_indication": ("中药", "药材", "草药", "本草", "药性", "性味"),
    "classical_theory": ("古籍", "原文", "条文", "黄帝内经", "素问", "灵枢", "难经", "伤寒论", "金匮"),
    "classical_acupuncture_principle": ("针灸", "针刺", "穴位", "经络", "腧穴"),
}


@dataclass(frozen=True)
class TermHit:
    term: str
    matched_text: str
    field: str
    entry_id: str
    title: str
    source_type: str
    formula: str = ""
    herb_name: str = ""


@dataclass
class LocalTermIndex:
    diagnostic_terms: dict[str, list[TermHit]] = field(default_factory=dict)
    pathogenesis_terms: dict[str, list[TermHit]] = field(default_factory=dict)
    source_hint_terms: dict[str, list[TermHit]] = field(default_factory=dict)
    diagnostic_buckets: dict[str, list[str]] = field(default_factory=dict)
    pathogenesis_buckets: dict[str, list[str]] = field(default_factory=dict)
    source_hint_buckets: dict[str, list[str]] = field(default_factory=dict)
    entry_count: int = 0


def clean_query(text: str) -> str:
    text = str(text or "").strip()
    text = META_SUFFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" 。；;")


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_term(term: Any) -> str:
    value = str(term or "").strip()
    value = re.sub(r"\s+", "", value)
    value = value.strip("，,。；;：:、（）()[]【】\"'")
    return value


def usable_term(term: str) -> bool:
    if len(term) < 2:
        return False
    if term in STOP_TERMS:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", term):
        return False
    if len(term) > 24:
        return False
    return True


def iter_list_values(value: Any) -> Iterator[str]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ("name", "term", "title"):
                    if item.get(key):
                        yield normalize_term(item[key])
                        break
            else:
                yield normalize_term(item)
    elif isinstance(value, str):
        yield normalize_term(value)


def add_term(mapping: dict[str, list[TermHit]], term: str, hit: TermHit) -> None:
    if not usable_term(term):
        return
    hits = mapping.setdefault(term, [])
    if not any(existing.entry_id == hit.entry_id and existing.field == hit.field for existing in hits):
        hits.append(hit)


def build_term_buckets(mapping: dict[str, list[TermHit]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for term in mapping:
        buckets[term[0]].append(term)
    return {
        key: sorted(terms, key=lambda value: (-len(value), value))
        for key, terms in buckets.items()
    }


def load_local_term_index(path: Path) -> LocalTermIndex:
    index = LocalTermIndex()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            entry = json.loads(raw_line)
            index.entry_count += 1
            base = {
                "entry_id": str(entry.get("entry_id", "")),
                "title": str(entry.get("title", "")),
                "source_type": str(entry.get("source_type", "")),
                "formula": str(entry.get("formula", "")),
                "herb_name": str(entry.get("herb_name", "")),
            }
            for field_name in DIAGNOSTIC_FIELDS:
                for term in iter_list_values(entry.get(field_name)):
                    add_term(
                        index.diagnostic_terms,
                        term,
                        TermHit(term=term, matched_text=term, field=field_name, **base),
                    )
            for field_name in PATHOGENESIS_FIELDS:
                for term in iter_list_values(entry.get(field_name)):
                    add_term(
                        index.pathogenesis_terms,
                        term,
                        TermHit(term=term, matched_text=term, field=field_name, **base),
                    )
            for field_name in SOURCE_HINT_FIELDS:
                for term in iter_list_values(entry.get(field_name)):
                    add_term(
                        index.source_hint_terms,
                        term,
                        TermHit(term=term, matched_text=term, field=field_name, **base),
                    )
    index.diagnostic_buckets = build_term_buckets(index.diagnostic_terms)
    index.pathogenesis_buckets = build_term_buckets(index.pathogenesis_terms)
    index.source_hint_buckets = build_term_buckets(index.source_hint_terms)
    return index


def match_terms(
    query: str,
    mapping: dict[str, list[TermHit]],
    buckets: dict[str, list[str]],
    max_terms: int = 80,
) -> list[TermHit]:
    hits: list[TermHit] = []
    seen: set[tuple[str, str, str]] = set()
    seen_terms: set[str] = set()
    candidate_terms: list[str] = []
    for char in dict.fromkeys(query):
        for term in buckets.get(char, []):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            candidate_terms.append(term)
    candidate_terms.sort(key=lambda value: (-len(value), value))
    for term in candidate_terms:
        if term not in query:
            continue
        for hit in mapping[term]:
            key = (hit.term, hit.field, hit.entry_id)
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
            if len(hits) >= max_terms:
                return hits
    return hits


def is_negated(query: str, term: str) -> bool:
    for match in re.finditer(re.escape(term), query):
        left = query[max(0, match.start() - 8) : match.start()]
        span = query[max(0, match.start() - 8) : match.end() + 2]
        direct_prefix = re.search(
            r"(?:没有|没|无|未|不见|否认|并无|并没有|不是)\s*(?:明显|什么|太|怎么|再)?[，,、。；;\s]*$",
            left,
        )
        adjacent_bu = re.search(r"(?:不|不太|不怎么)\s*$", left)
        if direct_prefix or adjacent_bu or re.search(rf"不(?:怎么|太)?{re.escape(term)}", span):
            return True
    return False


def inferred_source_types(query: str, hits: Iterable[TermHit]) -> list[dict[str, Any]]:
    scores: Counter[str] = Counter()
    evidence: dict[str, set[str]] = defaultdict(set)
    for source_type, keywords in SOURCE_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                scores[source_type] += 3
                evidence[source_type].add(keyword)
    for hit in hits:
        if hit.source_type:
            scores[hit.source_type] += 1
            evidence[hit.source_type].add(hit.term)
    return [
        {"source_type": source_type, "score": score, "evidence": sorted(evidence[source_type])[:12]}
        for source_type, score in scores.most_common()
    ]


def compact_hits(hits: Iterable[TermHit], limit: int = 30) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        key = (hit.term, hit.field, hit.entry_id)
        if key in seen:
            continue
        seen.add(key)
        compacted.append(
            {
                "term": hit.term,
                "matched_text": hit.matched_text,
                "field": hit.field,
                "entry_id": hit.entry_id,
                "title": hit.title,
                "source_type": hit.source_type,
                "formula": hit.formula,
                "herb_name": hit.herb_name,
            }
        )
        if len(compacted) >= limit:
            break
    return compacted


def extract_from_record(
    *,
    source_line: int,
    raw_record: dict[str, Any],
    term_index: LocalTermIndex,
    max_terms_per_query: int = 80,
) -> dict[str, Any] | None:
    query = clean_query(str(raw_record.get("query", "")))
    if not query or PII_RE.search(query):
        return None

    diagnostic_hits = match_terms(
        query,
        term_index.diagnostic_terms,
        term_index.diagnostic_buckets,
        max_terms=max_terms_per_query,
    )
    pathogenesis_hits = match_terms(
        query,
        term_index.pathogenesis_terms,
        term_index.pathogenesis_buckets,
        max_terms=max_terms_per_query,
    )
    source_hint_hits = match_terms(
        query,
        term_index.source_hint_terms,
        term_index.source_hint_buckets,
        max_terms=max_terms_per_query,
    )
    all_hits = diagnostic_hits + pathogenesis_hits + source_hint_hits
    negated_hits = [hit for hit in diagnostic_hits if is_negated(query, hit.term)]

    query_hash = stable_hash(query, 24)
    return {
        "candidate_id": f"chatmed_query_{source_line}_{query_hash}",
        "source_dataset": "ChatMed_TCM-v0.2.json",
        "source_line": source_line,
        "query_hash": query_hash,
        "query": query,
        "review_status": QUERY_CANDIDATE_STATUS,
        "evidence_origin": EVIDENCE_ORIGIN,
        "diagnostic_matches": compact_hits(diagnostic_hits),
        "pathogenesis_matches": compact_hits(pathogenesis_hits),
        "source_hint_matches": compact_hits(source_hint_hits),
        "negated_matches": compact_hits(negated_hits),
        "inferred_source_types": inferred_source_types(query, all_hits),
        "match_counts": {
            "diagnostic": len(diagnostic_hits),
            "pathogenesis": len(pathogenesis_hits),
            "source_hint": len(source_hint_hits),
            "negated": len(negated_hits),
        },
    }


def iter_jsonl(path: Path, limit: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit and line_number > limit:
                break
            if not raw_line.strip():
                continue
            yield line_number, json.loads(raw_line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


class CappedRows:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.rows: list[dict[str, Any]] = []
        self.seen = 0

    def add(self, row: dict[str, Any]) -> None:
        self.seen += 1
        if not self.cap or len(self.rows) < self.cap:
            self.rows.append(row)

    @property
    def truncated(self) -> bool:
        return bool(self.cap and self.seen > self.cap)


def project_candidate(record: dict[str, Any], candidate_type: str) -> dict[str, Any]:
    base = {
        "candidate_id": record["candidate_id"].replace("chatmed_query", f"chatmed_{candidate_type}"),
        "candidate_type": candidate_type,
        "source_dataset": record["source_dataset"],
        "source_line": record["source_line"],
        "query_hash": record["query_hash"],
        "query": record["query"],
        "review_status": record["review_status"],
        "evidence_origin": record["evidence_origin"],
    }
    if candidate_type == "diagnostic_key":
        base["matched_terms"] = record["diagnostic_matches"]
        base["negated_terms"] = record["negated_matches"]
    elif candidate_type == "pathogenesis":
        base["matched_terms"] = record["pathogenesis_matches"]
    elif candidate_type == "source_type":
        base["inferred_source_types"] = record["inferred_source_types"]
        base["source_hint_matches"] = record["source_hint_matches"]
    elif candidate_type == "negation_forbidden":
        base["forbidden_term_candidates"] = record["negated_matches"]
    elif candidate_type == "unmatched_or_unusable":
        base["reason"] = "no_local_query_term_match"
    return base


def extract_candidates(
    *,
    chatmed_path: Path,
    syndrome_dictionary_path: Path,
    output_dir: Path,
    limit: int = 0,
    max_rows_per_file: int = 5000,
    max_terms_per_query: int = 80,
) -> dict[str, Any]:
    term_index = load_local_term_index(syndrome_dictionary_path)
    buckets = {
        "diagnostic_key": CappedRows(max_rows_per_file),
        "pathogenesis": CappedRows(max_rows_per_file),
        "source_type": CappedRows(max_rows_per_file),
        "negation_forbidden": CappedRows(max_rows_per_file),
        "unmatched_or_unusable": CappedRows(max_rows_per_file),
    }
    counters: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()

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
        has_any_match = False
        if record["diagnostic_matches"]:
            has_any_match = True
            buckets["diagnostic_key"].add(project_candidate(record, "diagnostic_key"))
            counters["records_with_diagnostic_match"] += 1
        if record["pathogenesis_matches"]:
            has_any_match = True
            buckets["pathogenesis"].add(project_candidate(record, "pathogenesis"))
            counters["records_with_pathogenesis_match"] += 1
        if record["inferred_source_types"]:
            has_any_match = True
            buckets["source_type"].add(project_candidate(record, "source_type"))
            counters["records_with_source_type_hint"] += 1
            for item in record["inferred_source_types"][:3]:
                source_type_counter[str(item["source_type"])] += 1
        if record["negated_matches"]:
            has_any_match = True
            buckets["negation_forbidden"].add(project_candidate(record, "negation_forbidden"))
            counters["records_with_negated_terms"] += 1
        if not has_any_match:
            buckets["unmatched_or_unusable"].add(project_candidate(record, "unmatched_or_unusable"))
            counters["records_without_local_match"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "diagnostic_key": "diagnostic_key_candidates.jsonl",
        "pathogenesis": "pathogenesis_candidates.jsonl",
        "source_type": "source_type_candidates.jsonl",
        "negation_forbidden": "negation_forbidden_candidates.jsonl",
        "unmatched_or_unusable": "unmatched_or_unusable.jsonl",
    }
    written: dict[str, dict[str, Any]] = {}
    for key, filename in files.items():
        bucket = buckets[key]
        path = output_dir / filename
        written_count = write_jsonl(path, bucket.rows)
        written[key] = {
            "path": display_path(path),
            "written": written_count,
            "seen": bucket.seen,
            "truncated": bucket.truncated,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": display_path(chatmed_path),
        "syndrome_dictionary": display_path(syndrome_dictionary_path),
        "records_seen": counters["records_seen"],
        "local_entry_count": term_index.entry_count,
        "local_term_counts": {
            "diagnostic_terms": len(term_index.diagnostic_terms),
            "pathogenesis_terms": len(term_index.pathogenesis_terms),
            "source_hint_terms": len(term_index.source_hint_terms),
        },
        "review_status": QUERY_CANDIDATE_STATUS,
        "evidence_origin": EVIDENCE_ORIGIN,
        "counters": dict(counters),
        "top_inferred_source_types": source_type_counter.most_common(20),
        "output_files": written,
        "safety_note": (
            "Only ChatMed query text was used for candidate extraction. ChatMed response text is not treated "
            "as verified medical evidence and is not written to these candidate files."
        ),
    }
    summary_path = output_dir / "chatmed_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 ChatMed 问题侧抽取候选词表/过滤字段，输出隔离的 candidate_unverified 审计文件。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CHATMED, help="ChatMed JSONL 输入文件")
    parser.add_argument(
        "--syndrome-dictionary",
        type=Path,
        default=DEFAULT_SYNDROME_DICTIONARY,
        help="本地结构化方证词典 JSONL",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="候选输出目录")
    parser.add_argument("--limit", type=int, default=0, help="最多读取多少行；0 表示全量")
    parser.add_argument("--max-rows-per-file", type=int, default=5000, help="每类候选最多写出多少行；0 表示不限制")
    parser.add_argument("--max-terms-per-query", type=int, default=80, help="单条问题最多保留多少个本地词表命中")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = extract_candidates(
        chatmed_path=args.input,
        syndrome_dictionary_path=args.syndrome_dictionary,
        output_dir=args.output_dir,
        limit=args.limit,
        max_rows_per_file=args.max_rows_per_file,
        max_terms_per_query=args.max_terms_per_query,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
