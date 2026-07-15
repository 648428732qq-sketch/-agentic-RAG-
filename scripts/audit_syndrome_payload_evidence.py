from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_VERIFIED = ROOT / "datasets" / "structured" / "syndrome_dictionary_verified.jsonl"
DEFAULT_QUARANTINE = ROOT / "datasets" / "quarantine" / "syndrome_payload_rejected.jsonl"
DEFAULT_LABEL_POOL = ROOT / "datasets" / "structured" / "query_translator_evidence_label_pool.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "external" / "reports" / "payload_evidence_audit.json"

TEXT_NORMALIZER = re.compile(r"[^0-9a-z\u3400-\u9fff]+", re.IGNORECASE)
NOISE_MARKERS = (
    "元素。",
    "不可与（不可以用）（可以用）",
    "上主之（主治此证）深仁",
    "网页导航",
    "点击加载",
)
EVIDENCE_FIELDS = (
    "evidence",
    "raw_text",
    "ancient_symptoms",
    "indications",
    "contraindications",
    "formula_analysis",
    "theory_answer",
    "property_text",
    "usage_original",
    "functions",
    "treatment_method",
    "acupuncture_principle",
)
TERM_FIELDS = (
    "ancient_symptoms",
    "diagnostic_keys",
    "differential_keys",
    "theory_terms",
    "acupuncture_terms",
    "herb_aliases",
)
TERM_FIELD_ROLES = {
    "ancient_symptoms": "ancient_symptom",
    "diagnostic_keys": "diagnostic_key",
    "differential_keys": "differential_key",
    "theory_terms": "theory_term",
    "acupuncture_terms": "acupuncture_term",
    "herb_aliases": "herb_alias",
}
SCALAR_TERM_ROLES = {
    "syndrome_name": "syndrome_name",
    "formula": "formula_name",
    "herb_name": "herb_name",
    "theory_topic": "theory_topic",
    "diagnostic_method": "diagnostic_method",
}
GOLD_ELIGIBLE_TERM_ROLES = {
    "ancient_symptom",
    "diagnostic_key",
    "required_symptom",
    "syndrome_name",
}
EXPECTED_SOURCE_TYPES = {
    "formula_syndrome",
    "classical_clause",
    "classical_acupuncture",
    "classical_acupuncture_principle",
    "herb_indication",
    "classical_theory",
}
INDEPENDENT_CLINICAL_SINGLE_CHARACTERS = {
    "喘",
    "咳",
    "呕",
    "吐",
    "渴",
    "汗",
    "痛",
    "痒",
    "肿",
    "麻",
    "晕",
    "冷",
    "热",
}


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def normalize(value: str) -> str:
    return TEXT_NORMALIZER.sub("", value.casefold())


def evidence_text(entry: dict[str, Any]) -> str:
    return " ".join(text for field in EVIDENCE_FIELDS for text in iter_strings(entry.get(field)))


def is_supported(term: str, evidence: str) -> bool:
    normalized_term = normalize(term)
    meaningful_length = len(normalized_term) >= 2 or normalized_term in INDEPENDENT_CLINICAL_SINGLE_CHARACTERS
    return meaningful_length and normalized_term in normalize(evidence)


def required_terms(entry: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in entry.get("required_symptom_groups") or []:
        terms = list(iter_strings(group))
        if terms:
            groups.append(terms)
    return groups


def term_roles(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}

    def add(term: str, role: str) -> None:
        key = normalize(term)
        if not key:
            return
        item = values.setdefault(key, {"term": term, "roles": set()})
        item["roles"].add(role)

    for field, role in TERM_FIELD_ROLES.items():
        for term in iter_strings(entry.get(field)):
            add(term, role)
    for group in required_terms(entry):
        for term in group:
            add(term, "required_symptom")
    for field, role in SCALAR_TERM_ROLES.items():
        for term in iter_strings(entry.get(field)):
            add(term, role)
    return values


def unique_terms(entry: dict[str, Any]) -> list[str]:
    return [item["term"] for item in term_roles(entry).values()]


def audit_entry(entry: dict[str, Any]) -> dict[str, Any]:
    evidence = evidence_text(entry)
    role_map = term_roles(entry)
    evidence_terms = [item["term"] for item in role_map.values()]
    supported_terms = [term for term in evidence_terms if is_supported(term, evidence)]
    unsupported_terms = [term for term in evidence_terms if term not in supported_terms]
    diagnostic_terms = list(iter_strings(entry.get("diagnostic_keys")))
    supported_diagnostic = [term for term in diagnostic_terms if is_supported(term, evidence)]
    broad_terms = sorted(
        {
            term
            for term in evidence_terms
            if len(normalize(term)) < 2 and normalize(term) not in INDEPENDENT_CLINICAL_SINGLE_CHARACTERS
        }
    )
    groups = required_terms(entry)
    supported_groups = [group for group in groups if any(is_supported(term, evidence) for term in group)]
    forbidden = {normalize(term) for term in iter_strings(entry.get("forbidden_terms")) if normalize(term)}
    required = {normalize(term) for group in groups for term in group if normalize(term)}
    conflicts = sorted(required & forbidden)

    reasons: list[str] = []
    severe: list[str] = []
    if not evidence.strip():
        severe.append("missing_evidence")
    if entry.get("source_type") not in EXPECTED_SOURCE_TYPES:
        severe.append("unknown_source_type")
    if not entry.get("entry_id") or not entry.get("source_file") or not entry.get("source_book"):
        severe.append("missing_provenance")
    if conflicts:
        severe.append("required_forbidden_conflict")
    if groups and len(supported_groups) / len(groups) < 0.5:
        severe.append("required_groups_weakly_supported")
    if diagnostic_terms and len(supported_diagnostic) / len(diagnostic_terms) < 0.25:
        severe.append("diagnostic_keys_weakly_supported")
    if broad_terms:
        reasons.append("overbroad_single_character_terms")
    if any(marker in evidence for marker in NOISE_MARKERS):
        severe.append("known_source_noise")
    if not supported_terms:
        severe.append("no_evidence_grounded_terms")

    review_status = str(entry.get("review_status", ""))
    confidence = float(entry.get("confidence") or 0.0)
    if severe:
        tier = "quarantine"
    elif review_status in {"human_verified", "expert_verified"}:
        tier = "verified"
    elif (
        entry.get("source_type") == "formula_syndrome"
        and confidence >= 0.65
        and (not groups or len(supported_groups) == len(groups))
        and (not diagnostic_terms or len(supported_diagnostic) / len(diagnostic_terms) >= 0.5)
    ):
        tier = "high"
    else:
        tier = "medium"
        reasons.append("rule_extracted_or_source_not_independently_verified")

    return {
        "entry_id": entry.get("entry_id"),
        "source_type": entry.get("source_type"),
        "source_book": entry.get("source_book"),
        "review_status": review_status,
        "confidence": confidence,
        "tier": tier,
        "reasons": sorted(set(reasons)),
        "severe_reasons": sorted(set(severe)),
        "evidence_term_count": len(evidence_terms),
        "supported_term_count": len(supported_terms),
        "supported_terms": supported_terms,
        "supported_term_roles": {
            term: sorted(role_map[normalize(term)]["roles"])
            for term in supported_terms
        },
        "unsupported_terms": unsupported_terms,
        "broad_terms": broad_terms,
        "required_group_count": len(groups),
        "supported_required_group_count": len(supported_groups),
        "diagnostic_term_count": len(diagnostic_terms),
        "supported_diagnostic_count": len(supported_diagnostic),
    }


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            entries.append(value)
    return entries


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_label_records(entry: dict[str, Any], audit: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if audit["tier"] == "quarantine":
        return
    for term in audit["supported_terms"]:
        roles = audit.get("supported_term_roles", {}).get(term, [])
        eligible_role = bool(set(roles) & GOLD_ELIGIBLE_TERM_ROLES)
        yield {
            "term_id": f"{entry['entry_id']}::{normalize(term)}",
            "canonical_term": term,
            "entry_id": entry["entry_id"],
            "source_type": entry.get("source_type"),
            "source_book": entry.get("source_book"),
            "source_file": entry.get("source_file"),
            "evidence": entry.get("evidence") or entry.get("raw_text") or "",
            "evidence_tier": audit["tier"],
            "term_roles": roles,
            "eligible_as_gold_label": audit["tier"] in {"verified", "high"} and eligible_role,
        }


def run_audit(
    input_path: Path,
    verified_path: Path,
    quarantine_path: Path,
    label_pool_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    entries = load_entries(input_path)
    audits = [audit_entry(entry) for entry in entries]
    tier_counts = Counter(audit["tier"] for audit in audits)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    for audit in audits:
        by_source[str(audit["source_type"])][audit["tier"]] += 1
        reason_counts.update(audit["reasons"])
        reason_counts.update(audit["severe_reasons"])

    accepted = [entry for entry, audit in zip(entries, audits) if audit["tier"] in {"verified", "high"}]
    rejected = [
        {"audit": audit, "payload": entry}
        for entry, audit in zip(entries, audits)
        if audit["tier"] == "quarantine"
    ]
    labels = [
        label
        for entry, audit in zip(entries, audits)
        for label in build_label_records(entry, audit)
    ]
    verified_count = write_jsonl(verified_path, accepted)
    quarantine_count = write_jsonl(quarantine_path, rejected)
    label_count = write_jsonl(label_pool_path, labels)

    report = {
        "report_version": 1,
        "entry_count": len(entries),
        "tier_counts": dict(sorted(tier_counts.items())),
        "verified_output_count": verified_count,
        "quarantine_output_count": quarantine_count,
        "evidence_label_count": label_count,
        "gold_eligible_label_count": sum(label["eligible_as_gold_label"] for label in labels),
        "policy": {
            "verified": "human_or_expert_verified_only",
            "high": "formula_source_with_full_required_group_support_and_strong_diagnostic_support",
            "medium": "usable_for_candidate_retrieval_only_not_gold_supervision",
            "quarantine": "excluded_from_candidate_and_supervision_pools",
            "modern_aliases": "never_self_validate_against_model_generated_modern_text",
        },
        "by_source_type": {name: dict(sorted(counts.items())) for name, counts in sorted(by_source.items())},
        "reason_counts": dict(reason_counts.most_common()),
        "audits": audits,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit syndrome payload fields against source evidence")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--label-pool", type=Path, default=DEFAULT_LABEL_POOL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args.input, args.verified, args.quarantine, args.label_pool, args.report)
    print(json.dumps({key: report[key] for key in ("entry_count", "tier_counts", "verified_output_count", "quarantine_output_count", "evidence_label_count", "gold_eligible_label_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
