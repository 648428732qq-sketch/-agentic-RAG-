from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from scripts.build_syndrome_dictionary import (  # noqa: E402
    GENERIC_DIAGNOSTIC_TERMS,
    SYMPTOM_ALIASES,
    TERM_GROUP_RULES,
    SyndromeEntry,
    make_search_text,
    normalize_required_symptom_groups,
)


DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_REPLACEMENTS = ROOT / "datasets" / "structured" / "syndrome_dictionary_reviewed_replacements.jsonl"
DEFAULT_CANDIDATES = ROOT / "datasets" / "structured" / "syndrome_required_groups_candidates.jsonl"
DEFAULT_REPORT = ROOT / "datasets" / "structured" / "syndrome_required_groups_candidates_report.json"
QUARANTINE = ROOT / "datasets" / "quarantine"

DIAGNOSTIC_PATTERN = re.compile(
    r"(?:临床应用)?以(?P<phrase>[^。\n]{2,220}?)为(?:辨证(?:治)?|证治)要点",
    re.MULTILINE,
)
SPLIT_PATTERN = re.compile(r"[，、；;]+")
ALTERNATIVE_PATTERN = re.compile(r"(?:或|或见|以及|及)")
GRAMMAR_FRAGMENT = re.compile(
    r"^(?:甚则|并见|兼见|伴有|可见|出现|表现为|泻必|泻后|时有|而|且|与)+|(?:者|为宜|等)$"
)
ASCII_NOISE = re.compile(r"[A-Za-z]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def extract_diagnostic_phrase(raw_text: str) -> tuple[str, int, int]:
    matches = list(DIAGNOSTIC_PATTERN.finditer(raw_text or ""))
    if not matches:
        return "", -1, -1
    match = matches[-1]
    phrase = match.group("phrase").strip(" ：:，,。")
    start = match.start("phrase") + len(match.group("phrase")) - len(match.group("phrase").lstrip())
    return phrase, start, start + len(phrase)


def classify_axis(term: str) -> str:
    if term.startswith("舌") or term.startswith("苔"):
        return "tongue"
    if "脉" in term:
        return "pulse"
    if any(marker in term for marker in ("寒", "冷", "热", "温", "凉")):
        return "cold_heat"
    if any(marker in term for marker in ("汗", "渴", "饮")):
        return "fluid_surface"
    if any(marker in term for marker in ("便", "泻", "痢", "尿", "溺", "淋")):
        return "elimination"
    return "main_symptom"


def _known_terms(entry: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("diagnostic_keys", "differential_keys", "ancient_symptoms"):
        values.extend(str(item).strip() for item in entry.get(field, []) if str(item).strip())
    return sorted(
        {
            term
            for term in values
            if 1 < len(term) <= 20 and term not in GENERIC_DIAGNOSTIC_TERMS
        },
        key=lambda term: (-len(term), term),
    )


def _clean_fragment(value: str) -> str:
    cleaned = GRAMMAR_FRAGMENT.sub("", value.strip(" ，、；;。"))
    return cleaned.strip(" ，、；;。")


def _decompose_segment(segment: str, known_terms: list[str]) -> list[str]:
    segment = _clean_fragment(segment)
    if len(segment) < 2:
        return []
    tongue_position = segment.find("舌")
    if tongue_position > 0 and segment[tongue_position - 1] == "口":
        tongue_position = -1
    marker_positions = [position for position in (tongue_position, segment.find("脉")) if position >= 0]
    if segment.startswith("苔"):
        marker_positions.append(0)
    if marker_positions:
        first_marker = min(marker_positions)
        pieces: list[str] = []
        prefix = _clean_fragment(segment[:first_marker])
        if prefix:
            pieces.extend(_decompose_segment(prefix, known_terms))
        objective = segment[first_marker:]
        pulse_position = objective.find("脉", 1)
        if pulse_position > 0:
            tongue = _clean_fragment(objective[:pulse_position])
            pulse = _clean_fragment(objective[pulse_position:])
            if tongue:
                pieces.append(tongue)
            if pulse:
                pieces.append(pulse)
        elif objective:
            pieces.append(objective)
        return unique(pieces)

    occupied = [False] * len(segment)
    selected: list[tuple[int, int, str]] = []
    for term in known_terms:
        start = segment.find(term)
        if start < 0:
            continue
        end = start + len(term)
        if any(occupied[start:end]):
            continue
        selected.append((start, end, term))
        for index in range(start, end):
            occupied[index] = True
    selected.sort()

    fragments: list[str] = [term for _, _, term in selected]
    cursor = 0
    for start, end, _ in selected + [(len(segment), len(segment), "")]:
        if start > cursor:
            leftover = _clean_fragment(segment[cursor:start])
            if len(leftover) >= 2 and not leftover.endswith(("必", "而", "且", "伴")):
                fragments.append(leftover)
        cursor = max(cursor, end)
    if not fragments:
        fragments.append(segment)
    return unique(fragments)


def diagnostic_terms(phrase: str, entry: dict[str, Any]) -> list[tuple[str, str]]:
    known_terms = _known_terms(entry)
    terms: list[tuple[str, str]] = []
    for segment in SPLIT_PATTERN.split(phrase):
        segment = _clean_fragment(segment)
        if not segment:
            continue
        alternatives = [part for part in ALTERNATIVE_PATTERN.split(segment) if _clean_fragment(part)]
        if len(alternatives) > 1:
            alternative_terms = unique(
                [
                    term
                    for part in alternatives
                    for term in _decompose_segment(part, known_terms)
                ]
            )
            if alternative_terms:
                terms.append(("alternative", "|".join(alternative_terms)))
            continue
        for term in _decompose_segment(segment, known_terms):
            terms.append(("required", term))
    return terms


def alias_group(source_terms: list[str]) -> list[str]:
    aliases = list(source_terms)
    for source_term in source_terms:
        for _, rule_group in TERM_GROUP_RULES:
            if source_term in rule_group or any(term in source_term or source_term in term for term in rule_group):
                aliases.extend(rule_group)
        aliases.extend(SYMPTOM_ALIASES.get(source_term, []))
    return unique([term for term in aliases if 1 < len(term) <= 20])


def groups_from_phrase(phrase: str, entry: dict[str, Any]) -> tuple[list[list[str]], list[str]]:
    groups: list[list[str]] = []
    source_terms: list[str] = []
    for kind, value in diagnostic_terms(phrase, entry):
        terms = value.split("|") if kind == "alternative" else [value]
        terms = unique([_clean_fragment(term) for term in terms])
        if not terms:
            continue
        group = alias_group(terms)
        if not group:
            continue
        groups.append(group)
        source_terms.append(" / ".join(terms))

    existing_groups = [
        unique([str(term) for term in group])
        for group in entry.get("required_symptom_groups", [])
        if any(str(term) in phrase for term in group)
    ]
    merged = list(existing_groups)
    for group in groups:
        overlapping_index = next(
            (index for index, current in enumerate(merged) if _groups_overlap(group, current)),
            None,
        )
        if overlapping_index is not None:
            merged[overlapping_index] = unique(merged[overlapping_index] + group)
            continue
        merged.append(group)
    return merged, source_terms


def infer_clarify_fields(groups: list[list[str]], current: list[str]) -> list[str]:
    values = list(current)
    for group in groups:
        text = " ".join(group)
        if "舌" in text or "苔" in text:
            values.append("舌象")
        if "脉" in text:
            values.append("脉象")
        if "汗" in text:
            values.append("是否出汗")
        if "渴" in text or "饮" in text:
            values.append("是否口渴")
        if any(marker in text for marker in ("便", "泻", "痢")):
            values.append("大便情况")
        if any(marker in text for marker in ("尿", "溺", "淋")):
            values.append("小便情况")
    return unique(values)


def trace_groups(entry: dict[str, Any], phrase: str, phrase_start: int, groups: list[list[str]]) -> list[dict[str, Any]]:
    raw_text = str(entry.get("raw_text", ""))
    traces: list[dict[str, Any]] = []
    for group in groups:
        source_term = next((term for term in group if term in phrase), "")
        if not source_term:
            source_term = next((term for term in group if term in raw_text), "")
        local_start = phrase.find(source_term) if source_term else -1
        candidate_start = phrase_start + local_start if local_start >= 0 else -1
        start = (
            candidate_start
            if candidate_start >= 0 and raw_text[candidate_start : candidate_start + len(source_term)] == source_term
            else raw_text.find(source_term)
        )
        end = start + len(source_term) if start >= 0 else -1
        traces.append(
            {
                "axis": classify_axis(source_term or group[0]),
                "terms": group,
                "source_term": source_term,
                "source_field": "raw_text",
                "source_start": start,
                "source_end": end,
                "source_excerpt": raw_text[max(0, start - 24) : min(len(raw_text), end + 24)] if start >= 0 else "",
                "also_in_evidence": bool(source_term and source_term in str(entry.get("evidence", ""))),
            }
        )
    return traces


def _groups_overlap(first: list[str], second: list[str]) -> bool:
    return any(
        left == right or left in right or right in left
        for left in first
        for right in second
        if len(left) > 1 and len(right) > 1
    )


def supplement_from_evidence(entry: dict[str, Any], groups: list[list[str]]) -> list[list[str]]:
    evidence = str(entry.get("evidence", ""))
    supplemented = list(groups)
    has_main = any(classify_axis(group[0]) in {"main_symptom", "elimination"} for group in supplemented if group)
    if len(supplemented) >= 2 and has_main:
        return supplemented
    for term in entry.get("differential_keys", []):
        term = _clean_fragment(str(term))
        if not (2 < len(term) <= 16) or term not in evidence:
            continue
        if classify_axis(term) not in {"main_symptom", "elimination"}:
            continue
        if any(marker in term for marker in ("证", "所治", "因", "属于")):
            continue
        group = alias_group([term])
        if not group or any(_groups_overlap(group, current) for current in supplemented):
            continue
        supplemented.append(group)
        has_main = True
        if len(supplemented) >= 2 and has_main:
            break
    return supplemented


def build_candidate(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_text = str(entry.get("raw_text", ""))
    phrase, phrase_start, phrase_end = extract_diagnostic_phrase(raw_text)
    original_phrase = phrase
    normalizations: list[dict[str, str]] = []
    if "xian" in phrase and "焮" in str(entry.get("evidence", "")):
        phrase = phrase.replace("xian", "焮")
        normalizations.append({"from": "xian", "to": "焮", "evidence_field": "evidence"})
    groups, _ = groups_from_phrase(phrase, entry) if phrase else ([], [])
    groups = supplement_from_evidence(entry, groups)
    traces = trace_groups(entry, phrase, phrase_start, groups) if phrase else []
    required_terms = {term for group in groups for term in group}
    forbidden_overlap = sorted(required_terms & set(entry.get("forbidden_terms", [])))
    axes = Counter(trace["axis"] for trace in traces if trace.get("source_term"))
    reasons: list[str] = []
    if not phrase:
        reasons.append("missing_explicit_diagnostic_phrase")
    if len(groups) < 2:
        reasons.append("fewer_than_two_required_groups")
    if len(groups) > 8:
        reasons.append("too_many_required_groups")
    if any(not trace.get("source_term") or trace.get("source_start", -1) < 0 for trace in traces):
        reasons.append("untraced_required_group")
    if forbidden_overlap:
        reasons.append("required_forbidden_conflict")
    if ASCII_NOISE.search(phrase):
        reasons.append("ascii_source_noise")
    if not axes.get("main_symptom") and not axes.get("elimination"):
        reasons.append("missing_main_symptom_axis")

    eligible = not reasons
    replacement: dict[str, Any] | None = None
    if eligible:
        model = SyndromeEntry.model_validate(entry)
        model.required_symptom_groups = groups
        model.must_clarify_fields = infer_clarify_fields(groups, [])
        model.review_status = (
            f"{model.review_status}+required_groups_rule_validated"
            if "required_groups_rule_validated" not in model.review_status
            else model.review_status
        )
        normalize_required_symptom_groups(model)
        model.search_text = make_search_text(model)
        replacement = model.model_dump(mode="json")

    candidate = {
        "entry_id": entry.get("entry_id"),
        "formula": entry.get("formula"),
        "source_type": entry.get("source_type"),
        "source_book": entry.get("source_book"),
        "source_file": entry.get("source_file"),
        "source_url": entry.get("source_url"),
        "current_required_symptom_groups": entry.get("required_symptom_groups", []),
        "proposed_required_symptom_groups": groups,
        "proposed_must_clarify_fields": infer_clarify_fields(groups, []),
        "diagnostic_phrase": phrase,
        "diagnostic_phrase_original": original_phrase,
        "evidence_backed_normalizations": normalizations,
        "diagnostic_phrase_trace": {
            "source_field": "raw_text",
            "source_start": phrase_start,
            "source_end": phrase_end,
            "source_excerpt": raw_text[max(0, phrase_start - 32) : min(len(raw_text), phrase_end + 32)]
            if phrase_start >= 0
            else "",
        },
        "group_traces": traces,
        "evidence": entry.get("evidence", ""),
        "evidence_sha256": sha256_text(str(entry.get("evidence", ""))),
        "raw_text_sha256": sha256_text(raw_text),
        "validation": {
            "auto_apply_eligible": eligible,
            "reasons": reasons,
            "group_count": len(groups),
            "axis_counts": dict(axes),
            "required_forbidden_overlap": forbidden_overlap,
            "policy": "explicit_diagnostic_phrase_v1",
        },
        "review_status": "rule_validated_candidate" if eligible else "manual_review_required",
    }
    return candidate, replacement


def merge_replacements(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    positions = {str(row.get("entry_id")): index for index, row in enumerate(merged)}
    for row in additions:
        entry_id = str(row.get("entry_id"))
        if entry_id in positions:
            merged[positions[entry_id]] = row
        else:
            positions[entry_id] = len(merged)
            merged.append(row)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-traced required_symptom_groups candidates")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reviewed-replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--write-reviewed-replacements", action="store_true")
    args = parser.parse_args()

    dictionary = read_jsonl(args.dictionary)
    targets = [
        entry
        for entry in dictionary
        if entry.get("source_type") == "formula_syndrome"
        and len(entry.get("required_symptom_groups") or []) <= 1
    ]
    candidates: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    for entry in targets:
        candidate, replacement = build_candidate(entry)
        candidates.append(candidate)
        if replacement is not None:
            replacements.append(replacement)
    write_jsonl(args.candidates, candidates)

    backup = ""
    merged_count = 0
    if args.write_reviewed_replacements:
        existing = read_jsonl(args.reviewed_replacements) if args.reviewed_replacements.exists() else []
        if args.reviewed_replacements.exists():
            QUARANTINE.mkdir(parents=True, exist_ok=True)
            backup_path = QUARANTINE / (
                "syndrome_dictionary_reviewed_replacements_before_required_groups_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".jsonl"
            )
            shutil.copy2(args.reviewed_replacements, backup_path)
            backup = str(backup_path)
        merged = merge_replacements(existing, replacements)
        write_jsonl(args.reviewed_replacements, merged)
        merged_count = len(merged)

    reason_counts = Counter(
        reason
        for candidate in candidates
        for reason in candidate["validation"]["reasons"]
    )
    report = {
        "dictionary": str(args.dictionary.resolve()),
        "target_count": len(targets),
        "candidate_count": len(candidates),
        "eligible_replacement_count": len(replacements),
        "manual_review_count": len(candidates) - len(replacements),
        "reason_counts": dict(reason_counts.most_common()),
        "empty_before": sum(not entry.get("required_symptom_groups") for entry in targets),
        "single_before": sum(len(entry.get("required_symptom_groups") or []) == 1 for entry in targets),
        "candidates": str(args.candidates.resolve()),
        "reviewed_replacements_written": bool(args.write_reviewed_replacements),
        "reviewed_replacements_count": merged_count,
        "reviewed_replacements_backup": backup,
        "policy": "explicit_diagnostic_phrase_v1",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
