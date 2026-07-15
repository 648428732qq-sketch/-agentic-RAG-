from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

import config  # noqa: E402
from db.qdrant_client_factory import create_qdrant_client  # noqa: E402
from build_syndrome_dictionary import SyndromeEntry  # noqa: E402
from query_syndrome_dictionary import close_default_retriever, query_syndrome_dictionary  # noqa: E402


EXPECTED_SOURCE_TYPES = {
    "formula_syndrome",
    "classical_clause",
    "classical_acupuncture",
    "classical_acupuncture_principle",
    "herb_indication",
    "classical_theory",
}

SMOKE_TESTS = [
    {
        "query": "怕冷头疼不出汗还喘",
        "expected_formula": "麻黄汤",
        "expected_source_type": "formula_syndrome",
    },
    {
        "query": "吹风难受出汗头疼恶心",
        "expected_formula": "桂枝汤",
        "expected_source_type": "formula_syndrome",
    },
    {
        "query": "咳嗽喘痰很稀躺不下",
        "expected_formula": "小青龙汤",
        "expected_source_type": "formula_syndrome",
    },
    {
        "query": "胃胀胃疼吃不下",
        "expected_source_type": "classical_acupuncture",
        "expected_intervention_text": "三里",
    },
    {
        "query": "眼睛痛看不清",
        "expected_source_type": "herb_indication",
        "expected_intervention_type": "herb",
    },
    {
        "query": "十二经为什么独取寸口",
        "expected_source_type": "classical_theory",
        "expected_intervention_type": "theory",
    },
    {
        "query": "针刺补泻和迎随是什么意思",
        "expected_source_type": "classical_acupuncture_principle",
        "expected_intervention_type": "acupuncture_principle",
    },
]


def load_entries(path: Path) -> tuple[list[SyndromeEntry], list[dict[str, Any]]]:
    entries: list[SyndromeEntry] = []
    errors: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            entries.append(SyndromeEntry.model_validate(payload))
        except Exception as exc:
            errors.append({"line": line_number, "error": str(exc)})
    return entries, errors


def validate_entry(entry: SyndromeEntry) -> list[str]:
    problems: list[str] = []
    if entry.source_type not in EXPECTED_SOURCE_TYPES:
        problems.append(f"unknown source_type={entry.source_type}")
    if not entry.entry_id:
        problems.append("missing entry_id")
    if not entry.title:
        problems.append("missing title")
    if not entry.source_book:
        problems.append("missing source_book")
    if not entry.source_file:
        problems.append("missing source_file")
    if not entry.evidence and not entry.indications:
        problems.append("missing evidence/indications")
    if not entry.search_text:
        problems.append("missing search_text")
    if not entry.ancient_symptoms:
        problems.append("missing ancient_symptoms")
    required_terms = {
        str(term)
        for group in entry.required_symptom_groups
        for term in (group if isinstance(group, list) else [group])
    }
    required_forbidden_overlap = sorted(required_terms & set(entry.forbidden_terms))
    if required_forbidden_overlap:
        problems.append(f"required/forbidden conflict={required_forbidden_overlap}")
    if entry.source_type in {"formula_syndrome", "classical_clause"} and not entry.differential_keys:
        problems.append("missing differential_keys")

    if entry.source_type == "formula_syndrome":
        if entry.intervention_type != "formula":
            problems.append("formula_syndrome intervention_type must be formula")
        if not entry.formula:
            problems.append("formula_syndrome missing formula")
        if not entry.formula_composition:
            problems.append("formula_syndrome missing formula_composition")

    if entry.source_type == "classical_clause":
        if entry.intervention_type != "formula":
            problems.append("classical_clause intervention_type must be formula")
        if not entry.formula:
            problems.append("classical_clause missing formula")
        if "不可与" in entry.evidence and "可与" in entry.evidence:
            problems.append("classical_clause may include ambiguous negative/positive wording")

    if entry.source_type == "classical_acupuncture":
        if entry.intervention_type != "acupuncture":
            problems.append("classical_acupuncture intervention_type must be acupuncture")
        if entry.formula:
            problems.append("classical_acupuncture must not set formula")
        if not entry.treatment_method:
            problems.append("classical_acupuncture missing treatment_method")
        if "组成" in entry.search_text:
            problems.append("classical_acupuncture search_text looks formula-like")

    if entry.source_type == "classical_acupuncture_principle":
        if entry.intervention_type != "acupuncture_principle":
            problems.append("classical_acupuncture_principle intervention_type must be acupuncture_principle")
        if entry.formula:
            problems.append("classical_acupuncture_principle must not set formula")
        if not entry.acupuncture_principle:
            problems.append("classical_acupuncture_principle missing acupuncture_principle")
        if not entry.acupuncture_terms:
            problems.append("classical_acupuncture_principle missing acupuncture_terms")

    if entry.source_type == "herb_indication":
        if entry.intervention_type != "herb":
            problems.append("herb_indication intervention_type must be herb")
        if not entry.herb_name:
            problems.append("herb_indication missing herb_name")
        if entry.formula:
            problems.append("herb_indication must not set formula")
        if not entry.nature_flavor:
            problems.append("herb_indication missing nature_flavor")
        if not entry.indications:
            problems.append("herb_indication missing indications")

    if entry.source_type == "classical_theory":
        if entry.intervention_type != "theory":
            problems.append("classical_theory intervention_type must be theory")
        if entry.formula:
            problems.append("classical_theory must not set formula")
        if not entry.theory_topic:
            problems.append("classical_theory missing theory_topic")
        if not entry.theory_terms:
            problems.append("classical_theory missing theory_terms")
        if not entry.theory_answer:
            problems.append("classical_theory missing theory_answer")

    return problems


def validate_entries(entries: list[SyndromeEntry], max_examples: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    structured_field_counts: dict[str, Counter[str]] = {}
    for entry in entries:
        source_counter = structured_field_counts.setdefault(entry.source_type, Counter())
        for field in (
            "required_symptom_groups",
            "forbidden_terms",
            "differential_keys",
            "must_clarify_fields",
        ):
            if getattr(entry, field):
                source_counter[field] += 1
        entry_problems = validate_entry(entry)
        if entry_problems:
            problems.append(
                {
                    "entry_id": entry.entry_id,
                    "source_type": entry.source_type,
                    "source_book": entry.source_book,
                    "title": entry.title,
                    "problems": entry_problems,
                }
            )

    summary = {
        "entry_count": len(entries),
        "source_type_counts": dict(Counter(entry.source_type for entry in entries)),
        "source_book_counts": dict(Counter(entry.source_book for entry in entries)),
        "intervention_type_counts": dict(Counter(entry.intervention_type for entry in entries)),
        "structured_field_counts": {
            source_type: dict(counter) for source_type, counter in structured_field_counts.items()
        },
    }
    return problems[:max_examples], summary


def validate_qdrant(expected_count: int, expected_source_type_counts: dict[str, int]) -> dict[str, Any]:
    client = create_qdrant_client()
    payload_problems: list[dict[str, Any]] = []
    payload_source_type_counts: Counter[str] = Counter()
    payload_count = 0
    try:
        exists = client.collection_exists(config.SYNDROME_COLLECTION)
        count = client.count(collection_name=config.SYNDROME_COLLECTION, exact=True).count if exists else 0
        if exists:
            next_offset = None
            while True:
                records, next_offset = client.scroll(
                    collection_name=config.SYNDROME_COLLECTION,
                    offset=next_offset,
                    limit=256,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload_count += 1
                    try:
                        entry = SyndromeEntry.model_validate(record.payload or {})
                        payload_source_type_counts[entry.source_type] += 1
                    except Exception as exc:
                        if len(payload_problems) < 20:
                            payload_problems.append({"point_id": str(record.id), "error": str(exc)})
                if not next_offset:
                    break
    finally:
        client.close()

    payload_counts = dict(payload_source_type_counts)
    payloads_ok = (
        exists
        and payload_count == expected_count
        and not payload_problems
        and payload_counts == expected_source_type_counts
    )
    return {
        "collection": config.SYNDROME_COLLECTION,
        "exists": exists,
        "count": count,
        "expected_count": expected_count,
        "payload_count": payload_count,
        "payload_source_type_counts": payload_counts,
        "expected_source_type_counts": expected_source_type_counts,
        "payload_problems": payload_problems,
        "payloads_ok": payloads_ok,
        "ok": exists and count == expected_count and payloads_ok,
    }


def validate_smoke_tests() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    try:
        for test in SMOKE_TESTS:
            result = query_syndrome_dictionary(test["query"], limit=10)
            matches = result.get("matches", [])
            top = matches[0] if matches else {}

            def candidate_matches(candidate: dict[str, Any]) -> bool:
                if test.get("expected_formula") and candidate.get("formula") != test["expected_formula"]:
                    return False
                if test.get("expected_source_type") and candidate.get("source_type") != test["expected_source_type"]:
                    return False
                if (
                    test.get("expected_intervention_type")
                    and candidate.get("intervention_type") != test["expected_intervention_type"]
                ):
                    return False
                text = " ".join(
                    str(candidate.get(key, ""))
                    for key in ("title", "syndrome_name", "formula", "source_book", "intervention_name", "herb_name")
                )
                text += " " + " ".join(candidate.get("ancient_symptoms", []))
                return not test.get("expected_intervention_text") or test["expected_intervention_text"] in text

            ok = any(candidate_matches(candidate) for candidate in matches)
            results.append(
                {
                    "query": test["query"],
                    "ok": ok,
                    "expected": {k: v for k, v in test.items() if k != "query"},
                    "top": {
                        "title": top.get("title", ""),
                        "source_type": top.get("source_type", ""),
                        "intervention_type": top.get("intervention_type", ""),
                        "intervention_name": top.get("intervention_name", ""),
                        "herb_name": top.get("herb_name", ""),
                        "formula": top.get("formula", ""),
                        "source_book": top.get("source_book", ""),
                        "syndrome_name": top.get("syndrome_name", ""),
                        "overlap_score": top.get("overlap_score", 0),
                    },
                }
            )
    finally:
        close_default_retriever()
    return results


def validate_structured_answer_gate() -> list[dict[str, Any]]:
    from core.syndrome_retriever import SyndromeRetriever, should_use_structured_answer

    gate_tests = [
        {
            "query": "眼睛痛看不清",
            "expected_source_type": "herb_indication",
            "expected_gate": False,
        },
        {
            "query": "咳嗽喘不上气",
            "expected_gate": False,
        },
        {
            "query": "十二经为什么独取寸口",
            "expected_source_type": "classical_theory",
            "expected_gate": True,
        },
        {
            "query": "针刺补泻和迎随是什么意思",
            "expected_source_type": "classical_acupuncture_principle",
            "expected_gate": True,
        },
        {
            "query": "怕冷头疼不出汗还喘",
            "expected_source_type": "formula_syndrome",
            "expected_gate": False,
        },
    ]
    retriever = SyndromeRetriever()
    results: list[dict[str, Any]] = []
    try:
        for test in gate_tests:
            result = retriever.search(test["query"], limit=3)
            matches = result.get("matches", [])
            top_payload = matches[0].get("payload", {}) if matches else {}
            gate = should_use_structured_answer(result)
            ok = gate is test["expected_gate"]
            if test.get("expected_source_type"):
                ok = ok and bool(matches) and top_payload.get("source_type") == test["expected_source_type"]
            results.append(
                {
                    "query": test["query"],
                    "ok": ok,
                    "expected": {k: v for k, v in test.items() if k != "query"},
                    "gate": gate,
                    "top": {
                        "title": top_payload.get("title", ""),
                        "source_type": top_payload.get("source_type", ""),
                        "formula": top_payload.get("formula", ""),
                        "intervention_type": top_payload.get("intervention_type", ""),
                        "intervention_name": top_payload.get("intervention_name", ""),
                        "herb_name": top_payload.get("herb_name", ""),
                        "overlap_score": matches[0].get("overlap_score", 0) if matches else 0,
                    },
                }
            )
    finally:
        retriever.close()
    return results


def validate_llm_translator() -> dict[str, Any]:
    from core.llm_factory import create_query_translator_client
    from core.syndrome_retriever import SyndromeRetriever

    tests = [
        {
            "query": "鼻子不通气脑袋疼还怕吹空调",
            "expected_terms": {"鼻塞", "头痛", "恶风"},
            "expected_source_type": "formula_syndrome",
        },
        {
            "query": "针灸里顺着气走和逆着气走怎么补泻",
            "expected_terms": {"迎随", "补泻"},
            "expected_source_type": "classical_acupuncture_principle",
        },
        {
            "query": "为什么古书说只看寸口脉就能知道十二经",
            "expected_terms": {"寸口", "十二经"},
            "expected_source_type": "classical_theory",
        },
    ]
    retriever = SyndromeRetriever()
    retriever.set_llm(create_query_translator_client())
    cases: list[dict[str, Any]] = []
    try:
        for test in tests:
            result = retriever.search(test["query"], limit=3)
            query_info = result.get("query", {})
            matches = result.get("matches", [])
            canonical_terms = set(query_info.get("canonical_terms", []))
            top_payload = matches[0].get("payload", {}) if matches else {}
            ok = (
                query_info.get("translation_method") == "llm_structured"
                and test["expected_terms"].issubset(canonical_terms)
                and bool(matches)
                and top_payload.get("source_type") == test["expected_source_type"]
            )
            cases.append(
                {
                    "ok": ok,
                    "query": test["query"],
                    "expected_terms": sorted(test["expected_terms"]),
                    "expected_source_type": test["expected_source_type"],
                    "translation_method": query_info.get("translation_method"),
                    "query_intent": query_info.get("query_intent", ""),
                    "canonical_terms": query_info.get("canonical_terms", []),
                    "colloquial_terms": query_info.get("colloquial_terms", []),
                    "top": [
                        {
                            "title": item.get("payload", {}).get("title", ""),
                            "source_type": item.get("payload", {}).get("source_type", ""),
                            "formula": item.get("payload", {}).get("formula", ""),
                            "source_book": item.get("payload", {}).get("source_book", ""),
                            "overlap_score": item.get("overlap_score", 0),
                        }
                        for item in matches[:3]
                    ],
                }
            )
    finally:
        retriever.close()
    ok = all(case["ok"] for case in cases)
    return {
        "ok": ok,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证结构化方证/古籍/针刺 payload 与 Qdrant 索引")
    parser.add_argument("--jsonl", default=str(ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"))
    parser.add_argument("--report", default=str(ROOT / "datasets" / "structured" / "syndrome_dictionary_validation.json"))
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--check-llm", action="store_true", help="调用当前 LLM 提供商验证结构化口语症状翻译器")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    entries, parse_errors = load_entries(jsonl_path)
    entry_problems, summary = validate_entries(entries)
    qdrant_result = None if args.skip_qdrant else validate_qdrant(len(entries), summary["source_type_counts"])
    smoke_results = [] if args.skip_smoke else validate_smoke_tests()
    gate_results = [] if args.skip_smoke else validate_structured_answer_gate()
    llm_result = validate_llm_translator() if args.check_llm else None

    ok = not parse_errors and not entry_problems
    if qdrant_result is not None:
        ok = ok and qdrant_result["ok"]
    if smoke_results:
        ok = ok and all(item["ok"] for item in smoke_results)
    if gate_results:
        ok = ok and all(item["ok"] for item in gate_results)
    if llm_result is not None:
        ok = ok and llm_result["ok"]

    report = {
        "ok": ok,
        "summary": summary,
        "parse_errors": parse_errors,
        "entry_problems": entry_problems,
        "qdrant": qdrant_result,
        "smoke_tests": smoke_results,
        "structured_answer_gate": gate_results,
        "llm_translator": llm_result,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
