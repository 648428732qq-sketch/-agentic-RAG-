from __future__ import annotations

from scripts.apply_syndrome_dictionary_replacements import merge_entries
from scripts.evaluate_required_group_hard_negatives import summarize
from scripts.generate_required_group_hard_negatives import generate_cases


def formula_entry() -> dict:
    return {
        "entry_id": "formula::测试方",
        "source_type": "formula_syndrome",
        "formula": "测试方",
        "required_symptom_groups": [["腹痛"], ["呕吐", "干呕"], ["脉弦"]],
        "evidence": "腹痛，呕吐，脉弦。",
        "source_file": "方剂大全.md",
        "review_status": "rule_validated_required_groups",
    }


def test_merge_replacements_preserves_order_and_replaces_by_id() -> None:
    base = [{"entry_id": "a", "value": 1}, {"entry_id": "b", "value": 2}]
    merged, replaced, appended = merge_entries(base, [{"entry_id": "a", "value": 3}, {"entry_id": "c"}])
    assert [row["entry_id"] for row in merged] == ["a", "b", "c"]
    assert merged[0]["value"] == 3
    assert replaced == 1
    assert appended == 1


def test_generates_exactly_one_omission_per_required_group() -> None:
    entry = formula_entry()
    candidates = [{"entry_id": entry["entry_id"], "validation": {"auto_apply_eligible": True}}]
    cases = generate_cases([entry], candidates)
    missing = [case for case in cases if case["style"] == "missing_one_required_group"]
    assert len(cases) == 4
    assert len(missing) == 3
    assert {case["omitted_group_index"] for case in missing} == {0, 1, 2}
    for case in missing:
        query = case["query"]
        assert all(term not in query for term in case["omitted_required_group"])


def test_summary_cannot_pass_by_rejecting_complete_signatures() -> None:
    rows = [
        {
            "style": "full_required_signature",
            "checks": {
                "target_in_top_k": True,
                "grounded": False,
                "route_ok": True,
                "top_payload_conflict_free": True,
            },
            "decision": {
                "reason_details": [{"code": "low_query_coverage"}],
            },
        },
        {
            "style": "missing_one_required_group",
            "checks": {
                "safe_rejection": True,
                "reason_present": True,
                "route_ok": True,
                "top_payload_conflict_free": True,
            },
            "decision": {
                "reason_details": [{"code": "missing_required_symptom_groups"}],
            },
        },
    ]

    report = summarize(rows)

    assert not report["ok"]
    assert report["failed_thresholds"]["full_grounded_rate"]["required"] == 0.95
    assert report["full_signature_rejection_reason_counts"] == {"low_query_coverage": 1}
