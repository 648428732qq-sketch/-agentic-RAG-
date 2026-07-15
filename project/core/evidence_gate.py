from __future__ import annotations

from typing import Any


KNOWLEDGE_INTENTS = {
    "formula_knowledge",
    "herb_indication",
    "classical_theory",
    "acupuncture_principle",
}


REJECTION_REASON_MESSAGES = {
    "unknown_intent": "无法确认问题属于当前本地中医知识库可回答的范围。",
    "no_grounded_match": "Qdrant 中没有检索到同时满足本地证据约束的条目。",
    "collection_missing": "结构化方证集合不存在或尚未完成索引。",
    "missing_required_knowledge_groups": "候选知识条目的必要证据字段没有完整命中。",
    "insufficient_knowledge_match": "当前命中的知识要点过少，不能据此形成可靠回答。",
    "translator_needs_more_info": "问题中的症状信息不足，Query Translator 要求继续追问。",
    "no_canonical_terms": "没有从原问题中识别出可由本地 payload 核验的规范术语。",
    "negative_evidence_conflict": "用户明确否认的症状与候选条目发生冲突。",
    "forbidden_evidence_conflict": "候选条目包含与当前问题相冲突的排除性症状。",
    "missing_required_symptom_groups": "候选方证的必要鉴别症状尚未确认。",
    "payload_evidence_confidence_below_threshold": "候选 payload 的证据可信度低于门控阈值。",
    "too_few_payload_matches": "命中的 payload 诊断要素数量不足。",
    "low_query_coverage": "候选条目只能解释用户问题中的少部分表征。",
    "competing_indistinguishable_matches": "存在多个证据接近的候选，当前信息不足以区分。",
    "direct_knowledge_evidence_match": "本地知识条目与查询要点直接匹配。",
    "payload_evidence_requirements_satisfied": "候选 payload 已满足当前证据门控要求。",
}


def explain_reason(code: str) -> str:
    if code.startswith("retrieval_failure:"):
        failure_type = code.split(":", 1)[1] or "unknown"
        return f"检索组件运行失败（{failure_type}），为避免无依据回答已停止生成。"
    return REJECTION_REASON_MESSAGES.get(code, f"证据门控未通过（{code}）。")


def decision_result(status: str, reasons: list[str], **extra: Any) -> dict[str, Any]:
    unique_reasons = list(dict.fromkeys(str(reason) for reason in reasons if str(reason)))
    details = [{"code": code, "message": explain_reason(code)} for code in unique_reasons]
    if status == "grounded_answer":
        response_policy = "answer"
        summary = "本地证据满足回答条件。"
    elif status == "clarify":
        response_policy = "clarify"
        summary = "当前暂不提供方药或治疗结论，需要补充信息后重新检索。"
    else:
        response_policy = "refuse"
        summary = "当前拒绝回答，以避免使用本地数据库之外的知识补全。"
    return {
        "status": status,
        "reasons": unique_reasons,
        "reason_details": details,
        "response_policy": response_policy,
        "rejection": {
            "required": status != "grounded_answer",
            "mode": response_policy,
            "primary_reason": unique_reasons[0] if unique_reasons else "",
            "summary": summary,
        },
        **extra,
    }


def intervention_identity(payload: dict[str, Any]) -> str:
    return str(
        payload.get("formula")
        or payload.get("herb_name")
        or payload.get("intervention_name")
        or payload.get("title")
        or payload.get("entry_id")
        or ""
    )


def evidence_confidence(payload: dict[str, Any]) -> float:
    if str(payload.get("review_status", "")) in {"human_verified", "expert_verified"}:
        return 1.0
    try:
        # Legacy payloads predate the confidence field. Current rebuilt payloads
        # always carry it; the neutral fallback keeps migration deterministic.
        return float(payload["confidence"]) if "confidence" in payload else 0.6
    except (TypeError, ValueError):
        return 0.0


def payload_rank_signature(query_info: dict[str, Any], match: dict[str, Any]) -> int:
    """Return a source-agnostic evidence completeness signal for local ranking."""
    if query_info.get("query_intent") != "clinical_symptom":
        return 0
    if match.get("negative_conflicts") or match.get("forbidden_conflicts"):
        return 0
    required_groups = match.get("matched_required_symptom_groups", []) + match.get(
        "missing_required_symptom_groups", []
    )
    required_complete = not required_groups or float(match.get("required_group_coverage", 0.0)) >= 1.0
    canonical_matches = int(
        match.get(
            "primary_canonical_match_count",
            match.get("canonical_match_count", len(match.get("matched_terms", []))),
        )
    )
    query_term_count = len(query_info.get("primary_canonical_terms") or query_info.get("canonical_terms") or [])
    query_coverage = float(
        match.get("query_coverage", canonical_matches / query_term_count if query_term_count else 0.0)
    )
    if canonical_matches >= 3 and query_coverage >= 0.75:
        return 2
    return int(required_complete and query_coverage >= 0.5 and canonical_matches >= 2)


def _competing_matches(top: dict[str, Any], matches: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    top_identity = intervention_identity(top.get("payload", {}))
    top_count = int(top.get("primary_canonical_match_count", top.get("canonical_match_count", 0)))
    top_required = float(top.get("required_group_coverage", 0.0))
    return [
        item
        for item in matches[1 : limit + 1]
        if intervention_identity(item.get("payload", {})) != top_identity
        and int(item.get("primary_canonical_match_count", item.get("canonical_match_count", 0))) >= max(1, top_count - 1)
        and float(item.get("required_group_coverage", 0.0)) >= max(0.0, top_required - 0.25)
    ]


def evaluate_evidence_gate(
    query_info: dict[str, Any],
    matches: list[dict[str, Any]],
    *,
    min_evidence_confidence: float = 0.6,
    min_query_coverage: float = 0.67,
) -> dict[str, Any]:
    intent = str(query_info.get("query_intent", "unknown"))
    if intent == "unknown":
        return decision_result("no_match", ["unknown_intent"])
    if not matches:
        status = "clarify" if intent == "clinical_symptom" and query_info.get("canonical_terms") else "no_match"
        return decision_result(status, ["no_grounded_match"])

    top = matches[0]
    payload = top.get("payload", {})
    identity = intervention_identity(payload)
    if intent in KNOWLEDGE_INTENTS:
        canonical_count = int(top.get("canonical_match_count", 0))
        missing_required = top.get("missing_required_symptom_groups", [])
        if missing_required:
            return decision_result(
                "clarify",
                ["missing_required_knowledge_groups"],
                top_identity=identity,
            )
        if int(top.get("exact_match_count", 0)) > 0 or canonical_count >= 2:
            return decision_result(
                "grounded_answer",
                ["direct_knowledge_evidence_match"],
                top_identity=identity,
            )
        status = "clarify" if canonical_count else "no_match"
        return decision_result(status, ["insufficient_knowledge_match"], top_identity=identity)

    reasons: list[str] = []
    if query_info.get("needs_more_info"):
        reasons.append("translator_needs_more_info")
    if not query_info.get("canonical_terms"):
        reasons.append("no_canonical_terms")
    if top.get("negative_conflicts"):
        reasons.append("negative_evidence_conflict")
    if top.get("forbidden_conflicts"):
        reasons.append("forbidden_evidence_conflict")

    required_groups = top.get("matched_required_symptom_groups", []) + top.get(
        "missing_required_symptom_groups", []
    )
    if required_groups and float(top.get("required_group_coverage", 0.0)) < 1.0:
        reasons.append("missing_required_symptom_groups")
    confidence = evidence_confidence(payload)
    if confidence < min_evidence_confidence:
        reasons.append("payload_evidence_confidence_below_threshold")

    primary_match_count = int(top.get("primary_canonical_match_count", top.get("canonical_match_count", 0)))
    minimum_matches = 2 if required_groups else 3
    if primary_match_count < minimum_matches:
        reasons.append("too_few_payload_matches")
    if float(top.get("query_coverage", 0.0)) < min_query_coverage:
        reasons.append("low_query_coverage")

    competing = _competing_matches(top, matches)
    if competing:
        top_terms = set(top.get("matched_terms", []))
        competing_terms = set().union(*(set(item.get("matched_terms", [])) for item in competing))
        distinctive = {term for term in top_terms - competing_terms if len(str(term)) >= 2}
        if not distinctive:
            reasons.append("competing_indistinguishable_matches")

    return decision_result(
        "clarify" if reasons else "grounded_answer",
        reasons or ["payload_evidence_requirements_satisfied"],
        top_identity=identity,
        required_canonical_matches=minimum_matches,
        evidence_confidence=confidence,
        gate_policy="payload_evidence_v1",
    )
