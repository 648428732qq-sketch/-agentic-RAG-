from __future__ import annotations

import threading
import time
import os
from collections import OrderedDict
from typing import Any

import config
from core.evidence_gate import decision_result, evaluate_evidence_gate, payload_rank_signature
from core.hybrid_retrieval import (
    LocalBM25Index,
    collect_candidate_terms,
    collect_global_candidate_terms,
    collect_payload_evidence_terms,
    payload_contains_term,
    reciprocal_rank_fusion,
)
from core.query_translator_biencoder import QueryTranslatorBiEncoder
from core.syndrome_reranker import CrossEncoderReranker, build_rerank_query
from core.symptom_query_translator import infer_direct_candidate_terms, looks_like_symptom_query, translate_symptom_query
from core.syndrome_terms import clean_text, symptom_overlap_score, unique


def _debug_stage(message: str) -> None:
    if os.environ.get("QUERY_TRANSLATOR_DEBUG") == "1":
        print(f"[hybrid-retriever] {message}", flush=True)
    debug_file = os.environ.get("QUERY_TRANSLATOR_DEBUG_FILE")
    if debug_file:
        with open(debug_file, "a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} hybrid-retriever {message}\n")


STRUCTURED_ANSWER_MIN_OVERLAP = {
    "formula_syndrome": 3,
    "classical_clause": 2,
    "classical_acupuncture": 3,
    "classical_acupuncture_principle": 1,
    "herb_indication": 1,
    "classical_theory": 2,
}
CLINICAL_MIN_CANONICAL_MATCHES = {
    "formula_syndrome": 3,
    "classical_clause": 2,
    "classical_acupuncture": 3,
    "herb_indication": 2,
}
MISSING_REQUIRED_RETAIN_MIN_QUERY_COVERAGE = 0.8
WEAK_CLINICAL_TERMS = {
    "疼痛", "腹痛", "心痛", "胃脘当心而痛", "胃脘", "胃脘痞闷",
    "腹䐜胀", "腹胀满", "心下痞", "心下痞硬", "胸痞", "胸满",
}
STRICT_DIRECT_CLINICAL_SIGNATURES = (
    (
        {"恶寒"},
        {"身疼", "肢体酸楚疼痛", "四肢酸疼", "头身疼痛", "骨节疼痛"},
        {"无汗"},
        {"喘", "喘咳", "咳喘"},
    ),
    (
        {"恶寒"},
        {"头痛"},
        {"无汗"},
        {"喘", "喘咳", "咳喘"},
    ),
)
DIRECT_CLINICAL_SIGNATURES = (
    (
        {"痰涎清稀", "清稀色白"},
        {"不得平卧", "不得卧"},
        {"咳嗽", "喘", "喘咳", "咳喘", "咳逆"},
    ),
)
RANK_ONLY_CLINICAL_SIGNATURES = (
    (
        {"吐后"},
        {"渴欲得水", "渴欲饮水", "口渴", "欲饮水"},
    ),
)
THEORY_QUERY_MARKERS = (
    "为什么", "为何", "何谓", "什么意思", "怎么理解", "理论", "脉", "诊", "切脉", "经络",
    "十二经", "寸口", "脏腑", "五脏", "六腑", "阴阳", "虚实", "补泻", "难经",
)
ACUPUNCTURE_PRINCIPLE_QUERY_MARKERS = (
    "针刺", "针法", "针道", "九针", "补泻", "迎随", "得气", "气至", "留针",
    "浅刺", "深刺", "刺荣", "刺卫", "灸法",
)
SHORT_CLINICAL_RANK_TERMS = {"喘", "咳", "渴", "厥", "痰", "脉", "呕", "利"}
NEGATED_EVIDENCE_COMPATIBLE_TERMS = {
    "口渴": {"不渴", "不口渴", "没有明显口渴", "没口渴"},
    "汗出": {"无汗", "不出汗", "没有汗", "没汗", "汗不出"},
    "有汗": {"无汗", "不出汗", "没有汗", "没汗", "汗不出"},
    "自汗": {"无汗", "不出汗", "没有汗", "没汗", "汗不出"},
    "无汗": {"汗出", "有汗", "自汗", "汗大出", "多汗"},
}


def _is_rankable_term(term: str, max_len: int) -> bool:
    return (1 < len(term) <= max_len) or term in SHORT_CLINICAL_RANK_TERMS


def merge_biencoder_predictions(
    query_info: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    min_score: float,
    max_terms: int,
    primary_terms: int,
) -> dict[str, Any]:
    """Merge only local-catalog predictions while preserving explicit negatives."""
    result = dict(query_info)
    existing = list(result.get("canonical_terms", []))
    primary = list(result.get("primary_canonical_terms") or existing)
    negatives = set(result.get("negative_terms", []))
    accepted: list[dict[str, Any]] = []
    for prediction in predictions:
        term = clean_text(str(prediction.get("canonical_term", "")))
        try:
            score = float(prediction.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not term or term in negatives or score < min_score:
            continue
        accepted.append({**prediction, "canonical_term": term, "score": round(score, 6)})
        if len(accepted) >= max(0, max_terms):
            break
    if not accepted:
        result["biencoder_predictions"] = []
        return result

    predicted_terms = [item["canonical_term"] for item in accepted]
    result["canonical_terms"] = unique(existing + predicted_terms)
    if len(primary) < max(1, primary_terms):
        primary = unique(primary + predicted_terms[: max(0, primary_terms - len(primary))])
    result["primary_canonical_terms"] = primary
    result["biencoder_predictions"] = accepted
    result["translation_method"] = str(result.get("translation_method", "local_alias")) + "+biencoder_v3"
    if result.get("query_intent") == "clinical_symptom" and primary:
        result["needs_more_info"] = False
    expanded = str(result.get("expanded_query", "")).strip()
    result["expanded_query"] = clean_text(
        "\n".join(part for part in (expanded, "双塔候选 " + " ".join(predicted_terms)) if part)
    )
    return result


def merge_direct_payload_terms(
    query_info: dict[str, Any],
    direct_terms: list[str],
) -> dict[str, Any]:
    """Attach literal local-payload evidence without promoting derived aliases.

    Query Translator may expand one surface phrase into several related terms.
    Those expansions remain useful recall terms, but query coverage must be
    measured against the terms literally present in the user's query.
    """
    result = dict(query_info)
    negative_terms = set(result.get("negative_terms", []))
    literal_terms = [term for term in unique(direct_terms) if term not in negative_terms]
    result["literal_payload_terms"] = literal_terms
    if literal_terms:
        result["canonical_terms"] = unique(list(result.get("canonical_terms", [])) + literal_terms)
    return result


def should_use_structured_answer(result: dict[str, Any]) -> bool:
    decision = result.get("decision", {})
    if decision:
        return decision.get("status") == "grounded_answer"
    matches = result.get("matches", [])
    if not matches:
        return False
    query = result.get("query", {})
    if query.get("query_intent") == "unknown":
        return False
    top = matches[0]
    if top.get("negative_conflicts"):
        return False
    payload = top.get("payload", {})
    source_type = payload.get("source_type", "")
    if top.get("exact_match_count", 0) > 0 and query.get("query_intent") in {
        "formula_knowledge",
        "herb_indication",
        "classical_theory",
        "acupuncture_principle",
    }:
        return True
    min_overlap = STRUCTURED_ANSWER_MIN_OVERLAP.get(source_type, 4)
    return top.get("overlap_score", 0) >= min_overlap


def should_request_structured_clarification(result: dict[str, Any]) -> bool:
    query = result.get("query", {})
    return (
        query.get("query_intent") == "clinical_symptom"
        and result.get("decision", {}).get("status") == "clarify"
    )


def should_refuse_ungrounded_local_query(result: dict[str, Any]) -> bool:
    if not bool(getattr(config, "STRICT_LOCAL_EVIDENCE_MODE", True)):
        return False
    query = result.get("query", {})
    decision = result.get("decision", {})
    return (
        decision.get("status") == "no_match"
        and query.get("query_intent") in {"clinical_symptom", "unknown"}
    )


def format_local_no_match(result: dict[str, Any]) -> str:
    query = result.get("query", {})
    decision = result.get("decision", {})
    recognized = _join(query.get("canonical_terms", []), default="未识别出可核验的本地规范表征")
    lines = [
        "很抱歉，本地知识库中没有找到足够证据回答这个问题。",
        "",
        f"- **已识别内容**：{recognized}",
        f"- **处理结果**：{decision.get('rejection', {}).get('summary', '当前拒绝回答。')}",
        "- **拒答原因**：",
    ]
    details = decision.get("reason_details", [])
    if details:
        lines.extend(f"  - `{item.get('code', 'unknown')}`：{item.get('message', '')}" for item in details)
    else:
        lines.append("  - `unknown`：未取得可核验的本地证据。")
    lines.append("- 系统不会使用模型常识补充本地数据库中不存在的病症、病因或方药。")
    return "\n".join(lines)


def _list_values(payload: dict[str, Any], field: str) -> list[str]:
    value = payload.get(field, [])
    if isinstance(value, list):
        return [clean_text(str(item)) for item in value if clean_text(str(item))]
    return [clean_text(str(value))] if value else []


def _diagnostic_terms(payload: dict[str, Any]) -> list[str]:
    terms = _list_values(payload, "diagnostic_keys")
    if not terms:
        terms = _list_values(payload, "ancient_symptoms")
    return list(dict.fromkeys(term for term in terms if _is_rankable_term(term, 16)))


def _differential_terms(payload: dict[str, Any]) -> list[str]:
    terms = _list_values(payload, "differential_keys")
    return list(dict.fromkeys(term for term in terms if _is_rankable_term(term, 24)))


def _matched_terms(payload: dict[str, Any], terms: list[str]) -> list[str]:
    return list(dict.fromkeys(term for term in terms if term and payload_contains_term(payload, term)))


def _payload_contains_evidence_term(payload: dict[str, Any], term: str) -> bool:
    """Match terms against positive evidence, not exclusion-only metadata."""
    required_terms = {item for group in _required_symptom_groups(payload) for item in group}
    forbidden_terms = set(str(item) for item in payload.get("forbidden_terms", []) or [])
    compatible_terms = NEGATED_EVIDENCE_COMPATIBLE_TERMS.get(term, set())
    if compatible_terms & required_terms or term in forbidden_terms:
        return False
    evidence_payload = dict(payload)
    evidence_payload.pop("forbidden_terms", None)
    evidence_payload.pop("must_clarify_fields", None)
    return payload_contains_term(evidence_payload, term)


def _required_symptom_groups(payload: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for raw_group in payload.get("required_symptom_groups", []) or []:
        if isinstance(raw_group, list):
            group = [clean_text(str(term)) for term in raw_group if clean_text(str(term))]
        else:
            group = [clean_text(str(raw_group))]
        if group:
            groups.append(unique(group))
    return groups


def _matched_required_groups(payload: dict[str, Any], query_terms: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    term_set = set(query_terms)
    matched: list[list[str]] = []
    missing: list[list[str]] = []
    for group in _required_symptom_groups(payload):
        if term_set.intersection(group):
            matched.append(group)
        else:
            missing.append(group)
    return matched, missing


def _forbidden_term_conflicts(payload: dict[str, Any], query_terms: list[str]) -> list[str]:
    forbidden = set(str(term) for term in payload.get("forbidden_terms", []) or [])
    return unique([term for term in query_terms if term in forbidden])


def _intervention_identity(payload: dict[str, Any]) -> str:
    return str(
        payload.get("formula")
        or payload.get("herb_name")
        or payload.get("intervention_name")
        or payload.get("title")
        or payload.get("entry_id")
        or ""
    )


def _covers_signature(terms: set[str], signature: tuple[set[str], ...]) -> bool:
    return all(bool(group & terms) for group in signature)


def _has_clinical_signature(
    query_info: dict[str, Any],
    top: dict[str, Any],
    signatures: tuple[tuple[set[str], ...], ...],
    *,
    allow_missing_required: bool = False,
) -> bool:
    if query_info.get("negative_terms"):
        return False
    payload = top.get("payload", {})
    source_type = payload.get("source_type", "")
    if source_type not in {"formula_syndrome", "classical_clause"}:
        return False
    if not allow_missing_required and top.get("missing_required_symptom_groups"):
        return False
    query_terms = set(query_info.get("canonical_terms", []))
    matched_terms = set(top.get("matched_terms", []))
    return any(
        _covers_signature(query_terms, signature) and _covers_signature(matched_terms, signature)
        for signature in signatures
    )


def _is_direct_clinical_signature(query_info: dict[str, Any], top: dict[str, Any]) -> bool:
    return _has_clinical_signature(
        query_info,
        top,
        DIRECT_CLINICAL_SIGNATURES,
        allow_missing_required=True,
    ) or _has_clinical_signature(
        query_info,
        top,
        STRICT_DIRECT_CLINICAL_SIGNATURES,
    )


def _is_rank_clinical_signature(query_info: dict[str, Any], top: dict[str, Any]) -> int:
    return payload_rank_signature(query_info, top)


def _build_retrieval_decision(query_info: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_evidence_gate(
        query_info,
        matches,
        min_evidence_confidence=float(getattr(config, "SYNDROME_MIN_EVIDENCE_CONFIDENCE", 0.6)),
        min_query_coverage=float(getattr(config, "SYNDROME_MIN_QUERY_COVERAGE", 0.67)),
    )


def is_theory_query(query_info: dict[str, Any]) -> bool:
    query = str(query_info.get("original_query", ""))
    terms = " ".join(str(term) for term in query_info.get("canonical_terms", []))
    return any(marker in query or marker in terms for marker in THEORY_QUERY_MARKERS)


def is_acupuncture_principle_query(query_info: dict[str, Any]) -> bool:
    query = str(query_info.get("original_query", ""))
    terms = " ".join(str(term) for term in query_info.get("canonical_terms", []))
    return any(marker in query or marker in terms for marker in ACUPUNCTURE_PRINCIPLE_QUERY_MARKERS)


def source_priority(payload: dict[str, Any], query_info: dict[str, Any]) -> int:
    source_type = payload.get("source_type", "")
    intent = query_info.get("query_intent")
    if intent == "herb_indication":
        return 6 if source_type == "herb_indication" else 0
    if intent == "classical_theory":
        if is_theory_query(query_info) and source_type == "classical_theory":
            return 6
        return 6 if source_type == "classical_theory" else 0
    if intent == "acupuncture_principle":
        if is_acupuncture_principle_query(query_info) and source_type == "classical_acupuncture_principle":
            return 6
        return 6 if source_type == "classical_acupuncture_principle" else 0
    if intent == "formula_knowledge":
        if source_type == "formula_syndrome":
            return 6
        if source_type == "classical_clause":
            return 5
        if payload.get("intervention_type") == "formula":
            return 4
        return 0
    if intent == "clinical_symptom":
        if source_type == "formula_syndrome":
            return 5
        if source_type == "classical_acupuncture":
            return 5
        if source_type == "classical_clause":
            return 4
        return 0
    if source_type == "formula_syndrome":
        return 4
    if source_type == "classical_clause":
        return 3
    if payload.get("intervention_type") == "formula":
        return 2
    if source_type == "classical_theory":
        return 1
    return 0


def explanatory_priority(payload: dict[str, Any], query_info: dict[str, Any]) -> int:
    source_type = payload.get("source_type", "")
    if source_type not in {"classical_acupuncture_principle", "classical_theory"}:
        return 0

    text = " ".join(
        str(payload.get(key, ""))
        for key in (
            "syndrome_name",
            "theory_topic",
            "theory_question",
            "theory_answer",
            "acupuncture_principle",
            "evidence",
        )
        if payload.get(key)
    )
    direct_fields = " ".join(
        str(payload.get(key, ""))
        for key in (
            "theory_question",
            "theory_answer",
            "acupuncture_principle",
            "evidence",
        )
        if payload.get(key)
    )
    terms = [str(term) for term in query_info.get("canonical_terms", []) if len(str(term)) > 1]
    original_query = str(query_info.get("original_query", ""))
    theory_question = str(payload.get("theory_question", ""))

    bonus = sum(1 for term in terms if term in direct_fields)
    if source_type == "classical_theory" and terms and all(term in theory_question for term in terms):
        bonus += 4
    for phrase in ("独取寸口", "寸口脉平", "针有补泻"):
        if phrase in original_query and phrase in direct_fields:
            bonus += 5
    if "所谓" in text:
        bonus += 3
    if "故曰" in text or "故谓" in text:
        bonus += 2
    if "何谓" in text:
        bonus += 1
    if "经言" in text and not any(marker in text for marker in ("所谓", "故曰", "何谓")):
        bonus -= 1
    return bonus


def local_rank_key(query_info: dict[str, Any], item: dict[str, Any]) -> tuple[Any, ...]:
    required_group_coverage = float(item.get("required_group_coverage", 0.0))
    query_coverage = float(item.get("query_coverage", 0.0))
    canonical_match_count = int(item.get("canonical_match_count", 0))
    return (
        _is_rank_clinical_signature(query_info, item),
        source_priority(item["payload"], query_info),
        int(required_group_coverage >= 1.0 and query_coverage >= 0.5),
        int(query_coverage >= 0.75 and canonical_match_count >= 3),
        query_coverage,
        int(required_group_coverage >= 1.0),
        required_group_coverage,
        canonical_match_count,
        item.get("required_group_match_count", 0),
        len(item["matched_diagnostic_terms"]),
        float(item.get("diagnostic_coverage", 0.0)),
        len(item.get("matched_differential_terms", [])),
        float(item.get("differential_coverage", 0.0)),
        item["specificity_score"],
        item["exact_match_count"],
        explanatory_priority(item["payload"], query_info),
        item["overlap_score"],
        item["route_count"],
        item["rrf_score"],
        item["score"],
    )


def rerank_evidence_key(query_info: dict[str, Any], item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _is_rank_clinical_signature(query_info, item),
        item["canonical_match_count"],
        len(item["matched_diagnostic_terms"]),
        int(float(item.get("query_coverage", 0.0)) >= 1.0),
        source_priority(item["payload"], query_info),
        explanatory_priority(item["payload"], query_info),
    )


def _should_retain_missing_required_candidate(query_info: dict[str, Any], match: dict[str, Any]) -> bool:
    payload = match.get("payload", {})
    return (
        query_info.get("query_intent") == "clinical_symptom"
        and payload.get("source_type") in {"formula_syndrome", "classical_clause"}
        and bool(match.get("missing_required_symptom_groups"))
        and not match.get("negative_conflicts")
        and not match.get("forbidden_conflicts")
        and float(match.get("query_coverage", 0.0)) >= MISSING_REQUIRED_RETAIN_MIN_QUERY_COVERAGE
        and int(match.get("primary_canonical_match_count", match.get("canonical_match_count", 0))) >= 3
    )


def _retain_high_coverage_missing_candidates(
    selected: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    query_info: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not query_info or query_info.get("query_intent") != "clinical_symptom" or not selected:
        return selected
    selected_identities = {_intervention_identity(match.get("payload", {})) for match in selected}
    for candidate in matches:
        identity = _intervention_identity(candidate.get("payload", {}))
        if not identity or identity in selected_identities:
            continue
        if not _should_retain_missing_required_candidate(query_info, candidate):
            continue
        replaceable_indexes = range(1, len(selected)) if len(selected) > 1 else range(len(selected))
        replace_index = max(
            replaceable_indexes,
            key=lambda index: (
                not _should_retain_missing_required_candidate(query_info, selected[index]),
                -float(selected[index].get("query_coverage", 0.0)),
                -int(selected[index].get("primary_canonical_match_count", selected[index].get("canonical_match_count", 0))),
            ),
        )
        replace_target = selected[replace_index]
        if _should_retain_missing_required_candidate(query_info, replace_target):
            continue
        if float(replace_target.get("query_coverage", 0.0)) >= float(candidate.get("query_coverage", 0.0)):
            continue
        selected_identities.discard(_intervention_identity(replace_target.get("payload", {})))
        selected[replace_index] = candidate
        selected_identities.add(identity)
    return selected


def _duplicate_preference_key(match: dict[str, Any], query_info: dict[str, Any] | None) -> tuple[Any, ...]:
    if not query_info:
        return (0,)
    return (
        source_priority(match.get("payload", {}), query_info),
        _is_rank_clinical_signature(query_info, match),
        int(not match.get("missing_required_symptom_groups")),
        float(match.get("query_coverage", 0.0)),
        int(match.get("primary_canonical_match_count", match.get("canonical_match_count", 0))),
        int(match.get("canonical_match_count", 0)),
        float(match.get("required_group_coverage", 0.0)),
        int(match.get("required_group_match_count", 0)),
    )


def diversify_matches(
    matches: list[dict[str, Any]],
    limit: int,
    query_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prefer the best evidence per intervention before showing duplicates."""
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    selected_by_identity: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for match in matches:
        identity = _intervention_identity(match.get("payload", {}))
        if identity and identity in seen:
            selected_index = selected_by_identity.get(identity)
            if (
                selected_index is not None
                and _duplicate_preference_key(match, query_info)
                > _duplicate_preference_key(selected[selected_index], query_info)
            ):
                deferred.append(selected[selected_index])
                selected[selected_index] = match
            else:
                deferred.append(match)
            continue
        if identity:
            seen.add(identity)
        if len(selected) < limit:
            if identity:
                selected_by_identity[identity] = len(selected)
            selected.append(match)
        else:
            deferred.append(match)
    for match in deferred:
        if len(selected) >= limit:
            break
        selected.append(match)
    return _retain_high_coverage_missing_candidates(selected, matches, query_info)


class SyndromeRetriever:
    def __init__(self, vector_db_manager=None, *, client=None, embedding=None):
        self._vector_db_manager = vector_db_manager
        self._client = client or getattr(vector_db_manager, "_VectorDbManager__client", None)
        self._embedding = embedding or getattr(vector_db_manager, "_VectorDbManager__dense_embeddings", None)
        self._owns_client = self._client is None
        self._llm = None
        self._payload_cache: list[dict[str, Any]] = []
        self._global_candidate_terms: list[str] = []
        self._payload_term_index: dict[str, list[dict[str, Any]]] = {}
        self._lexical_index: LocalBM25Index | None = None
        self._payload_cache_loaded_at = 0.0
        self._payload_cache_lock = threading.Lock()
        self._translation_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._translator_failures = 0
        self._translator_disabled_until = 0.0
        self._reranker: CrossEncoderReranker | None = None
        self._biencoder: QueryTranslatorBiEncoder | None = None

    def set_llm(self, llm) -> None:
        self._llm = llm

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_runtime(self):
        owns_client = False
        if self._embedding is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_kwargs = {"local_files_only": config.EMBEDDING_LOCAL_FILES_ONLY}
            embedding_device = str(getattr(config, "EMBEDDING_DEVICE", "")).strip().lower()
            if embedding_device and embedding_device != "auto":
                model_kwargs["device"] = embedding_device

            self._embedding = HuggingFaceEmbeddings(
                model_name=config.DENSE_MODEL,
                model_kwargs=model_kwargs,
            )
        if self._client is None:
            from db.qdrant_client_factory import create_qdrant_client

            self._client = create_qdrant_client()
            owns_client = True
            self._owns_client = True
        return self._client, self._embedding, owns_client

    def _ensure_payload_index(self, client) -> LocalBM25Index:
        ttl = getattr(config, "SYNDROME_PAYLOAD_CACHE_TTL_SECONDS", 300)
        now = time.monotonic()
        if self._lexical_index is not None and now - self._payload_cache_loaded_at < ttl:
            return self._lexical_index
        with self._payload_cache_lock:
            now = time.monotonic()
            if self._lexical_index is not None and now - self._payload_cache_loaded_at < ttl:
                return self._lexical_index
            payloads: list[dict[str, Any]] = []
            next_offset = None
            while True:
                records, next_offset = client.scroll(
                    collection_name=config.SYNDROME_COLLECTION,
                    offset=next_offset,
                    limit=256,
                    with_payload=True,
                    with_vectors=False,
                )
                payloads.extend(record.payload or {} for record in records if record.payload)
                if not next_offset:
                    break
            self._payload_cache = payloads
            self._global_candidate_terms = collect_global_candidate_terms(
                payloads,
                limit=getattr(config, "SYNDROME_TRANSLATOR_MAX_TERMS", 800),
            )
            term_index: dict[str, list[dict[str, Any]]] = {}
            for payload in payloads:
                for term in collect_payload_evidence_terms([payload]):
                    term_index.setdefault(term, []).append(payload)
            self._payload_term_index = term_index
            self._lexical_index = LocalBM25Index(payloads)
            self._payload_cache_loaded_at = time.monotonic()
            return self._lexical_index

    @staticmethod
    def _dense_search(client, embedding, query: str, limit: int) -> list[dict[str, Any]]:
        normalized_query = clean_text(str(query))
        if not normalized_query:
            return []
        vector = embedding.embed_query(normalized_query)
        response = client.query_points(
            collection_name=config.SYNDROME_COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "entry_id": str((point.payload or {}).get("entry_id", point.id)),
                "score": float(point.score),
                "payload": point.payload or {},
            }
            for point in response.points
            if point.payload
        ]

    @staticmethod
    def _lexical_search(index: LocalBM25Index, query: str, limit: int) -> list[dict[str, Any]]:
        return [
            {
                "entry_id": hit.entry_id,
                "score": hit.score,
                "exact_matches": hit.exact_matches,
                "payload": hit.payload,
            }
            for hit in index.search(query, limit=limit)
        ]

    @staticmethod
    def _payload_overlap_search(
        payloads: list[dict[str, Any]],
        term_index: dict[str, list[dict[str, Any]]],
        query_info: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        positive_terms = list(query_info.get("canonical_terms", []))
        primary_terms = list(query_info.get("primary_canonical_terms") or positive_terms)
        if not positive_terms:
            return []

        candidate_payloads: dict[str, dict[str, Any]] = {}
        for term in unique(primary_terms + positive_terms):
            for payload in term_index.get(term, []):
                entry_id = str(payload.get("entry_id", "")).strip()
                if entry_id:
                    candidate_payloads[entry_id] = payload
        scan_payloads = list(candidate_payloads.values()) if candidate_payloads else payloads
        hits: list[dict[str, Any]] = []
        primary_count = max(len(primary_terms), 1)
        for payload in scan_payloads:
            entry_id = str(payload.get("entry_id", "")).strip()
            if not entry_id or source_priority(payload, query_info) <= 0:
                continue
            matched_terms = [term for term in positive_terms if _payload_contains_evidence_term(payload, term)]
            if not matched_terms:
                continue
            primary_matched = [
                term for term in primary_terms if _payload_contains_evidence_term(payload, term)
            ]
            if not primary_matched:
                continue
            required_groups = _required_symptom_groups(payload)
            matched_required, _ = _matched_required_groups(payload, positive_terms)
            query_coverage = len(primary_matched) / primary_count
            required_coverage = len(matched_required) / len(required_groups) if required_groups else 0.0
            score = (
                len(primary_matched) * 10.0
                + len(matched_terms) * 2.0
                + query_coverage * 5.0
                + required_coverage
                + source_priority(payload, query_info)
            )
            hits.append(
                {
                    "entry_id": entry_id,
                    "score": round(score, 6),
                    "exact_matches": 0,
                    "payload": payload,
                }
            )
        hits.sort(
            key=lambda item: (
                item["score"],
                source_priority(item["payload"], query_info),
                len(str(item["payload"].get("title", ""))),
            ),
            reverse=True,
        )
        return hits[:limit]

    def _translate(self, query: str, candidate_terms: list[str], should_call_llm: bool) -> dict[str, Any]:
        cache_key = query + "\n" + "|".join(candidate_terms)
        if cache_key in self._translation_cache:
            cached = self._translation_cache.pop(cache_key)
            self._translation_cache[cache_key] = cached
            return dict(cached)

        now = time.monotonic()
        circuit_open = now < self._translator_disabled_until
        llm = self._llm if should_call_llm and not circuit_open else None
        result = translate_symptom_query(
            query,
            llm=llm,
            force_llm=bool(llm),
            candidate_terms=candidate_terms,
        )
        if circuit_open:
            result["translation_errors"] = list(result.get("translation_errors", [])) + ["llm_circuit_open"]
        failed = any(str(error).startswith("llm_failure:") for error in result.get("translation_errors", []))
        if failed:
            self._translator_failures += 1
            threshold = getattr(config, "LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD", 3)
            if self._translator_failures >= threshold:
                self._translator_disabled_until = now + getattr(config, "LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS", 60)
        elif result.get("llm_used"):
            self._translator_failures = 0
            self._translator_disabled_until = 0.0

        self._translation_cache[cache_key] = dict(result)
        cache_size = getattr(config, "LLM_SYMPTOM_TRANSLATOR_CACHE_SIZE", 128)
        while len(self._translation_cache) > cache_size:
            self._translation_cache.popitem(last=False)
        return result

    def _ensure_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                getattr(config, "SYNDROME_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
                device=getattr(config, "SYNDROME_RERANK_DEVICE", "") or None,
                cache_folder=getattr(config, "SYNDROME_RERANK_CACHE_FOLDER", "") or None,
                local_files_only=bool(getattr(config, "SYNDROME_RERANK_LOCAL_FILES_ONLY", True)),
                max_length=int(getattr(config, "SYNDROME_RERANK_MAX_LENGTH", 512)),
                batch_size=int(getattr(config, "SYNDROME_RERANK_BATCH_SIZE", 16)),
                trust_remote_code=bool(getattr(config, "SYNDROME_RERANK_TRUST_REMOTE_CODE", False)),
            )
        return self._reranker

    def _ensure_biencoder(self) -> QueryTranslatorBiEncoder:
        if self._biencoder is None:
            self._biencoder = QueryTranslatorBiEncoder(
                getattr(config, "QUERY_TRANSLATOR_BIENCODER_MODEL"),
                getattr(config, "QUERY_TRANSLATOR_BIENCODER_CATALOG"),
                device=getattr(config, "QUERY_TRANSLATOR_BIENCODER_DEVICE", "cpu") or "cpu",
            )
        return self._biencoder

    def _augment_with_biencoder(self, query: str, query_info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not bool(getattr(config, "ENABLE_QUERY_TRANSLATOR_BIENCODER", False)):
            return query_info, {"enabled": False, "used": False}
        local_threshold = max(0, int(getattr(config, "QUERY_TRANSLATOR_BIENCODER_LOCAL_TERM_THRESHOLD", 3)))
        local_terms = list(query_info.get("primary_canonical_terms") or query_info.get("canonical_terms") or [])
        if len(local_terms) >= local_threshold:
            return query_info, {
                "enabled": True,
                "used": False,
                "reason": "local_terms_sufficient",
                "local_term_count": len(local_terms),
            }
        top_k = max(1, int(getattr(config, "QUERY_TRANSLATOR_BIENCODER_TOP_K", 5)))
        try:
            predictions = self._ensure_biencoder().predict(query, limit=top_k)
            augmented = merge_biencoder_predictions(
                query_info,
                predictions,
                min_score=float(getattr(config, "QUERY_TRANSLATOR_BIENCODER_MIN_SCORE", 0.55)),
                max_terms=top_k,
                primary_terms=max(1, int(getattr(config, "QUERY_TRANSLATOR_BIENCODER_PRIMARY_K", 3))),
            )
            accepted = augmented.get("biencoder_predictions", [])
            return augmented, {
                "enabled": True,
                "used": bool(accepted),
                "prediction_count": len(predictions),
                "accepted_count": len(accepted),
                "accepted_terms": [item.get("canonical_term", "") for item in accepted],
            }
        except Exception as exc:
            fallback = dict(query_info)
            fallback["translation_errors"] = list(fallback.get("translation_errors", [])) + [
                f"biencoder_failure:{type(exc).__name__}"
            ]
            return fallback, {
                "enabled": True,
                "used": False,
                "error": type(exc).__name__,
            }

    def _rerank_matches(self, query_info: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
        if not bool(getattr(config, "ENABLE_SYNDROME_RERANK", True)):
            return {"rerank_enabled": False, "rerank_used": False}
        allowed_intents = {
            item.strip()
            for item in str(getattr(config, "SYNDROME_RERANK_INTENTS", "clinical_symptom")).split(",")
            if item.strip()
        }
        query_intent = str(query_info.get("query_intent", "unknown"))
        if allowed_intents and query_intent not in allowed_intents:
            return {
                "rerank_enabled": True,
                "rerank_used": False,
                "reason": "intent_not_enabled",
                "query_intent": query_intent,
                "allowed_intents": sorted(allowed_intents),
            }
        candidate_count = min(
            len(matches),
            max(0, int(getattr(config, "SYNDROME_RERANK_CANDIDATES", 40))),
        )
        if candidate_count <= 1:
            return {
                "rerank_enabled": True,
                "rerank_used": False,
                "reason": "not_enough_candidates",
                "candidate_count": candidate_count,
            }

        query_text = build_rerank_query(query_info)
        reranker = self._ensure_reranker()
        scores, debug = reranker.score(query_text, matches[:candidate_count])
        debug = {"rerank_enabled": True, **debug}
        if not scores:
            return debug

        reranked: list[dict[str, Any]] = []
        for index, (match, score) in enumerate(zip(matches[:candidate_count], scores), start=1):
            updated = dict(match)
            updated["pre_rerank_rank"] = index
            updated["rerank_score"] = round(float(score), 6)
            updated["rerank_model"] = reranker.model_name
            reranked.append(updated)

        mode = str(getattr(config, "SYNDROME_RERANK_MODE", "evidence_first")).lower()
        if mode == "score_first":
            reranked.sort(
                key=lambda item: (
                    item["rerank_score"],
                    rerank_evidence_key(query_info, item),
                    local_rank_key(query_info, item),
                ),
                reverse=True,
            )
        else:
            reranked.sort(
                key=lambda item: (
                    rerank_evidence_key(query_info, item),
                    item["rerank_score"],
                    local_rank_key(query_info, item),
                ),
                reverse=True,
            )
        matches[:candidate_count] = reranked
        debug["rerank_mode"] = mode
        debug["rerank_top_score"] = reranked[0].get("rerank_score") if reranked else None
        return debug

    def search(self, query: str, limit: int = 3, candidate_limit: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        hybrid_enabled = bool(getattr(config, "ENABLE_HYBRID_SYNDROME_RETRIEVAL", True))
        _debug_stage("search:start")
        local_info = translate_symptom_query(query)
        _debug_stage("local-translation:done")
        client, embedding, owns_client = self._ensure_runtime()
        _debug_stage("runtime:ready")
        route_errors: list[str] = []
        routes: dict[str, list[dict[str, Any]]] = {}
        try:
            if not client.collection_exists(config.SYNDROME_COLLECTION):
                return {
                    "query": local_info,
                    "matches": [],
                    "decision": decision_result("no_match", ["collection_missing"]),
                    "retrieval_debug": {"errors": ["collection_missing"]},
                }

            candidate_limit = candidate_limit or max(limit * 10, 40)
            raw_limit = max(candidate_limit, getattr(config, "SYNDROME_RAW_DENSE_CANDIDATES", 40))
            lexical_limit = max(candidate_limit, getattr(config, "SYNDROME_LEXICAL_CANDIDATES", 40))
            try:
                routes["raw_dense"] = self._dense_search(client, embedding, query, raw_limit)
                _debug_stage("raw-dense:done")
            except Exception as exc:
                routes["raw_dense"] = []
                route_errors.append(f"raw_dense:{type(exc).__name__}")
            if hybrid_enabled:
                try:
                    lexical_index = self._ensure_payload_index(client)
                    routes["lexical"] = self._lexical_search(lexical_index, query, lexical_limit)
                    _debug_stage("lexical:done")
                except Exception as exc:
                    routes["lexical"] = []
                    route_errors.append(f"lexical:{type(exc).__name__}")

            pre_fused = reciprocal_rank_fusion(
                routes,
                rank_constant=getattr(config, "SYNDROME_RRF_K", 60),
            )
            translator_context_limit = getattr(config, "SYNDROME_TRANSLATOR_CONTEXT_CANDIDATES", 16)
            retrieval_candidate_terms = collect_candidate_terms(
                pre_fused[:translator_context_limit],
                limit=getattr(config, "SYNDROME_TRANSLATOR_MAX_TERMS", 80),
            )
            direct_payload_terms = infer_direct_candidate_terms(query, list(self._payload_term_index))
            candidate_terms = unique(direct_payload_terms + retrieval_candidate_terms + self._global_candidate_terms)[
                : getattr(config, "SYNDROME_TRANSLATOR_MAX_TERMS", 800)
            ]
            should_call_llm = (
                bool(getattr(config, "ENABLE_LLM_SYMPTOM_TRANSLATOR", False))
                and self._llm is not None
                and (looks_like_symptom_query(query) or local_info.get("query_intent") == "unknown")
                and (
                    bool(getattr(config, "LLM_SYMPTOM_TRANSLATOR_ALWAYS", False))
                    or len(local_info.get("canonical_terms", [])) < getattr(config, "LLM_SYMPTOM_TRANSLATOR_MIN_LOCAL_TERMS", 2)
                    or bool(local_info.get("negative_phrases"))
                )
            )
            query_info = self._translate(query, candidate_terms, should_call_llm)
            _debug_stage(f"translator:done:{query_info.get('translation_method')}")
            query_info = merge_direct_payload_terms(query_info, direct_payload_terms)
            query_info, biencoder_debug = self._augment_with_biencoder(query, query_info)
            _debug_stage(f"biencoder:done:{biencoder_debug.get('used')}")

            if hybrid_enabled and (query_info.get("canonical_terms") or query_info.get("colloquial_terms")):
                canonical_limit = max(candidate_limit, getattr(config, "SYNDROME_CANONICAL_DENSE_CANDIDATES", 40))
                try:
                    routes["canonical_dense"] = self._dense_search(
                        client,
                        embedding,
                        str(query_info.get("expanded_query", query)),
                        canonical_limit,
                    )
                    _debug_stage("canonical-dense:done")
                except Exception as exc:
                    routes["canonical_dense"] = []
                    route_errors.append(f"canonical_dense:{type(exc).__name__}")
                try:
                    payload_limit = max(candidate_limit, getattr(config, "SYNDROME_PAYLOAD_OVERLAP_CANDIDATES", candidate_limit))
                    routes["payload_overlap"] = self._payload_overlap_search(
                        self._payload_cache,
                        self._payload_term_index,
                        query_info,
                        payload_limit,
                    )
                    _debug_stage("payload-overlap:done")
                except Exception as exc:
                    routes["payload_overlap"] = []
                    route_errors.append(f"payload_overlap:{type(exc).__name__}")

            fused = reciprocal_rank_fusion(
                routes,
                rank_constant=getattr(config, "SYNDROME_RRF_K", 60),
            )
        finally:
            pass

        matches: list[dict[str, Any]] = []
        positive_terms = list(query_info.get("canonical_terms", []))
        primary_positive_terms = list(
            query_info.get("literal_payload_terms")
            or query_info.get("primary_canonical_terms")
            or positive_terms
        )
        colloquial_terms = list(query_info.get("colloquial_terms", []))
        negative_terms = list(query_info.get("negative_terms", []))
        knowledge_intents = {"formula_knowledge", "herb_indication", "classical_theory", "acupuncture_principle"}
        for item in fused:
            payload = item.get("payload", {})
            overlap = symptom_overlap_score(
                payload,
                positive_terms,
                colloquial_terms,
            )
            negative_conflicts = [
                term for term in negative_terms if _payload_contains_evidence_term(payload, term)
            ]
            forbidden_conflicts = _forbidden_term_conflicts(payload, positive_terms)
            exact = int(item.get("exact_match_count", 0))
            if negative_conflicts and query_info.get("query_intent") == "clinical_symptom":
                continue
            if (
                forbidden_conflicts
                and query_info.get("query_intent") == "clinical_symptom"
                and payload.get("source_type") in {"formula_syndrome", "classical_clause"}
            ):
                continue
            if overlap <= 0 and not (exact > 0 and query_info.get("query_intent") in knowledge_intents):
                continue
            dense_scores = item.get("dense_scores", {})
            matched_terms = _matched_terms(payload, positive_terms)
            primary_matched_terms = _matched_terms(payload, primary_positive_terms)
            matched_required_groups, missing_required_groups = _matched_required_groups(payload, positive_terms)
            diagnostic_terms = _diagnostic_terms(payload)
            differential_terms = _differential_terms(payload)
            matched_diagnostic_terms = _matched_terms(
                {"diagnostic_keys": diagnostic_terms},
                positive_terms,
            )
            matched_differential_terms = _matched_terms(
                {"differential_keys": differential_terms},
                positive_terms,
            )
            canonical_match_count = len(matched_terms)
            primary_canonical_match_count = len(primary_matched_terms)
            query_coverage = (
                primary_canonical_match_count / len(primary_positive_terms)
                if primary_positive_terms
                else 0.0
            )
            diagnostic_coverage = (
                len(matched_diagnostic_terms) / len(diagnostic_terms)
                if diagnostic_terms
                else 0.0
            )
            differential_coverage = (
                len(matched_differential_terms) / len(differential_terms)
                if differential_terms
                else 0.0
            )
            required_groups = _required_symptom_groups(payload)
            required_group_coverage = (
                len(matched_required_groups) / len(required_groups)
                if required_groups
                else 0.0
            )
            matches.append(
                {
                    "score": max(dense_scores.values(), default=0.0),
                    "overlap_score": overlap,
                    "rrf_score": item.get("rrf_score", 0.0),
                    "route_count": len(item.get("routes", {})),
                    "routes": item.get("routes", {}),
                    "dense_scores": dense_scores,
                    "lexical_score": item.get("lexical_score", 0.0),
                    "exact_match_count": exact,
                    "matched_terms": matched_terms,
                    "primary_matched_terms": primary_matched_terms,
                    "matched_required_symptom_groups": matched_required_groups,
                    "missing_required_symptom_groups": missing_required_groups,
                    "required_group_match_count": len(matched_required_groups),
                    "required_group_coverage": round(required_group_coverage, 4),
                    "canonical_match_count": canonical_match_count,
                    "primary_canonical_match_count": primary_canonical_match_count,
                    "query_coverage": round(query_coverage, 4),
                    "matched_diagnostic_terms": matched_diagnostic_terms,
                    "missing_diagnostic_terms": [
                        term for term in diagnostic_terms if term not in matched_diagnostic_terms
                    ][:12],
                    "diagnostic_coverage": round(diagnostic_coverage, 4),
                    "matched_differential_terms": matched_differential_terms,
                    "missing_differential_terms": [
                        term for term in differential_terms if term not in matched_differential_terms
                    ][:12],
                    "differential_coverage": round(differential_coverage, 4),
                    "specificity_score": sum(len(term) for term in matched_terms),
                    "negative_conflicts": negative_conflicts,
                    "forbidden_conflicts": forbidden_conflicts,
                    "payload": payload,
                }
            )

        matches.sort(key=lambda item: local_rank_key(query_info, item), reverse=True)
        rerank_debug = self._rerank_matches(query_info, matches)
        limited_matches = diversify_matches(matches, limit, query_info)
        decision = _build_retrieval_decision(query_info, limited_matches)
        if decision.get("status") == "clarify":
            query_info = dict(query_info)
            query_info["needs_more_info"] = True
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        _debug_stage(f"search:done:{elapsed_ms}ms")
        return {
            "query": query_info,
            "candidate_limit": candidate_limit,
            "matches": limited_matches,
            "decision": decision,
            "retrieval_debug": {
                "hybrid_enabled": hybrid_enabled,
                "route_counts": {name: len(values) for name, values in routes.items()},
                "fused_candidates": len(fused),
                "candidate_term_count": len(query_info.get("candidate_terms", [])),
                "llm_requested": should_call_llm,
                "llm_used": bool(query_info.get("llm_used")),
                "biencoder": biencoder_debug,
                "rerank": rerank_debug,
                "translator_failures": self._translator_failures,
                "errors": route_errors,
                "elapsed_ms": elapsed_ms,
            },
        }


def _join(values: Any, default: str = "未抽取") -> str:
    if isinstance(values, list):
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        return "、".join(cleaned) if cleaned else default
    if values:
        return str(values).strip()
    return default


def _format_composition(payload: dict[str, Any]) -> str:
    herbs = payload.get("formula_composition") or []
    formatted: list[str] = []
    for herb in herbs:
        if isinstance(herb, dict):
            name = str(herb.get("name", "")).strip()
            dose = str(herb.get("original_dose", "")).strip()
            if name and dose:
                formatted.append(f"{name}（{dose}）")
            elif name:
                formatted.append(name)
    return "、".join(formatted) if formatted else "未抽取"


def format_syndrome_card(result: dict[str, Any]) -> str:
    query = result.get("query", {})
    matches = result.get("matches", [])
    decision = result.get("decision", {})
    lines = [
        "**结构化方证匹配**",
        "",
        f"- 翻译方式：{query.get('translation_method', 'local_alias')}",
        f"- 识别出的中医表征：{_join(query.get('canonical_terms', []))}",
        f"- 识别出的口语表达：{_join(query.get('colloquial_terms', []))}",
        f"- 匹配状态：{'证据足够' if decision.get('status') == 'grounded_answer' else '需要补充信息'}",
    ]
    if query.get("body_parts"):
        lines.append(f"- 涉及部位：{_join(query.get('body_parts', []))}")
    if query.get("pathogenesis_hints"):
        lines.append(f"- 病因病机线索：{_join(query.get('pathogenesis_hints', []))}")
    for idx, match in enumerate(matches, start=1):
        payload = match.get("payload", {})
        intervention = payload.get("formula") or payload.get("intervention_name") or payload.get("treatment_method") or "未抽取"
        lines.extend(
            [
                "",
                f"{idx}. **{payload.get('title', payload.get('formula', '未知方证'))}**",
                f"   - 证候：{payload.get('syndrome_name', '未抽取')}",
                f"   - 干预/方剂：{intervention}",
                f"   - 来源类型：{payload.get('source_type', 'unknown')}",
                f"   - 规范词命中：{match.get('canonical_match_count', 0)}（{_join(match.get('matched_terms', []))}）",
                f"   - 诊断要素覆盖：{match.get('diagnostic_coverage', 0.0):.0%}",
            ]
        )
    return "\n".join(lines)


def format_syndrome_clarification(result: dict[str, Any]) -> str:
    query = result.get("query", {})
    matches = result.get("matches", [])
    decision = result.get("decision", {})
    identified = _join(query.get("canonical_terms", []), default="尚未识别出足够的规范表征")
    missing: list[str] = []
    for match in matches[:3]:
        for term in match.get("missing_diagnostic_terms", []):
            if term not in missing and term not in query.get("negative_terms", []):
                missing.append(term)
    missing = [term for term in missing if 1 < len(term) <= 12][:8]
    lines = [
        "目前证据不足，不能把这些症状唯一对应到某一个方证或治疗方案。",
        "",
        f"- **已确认的表征**：{identified}",
        f"- **暂不回答原因**：{decision.get('rejection', {}).get('summary', '证据门控未通过。')}",
    ]
    for detail in decision.get("reason_details", []):
        lines.append(f"  - `{detail.get('code', 'unknown')}`：{detail.get('message', '')}")
    if missing:
        lines.extend(
            [
                f"- **请补充是否存在**：{'、'.join(missing)}",
                "- 对每一项可直接回答“有 / 没有 / 不确定”，否定信息也会参与排除。",
            ]
        )
    else:
        lines.append("- 请补充症状部位、出现时间、寒热、出汗、舌象和脉象等可核实信息。")
    lines.append("")
    lines.append("系统不会用缺失信息补齐病因，也不会在当前证据下自动定方。")
    return "\n".join(lines)


def format_syndrome_answer(result: dict[str, Any]) -> str:
    matches = result.get("matches", [])
    if not matches:
        return ""

    query = result.get("query", {})
    top = matches[0]
    payload = top.get("payload", {})
    source = payload.get("source_book", "")
    source_url = payload.get("source_url", "")
    source_type = payload.get("source_type", "")

    if source_type == "classical_acupuncture" or payload.get("intervention_type") == "acupuncture":
        lines = [
            f"从结构化古籍库看，最接近的是 **{payload.get('title', '针刺/灸法条文')}**。",
            "",
            f"- **识别出的表征**：{_join(query.get('canonical_terms', []))}",
            f"- **古籍症状/病名线索**：{_join(payload.get('ancient_symptoms', []))}",
            f"- **针刺/灸法方法**：{payload.get('treatment_method', '未抽取') or '未抽取'}",
            f"- **穴位/经脉**：{_join(payload.get('acupoints_or_channels', []))}",
            f"- **原文依据**：{payload.get('evidence', '未抽取')}",
            "",
            "这类条目只作为古籍针刺/灸法线索，不等于现代临床操作建议。针刺、放血、灸法等必须由具备资质的医师结合禁忌证和体质执行。",
            "",
            "---",
            "**参考来源:**",
            f"- {source} {source_url}".strip(),
        ]
        return "\n".join(lines)

    if source_type == "classical_acupuncture_principle" or payload.get("intervention_type") == "acupuncture_principle":
        lines = [
            f"从结构化针法原则库看，最接近的是 **{payload.get('title', '针法原则条文')}**。",
            "",
            f"- **识别出的检索词**：{_join(query.get('canonical_terms', []))}",
            f"- **针法术语**：{_join(payload.get('acupuncture_terms', []))}",
            f"- **针刺/灸法原则**：{payload.get('acupuncture_principle', '未抽取') or '未抽取'}",
            f"- **方法摘要**：{payload.get('treatment_method', '未抽取') or '未抽取'}",
            f"- **穴位/经脉**：{_join(payload.get('acupoints_or_channels', []))}",
            f"- **原文依据**：{payload.get('evidence', '未抽取')}",
            "",
            "这类条目只作为古籍针刺/灸法原则依据，不等于现代临床操作建议。",
            "",
            "---",
            "**参考来源:**",
            f"- {source} {source_url}".strip(),
        ]
        return "\n".join(lines)

    if source_type == "herb_indication" or payload.get("intervention_type") == "herb":
        lines = [
            f"从结构化本草库看，最接近的是 **{payload.get('herb_name') or payload.get('title', '本草条文')}**。",
            "",
            f"- **识别出的表征**：{_join(query.get('canonical_terms', []))}",
            f"- **现代口语对应**：{_join(payload.get('modern_symptoms', []))}",
            f"- **古籍主治线索**：{_join(payload.get('ancient_symptoms', []))}",
            f"- **性味**：{_join(payload.get('nature_flavor', []))}",
            f"- **本草部类**：{_join([payload.get('herb_grade', ''), payload.get('herb_category', '')], default='未抽取')}",
            f"- **别名**：{_join(payload.get('herb_aliases', []))}",
            f"- **产地**：{payload.get('origin_habitat', '未抽取') or '未抽取'}",
            f"- **原文依据**：{payload.get('evidence', '未抽取')}",
        ]
        contraindications = payload.get("contraindications", "")
        if contraindications:
            lines.append(f"- **安全提示**：{contraindications}")
        lines.extend(
            [
                "",
                "这类条目只表示古籍单味药主治的检索命中，不等于可直接用药。单味药仍需要辨证、炮制、剂量、配伍和禁忌校验。",
                "",
                "---",
                "**参考来源:**",
                f"- {source} {source_url}".strip(),
            ]
        )
        return "\n".join(lines)

    if source_type == "classical_theory" or payload.get("intervention_type") == "theory":
        lines = [
            f"从结构化古籍理论库看，最接近的是 **{payload.get('title', '古籍理论条文')}**。",
            "",
            f"- **识别出的检索词**：{_join(query.get('canonical_terms', []))}",
            f"- **理论主题**：{payload.get('theory_topic', '未抽取') or '未抽取'}",
            f"- **理论术语**：{_join(payload.get('theory_terms', []))}",
            f"- **诊法/理论类型**：{payload.get('diagnostic_method', '未抽取') or '未抽取'}",
            f"- **原文问句**：{payload.get('theory_question', '未抽取') or '未抽取'}",
            f"- **原文答文**：{payload.get('theory_answer', '未抽取') or '未抽取'}",
            "",
            "这类条目只作为古籍理论和诊法依据，不等于现代临床诊断结论。",
            "",
            "---",
            "**参考来源:**",
            f"- {source} {source_url}".strip(),
        ]
        return "\n".join(lines)

    lines = [
        f"从结构化方证库看，最接近的是 **{payload.get('title', payload.get('formula', '未知方证'))}**。",
        "",
        f"- **识别出的表征**：{_join(query.get('canonical_terms', []))}",
        f"- **现代口语对应**：{_join(payload.get('modern_symptoms', []))}",
        f"- **古代症状/辨证依据**：{_join(payload.get('ancient_symptoms', []))}",
        f"- **核心病因病机**：{_join(payload.get('pathogenesis', []))}",
        f"- **对应方剂**：{payload.get('formula', '未抽取')}",
        f"- **组成**：{_format_composition(payload)}",
        f"- **功用/治法**：{payload.get('functions', '未抽取') or '未抽取'}",
        f"- **原文用法**：{payload.get('usage_original', '未抽取') or '未抽取'}",
    ]

    contraindications = payload.get("contraindications", "")
    if contraindications:
        lines.append(f"- **使用注意**：{contraindications}")

    lines.extend(
        [
            "",
            "这只是基于文献方证的检索匹配，不等于对个人病情作出处方。若要继续细分，需要补充舌象、脉象、出汗情况、寒热轻重、痰色痰质等信息。",
            "",
            "---",
            "**参考来源:**",
            f"- {source} {source_url}".strip(),
        ]
    )
    return "\n".join(lines)
