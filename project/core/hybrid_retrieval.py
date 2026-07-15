from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from core.syndrome_terms import SYMPTOM_TERMS, clean_text, unique


LEXICAL_FIELDS = (
    "title",
    "syndrome_name",
    "formula",
    "herb_name",
    "intervention_name",
    "treatment_method",
    "ancient_symptoms",
    "modern_symptoms",
    "symptom_aliases",
    "diagnostic_keys",
    "pathogenesis",
    "required_symptom_groups",
    "forbidden_terms",
    "differential_keys",
    "must_clarify_fields",
    "acupoints_or_channels",
    "theory_topic",
    "theory_question",
    "theory_terms",
    "diagnostic_method",
    "acupuncture_principle",
    "acupuncture_terms",
    "functions",
    "indications",
    "evidence",
)
EXACT_FIELDS = (
    "formula",
    "herb_name",
    "intervention_name",
    "title",
    "theory_terms",
    "acupuncture_terms",
    "acupoints_or_channels",
    "differential_keys",
)
CANONICAL_TERM_FIELDS = (
    "ancient_symptoms",
    "modern_symptoms",
    "symptom_aliases",
    "diagnostic_keys",
    "pathogenesis",
    "required_symptom_groups",
    # Exclusion-only terms still need to be visible to Query Translator so the
    # retriever can filter contraindicated formula-syndrome payloads.
    "forbidden_terms",
    "differential_keys",
    "theory_terms",
    "acupuncture_terms",
    "acupoints_or_channels",
    "diagnostic_method",
)
_INTERVENTION_CUE_RE = re.compile(r"(?:主之|可与|宜用|方剂|处方|加减|剂量)")
LEXICAL_STOP_TOKENS = {
    "这个", "那个", "以后", "最近", "一直", "完全", "就是", "还是", "已经",
    "哪里", "怎么", "什么", "为什么", "哪些", "具体", "里面", "古书", "记载",
    "感觉", "一下", "这里", "那里", "自己", "整个人", "说不上", "不舒服",
}
_SEGMENT_RE = re.compile(r"[\u3400-\u9fff\U00020000-\U0002ffff]+|[a-zA-Z0-9]+")


def _iter_values(payload: dict[str, Any], fields: Iterable[str]) -> Iterable[str]:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for nested in item.values():
                        if nested:
                            yield str(nested)
                elif isinstance(item, list):
                    for nested in item:
                        if nested:
                            yield str(nested)
                elif item:
                    yield str(item)
        elif value:
            yield str(value)


def payload_lexical_text(payload: dict[str, Any]) -> str:
    return clean_text(" ".join(_iter_values(payload, LEXICAL_FIELDS)))


def tokenize_zh(text: str) -> list[str]:
    tokens: list[str] = []
    for segment in _SEGMENT_RE.findall(clean_text(text).lower()):
        if re.fullmatch(r"[a-z0-9]+", segment):
            if len(segment) > 1 and segment not in LEXICAL_STOP_TOKENS:
                tokens.append(segment)
            continue
        if 1 < len(segment) <= 8 and segment not in LEXICAL_STOP_TOKENS:
            tokens.append(segment)
        for width in (2, 3):
            if len(segment) < width:
                continue
            for idx in range(len(segment) - width + 1):
                token = segment[idx:idx + width]
                if token not in LEXICAL_STOP_TOKENS:
                    tokens.append(token)
    return tokens


def exact_match_count(query: str, payload: dict[str, Any]) -> int:
    compact_query = re.sub(r"\s+", "", clean_text(query).lower())
    count = 0
    for value in unique(list(_iter_values(payload, EXACT_FIELDS))):
        compact_value = re.sub(r"\s+", "", value.lower())
        if len(compact_value) < 2:
            continue
        if compact_value in compact_query or compact_query in compact_value:
            count += 1
    return count


@dataclass(frozen=True)
class LexicalHit:
    entry_id: str
    score: float
    exact_matches: int
    payload: dict[str, Any]


class LocalBM25Index:
    """Small deterministic BM25 index for the structured syndrome payloads."""

    def __init__(self, payloads: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self._payloads: dict[str, dict[str, Any]] = {}
        self._term_counts: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._document_frequency: Counter[str] = Counter()
        self._k1 = k1
        self._b = b

        for payload in payloads:
            entry_id = str(payload.get("entry_id", "")).strip()
            if not entry_id:
                continue
            counts = Counter(tokenize_zh(payload_lexical_text(payload)))
            self._payloads[entry_id] = payload
            self._term_counts[entry_id] = counts
            self._doc_lengths[entry_id] = sum(counts.values())
            self._document_frequency.update(counts.keys())
        lengths = list(self._doc_lengths.values())
        self._average_length = sum(lengths) / len(lengths) if lengths else 1.0

    @property
    def size(self) -> int:
        return len(self._payloads)

    def search(self, query: str, limit: int = 30) -> list[LexicalHit]:
        query_terms = Counter(tokenize_zh(query))
        if not query_terms or not self._payloads:
            return []
        document_count = len(self._payloads)
        hits: list[LexicalHit] = []
        for entry_id, counts in self._term_counts.items():
            score = 0.0
            length = self._doc_lengths[entry_id]
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self._k1 * (1 - self._b + self._b * length / self._average_length)
                score += inverse_frequency * (frequency * (self._k1 + 1) / denominator) * min(query_frequency, 2)
            payload = self._payloads[entry_id]
            exact = exact_match_count(query, payload)
            if exact:
                score += 4.0 * exact
            if score > 0:
                hits.append(LexicalHit(entry_id=entry_id, score=score, exact_matches=exact, payload=payload))
        hits.sort(key=lambda hit: (hit.exact_matches, hit.score), reverse=True)
        return hits[:limit]


def reciprocal_rank_fusion(
    routes: dict[str, list[dict[str, Any]]],
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for route_name, hits in routes.items():
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            entry_id = str(payload.get("entry_id") or hit.get("entry_id") or "")
            if not entry_id:
                continue
            item = fused.setdefault(
                entry_id,
                {
                    "entry_id": entry_id,
                    "payload": payload,
                    "rrf_score": 0.0,
                    "routes": {},
                    "dense_scores": {},
                    "lexical_score": 0.0,
                    "exact_match_count": 0,
                },
            )
            item["rrf_score"] += 1.0 / (rank_constant + rank)
            item["routes"][route_name] = rank
            if "score" in hit:
                if route_name.endswith("dense"):
                    item["dense_scores"][route_name] = float(hit["score"])
                elif route_name == "lexical":
                    item["lexical_score"] = float(hit["score"])
            item["exact_match_count"] = max(item["exact_match_count"], int(hit.get("exact_matches", 0)))
    return sorted(
        fused.values(),
        key=lambda item: (item["rrf_score"], item["exact_match_count"], max(item["dense_scores"].values(), default=0.0)),
        reverse=True,
    )


def collect_candidate_terms(items: list[dict[str, Any]], limit: int = 80) -> list[str]:
    terms: list[str] = []
    derived_terms: list[str] = []
    raw_limit = max(1, int(limit * 0.75))
    for item in items:
        payload = item.get("payload") or item
        for value in _iter_values(payload, CANONICAL_TERM_FIELDS):
            cleaned = clean_text(value)
            if 1 < len(cleaned) <= 16 and not _INTERVENTION_CUE_RE.search(cleaned):
                terms.append(cleaned)
                # Payloads sometimes store a compound such as "喘咳" or a full
                # classical clause. Expose only canonical atoms already present
                # in the local terminology table; this keeps the LLM whitelist
                # grounded without requiring per-query synonym patches.
                derived_terms.extend(
                    canonical
                    for canonical in SYMPTOM_TERMS
                    if canonical != cleaned and canonical in cleaned
                )
        if len(unique(terms)) >= raw_limit:
            break
    return unique(terms + derived_terms)[:limit]


def collect_payload_evidence_terms(payloads: list[dict[str, Any]]) -> list[str]:
    """Collect exact, locally evidenced terms without a global vocabulary cap.

    This index is used only for literal query-to-payload matching.  Keeping it
    separate from the bounded LLM whitelist prevents uncommon formula signs
    from disappearing merely because they fall after a global top-N cutoff.
    """
    terms: list[str] = []
    for payload in payloads:
        for value in _iter_values(payload, CANONICAL_TERM_FIELDS):
            cleaned = clean_text(value)
            if not (1 < len(cleaned) <= 16):
                continue
            if _INTERVENTION_CUE_RE.search(cleaned):
                continue
            terms.append(cleaned)
    return unique(terms)


def collect_global_candidate_terms(payloads: list[dict[str, Any]], limit: int = 800) -> list[str]:
    """Return local terms that actually exist in payloads, including exclusion fields."""
    corpus = "\n".join(payload_lexical_text(payload) for payload in payloads)
    payload_terms: list[str] = []
    for payload in payloads:
        for value in _iter_values(payload, ("forbidden_terms",)):
            cleaned = clean_text(value)
            if 1 < len(cleaned) <= 16 and not _INTERVENTION_CUE_RE.search(cleaned):
                payload_terms.append(cleaned)
    symptom_terms = [term for term in SYMPTOM_TERMS if term in corpus]
    return unique(payload_terms + symptom_terms)[:limit]


def payload_contains_term(payload: dict[str, Any], term: str) -> bool:
    if not term:
        return False
    return term in payload_lexical_text(payload)
