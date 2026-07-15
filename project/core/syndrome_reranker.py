from __future__ import annotations

from collections.abc import Callable
from typing import Any


RERANK_TEXT_FIELDS = (
    "title",
    "source_type",
    "source_book",
    "chapter",
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
    "treatment_principle",
    "functions",
    "indications",
    "contraindications",
    "evidence",
    "raw_text",
)


def _flatten_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(_flatten_value(nested))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_flatten_value(item))
        return parts
    text = str(value).strip()
    return [text] if text else []


def payload_to_rerank_text(payload: dict[str, Any], max_chars: int = 1800) -> str:
    parts: list[str] = []
    for field in RERANK_TEXT_FIELDS:
        values = _flatten_value(payload.get(field))
        if values:
            parts.append(f"{field}: {' '.join(values)}")
    text = "\n".join(parts)
    return text[:max_chars] if max_chars > 0 else text


def build_rerank_query(query_info: dict[str, Any], max_chars: int = 900) -> str:
    parts = [
        f"original_query: {query_info.get('original_query', '')}",
        f"intent: {query_info.get('query_intent', '')}",
        "canonical_terms: " + " ".join(str(term) for term in query_info.get("canonical_terms", [])),
        "primary_terms: " + " ".join(str(term) for term in query_info.get("primary_canonical_terms", [])),
        "colloquial_terms: " + " ".join(str(term) for term in query_info.get("colloquial_terms", [])),
        "negative_terms: " + " ".join(str(term) for term in query_info.get("negative_terms", [])),
        "pathogenesis_hints: " + " ".join(str(term) for term in query_info.get("pathogenesis_hints", [])),
    ]
    text = "\n".join(part for part in parts if part.strip())
    return text[:max_chars] if max_chars > 0 else text


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder wrapper for second-stage rerank."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        cache_folder: str | None = None,
        local_files_only: bool = True,
        max_length: int = 512,
        batch_size: int = 16,
        trust_remote_code: bool = False,
        scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or None
        self.cache_folder = cache_folder or None
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self._scorer = scorer
        self._model = None
        self._load_error = ""

    @property
    def load_error(self) -> str:
        return self._load_error

    def _ensure_model(self):
        if self._scorer is not None:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
                max_length=self.max_length,
            )
            self._load_error = ""
            return self._model
        except Exception as exc:  # pragma: no cover - depends on local model cache.
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    def score(self, query_text: str, matches: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        if not matches:
            return [], {"rerank_used": False, "reason": "no_candidates"}

        pairs = [
            (query_text, payload_to_rerank_text(match.get("payload", {})))
            for match in matches
        ]
        try:
            if self._scorer is not None:
                raw_scores = self._scorer(pairs)
            else:
                model = self._ensure_model()
                raw_scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            scores = [float(getattr(score, "item", lambda: score)()) for score in raw_scores]
        except Exception as exc:
            return [], {
                "rerank_used": False,
                "model": self.model_name,
                "error": f"{type(exc).__name__}: {exc}",
                "load_error": self._load_error,
            }

        return scores, {
            "rerank_used": True,
            "model": self.model_name,
            "candidate_count": len(matches),
            "local_files_only": self.local_files_only,
        }
