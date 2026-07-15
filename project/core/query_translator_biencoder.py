from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class TermCatalogItem:
    canonical_term: str
    search_text: str
    term_id: str = ""
    entry_id: str = ""
    evidence_tier: str = ""


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def catalog_search_text(record: dict[str, Any], max_evidence_chars: int = 220) -> str:
    term = _clean(record.get("canonical_term"))
    evidence = _clean(record.get("evidence"))[:max_evidence_chars]
    roles = " ".join(_clean(role) for role in record.get("term_roles") or [] if _clean(role))
    parts = [term]
    if roles:
        parts.append(roles)
    if evidence and evidence != term:
        parts.append(evidence)
    return "。".join(parts)


def build_term_catalog(
    label_path: Path,
    tiers: Sequence[str] = ("high", "medium"),
    eligible_only: bool = False,
) -> list[TermCatalogItem]:
    allowed_tiers = {str(value) for value in tiers}
    items: list[TermCatalogItem] = []
    seen: set[tuple[str, str, str]] = set()
    for record in iter_jsonl(label_path):
        tier = _clean(record.get("evidence_tier"))
        if tier not in allowed_tiers:
            continue
        if eligible_only and not bool(record.get("eligible_as_gold_label")):
            continue
        term = _clean(record.get("canonical_term"))
        if not term:
            continue
        item = TermCatalogItem(
            canonical_term=term,
            search_text=catalog_search_text(record),
            term_id=_clean(record.get("term_id")),
            entry_id=_clean(record.get("entry_id")),
            evidence_tier=tier,
        )
        identity = (item.canonical_term, item.entry_id, item.search_text)
        if identity in seen:
            continue
        seen.add(identity)
        items.append(item)
    if not items:
        raise ValueError(f"term catalog is empty: {label_path}")
    return items


def catalog_records(items: Iterable[TermCatalogItem]) -> Iterator[dict[str, str]]:
    for item in items:
        yield {
            "canonical_term": item.canonical_term,
            "search_text": item.search_text,
            "term_id": item.term_id,
            "entry_id": item.entry_id,
            "evidence_tier": item.evidence_tier,
        }


def load_catalog(path: Path) -> list[TermCatalogItem]:
    return [
        TermCatalogItem(
            canonical_term=_clean(record.get("canonical_term")),
            search_text=_clean(record.get("search_text")),
            term_id=_clean(record.get("term_id")),
            entry_id=_clean(record.get("entry_id")),
            evidence_tier=_clean(record.get("evidence_tier")),
        )
        for record in iter_jsonl(path)
        if _clean(record.get("canonical_term")) and _clean(record.get("search_text"))
    ]


def expected_terms_from_supervision(record: dict[str, Any]) -> set[str]:
    return {
        _clean(mapping.get("canonical_term"))
        for mapping in record.get("mappings") or []
        if _clean(mapping.get("canonical_term")) and mapping.get("polarity", "present") == "present"
    }


def expected_groups_from_frozen(record: dict[str, Any]) -> list[set[str]]:
    groups = record.get("expected_term_groups") or []
    if groups:
        return [{_clean(term) for term in group if _clean(term)} for group in groups]
    return [{_clean(term)} for term in record.get("expected_terms") or [] if _clean(term)]


def collapse_catalog_scores(
    items: Sequence[TermCatalogItem],
    scores: Sequence[float],
    limit: int,
) -> list[dict[str, Any]]:
    best: dict[str, tuple[float, TermCatalogItem]] = {}
    for item, score in zip(items, scores):
        numeric_score = float(score)
        previous = best.get(item.canonical_term)
        if previous is None or numeric_score > previous[0]:
            best[item.canonical_term] = (numeric_score, item)
    ranked = sorted(best.values(), key=lambda value: (value[0], value[1].canonical_term), reverse=True)
    return [
        {
            "canonical_term": item.canonical_term,
            "score": round(score, 6),
            "entry_id": item.entry_id,
            "evidence_tier": item.evidence_tier,
        }
        for score, item in ranked[:limit]
    ]


def score_predictions(cases: Sequence[dict[str, Any]], top_ks: Sequence[int]) -> dict[str, Any]:
    if not cases:
        return {"case_count": 0}
    metrics: dict[str, Any] = {"case_count": len(cases)}
    reciprocal_ranks: list[float] = []
    for case in cases:
        ranked_terms = [str(item["canonical_term"]) for item in case.get("predictions") or []]
        groups = case.get("expected_groups") or []
        first_ranks: list[int] = []
        for group in groups:
            ranks = [ranked_terms.index(term) + 1 for term in group if term in ranked_terms]
            first_ranks.append(min(ranks) if ranks else 0)
        positive_ranks = [rank for rank in first_ranks if rank]
        reciprocal_ranks.append(1.0 / min(positive_ranks) if positive_ranks else 0.0)
        case["group_first_ranks"] = first_ranks
    metrics["mrr"] = round(sum(reciprocal_ranks) / len(cases), 6)
    for top_k in sorted(set(int(value) for value in top_ks)):
        covered_groups = 0
        total_groups = 0
        all_groups_cases = 0
        any_group_cases = 0
        for case in cases:
            groups = case.get("expected_groups") or []
            predictions = {
                str(item["canonical_term"])
                for item in (case.get("predictions") or [])[:top_k]
            }
            covered = [bool(set(group) & predictions) for group in groups]
            covered_groups += sum(covered)
            total_groups += len(covered)
            all_groups_cases += int(bool(covered) and all(covered))
            any_group_cases += int(any(covered))
        metrics[f"group_recall_at_{top_k}"] = round(covered_groups / max(total_groups, 1), 6)
        metrics[f"case_all_at_{top_k}"] = round(all_groups_cases / len(cases), 6)
        metrics[f"case_any_at_{top_k}"] = round(any_group_cases / len(cases), 6)
    return metrics


def fuse_prediction_lists(
    prediction_lists: Sequence[Sequence[dict[str, Any]]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for predictions in prediction_lists:
        for rank, prediction in enumerate(predictions, start=1):
            term = str(prediction.get("canonical_term", "")).strip()
            if not term:
                continue
            item = fused.setdefault(
                term,
                {
                    "canonical_term": term,
                    "score": 0.0,
                    "entry_id": prediction.get("entry_id", ""),
                    "evidence_tier": prediction.get("evidence_tier", ""),
                },
            )
            item["score"] += 1.0 / (rank_constant + rank)
    ranked = sorted(fused.values(), key=lambda item: (float(item["score"]), item["canonical_term"]), reverse=True)
    for item in ranked:
        item["score"] = round(float(item["score"]), 8)
    return ranked[:limit]


def cosine_scores(query_embeddings: Any, catalog_embeddings: Any) -> Any:
    # Inputs are normalized by SentenceTransformer.encode, so matrix multiplication is cosine similarity.
    return query_embeddings @ catalog_embeddings.T


class QueryTranslatorBiEncoder:
    """Lazy local-catalog predictor for online candidate-term expansion."""

    def __init__(self, model_path: str | Path, catalog_path: str | Path, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.catalog_path = Path(catalog_path)
        self.device = device or "cpu"
        self._model = None
        self._catalog: list[TermCatalogItem] = []
        self._catalog_embeddings = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"bi-encoder model directory missing: {self.model_path}")
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"bi-encoder catalog missing: {self.catalog_path}")
        from sentence_transformers import SentenceTransformer

        self._catalog = load_catalog(self.catalog_path)
        if not self._catalog:
            raise ValueError(f"bi-encoder catalog is empty: {self.catalog_path}")
        self._model = SentenceTransformer(str(self.model_path), device=self.device, local_files_only=True)
        self._model.max_seq_length = 160
        self._catalog_embeddings = self._model.encode(
            [item.search_text for item in self._catalog],
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def predict(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        normalized = _clean(query)
        if not normalized or limit <= 0:
            return []
        self._ensure_loaded()
        query_embedding = self._model.encode(
            [normalized],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = cosine_scores(query_embedding, self._catalog_embeddings)[0]
        return collapse_catalog_scores(self._catalog, scores, limit)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def group_catalog_by_term(items: Sequence[TermCatalogItem]) -> dict[str, list[TermCatalogItem]]:
    grouped: dict[str, list[TermCatalogItem]] = defaultdict(list)
    for item in items:
        grouped[item.canonical_term].append(item)
    return dict(grouped)
