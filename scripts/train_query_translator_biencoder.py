from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.query_translator_biencoder import (  # noqa: E402
    build_term_catalog,
    catalog_records,
    group_catalog_by_term,
    iter_jsonl,
    write_jsonl,
)


DEFAULT_TRAIN = ROOT / "datasets" / "external" / "splits" / "train"
DEFAULT_LABELS = ROOT / "datasets" / "structured" / "query_translator_evidence_label_pool.jsonl"
DEFAULT_BASE_MODEL = (
    ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-small-zh-v1.5"
    / "snapshots"
    / "7999e1d3359715c523056ef9478215996d62a620"
)


def stable_index(value: str, size: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def make_training_examples(
    query_path: Path,
    semantic_path: Path,
    catalog,
    semantic_limit: int,
    seed: int,
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    grouped = group_catalog_by_term(catalog)
    terms = sorted(grouped)
    examples: list[tuple[str, str, float]] = []
    counts = {
        "query_term_positive": 0,
        "query_term_negative": 0,
        "semantic_positive": 0,
        "semantic_negative": 0,
    }
    for record in iter_jsonl(query_path):
        query = str(record.get("query", "")).strip()
        expected = {
            str(mapping.get("canonical_term", "")).strip()
            for mapping in record.get("mappings") or []
            if mapping.get("polarity", "present") == "present" and str(mapping.get("canonical_term", "")).strip()
        }
        if not query or not expected:
            continue
        for term in sorted(expected):
            candidates = grouped.get(term)
            if not candidates:
                continue
            positive = candidates[stable_index(f"{record.get('query_id')}:{term}", len(candidates))]
            examples.append((query, positive.search_text, 1.0))
            counts["query_term_positive"] += 1

            offset = stable_index(f"negative:{seed}:{record.get('query_id')}:{term}", len(terms))
            negative_term = terms[offset]
            if negative_term in expected:
                negative_term = terms[(offset + 1) % len(terms)]
            if negative_term in expected:
                continue
            negative_candidates = grouped[negative_term]
            negative = negative_candidates[stable_index(query + negative_term, len(negative_candidates))]
            examples.append((query, negative.search_text, 0.0))
            counts["query_term_negative"] += 1

    semantic_count = 0
    for record in iter_jsonl(semantic_path):
        if semantic_limit > 0 and semantic_count >= semantic_limit:
            break
        left = str(record.get("text_a", "")).strip()
        right = str(record.get("text_b", "")).strip()
        if not left or not right:
            continue
        positive = bool(record.get("is_positive"))
        examples.append((left, right, 1.0 if positive else 0.0))
        counts["semantic_positive" if positive else "semantic_negative"] += 1
        semantic_count += 1

    random.Random(seed).shuffle(examples)
    return examples, counts


def make_positive_groups(query_path: Path, catalog) -> dict[str, list[tuple[str, str]]]:
    grouped_catalog = group_catalog_by_term(catalog)
    groups: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in iter_jsonl(query_path):
        query = str(record.get("query", "")).strip()
        if not query:
            continue
        expected = {
            str(mapping.get("canonical_term", "")).strip()
            for mapping in record.get("mappings") or []
            if mapping.get("polarity", "present") == "present" and str(mapping.get("canonical_term", "")).strip()
        }
        for term in sorted(expected):
            candidates = grouped_catalog.get(term)
            if not candidates or (query, term) in seen:
                continue
            seen.add((query, term))
            target = candidates[stable_index(f"{record.get('query_id')}:{term}", len(candidates))]
            groups.setdefault(term, []).append((query, target.search_text))
    return groups


def train_model(
    model: Any,
    examples: list[tuple[str, str, float]],
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_amp: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from sentence_transformers.util import batch_to_device
    from transformers import get_linear_schedule_with_warmup

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.08))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp_enabled = bool(use_amp and str(model.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"translator epoch {epoch + 1}/{epochs}")
        for texts_a, texts_b, labels in progress:
            features_a = batch_to_device(model.preprocess(list(texts_a)), model.device)
            features_b = batch_to_device(model.preprocess(list(texts_b)), model.device)
            labels = labels.to(model.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                embeddings_a = functional.normalize(model(features_a)["sentence_embedding"], p=2, dim=1)
                embeddings_b = functional.normalize(model(features_b)["sentence_embedding"], p=2, dim=1)
                similarities = (embeddings_a * embeddings_b).sum(dim=1)
                loss = functional.mse_loss(similarities, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            numeric_loss = float(loss.detach().cpu())
            losses.append(numeric_loss)
            if len(losses) % 20 == 0:
                progress.set_postfix(loss=f"{sum(losses[-20:]) / 20:.4f}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return {
        "steps": total_steps,
        "warmup_steps": warmup_steps,
        "final_loss": round(losses[-1], 6),
        "mean_loss": round(sum(losses) / len(losses), 6),
        "amp_enabled": amp_enabled,
    }


def train_mnrl_distill(
    model: Any,
    teacher: Any,
    groups: dict[str, list[tuple[str, str]]],
    output_path: Path,
    rounds: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    preservation_weight: float,
    use_amp: bool,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from sentence_transformers.util import batch_to_device
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    terms = sorted(groups)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    batches_per_round = (len(terms) + batch_size - 1) // batch_size
    total_steps = rounds * batches_per_round
    warmup_steps = max(1, int(total_steps * 0.08))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp_enabled = bool(use_amp and str(model.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    losses: list[float] = []
    retrieval_losses: list[float] = []
    preservation_losses: list[float] = []
    model.train()
    progress = tqdm(total=total_steps, desc="translator mnrl+distill")
    for round_index in range(rounds):
        round_terms = list(terms)
        rng.shuffle(round_terms)
        for offset in range(0, len(round_terms), batch_size):
            batch_terms = round_terms[offset : offset + batch_size]
            pairs = [groups[term][round_index % len(groups[term])] for term in batch_terms]
            texts_a = [pair[0] for pair in pairs]
            texts_b = [pair[1] for pair in pairs]
            features_a = batch_to_device(model.preprocess(texts_a), model.device)
            features_b = batch_to_device(model.preprocess(texts_b), model.device)
            teacher_features_a = batch_to_device(teacher.preprocess(texts_a), teacher.device)
            teacher_features_b = batch_to_device(teacher.preprocess(texts_b), teacher.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_a = functional.normalize(teacher(teacher_features_a)["sentence_embedding"], p=2, dim=1)
                teacher_b = functional.normalize(teacher(teacher_features_b)["sentence_embedding"], p=2, dim=1)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                student_a = functional.normalize(model(features_a)["sentence_embedding"], p=2, dim=1)
                student_b = functional.normalize(model(features_b)["sentence_embedding"], p=2, dim=1)
                logits = student_a @ student_b.T / temperature
                labels = torch.arange(len(batch_terms), device=model.device)
                retrieval_loss = functional.cross_entropy(logits, labels)
                preservation_loss = (
                    (1.0 - (student_a * teacher_a).sum(dim=1)).mean()
                    + (1.0 - (student_b * teacher_b).sum(dim=1)).mean()
                ) / 2.0
                loss = retrieval_loss + preservation_weight * preservation_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
            retrieval_losses.append(float(retrieval_loss.detach().cpu()))
            preservation_losses.append(float(preservation_loss.detach().cpu()))
            progress.update(1)
            if len(losses) % 10 == 0:
                progress.set_postfix(
                    loss=f"{sum(losses[-10:]) / 10:.4f}",
                    preserve=f"{sum(preservation_losses[-10:]) / 10:.4f}",
                )
    progress.close()
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return {
        "mode": "multiple_negatives_with_frozen_teacher",
        "target_term_count": len(groups),
        "available_positive_pairs": sum(len(values) for values in groups.values()),
        "rounds": rounds,
        "steps": total_steps,
        "warmup_steps": warmup_steps,
        "temperature": temperature,
        "preservation_weight": preservation_weight,
        "final_loss": round(losses[-1], 6),
        "mean_loss": round(sum(losses) / len(losses), 6),
        "mean_retrieval_loss": round(sum(retrieval_losses) / len(retrieval_losses), 6),
        "mean_preservation_loss": round(sum(preservation_losses) / len(preservation_losses), 6),
        "amp_enabled": amp_enabled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline evidence-grounded Query Translator bi-encoder")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--semantic-limit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-mode", choices=("mnrl_distill", "cosine_pair"), default="mnrl_distill")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument("--augmentation-query-pairs", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    if "eval" in str(args.train_root).lower() or "frozen" in str(args.train_root).lower():
        raise ValueError("training input must not point at an eval or frozen directory")
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    catalog = build_term_catalog(args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / "term_catalog.jsonl"
    write_jsonl(catalog_path, catalog_records(catalog))
    model = SentenceTransformer(str(args.base_model), device=args.device, local_files_only=True)
    model.max_seq_length = 160
    if args.loss_mode == "mnrl_distill":
        positive_groups = make_positive_groups(args.train_root / "query_term_pairs.jsonl", catalog)
        augmentation_pairs = 0
        for augmentation_path in args.augmentation_query_pairs:
            if "eval" in str(augmentation_path).lower() or "frozen" in str(augmentation_path).lower():
                raise ValueError("augmentation input must not point at an eval or frozen directory")
            augmented_groups = make_positive_groups(augmentation_path, catalog)
            augmentation_pairs += sum(len(values) for values in augmented_groups.values())
            for term, values in augmented_groups.items():
                positive_groups.setdefault(term, []).extend(values)
        if len(positive_groups) < 2:
            raise ValueError("not enough unique target terms for contrastive training")
        teacher = SentenceTransformer(str(args.base_model), device=args.device, local_files_only=True)
        teacher.max_seq_length = model.max_seq_length
        optimization = train_mnrl_distill(
            model,
            teacher,
            positive_groups,
            args.output / "model",
            args.rounds,
            args.batch_size,
            args.learning_rate,
            args.temperature,
            args.preservation_weight,
            args.use_amp,
            args.seed,
        )
        training_examples = sum(len(values) for values in positive_groups.values())
        example_counts = {
            "query_term_positive": training_examples,
            "query_term_negative": 0,
            "semantic_positive": 0,
            "semantic_negative": 0,
            "augmentation_query_term_positive": augmentation_pairs,
        }
    else:
        examples, example_counts = make_training_examples(
            args.train_root / "query_term_pairs.jsonl",
            args.train_root / "semantic_pair_supervision.jsonl",
            catalog,
            args.semantic_limit,
            args.seed,
        )
        if not examples:
            raise ValueError("no training examples were built")
        optimization = train_model(
            model,
            examples,
            args.output / "model",
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.use_amp,
        )
        training_examples = len(examples)
    report = {
        "report_version": 1,
        "base_model": str(args.base_model.resolve()),
        "output_model": str((args.output / "model").resolve()),
        "train_root": str(args.train_root.resolve()),
        "labels": str(args.labels.resolve()),
        "frozen_eval_read_during_training": False,
        "catalog_items": len(catalog),
        "catalog_terms": len({item.canonical_term for item in catalog}),
        "training_examples": training_examples,
        "example_counts": example_counts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_mode": args.loss_mode,
        "augmentation_query_pairs": [str(path.resolve()) for path in args.augmentation_query_pairs],
        "seed": args.seed,
        "optimization": optimization,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
