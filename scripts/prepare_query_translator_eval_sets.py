from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = ROOT / "tests" / "evals" / "query_translator_mixed_300" / "questions_mixed.jsonl"
DEFAULT_GOLD = ROOT / "tests" / "evals" / "query_translator_mixed_300" / "private" / "gold_keys.jsonl"
DEFAULT_OUTPUT = ROOT / "tests" / "evals" / "query_translator_benchmarks"

TARGETS = {
    "term_recall": 0.95,
    "negative_terms": 0.98,
    "no_forbidden_positive": 0.995,
    "decision": 0.95,
    "must_clarify": 0.98,
    "retrieval": 0.98,
    "unsupported_answer_rate": 0.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def public_view(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"id": str(record["id"]), "query": str(record["query"])} for record in records]


def count_styles(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("generation_style", "unknown")) for record in records).items()))


def stable_key(record: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{record['id']}".encode("utf-8")).hexdigest()


def holdout_quotas(styles: list[str], total: int) -> dict[str, int]:
    base, extra = divmod(total, len(styles))
    return {style: base + (1 if index < extra else 0) for index, style in enumerate(styles)}


def split_holdout(gold: list[dict[str, Any]], holdout_size: int, salt: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold:
        grouped[str(record.get("generation_style", "unknown"))].append(record)
    styles = sorted(grouped)
    quotas = holdout_quotas(styles, holdout_size)
    holdout_ids: set[str] = set()
    for style in styles:
        rows = sorted(grouped[style], key=lambda item: stable_key(item, salt))
        quota = quotas[style]
        if quota > len(rows):
            raise ValueError(f"holdout quota {quota} exceeds style count {len(rows)} for {style}")
        holdout_ids.update(str(record["id"]) for record in rows[:quota])
    dev = [record for record in gold if str(record["id"]) not in holdout_ids]
    holdout = [record for record in gold if str(record["id"]) in holdout_ids]
    return dev, holdout


def is_hard_negative(record: dict[str, Any]) -> bool:
    return (
        record.get("generation_style") == "hard_negative_clarify"
        or bool(record.get("expected_negative_terms"))
        or bool(record.get("forbidden_terms"))
        or bool(record.get("omitted_term_group"))
    )


def write_set(output_dir: Path, name: str, records: list[dict[str, Any]], description: str, locked: bool) -> None:
    set_dir = output_dir / name
    write_jsonl(set_dir / "questions.jsonl", public_view(records))
    write_jsonl(set_dir / "private" / "gold_keys.jsonl", records)
    write_json(
        set_dir / "manifest.json",
        {
            "name": name,
            "description": description,
            "locked_for_tuning": locked,
            "case_count": len(records),
            "style_counts": count_styles(records),
            "public_questions": str(set_dir / "questions.jsonl"),
            "private_gold": str(set_dir / "private" / "gold_keys.jsonl"),
            "targets": TARGETS,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从300条Query Translator混合盲测候选池拆出开发集、Hard Negative专项集和内部留出回归集")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-size", type=int, default=100)
    parser.add_argument("--salt", default="query-translator-final-holdout-20260622")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    if any(set(record) != {"id", "query"} for record in questions):
        raise ValueError("public question file must contain only id/query fields")
    public_by_id = {str(record["id"]): record for record in questions}
    gold = read_jsonl(args.gold)
    missing_public = [record["id"] for record in gold if str(record["id"]) not in public_by_id]
    if missing_public:
        raise ValueError(f"gold has ids missing from public questions: {missing_public[:10]}")
    if not 0 < args.holdout_size < len(gold):
        raise ValueError("--holdout-size must be between 1 and total case count - 1")

    dev, final_holdout = split_holdout(gold, args.holdout_size, args.salt)
    hard_negative = [record for record in dev if is_hard_negative(record)]

    write_set(
        args.output_dir,
        "dev_tuning",
        dev,
        "可查看私有金标，用于Prompt、别名、归一化和门控阈值调参；不能用于最终宣传指标。",
        locked=False,
    )
    write_set(
        args.output_dir,
        "hard_negative_special",
        hard_negative,
        "从开发集中抽出的Hard Negative专项集，覆盖关键症状缺失、否定词和禁止阳性误判。",
        locked=False,
    )
    write_set(
        args.output_dir,
        "final_holdout",
        final_holdout,
        "内部留出/回归集；300条整体报告已被看过，不作为严格最终盲测。",
        locked=True,
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "source_questions": str(args.questions),
            "source_gold": str(args.gold),
            "total_source_cases": len(gold),
            "holdout_size": args.holdout_size,
            "salt": args.salt,
            "sets": {
                "dev_tuning": {
                    "case_count": len(dev),
                    "style_counts": count_styles(dev),
                },
                "hard_negative_special": {
                    "case_count": len(hard_negative),
                    "style_counts": count_styles(hard_negative),
                },
                "final_holdout": {
                    "case_count": len(final_holdout),
                    "style_counts": count_styles(final_holdout),
                },
            },
            "targets": TARGETS,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "dev_tuning": len(dev),
                "hard_negative_special": len(hard_negative),
                "final_holdout": len(final_holdout),
                "final_holdout_style_counts": count_styles(final_holdout),
                "targets": TARGETS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
