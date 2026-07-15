from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def stable_key(record: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{record['id']}".encode("utf-8")).hexdigest()


def quotas(styles: list[str], target_count: int) -> dict[str, int]:
    base, extra = divmod(target_count, len(styles))
    return {style: base + (1 if index < extra else 0) for index, style in enumerate(styles)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从Query Translator生成结果中固定抽取一个公开/私有成对子集")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--salt", default="fresh-final-100-20260622")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public = {str(record["id"]): record for record in read_jsonl(args.questions)}
    gold = read_jsonl(args.gold)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold:
        case_id = str(record["id"])
        if case_id not in public:
            raise ValueError(f"gold id missing from public questions: {case_id}")
        grouped[str(record.get("generation_style", "unknown"))].append(record)
    styles = sorted(grouped)
    if args.target_count > len(gold):
        raise ValueError("--target-count exceeds available records")
    target_quotas = quotas(styles, args.target_count)
    selected: list[dict[str, Any]] = []
    for style in styles:
        rows = sorted(grouped[style], key=lambda record: stable_key(record, args.salt))
        quota = target_quotas[style]
        if quota > len(rows):
            raise ValueError(f"quota {quota} exceeds available rows {len(rows)} for {style}")
        selected.extend(rows[:quota])
    selected_ids = {str(record["id"]) for record in selected}
    selected_public = [record for record in read_jsonl(args.questions) if str(record["id"]) in selected_ids]
    write_jsonl(args.output_dir / "questions_mixed.jsonl", selected_public)
    write_jsonl(args.output_dir / "private" / "gold_keys.jsonl", selected)
    style_counts = {style: sum(1 for record in selected if record.get("generation_style") == style) for style in styles}
    write_json(
        args.output_dir / "manifest.json",
        {
            "source_questions": str(args.questions),
            "source_gold": str(args.gold),
            "target_count": args.target_count,
            "selected_count": len(selected),
            "public_fields": ["id", "query"],
            "style_counts": style_counts,
            "locked_for_tuning": True,
        },
    )
    print(json.dumps({"selected_count": len(selected), "style_counts": style_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
