from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

from build_syndrome_dictionary import SyndromeEntry, make_search_text  # noqa: E402
from audit_syndrome_dictionary_quality import score_entry  # noqa: E402
from core.llm_factory import create_llm  # noqa: E402
from core.syndrome_terms import clean_text, unique  # noqa: E402


DEFAULT_QUEUE = ROOT / "datasets" / "structured" / "syndrome_dictionary_llm_review_queue.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "structured" / "syndrome_dictionary_reviewed_replacements.jsonl"
UNSUPPORTED_MODERN_DIAGNOSIS_TERMS = (
    "腰肌劳损",
    "颈椎病",
    "肩周炎",
    "高血压",
    "糖尿病",
    "冠心病",
    "肿瘤",
    "癌",
    "感染",
    "炎症",
)


class ReviewedEntryPatch(BaseModel):
    modern_symptoms: list[str] = Field(default_factory=list, description="白话症状/理论问法/针法说法")
    symptom_aliases: list[str] = Field(default_factory=list, description="用户可能输入的口语近义说法")
    pathogenesis: list[str] = Field(default_factory=list, description="只在原文或现有payload支持时补病机")
    treatment_method: str = ""
    acupoints_or_channels: list[str] = Field(default_factory=list)
    treatment_principle: str = ""
    theory_terms: list[str] = Field(default_factory=list)
    acupuncture_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    review_note: str = Field(default="", description="一句话说明补全依据")


def load_queue(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            items.append(json.loads(raw_line))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return items


def load_existing_replacements(path: Path) -> list[SyndromeEntry]:
    if not path.exists():
        return []
    entries: list[SyndromeEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            entries.append(SyndromeEntry.model_validate(json.loads(raw_line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return entries


def merge_replacements(entries: list[SyndromeEntry]) -> list[SyndromeEntry]:
    merged: list[SyndromeEntry] = []
    positions: dict[str, int] = {}
    for entry in entries:
        if entry.entry_id in positions:
            merged[positions[entry.entry_id]] = entry
            continue
        positions[entry.entry_id] = len(merged)
        merged.append(entry)
    return merged


def filter_unsupported_modern_terms(values: list[str], evidence: str) -> list[str]:
    filtered: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        if any(term in text and term not in evidence for term in UNSUPPORTED_MODERN_DIAGNOSIS_TERMS):
            continue
        filtered.append(text)
    return unique(filtered)


def patch_entry(item: dict[str, Any], patch: ReviewedEntryPatch) -> SyndromeEntry:
    payload = dict(item["current_payload"])
    current = SyndromeEntry.model_validate(payload)
    requested = set(item.get("requested_fields", []))
    evidence = clean_text(item.get("evidence", "") or payload.get("evidence", ""))

    def should_update(field: str, value: Any) -> bool:
        if field not in requested:
            return False
        if isinstance(value, list):
            return bool(value)
        return bool(clean_text(str(value)))

    updates: dict[str, Any] = {}
    if should_update("modern_symptoms", patch.modern_symptoms):
        updates["modern_symptoms"] = filter_unsupported_modern_terms(patch.modern_symptoms, evidence)
    if should_update("symptom_aliases", patch.symptom_aliases):
        updates["symptom_aliases"] = filter_unsupported_modern_terms(patch.symptom_aliases, evidence)
    if should_update("pathogenesis", patch.pathogenesis):
        updates["pathogenesis"] = unique(patch.pathogenesis)
    if should_update("treatment_method", patch.treatment_method):
        updates["treatment_method"] = clean_text(patch.treatment_method)
    if should_update("acupoints_or_channels", patch.acupoints_or_channels):
        updates["acupoints_or_channels"] = unique(patch.acupoints_or_channels)
    if should_update("treatment_principle", patch.treatment_principle):
        updates["treatment_principle"] = clean_text(patch.treatment_principle)
    if should_update("theory_terms", patch.theory_terms):
        updates["theory_terms"] = unique(patch.theory_terms)
    if should_update("acupuncture_terms", patch.acupuncture_terms):
        updates["acupuncture_terms"] = unique(patch.acupuncture_terms)

    for key, value in updates.items():
        payload[key] = value
    payload["review_status"] = "llm_reviewed_replacement"
    payload["confidence"] = min(max(float(patch.confidence or current.confidence), current.confidence), 0.68)

    entry = SyndromeEntry.model_validate(payload)
    entry.search_text = make_search_text(entry)
    return entry


def review_item(llm, item: dict[str, Any]) -> ReviewedEntryPatch:
    current_payload = item["current_payload"]
    prompt = """你是中医古籍结构化 payload 复核器。只补全 requested_fields，不改原文证据。

硬性规则：
1. evidence 是唯一证据；不得编造 evidence 中没有支持的方剂、穴位、药名。
2. 不做现代剂量换算，不给临床处方建议。
3. modern_symptoms 必须是普通用户能输入的白话说法，不能复制“黄帝曰/经言/何谓”等原文句式。
4. symptom_aliases 写用户可能输入的近义说法。
5. 如果 evidence 只讲理论，不要硬改成症状；可把 modern_symptoms 写成白话理论问题。
6. 如果某字段无证据支持，返回空列表或空字符串。
"""
    task = {
        "entry_id": item["entry_id"],
        "source_type": item["source_type"],
        "requested_fields": item.get("requested_fields", []),
        "reasons": item.get("reasons", []),
        "evidence": item.get("evidence", ""),
        "current_payload_subset": {
            "title": current_payload.get("title"),
            "source_type": current_payload.get("source_type"),
            "ancient_symptoms": current_payload.get("ancient_symptoms"),
            "modern_symptoms": current_payload.get("modern_symptoms"),
            "symptom_aliases": current_payload.get("symptom_aliases"),
            "pathogenesis": current_payload.get("pathogenesis"),
            "treatment_method": current_payload.get("treatment_method"),
            "acupoints_or_channels": current_payload.get("acupoints_or_channels"),
            "theory_question": current_payload.get("theory_question"),
            "theory_answer": current_payload.get("theory_answer"),
            "acupuncture_principle": current_payload.get("acupuncture_principle"),
        },
    }
    structured = llm.with_config(temperature=0).with_structured_output(ReviewedEntryPatch)
    return structured.invoke([SystemMessage(content=prompt), HumanMessage(content=json.dumps(task, ensure_ascii=False))])


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 LLM 复核质量队列，输出可替换入库的 SyndromeEntry JSONL")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--min-quality", type=int, default=4)
    parser.add_argument("--source-type", default="", help="可选，仅处理某类 source_type")
    parser.add_argument("--overwrite", action="store_true", help="覆盖 output；默认追加并按 entry_id 去重")
    parser.add_argument("--allow-non-improving", action="store_true", help="允许写入质量分未降低的替换项")
    args = parser.parse_args()

    output_path = Path(args.output)
    existing = [] if args.overwrite else load_existing_replacements(output_path)
    existing_ids = {entry.entry_id for entry in existing}
    queue = [
        item
        for item in load_queue(Path(args.queue))
        if int(item.get("quality_score", 0)) >= args.min_quality
        and (not args.source_type or item.get("source_type") == args.source_type)
        and (args.overwrite or item.get("entry_id") not in existing_ids)
    ]
    if args.max_items:
        queue = queue[: args.max_items]

    llm = create_llm()
    replacements: list[SyndromeEntry] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in queue:
        try:
            patch = review_item(llm, item)
            entry = patch_entry(item, patch)
            old_score = int(item.get("quality_score", 0))
            new_score, reasons, _ = score_entry(entry)
            if new_score >= old_score and not args.allow_non_improving:
                skipped.append(
                    {
                        "entry_id": item.get("entry_id"),
                        "old_score": old_score,
                        "new_score": new_score,
                        "reasons": reasons,
                    }
                )
                continue
            replacements.append(entry)
        except Exception as exc:
            failures.append({"entry_id": item.get("entry_id"), "error": str(exc)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_replacements = merge_replacements(existing + replacements)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in all_replacements:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "queue": str(Path(args.queue)),
                "processed": len(queue),
                "new_replacements": len(replacements),
                "total_replacements": len(all_replacements),
                "skipped": skipped,
                "failures": failures,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
