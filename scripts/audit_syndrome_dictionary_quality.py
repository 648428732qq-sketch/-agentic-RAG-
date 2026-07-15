from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

from build_syndrome_dictionary import SyndromeEntry  # noqa: E402
from core.syndrome_terms import clean_text  # noqa: E402


DEFAULT_JSONL = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_REPORT_JSON = ROOT / "datasets" / "structured" / "syndrome_dictionary_quality_audit.json"
DEFAULT_REPORT_MD = ROOT / "datasets" / "structured" / "syndrome_dictionary_quality_audit.md"
DEFAULT_QUEUE_JSONL = ROOT / "datasets" / "structured" / "syndrome_dictionary_llm_review_queue.jsonl"

NOISY_MODERN_MARKERS = (
    "黄帝曰",
    "岐伯曰",
    "问曰",
    "难曰",
    "经言",
    "何谓",
    "奈何",
    "主之",
    "可与",
    "不可",
    "可令",
    "取之",
    "刺之",
    "灸之",
    "故曰",
)
GENERIC_METHODS = (
    "针刺/取穴相关条文",
    "古籍单味药主治线索",
    "脉诊/诊法理论",
    "古籍理论问答",
)


def load_entries(path: Path) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            entries.append(SyndromeEntry.model_validate(json.loads(raw_line)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return entries


def _same_modern_as_ancient(entry: SyndromeEntry) -> bool:
    return bool(entry.ancient_symptoms) and entry.modern_symptoms == entry.ancient_symptoms


def _has_noisy_modern(entry: SyndromeEntry) -> bool:
    text = " ".join(entry.modern_symptoms + entry.symptom_aliases)
    return any(marker in text for marker in NOISY_MODERN_MARKERS)


def _has_untranslated_classical_suffix(entry: SyndromeEntry) -> bool:
    text = " ".join(entry.modern_symptoms)
    return any(token in text for token in ("者", "也", "焉", "矣")) and not entry.symptom_aliases


def score_entry(entry: SyndromeEntry) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    requested_fields: list[str] = []

    if not entry.modern_symptoms:
        score += 5
        reasons.append("missing modern_symptoms")
        requested_fields.append("modern_symptoms")
    elif _same_modern_as_ancient(entry):
        score += 4
        reasons.append("modern_symptoms identical to ancient_symptoms")
        requested_fields.append("modern_symptoms")

    if not entry.symptom_aliases and entry.source_type not in {"classical_theory"}:
        score += 2
        reasons.append("missing symptom_aliases")
        requested_fields.append("symptom_aliases")

    if _has_noisy_modern(entry):
        score += 3
        reasons.append("modern fields contain classical or source-dialogue noise")
        requested_fields.extend(["modern_symptoms", "symptom_aliases"])

    if _has_untranslated_classical_suffix(entry):
        score += 2
        reasons.append("modern_symptoms still look like untranslated classical snippets")
        requested_fields.append("modern_symptoms")

    if entry.source_type == "classical_clause":
        if not entry.pathogenesis or entry.pathogenesis == [entry.evidence]:
            score += 2
            reasons.append("classical_clause pathogenesis is missing or raw evidence copy")
            requested_fields.append("pathogenesis")
        if not entry.formula:
            score += 4
            reasons.append("classical_clause missing formula")
            requested_fields.append("formula")

    if entry.source_type == "classical_acupuncture":
        if not entry.acupoints_or_channels:
            score += 2
            reasons.append("classical_acupuncture missing acupoints_or_channels")
            requested_fields.append("acupoints_or_channels")
        if entry.treatment_method in GENERIC_METHODS or not entry.treatment_method:
            score += 2
            reasons.append("classical_acupuncture treatment_method is generic or missing")
            requested_fields.append("treatment_method")

    if entry.source_type == "classical_acupuncture_principle":
        if not entry.acupuncture_terms:
            score += 4
            reasons.append("acupuncture_principle missing acupuncture_terms")
            requested_fields.append("acupuncture_terms")
        if entry.treatment_method in GENERIC_METHODS or entry.intervention_name in GENERIC_METHODS:
            score += 1
            reasons.append("acupuncture_principle method summary is generic")
            requested_fields.append("treatment_method")

    if entry.source_type == "herb_indication":
        if not entry.nature_flavor:
            score += 3
            reasons.append("herb_indication missing nature_flavor")
            requested_fields.append("nature_flavor")
        if not entry.indications:
            score += 4
            reasons.append("herb_indication missing indications")
            requested_fields.append("indications")

    if entry.source_type == "classical_theory":
        if not entry.theory_question and "何谓" in entry.evidence:
            score += 2
            reasons.append("classical_theory likely has question but theory_question is empty")
            requested_fields.append("theory_question")
        if not entry.theory_answer:
            score += 4
            reasons.append("classical_theory missing theory_answer")
            requested_fields.append("theory_answer")

    requested_fields = sorted(set(requested_fields))
    return score, reasons, requested_fields


def queue_item(entry: SyndromeEntry, score: int, reasons: list[str], requested_fields: list[str]) -> dict[str, Any]:
    payload = entry.model_dump(mode="json")
    evidence = clean_text(entry.evidence or entry.raw_text or entry.indications)
    return {
        "entry_id": entry.entry_id,
        "title": entry.title,
        "source_type": entry.source_type,
        "source_book": entry.source_book,
        "source_file": entry.source_file,
        "source_url": entry.source_url,
        "quality_score": score,
        "reasons": reasons,
        "requested_fields": requested_fields,
        "evidence": evidence,
        "current_payload": payload,
        "llm_instruction": (
            "只根据 evidence 和 current_payload 补全 requested_fields；"
            "不得改写 evidence，不得编造原文未出现的方剂/穴位/药名，"
            "不得做现代剂量换算。输出仍必须符合 SyndromeEntry 字段结构。"
        ),
    }


def write_queue(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_markdown(report: dict[str, Any], items: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 结构化方证字典质量审计",
        "",
        f"- 总条目：{report['entry_count']}",
        f"- 进入复核队列：{report['queue_count']}",
        f"- 阈值：quality_score >= {report['threshold']}",
        "",
        "## 按来源类型统计",
        "",
    ]
    for source_type, data in sorted(report["by_source_type"].items()):
        lines.append(
            f"- {source_type}: total={data['total']}, queued={data['queued']}, "
            f"avg_score={data['avg_score']:.2f}, max_score={data['max_score']}"
        )
    lines.extend(["", "## 高频原因", ""])
    for reason, count in report["reason_counts"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## 复核队列样例", ""])
    for item in items[:30]:
        lines.extend(
            [
                f"### {item['title']}",
                f"- 来源：{item['source_book']} / {item['source_type']}",
                f"- 分数：{item['quality_score']}",
                f"- 原因：{'；'.join(item['reasons'])}",
                f"- 需补字段：{', '.join(item['requested_fields']) or '无'}",
                f"- 证据：{item['evidence'][:180]}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="审计结构化方证字典质量，并导出 LLM/人工复核队列")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--max-queue", type=int, default=300, help="0 表示不限制")
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--queue-jsonl", default=str(DEFAULT_QUEUE_JSONL))
    args = parser.parse_args()

    entries = load_entries(Path(args.jsonl))
    scored: list[tuple[int, SyndromeEntry, list[str], list[str]]] = []
    by_source: dict[str, list[int]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    for entry in entries:
        score, reasons, requested_fields = score_entry(entry)
        scored.append((score, entry, reasons, requested_fields))
        by_source[entry.source_type].append(score)
        for reason in reasons:
            reason_counts[reason] += 1

    queue = [
        queue_item(entry, score, reasons, requested_fields)
        for score, entry, reasons, requested_fields in sorted(scored, key=lambda item: item[0], reverse=True)
        if score >= args.threshold
    ]
    if args.max_queue:
        queue = queue[: args.max_queue]

    queued_by_source = Counter(item["source_type"] for item in queue)
    report = {
        "entry_count": len(entries),
        "threshold": args.threshold,
        "queue_count": len(queue),
        "queue_jsonl": str(Path(args.queue_jsonl)),
        "by_source_type": {
            source_type: {
                "total": len(scores),
                "queued": int(queued_by_source[source_type]),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores) if scores else 0,
            }
            for source_type, scores in sorted(by_source.items())
        },
        "reason_counts": dict(reason_counts.most_common()),
    }

    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_queue(queue, Path(args.queue_jsonl))
    write_markdown(report, queue, Path(args.report_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
