from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from generate_chatmed_blind_questions import clean_source_question, contains_pii, file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ChatMed_TCM-v0.2.json"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "evals" / "gold" / "seeds" / "chatmed" / "private"

QUOTAS = OrderedDict(
    [
        ("single_symptom", 80),
        ("multi_symptom", 140),
        ("fuzzy_colloquial", 80),
        ("negation_contradiction", 60),
        ("typo_asr", 40),
        ("formula_herb", 70),
        ("classics_theory", 50),
        ("multi_turn", 40),
        ("insufficient_safety", 40),
    ]
)

CATEGORY_LABELS = {
    "single_symptom": "单一明确症状",
    "multi_symptom": "多症状组合",
    "fuzzy_colloquial": "模糊口语、隐喻表达",
    "negation_contradiction": "否定、矛盾、反问",
    "typo_asr": "错别字、语音转写、方言",
    "formula_herb": "方剂、本草知识",
    "classics_theory": "古籍理论、针法知识",
    "multi_turn": "多轮上下文",
    "insufficient_safety": "信息不足、域外及危险问题",
}

NO_SWEAT_RE = re.compile(r"无汗|不出汗|没出汗|没有汗|汗不出来")
HEADACHE_RE = re.compile(r"头痛|头疼|脑袋疼|脑壳疼")
INSUFFICIENT_RE = re.compile(r"没有其他症状|无其他症状|只有.{0,12}(?:症状|不适)|信息不足")
NEGATION_RE = re.compile(r"没有|无|不|未|否认|并非|不是")
DANGER_RE = re.compile(
    r"半身不遂|昏迷|胸痛|呼吸困难|大出血|中毒|自杀|过量|抽搐|痫病|高热|孕妇|怀孕|婴儿|儿童|癌"
)
CLASSICS_RE = re.compile(
    r"伤寒论|金匮要略|黄帝内经|素问|灵枢|难经|神农本草经|本草纲目|温病条辨|六经|十二经|"
    r"经络|经脉|针刺|针灸|穴位|腧穴|补泻|迎随|阴阳五行|脏腑理论|气血津液|八纲"
)
FORMULA_RE = re.compile(r"中药|方剂|药材|功效|主治|配伍|组成|禁忌|用法|汤|丸|散|饮")
KNOWLEDGE_RE = re.compile(r"什么是|是什么|为何|为什么|如何鉴别|有何不同|区别|功效|作用|组成|主治|配伍|禁忌")
PATIENT_RE = re.compile(r"我|患者|最近|这几天|怎么办|怎么回事|如何治疗|推荐")
FIRST_PERSON_RE = re.compile(r"我|本人|最近|这几天|怎么办|怎么回事")
META_REASONING_RE = re.compile(r"推理过程|考虑所有症状|一步步")
TEMPLATE_OR_PII_RE = re.compile(
    r"姓名\s*[:：]|身份证|手机号|电话\s*[:：]|<[^>]+>|为我写一篇|请给出一篇|推荐一本|"
    r"科研|论文|文献|获取全文|写文章"
)
THEORY_QUESTION_RE = re.compile(r"是什么意思|为什么|为何|如何理解|理论|原文|条文|关系|规律|作用|原则")
FORMULA_KNOWLEDGE_RE = re.compile(r"组成|功效|药性|配伍|方子|药方|处方|每味|用法用量|药材列表|方名")
EDIT_TEMPLATE_RE = re.compile(r"请分析|请编辑|请抽取|请将下面|请解析以下|根据患者.{0,12}(?:诊断|判断)")
BOILERPLATE_RE = re.compile(
    r"请帮我推荐中药或者方剂|请推荐中药(?:或者方剂)?|有什么中药推荐|有什么方剂推荐|请问该怎么治疗"
)
LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")


@dataclass(frozen=True)
class Candidate:
    source_line: int
    question: str
    answer: str
    question_hash: str


def iter_candidates(path: Path) -> Iterator[Candidate]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            question = clean_source_question(str(item.get("query", "")))
            answer = str(item.get("response", "")).strip()
            normalized_source = BOILERPLATE_RE.sub("", question)
            normalized = re.sub(r"[\s，,。；;、？?！!]", "", normalized_source)
            if (
                not 8 <= len(question) <= 300
                or len(answer) < 8
                or contains_pii(question)
                or TEMPLATE_OR_PII_RE.search(question)
                or LATIN_WORD_RE.search(question)
            ):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            yield Candidate(
                source_line=line_number,
                question=question,
                answer=answer,
                question_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )


def _clause_count(question: str) -> int:
    return len([part for part in re.split(r"[，,。；;、]|而且|并且|还有|同时|伴有", question) if len(part.strip()) >= 2])


def _clinical_candidate(candidate: Candidate) -> bool:
    question = candidate.question
    return bool(FIRST_PERSON_RE.search(question)) and not bool(
        FORMULA_KNOWLEDGE_RE.search(question) or CLASSICS_RE.search(question) or EDIT_TEMPLATE_RE.search(question)
    )


def category_score(category: str, candidate: Candidate) -> int:
    question = candidate.question
    clauses = _clause_count(question)
    length = len(question)
    insufficient = bool(INSUFFICIENT_RE.search(question))
    knowledge = bool(KNOWLEDGE_RE.search(question))
    formula = bool(FORMULA_RE.search(question))
    classics = bool(CLASSICS_RE.search(question))
    danger = bool(DANGER_RE.search(question))

    if category == "classics_theory":
        return 180 * classics + 70 * bool(THEORY_QUESTION_RE.search(question)) + 20 * knowledge - 80 * bool(re.search(r"推荐|如何治疗", question)) + min(length, 80)
    if category == "formula_herb":
        return 90 * formula + 35 * knowledge - 30 * classics + min(length, 80)
    if category == "insufficient_safety":
        return 100 * insufficient + 80 * danger + 220 * bool(NO_SWEAT_RE.search(question)) + min(length, 80)
    if category == "multi_turn":
        return 15 * clauses + min(length, 160) + 60 * _clinical_candidate(candidate) - 100 * knowledge
    if category == "negation_contradiction":
        return 40 * len(NEGATION_RE.findall(question)) + 30 * insufficient + min(length, 80)
    if category == "multi_symptom":
        return 25 * clauses + min(length, 140) + 60 * _clinical_candidate(candidate) - 80 * insufficient - 70 * knowledge
    if category == "single_symptom":
        return 110 * insufficient + max(0, 80 - length) - 15 * max(0, clauses - 2) - 40 * knowledge
    if category == "fuzzy_colloquial":
        return 100 * _clinical_candidate(candidate) + max(0, 100 - length) - 50 * knowledge
    if category == "typo_asr":
        return 100 * _clinical_candidate(candidate) + max(0, 100 - abs(length - 55)) - 60 * knowledge
    raise KeyError(category)


def collect_pools(path: Path, pool_multiplier: int) -> tuple[dict[str, list[Candidate]], int]:
    heaps: dict[str, list[tuple[int, str, Candidate]]] = {category: [] for category in QUOTAS}
    eligible = 0
    for candidate in iter_candidates(path):
        eligible += 1
        for category, quota in QUOTAS.items():
            score = category_score(category, candidate)
            entry = (score, candidate.question_hash, candidate)
            heap = heaps[category]
            capacity = quota * pool_multiplier
            if len(heap) < capacity:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    pools = {
        category: [entry[2] for entry in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)]
        for category, heap in heaps.items()
    }
    return pools, eligible


StylePredicate = Callable[[Candidate], bool]


def style_allocations(category: str) -> list[tuple[str, int, StylePredicate, str]]:
    any_candidate: StylePredicate = lambda candidate: True
    if category == "fuzzy_colloquial":
        return [
            (
                "headache_metaphor",
                25,
                lambda candidate: bool(HEADACHE_RE.search(candidate.question)) and _clinical_candidate(candidate),
                "answer",
            ),
            ("colloquial_dialect", 30, _clinical_candidate, "answer"),
            ("fuzzy_general", 25, _clinical_candidate, "answer"),
        ]
    if category == "insufficient_safety":
        return [
            (
                "missing_no_sweat",
                20,
                lambda candidate: bool(NO_SWEAT_RE.search(candidate.question)) and _clinical_candidate(candidate),
                "clarify",
            ),
            (
                "safety_clarify",
                20,
                lambda candidate: _clinical_candidate(candidate)
                and bool(INSUFFICIENT_RE.search(candidate.question) or DANGER_RE.search(candidate.question)),
                "clarify_or_referral",
            ),
        ]
    style_map = {
        "single_symptom": "single_symptom_colloquial",
        "multi_symptom": "multi_symptom_colloquial",
        "negation_contradiction": "negative_rhetorical",
        "typo_asr": "input_typos",
        "formula_herb": "knowledge_paraphrase",
        "classics_theory": "theory_paraphrase",
        "multi_turn": "multi_turn",
    }
    predicates: dict[str, StylePredicate] = {
        "single_symptom": lambda candidate: bool(INSUFFICIENT_RE.search(candidate.question)) and _clinical_candidate(candidate),
        "multi_symptom": lambda candidate: 3 <= _clause_count(candidate.question) <= 9 and _clinical_candidate(candidate),
        "negation_contradiction": lambda candidate: bool(NEGATION_RE.search(candidate.question)) and _clinical_candidate(candidate),
        "typo_asr": _clinical_candidate,
        "formula_herb": lambda candidate: bool(FORMULA_RE.search(candidate.question)),
        "classics_theory": lambda candidate: bool(CLASSICS_RE.search(candidate.question)),
        "multi_turn": lambda candidate: len(candidate.question) >= 50
        and 3 <= _clause_count(candidate.question) <= 10
        and _clinical_candidate(candidate),
    }
    return [(style_map[category], QUOTAS[category], predicates[category], "answer")]


def select_disjoint_seeds(pools: dict[str, list[Candidate]]) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {category: [] for category in QUOTAS}
    used_hashes: set[str] = set()
    category_order = [
        "classics_theory",
        "formula_herb",
        "insufficient_safety",
        "multi_turn",
        "negation_contradiction",
        "multi_symptom",
        "single_symptom",
        "fuzzy_colloquial",
        "typo_asr",
    ]
    for category in category_order:
        for style, count, predicate, expected_behavior in style_allocations(category):
            picked = 0
            for candidate in pools[category]:
                if candidate.question_hash in used_hashes or not predicate(candidate):
                    continue
                used_hashes.add(candidate.question_hash)
                seed_id = f"seed_{category}_{len(selected[category]) + 1:04d}"
                selected[category].append(
                    {
                        "seed_id": seed_id,
                        "category_key": category,
                        "category": CATEGORY_LABELS[category],
                        "generation_style": style,
                        "expected_behavior": expected_behavior,
                        "source_dataset": DEFAULT_INPUT.name,
                        "source_line": candidate.source_line,
                        "source_question_hash": candidate.question_hash,
                        "source_question": candidate.question,
                        "reference_answer": candidate.answer,
                        "source_answer_status": "imported_unverified",
                        "review_status": "seed_unreviewed",
                    }
                )
                picked += 1
                if picked == count:
                    break
            if picked != count:
                raise RuntimeError(f"{category}/{style} 仅选出 {picked}/{count} 条，请扩大候选池")
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按金标准清单从 ChatMed TCM 抽取600条分层盲测种子")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool-multiplier", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pool_multiplier < 2:
        raise SystemExit("--pool-multiplier 必须至少为2")
    pools, eligible = collect_pools(args.input, args.pool_multiplier)
    selected = select_disjoint_seeds(pools)
    for category, records in selected.items():
        write_jsonl(args.output_dir / f"{category}.jsonl", records)

    all_hashes = [record["source_question_hash"] for records in selected.values() for record in records]
    manifest = {
        "source_dataset": str(args.input),
        "source_sha256": file_sha256(args.input),
        "eligible_unique_records": eligible,
        "total_seed_count": len(all_hashes),
        "unique_source_questions": len(set(all_hashes)),
        "quotas": dict(QUOTAS),
        "actual_counts": {category: len(records) for category, records in selected.items()},
        "style_counts": {
            style: sum(record["generation_style"] == style for records in selected.values() for record in records)
            for style in sorted({record["generation_style"] for records in selected.values() for record in records})
        },
        "contains_reference_answers": True,
        "rag_must_not_read_this_directory": True,
    }
    manifest_path = args.output_dir.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
