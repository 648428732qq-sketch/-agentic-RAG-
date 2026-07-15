"""从结构化字典分层抽取未被旧评测使用的 Query Translator V2 种子。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_EVAL_ROOT = ROOT / "tests" / "evals"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_ROOT / "query_translator_v2_unseen_seed_pool_100"

SOURCE_QUOTAS = {
    "formula_syndrome": 30,
    "classical_clause": 14,
    "herb_indication": 21,
    "classical_acupuncture": 15,
    "classical_theory": 10,
    "classical_acupuncture_principle": 10,
}
STYLE_QUOTAS = {
    "metaphor_incomplete": 24,
    "asr_homophone_typos": 24,
    "negation_uncertainty": 4,
    "multi_symptom_one_to_n": 24,
    "hard_negative_clarify": 24,
}
NEGATION_SOURCE_RESERVES: dict[str, int] = {}
DIRECTION_KEYWORDS = {
    "respiratory": ("咳", "喘", "痰", "肺", "胸闷", "不得平卧", "鼻"),
    "digestive": ("胃", "腹", "呕", "吐", "下利", "泄泻", "便秘", "食", "口渴"),
    "exterior_cold_heat": ("恶寒", "恶风", "发热", "无汗", "汗出", "表", "风寒", "温病"),
    "pain_musculoskeletal": ("痛", "疼", "痿", "痹", "筋", "骨", "关节", "项背", "腰"),
    "fluid_urinary": ("小便", "水肿", "浮肿", "水气", "淋", "尿", "湿"),
    "spirit_neurology": ("神", "魂", "魄", "失眠", "不寐", "眩", "惊", "癫", "痫", "悸"),
    "sensory_head_face": ("目", "眼", "耳", "鼻", "口", "咽", "头", "面"),
    "circulation_pulse": ("脉", "血", "寸口", "经络", "经脉", "气上冲"),
    "reproductive": ("胎", "产", "妇", "月经", "经闭", "妊娠", "带下", "胞", "遗精", "阳痿"),
    "toxin_skin_trauma": ("疮", "痈", "肿", "毒", "虫", "创", "皮", "肌"),
    "acupuncture_method": ("刺", "针", "灸", "补泻", "迎随", "得气", "九针"),
}

GENERIC_TERMS = {
    "主之",
    "令人",
    "其实",
    "针法",
    "刺法",
    "相关条文",
    "五脏百病",
    "身体五脏百病",
}
TERM_SPLIT_RE = re.compile(r"[，,、。；;：:\n]")
NEGATIVE_MARKERS = ("不", "无", "未", "没", "勿", "莫", "非")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: 顶层必须为对象")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def stable_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def clean_term(value: Any) -> str:
    text = str(value or "").strip().strip("。；;，,、：:")
    if not text or text in GENERIC_TERMS or len(text) > 24:
        return ""
    if TERM_SPLIT_RE.search(text) or "http" in text.lower():
        return ""
    return text


def answer_terms(entry: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for field in ("formula", "herb_name", "intervention_name"):
        raw = str(entry.get(field, "")).strip().strip("。；;，,、：:")
        if not raw:
            continue
        if len(raw) <= 48:
            terms.add(raw)
        terms.update(
            term
            for part in TERM_SPLIT_RE.split(raw)
            if (term := clean_term(part))
        )
    return terms


def overlaps_answer(term: str, blocked: set[str]) -> bool:
    return any(term in answer or answer in term for answer in blocked)


def is_usable_negative(term: str) -> bool:
    return bool(term) and not any(marker in term for marker in NEGATIVE_MARKERS)


def extract_groups(entry: dict[str, Any], limit: int = 4) -> list[list[str]]:
    blocked = answer_terms(entry)
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_group in entry.get("required_symptom_groups", []):
        group = unique([clean_term(term) for term in raw_group])
        group = [term for term in group if term and not overlaps_answer(term, blocked)]
        key = tuple(sorted(group))
        if group and key not in seen:
            groups.append(group[:8])
            seen.add(key)
        if len(groups) >= limit:
            return groups
    occupied = {term for group in groups for term in group}
    for raw_term in entry.get("diagnostic_keys", []):
        term = clean_term(raw_term)
        if not term or overlaps_answer(term, blocked) or term in occupied:
            continue
        if any(term in existing or existing in term for existing in occupied if len(existing) >= 2):
            continue
        groups.append([term])
        occupied.add(term)
        if len(groups) >= limit:
            break
    return groups


def signature(groups: list[list[str]]) -> set[str]:
    return {term for group in groups for term in group}


def signatures_too_similar(left: set[str], right: set[str]) -> bool:
    overlap = left & right
    if len(overlap) < 2:
        return False
    union = left | right
    jaccard = len(overlap) / len(union)
    containment = len(overlap) / min(len(left), len(right))
    return jaccard >= 0.72 or containment >= 0.85


def collect_old_usage(eval_root: Path, output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    output_resolved = output_dir.resolve()
    for path in sorted(eval_root.rglob("*.jsonl")):
        if output_resolved in path.resolve().parents:
            continue
        records.extend(read_jsonl(path))
    formulas: set[str] = set()
    entry_ids: set[str] = set()
    signatures: list[set[str]] = []
    seed_ids: set[str] = set()
    for record in records:
        seed_id = str(record.get("source_seed_id", "")).strip()
        if seed_id:
            seed_ids.add(seed_id)
        for field in ("source_entry_id", "expected_entry_id", "expected_entry_id_in_top_k"):
            value = str(record.get(field, "")).strip()
            if value:
                entry_ids.add(value)
        for field in ("expected_formula", "expected_formula_in_top_k"):
            value = str(record.get(field, "")).strip()
            if value:
                formulas.add(value)
        formulas.update(str(value).strip() for value in record.get("expected_any_formula_in_top_k", []) if value)
        groups = [
            [clean_term(term) for term in group if clean_term(term)]
            for group in record.get("expected_term_groups", [])
        ]
        current = signature([group for group in groups if group])
        if current:
            signatures.append(current)
    return {
        "records": len(records),
        "seed_ids": seed_ids,
        "entry_ids": entry_ids,
        "formulas": formulas,
        "signatures": signatures,
    }


def concept_key(entry: dict[str, Any], groups: list[list[str]]) -> str:
    source_type = str(entry.get("source_type", ""))
    if source_type in {"formula_syndrome", "classical_clause"} and entry.get("formula"):
        return f"formula:{entry['formula']}"
    if source_type == "herb_indication" and entry.get("herb_name"):
        return f"herb:{entry['herb_name']}"
    if source_type == "classical_acupuncture" and entry.get("intervention_name"):
        return f"acupuncture:{entry['intervention_name']}:{'|'.join(groups[0])}"
    return f"{source_type}:{'|'.join(sorted(signature(groups)))}"


def classify_direction(entry: dict[str, Any], groups: list[list[str]]) -> str:
    text = " ".join(
        list(signature(groups))
        + [str(value) for value in entry.get("pathogenesis", [])]
        + [str(entry.get("title", "")), str(entry.get("source_book", ""))]
    )
    scores = {
        direction: sum(text.count(keyword) for keyword in keywords)
        for direction, keywords in DIRECTION_KEYWORDS.items()
    }
    best = max(scores, key=lambda direction: (scores[direction], -list(scores).index(direction)))
    if scores[best] > 0:
        return best
    source_type = str(entry.get("source_type", ""))
    if "acupuncture" in source_type:
        return "acupuncture_method"
    if source_type == "classical_theory":
        return "circulation_pulse"
    return "general_other"


def build_seed_query(entry: dict[str, Any], groups: list[list[str]], negative_term: str = "") -> str:
    symptoms = "、".join(group[0] for group in groups)
    source_type = str(entry.get("source_type", ""))
    if source_type in {"formula_syndrome", "classical_clause"}:
        query = f"出现{symptoms}，从本地方证库看应匹配哪类方证？"
    elif source_type == "herb_indication":
        query = f"古籍本草中，哪些药物适用于{symptoms}？"
    elif source_type == "classical_acupuncture":
        query = f"出现{symptoms}，古籍针灸条文如何处理？"
    elif source_type == "classical_acupuncture_principle":
        query = f"古籍针法中，{symptoms}应该怎样理解？"
    else:
        query = f"古籍理论如何解释{symptoms}？"
    if negative_term:
        query = query.rstrip("？") + f"，并且明确没有{negative_term}？"
    return query


def candidate_from_entry(entry: dict[str, Any], old_usage: dict[str, Any]) -> dict[str, Any] | None:
    entry_id = str(entry.get("entry_id", "")).strip()
    source_type = str(entry.get("source_type", "")).strip()
    formula = str(entry.get("formula", "")).strip()
    if not entry_id or source_type not in SOURCE_QUOTAS:
        return None
    if entry_id in old_usage["entry_ids"] or (formula and formula in old_usage["formulas"]):
        return None
    groups = extract_groups(entry)
    if len(groups) < 2:
        return None
    current_signature = signature(groups)
    if any(signatures_too_similar(current_signature, old) for old in old_usage["signatures"]):
        return None
    forbidden = unique([clean_term(term) for term in entry.get("forbidden_terms", [])])
    forbidden = [term for term in forbidden if term and term not in current_signature][:4]
    negatable_forbidden = [term for term in forbidden if is_usable_negative(term)]
    return {
        "entry": entry,
        "entry_id": entry_id,
        "source_type": source_type,
        "groups": groups,
        "signature": current_signature,
        "forbidden_terms": forbidden,
        "negatable_forbidden_terms": negatable_forbidden,
        "direction": classify_direction(entry, groups),
        "concept_key": concept_key(entry, groups),
        "quality": len(current_signature) + len(groups) + (2 if entry.get("evidence") else 0),
    }


def select_source_candidates(
    candidates: list[dict[str, Any]],
    quota: int,
    salt: str,
    selected_concepts: set[str],
    selected_signatures: list[set[str]],
    minimum_negatable: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    direction_counts: Counter[str] = Counter()
    remaining = list(candidates)
    while len(selected) < quota:
        eligible = [
            item
            for item in remaining
            if item["concept_key"] not in selected_concepts
            and not any(signatures_too_similar(item["signature"], old) for old in selected_signatures)
        ]
        reserved_needed = max(0, minimum_negatable - len([item for item in selected if item["negatable_forbidden_terms"]]))
        slots_left = quota - len(selected)
        if reserved_needed >= slots_left:
            eligible = [item for item in eligible if item["negatable_forbidden_terms"]]
        if not eligible:
            raise RuntimeError(f"无法满足来源配额 {quota}，当前仅选出 {len(selected)}")
        chosen = min(
            eligible,
            key=lambda item: (
                0 if reserved_needed == 0 or item["negatable_forbidden_terms"] else 1,
                direction_counts[item["direction"]],
                -item["quality"],
                stable_hash(item["entry_id"], salt),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        selected_concepts.add(chosen["concept_key"])
        selected_signatures.append(chosen["signature"])
        direction_counts[chosen["direction"]] += 1
    return selected


def assign_styles(selected: list[dict[str, Any]], salt: str) -> None:
    unassigned = list(selected)
    style_source_counts: dict[str, Counter[str]] = {style: Counter() for style in STYLE_QUOTAS}
    style_direction_counts: dict[str, Counter[str]] = {style: Counter() for style in STYLE_QUOTAS}
    style_indexes: Counter[str] = Counter()

    def assign_one(style: str) -> None:
        index = style_indexes[style]
        eligible = [
            item
            for item in unassigned
            if style != "negation_uncertainty" or item["negatable_forbidden_terms"]
        ]
        if style in {"hard_negative_clarify", "multi_symptom_one_to_n"}:
            eligible = [item for item in eligible if len(item["groups"]) >= 2]
        if not eligible:
            raise RuntimeError(f"无法为 {style} 分配足够种子")
        chosen = min(
            eligible,
            key=lambda item: (
                style_source_counts[style][item["source_type"]],
                style_direction_counts[style][item["direction"]],
                stable_hash(f"{style}|{item['entry_id']}", salt),
            ),
        )
        chosen["planned_style"] = style
        chosen["planned_variant_index"] = index
        if style == "hard_negative_clarify":
            chosen["planned_omitted_group_index"] = int(
                stable_hash(chosen["entry_id"], salt)[:8], 16
            ) % len(chosen["groups"])
        if style == "negation_uncertainty":
            chosen["expected_negative_terms"] = [chosen["negatable_forbidden_terms"][0]]
        unassigned.remove(chosen)
        style_source_counts[style][chosen["source_type"]] += 1
        style_direction_counts[style][chosen["direction"]] += 1
        style_indexes[style] += 1

    # Negation has the narrowest eligibility, so reserve it first. The other
    # styles rotate over the shared pool to keep each style source-diverse.
    for _ in range(STYLE_QUOTAS["negation_uncertainty"]):
        assign_one("negation_uncertainty")
    rotating_styles = (
        "metaphor_incomplete",
        "asr_homophone_typos",
        "multi_symptom_one_to_n",
        "hard_negative_clarify",
    )
    while any(style_indexes[style] < STYLE_QUOTAS[style] for style in rotating_styles):
        for style in rotating_styles:
            if style_indexes[style] < STYLE_QUOTAS[style]:
                assign_one(style)
    if unassigned:
        raise AssertionError(f"存在未分配风格的种子: {len(unassigned)}")


def export_seed(item: dict[str, Any], salt: str) -> dict[str, Any]:
    entry = item["entry"]
    entry_id = item["entry_id"]
    negative_terms = item.get("expected_negative_terms", [])
    payload_json = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    seed: dict[str, Any] = {
        "id": f"v2_seed_{stable_hash(entry_id, salt)[:16]}",
        "source_seed_id": f"entry::{entry_id}",
        "source_entry_id": entry_id,
        "source_payload_hash": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "source_type": item["source_type"],
        "source_book": entry.get("source_book", ""),
        "source_file": entry.get("source_file", ""),
        "direction": item["direction"],
        "planned_style": item["planned_style"],
        "planned_variant_index": item["planned_variant_index"],
        "query": build_seed_query(entry, item["groups"], negative_terms[0] if negative_terms else ""),
        "expected_term_groups": item["groups"],
        "expected_negative_terms": negative_terms,
        "forbidden_terms": item["forbidden_terms"],
        "expected_entry_id_in_top_k": entry_id,
        "expected_source_type_in_top_k": item["source_type"],
        "expected_gate": True,
        "review_status": "auto_extracted_from_local_payload_pending_generation",
        "source_evidence": str(entry.get("evidence", ""))[:1200],
    }
    if item.get("planned_region"):
        seed["planned_region"] = item["planned_region"]
    if "planned_omitted_group_index" in item:
        seed["planned_omitted_group_index"] = item["planned_omitted_group_index"]
    formula = str(entry.get("formula", "")).strip()
    if formula:
        seed["expected_formula_in_top_k"] = formula
    intervention = str(entry.get("herb_name") or entry.get("intervention_name") or "").strip()
    if intervention and intervention not in {"针刺/取穴相关条文", "脉诊/诊法理论"}:
        seed["expected_intervention_text"] = intervention
    return seed


def validate_seeds(
    seeds: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    old_usage: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    source_counts = Counter(seed["source_type"] for seed in seeds)
    style_counts = Counter(seed["planned_style"] for seed in seeds)
    style_sources: dict[str, set[str]] = {
        style: {seed["source_type"] for seed in seeds if seed["planned_style"] == style}
        for style in STYLE_QUOTAS
    }
    for source_type, quota in SOURCE_QUOTAS.items():
        if source_counts[source_type] != quota:
            errors.append(f"来源配额错误: {source_type}={source_counts[source_type]} != {quota}")
    for style, quota in STYLE_QUOTAS.items():
        if style_counts[style] != quota:
            errors.append(f"风格配额错误: {style}={style_counts[style]} != {quota}")
        minimum_sources = 1 if style == "negation_uncertainty" and quota <= 5 else (3 if style == "negation_uncertainty" else 4)
        if len(style_sources[style]) < minimum_sources:
            errors.append(
                f"风格来源多样性不足: {style} 仅 {len(style_sources[style])} 类，要求 {minimum_sources} 类"
            )
    for seed in seeds:
        entry_id = seed["source_entry_id"]
        entry = raw_by_id[entry_id]
        query = seed["query"]
        if entry_id in old_usage["entry_ids"]:
            errors.append(f"旧条目泄漏: {seed['id']}:{entry_id}")
        for answer in answer_terms(entry):
            if len(answer) >= 2 and answer in query:
                errors.append(f"答案泄漏: {seed['id']}:{answer}")
        if "没有不" in query or "没有无" in query:
            errors.append(f"双重否定: {seed['id']}:{query}")
        if seed["planned_style"] == "negation_uncertainty":
            negatives = seed.get("expected_negative_terms", [])
            if not negatives or not all(is_usable_negative(term) for term in negatives):
                errors.append(f"无效否定标签: {seed['id']}:{negatives}")
        payload_json = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        actual_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if seed["source_payload_hash"] != actual_hash:
            errors.append(f"payload 哈希错误: {seed['id']}")
    if errors:
        raise AssertionError("种子完整性校验失败:\n" + "\n".join(errors[:30]))
    return {
        "answer_leaks": 0,
        "double_negations": 0,
        "old_entry_id_overlap": 0,
        "payload_hash_mismatches": 0,
        "minimum_source_types_per_style": min(map(len, style_sources.values())),
        "style_source_type_counts": {
            style: len(style_sources[style]) for style in sorted(style_sources)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--salt", default="query-translator-v2-unseen-20260629")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(SOURCE_QUOTAS.values()) != 100 or sum(STYLE_QUOTAS.values()) != 100:
        raise AssertionError("当前配额必须各自合计 100")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在；如需重建请显式使用 --overwrite: {args.output_dir}")
    old_usage = collect_old_usage(args.eval_root, args.output_dir)
    raw_entries = read_jsonl(args.dictionary)
    raw_by_id = {str(entry.get("entry_id", "")): entry for entry in raw_entries}
    candidates = [candidate_from_entry(entry, old_usage) for entry in raw_entries]
    candidates = [candidate for candidate in candidates if candidate]
    by_source: dict[str, list[dict[str, Any]]] = {
        source_type: [item for item in candidates if item["source_type"] == source_type]
        for source_type in SOURCE_QUOTAS
    }
    selected: list[dict[str, Any]] = []
    selected_concepts: set[str] = set()
    selected_signatures: list[set[str]] = []
    selection_order = (
        "classical_theory",
        "classical_clause",
        "formula_syndrome",
        "herb_indication",
        "classical_acupuncture",
        "classical_acupuncture_principle",
    )
    for source_type in selection_order:
        quota = SOURCE_QUOTAS[source_type]
        picked = select_source_candidates(
            by_source[source_type],
            quota,
            args.salt,
            selected_concepts,
            selected_signatures,
            NEGATION_SOURCE_RESERVES.get(source_type, 0),
        )
        selected.extend(picked)
    assign_styles(selected, args.salt)
    seeds = [export_seed(item, args.salt) for item in selected]
    seeds.sort(key=lambda seed: stable_hash(seed["id"], args.salt))
    ids = [seed["id"] for seed in seeds]
    entry_ids = [seed["source_entry_id"] for seed in seeds]
    if len(seeds) != 100 or len(set(ids)) != 100 or len(set(entry_ids)) != 100:
        raise AssertionError("种子数或唯一性检查失败")
    if set(entry_ids) & old_usage["entry_ids"]:
        raise AssertionError("新种子与旧 entry_id 泄漏")
    integrity_checks = validate_seeds(seeds, raw_by_id, old_usage)
    write_jsonl(args.output_dir / "private" / "seeds.jsonl", seeds)
    manifest = {
        "status": "seed_pool_frozen_pending_qwen_generation",
        "count": len(seeds),
        "dictionary": str(args.dictionary),
        "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
        "salt": args.salt,
        "source_type_counts": dict(sorted(Counter(seed["source_type"] for seed in seeds).items())),
        "style_counts": dict(sorted(Counter(seed["planned_style"] for seed in seeds).items())),
        "direction_counts": dict(sorted(Counter(seed["direction"] for seed in seeds).items())),
        "source_style_matrix": {
            style: dict(
                sorted(
                    Counter(
                        seed["source_type"]
                        for seed in seeds
                        if seed["planned_style"] == style
                    ).items()
                )
            )
            for style in sorted(STYLE_QUOTAS)
        },
        "unique_entry_ids": len(set(entry_ids)),
        "unique_concepts": len(selected_concepts),
        "old_gold_records_scanned": old_usage["records"],
        "old_seed_ids_excluded": len(old_usage["seed_ids"]),
        "old_entry_ids_excluded": len(old_usage["entry_ids"]),
        "old_formulas_excluded": sorted(old_usage["formulas"]),
        "public_questions_generated": False,
        "private_labels_source": "local_syndrome_dictionary_payload",
        "llm_generates_answers": False,
        "allowed_usage": "generate_once_then_final_holdout",
        "integrity_checks": integrity_checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
