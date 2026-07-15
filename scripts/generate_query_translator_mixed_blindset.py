from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = ROOT / "tests" / "evals" / "query_translator_cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "evals" / "query_translator_mixed_300"

STYLE_KEYS = (
    "metaphor_incomplete",
    "asr_homophone_typos",
    "negation_uncertainty",
    "multi_symptom_one_to_n",
    "hard_negative_clarify",
)
STYLE_INSTRUCTIONS = {
    "metaphor_incomplete": (
        "使用生活化隐喻、夸张、倒装或不完整句表达全部症状，句子可以松散，"
        "但不得新增、删除或改变症状和否定关系。"
    ),
    "asr_homophone_typos": (
        "模拟口齿不清的老年人使用微信语音转文字，保留全部事实，并制造2到3处"
        "同音或近音错字；错字不能改变真实语义。"
    ),
    "negation_uncertainty": (
        "使用带情绪的否定、反问或不确定句式，必须准确保留输入中明确给出的阳性症状"
        "和阴性症状，不得把阴性症状翻转为阳性。"
    ),
    "multi_symptom_one_to_n": (
        "把全部独立症状糅合为一句连贯抱怨，不使用逗号、顿号或分号分隔症状，"
        "不得遗漏任何一组症状。"
    ),
    "hard_negative_clarify": (
        "生成必须追问的困难反例：完整保留其余症状，但绝口不提指定遗漏症状及其任何近义表达；"
        "不得暗示答案或方剂，must_clarify必须为true。"
    ),
}

SYSTEM_PROMPT = """
你是本地中医RAG的Query Translator盲测问题生成器。你只改写用户问题，不回答、不诊断、不推荐方剂。

硬性规则：
1. 只能表达输入给出的阳性症状、阴性症状和指定遗漏项，不得增加病因、证候、病史、舌脉、药物或方剂。
2. expected_term_groups中的每一组是同一症状的允许规范表达；除hard negative指定遗漏组外，每组必须在问句中保留一个可识别的口语证据片段。
3. expected_negative_terms必须保持为明确否认，polarity=absent；不得翻转。
4. 每个evidence_mapping的source_phrase必须逐字出现在query中；canonical_term只能选自输入允许的本地规范词。
5. hard negative不得出现omitted_term_group中任何词或近义暗示。
6. 不得输出分类标签、答案、解释或Markdown，只输出JSON对象。

JSON格式：
{
  "query": "混淆后的患者原话",
  "evidence_mappings": [
    {"source_phrase": "逐字证据", "canonical_term": "允许规范词", "polarity": "present或absent"}
  ],
  "typo_pairs": [{"correct": "正确字词", "typo": "query中实际错字"}],
  "must_clarify": false,
  "added_facts": [],
  "omitted_terms_present": false
}
""".strip()


@dataclass(frozen=True)
class Plan:
    case_id: str
    style: str
    region: str
    seed: dict[str, Any]
    variant_index: int
    omitted_group_index: int | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: 顶层必须是对象")
        records.append(value)
    return records


def write_atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_generation_manifest(output_dir: Path, summary: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            existing = value
    if isinstance(existing.get("integrity_checks"), dict) and "count" in existing:
        manifest = {**existing, "generation": summary}
        if (
            summary.get("complete")
            and existing.get("status") == "seed_pool_frozen_pending_qwen_generation"
        ):
            manifest["status"] = "final_holdout_generated_pending_system_evaluation"
    else:
        manifest = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def term_groups(seed: dict[str, Any]) -> list[list[str]]:
    groups = seed.get("expected_term_groups") or [[term] for term in seed.get("expected_terms", [])]
    return [[str(term) for term in group if str(term).strip()] for group in groups if group]


def eligible(style: str, seed: dict[str, Any]) -> bool:
    groups = term_groups(seed)
    if style == "negation_uncertainty":
        return bool(seed.get("expected_negative_terms")) and bool(groups)
    if style in {"multi_symptom_one_to_n", "hard_negative_clarify"}:
        return len(groups) >= 2
    return bool(groups)


def build_plans(
    seeds: list[dict[str, Any]],
    per_style: int,
    styles: tuple[str, ...] = STYLE_KEYS,
) -> list[Plan]:
    plans: list[Plan] = []
    for style_index, style in enumerate(styles):
        candidates = [seed for seed in seeds if eligible(style, seed)]
        if not candidates:
            raise ValueError(f"没有适用于{style}的种子")
        for index in range(per_style):
            seed = candidates[(index * 7 + style_index) % len(candidates)]
            groups = term_groups(seed)
            omitted = (index + style_index) % len(groups) if style == "hard_negative_clarify" else None
            region = ""
            identity = f"{seed['id']}|{style}|{index}|{omitted}|{region}"
            case_id = f"qt_mix_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
            plans.append(Plan(case_id, style, region, seed, index, omitted))
    return plans


def build_one_per_seed_plans(seeds: list[dict[str, Any]]) -> list[Plan]:
    plans: list[Plan] = []
    for index, seed in enumerate(seeds):
        style = str(seed.get("planned_style", ""))
        if style not in STYLE_KEYS:
            continue
        if not eligible(style, seed):
            raise ValueError(f"种子 {seed.get('id')} 不适用于 {style}")
        groups = term_groups(seed)
        omitted = None
        if style == "hard_negative_clarify":
            omitted = int(seed.get("planned_omitted_group_index", index % len(groups)))
            if not 0 <= omitted < len(groups):
                raise ValueError(f"种子 {seed.get('id')} 的遗漏组索引越界")
        region = ""
        variant_index = int(seed.get("planned_variant_index", 0))
        identity = f"{seed['id']}|{style}|{variant_index}|{omitted}|{region}"
        case_id = f"qt_mix_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        plans.append(Plan(case_id, style, region, seed, variant_index, omitted))
    return plans


def effective_groups(plan: Plan) -> list[list[str]]:
    return [
        group
        for index, group in enumerate(term_groups(plan.seed))
        if index != plan.omitted_group_index
    ]


def build_prompt(plan: Plan, previous_error: str = "") -> str:
    groups = term_groups(plan.seed)
    omitted = groups[plan.omitted_group_index] if plan.omitted_group_index is not None else []
    payload = {
        "原始问题": plan.seed["query"],
        "允许的本地规范症状组": groups,
        "必须保持否认的本地规范词": plan.seed.get("expected_negative_terms", []),
        "禁止变成阳性的规范词": plan.seed.get("forbidden_terms", []),
        "指定遗漏症状组": omitted,
        "目标改写规则": STYLE_INSTRUCTIONS[plan.style].format(region=plan.region),
        "输出约束": {
            "must_clarify": plan.style == "hard_negative_clarify",
            "typo_pairs_count": "2到3" if plan.style == "asr_homophone_typos" else 0,
            "added_facts": [],
            "omitted_terms_present": False,
        },
    }
    prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    if previous_error:
        prompt += f"\n上一版未通过程序校验，必须修正：{previous_error}"
    return prompt


def parse_json_object(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型返回JSON顶层不是对象")
    return value


def validate(plan: Plan, generated: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    query = generated.get("query")
    if not isinstance(query, str) or not 4 <= len(query.strip()) <= 280:
        return ["query必须是4到280字的字符串"]
    mappings = generated.get("evidence_mappings")
    if not isinstance(mappings, list):
        return ["evidence_mappings必须是数组"]

    expected_groups = effective_groups(plan)
    allowed_positive = {term for group in expected_groups for term in group}
    expected_negative = {str(term) for term in plan.seed.get("expected_negative_terms", [])}
    covered_positive: set[str] = set()
    covered_negative: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            errors.append("evidence_mapping必须是对象")
            continue
        phrase = str(mapping.get("source_phrase", ""))
        canonical = str(mapping.get("canonical_term", ""))
        polarity = str(mapping.get("polarity", ""))
        if not phrase or phrase not in query:
            errors.append("映射证据未逐字出现在query")
        if polarity == "present" and canonical in allowed_positive:
            covered_positive.add(canonical)
        elif polarity == "absent" and canonical in expected_negative:
            covered_negative.add(canonical)
        else:
            errors.append(f"越界映射:{canonical}:{polarity}")
    for group in expected_groups:
        if not (set(group) & covered_positive):
            errors.append(f"缺少症状组映射:{group}")
    if not expected_negative.issubset(covered_negative):
        errors.append("缺少阴性症状映射")

    omitted = (
        term_groups(plan.seed)[plan.omitted_group_index]
        if plan.omitted_group_index is not None
        else []
    )
    if any(term in query for term in omitted) or generated.get("omitted_terms_present") is not False:
        errors.append("指定遗漏症状仍出现在query")
    if generated.get("must_clarify") is not (plan.style == "hard_negative_clarify"):
        errors.append("must_clarify不符合计划")
    if generated.get("added_facts") != []:
        errors.append("added_facts必须为空")

    typo_pairs = generated.get("typo_pairs")
    if not isinstance(typo_pairs, list):
        errors.append("typo_pairs必须是数组")
    elif plan.style == "asr_homophone_typos":
        if not 2 <= len(typo_pairs) <= 3:
            errors.append("ASR错字必须为2到3处")
        for pair in typo_pairs:
            if not isinstance(pair, dict) or not str(pair.get("typo", "")) in query:
                errors.append("typo_pairs没有记录query中的真实错字")
    elif typo_pairs:
        errors.append("非ASR风格不得输出typo_pairs")
    if plan.style == "multi_symptom_one_to_n" and any(char in query for char in "，、；;"):
        errors.append("1-to-N句子不能使用症状分隔标点")
    return errors


def request_generation(client: Any, args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        if "response_format" not in str(exc).lower():
            raise
        request.pop("response_format", None)
        response = client.chat.completions.create(**request)
    return parse_json_object(response.choices[0].message.content)


def build_private_record(plan: Plan, generated: dict[str, Any], model: str) -> dict[str, Any]:
    expected_groups = effective_groups(plan)
    omitted_group = (
        term_groups(plan.seed)[plan.omitted_group_index]
        if plan.omitted_group_index is not None
        else []
    )
    forbidden = list(
        dict.fromkeys(
            [str(term) for term in plan.seed.get("forbidden_terms", [])]
            + omitted_group
        )
    )
    record = {
        "id": plan.case_id,
        "query": generated["query"].strip(),
        "source_seed_id": plan.seed["id"],
        "generation_style": plan.style,
        "region": plan.region,
        "expected_term_groups": expected_groups,
        "expected_negative_terms": plan.seed.get("expected_negative_terms", []),
        "forbidden_terms": forbidden,
        "must_clarify": plan.style == "hard_negative_clarify",
        "expected_decision": "clarify" if plan.style == "hard_negative_clarify" else plan.seed.get("expected_decision"),
        "omitted_term_group": omitted_group,
        "typo_pairs": generated.get("typo_pairs", []),
        "evidence_mappings": generated.get("evidence_mappings", []),
        "review_status": "qwen_generated_structurally_validated",
        "generation": {
            "provider": "dashscope_openai_compatible",
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "variant_index": plan.variant_index,
        },
    }
    retrieval_expectation_fields = (
        "expected_entry_id",
        "expected_entry_id_in_top_k",
        "expected_formula",
        "expected_formula_in_top_k",
        "expected_any_formula_in_top_k",
        "expected_source_type",
        "expected_source_type_in_top_k",
        "expected_intervention_text",
        "expected_gate",
        "expected_needs_more_info",
    )
    for field in retrieval_expectation_fields:
        if field in plan.seed:
            record[field] = plan.seed[field]
    if plan.style == "hard_negative_clarify":
        record["expected_gate"] = False
        record["expected_needs_more_info"] = True
    return record


def generate_validated_record(
    client: Any,
    args: argparse.Namespace,
    plan: Plan,
) -> tuple[dict[str, Any] | None, str]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            generated = request_generation(client, args, build_prompt(plan, last_error))
            errors = validate(plan, generated)
            if errors:
                raise ValueError("；".join(errors))
            return build_private_record(plan, generated, args.model), ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:800]
            print(
                f"[retry] id={plan.case_id} attempt={attempt}/{args.retries} error={last_error}",
                flush=True,
            )
            if attempt < args.retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    return None, last_error


def export_mixed(records: list[dict[str, Any]], output_dir: Path, shuffle_seed: int) -> None:
    public = [{"id": record["id"], "query": record["query"]} for record in records]
    random.Random(shuffle_seed).shuffle(public)
    write_atomic_jsonl(output_dir / "questions_mixed.jsonl", public)
    write_atomic_jsonl(output_dir / "private" / "gold_keys.jsonl", records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用Qwen生成300条不带分类标签的Query Translator混合盲测题")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-style", type=int, default=60)
    parser.add_argument(
        "--styles",
        default=",".join(STYLE_KEYS),
        help="Comma-separated generation styles; regional dialect is not supported",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260622)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1, help="并发生成数；默认串行，建议最多4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--one-per-seed",
        action="store_true",
        help="每个种子只按 planned_style 生成一条，适用于独立最终集",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_style < 1 or args.retries < 1 or not 1 <= args.workers <= 8:
        raise SystemExit("数量参数无效")
    seeds = read_jsonl(args.seeds)
    selected_styles = tuple(style.strip() for style in args.styles.split(",") if style.strip())
    unknown_styles = sorted(set(selected_styles) - set(STYLE_KEYS))
    if unknown_styles:
        raise ValueError(f"unsupported styles: {unknown_styles}")
    if not selected_styles:
        raise ValueError("at least one generation style is required")
    plans = (
        build_one_per_seed_plans(seeds)
        if args.one_per_seed
        else build_plans(seeds, args.per_style, selected_styles)
    )
    summary = {
        "seed_count": len(seeds),
        "style_count": len(STYLE_KEYS),
        "per_style": None if args.one_per_seed else args.per_style,
        "selected_styles": list(selected_styles),
        "one_per_seed": args.one_per_seed,
        "planned_count": len(plans),
        "public_fields": ["id", "query"],
        "public_contains_style_or_category": False,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    from dotenv import load_dotenv

    load_dotenv(ROOT / "project" / ".env")
    api_key = os.getenv(args.api_key_env)
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("请输入DashScope API Key（不会显示或写入文件）: ").strip()
    if not api_key:
        raise SystemExit(f"未设置{args.api_key_env}；请通过环境变量或--prompt-api-key提供")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0)
    master_path = args.output_dir / "private" / "generation_master.jsonl"
    failure_path = args.output_dir / "private" / "generation_failures.jsonl"
    gold_path = args.output_dir / "private" / "gold_keys.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    for resume_path in (gold_path, master_path):
        existing.update({record["id"]: record for record in read_jsonl(resume_path)})
    planned_ids = {plan.case_id for plan in plans}
    existing = {case_id: record for case_id, record in existing.items() if case_id in planned_ids}
    write_atomic_jsonl(master_path, list(existing.values()))
    failures: list[dict[str, Any]] = []
    pending = [plan for plan in plans if plan.case_id not in existing]
    completed_before = len(plans) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_plans = {
            executor.submit(generate_validated_record, client, args, plan): plan
            for plan in pending
        }
        for completed_now, future in enumerate(as_completed(future_plans), start=1):
            plan = future_plans[future]
            record, last_error = future.result()
            if record is None:
                failures.append(
                    {"id": plan.case_id, "source_seed_id": plan.seed["id"], "error": last_error}
                )
            else:
                existing[plan.case_id] = record
                write_atomic_jsonl(master_path, list(existing.values()))
            position = completed_before + completed_now
            print(
                f"[{position}/{len(plans)}] generated={len(existing)} failed={len(failures)}",
                flush=True,
            )
            if args.request_delay:
                time.sleep(args.request_delay)

    records = [existing[plan.case_id] for plan in plans if plan.case_id in existing]
    export_mixed(records, args.output_dir, args.shuffle_seed)
    write_atomic_jsonl(failure_path, failures)
    summary.update(
        {
            "generated_count": len(records),
            "failed_count": len(failures),
            "complete": len(records) == len(plans),
            "questions": str(args.output_dir / "questions_mixed.jsonl"),
            "private_gold": str(args.output_dir / "private" / "gold_keys.jsonl"),
        }
    )
    write_generation_manifest(args.output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
