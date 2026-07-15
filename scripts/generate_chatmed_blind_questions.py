from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_DIR = ROOT / "tests" / "evals" / "gold" / "seeds" / "chatmed" / "private"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "evals" / "gold" / "candidates" / "chatmed"
DEFAULT_MASTER = DEFAULT_OUTPUT_DIR / "private" / "chatmed_blind_master.jsonl"

SYSTEM_PROMPT = """
你是中医 RAG 盲测问句改写器。输入包含一条原始问诊问题和一条隐藏参考答案。你只能改写问题，不能回答问题。

硬性规则：
1. 保留原问题中的事实，不增加新的症状、病史、检查结果、证候、病因、药名或方剂名。
2. 参考答案仅用于确认改写后问题仍与原答案对应，绝不能把其中的结论、方剂、药物、诊断或解释泄漏到问句。
3. 删除“请输出推理过程”“要求考虑所有症状”等面向模型的元指令，改写必须像真实用户输入。
4. 除指定的信息缺失风格外，不得遗漏会改变参考答案的关键信息。
5. 不得生成姓名、电话、身份证号、邮箱、地址等个人信息。
6. 只输出 JSON 对象，不输出答案、解释或 Markdown。

JSON 格式：
{
  "history": [],
  "query": "最终一轮用户问题",
  "only_intended_change": true,
  "reference_answer_still_applicable": true,
  "intentional_omissions": [],
  "added_facts": [],
  "answer_leakage": false,
  "typo_pairs": [{"correct": "正确词", "typo": "实际出现在history或query中的错别字"}]
}

只有多轮风格允许 history 非空，格式必须为：
[{"role":"user","content":"..."},{"role":"assistant","content":"仅用于澄清的简短追问"}]
assistant 历史不得包含诊断、处方或参考答案。
""".strip()


@dataclass(frozen=True)
class StyleSpec:
    key: str
    instruction: str
    multi_turn: bool = False
    omit_no_sweat: bool = False
    headache_metaphor: bool = False
    typo_min: int = 0
    typo_max: int = 0


STYLES: dict[str, StyleSpec] = {
    "single_symptom_colloquial": StyleSpec(
        "single_symptom_colloquial", "把单一症状改成自然患者口语，保持信息量不变。"
    ),
    "multi_symptom_colloquial": StyleSpec(
        "multi_symptom_colloquial", "用患者大白话重述，必须保留原问题中的每一项症状和否定信息。"
    ),
    "colloquial_dialect": StyleSpec(
        "colloquial_dialect", "使用自然方言或极度口语化表达，允许语序凌乱，但医学事实不能变化。"
    ),
    "fuzzy_general": StyleSpec(
        "fuzzy_general", "使用模糊、非医学化但仍可理解的表达，不得真正删除原问题中的关键事实。"
    ),
    "headache_metaphor": StyleSpec(
        "headache_metaphor", "把头痛改成脑袋像要炸开、裂开或被箍住一类隐喻，其他事实不变。",
        headache_metaphor=True,
    ),
    "negative_rhetorical": StyleSpec(
        "negative_rhetorical", "改成自然的否定或反问语气，不得新增真正被否定的医学事实。"
    ),
    "input_typos": StyleSpec(
        "input_typos", "保留原意并加入2到3个常见输入法错别字，在typo_pairs中逐项记录。",
        typo_min=2,
        typo_max=3,
    ),
    "knowledge_paraphrase": StyleSpec(
        "knowledge_paraphrase", "把方剂或本草知识问题换一种自然问法，不得把参考答案写进问题。"
    ),
    "theory_paraphrase": StyleSpec(
        "theory_paraphrase", "把古籍、证候或针法理论问题换一种问法，保留考查知识点。"
    ),
    "multi_turn": StyleSpec(
        "multi_turn", "把原问题拆成真实多轮问诊：history提供前置信息，query作为最后追问；全部事实合并后与原问题等价。",
        multi_turn=True,
    ),
    "missing_no_sweat": StyleSpec(
        "missing_no_sweat", "只删除无汗、不出汗或汗出不来这一项，保留其他事实，使系统必须追问出汗情况。",
        omit_no_sweat=True,
    ),
    "safety_clarify": StyleSpec(
        "safety_clarify", "用真实患者口语重述，保持危险信号或信息不足状态，不得弱化风险。"
    ),
}

STYLE_PROMPTS = {
    "single_symptom_colloquial": """
任务：把单一症状题改成一条简短、自然的患者口语。
必须：只保留原有单一症状和原有否定信息；使用生活化表达。
禁止：扩展成多症状、补充舌脉、病史、病因或药名；history必须为空。
""",
    "multi_symptom_colloquial": """
任务：把多症状组合题改成真实患者的大白话主诉。
必须：逐项保留原问题中的所有症状、程度、时间、否定和相互关系，可以打乱语序。
禁止：丢掉任何症状、合并掉关键差异、添加诊断或药物；history必须为空。
""",
    "colloquial_dialect": """
任务：生成带自然方言色彩或极度口语化的问法。
必须：使用如“咋回事、难受得慌、脑壳”等自然表达，但医学事实与原问题完全等价。
禁止：堆砌方言词、改变地区含义、增加症状或泄漏答案；history必须为空。
""",
    "fuzzy_general": """
任务：把医学化症状改成患者可能使用的模糊生活描述。
必须：每项原始事实仍能从模糊表达中被推断出来，参考答案继续适用。
禁止：真正删除关键信息、把模糊改成错误、出现方剂或诊断结论；history必须为空。
""",
    "headache_metaphor": """
任务：专门测试头痛隐喻理解。
必须：把原有头痛写成“脑袋像要炸开、裂开、被箍住”等自然隐喻；保留其他所有事实。
禁止：新增眩晕、呕吐等原问题没有的伴随症状；history必须为空。
""",
    "negative_rhetorical": """
任务：用否定、反问或带情绪的句式重述原问题。
必须：保留原问题已有的否定关系；“不出汗”仍表示无汗，不可误判为删除症状。
禁止：凭空否定新的症状、制造与原问题不同的矛盾、泄漏答案；history必须为空。
""",
    "input_typos": """
任务：生成输入法错别字鲁棒性测试。
必须：只制造2到3处常见同音、近音或形近错字；在typo_pairs逐项记录，错字必须真实出现在query。
禁止：错到无法理解、改变药名或症状含义、添加事实；history必须为空。
""",
    "knowledge_paraphrase": """
任务：改写方剂或本草知识问题。
必须：保留原问题考查的方剂、药材、功效、组成或禁忌范围，只改变提问方式。
禁止：把隐藏参考答案中的组成、功效、剂量或结论补进问题；history必须为空。
""",
    "theory_paraphrase": """
任务：改写古籍理论、经络或针法知识问题。
必须：保留原书名、理论对象和考查点，换一种清晰问法。
禁止：替用户作答、引用参考答案中的解释、把理论题改成临床处方题；history必须为空。
""",
    "multi_turn": """
任务：把一个完整问诊问题拆成多轮上下文测试。
必须：history至少包含一条user陈述和一条assistant澄清追问，query是用户最后补充或提问；合并后与原问题事实等价。
禁止：assistant历史给出诊断、处方、提示答案；不得在拆分时遗漏事实或新增事实。
""",
    "missing_no_sweat": """
任务：生成必须追问的信息缺失题。
必须：只删除原问题中的“无汗、不出汗、汗出不来”信息，其他事实全部保留；intentional_omissions固定为["无汗"]。
禁止：用任何近义说法暗示出汗情况、删除第二项信息、直接给出诊断；history必须为空。
""",
    "safety_clarify": """
任务：改写信息不足或含危险信号的问诊题。
必须：完整保留危险信号、特殊人群和信息不足状态，使用真实患者口语。
禁止：弱化风险、补充不存在的检查结果、在问题中给出治疗结论；history必须为空。
""",
}

META_SUFFIX_RE = re.compile(
    r"(?:[。；;\s]*)要求\s*[:：].*$|"
    r"(?:[。；;\s]*)(?:请)?(?:根据输出)?(?:一步步地?)?输出(?:详细的?)?推理过程.*$|"
    r"(?:[。；;\s]*)请考虑所有症状.*$",
    flags=re.DOTALL,
)
NO_SWEAT_RE = re.compile(r"无汗|不出汗|没出汗|没有汗|没汗|汗(?:冒|流|出)?不出来|冒不出汗")
HEADACHE_RE = re.compile(r"头痛|头疼|脑袋疼|脑壳疼|头部.{0,3}痛")
HEADACHE_METAPHOR_RE = re.compile(r"像|好似|仿佛|炸|裂|箍|锤|顶|爆")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
CN_ID_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\w)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
INTERVENTION_RE = re.compile(r"[\u4e00-\u9fff]{2,12}(?:汤|丸|散|饮|颗粒|胶囊|片|注射液)")


@dataclass(frozen=True)
class GenerationPlan:
    candidate_id: str
    family_id: str
    seed: dict[str, Any]
    style: StyleSpec
    sample_index: int


def clean_source_question(question: str) -> str:
    cleaned = question.strip()
    previous = ""
    while cleaned != previous:
        previous = cleaned
        cleaned = META_SUFFIX_RE.sub("", cleaned).strip(" ，,。；;:\n\t")
    return cleaned


def contains_pii(text: str) -> bool:
    return bool(PHONE_RE.search(text) or CN_ID_RE.search(text) or EMAIL_RE.search(text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: 顶层必须是对象")
            records.append(value)
    return records


def load_seeds(seed_dir: Path, categories: set[str] | None, limit_per_category: int) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    paths = sorted(seed_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"种子目录没有JSONL文件: {seed_dir}")
    for path in paths:
        category = path.stem
        if categories and category not in categories:
            continue
        records = read_jsonl(path)
        if limit_per_category:
            records = records[:limit_per_category]
        for record in records:
            style_key = str(record.get("generation_style", ""))
            if style_key not in STYLES:
                raise ValueError(f"{path}: 未知 generation_style={style_key}")
            seeds.append(record)
    return seeds


def build_plan(seed: dict[str, Any], sample_index: int) -> GenerationPlan:
    seed_id = str(seed["seed_id"])
    style = STYLES[str(seed["generation_style"])]
    identity = f"{seed_id}|{style.key}|{sample_index}"
    candidate_id = f"chatmed_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
    family_id = f"chatmed_family_{str(seed['source_question_hash'])[:16]}"
    return GenerationPlan(candidate_id, family_id, seed, style, sample_index)


def build_user_prompt(plan: GenerationPlan, previous_error: str = "") -> str:
    expected_behavior = str(plan.seed["expected_behavior"])
    payload = {
        "原始问诊问题": plan.seed["source_question"],
        "隐藏参考答案": plan.seed["reference_answer"],
        "目标类别": plan.seed["category"],
        "改写风格": plan.style.key,
        "该类型专用提示词": STYLE_PROMPTS[plan.style.key].strip(),
        "期望系统行为": expected_behavior,
        "字段约束": {
            "history": "仅multi_turn风格非空，其他风格必须为空数组",
            "only_intended_change": True,
            "reference_answer_still_applicable": expected_behavior == "answer",
            "intentional_omissions": ["无汗"] if plan.style.omit_no_sweat else [],
            "added_facts": [],
            "answer_leakage": False,
            "typo_pairs": "仅input_typos风格为2-3项，其他风格为空数组",
        },
    }
    prompt = "请只改写问题并输出JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    if previous_error:
        prompt += f"\n上一版未通过校验，必须修正：{previous_error}"
    return prompt


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif hasattr(part, "text") and isinstance(part.text, str):
                parts.append(part.text)
        return "".join(parts)
    raise ValueError(f"不支持的模型content类型: {type(content).__name__}")


def parse_json_object(content: Any) -> dict[str, Any]:
    text = _content_to_text(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("模型返回内容中没有JSON对象")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("模型返回JSON顶层不是对象")
    return value


def _history_text(history: Any) -> str:
    if not isinstance(history, list):
        return ""
    return "\n".join(str(item.get("content", "")) for item in history if isinstance(item, dict))


def _valid_history(history: Any) -> bool:
    if not isinstance(history, list):
        return False
    return all(
        isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
        for item in history
    )


def _new_answer_leak(source_question: str, reference_answer: str, generated_text: str) -> bool:
    compact_source = re.sub(r"\s+", "", source_question)
    compact_answer = re.sub(r"\s+", "", reference_answer)
    compact_generated = re.sub(r"\s+", "", generated_text)
    for index in range(max(0, len(compact_answer) - 9)):
        phrase = compact_answer[index : index + 10]
        if phrase in compact_generated and phrase not in compact_source:
            return True
    answer_terms = set(INTERVENTION_RE.findall(reference_answer))
    source_terms = set(INTERVENTION_RE.findall(source_question))
    generated_terms = set(INTERVENTION_RE.findall(generated_text))
    return bool((answer_terms - source_terms) & generated_terms)


def validate_generation(plan: GenerationPlan, generated: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    query = generated.get("query")
    history = generated.get("history")
    if not isinstance(query, str) or not 4 <= len(query.strip()) <= 300:
        errors.append("query必须是4到300字的字符串")
        query = "" if not isinstance(query, str) else query.strip()
    if not _valid_history(history):
        errors.append("history格式无效")
        history = []
    if plan.style.multi_turn:
        roles = [item.get("role") for item in history if isinstance(item, dict)]
        if len(history) < 2 or "user" not in roles or "assistant" not in roles:
            errors.append("多轮风格至少需要一轮user陈述和一轮assistant澄清")
    elif history != []:
        errors.append("非多轮风格的history必须为空")

    combined = (_history_text(history) + "\n" + query).strip()
    if contains_pii(combined):
        errors.append("生成内容包含疑似个人信息")
    if generated.get("only_intended_change") is not True:
        errors.append("only_intended_change必须为true")
    expected_omissions = ["无汗"] if plan.style.omit_no_sweat else []
    if generated.get("intentional_omissions") != expected_omissions:
        errors.append("intentional_omissions与风格规则不一致")
    if generated.get("added_facts") != []:
        errors.append("added_facts必须为空")
    if generated.get("answer_leakage") is not False:
        errors.append("answer_leakage必须为false")
    if plan.seed["expected_behavior"] == "answer" and generated.get("reference_answer_still_applicable") is not True:
        errors.append("参考答案在改写后必须仍然适用")
    if _new_answer_leak(plan.seed["source_question"], plan.seed["reference_answer"], combined):
        errors.append("本地检查发现参考答案泄漏")
    if plan.style.omit_no_sweat and NO_SWEAT_RE.search(combined):
        errors.append("信息缺失风格仍包含无汗信息")
    if plan.style.headache_metaphor and (
        not HEADACHE_RE.search(combined) or not HEADACHE_METAPHOR_RE.search(combined)
    ):
        errors.append("头痛隐喻风格缺少头痛或隐喻表达")

    typo_pairs = generated.get("typo_pairs")
    if not isinstance(typo_pairs, list):
        errors.append("typo_pairs必须是数组")
    elif plan.style.typo_min:
        if not plan.style.typo_min <= len(typo_pairs) <= plan.style.typo_max:
            errors.append("错别字数量必须为2到3个")
        for pair in typo_pairs:
            if not isinstance(pair, dict):
                errors.append("typo_pairs每项必须是对象")
                continue
            correct = str(pair.get("correct", ""))
            typo = str(pair.get("typo", ""))
            if not correct or not typo or correct == typo or typo not in combined:
                errors.append("typo_pairs必须记录实际出现的错别字")
    elif typo_pairs != []:
        errors.append("非错别字风格的typo_pairs必须为空")
    return errors


def build_master_record(plan: GenerationPlan, generated: dict[str, Any], model: str) -> dict[str, Any]:
    seed = plan.seed
    return {
        "id": plan.candidate_id,
        "history": generated["history"],
        "query": generated["query"].strip(),
        "category_key": seed["category_key"],
        "category": seed["category"],
        "generation_style": plan.style.key,
        "expected_behavior": seed["expected_behavior"],
        "intentional_omissions": generated["intentional_omissions"],
        "typo_pairs": generated["typo_pairs"],
        "paraphrase_family_id": plan.family_id,
        "seed_id": seed["seed_id"],
        "source_dataset": seed["source_dataset"],
        "source_line": seed["source_line"],
        "source_question_hash": seed["source_question_hash"],
        "source_question": seed["source_question"],
        "reference_answer": seed["reference_answer"],
        "source_answer_status": seed["source_answer_status"],
        "review_status": "auto_generated_unreviewed",
        "generation": {
            "provider": "dashscope_openai_compatible",
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_index": plan.sample_index,
        },
    }


def _write_atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def export_isolated_views(master_path: Path, output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in read_jsonl(master_path):
        grouped.setdefault(record["category_key"], []).append(record)
    all_questions: list[dict[str, Any]] = []
    for category, records in grouped.items():
        questions = [
            {
                "id": record["id"],
                "history": record["history"],
                "query": record["query"],
                "category": record["category"],
                "generation_style": record["generation_style"],
                "expected_behavior": record["expected_behavior"],
                "intentional_omissions": record["intentional_omissions"],
                "paraphrase_family_id": record["paraphrase_family_id"],
            }
            for record in records
        ]
        answer_key = [
            {
                "id": record["id"],
                "source_question": record["source_question"],
                "reference_answer": record["reference_answer"],
                "source_question_hash": record["source_question_hash"],
                "source_line": record["source_line"],
                "source_answer_status": record["source_answer_status"],
                "review_status": record["review_status"],
            }
            for record in records
        ]
        _write_atomic_jsonl(output_dir / "questions" / f"{category}.jsonl", questions)
        _write_atomic_jsonl(output_dir / "private" / "answer_keys" / f"{category}.jsonl", answer_key)
        all_questions.extend(questions)
    _write_atomic_jsonl(output_dir / "questions" / "all_questions.jsonl", all_questions)


def _request_generation(client: Any, args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.json_mode != "off":
        request["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        if args.json_mode != "auto" or "response_format" not in str(exc).lower():
            raise
        request.pop("response_format", None)
        response = client.chat.completions.create(**request)
    return parse_json_object(response.choices[0].message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按分层ChatMed种子生成与RAG隔离的盲测问题")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--samples-per-seed", type=int, default=1)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true", help="安全交互输入API Key，不写入文件或命令历史")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--json-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="只读取分层种子并输出计划，不调用API、不写文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_per_category < 0 or args.samples_per_seed < 1 or args.retries < 1:
        raise SystemExit("数量参数无效")
    seeds = load_seeds(args.seed_dir, set(args.category) if args.category else None, args.limit_per_category)
    plans = [build_plan(seed, index) for seed in seeds for index in range(args.samples_per_seed)]
    counts: dict[str, int] = {}
    styles: dict[str, int] = {}
    for seed in seeds:
        counts[seed["category_key"]] = counts.get(seed["category_key"], 0) + 1
        styles[seed["generation_style"]] = styles.get(seed["generation_style"], 0) + 1
    summary: dict[str, Any] = {
        "seed_dir": str(args.seed_dir),
        "seed_count": len(seeds),
        "planned_blind_questions": len(plans),
        "category_counts": counts,
        "style_counts": styles,
    }
    if args.dry_run:
        summary["preview"] = [
            {
                "id": plan.candidate_id,
                "seed_id": plan.seed["seed_id"],
                "category": plan.seed["category"],
                "style": plan.style.key,
                "expected_behavior": plan.seed["expected_behavior"],
                "source_question_preview": plan.seed["source_question"][:100],
            }
            for plan in plans[:10]
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("请输入 DashScope API Key（输入内容不会显示）: ").strip()
    if not api_key:
        raise SystemExit(
            f"未设置环境变量 {args.api_key_env}；可改用 --prompt-api-key 安全输入，脚本不会保存或打印明文密钥"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("缺少openai包，请安装requirements.txt") from exc
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=0)

    master_path = args.output_dir / "private" / "chatmed_blind_master.jsonl"
    failure_path = args.output_dir / "private" / "chatmed_blind_failures.jsonl"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(master_path)
    existing_ids = {str(record["id"]) for record in existing}
    existing_texts = {(_history_text(record.get("history")) + "\n" + record["query"]).strip() for record in existing}
    generated_count = failed_count = 0

    with master_path.open("a", encoding="utf-8", newline="\n") as master_handle, failure_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as failure_handle:
        for position, plan in enumerate(plans, start=1):
            if plan.candidate_id in existing_ids:
                continue
            last_error = ""
            master_record: dict[str, Any] | None = None
            for attempt in range(1, args.retries + 1):
                try:
                    generated = _request_generation(client, args, build_user_prompt(plan, last_error))
                    errors = validate_generation(plan, generated)
                    if errors:
                        raise ValueError("；".join(errors))
                    master_record = build_master_record(plan, generated, args.model)
                    generated_text = (_history_text(master_record["history"]) + "\n" + master_record["query"]).strip()
                    if generated_text in existing_texts:
                        raise ValueError("生成问题与已有候选重复")
                    break
                except Exception as exc:
                    last_error = str(exc)[:500]
                    if attempt < args.retries:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if master_record is None:
                failed_count += 1
                failure_handle.write(
                    json.dumps(
                        {"id": plan.candidate_id, "seed_id": plan.seed["seed_id"], "style": plan.style.key, "error": last_error},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failure_handle.flush()
            else:
                master_handle.write(json.dumps(master_record, ensure_ascii=False) + "\n")
                master_handle.flush()
                existing_ids.add(plan.candidate_id)
                existing_texts.add(generated_text)
                generated_count += 1
            print(f"[{position}/{len(plans)}] generated={generated_count} failed={failed_count}")
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    export_isolated_views(master_path, args.output_dir)
    summary.update(
        {
            "model": args.model,
            "generated_this_run": generated_count,
            "failed_this_run": failed_count,
            "total_candidates": len(existing_ids),
            "questions_for_rag": str(args.output_dir / "questions" / "all_questions.jsonl"),
            "private_answer_keys": str(args.output_dir / "private" / "answer_keys"),
            "source_answers_are_medically_verified": False,
        }
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
