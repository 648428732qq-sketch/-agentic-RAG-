from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from db.qdrant_client_factory import create_qdrant_client  # noqa: E402
from core.syndrome_terms import (  # noqa: E402
    SYMPTOM_ALIASES,
    SYMPTOM_STOPWORDS,
    SYMPTOM_TERMS,
    clean_text,
    unique,
)


LABELS = ("组成", "用法", "功用", "主治", "方解", "运用", "原文解析")
LABEL_ALIASES = {
    "组成": "组成",
    "方剂组成": "组成",
    "组方": "组成",
    "用法": "用法",
    "功用": "功用",
    "主治": "主治",
    "方解": "方解",
    "运用": "运用",
    "原文解析": "原文解析",
}
DEFAULT_CLASSIC_SOURCES = [
    ROOT / "markdown_docs" / "古籍_伤寒论.md",
    ROOT / "markdown_docs" / "古籍_金匮要略.md",
    ROOT / "markdown_docs" / "古籍_温病条辨.md",
]
DEFAULT_ACUPUNCTURE_SOURCES = [
    ROOT / "markdown_docs" / "古籍_黄帝内经_灵枢.md",
    ROOT / "markdown_docs" / "古籍_黄帝内经_素问.md",
    ROOT / "markdown_docs" / "古籍_难经.md",
]
DEFAULT_HERB_SOURCES = [
    ROOT / "markdown_docs" / "古籍_神农本草经.md",
]
DEFAULT_THEORY_SOURCES = [
    ROOT / "markdown_docs" / "古籍_难经.md",
]
DEFAULT_REVIEWED_REPLACEMENTS = ROOT / "datasets" / "structured" / "syndrome_dictionary_reviewed_replacements.jsonl"
POSITIVE_FORMULA_MARKERS = ("主之", "宜", "可与")
NEGATIVE_FORMULA_MARKERS = ("不可与", "不可服", "禁用")
ACUPUNCTURE_MARKERS = ("取之", "刺之", "灸之", "当刺", "可刺", "补之", "泻之", "深刺", "浅刺")
CLINICAL_HINT_TERMS = (
    "病", "痛", "痹", "热", "寒", "胀", "厥", "痿", "咳", "喘", "疟", "泄", "利",
    "小便", "大便", "头", "胸", "腹", "腰", "身", "汗", "呕", "口苦", "烦", "渴",
)
ACUPUNCTURE_CLINICAL_TERMS = (
    "病者", "病也", "病在", "胃病", "胆病", "痛", "痹", "厥", "痿", "胀",
    "寒热", "疟", "热厥", "寒厥", "头痛", "腹痛", "腰痛", "心痛", "小便", "大便",
)
ACUPUNCTURE_EXCLUDE_MARKERS = (
    "黄帝曰", "问曰", "奈何", "凡刺", "刺之法", "刺之道", "持针", "针道",
    "小针", "九针", "方刺之时", "用针者", "针太深", "不欲深刺", "所取之处",
    "可得同之乎", "十二禁", "针害", "气至", "气不至",
)
ACUPOINT_OR_CHANNEL_TERMS = [
    "三里", "下陵三里", "阴陵泉", "阳陵泉", "涌泉", "大敦", "隐白", "廉泉",
    "阴𫏋", "阳𫏋", "阴跷", "阳跷", "踝后", "所别", "皮肤之血", "分腠", "井荥分输",
    "取之井", "取之于合", "三阴", "阴分", "井穴", "合穴",
    "足太阳", "足阳明", "足少阳", "足太阴", "足少阴", "足厥阴",
    "手太阳", "手阳明", "手少阳", "手太阴", "手少阴", "手厥阴",
    "太阳", "阳明", "少阳", "太阴", "少阴", "厥阴",
]
ACUPUNCTURE_PRINCIPLE_TERMS = [
    "补之", "泻之", "虚则补之", "实则泻之", "浅刺", "深刺", "浅取", "深取",
    "迎随", "荣卫", "刺荣", "刺卫", "气至", "气不至", "得气", "留针",
    "九针", "针道", "刺之道", "刺法", "持针", "正指直刺", "卧针",
    "无伤荣", "无伤卫", "先补", "后泻", "疾徐", "开阖",
]
ACUPUNCTURE_PRINCIPLE_CONTEXT_TERMS = (
    "针", "刺", "取", "补", "泻", "荣卫", "经脉", "气至", "得气", "留针", "持针", "九针"
)
NON_ACUPUNCTURE_YINGSUI_CONTEXT = (
    "五运", "六气", "寒暑", "天期", "五气", "太过", "不及", "岁气", "司天", "在泉"
)
HERB_PUNCTUATION_LINES = {"，", "。", "；", "、", "：", ":", ",", ".", ";", "?"}
HERB_STOP_NAMES = {
    "上经", "中经", "下经", "卷一", "卷二", "卷三", "序录", "旧同",
    "玉石", "草", "木", "人", "兽", "禽", "虫鱼", "果", "米谷", "菜",
    "上品", "中品", "下品",
}
HERB_NOTE_STARTERS = (
    "《", "案", "按", "吴普", "名医", "雷公", "别录", "唐本", "陶隐居",
    "说文", "山海经", "范子", "御览", "大观", "本草",
)
HERB_INDICATION_STOPS = ("久服", "一名", "生", "能化", "畏", "恶")
HERB_TERM_STOPWORDS = {
    "轻身", "延年", "不老", "神仙", "生山谷", "一名", "采无时", "多有",
    "俱作", "宜改", "后人", "经文", "黑字", "白字",
}
HERB_GRADE_PREFIX = {"上": "上品", "中": "中品", "下": "下品"}
HERB_NAME_VARIANTS = {
    "丹沙": "丹砂",
    "太乙余食": "太乙余粮",
    "班苗": "斑蝥",
    "荧火": "萤火",
}
THEORY_TERMS = [
    "寸口", "脉", "脉诊", "尺寸", "寸关尺", "浮", "沉", "迟", "数", "太过", "不及",
    "阴阳", "五藏", "六府", "荣卫", "经络", "十二经", "奇经八脉", "任脉", "督脉",
    "冲脉", "带脉", "阴维", "阳维", "阴跷", "阳跷", "三焦", "命门", "肾间动气",
    "八会", "井荥俞经合", "虚实", "补泻", "寒热", "死生", "吉凶", "关格", "覆溢",
]

PATHOGENESIS_TERMS = [
    "外感风寒",
    "外感风寒表实",
    "外感风寒表虚",
    "外感风寒湿邪",
    "风寒束表",
    "寒邪束表",
    "肺气失宣",
    "营卫不和",
    "卫阳被遏",
    "腠理闭塞",
    "寒饮内停",
    "外寒里饮",
    "水饮内停",
    "气郁不舒",
    "气机郁滞",
    "内有蕴热",
    "风邪犯肺",
    "肺失宣降",
    "表邪未尽",
]

TERM_GROUP_RULES = [
    ("无汗", ["无汗"]),
    ("汗出", ["汗出", "有汗", "自汗", "汗大出", "多汗"]),
    ("恶寒", ["恶寒"]),
    ("恶风", ["恶风"]),
    ("发热", ["发热", "身热", "大热", "壮热"]),
    ("头痛", ["头痛"]),
    ("身疼", ["身疼", "头身疼痛", "肢体酸楚疼痛", "骨节疼痛", "身体疼重"]),
    ("喘", ["喘", "喘咳", "咳喘"]),
    ("咳嗽", ["咳嗽", "喘咳", "咳喘"]),
    ("痰多", ["痰多", "痰涎清稀"]),
    ("不得平卧", ["不得平卧"]),
    ("口渴", ["口渴", "口大渴", "烦渴引饮", "口干"]),
    ("不渴", ["不渴"]),
    ("呕吐", ["呕吐", "干呕", "气逆欲呕"]),
    ("下利", ["下利", "泄泻", "腹泻", "大便稀溏"]),
    ("厥", ["厥", "四肢厥冷", "畏寒肢冷"]),
    ("食少", ["食少", "食少便溏", "脘痞食少"]),
    ("气短乏力", ["气短乏力", "乏力"]),
]
SWEAT_PRESENT_TERMS = {"汗出", "有汗", "自汗", "汗大出", "多汗"}
SWEAT_ABSENT_TERMS = {"无汗"}
THIRST_PRESENT_TERMS = {"口渴", "口大渴", "烦渴引饮"}
THIRST_ABSENT_TERMS = {"不渴"}

GENERIC_DIAGNOSTIC_TERMS = {
    "脉", "浮", "沉", "迟", "数", "苔白", "舌苔白", "舌淡", "舌红",
    "舌苔薄白", "舌苔白滑", "脉浮", "脉沉",
}
FORMULA_DIFFERENTIAL_RULES = {
    "麻黄汤": {
        "required": [["恶寒"], ["无汗"], ["喘"]],
        "forbidden": ["汗出", "有汗", "自汗"],
        "differential": ["无汗", "汗出", "喘", "表实"],
        "clarify": ["是否出汗", "是否喘咳"],
    },
    "桂枝汤": {
        "required": [["恶风"], ["汗出"]],
        "forbidden": ["无汗"],
        "differential": ["汗出", "无汗", "恶风", "营卫不和"],
        "clarify": ["是否出汗", "怕风还是怕冷"],
    },
    "小青龙汤": {
        "required": [["恶寒"], ["无汗"], ["喘", "喘咳", "咳喘"], ["痰多", "痰涎清稀"]],
        "forbidden": ["汗出", "有汗", "痰热", "黄痰"],
        "differential": ["外寒里饮", "痰涎清稀", "不得平卧"],
        "clarify": ["是否出汗", "痰是清稀还是黄稠", "能否平卧"],
    },
    "苏子降气汤": {
        "required": [["喘", "喘咳", "咳喘"], ["痰多", "痰涎壅盛"], ["胸膈满闷"], ["呼多吸少", "短气"]],
        "forbidden": ["痰涎清稀", "不得平卧"],
        "differential": ["上实下虚", "痰涎壅盛", "胸膈满闷", "呼多吸少", "腰疼脚弱"],
        "clarify": ["是否胸膈满闷", "是否呼多吸少", "是否腰膝酸软"],
    },
    "苓甘五味姜辛汤": {
        "required": [["咳嗽", "喘咳", "咳喘"], ["痰多", "清稀色白"], ["胸满", "喜唾"]],
        "forbidden": ["不得平卧", "黄痰", "痰热"],
        "differential": ["寒饮咳嗽", "清稀色白", "胸满", "喜唾"],
        "clarify": ["痰是清稀还是黄稠", "是否胸满喜唾"],
    },
    "白虎汤": {
        "required": [["发热", "大热", "壮热"], ["汗出", "汗大出"], ["口渴", "口大渴"]],
        "forbidden": ["无汗", "不渴"],
        "differential": ["大热", "汗大出", "口大渴", "气分热盛"],
        "clarify": ["是否口渴", "是否出汗"],
    },
    "竹叶石膏汤": {
        "required": [["身热", "发热"], ["口渴", "口干"], ["气逆欲呕"]],
        "forbidden": ["无汗", "不渴"],
        "differential": ["余热未清", "气津两伤", "气逆欲呕"],
        "clarify": ["是否口干口渴", "是否恶心欲呕"],
    },
    "理中丸": {
        "required": [["呕吐"], ["下利", "大便稀溏"], ["不渴"]],
        "forbidden": ["口渴", "湿热", "阴虚"],
        "differential": ["脾胃虚寒", "不渴", "喜温喜按"],
        "clarify": ["是否口渴", "大便是否稀溏", "腹痛是否喜温喜按"],
    },
    "四逆汤": {
        "required": [["厥", "四肢厥冷"], ["恶寒"], ["下利"]],
        "forbidden": ["口渴", "汗大出", "真热假寒"],
        "differential": ["四肢厥逆", "心肾阳衰", "阴寒内盛"],
        "clarify": ["四肢是否厥冷", "是否腹痛下利"],
    },
    "四君子汤": {
        "required": [["气短乏力"], ["食少"]],
        "forbidden": ["痰多", "恶寒", "发热"],
        "differential": ["脾胃气虚", "食少便溏", "气短乏力"],
        "clarify": ["是否食少便溏", "是否气短乏力"],
    },
    "六君子汤": {
        "required": [["气短乏力"], ["食少"], ["痰多"]],
        "forbidden": ["无痰", "恶寒", "发热"],
        "differential": ["脾胃气虚", "痰湿", "食少便溏"],
        "clarify": ["是否痰多", "是否食少便溏"],
    },
}


class FormulaHerb(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    original_dose: str = ""


METRIC_DOSE_PATTERN = re.compile(
    r"[（(]\s*\d+(?:\.\d+)?(?:\s*[-~—–至]\s*\d+(?:\.\d+)?)?\s*(?:g|G|克|kg|KG|千克|ml|mL|ML|毫升)\s*[）)]"
)


def sanitize_formula_dose(dose: str) -> str:
    """Keep source-era dose text and remove modern metric conversions."""
    cleaned = METRIC_DOSE_PATTERN.sub("", dose)
    cleaned = re.sub(r"[，,；;、]\s*$", "", cleaned)
    return clean_text(cleaned)


class SyndromeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entry_id: str
    title: str
    source_type: str = "formula_syndrome"
    source_book: str = "方剂大全"
    source_file: str = "方剂大全.md"
    source_url: str = ""
    chapter: str = "方剂大全"
    syndrome_name: str = ""
    ancient_symptoms: list[str] = Field(default_factory=list)
    modern_symptoms: list[str] = Field(default_factory=list)
    symptom_aliases: list[str] = Field(default_factory=list)
    diagnostic_keys: list[str] = Field(default_factory=list)
    pathogenesis: list[str] = Field(default_factory=list)
    required_symptom_groups: list[list[str]] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    differential_keys: list[str] = Field(default_factory=list)
    must_clarify_fields: list[str] = Field(default_factory=list)
    intervention_type: str = "formula"
    intervention_name: str = ""
    treatment_method: str = ""
    acupoints_or_channels: list[str] = Field(default_factory=list)
    treatment_principle: str = ""
    formula: str
    formula_composition: list[FormulaHerb] = Field(default_factory=list)
    herb_name: str = ""
    herb_grade: str = ""
    herb_category: str = ""
    herb_aliases: list[str] = Field(default_factory=list)
    nature_flavor: list[str] = Field(default_factory=list)
    origin_habitat: str = ""
    property_text: str = ""
    theory_topic: str = ""
    theory_question: str = ""
    theory_answer: str = ""
    theory_terms: list[str] = Field(default_factory=list)
    diagnostic_method: str = ""
    acupuncture_principle: str = ""
    acupuncture_terms: list[str] = Field(default_factory=list)
    usage_original: str = ""
    functions: str = ""
    indications: str = ""
    formula_analysis: str = ""
    modifications: str = ""
    modern_applications: str = ""
    contraindications: str = ""
    evidence: str = ""
    review_status: str = "rule_extracted"
    confidence: float = 0.65
    search_text: str = ""
    raw_text: str = ""
    payload_version: str = "syndrome_entry_v1"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, re.MULTILINE))
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        title = clean_text(match.group(1))
        body = markdown[start:end].strip()
        if title and body:
            sections.append((title, body))
    return sections


def parse_labeled_fields(title: str, body: str) -> tuple[str, dict[str, list[str]]]:
    source_match = re.search(r"^>\s*来源[：:]\s*(\S+)", body, re.MULTILINE)
    source_url = source_match.group(1) if source_match else ""

    fields: dict[str, list[str]] = {label: [] for label in LABELS}
    current: str | None = None
    for raw_line in body.splitlines():
        line = clean_text(raw_line)
        if not line or line.startswith(">"):
            continue
        if line in LABEL_ALIASES:
            current = LABEL_ALIASES[line]
            continue
        if current is None:
            continue
        if line == title:
            continue
        fields[current].append(line)
    return source_url, fields


def field_text(fields: dict[str, list[str]], label: str) -> str:
    return clean_text("\n".join(fields.get(label, [])))


def parse_composition(lines: list[str]) -> list[FormulaHerb]:
    herbs: list[FormulaHerb] = []
    idx = 0
    while idx < len(lines):
        name = clean_text(lines[idx])
        dose = ""
        if idx + 1 < len(lines):
            dose = sanitize_formula_dose(lines[idx + 1])
            idx += 2
        else:
            idx += 1
        if name:
            herbs.append(FormulaHerb(name=name, original_dose=dose))
    return herbs


def extract_numbered(text: str, label: str) -> str:
    pattern = rf"{label}：(.+?)(?=\n\d+、|$)"
    match = re.search(pattern, text, re.DOTALL)
    return clean_text(match.group(1)) if match else ""


def split_symptom_phrase(text: str) -> list[str]:
    candidates: list[str] = []
    for term in SYMPTOM_TERMS:
        if term in text:
            candidates.append(term)

    chunks = re.split(r"[，、；;。.\n]|或|及|并|而|兼见|伴", text)
    for chunk in chunks:
        chunk = clean_text(chunk)
        chunk = re.sub(r"^(黄帝曰|岐伯曰|问曰|曰)[:：]", "", chunk)
        chunk = re.sub(r"(者|也)$", "", chunk)
        if re.fullmatch(r"(取|刺|灸|补|泻|浅刺|深刺|当刺|可刺).{0,8}", chunk):
            continue
        if any(marker in chunk for marker in ("取之", "刺之", "灸之", "补之", "泻之")):
            continue
        if any(marker in chunk for marker in ("可令", "立快", "主之", "宜之")):
            continue
        if any(skip in chunk for skip in ("方见", "见上", "见中焦篇")):
            continue
        if re.fullmatch(r"(下之则愈|下之则和|当发其汗|缓中补虚|五日可治|七日不可治|乃治痞|当下其癥|止逆下气)", chunk):
            continue
        if re.fullmatch(r"宜.+(汤|丸|散|饮|方).{0,8}", chunk):
            continue
        if chunk.startswith(("当须", "见")):
            continue
        if any(skip in chunk for skip in ("本方", "临床应用", "剂量", "方宜", "代表方", "治疗", "使用", "现代", "属于")):
            continue
        if any(skip in chunk for skip in ("主之", "可与", "方用前法", "如前法", "不可", "此误也", "勿", "禁")):
            continue
        if re.search(r"与.+(汤|丸|散|饮|方)", chunk):
            continue
        if re.search(r"(汤|丸|散|饮|方)$", chunk):
            continue
        if any(term in chunk and term != chunk for term in SYMPTOM_TERMS):
            continue
        if chunk in SYMPTOM_STOPWORDS:
            continue
        if 1 < len(chunk) <= 12 and not any(skip in chunk for skip in ("证", "主治", "用于")):
            candidates.append(chunk)
    return unique(candidates)


def extract_syndrome_name(indications: str) -> str:
    first = clean_text(re.split(r"[。.\n]", indications, maxsplit=1)[0])
    return first[:40]


def extract_diagnostic_text(operation: str) -> str:
    numbered = extract_numbered(operation, "辨证要点")
    if not numbered:
        return ""
    match = re.search(r"以(.{2,80}?)(?:为辨证要点|为主|为要)", numbered)
    if match:
        return clean_text(match.group(1))
    match = re.search(r"临床应用以(.{2,80}?)(?:为辨证要点|。|$)", numbered)
    if match:
        return clean_text(match.group(1))
    return clean_text(re.split(r"[。.\n]", numbered, maxsplit=1)[0])


def extract_symptoms(indications: str, diagnostic_keys: str) -> list[str]:
    symptom_texts: list[str] = []
    parts = re.split(r"[。.\n]", indications, maxsplit=1)
    if len(parts) > 1:
        symptom_texts.append(parts[1])
    else:
        symptom_texts.append(indications)
    symptom_texts.append(diagnostic_keys)
    return unique([sym for text in symptom_texts for sym in split_symptom_phrase(text)])


def modernize_symptoms(ancient_symptoms: list[str]) -> tuple[list[str], list[str]]:
    modern: list[str] = []
    aliases: list[str] = []
    for symptom in ancient_symptoms:
        mapped = SYMPTOM_ALIASES.get(symptom)
        if mapped:
            modern.extend(mapped[:2])
            aliases.extend(mapped)
            continue

        partial_aliases: list[str] = []
        for term, term_aliases in SYMPTOM_ALIASES.items():
            if len(term) < 2 or term not in symptom:
                continue
            partial_aliases.extend(term_aliases)
        if partial_aliases:
            modern.extend(partial_aliases[:3])
            aliases.extend(partial_aliases)
        else:
            modern.append(symptom)
    return unique(modern), unique(aliases)


def extract_pathogenesis(indications: str, analysis: str) -> list[str]:
    values: list[str] = []
    syndrome = extract_syndrome_name(indications)
    if syndrome:
        values.append(syndrome)
    for term in PATHOGENESIS_TERMS:
        if term in indications or term in analysis:
            values.append(term)

    first_sentence = clean_text(re.split(r"[。.\n]", analysis, maxsplit=1)[0]) if analysis else ""
    match = re.search(r"本方(?:证|主治|治证)?(?:为|由|治)?(.{2,80}?)(?:所致|证)", first_sentence)
    if match:
        values.append(match.group(1))
    return unique(values)


def _entry_text_for_rules(entry: SyndromeEntry) -> str:
    values: list[str] = [
        entry.formula,
        entry.syndrome_name,
        entry.indications,
        entry.functions,
        entry.formula_analysis,
        entry.contraindications,
        entry.evidence,
    ]
    values.extend(entry.ancient_symptoms)
    values.extend(entry.modern_symptoms)
    values.extend(entry.diagnostic_keys)
    values.extend(entry.pathogenesis)
    return clean_text(" ".join(str(value) for value in values if value))


def _entry_clinical_text_for_forbidden(entry: SyndromeEntry) -> str:
    values: list[str] = [
        entry.formula,
        entry.syndrome_name,
        entry.indications,
        entry.functions,
    ]
    values.extend(entry.ancient_symptoms)
    values.extend(entry.modern_symptoms)
    values.extend(entry.diagnostic_keys)
    values.extend(entry.pathogenesis)
    return clean_text(" ".join(str(value) for value in values if value))


def _groups_from_terms(terms: list[str], text: str, max_groups: int = 5) -> list[list[str]]:
    groups: list[list[str]] = []
    seen_keys: set[str] = set()
    term_set = set(terms)
    for key, group in TERM_GROUP_RULES:
        if key in seen_keys:
            continue
        if term_set.intersection(group) or any(value in text for value in group):
            groups.append(group)
            seen_keys.add(key)
        if len(groups) >= max_groups:
            break
    return groups


def _derive_forbidden_terms(entry: SyndromeEntry, text: str) -> list[str]:
    forbidden: list[str] = []
    if any(term in text for term in ("无汗", "汗不出", "不出汗", "表实")):
        forbidden.extend(["汗出", "有汗", "自汗"])
    if any(term in text for term in ("汗出", "自汗", "有汗", "表虚")):
        forbidden.append("无汗")
    if "不渴" in text:
        forbidden.extend(["口渴", "口大渴"])
    if any(term in text for term in ("口渴", "口大渴", "烦渴引饮")):
        forbidden.append("不渴")

    contraindications = entry.contraindications
    if "无汗" in contraindications:
        forbidden.append("无汗")
    if any(term in contraindications for term in ("汗出", "自汗")):
        forbidden.extend(["汗出", "有汗", "自汗"])
    if "阴虚" in contraindications:
        forbidden.append("阴虚")
    if "湿热" in contraindications:
        forbidden.append("湿热")
    return unique([term for term in forbidden if term])


def derive_differential_fields(entry: SyndromeEntry) -> None:
    text = _entry_text_for_rules(entry)
    clinical_text = _entry_clinical_text_for_forbidden(entry)
    rule = FORMULA_DIFFERENTIAL_RULES.get(entry.formula, {})
    base_terms = [
        term
        for term in unique(entry.diagnostic_keys + entry.ancient_symptoms + entry.pathogenesis)
        if term not in GENERIC_DIAGNOSTIC_TERMS
    ]

    if not entry.required_symptom_groups:
        groups = [list(group) for group in rule.get("required", [])]
        if not groups:
            groups = _groups_from_terms(base_terms, text)
        entry.required_symptom_groups = [unique(group) for group in groups if group]

    if not entry.forbidden_terms:
        forbidden_terms = unique(list(rule.get("forbidden", [])) + _derive_forbidden_terms(entry, clinical_text))
        required_terms = {term for group in entry.required_symptom_groups for term in group}
        entry.forbidden_terms = unique([term for term in forbidden_terms if term not in required_terms])

    if not entry.differential_keys:
        keys = list(rule.get("differential", []))
        keys.extend(term for term in base_terms if len(term) > 1 and term not in keys)
        entry.differential_keys = unique(keys[:12])

    if not entry.must_clarify_fields:
        clarify = list(rule.get("clarify", []))
        group_text = " ".join(" ".join(group) for group in entry.required_symptom_groups)
        if "无汗" in group_text or "汗出" in group_text or entry.forbidden_terms and any(
            term in {"无汗", "汗出", "有汗", "自汗"} for term in entry.forbidden_terms
        ):
            clarify.append("是否出汗")
        if "喘" in group_text or "喘咳" in group_text:
            clarify.append("是否喘咳")
        if "口渴" in group_text or "不渴" in group_text:
            clarify.append("是否口渴")
        entry.must_clarify_fields = unique(clarify)


def normalize_required_symptom_groups(entry: SyndromeEntry) -> None:
    flat_required = {term for group in entry.required_symptom_groups for term in group}
    evidence_terms = set(entry.ancient_symptoms + entry.modern_symptoms + entry.diagnostic_keys)
    has_no_sweat = bool(flat_required & SWEAT_ABSENT_TERMS)
    has_sweat = bool(flat_required & SWEAT_PRESENT_TERMS)
    has_no_thirst = bool((flat_required | evidence_terms) & THIRST_ABSENT_TERMS)

    normalized_groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in entry.required_symptom_groups:
        cleaned = [term for term in group if term]
        if has_no_sweat:
            cleaned = [term for term in cleaned if term not in SWEAT_PRESENT_TERMS]
        if has_sweat:
            cleaned = [term for term in cleaned if term not in SWEAT_ABSENT_TERMS]
        if has_no_thirst:
            cleaned = [term for term in cleaned if term not in THIRST_PRESENT_TERMS]
        cleaned = unique(cleaned)
        key = tuple(cleaned)
        if cleaned and key not in seen:
            normalized_groups.append(cleaned)
            seen.add(key)
    entry.required_symptom_groups = normalized_groups

    required_terms = {term for group in entry.required_symptom_groups for term in group}
    forbidden = list(entry.forbidden_terms)
    if has_no_sweat:
        forbidden.extend(SWEAT_PRESENT_TERMS)
    if has_sweat:
        forbidden.extend(SWEAT_ABSENT_TERMS)
    if has_no_thirst:
        forbidden.extend(THIRST_PRESENT_TERMS)
    entry.forbidden_terms = unique([term for term in forbidden if term and term not in required_terms])


def extract_treatment_principle(analysis: str, functions: str) -> str:
    match = re.search(r"治当(.{2,40}?)[。；;]", analysis)
    if match:
        return clean_text(match.group(1))
    return functions


def make_search_text(entry: SyndromeEntry) -> str:
    parts = [
        f"方剂 {entry.formula}",
        f"证候 {entry.syndrome_name}",
        "古代症状 " + " ".join(entry.ancient_symptoms),
        "口语症状 " + " ".join(entry.modern_symptoms + entry.symptom_aliases),
        "核心病因 " + " ".join(entry.pathogenesis),
        "必需症状组 " + " ".join(" ".join(group) for group in entry.required_symptom_groups),
        "排除症状 " + " ".join(entry.forbidden_terms),
        "鉴别要点 " + " ".join(entry.differential_keys),
        "需追问字段 " + " ".join(entry.must_clarify_fields),
        "干预类型 " + entry.intervention_type,
        "干预名称 " + entry.intervention_name,
        "治疗方法 " + entry.treatment_method,
        "穴位经脉 " + " ".join(entry.acupoints_or_channels),
        "本草药物 " + entry.herb_name,
        "本草品级 " + entry.herb_grade,
        "本草部类 " + entry.herb_category,
        "本草别名 " + " ".join(entry.herb_aliases),
        "性味 " + " ".join(entry.nature_flavor),
        "产地 " + entry.origin_habitat,
        "本草功效 " + entry.property_text,
        "理论主题 " + entry.theory_topic,
        "理论问句 " + entry.theory_question,
        "理论答文 " + entry.theory_answer,
        "理论术语 " + " ".join(entry.theory_terms),
        "诊法 " + entry.diagnostic_method,
        "针法原则 " + entry.acupuncture_principle,
        "针法术语 " + " ".join(entry.acupuncture_terms),
        "功用 " + entry.functions,
        "主治 " + entry.indications,
        "辨证要点 " + " ".join(entry.diagnostic_keys),
    ]
    return clean_text("\n".join(part for part in parts if part.strip()))


def build_entry(title: str, body: str, source_file: str) -> SyndromeEntry:
    source_url, fields = parse_labeled_fields(title, body)
    usage = field_text(fields, "用法")
    functions = field_text(fields, "功用")
    indications = field_text(fields, "主治")
    analysis = field_text(fields, "方解")
    operation = field_text(fields, "运用")
    diagnostic_text = extract_diagnostic_text(operation)
    ancient_symptoms = extract_symptoms(indications, diagnostic_text)
    modern_symptoms, aliases = modernize_symptoms(ancient_symptoms)
    pathogenesis = extract_pathogenesis(indications, analysis)
    entry = SyndromeEntry(
        entry_id=f"formula::{title}",
        title=f"{title}证",
        source_file=source_file,
        source_url=source_url,
        syndrome_name=extract_syndrome_name(indications),
        ancient_symptoms=ancient_symptoms,
        modern_symptoms=modern_symptoms,
        symptom_aliases=aliases,
        diagnostic_keys=split_symptom_phrase(diagnostic_text),
        pathogenesis=pathogenesis,
        treatment_principle=extract_treatment_principle(analysis, functions),
        formula=title,
        formula_composition=parse_composition(fields.get("组成", [])),
        usage_original=usage,
        functions=functions,
        indications=indications,
        formula_analysis=analysis,
        modifications=extract_numbered(operation, "加减变化"),
        modern_applications=extract_numbered(operation, "现代运用"),
        contraindications=extract_numbered(operation, "使用注意"),
        evidence=indications,
        raw_text=clean_text(body),
    )
    entry.search_text = make_search_text(entry)
    return entry


def build_entries(source_path: Path) -> list[SyndromeEntry]:
    markdown = source_path.read_text(encoding="utf-8")
    entries = [build_entry(title, body, source_path.name) for title, body in split_sections(markdown)]
    return [entry for entry in entries if entry.ancient_symptoms or entry.pathogenesis or entry.indications]


def collect_classical_formula_names(source_paths: list[Path]) -> set[str]:
    formula_names: set[str] = set()
    heading_re = re.compile(r"^(.{2,30}?(?:汤|丸|散|饮|方))方?[：:]?$")
    bad_name_markers = ("方论", "原方", "无方", "制方", "偶方", "方法", "处方", "药方")
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            line = clean_text(raw_line)
            match = heading_re.match(line)
            if not match:
                continue
            name = clean_text(match.group(1))
            if len(name) > 16 or any(skip in name for skip in ("用后方", "前方", "本方", "汤方")):
                continue
            if any(marker in name for marker in bad_name_markers):
                continue
            formula_names.add(name)
    return formula_names


def extract_formula_mentions(line: str, known_formulas: set[str]) -> list[str]:
    selected: list[tuple[int, int, str]] = []
    for formula in sorted(known_formulas, key=len, reverse=True):
        start = line.find(formula)
        while start != -1:
            end = start + len(formula)
            before = line[max(0, start - 2):start]
            before3 = line[max(0, start - 3):start]
            after = line[end:end + 3]
            positive = after.startswith("主之")
            positive = positive or (before.endswith("宜") and not before.endswith("不宜"))
            positive = positive or (before.endswith("可与") and not before3.endswith("不可与"))
            positive = positive or (before.endswith("宜") and not before.endswith("不宜") and after.startswith("方"))
            if positive and not any(start < old_end and end > old_start for old_start, old_end, _ in selected):
                selected.append((start, end, formula))
            start = line.find(formula, start + 1)

    if selected:
        selected.sort(key=lambda item: item[0])
        return [formula for _, _, formula in selected]
    return []


def iter_classical_clauses(source_path: Path) -> list[dict[str, str]]:
    book_title = source_path.stem.replace("古籍_", "")
    chapter = ""
    chapter_url = ""
    clauses: list[dict[str, str]] = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if line.startswith("## "):
            chapter = clean_text(line[3:])
            chapter_url = ""
            continue
        if line.startswith("> 来源:"):
            chapter_url = clean_text(line.split(":", 1)[-1])
            continue
        if line.startswith(">"):
            continue
        for clause_line in re.split(r"[。；;]", line):
            clause_line = clean_text(clause_line)
            if not any(marker in clause_line for marker in POSITIVE_FORMULA_MARKERS):
                continue
            if any(marker in clause_line for marker in NEGATIVE_FORMULA_MARKERS):
                continue
            clauses.append(
                {
                    "book_title": book_title,
                    "chapter": chapter,
                    "chapter_url": chapter_url,
                    "line": clause_line + "。",
                }
            )
    return clauses


def build_classical_entries(source_paths: list[Path], formula_entries: list[SyndromeEntry]) -> list[SyndromeEntry]:
    known_formulas = {entry.formula for entry in formula_entries if entry.formula}
    known_formulas.update(collect_classical_formula_names(source_paths))
    formula_lookup = {entry.formula: entry for entry in formula_entries}

    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for clause in iter_classical_clauses(source_path):
            line = clause["line"]
            for formula in extract_formula_mentions(line, known_formulas):
                key = f"{source_path.name}::{clause['chapter']}::{formula}::{line}"
                if key in seen:
                    continue
                seen.add(key)
                ancient_symptoms = split_symptom_phrase(line)
                if not ancient_symptoms:
                    continue
                modern_symptoms, aliases = modernize_symptoms(ancient_symptoms)
                pathogenesis = extract_pathogenesis(line, line)
                formula_entry = formula_lookup.get(formula)
                contraindications = ""
                if any(marker in line for marker in NEGATIVE_FORMULA_MARKERS):
                    contraindications = line

                entry = SyndromeEntry(
                    entry_id=f"classical::{key}",
                    title=f"{formula}古籍条文证",
                    source_type="classical_clause",
                    source_book=clause["book_title"],
                    source_file=source_path.name,
                    source_url=clause["chapter_url"],
                    chapter=clause["chapter"],
                    syndrome_name=extract_syndrome_name(line) or f"{formula}相关条文",
                    ancient_symptoms=ancient_symptoms,
                    modern_symptoms=modern_symptoms,
                    symptom_aliases=aliases,
                    diagnostic_keys=ancient_symptoms,
                    pathogenesis=pathogenesis,
                    treatment_principle=formula_entry.treatment_principle if formula_entry else "",
                    formula=formula,
                    formula_composition=formula_entry.formula_composition if formula_entry else [],
                    usage_original="",
                    functions=formula_entry.functions if formula_entry else "",
                    indications=line,
                    formula_analysis="",
                    modifications="",
                    modern_applications="",
                    contraindications=contraindications,
                    evidence=line,
                    review_status="rule_extracted_classical_clause",
                    confidence=0.52,
                    raw_text=line,
                )
                entry.search_text = make_search_text(entry)
                entries.append(entry)
    return entries


def is_plain_chapter_title(line: str) -> bool:
    if not (2 <= len(line) <= 30):
        return False
    if any(mark in line for mark in "，。；：？！、,.?;:"):
        return False
    return bool(re.search(r"第[一二三四五六七八九十百〇○零]+$", line))


def iter_chapter_sentences(source_path: Path) -> list[dict[str, str]]:
    book_title = source_path.stem.replace("古籍_", "")
    chapter = ""
    chapter_url = ""
    buffer: list[str] = []
    records: list[dict[str, str]] = []

    def flush() -> None:
        if not buffer:
            return
        text = clean_text("".join(buffer))
        for sentence in re.split(r"[。？！]", text):
            sentence = clean_text(sentence)
            if len(sentence) >= 8:
                records.append(
                    {
                        "book_title": book_title,
                        "chapter": chapter,
                        "chapter_url": chapter_url,
                        "sentence": sentence,
                    }
                )
        buffer.clear()

    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if line.startswith("## "):
            flush()
            chapter = clean_text(line[3:])
            chapter_url = ""
            continue
        if line.startswith("# "):
            continue
        if line.startswith("> 来源页:") or line.startswith("> 章节来源:"):
            chapter_url = clean_text(line.split(":", 1)[-1])
            continue
        if line.startswith(">"):
            continue
        if is_plain_chapter_title(line):
            flush()
            chapter = line
            continue
        buffer.append(line)
    flush()
    return records


def extract_acupoints_or_channels(sentence: str) -> list[str]:
    selected: list[str] = []
    for term in sorted(ACUPOINT_OR_CHANNEL_TERMS, key=len, reverse=True):
        if term not in sentence:
            continue
        if any(term in old for old in selected):
            continue
        selected.append(term)
    return selected


def extract_treatment_method(sentence: str) -> str:
    patterns = [
        r"(取之.{0,20})",
        r"(取以.{0,20})",
        r"(当刺.{0,20})",
        r"(可刺.{0,20})",
        r"(刺之.{0,20})",
        r"(灸之.{0,20})",
        r"(深刺之?)",
        r"(浅刺之?)",
        r"(则[补泻]之)",
        r"([虚实盛衰寒热陷下不盛不虚]{1,8}则[补泻疾留灸][之]?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            return clean_text(match.group(1).rstrip("，；、"))
    return "针刺/取穴相关条文"


def is_specific_acupuncture_clause(sentence: str) -> bool:
    if any(marker in sentence for marker in ACUPUNCTURE_EXCLUDE_MARKERS):
        return False
    if not any(marker in sentence for marker in ACUPUNCTURE_MARKERS):
        return False
    if not any(term in sentence for term in ACUPUNCTURE_CLINICAL_TERMS):
        return False
    method = extract_treatment_method(sentence)
    if any(marker in method for marker in ("奈何", "处也", "之道", "之法")):
        return False
    return True


def build_acupuncture_entries(source_paths: list[Path]) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for record in iter_chapter_sentences(source_path):
            sentence = record["sentence"]
            if not is_specific_acupuncture_clause(sentence):
                continue
            ancient_symptoms = split_symptom_phrase(sentence)
            if not ancient_symptoms:
                ancient_symptoms = unique([term for term in CLINICAL_HINT_TERMS if term in sentence and len(term) > 1])
            if not ancient_symptoms:
                continue
            method = extract_treatment_method(sentence)
            acupoints_or_channels = extract_acupoints_or_channels(sentence)
            key = f"{source_path.name}::{record['chapter']}::{sentence}"
            if key in seen:
                continue
            seen.add(key)
            modern_symptoms, aliases = modernize_symptoms(ancient_symptoms)
            intervention_name = "、".join(acupoints_or_channels[:3]) if acupoints_or_channels else method
            entry = SyndromeEntry(
                entry_id=f"acupuncture::{key}",
                title=f"{record['book_title']}针刺条文",
                source_type="classical_acupuncture",
                source_book=record["book_title"],
                source_file=source_path.name,
                source_url=record["chapter_url"],
                chapter=record["chapter"],
                syndrome_name=clean_text(sentence[:40]),
                ancient_symptoms=ancient_symptoms,
                modern_symptoms=modern_symptoms,
                symptom_aliases=aliases,
                diagnostic_keys=ancient_symptoms,
                pathogenesis=extract_pathogenesis(sentence, sentence),
                intervention_type="acupuncture",
                intervention_name=intervention_name,
                treatment_method=method,
                acupoints_or_channels=acupoints_or_channels,
                treatment_principle=method,
                formula="",
                formula_composition=[],
                usage_original=method,
                functions="",
                indications=sentence,
                evidence=sentence,
                review_status="rule_extracted_classical_acupuncture",
                confidence=0.45,
                raw_text=sentence,
            )
            entry.search_text = make_search_text(entry)
            entries.append(entry)
    return entries


def normalize_herb_lines(source_path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = clean_text(raw_line)
        if line:
            lines.append(line)
    return lines


def next_non_punctuation(lines: list[str], idx: int, max_lookahead: int = 7) -> str:
    for lookahead in range(idx + 1, min(len(lines), idx + max_lookahead + 1)):
        value = lines[lookahead].replace(" ", "")
        if value in HERB_PUNCTUATION_LINES:
            continue
        return value
    return ""


def is_herb_name_candidate(lines: list[str], idx: int) -> bool:
    compact = lines[idx].replace(" ", "")
    if not (1 <= len(compact) <= 10):
        return False
    if compact in HERB_STOP_NAMES:
        return False
    if any(mark in compact for mark in "，。；：？！、,.?;:（）()《》·"):
        return False
    if compact.endswith(("种", "同", "曰", "云", "篇", "经", "本草", "之药", "之草", "之木")):
        return False
    if not re.fullmatch(r"[\u3400-\u9fff]+", compact):
        return False
    return next_non_punctuation(lines, idx).startswith("味")


def update_herb_scope(line: str, current_grade: str, current_category: str) -> tuple[str, str]:
    compact = line.replace(" ", "")
    grade_map = {"上经": "上品", "中经": "中品", "下经": "下品"}
    if compact in grade_map:
        current_grade = grade_map[compact]

    category_match = re.match(r"^[上中下]\s+(.{1,16})$", line)
    if category_match:
        category = clean_text(category_match.group(1))
        category = re.sub(r"[上中下]品.*$", "", category)
        category = re.sub(r"[，。].*$", "", category)
        category = clean_text(category)
        if category and 1 <= len(category) <= 6:
            current_category = category
    return current_grade, current_category


def parse_herb_category_line(line: str) -> tuple[str, str] | None:
    match = re.match(r"^([上中下])\s*(.{1,16})$", line)
    if not match:
        return None
    grade = HERB_GRADE_PREFIX.get(match.group(1), "")
    category = clean_text(match.group(2))
    category = re.sub(r"[上中下]品.*$", "", category)
    category = re.sub(r"[，。].*$", "", category)
    category = clean_text(category)
    if not grade or not category or len(category) > 6:
        return None
    return grade, category


def split_herb_catalog_names(line: str) -> list[str]:
    if any(mark in line for mark in "，。；：？！,.?;:（）()《》"):
        return []
    names: list[str] = []
    for raw_name in re.split(r"[ 　、]+", line):
        name = clean_text(raw_name)
        if not name or name in HERB_STOP_NAMES:
            continue
        if 1 <= len(name) <= 10 and re.fullmatch(r"[\u3400-\u9fff]+", name):
            names.append(name)
    return names


def build_herb_catalog_metadata(lines: list[str]) -> dict[str, tuple[str, str]]:
    catalog: dict[str, tuple[str, str]] = {}
    for idx, line in enumerate(lines):
        names = split_herb_catalog_names(line)
        if not names:
            continue
        category: tuple[str, str] | None = None
        for lookahead in range(idx + 1, min(len(lines), idx + 6)):
            value = lines[lookahead]
            if value.replace(" ", "") in HERB_PUNCTUATION_LINES:
                continue
            category = parse_herb_category_line(value)
            break
        if category is None:
            continue
        for name in names:
            catalog[name] = category
            if name.endswith("砂"):
                catalog[name[:-1] + "沙"] = category
    return catalog


def has_previous_herb_boundary(lines: list[str], idx: int) -> bool:
    for lookback in range(idx - 1, max(-1, idx - 8), -1):
        value = lines[lookback].replace(" ", "")
        if not value:
            continue
        if value in {"。", "）", "》"}:
            return True
        if value in {"，", "、", "；", "：", ",", ";", ":"}:
            return False
        return False
    return False


def collect_herb_starts(lines: list[str]) -> tuple[list[int], dict[int, tuple[str, str]]]:
    starts: list[int] = []
    metadata: dict[int, tuple[str, str]] = {}
    catalog_metadata = build_herb_catalog_metadata(lines)
    current_grade = ""
    current_category = ""
    for idx, line in enumerate(lines):
        current_grade, current_category = update_herb_scope(line, current_grade, current_category)
        if is_herb_name_candidate(lines, idx):
            herb_name = line.replace(" ", "")
            catalog_key = HERB_NAME_VARIANTS.get(herb_name, herb_name)
            in_catalog = catalog_key in catalog_metadata
            if catalog_metadata and not in_catalog and not has_previous_herb_boundary(lines, idx):
                continue
            starts.append(idx)
            metadata[idx] = catalog_metadata.get(catalog_key, (current_grade, current_category) if not catalog_metadata else ("", ""))
    return starts, metadata


def join_herb_fragments(parts: list[str]) -> str:
    text = ""
    for raw_part in parts:
        part = clean_text(raw_part)
        if not part:
            continue
        if part in HERB_PUNCTUATION_LINES:
            text = text.rstrip() + part
        elif part in {"（", "(", "《"}:
            text += part
        elif part in {"）", ")", "》"}:
            text = text.rstrip() + part
        else:
            text += part
    text = re.sub(r"（.*?）", "", text)
    text = re.sub(r"[（(《].*$", "", text)
    return clean_text(text.strip("，。；、 "))


def trim_herb_core(block_lines: list[str]) -> list[str]:
    if not block_lines:
        return []
    core = [block_lines[0]]
    seen_indication = False
    for line in block_lines[1:]:
        compact = line.replace(" ", "")
        if seen_indication and (compact in {"（", "("} or compact.startswith(HERB_NOTE_STARTERS)):
            break
        if compact.startswith("主"):
            seen_indication = True
        core.append(line)
    return core


def extract_herb_nature(core_lines: list[str]) -> str:
    start = -1
    end = len(core_lines)
    for idx, line in enumerate(core_lines):
        compact = line.replace(" ", "")
        if start < 0 and compact.startswith("味"):
            start = idx
        if start >= 0 and compact.startswith("主"):
            end = idx
            break
    if start < 0:
        return ""
    return join_herb_fragments(core_lines[start:end])


def extract_herb_indications(core_lines: list[str]) -> str:
    start = -1
    for idx, line in enumerate(core_lines):
        if line.replace(" ", "").startswith("主"):
            start = idx
            break
    if start < 0:
        return ""

    parts: list[str] = []
    for line in core_lines[start:]:
        compact = line.replace(" ", "")
        if parts and compact.startswith(HERB_INDICATION_STOPS):
            break
        if parts and compact.startswith(HERB_NOTE_STARTERS):
            break
        parts.append(line)
    return join_herb_fragments(parts)


def extract_herb_aliases(core_lines: list[str]) -> list[str]:
    text = join_herb_fragments(core_lines)
    aliases = []
    for match in re.finditer(r"一名([^，。；、（）()《》]{1,12})", text):
        alias = clean_text(match.group(1))
        if alias:
            aliases.append(alias)
    return unique(aliases)


def extract_herb_origin(core_lines: list[str]) -> str:
    for idx, line in enumerate(core_lines):
        compact = line.replace(" ", "")
        if not compact.startswith("生"):
            continue
        if compact.startswith(("生肌", "生肉", "生津")):
            continue
        origin = join_herb_fragments(core_lines[idx:idx + 5])
        return clean_text(re.split(r"[。；;]", origin, maxsplit=1)[0])
    return ""


def extract_herb_indication_terms(indications: str) -> list[str]:
    text = re.sub(r"^主", "", clean_text(indications))
    candidates: list[str] = []
    for term in SYMPTOM_TERMS:
        if term in text:
            candidates.append(term)

    for raw_chunk in re.split(r"[，、；;。.\n]|或|及|并|而|与", text):
        chunk = clean_text(raw_chunk)
        chunk = re.sub(r"^(主|治|疗|除|止|利|下|破|去)", "", chunk)
        chunk = re.sub(r"[（(《].*$", "", chunk)
        chunk = clean_text(chunk)
        if not chunk or chunk in SYMPTOM_STOPWORDS:
            continue
        if any(stop in chunk for stop in HERB_TERM_STOPWORDS):
            continue
        if 1 < len(chunk) <= 14:
            candidates.append(chunk)
    return unique(candidates)


def iter_herb_blocks(source_path: Path) -> list[dict[str, Any]]:
    lines = normalize_herb_lines(source_path)
    starts, metadata = collect_herb_starts(lines)
    blocks: list[dict[str, Any]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        raw_block = lines[start:end]
        core_lines = trim_herb_core(raw_block)
        indications = extract_herb_indications(core_lines)
        if not indications:
            continue
        herb_name = clean_text(lines[start].replace(" ", ""))
        grade, category = metadata.get(start, ("", ""))
        blocks.append(
            {
                "herb_name": herb_name,
                "grade": grade,
                "category": category,
                "core_lines": core_lines,
                "raw_block": raw_block,
                "nature": extract_herb_nature(core_lines),
                "indications": indications,
                "aliases": extract_herb_aliases(core_lines),
                "origin": extract_herb_origin(core_lines),
            }
        )
    return blocks


def build_herb_entries(source_paths: list[Path]) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        book_title = source_path.stem.replace("古籍_", "")
        for block in iter_herb_blocks(source_path):
            herb_name = block["herb_name"]
            indications = block["indications"]
            ancient_symptoms = extract_herb_indication_terms(indications)
            if not ancient_symptoms:
                continue
            key = f"{source_path.name}::{block['grade']}::{block['category']}::{herb_name}::{indications}"
            if key in seen:
                continue
            seen.add(key)
            modern_symptoms, aliases = modernize_symptoms(ancient_symptoms)
            nature = block["nature"]
            toxic_note = "原文性味含“有毒”，仅作古籍索引，不可自行服用。" if "有毒" in nature else ""
            property_text = clean_text("；".join(part for part in (nature, indications) if part))
            entry = SyndromeEntry(
                entry_id=f"herb::{key}",
                title=f"{herb_name}本草主治",
                source_type="herb_indication",
                source_book=book_title,
                source_file=source_path.name,
                source_url="",
                chapter=clean_text(" ".join(part for part in (block["grade"], block["category"]) if part)),
                syndrome_name=f"{herb_name}主治",
                ancient_symptoms=ancient_symptoms,
                modern_symptoms=modern_symptoms,
                symptom_aliases=aliases,
                diagnostic_keys=ancient_symptoms,
                pathogenesis=[],
                intervention_type="herb",
                intervention_name=herb_name,
                treatment_method="古籍单味药主治线索",
                acupoints_or_channels=[],
                treatment_principle=indications,
                formula="",
                formula_composition=[],
                herb_name=herb_name,
                herb_grade=block["grade"],
                herb_category=block["category"],
                herb_aliases=block["aliases"],
                nature_flavor=[nature] if nature else [],
                origin_habitat=block["origin"],
                property_text=property_text,
                usage_original="",
                functions=property_text,
                indications=indications,
                formula_analysis="",
                modifications="",
                modern_applications="",
                contraindications=toxic_note,
                evidence=join_herb_fragments(block["core_lines"][1:]),
                review_status="rule_extracted_herb_indication",
                confidence=0.42,
                raw_text=join_herb_fragments(block["raw_block"][:120]),
            )
            entry.search_text = make_search_text(entry)
            entries.append(entry)
    return entries


def extract_acupuncture_terms(sentence: str) -> list[str]:
    terms: list[str] = []
    for term in sorted(ACUPUNCTURE_PRINCIPLE_TERMS + ACUPOINT_OR_CHANNEL_TERMS, key=len, reverse=True):
        if term not in sentence:
            continue
        if any(term in old for old in terms):
            continue
        terms.append(term)
    if "刺" in sentence and "刺法" not in terms:
        terms.append("刺法")
    if "灸" in sentence and "灸法" not in terms:
        terms.append("灸法")
    if "针" in sentence and "针法" not in terms:
        terms.append("针法")
    return unique(terms)


def is_acupuncture_principle_clause(sentence: str) -> bool:
    if not any(marker in sentence for marker in ACUPUNCTURE_MARKERS) and not any(term in sentence for term in ACUPUNCTURE_PRINCIPLE_TERMS):
        return False
    if any(marker in sentence for marker in ("黄帝曰", "岐伯曰", "何谓", "奈何", "可得闻乎")) and len(sentence) < 30:
        return False
    terms = extract_acupuncture_terms(sentence)
    if not terms:
        return False
    if "迎随" in terms and not any(term in sentence for term in ACUPUNCTURE_PRINCIPLE_CONTEXT_TERMS):
        return False
    if "迎随" in terms and any(term in sentence for term in NON_ACUPUNCTURE_YINGSUI_CONTEXT):
        if not any(term in sentence for term in ("针", "刺", "取", "补", "泻", "荣卫", "经脉")):
            return False
    if len(sentence) < 8:
        return False
    return True


def build_acupuncture_principle_entries(source_paths: list[Path]) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for record in iter_chapter_sentences(source_path):
            sentence = record["sentence"]
            if is_specific_acupuncture_clause(sentence):
                continue
            if not is_acupuncture_principle_clause(sentence):
                continue
            terms = extract_acupuncture_terms(sentence)
            key = f"{source_path.name}::{record['chapter']}::{sentence}"
            if key in seen:
                continue
            seen.add(key)
            modern_terms, aliases = modernize_symptoms(terms)
            method = extract_treatment_method(sentence)
            acupoints_or_channels = extract_acupoints_or_channels(sentence)
            topic = clean_text(sentence[:40])
            entry = SyndromeEntry(
                entry_id=f"acupuncture_principle::{key}",
                title=f"{record['book_title']}针法原则",
                source_type="classical_acupuncture_principle",
                source_book=record["book_title"],
                source_file=source_path.name,
                source_url=record["chapter_url"],
                chapter=record["chapter"],
                syndrome_name=topic,
                ancient_symptoms=terms,
                modern_symptoms=modern_terms,
                symptom_aliases=aliases,
                diagnostic_keys=terms,
                pathogenesis=[],
                intervention_type="acupuncture_principle",
                intervention_name=method,
                treatment_method=method,
                acupoints_or_channels=acupoints_or_channels,
                treatment_principle=method,
                formula="",
                formula_composition=[],
                theory_topic=topic,
                theory_question="",
                theory_answer=sentence,
                theory_terms=terms,
                diagnostic_method="针刺/灸法原则",
                acupuncture_principle=sentence,
                acupuncture_terms=terms,
                functions="",
                indications=sentence,
                evidence=sentence,
                review_status="rule_extracted_acupuncture_principle",
                confidence=0.4,
                raw_text=sentence,
            )
            entry.search_text = make_search_text(entry)
            entries.append(entry)
    return entries


def join_classic_fragments(parts: list[str]) -> str:
    text = ""
    for raw_part in parts:
        part = clean_text(raw_part)
        if not part:
            continue
        if part in {"，", "。", "；", "：", "？", "！", "、", ",", ".", ";", ":", "?", "!"}:
            text = text.rstrip() + part
        elif part in {"）", ")", "》", "」"}:
            text = text.rstrip() + part
        elif part in {"（", "(", "《", "「"}:
            text += part
        else:
            text += part
    text = re.sub(r"（.*?）", "", text)
    text = re.sub(r"「.*?」", "", text)
    return clean_text(text)


def is_theory_section_title(line: str) -> bool:
    return bool(re.fullmatch(r"[一二三四五六七八九十百]+难", line))


def iter_theory_sections(source_path: Path) -> list[dict[str, str]]:
    book_title = source_path.stem.replace("古籍_", "")
    chapter_url = ""
    title = ""
    buffer: list[str] = []
    sections: list[dict[str, str]] = []

    def flush() -> None:
        if not title or not buffer:
            return
        text = join_classic_fragments(buffer)
        if text:
            sections.append(
                {
                    "book_title": book_title,
                    "chapter": title,
                    "chapter_url": chapter_url,
                    "text": text,
                }
            )

    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if line.startswith("> 来源页:") or line.startswith("> 章节来源:"):
            chapter_url = clean_text(line.split(":", 1)[-1])
            continue
        if line.startswith("#") or line.startswith(">"):
            continue
        if is_theory_section_title(line):
            flush()
            title = line
            buffer = [line]
            continue
        if title:
            buffer.append(line)
    flush()
    return sections


def extract_theory_question(text: str) -> str:
    match = re.search(r"难曰[:：](.+?？)", text)
    if match:
        return clean_text(match.group(1))
    match = re.search(r"曰[:：](.+?？)", text)
    return clean_text(match.group(1)) if match else ""


def extract_theory_answer(text: str, question: str) -> str:
    if question and question in text:
        answer = text.split(question, 1)[-1]
    else:
        answer = text
    answer = re.sub(r"^[，。；：然\s]+", "", answer)
    return clean_text(answer)


def extract_theory_terms(text: str) -> list[str]:
    terms: list[str] = []
    for term in sorted(THEORY_TERMS, key=len, reverse=True):
        if term in text:
            terms.append(term)
    for match in re.finditer(r"(?:谓|曰|为)([^，。；：？]{1,12})", text):
        value = clean_text(match.group(1))
        if 1 < len(value) <= 8 and not any(skip in value for skip in ("何", "也", "者")):
            terms.append(value)
    return unique(terms)


def extract_theory_fallback_terms(question: str, answer: str, chapter: str) -> list[str]:
    text = clean_text(question or answer)
    terms: list[str] = [chapter]
    for raw_chunk in re.split(r"[，。；：？！、,.?;:\s]|何谓也|何以|奈何|何如", text):
        chunk = clean_text(raw_chunk)
        if not chunk:
            continue
        chunk = re.sub(r"^(然|故|曰|谓|其|有|无)", "", chunk)
        chunk = clean_text(chunk)
        if 1 < len(chunk) <= 10 and not any(skip in chunk for skip in ("难曰", "何谓", "也")):
            terms.append(chunk)
    return unique(terms[:8])


def infer_diagnostic_method(terms: list[str], text: str) -> str:
    if any(term in terms for term in ("寸口", "脉", "脉诊", "尺寸", "寸关尺", "浮", "沉", "迟", "数")):
        return "脉诊/诊法理论"
    if any(term in terms for term in ("经络", "十二经", "奇经八脉", "任脉", "督脉")):
        return "经络理论"
    if any(term in terms for term in ("补泻", "虚实", "井荥俞经合")) or "刺" in text:
        return "针法/补泻理论"
    if any(term in terms for term in ("五藏", "六府", "三焦", "命门")):
        return "脏腑理论"
    return "古籍理论问答"


def build_theory_entries(source_paths: list[Path]) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for section in iter_theory_sections(source_path):
            text = section["text"]
            question = extract_theory_question(text)
            answer = extract_theory_answer(text, question)
            terms = extract_theory_terms(text)
            if not terms:
                terms = extract_theory_fallback_terms(question, answer, section["chapter"])
            if not terms:
                continue
            modern_terms, aliases = modernize_symptoms(terms)
            topic = clean_text(question.rstrip("？") if question else answer[:40])
            key = f"{source_path.name}::{section['chapter']}::{topic}"
            if key in seen:
                continue
            seen.add(key)
            method = infer_diagnostic_method(terms, text)
            entry = SyndromeEntry(
                entry_id=f"theory::{key}",
                title=f"{section['book_title']}{section['chapter']}理论",
                source_type="classical_theory",
                source_book=section["book_title"],
                source_file=source_path.name,
                source_url=section["chapter_url"],
                chapter=section["chapter"],
                syndrome_name=topic,
                ancient_symptoms=terms,
                modern_symptoms=modern_terms,
                symptom_aliases=aliases,
                diagnostic_keys=terms,
                pathogenesis=[],
                intervention_type="theory",
                intervention_name=method,
                treatment_method=method,
                acupoints_or_channels=[],
                treatment_principle="",
                formula="",
                formula_composition=[],
                theory_topic=topic,
                theory_question=question,
                theory_answer=answer,
                theory_terms=terms,
                diagnostic_method=method,
                functions="",
                indications=answer,
                evidence=text,
                review_status="rule_extracted_classical_theory",
                confidence=0.48,
                raw_text=text,
            )
            entry.search_text = make_search_text(entry)
            entries.append(entry)
    return entries


def load_extra_entries(paths: list[Path]) -> list[SyndromeEntry]:
    entries: list[SyndromeEntry] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            entry = SyndromeEntry.model_validate(payload)
            if not entry.search_text:
                entry.search_text = make_search_text(entry)
            if entry.entry_id in seen:
                continue
            seen.add(entry.entry_id)
            entries.append(entry)
    return entries


def merge_entries_with_replacements(entries: list[SyndromeEntry]) -> tuple[list[SyndromeEntry], int]:
    merged: list[SyndromeEntry] = []
    positions: dict[str, int] = {}
    replacement_count = 0
    for entry in entries:
        if entry.entry_id in positions:
            merged[positions[entry.entry_id]] = entry
            replacement_count += 1
            continue
        positions[entry.entry_id] = len(merged)
        merged.append(entry)
    return merged, replacement_count


def sanitize_entries(entries: list[SyndromeEntry]) -> list[SyndromeEntry]:
    for entry in entries:
        for herb in entry.formula_composition:
            herb.original_dose = sanitize_formula_dose(herb.original_dose)
        derive_differential_fields(entry)
        normalize_required_symptom_groups(entry)
        entry.search_text = make_search_text(entry)
    return entries


def write_jsonl(entries: list[SyndromeEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")


def write_markdown_preview(entries: list[SyndromeEntry], output_path: Path, limit: int = 30) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 结构化方证字典预览", ""]
    for entry in entries[:limit]:
        lines.extend(
            [
                f"## {entry.title}",
                f"- **古代症状**：{'、'.join(entry.ancient_symptoms) or '未抽取'}",
                f"- **现代对照症状**：{'、'.join(entry.modern_symptoms) or '未抽取'}",
                f"- **口语近义表达**：{'、'.join(entry.symptom_aliases) or '未抽取'}",
                f"- **核心病因**：{'、'.join(entry.pathogenesis) or '未抽取'}",
                f"- **必需症状组**：{'; '.join(' / '.join(group) for group in entry.required_symptom_groups) or '未抽取'}",
                f"- **排除症状**：{'、'.join(entry.forbidden_terms) or '未抽取'}",
                f"- **鉴别要点**：{'、'.join(entry.differential_keys) or '未抽取'}",
                f"- **需追问字段**：{'、'.join(entry.must_clarify_fields) or '未抽取'}",
                f"- **干预类型**：{entry.intervention_type}",
                f"- **干预名称**：{entry.intervention_name or entry.formula or '未抽取'}",
                f"- **对应方剂**：{entry.formula or '不适用'}",
                f"- **本草药物**：{entry.herb_name or '不适用'}",
                f"- **性味**：{'、'.join(entry.nature_flavor) or '未抽取'}",
                f"- **本草部类**：{entry.herb_grade} {entry.herb_category}".strip(),
                f"- **理论主题**：{entry.theory_topic or '不适用'}",
                f"- **理论术语**：{'、'.join(entry.theory_terms) or '未抽取'}",
                f"- **诊法/理论类型**：{entry.diagnostic_method or '未抽取'}",
                f"- **针法原则**：{entry.acupuncture_principle or '不适用'}",
                f"- **针法术语**：{'、'.join(entry.acupuncture_terms) or '未抽取'}",
                f"- **穴位/经脉**：{'、'.join(entry.acupoints_or_channels) or '未抽取'}",
                f"- **功用**：{entry.functions or '未抽取'}",
                f"- **正确用法/方法**：{entry.usage_original or entry.treatment_method or '未抽取'}",
                f"- **使用注意**：{entry.contraindications or '未抽取'}",
                f"- **来源**：{entry.source_book} {entry.source_url}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_qdrant(entries: list[SyndromeEntry], recreate: bool = True) -> int:
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client.http import models as qmodels

    client = create_qdrant_client()
    try:
        model_kwargs = {"local_files_only": config.EMBEDDING_LOCAL_FILES_ONLY}
        if config.EMBEDDING_DEVICE and config.EMBEDDING_DEVICE != "auto":
            model_kwargs["device"] = config.EMBEDDING_DEVICE
        embedding = HuggingFaceEmbeddings(
            model_name=config.DENSE_MODEL,
            model_kwargs=model_kwargs,
        )
        vector_size = len(embedding.embed_query("test"))
        collection_name = config.SYNDROME_COLLECTION

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        elif recreate:
            while True:
                records, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    with_payload=False,
                    with_vectors=False,
                )
                point_ids = [record.id for record in records]
                if point_ids:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=qmodels.PointIdsList(points=point_ids),
                        wait=True,
                    )
                if not next_offset:
                    break
            cleared_count = client.count(collection_name=collection_name, exact=True).count
            if cleared_count != 0:
                raise RuntimeError(f"Qdrant clear failed: remaining points {cleared_count}")

        vectors = embedding.embed_documents([entry.search_text for entry in entries])
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, entry.entry_id)),
                vector=vector,
                payload=entry.model_dump(mode="json"),
            )
            for entry, vector in zip(entries, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        count = client.count(collection_name=collection_name, exact=True).count
        if recreate and count != len(entries):
            raise RuntimeError(f"Qdrant rebuild count mismatch: expected {len(entries)}, got {count}")
        return count
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建结构化方证字典并可写入 Qdrant")
    parser.add_argument("--source", default=str(ROOT / "markdown_docs" / "方剂大全.md"))
    parser.add_argument(
        "--classic-source",
        action="append",
        default=None,
        help="要抽取正向用方条文的古籍 Markdown，可重复传入；默认使用伤寒论、金匮要略、温病条辨",
    )
    parser.add_argument(
        "--acupuncture-source",
        action="append",
        default=None,
        help="要抽取针刺/灸法条文的古籍 Markdown，可重复传入；默认使用灵枢、素问、难经",
    )
    parser.add_argument(
        "--herb-source",
        action="append",
        default=None,
        help="要抽取单味药主治的本草 Markdown，可重复传入；默认使用神农本草经",
    )
    parser.add_argument(
        "--theory-source",
        action="append",
        default=None,
        help="要抽取理论问答/诊法的古籍 Markdown，可重复传入；默认使用难经",
    )
    parser.add_argument("--no-classics", action="store_true", help="只生成《方剂大全》结构化方证，不抽取古籍条文")
    parser.add_argument("--no-acupuncture", action="store_true", help="不抽取针刺/灸法条文")
    parser.add_argument("--no-acupuncture-principles", action="store_true", help="不抽取针刺/灸法原则条文")
    parser.add_argument("--no-herbs", action="store_true", help="不抽取本草单味药主治条文")
    parser.add_argument("--no-theory", action="store_true", help="不抽取古籍理论问答/诊法条文")
    parser.add_argument("--no-reviewed-replacements", action="store_true", help="不自动加载已审核替换 JSONL")
    parser.add_argument(
        "--reviewed-replacements",
        default=str(DEFAULT_REVIEWED_REPLACEMENTS),
        help="默认自动加载的已审核替换 JSONL；同 entry_id 会替换基础条目",
    )
    parser.add_argument(
        "--extra-entry-jsonl",
        action="append",
        default=None,
        help="追加已审核的 SyndromeEntry JSONL，可用于热插拔新古籍；若 entry_id 已存在则替换基础条目；可重复传入",
    )
    parser.add_argument("--output", default=str(ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"))
    parser.add_argument("--preview", default=str(ROOT / "datasets" / "structured" / "syndrome_dictionary_preview.md"))
    parser.add_argument("--write-qdrant", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="不删除已有结构化方证 collection")
    args = parser.parse_args()

    source_path = Path(args.source)
    formula_entries = build_entries(source_path)
    classic_sources = [Path(path) for path in args.classic_source] if args.classic_source else DEFAULT_CLASSIC_SOURCES
    acupuncture_sources = [Path(path) for path in args.acupuncture_source] if args.acupuncture_source else DEFAULT_ACUPUNCTURE_SOURCES
    herb_sources = [Path(path) for path in args.herb_source] if args.herb_source else DEFAULT_HERB_SOURCES
    theory_sources = [Path(path) for path in args.theory_source] if args.theory_source else DEFAULT_THEORY_SOURCES
    classical_entries = [] if args.no_classics else build_classical_entries(classic_sources, formula_entries)
    acupuncture_entries = [] if args.no_acupuncture else build_acupuncture_entries(acupuncture_sources)
    acupuncture_principle_entries = [] if args.no_acupuncture_principles else build_acupuncture_principle_entries(acupuncture_sources)
    herb_entries = [] if args.no_herbs else build_herb_entries(herb_sources)
    theory_entries = [] if args.no_theory else build_theory_entries(theory_sources)
    reviewed_path = Path(args.reviewed_replacements)
    reviewed_entries = (
        []
        if args.no_reviewed_replacements or not reviewed_path.exists()
        else load_extra_entries([reviewed_path])
    )
    extra_entries = load_extra_entries([Path(path) for path in args.extra_entry_jsonl]) if args.extra_entry_jsonl else []
    entries = (
        formula_entries
        + classical_entries
        + acupuncture_entries
        + acupuncture_principle_entries
        + herb_entries
        + theory_entries
        + reviewed_entries
        + extra_entries
    )
    entries, replacement_count = merge_entries_with_replacements(entries)
    entries = sanitize_entries(entries)
    write_jsonl(entries, Path(args.output))
    write_markdown_preview(entries, Path(args.preview))
    source_type_counts: dict[str, int] = {}
    for entry in entries:
        source_type_counts[entry.source_type] = source_type_counts.get(entry.source_type, 0) + 1
    print(json.dumps({
        "entries": len(entries),
        "source_type_counts": source_type_counts,
        "jsonl": args.output,
        "preview": args.preview,
        "collection": config.SYNDROME_COLLECTION,
        "replacement_count": replacement_count,
        "reviewed_replacements": len(reviewed_entries),
    }, ensure_ascii=False, indent=2))

    if args.write_qdrant:
        count = write_qdrant(entries, recreate=not args.keep_existing)
        print(json.dumps({"qdrant_collection": config.SYNDROME_COLLECTION, "points": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
