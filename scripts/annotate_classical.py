"""为中医古籍生成独立、可追溯的医学术语通俗释义伴随稿。

安全边界：
1. 原文不做括号注入或字符串改写。
2. 只解释词表中明确列出的术语，不自动翻译整句。
3. 六经、脉舌、治法等只作术语提示，仍须结合篇章和上下文。
4. 古代剂量保持原单位，不自动换算。
5. 旧版自动注释词表仅用于可逆清理。
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "tcm_knowledge" / "classics" / "plain_language"
DEFAULT_CLASSICS = (
    ROOT / "datasets/tcm_knowledge/classics/jicheng/simplified/黄帝内经_素问.md",
    ROOT / "datasets/tcm_knowledge/classics/jicheng/simplified/黄帝内经_灵枢.md",
    ROOT / "datasets/tcm_knowledge/classics/jicheng/simplified/难经.md",
    ROOT / "datasets/tcm_knowledge/classics/jicheng/simplified/神农本草经.md",
    ROOT / "datasets/tcm_knowledge/classics/jicheng/simplified/温病条辨.md",
    ROOT / "datasets/tcm_knowledge/classics/gushiwen/伤寒论/伤寒论.md",
    ROOT / "datasets/tcm_knowledge/classics/gushiwen/金匮要略/金匮要略.md",
)
COMPANION_MARKER = "> 生成器: scripts/annotate_classical.py v2"


# 旧版词表只能用于清除历史污染，不能重新写回古籍。
LEGACY_ANNOTATIONS = {
    "太阳病": "（即太阳经表证，病在体表，多属外感初起）",
    "阳明病": "（即阳明经热证，病在肠胃，多属实热）",
    "少阳病": "（即少阳经半表半里证，邪在表里之间）",
    "太阴病": "（即太阴经虚寒证，病在脾胃，多属虚寒）",
    "少阴病": "（即少阴经虚寒证，病在心肾，多属危重）",
    "厥阴病": "（即厥阴经寒热错杂证，病在肝经）",
    "恶寒": "（怕冷）",
    "恶风": "（怕风）",
    "发热": "（发烧）",
    "汗出": "（出汗）",
    "无汗": "（不出汗）",
    "头项强痛": "（头项部僵硬疼痛）",
    "往来寒热": "（一阵冷一阵热交替出现）",
    "但欲寐": "（总想睡觉，精神萎靡）",
    "心烦": "（心中烦躁不安）",
    "不得卧": "（无法入睡）",
    "胸胁苦满": "（胸胁部胀满不适）",
    "心下痞": "（胃脘部胀满堵塞感）",
    "腹痛": "（腹部疼痛）",
    "腹胀": "（腹部胀满）",
    "下利": "（腹泻）",
    "便秘": "（大便不通）",
    "小便不利": "（排尿不畅）",
    "口渴": "（口干想喝水）",
    "不渴": "（不觉得口干）",
    "脉浮": "（脉象轻取即得，多主表证）",
    "脉沉": "（脉象重按始得，多主里证）",
    "脉紧": "（脉象紧绷如绳，多主寒证痛证）",
    "脉缓": "（脉象松弛和缓，多主虚证湿证）",
    "脉数": "（脉象频率快，多主热证）",
    "脉迟": "（脉象频率慢，多主寒证）",
    "苔白": "（舌苔呈白色，多主寒证表证）",
    "苔黄": "（舌苔呈黄色，多主热证）",
    "苔腻": "（舌苔粘腻，多主湿证）",
    "舌淡": "（舌色淡白，多主虚证）",
    "舌红": "（舌色偏红，多主热证）",
    "发汗": "（通过发汗驱散表邪）",
    "解表": "（解除表证）",
    "清热": "（清除热邪）",
    "攻下": "（通便攻逐实邪）",
    "和解": "（调和表里或脏腑关系）",
    "温里": "（温补体内阳气驱散寒邪）",
    "补气": "（补益人体元气）",
    "补血": "（补养血液）",
    "活血": "（促进血液循环，消除瘀滞）",
    "利水": "（促进水液代谢排出）",
    "一两": "（约15克，汉代度量衡）",
    "一升": "（约200毫升，汉代度量衡）",
    "一方寸匕": "（约5-10克，散剂计量容器）",
    "主之": "（主治此证）",
    "宜": "（适宜用）",
    "可与": "（可以用）",
    "不可与": "（不可以用）",
    "禁": "（禁止使用）",
}


@dataclass(frozen=True)
class GlossaryEntry:
    category: str
    plain_language: str
    context_required: bool = False


def _six_channel_entry(term: str) -> GlossaryEntry:
    return GlossaryEntry(
        "六经病名",
        f"《伤寒论》六经辨证中的“{term}”病证名称；"
        "具体病位、病性和治法必须依据所在条文的症状与脉证判断。",
        True,
    )


# 仅收录可保守表达的术语。“主之、宜、可与、不可与、禁”和剂量不在此表。
PLAIN_LANGUAGE_GLOSSARY = {
    term: _six_channel_entry(term)
    for term in ("太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病")
}
PLAIN_LANGUAGE_GLOSSARY.update(
    {
        "恶寒": GlossaryEntry("症状", "文中指怕冷的感觉；程度和原因需结合上下文。"),
        "恶风": GlossaryEntry("症状", "文中指对风敏感，遇风容易感觉不适或怕冷。"),
        "发热": GlossaryEntry("症状", "文中指身体发热或自觉发热，不一定等同于已测得体温升高。"),
        "汗出": GlossaryEntry("症状", "出汗。"),
        "无汗": GlossaryEntry("症状", "没有出汗。"),
        "头项强痛": GlossaryEntry("症状", "头部和后颈僵硬、疼痛。"),
        "往来寒热": GlossaryEntry("症状", "冷感与热感交替出现。"),
        "但欲寐": GlossaryEntry("症状", "精神困倦，总想睡眠。"),
        "心烦": GlossaryEntry("症状", "自觉心中烦躁、不安。"),
        "不得卧": GlossaryEntry("症状", "不能安卧或难以入睡，具体表现需结合原文。", True),
        "胸胁苦满": GlossaryEntry("症状", "胸部两侧至胁肋部有胀满、难受的感觉。"),
        "心下痞": GlossaryEntry("症状", "上腹部自觉堵塞、胀满，通常按之不以明显疼痛为主。", True),
        "腹痛": GlossaryEntry("症状", "腹部疼痛。"),
        "腹胀": GlossaryEntry("症状", "腹部胀满。"),
        "下利": GlossaryEntry("症状", "大便次数增多或便质稀薄；古籍具体所指需结合上下文。", True),
        "便秘": GlossaryEntry("症状", "排便困难或大便长时间不通。"),
        "小便不利": GlossaryEntry("症状", "排尿不顺畅或尿量减少，具体表现需结合上下文。", True),
        "口渴": GlossaryEntry("症状", "口中干渴并有饮水需要。"),
        "不渴": GlossaryEntry("症状", "没有明显口渴感。"),
        "短气": GlossaryEntry("症状", "呼吸短促，感觉气不够用。"),
        "喘促": GlossaryEntry("症状", "呼吸急促或费力；严重程度需结合全文。", True),
        "气喘": GlossaryEntry("症状", "呼吸急促或费力；严重程度需结合全文。", True),
        "呕吐": GlossaryEntry("症状", "胃内容物从口中吐出。"),
        "欲呕": GlossaryEntry("症状", "有想要呕吐的感觉。"),
        "干呕": GlossaryEntry("症状", "有呕吐动作或声音，但通常没有吐出食物。", True),
        "厥逆": GlossaryEntry("症状/病机", "古籍含义不一，常涉及四肢逆冷等表现，不能脱离原文定解。", True),
        "四肢厥冷": GlossaryEntry("症状", "四肢末端发冷。", True),
        "手足厥冷": GlossaryEntry("症状", "手脚末端发冷。", True),
        "脉浮": GlossaryEntry("脉象", "诊脉时轻按较明显。", True),
        "脉沉": GlossaryEntry("脉象", "诊脉时轻按不明显，较重按压才较明显。", True),
        "脉紧": GlossaryEntry("脉象", "按起来有紧张、绷急感；临床含义需结合其他脉证。", True),
        "脉缓": GlossaryEntry("脉象", "脉来较和缓或较迟缓；古籍不同语境所指可能不同。", True),
        "脉数": GlossaryEntry("脉象", "脉搏频率较快；判断还受年龄和当时状态影响。", True),
        "脉迟": GlossaryEntry("脉象", "脉搏频率较慢；判断还受年龄和当时状态影响。", True),
        "脉弦": GlossaryEntry("脉象", "脉形较直而长，按之有弦张感。", True),
        "脉滑": GlossaryEntry("脉象", "脉来往较流利、圆滑。", True),
        "脉涩": GlossaryEntry("脉象", "脉来往不够流利，有艰涩感。", True),
        "脉微": GlossaryEntry("脉象", "脉势很弱，触感不明显。", True),
        "脉弱": GlossaryEntry("脉象", "脉势较弱、按之无力；具体含义需结合脉位。", True),
        "苔白": GlossaryEntry("舌象", "舌苔颜色偏白；不能仅凭颜色判断证候。", True),
        "苔黄": GlossaryEntry("舌象", "舌苔颜色偏黄；不能仅凭颜色判断证候。", True),
        "苔腻": GlossaryEntry("舌象", "舌苔外观较厚腻、颗粒不易分清。", True),
        "舌淡": GlossaryEntry("舌象", "舌体颜色较淡。", True),
        "舌红": GlossaryEntry("舌象", "舌体颜色较红。", True),
        "发汗": GlossaryEntry("治法", "以使身体适度出汗为目标的治法名称，不代表任何人都适用。", True),
        "解表": GlossaryEntry("治法", "针对中医所说“表证”的治法类别，须先有辨证依据。", True),
        "清热": GlossaryEntry("治法", "针对中医所说“热证”的治法类别，不能据词名自行用药。", True),
        "攻下": GlossaryEntry("治法", "使里实邪从下排出的治法类别，适用条件和禁忌须依原文辨证。", True),
        "和解": GlossaryEntry("治法", "以调和方式处理特定证候的治法类别，具体所指依上下文。", True),
        "温里": GlossaryEntry("治法", "针对中医所说“里寒证”的治法类别，须先有辨证依据。", True),
        "补气": GlossaryEntry("治法", "针对中医所说“气虚证”的补益治法类别。", True),
        "补血": GlossaryEntry("治法", "针对中医所说“血虚证”的补益治法类别。", True),
        "活血": GlossaryEntry("治法", "用于改善中医所说血行不畅、瘀滞的治法类别。", True),
        "利水": GlossaryEntry("治法", "促进中医所说水液排出的治法类别。", True),
        "心下": GlossaryEntry("部位", "古籍部位词，多指上腹或胃脘区域，并非简单指现代解剖的心脏下方。", True),
        "胸胁": GlossaryEntry("部位", "前胸两侧至胁肋的区域。"),
        "腠理": GlossaryEntry("中医概念", "古代对皮肤肌表纹理及其开合功能的概念，不等同于单一现代解剖结构。", True),
        "营卫": GlossaryEntry("中医概念", "中医用于说明体表防护、营养运行等功能的一组相对概念。", True),
        "荣卫": GlossaryEntry("中医概念", "“营卫”的古籍常见写法。", True),
        "寸口": GlossaryEntry("诊脉部位", "腕部桡动脉附近的诊脉部位。", True),
        "关上": GlossaryEntry("诊脉部位", "寸、关、尺三部中的“关”部。", True),
        "尺中": GlossaryEntry("诊脉部位", "寸、关、尺三部中的“尺”部。", True),
    }
)

# 保留旧名称，避免外部 import 失败；值不再用于原文注入。
CLASSICAL_ANNOTATIONS: dict[str, str] = {}

DOSAGE_PATTERN = re.compile(
    r"(?:[一二三四五六七八九十百千万半\d]+)(?:两|钱|分|升|合|斗|枚|方寸匕)"
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    count: int


def annotate_classical_text(text: str) -> str:
    """兼容旧调用：返回未改动原文，不再执行括号注入。"""
    warnings.warn(
        "原文内自动注释已禁用；请使用 build_plain_language_companion() 生成独立伴随稿。",
        DeprecationWarning,
        stacklevel=2,
    )
    return text


def strip_legacy_annotations(text: str) -> str:
    """移除旧版脚本插入的注释，支持重复执行后的多重注释。"""
    cleaned = text
    for classical in sorted(LEGACY_ANNOTATIONS, key=len, reverse=True):
        annotation = LEGACY_ANNOTATIONS[classical]
        token = classical + annotation
        while token in cleaned:
            cleaned = cleaned.replace(token, classical)

    legacy_values = sorted(set(LEGACY_ANNOTATIONS.values()), key=len, reverse=True)
    alternation = "|".join(re.escape(value) for value in legacy_values)
    for classical in sorted(LEGACY_ANNOTATIONS, key=len, reverse=True):
        pattern = re.compile(re.escape(classical) + rf"(?:(?:{alternation}))+")
        cleaned = pattern.sub(classical, cleaned)
    return cleaned


def extract_glossary_entries(text: str) -> list[tuple[str, GlossaryEntry]]:
    """按首次出现顺序返回安全术语，长词优先且不重复解释重叠词。"""
    candidates = []
    for term, entry in PLAIN_LANGUAGE_GLOSSARY.items():
        candidates.extend(
            (match.start(), match.end(), -len(term), term, entry)
            for match in re.finditer(re.escape(term), text)
        )
    candidates.sort(key=lambda item: (item[0], item[2]))

    selected: list[tuple[int, int, str, GlossaryEntry]] = []
    seen: set[str] = set()
    for start, end, _, term, entry in candidates:
        if term in seen:
            continue
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _, _ in selected):
            continue
        selected.append((start, end, term, entry))
        seen.add(term)
    selected.sort(key=lambda item: item[0])
    return [(term, entry) for _, _, term, entry in selected]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？；])", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = ""
        while len(sentence) > max_chars:
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        current += sentence
    if current:
        chunks.append(current)
    return chunks


def _iter_source_segments(text: str, max_chars: int) -> list[tuple[str, str, str]]:
    section = "正文"
    section_source = ""
    paragraph: list[str] = []
    segments: list[tuple[str, str, str]] = []

    def flush() -> None:
        if not paragraph:
            return
        joined = "".join(paragraph).strip()
        paragraph.clear()
        segments.extend(
            (section, section_source, chunk)
            for chunk in _split_long_text(joined, max_chars)
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            continue
        if line.startswith("##"):
            flush()
            section = line.lstrip("#").strip() or "正文"
            section_source = ""
            continue
        if line.startswith("> 章节来源:"):
            section_source = line
            continue
        if line.startswith(">"):
            continue
        if not line:
            flush()
            continue
        paragraph.append(line)
        if line[-1:] in "。！？；" and sum(map(len, paragraph)) >= max_chars // 2:
            flush()
    flush()
    return segments


def _document_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def build_plain_language_companion(
    text: str,
    *,
    title: str = "古籍",
    max_chars: int = 700,
) -> str:
    """生成“原文片段 + 命中术语释义”的独立 Markdown 伴随稿。"""
    if COMPANION_MARKER in text:
        return text
    if max_chars < 200:
        raise ValueError("max_chars 不能小于 200")

    actual_title = _document_title(text, title)
    metadata = [
        line
        for line in text.splitlines()
        if line.startswith(">")
        and any(
            label in line
            for label in ("目录来源:", "来源页:", "来源站点:", "底本状态:")
        )
    ]
    output = [
        f"# {actual_title}（医学术语通俗释义伴随稿）",
        *metadata,
        COMPANION_MARKER,
        "> 生成方式: 规则词表逐片段匹配；原文未改写；不是逐句白话翻译",
        "> 审核状态: 机器生成，尚未经中医专家逐条审校",
        "> 使用限制: 只能采用下方明确列出的释义，不得扩写、推断诊断或据此开方",
        "> 剂量限制: 古代单位保持原文，不换算为克、毫升等现代单位",
        "",
    ]

    current_section = ""
    for index, (section, section_source, segment) in enumerate(
        _iter_source_segments(text, max_chars),
        start=1,
    ):
        if section != current_section:
            output.extend((f"## {section}", ""))
            if section_source:
                output.extend((section_source, ""))
            current_section = section
        output.extend((f"### 原文片段 {index}", "", segment, ""))

        entries = extract_glossary_entries(segment)
        if entries:
            output.extend(("### 医学术语通俗释义", ""))
            for term, entry in entries:
                context = "；须结合本段上下文" if entry.context_required else ""
                output.append(
                    f"- **{term}**（{entry.category}{context}）：{entry.plain_language}"
                )
        else:
            output.append("> 术语释义: 本片段未命中安全词表，禁止自行补译。")

        if DOSAGE_PATTERN.search(segment):
            output.append(
                "> 剂量提示: 本片段含古代计量表达，具体量值可能受时代、版本和用法影响，未作现代换算。"
            )
        output.append("")

    return "\n".join(output).rstrip() + "\n"


def validate_classical_text(text: str) -> list[ValidationIssue]:
    """检查已知旧注释污染和抓取噪声。"""
    checks = {
        "legacy_annotation": (
            "仍包含旧版自动释义",
            sum(text.count(value) for value in set(LEGACY_ANNOTATIONS.values())),
        ),
        "contradictory_annotation": (
            "包含相互矛盾的“不可与”旧释义",
            text.count("不可与（不可以用）（可以用）"),
        ),
        "crawler_noise_element": (
            "包含网页抓取噪声“元素。”",
            text.count("元素。"),
        ),
    }
    return [
        ValidationIssue(code=code, message=message, count=count)
        for code, (message, count) in checks.items()
        if count
    ]


def run_self_test() -> None:
    sample = "上主之深仁。太阳病，恶寒，脉浮，不可与之。一两。"
    companion = build_plain_language_companion(sample, title="测试古籍", max_chars=200)
    assert annotate_classical_text(sample) == sample
    assert "上主之（主治此证）" not in companion
    assert "约15克" not in companion
    assert "**太阳病**" in companion
    assert "**恶寒**" in companion
    assert "**脉浮**" in companion
    assert "**可与**" not in companion
    assert extract_glossary_entries("厥后越人得其一二") == []
    assert [term for term, _ in extract_glossary_entries("心下痞")] == ["心下痞"]
    assert "古代计量表达" in companion
    assert build_plain_language_companion(companion) == companion
    assert strip_legacy_annotations("不可与（不可以用）（可以用）") == "不可与"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="待处理的古籍 Markdown")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--apply", action="store_true", help="写入伴随稿；默认仅预览")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("Self-test: OK")
        return

    sources = args.inputs or list(DEFAULT_CLASSICS)
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = (args.output_dir / source.name).resolve()
        if source == destination:
            raise ValueError("输出目录不能覆盖原始古籍")

        original = source.read_text(encoding="utf-8")
        companion = build_plain_language_companion(
            original,
            title=source.stem,
            max_chars=args.max_chars,
        )
        print(
            f"{source.name}: {len(original):,} source chars -> "
            f"{len(companion):,} companion chars"
        )
        if args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(companion, encoding="utf-8")
            print(f"  wrote {destination}")

    if not args.apply:
        print("DRY RUN: no files written. Re-run with --apply to generate companions.")


if __name__ == "__main__":
    main()
