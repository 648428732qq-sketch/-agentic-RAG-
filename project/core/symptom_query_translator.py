from __future__ import annotations

import re
import queue
import threading
import os
from typing import Any, Literal

import config
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.syndrome_terms import (
    SYMPTOM_ALIASES,
    SYMPTOM_TERMS,
    clean_text,
    expand_symptom_query,
    find_non_overlapping_aliases,
    unique,
)


QueryIntent = Literal[
    "clinical_symptom",
    "acupuncture_principle",
    "classical_theory",
    "herb_indication",
    "formula_knowledge",
    "unknown",
]


class QueryEvidenceMapping(BaseModel):
    source_phrase: str = Field(description="必须逐字来自用户原话的连续片段")
    source_start: int = Field(default=-1, description="source_phrase 在用户原话中的起始字符下标")
    source_end: int = Field(default=-1, description="source_phrase 在用户原话中的结束字符下标（不含）")
    canonical_term: str = Field(description="只能从系统提供的本地候选规范词中选择")
    polarity: Literal["present", "absent", "uncertain"] = "present"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SymptomQueryTranslation(BaseModel):
    query_intent: QueryIntent = "clinical_symptom"
    evidence_mappings: list[QueryEvidenceMapping] = Field(default_factory=list)
    canonical_terms: list[str] = Field(default_factory=list, description="兼容字段；最终仍以 evidence_mappings 校验结果为准")
    colloquial_terms: list[str] = Field(default_factory=list)
    body_parts: list[str] = Field(default_factory=list)
    pathogenesis_hints: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    time_modifiers: list[str] = Field(default_factory=list)
    severity_terms: list[str] = Field(default_factory=list)
    unknown_phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_more_info: bool = False


SYMPTOM_HINT_CHARS = (
    "疼", "痛", "咳", "喘", "痰", "汗", "冷", "热", "怕", "恶", "晕", "胀",
    "闷", "堵", "吐", "呕", "泻", "便", "尿", "痒", "肿", "麻", "烦", "渴",
    "酸", "乏", "抖", "跳", "烧", "僵", "睡", "食", "饿",
)
KNOWLEDGE_QUERY_MARKERS = (
    "组成", "用法", "剂量", "方解", "功用", "主治", "出处", "来源", "原文",
    "有哪些", "哪几", "是什么", "什么意思", "解释", "注意事项", "怎么理解",
)
THEORY_QUERY_MARKERS = (
    "脉络", "经络", "经脉", "搭脉", "号脉", "摸脉", "手腕",
    "理论", "为什么", "为何", "何谓", "经脉", "寸口", "号脉", "诊脉", "切脉", "脉法",
    "脏腑", "五脏", "六腑", "阴阳", "虚实", "难经",
)
ACUPUNCTURE_QUERY_MARKERS = (
    "针刺", "针灸", "针法", "扎针", "针扎", "下针", "灸法", "补泻", "迎随", "得气",
    "气至", "留针", "浅刺", "深刺", "九针", "穴位",
)
ACUPUNCTURE_PAIN_SIMILE_RE = re.compile(
    r"(?:像|如|跟|仿佛|好似).{0,5}针扎|针扎.{0,5}(?:似|一样|一般|般|感觉)"
)
HERB_QUERY_MARKERS = ("本草", "单味药", "药材", "性味", "哪味药", "哪些药")
GENERAL_CLINICAL_MARKERS = ("不舒服", "难受", "不对劲", "身体不适", "人不舒服")
PERSONAL_CLINICAL_RE = re.compile(
    r"(?:我|本人|患者).{0,18}(?:得了|患有|查出|诊断|不舒服|难受|怎么办|怎么治|如何治|吃什么|用什么)"
    r"|(?:怎么办|怎么治疗|如何治疗|吃什么药|用什么药|用什么方|开什么方)"
)
NEGATION_RE = re.compile(r"(?:完全|并|也|一直|从来|压根|绝对)?(?:不|没(?:有)?|无|未|勿)\s*([^，。；;、！？!?]{1,12})")
RHETORICAL_NEGATION_RE = re.compile(
    r"(?:(?:怎么|咋)(?:会|能|可能)|(?:哪|哪里|谁说)(?:就|还|会|能|可能|有)?)([^，。；;、！？!?呢啊呀]{1,10})(?:呢|啊|呀)?"
)
FORMULA_NAME_RE = re.compile(r"[\u3400-\u9fff\U00020000-\U0002ffff]{2,16}(?:汤|丸|散|饮|方)")
INTERVENTION_CANDIDATE_RE = re.compile(
    r"(?:主之|可与|宜用|方剂|处方|加减|剂量|[\u3400-\u9fff\U00020000-\U0002ffff]{1,12}(?:汤|丸|散|饮|方)$)"
)

NOISY_QUERY_REPLACEMENTS = (
    ("刻个不停", "咳个不停"),
    ("谈像清水", "痰像清水"),
    ("谈象清水", "痰像清水"),
    ("谈像清西", "痰像清稀"),
    ("瓶躺", "平躺"),
    ("平放", "平躺"),
    ("发张", "发胀"),
    ("吐玩", "吐完"),
    ("发扫", "发烧"),
    ("怕领", "怕冷"),
    ("咳嗦", "咳嗽"),
    ("经慢", "经脉"),
    ("正经", "经脉"),
    ("那出", "那处"),
    ("發燒", "发烧"),
    ("發熱", "发热"),
    ("風吹", "风吹"),
    ("過來", "过来"),
    ("難受", "难受"),
    ("一陣", "一阵"),
    ("無", "无"),
    ("淌細汗", "淌细汗"),
    ("筋络", "经络"),
    ("腕子", "手腕"),
    ("一吹峰难瘦", "一吹风难受"),
    ("一吹锋纳受", "一吹风难受"),
    ("一吹锋难受", "一吹风难受"),
    ("南受", "难受"),
    ("难收", "难受"),
    ("冒汉", "冒汗"),
    ("贸汗", "冒汗"),
    ("老冒汉", "老冒汗"),
    ("客扣", "咳嗽"),
    ("木有", "没有"),
    ("冇得", "没有"),
    ("没得", "没有"),
    ("一点汉", "一点汗"),
    ("半点汉", "半点汗"),
    ("貌不出来", "冒不出来"),
    ("茂不出来", "冒不出来"),
    ("冒不粗来", "冒不出来"),
    ("法紧", "发紧"),
    ("发进", "发紧"),
    ("涨疼", "胀疼"),
    ("发张疼", "发胀疼"),
    ("鼻赛", "鼻塞"),
    ("鼻封", "鼻塞"),
    ("个鼻", "鼻子"),
    ("赛住", "塞住"),
    ("冷疯", "冷风"),
    ("空条", "空调"),
    ("锅膛风", "过堂风"),
    ("果着", "裹着"),
    ("范吃", "饭吃"),
    ("不夏", "不下"),
    ("单卫药", "单味药"),
    ("固书", "古书"),
    ("青楚", "清楚"),
    ("苏腾", "酸疼"),
    ("酸腾", "酸疼"),
    ("饭厌不进去", "饭咽不进去"),
    ("咽不进去", "咽不下去"),
    ("腾疼", "疼"),
    ("围这儿", "胃这儿"),
    ("发涨右疼", "发胀又疼"),
    ("肚起", "肚脐"),
    ("一挑一挑", "一跳一跳"),
    ("手碗", "手腕"),
    ("经迈", "经脉"),
    ("呼脉", "号脉"),
    ("好脉", "号脉"),
    ("为神马", "为什么"),
    ("那一出", "那处"),
    ("合水", "喝水"),
    ("每有", "没有"),
    ("布急", "不急"),
    ("别憋", "不憋"),
    ("不串", "不喘"),
    ("平趟", "平躺"),
    ("痰象", "痰像"),
    ("淡像", "痰像"),
    ("克个不停", "咳个不停"),
    ("吐晚以后", "吐完以后"),
    ("一吹分", "一吹风"),
    ("难守", "难受"),
    ("打风吹", "风吹"),
    ("腕壳", "手腕"),
    ("手碗", "手腕"),
    ("咳嗖", "咳嗽"),
    ("咳嗦", "咳嗽"),
    ("冇咳", "没有咳嗽"),
    ("莫得咳", "没有咳嗽"),
    ("唔怕冻", "不怕冷"),
    ("怕冻", "怕冷"),
    ("望野", "看东西"),
    ("望嘢", "看东西"),
    ("清晰啲", "清楚点"),
    ("清出点", "清楚点"),
    ("明木", "明目"),
    ("眼紧", "眼睛"),
    ("痴不下", "吃不下"),
    ("痴", "吃"),
    ("吃不进去", "吃不下"),
    ("送不进嘴", "吃不下"),
    ("进不去", "不下"),
    ("突玩", "吐完"),
    ("趣", "去"),
    ("喝睡", "喝水"),
    ("酸马胀", "酸麻胀"),
    ("感脚", "感觉"),
    ("蒜气", "得气"),
    ("又腾", "又疼"),
    ("瞓落去", "躺下"),
    ("瞓落", "躺下"),
    ("训落去", "躺下"),
    ("涌返上嚟", "往上返"),
    ("涌返上黎", "往上返"),
    ("饮嘢", "喝水"),
)

NEGATION_INTENSIFIER_PREFIXES = (
    "不停",
    "不住",
    "不由自主",
    "不间断",
    "没完没了",
)

REGEX_TERM_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"(?:十二条|十二).{0,4}(?:正经|经脉|经络)"), ("十二经", "经络")),
    (re.compile(r"(?:满身|身上|全身).{0,8}(?:经络|经脉)|(?:经络|经脉).{0,8}(?:多条|哪条|那么多)"), ("十二经", "经络")),
    (re.compile(r"(?:号脉|把脉|诊脉|切脉|摸脉).{0,28}(?:手腕|手腕子|腕内|腕里|腕横纹).{0,12}(?:那一处|一处|那处|那儿|那块|那个位置|那点位置|那点|一小块|寸半截|寸半截地儿)"), ("寸口",)),
    (re.compile(r"(?:手腕|手腕子|腕内|腕里|腕横纹).{0,12}(?:一小块|那块|那个位置|那点位置|寸地|寸口|搭脉|摸脉|那头|那处|那儿)"), ("寸口",)),
    (re.compile(r"(?:脑袋|脑门|脑门子|后脑勺|头).{0,18}(?:发紧|紧绷|绷得|胀疼|胀痛|发沉|勒着|抽着疼|揪痛|针扎|炸裂)"), ("头痛",)),
    (re.compile(r"(?:后脑勺|脑袋|脑门|头).{0,24}(?:扯得慌|勒了|勒着|绷紧)"), ("头痛",)),
    (re.compile(r"(?:风|冷风|穿堂风|过堂风).{0,18}(?:难受|不舒服|扛不住|遭殃|不自在|发毛|遭不住|不得行|不得劲|挺不得劲|往身上钻|躲着)"), ("恶风",)),
    (re.compile(r"(?:怕|害怕|受不了|遭不住|扛不住|挺不住|不得劲).{0,10}(?:空调冷风|空调风|冷风|风吹|吹风|过堂风|穿堂风)"), ("恶风",)),
    (re.compile(r"(?:冻得|冷得|透骨|裹着|厚棉被|棉被).{0,18}(?:冷|冻|打颤|打抖|打哆嗦)"), ("恶寒",)),
    (re.compile(r"(?:周身|浑身|身上|全身|身体).{0,12}(?:酸痛|酸疼|酸胀|酸麻|灌了铅|沉重)"), ("身疼",)),
    (re.compile(r"(?:一滴汗|汗星子|汗)(?![^。！？,，；;]{0,12}没完没了).{0,8}(?:冇|没(?!完)|无).{0,6}(?:出|冒)"), ("无汗",)),
    (re.compile(r"(?:没见|没看到|没(?!完)|没有|无|未|不(?!停|住|下来)).{0,6}(?:出汗|汗出|汗出来|汗冒出来|汗)"), ("无汗",)),
    (re.compile(r"(?:汗星子|一滴汗|半点汗|一点汗).{0,12}(?:冒不出来|挤不出来|逼不出来|憋不出来|出不来|都没|没有)"), ("无汗",)),
    (re.compile(r"(?:钝疼|绞痛|阵痛|阵阵疼|阵阵痛)"), ("疼痛",)),
    (re.compile(r"(?:想|要|狂想|直想|拼了命想).{0,12}(?:喝水|合水|灌水|灌口水|接水|接凉水)|(?:嘴里|口里|嗓子|喉咙).{0,12}(?:干|着火|冒火|冒烟).{0,12}(?:水|喝|灌)"), ("渴欲得水", "渴欲饮水", "口渴", "欲饮水")),
    (re.compile(r"(?:鼻孔|鼻子|鼻腔|鼻).{0,14}(?:封|堵|塞|透不进气|不通|闷住|憋得死紧|连不上风|喘不过气)"), ("鼻塞",)),
    (re.compile(r"(?:胃|胃脘|心口|上腹|肚|腹).{0,18}(?:疼|痛|绞|揪|钝疼|阵痛|隐痛)|(?:疼|痛|绞|揪|钝疼|阵痛).{0,18}(?:胃|胃脘|心口|上腹|肚|腹)"), ("胃脘当心而痛", "疼痛", "腹痛")),
    (re.compile(r"(?:多条|好多条|许多条|噶多条|全身|身上|周身).{0,10}(?:脉络|经络|经脉)"), ("十二经", "经络")),
    (re.compile(r"(?:胃|肚|腹|心口|心口窝|上腹).{0,8}(?:胀|涨|滞|撑|鼓|堵)"), ("腹䐜胀", "腹胀满")),
    (re.compile(r"(?:胃|心口|心口窝|上腹).{0,10}(?:疼|痛|抽|刺)"), ("胃脘当心而痛",)),
    (re.compile(r"(?:吃|食|饭|饭菜|食物|咽|吞|茶饭).{0,8}(?:不下|下不去|不进|进不去|进不下|吃不下|咽不下|吞不下|落不下|食唔落|咽勿下|塞不进|送不进)"), ("食饮不下",)),
    (re.compile(r"(?:小便|尿).{0,8}(?:不通|不畅|不利|排不出|尿不出)"), ("小便不通", "小便不利")),
    (re.compile(r"(?:肚脐|脐下|小腹).{0,8}(?:跳|扑通|突突|打跳|敲小鼓|一跳一跳)"), ("脐下悸", "脐下有悸")),
    (re.compile(r"(?:气|一股气|有股气|那股气|股气|条气).{0,12}(?:小腹|小肚子|下肚|下面).{0,12}(?:胸|心口|心窝|上).{0,6}(?:冲|窜|串|顶|拱)"), ("气上冲", "欲作奔豚")),
    (re.compile(r"(?:气|一股气|有股气|那股气|股气|条气).{0,16}(?:小腹|小肚子|下肚|下面).{0,12}(?:冲|窜|串|顶|拱).{0,10}(?:胸|心口|心窝|上)"), ("气上冲", "欲作奔豚")),
    (re.compile(r"(?:咳|克|嗓子|喉).{0,10}(?:不停|没完|连轴|拉风箱|刹弗停)"), ("咳嗽",)),
    (re.compile(r"(?:痰|痰水|吐出来的).{0,10}(?:清水|清汤|凉水|白开水|自来水|水灵灵)"), ("痰涎清稀",)),
    (re.compile(r"(?:平躺|平趟|躺下|一躺|一瞓|瞓落|沾枕头|往床上一躺).{0,12}(?:憋|喘|闷醒|憋醒|没法躺|平不起睡|喊醒)"), ("不得平卧", "喘")),
    (re.compile(r"(?:吃|饭|饭菜|食|咽|吞|下肚|过嘴|一落肚).{0,12}(?:反上|往上反|往上返|格出来|吐|呕|翻涌上来|翻上来|涌返上)"), ("食已即吐", "食入口即吐", "胃反")),
    (re.compile(r"(?:吐完|吐后|吐过后|吐巴了|吐腾|吐利索|吐脱末|呕脱末|呕出来|呕完|全呕出来)"), ("吐后",)),
    (re.compile(r"(?:眼|眼睛|眼目|眼皮|视线|视物).{0,20}(?:看清|看不清|看东西更清楚|看东西清楚|视物不清|显影不清|擦亮|透亮|雾水|雾|朦胧|模糊|散尽|分明|清楚)"), ("明目",)),
    (re.compile(r"(?:身上|身体|浑身).{0,6}(?:发烫|发热|燥热)|(?:发烧|烧起来|一点都不烧|单是发烧)"), ("发热",)),
    (re.compile(r"(?:一点|半点|一丝|滴|汗星子).{0,8}(?:汗|汗星子).{0,8}(?:没(?!完)|没有|冒不出来|挤不出来|逼不出来|憋不出来|出不来)|(?:一点汗都没出过)"), ("无汗",)),
    (re.compile(r"(?:冒汗|流汗|淌汗|出汗|渗汗|汗珠子|汗一直|一直冒汗|一直流个不停|往外渗汗)"), ("汗出",)),
)

LEXICALIZED_PRESENT_TERMS = {
    "无汗",
    "不寐",
    "不得卧",
    "不得平卧",
    "不渴",
    "不欲食",
    "食饮不下",
    "吞咽不下",
    "膈咽不通",
}

SWEAT_PRESENT_TERMS = {"汗出", "出汗", "有汗", "自汗", "多汗", "汗多", "冒汗", "流汗", "淌汗", "渗汗", "汗大出"}
SWEAT_ABSENT_MARKERS = ("无汗", "不出汗", "没有汗", "没汗", "没见出汗", "汗不出", "汗出不来")
SWEAT_PRESENT_MARKERS = ("汗出", "出汗", "有汗", "自汗", "多汗", "汗多", "冒汗", "流汗", "淌汗", "渗汗")
SWEAT_ABSENT_SURFACE_RE = re.compile(
    r"(?:没见|没看到|没(?!完)|没有|无|未|不(?!停|住|下来|间断|由自主)|"
    r"(?:憋|逼|挤|捂|冒|流|淌|渗)不)[^，,、。！？；;：:\n]{0,10}(?:汗珠子|汗星子|汗)|"
    r"(?:汗珠子|汗星子|一滴汗|半点汗|一点汗|汗)[^，,、。！？；;：:\n]{0,12}"
    r"(?:冒不出|挤不出|逼不出|憋不出|捂不出|出不来|都没|没有)"
)
SWEAT_PRESENT_SURFACE_RE = re.compile(
    r"(?:多汗|汗多|自汗|老冒汗|总冒汗|一直冒汗|冒汗|流汗|淌汗|渗汗|往外渗汗|汗珠子|"
    r"(?<!没有)(?<!没)(?<!无)(?<!不)(?<!未)(?:出汗|汗出|有汗)(?![不没未]))"
)
THIRST_ABSENT_MARKERS = ("不渴", "不口渴", "没有口渴", "没口渴", "不觉得口渴")
THIRST_PRESENT_MARKERS = ("口渴", "口大渴", "烦渴引饮", "想喝水", "想饮水", "要喝水")
CONTRAST_MARKERS = ("但是", "但", "可是", "却", "反而", "同时", "又")

PRESENT_OVERRIDE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "汗出": (
        re.compile(r"(?:不停|没完没了|停不下来|止不住).{0,8}(?:冒汗|出汗|流汗|淌汗|渗汗|汗)"),
        re.compile(r"(?:汗|渗汗).{0,8}(?:没完没了|停不下来|止不住|往外冒|往外渗)"),
        re.compile(r"(?:平时|就|直|一直|老|总|还|仍).{0,4}(?:冒汗|出汗|流汗|淌汗|渗汗)"),
    ),
    "明目": (
        re.compile(r"(?:能不能|能否|可不可以).{0,12}(?:看清|擦亮|透亮)"),
    ),
}

TERM_EQUIVALENTS: dict[str, list[str]] = {
    "身热": ["发热"],
    "大热": ["发热"],
    "皮肤热": ["发热"],
    "骨节疼痛": ["身疼", "头身疼痛", "肢体酸楚疼痛"],
    "身体痛": ["身疼"],
    "身疼": ["肢体酸楚疼痛", "头身疼痛"],
    "四肢酸疼": ["身疼", "肢体酸楚疼痛"],
    "青盲": ["明目"],
    "眚盲": ["明目"],
    "目盲无所见": ["明目"],
    "目瞑": ["明目"],
    "脉诊": ["脉"],
    "经络": ["十二经"],
    "腹䐜胀": ["腹胀满"],
    "腹胀满": ["腹䐜胀"],
    "胃脘": ["胃脘当心而痛"],
    "胃脘痞闷": ["胃脘当心而痛"],
    "心下痞": ["腹胀满"],
    "心下痞硬": ["腹胀满"],
    "吞咽不下": ["食饮不下"],
    "口干": ["口渴"],
    "欲饮水": ["渴欲饮水", "渴欲得水"],
    "渴欲饮水": ["欲饮水", "渴欲得水"],
    "食入口即吐": ["食已即吐", "胃反"],
    "胃反": ["食已即吐", "食入口即吐"],
    "上气": ["喘"],
    "短气": ["喘"],
    "胸满": ["喘"],
}

NEGATIVE_TERM_EQUIVALENTS: dict[str, list[str]] = {
    "恶风": ["恶寒"],
    "身热": ["发热"],
    "大热": ["发热", "高热"],
    "发热": ["高热"],
    "胸痞": ["胸闷"],
    "胸中窒": ["胸闷"],
    "胸脘痞闷": ["胸闷"],
}

NEGATIVE_SURFACE_PATTERNS: dict[str, tuple[str, ...]] = {
    "喘": (
        "呼吸不急", "呼吸完全不急", "不憋", "不喘", "不串",
        "气儿顺畅", "气儿顺溜", "急促憋闷压根没沾过边",
        "喘不上气没沾过边", "气路好顺", "完全唔喘", "呼吸平平顺顺",
        "勿急", "勿窒", "不急", "不窒", "喘象也无", "喘气儿老稳当",
        "既不急促也不觉得憋闷", "这就叫喘吗",
        "气儿走得匀实", "半点儿喘促的迹象都没有", "半点喘促的迹象都没有",
    ),
    "发热": (
        "不烫", "一点热都没有", "发热是一丁点没有", "身上倒是不烫", "勿发热",
        "没有发热", "压根没有发热", "发热都没有", "一点儿发热都没有",
        "一点发热都没有", "一点发烧也无", "发烧也无", "没有烧起", "没烧起",
        "没有火气", "一点火气都没有",
    ),
    "恶寒": ("不怕冷", "一点都不怕冷", "一点寒气都不怕", "不觉得冷", "勿觉得挂风", "哪有一点怕冷"),
    "咳嗽": (
        "没有咳嗽", "没咳嗽", "咳嗽的影子都找不着", "一声咳嗽都没",
        "一点咳嗽的迹象都找不着", "咳嗽更是一点都没有", "咳嗽更是想都没想过",
        "咳嗽更是完全没有", "更别提咳嗽", "连半声咳嗽", "咳声也呒没",
        "咳嗽更是完全没有的事", "咳嗽更是一点都没有", "嗓子里连咳嗽",
        "不咳嗽", "并不咳嗽", "半点咳嗽", "半点儿咳嗽", "一点咳嗽都没有",
        "一点咳嗽", "没有咳", "冇咳", "连咳嗽都没有",
    ),
    "口渴": ("不口渴", "并不口渴", "不觉得口渴", "没有口渴", "没口渴"),
    "头晕": ("不头晕", "没有头晕", "没头晕", "并不头晕"),
    "胸闷": ("没有胸闷", "没胸闷", "不胸闷", "并不胸闷"),
    "高热": ("没有高热", "没高热", "不高热", "并不高热"),
}

DIRECT_CANDIDATE_TERM_STOPWORDS = {
    "中药",
    "方剂",
    "药方",
    "处方",
    "治疗",
    "调理",
    "推荐",
    "建议",
    "症状",
    "疾病",
    "患者",
    "其他症状",
    "饮食",
}


def normalize_noisy_query(query: str) -> str:
    normalized = clean_text(query)
    for source, target in NOISY_QUERY_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return normalized


def infer_direct_candidate_terms(query: str, candidate_terms: list[str] | None = None) -> list[str]:
    normalized_query = normalize_noisy_query(query)
    terms: list[str] = []
    for term in sorted(unique(candidate_terms or []), key=len, reverse=True):
        if term in DIRECT_CANDIDATE_TERM_STOPWORDS:
            continue
        if not (1 < len(term) <= 16):
            continue
        if INTERVENTION_CANDIDATE_RE.search(term):
            continue
        if term in normalized_query:
            terms.append(term)
    return unique(terms)


def infer_regex_terms(query: str) -> list[str]:
    terms: list[str] = []
    for pattern, canonical_terms in REGEX_TERM_RULES:
        if pattern.search(query):
            terms.extend(canonical_terms)
    return unique(terms)


def _is_lexicalized_present_term(term: str) -> bool:
    return term in LEXICALIZED_PRESENT_TERMS or term.endswith(("不下", "不通")) or term.startswith(("无", "不", "未", "不得"))


def _is_candidate_lexicalized_present_term(term: str) -> bool:
    return term in LEXICALIZED_PRESENT_TERMS or term.endswith(("不下", "不通")) or term.startswith(("无", "不得"))


def _has_present_override(term: str, query: str) -> bool:
    return any(pattern.search(query) for pattern in PRESENT_OVERRIDE_PATTERNS.get(term, ()))


def _augment_equivalent_terms(terms: list[str], allowed_terms: set[str]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        for equivalent in TERM_EQUIVALENTS.get(term, []):
            if equivalent in allowed_terms:
                expanded.append(equivalent)
    return unique(expanded)


def _augment_negative_terms(terms: list[str], allowed_terms: set[str]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        for equivalent in NEGATIVE_TERM_EQUIVALENTS.get(term, []):
            if equivalent in allowed_terms:
                expanded.append(equivalent)
    return unique(expanded)


def _normalize_alias_terms_to_allowed_canonicals(terms: list[str], allowed_terms: set[str]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        if term in SYMPTOM_ALIASES:
            continue
        for canonical, aliases in SYMPTOM_ALIASES.items():
            if canonical not in allowed_terms or canonical == term:
                continue
            if term in aliases:
                expanded.append(canonical)
    return unique(expanded)


def _apply_mutual_exclusions(
    positive_terms: list[str],
    primary_positive_terms: list[str],
    negative_terms: list[str],
    query: str = "",
) -> tuple[list[str], list[str], list[str]]:
    sweat_polarity_surface = query
    for intensifier in NEGATION_INTENSIFIER_PREFIXES:
        sweat_polarity_surface = sweat_polarity_surface.replace(intensifier, "")
    has_sweat_absent_surface = bool(SWEAT_ABSENT_SURFACE_RE.search(sweat_polarity_surface))
    has_sweat_present_surface = bool(SWEAT_PRESENT_SURFACE_RE.search(query))
    has_explicit_sweat_contradiction = (
        has_sweat_absent_surface
        and has_sweat_present_surface
        and any(marker in query for marker in CONTRAST_MARKERS)
    )
    if "无汗" in positive_terms and not has_explicit_sweat_contradiction:
        if has_sweat_present_surface and not has_sweat_absent_surface:
            positive_terms = [term for term in positive_terms if term != "无汗"]
            primary_positive_terms = [term for term in primary_positive_terms if term != "无汗"]
            negative_terms = [term for term in negative_terms if term != "无汗"]
        else:
            positive_terms = [term for term in positive_terms if term not in SWEAT_PRESENT_TERMS]
            primary_positive_terms = [term for term in primary_positive_terms if term not in SWEAT_PRESENT_TERMS]
            negative_terms = [term for term in negative_terms if term not in SWEAT_PRESENT_TERMS]
    return positive_terms, primary_positive_terms, negative_terms


def _has_explicit_thirst_contradiction(query: str) -> bool:
    return (
        any(marker in query for marker in THIRST_ABSENT_MARKERS)
        and any(marker in query for marker in THIRST_PRESENT_MARKERS)
        and any(marker in query for marker in CONTRAST_MARKERS)
    )


def infer_query_intent(query: str) -> QueryIntent:
    text = clean_text(query)
    if PERSONAL_CLINICAL_RE.search(text):
        return "clinical_symptom"
    acupuncture_markers = [marker for marker in ACUPUNCTURE_QUERY_MARKERS if marker != "针扎"]
    has_acupuncture_marker = any(marker in text for marker in acupuncture_markers)
    has_literal_needle_action = "针扎" in text and not ACUPUNCTURE_PAIN_SIMILE_RE.search(text)
    if has_acupuncture_marker or has_literal_needle_action:
        return "acupuncture_principle"
    if any(marker in text for marker in THEORY_QUERY_MARKERS):
        return "classical_theory"
    if FORMULA_NAME_RE.search(text) and any(marker in text for marker in KNOWLEDGE_QUERY_MARKERS + ("药", "放", "里面")):
        return "formula_knowledge"
    if any(marker in text for marker in HERB_QUERY_MARKERS):
        return "herb_indication"
    if any(marker in text for marker in GENERAL_CLINICAL_MARKERS):
        return "clinical_symptom"
    if any(ch in text for ch in SYMPTOM_HINT_CHARS):
        return "clinical_symptom"
    return "unknown"


def looks_like_symptom_query(query: str) -> bool:
    return infer_query_intent(query) in {
        "clinical_symptom",
        "acupuncture_principle",
        "classical_theory",
        "herb_indication",
    }


def _surface_to_canonical(surface: str, candidate_terms: list[str] | None = None) -> list[str]:
    info = expand_symptom_query(normalize_noisy_query(surface))
    return unique(list(info.get("canonical_terms", [])) + infer_direct_candidate_terms(surface, candidate_terms))


def extract_negated_terms(query: str, candidate_terms: list[str] | None = None) -> tuple[list[str], list[str]]:
    negative_terms: list[str] = []
    negative_phrases: list[str] = []
    normalized_query = normalize_noisy_query(query)
    for match in NEGATION_RE.finditer(normalized_query):
        negated_span = clean_text(match.group(0))
        if match.start() > 0 and normalized_query[match.start() - 1] in {"能", "可", "有"}:
            continue
        if (
            match.start() > 0
            and normalized_query[match.start() - 1] == "是"
            and normalized_query[match.start(): match.start() + 2] == "不是"
        ):
            continue
        if (
            negated_span.startswith(NEGATION_INTENSIFIER_PREFIXES)
            or "不停" in negated_span[:8]
            or "没完没了" in negated_span[:8]
            or "停不下来" in negated_span[:8]
            or (negated_span.startswith("不住") and match.start() > 0 and normalized_query[match.start() - 1] == "止")
        ):
            continue
        if match.start() > 0 and normalized_query[match.start() - 1] in {"莫", "岂"}:
            continue
        lexicalized_absence = any(
            _is_lexicalized_present_term(canonical)
            for canonical, _ in find_non_overlapping_aliases(clean_text(match.group(0)))
        )
        if not lexicalized_absence:
            context_span = normalized_query[max(0, match.start() - 4): match.end()]
            lexicalized_absence = any(
                term in context_span and _is_candidate_lexicalized_present_term(term)
                for term in unique(candidate_terms or [])
            )
        if lexicalized_absence:
            continue
        surface = clean_text(match.group(1))
        surface = re.split(r"(?:但是|只是|就是|而且|还|并且|也|就|是|吗)", surface, maxsplit=1)[0]
        surface = clean_text(surface)
        if not surface:
            continue
        mapped_terms = _surface_to_canonical(surface, candidate_terms)
        if not mapped_terms:
            continue
        negative_phrases.append(surface)
        negative_terms.extend(mapped_terms)
    for match in RHETORICAL_NEGATION_RE.finditer(normalized_query):
        if normalized_query[match.start(): match.start() + 2] in {"哪些", "哪几", "哪个", "哪味"}:
            continue
        surface = clean_text(match.group(1))
        mapped_terms = _surface_to_canonical(surface, candidate_terms)
        if not mapped_terms:
            continue
        negative_phrases.append(surface)
        negative_terms.extend(mapped_terms)
    for canonical, phrases in NEGATIVE_SURFACE_PATTERNS.items():
        for phrase in phrases:
            if phrase in normalized_query:
                negative_phrases.append(phrase)
                negative_terms.append(canonical)
    negative_terms = [
        term
        for term in unique(negative_terms)
        if not _has_present_override(term, normalized_query)
    ]
    return negative_terms, unique(negative_phrases)


def _evidence_in_query(source_phrase: str, query: str) -> bool:
    phrase = re.sub(r"\s+", "", clean_text(source_phrase))
    normalized_query = re.sub(r"\s+", "", clean_text(query))
    return bool(phrase) and phrase in normalized_query


def _evidence_from_mapping(mapping: QueryEvidenceMapping, query: str) -> str:
    normalized_query = clean_text(query)
    source_phrase = clean_text(mapping.source_phrase)
    if _evidence_in_query(source_phrase, normalized_query):
        return source_phrase
    start = mapping.source_start
    end = mapping.source_end
    if 0 <= start < end <= len(normalized_query):
        return clean_text(normalized_query[start:end])
    return ""


def _invoke_with_hard_timeout(callable_, timeout_seconds: float):
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, callable_()))
        except Exception as exc:  # pragma: no cover - provider-specific errors
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, name="query-translator-llm", daemon=True)
    thread.start()
    if os.environ.get("QUERY_TRANSLATOR_DEBUG") == "1":
        print(f"[query-translator] llm:start timeout={timeout_seconds:g}s", flush=True)
    debug_file = os.environ.get("QUERY_TRANSLATOR_DEBUG_FILE")
    if debug_file:
        with open(debug_file, "a", encoding="utf-8") as handle:
            handle.write(f"query-translator llm:start timeout={timeout_seconds:g}s\n")
    try:
        succeeded, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        if os.environ.get("QUERY_TRANSLATOR_DEBUG") == "1":
            print("[query-translator] llm:hard-timeout", flush=True)
        if debug_file:
            with open(debug_file, "a", encoding="utf-8") as handle:
                handle.write("query-translator llm:hard-timeout\n")
        raise TimeoutError(f"query translator exceeded {timeout_seconds:g}s") from exc
    if os.environ.get("QUERY_TRANSLATOR_DEBUG") == "1":
        print(f"[query-translator] llm:returned success={succeeded}", flush=True)
    if succeeded:
        return value
    raise value


def _build_expanded_query(data: dict[str, Any]) -> str:
    parts = [
        "用户原话 " + str(data.get("original_query", "")),
        "查询意图 " + str(data.get("query_intent", "")),
        "标准中医表征 " + " ".join(data.get("canonical_terms", [])),
        "口语证据 " + " ".join(data.get("colloquial_terms", [])),
        "部位 " + " ".join(data.get("body_parts", [])),
        "病因病机线索 " + " ".join(data.get("pathogenesis_hints", [])),
        "明确否认 " + " ".join(data.get("negative_terms", [])),
    ]
    return clean_text("\n".join(part for part in parts if part.strip()))


def _local_translation(query: str, candidate_terms: list[str] | None = None) -> dict[str, Any]:
    normalized_query = normalize_noisy_query(query)
    local_info = expand_symptom_query(normalized_query)
    direct_candidate_terms = infer_direct_candidate_terms(normalized_query, candidate_terms)
    if direct_candidate_terms:
        local_info["canonical_terms"] = unique(list(local_info.get("canonical_terms", [])) + direct_candidate_terms)
    regex_terms = infer_regex_terms(normalized_query)
    if regex_terms:
        local_info["canonical_terms"] = unique(list(local_info.get("canonical_terms", [])) + regex_terms)
    negative_terms, negative_phrases = extract_negated_terms(query, candidate_terms)
    allowed_terms = (
        set(candidate_terms or [])
        | set(SYMPTOM_TERMS)
        | set(local_info.get("canonical_terms", []))
        | set(negative_terms)
    )
    negative_terms = _normalize_alias_terms_to_allowed_canonicals(negative_terms, allowed_terms)
    negative_terms = _augment_equivalent_terms(negative_terms, allowed_terms)
    negative_terms = _augment_negative_terms(negative_terms, allowed_terms)
    positive_terms = [term for term in local_info.get("canonical_terms", []) if term not in negative_terms]
    colloquial_evidence_terms = set(local_info.get("colloquial_terms", []))
    primary_positive_terms = unique([term for term in positive_terms if term not in colloquial_evidence_terms])
    if not primary_positive_terms and positive_terms:
        primary_positive_terms = unique(positive_terms)
    primary_positive_terms = _normalize_alias_terms_to_allowed_canonicals(primary_positive_terms, allowed_terms)
    positive_terms = _augment_equivalent_terms(primary_positive_terms, allowed_terms)
    if any(term in positive_terms for term in ("不渴", "不口渴")) and not _has_explicit_thirst_contradiction(normalized_query):
        negative_terms = unique(negative_terms + ["口渴"])
    positive_terms = [term for term in positive_terms if term not in negative_terms]
    positive_terms, primary_positive_terms, negative_terms = _apply_mutual_exclusions(
        positive_terms,
        primary_positive_terms,
        negative_terms,
        normalized_query,
    )
    intent = infer_query_intent(normalized_query)
    if positive_terms and intent in {"unknown", "formula_knowledge"} and any(
        marker in normalized_query
        for marker in (
            "患者",
            "目前有",
            "表现为",
            "症状为",
            "能确认的表现",
            "已知症状",
            "我感觉",
            "我最近",
            "我的",
            "患者出现",
            "如何判断",
            "典型症状",
            "体现",
            "病因病机",
            "怎么治疗",
            "如何治疗",
            "怎么治",
            "怎么办",
            "推荐",
        )
    ):
        intent = "clinical_symptom"
    if any(phrase in normalized_query for phrase in ("气儿顺", "气好顺", "气路好顺")):
        positive_terms = [term for term in positive_terms if term not in {"得气", "气至"}]
        primary_positive_terms = [term for term in primary_positive_terms if term not in {"得气", "气至"}]
    result: dict[str, Any] = {
        "original_query": clean_text(query),
        "normalized_query": normalized_query,
        "query_intent": intent,
        "canonical_terms": unique(positive_terms),
        "primary_canonical_terms": primary_positive_terms,
        "colloquial_terms": unique(list(local_info.get("colloquial_terms", []))),
        "body_parts": [],
        "pathogenesis_hints": [],
        "negative_terms": negative_terms,
        "negative_phrases": negative_phrases,
        "time_modifiers": [],
        "severity_terms": [],
        "unknown_phrases": [],
        "confidence": 1.0 if positive_terms else 0.0,
        "translation_method": "local_alias",
        "needs_more_info": intent == "clinical_symptom" and not positive_terms,
        "evidence_mappings": [],
        "candidate_terms": unique(candidate_terms or []),
        "translation_errors": [],
        "llm_used": False,
    }
    result["expanded_query"] = _build_expanded_query(result)
    return result


def _merge_grounded_translation(
    query: str,
    local: dict[str, Any],
    llm_info: SymptomQueryTranslation,
    candidate_terms: list[str],
) -> dict[str, Any]:
    allowed_terms = set(candidate_terms) | set(local.get("canonical_terms", [])) | set(local.get("negative_terms", []))
    positive_terms = list(local.get("canonical_terms", []))
    primary_positive_terms = list(local.get("primary_canonical_terms", local.get("canonical_terms", [])))
    negative_terms = list(local.get("negative_terms", []))
    colloquial_terms = list(local.get("colloquial_terms", []))
    accepted_mappings: list[dict[str, Any]] = []
    errors: list[str] = []
    present_evidence_sources: set[str] = set()

    for mapping in llm_info.evidence_mappings:
        source_phrase = _evidence_from_mapping(mapping, query)
        canonical_term = clean_text(mapping.canonical_term)
        polarity = mapping.polarity
        if not source_phrase:
            errors.append(f"rejected_non_evidence:{canonical_term}:{clean_text(mapping.source_phrase)}")
            continue
        if polarity == "absent" and _is_lexicalized_present_term(canonical_term):
            polarity = "present"
            errors.append(f"normalized_lexicalized_absence:{canonical_term}")
        if polarity == "absent":
            narrower_terms = [
                term
                for term in allowed_terms
                if term != canonical_term and term in canonical_term and len(term) < len(canonical_term)
            ]
            if len(narrower_terms) == 1:
                errors.append(f"narrowed_negative_compound:{canonical_term}:{narrower_terms[0]}")
                canonical_term = narrower_terms[0]
        if canonical_term not in allowed_terms:
            errors.append(f"rejected_outside_local_candidates:{canonical_term}")
            continue
        if canonical_term == "寸口" and not any(marker in source_phrase for marker in ("寸口", "手腕", "手颈")):
            errors.append(f"rejected_vague_cunkou_evidence:{source_phrase}")
            continue
        if canonical_term == "寸口" and (
            source_phrase in {"手腕那一处", "某一个特定位置"}
            or "某一个特定位置" in source_phrase
        ):
            errors.append(f"rejected_vague_cunkou_evidence:{source_phrase}")
            continue
        if canonical_term == "咳逆上气" and not any(marker in source_phrase for marker in ("喘", "上气", "气急", "气促", "憋")):
            errors.append(f"rejected_over_specific_cough_upward_qi:{source_phrase}")
            continue
        min_confidence = float(
            getattr(config, "LLM_SYMPTOM_TRANSLATOR_MIN_MAPPING_CONFIDENCE", 0.72)
        )
        if mapping.confidence < min_confidence:
            errors.append(f"rejected_low_confidence:{canonical_term}:{mapping.confidence:.3f}")
            continue
        accepted_mapping = mapping.model_dump(mode="json")
        accepted_mapping["source_phrase"] = source_phrase
        accepted_mapping["canonical_term"] = canonical_term
        accepted_mapping["polarity"] = polarity
        accepted_mappings.append(accepted_mapping)
        colloquial_terms.append(source_phrase)
        if polarity == "present":
            positive_terms.append(canonical_term)
            # Multiple local canonical terms can legitimately describe the same
            # colloquial phrase. Count the phrase once for gate coverage, while
            # still keeping all accepted terms for retrieval recall.
            if source_phrase not in present_evidence_sources:
                primary_positive_terms.append(canonical_term)
                present_evidence_sources.add(source_phrase)
        elif polarity == "absent":
            negative_terms.append(canonical_term)

    negative_terms = _normalize_alias_terms_to_allowed_canonicals(
        unique(negative_terms),
        allowed_terms | set(negative_terms),
    )
    negative_terms = _augment_equivalent_terms(unique(negative_terms), allowed_terms | set(negative_terms))
    negative_terms = _augment_negative_terms(negative_terms, allowed_terms | set(negative_terms))
    primary_positive_terms = unique(primary_positive_terms)
    positive_terms = _normalize_alias_terms_to_allowed_canonicals(
        unique(positive_terms),
        allowed_terms | set(positive_terms),
    )
    positive_terms = _augment_equivalent_terms(unique(positive_terms), allowed_terms | set(positive_terms))
    normalized_query = normalize_noisy_query(query)
    if any(term in positive_terms for term in ("不渴", "不口渴")) and not _has_explicit_thirst_contradiction(normalized_query):
        negative_terms = unique(negative_terms + ["口渴"])
    positive_terms, primary_positive_terms, negative_terms = _apply_mutual_exclusions(
        positive_terms,
        primary_positive_terms,
        negative_terms,
        normalized_query,
    )
    negative_terms = [
        term
        for term in negative_terms
        if not (
            term in positive_terms
            and (
                _is_lexicalized_present_term(term)
                or (term in {"寸口", "明目"} and term in normalized_query)
                or _has_present_override(term, normalized_query)
            )
        )
    ]
    positive_terms = [term for term in positive_terms if term not in negative_terms]
    primary_positive_terms = [term for term in primary_positive_terms if term not in negative_terms]
    unknown_phrases = [
        clean_text(phrase)
        for phrase in llm_info.unknown_phrases
        if _evidence_in_query(phrase, query)
    ]
    intent = local["query_intent"]
    if intent == "unknown" and llm_info.query_intent in {
        "clinical_symptom",
        "acupuncture_principle",
        "classical_theory",
        "herb_indication",
        "formula_knowledge",
    }:
        intent = llm_info.query_intent
    grounded_pathogenesis = [
        hint
        for hint in unique(llm_info.pathogenesis_hints)
        if hint in positive_terms
    ]
    if any(phrase in normalized_query for phrase in ("气儿顺", "气好顺", "气路好顺")):
        positive_terms = [term for term in positive_terms if term not in {"得气", "气至"}]
        primary_positive_terms = [term for term in primary_positive_terms if term not in {"得气", "气至"}]
    result = dict(local)
    result.update(
        {
            "query_intent": intent,
            "canonical_terms": positive_terms,
            "primary_canonical_terms": primary_positive_terms or positive_terms,
            "colloquial_terms": unique(colloquial_terms),
            "body_parts": unique([value for value in llm_info.body_parts if _evidence_in_query(value, query)]),
            "pathogenesis_hints": grounded_pathogenesis,
            "negative_terms": negative_terms,
            "time_modifiers": unique(
                [value for value in llm_info.time_modifiers if _evidence_in_query(value, query)]
            ),
            "severity_terms": unique(
                [value for value in llm_info.severity_terms if _evidence_in_query(value, query)]
            ),
            "unknown_phrases": unique(unknown_phrases),
            "confidence": llm_info.confidence,
            "translation_method": "llm_structured",
            "needs_more_info": bool(llm_info.needs_more_info) or (intent == "clinical_symptom" and not positive_terms),
            "evidence_mappings": accepted_mappings,
            "translation_errors": errors,
            "llm_used": True,
        }
    )
    result["expanded_query"] = _build_expanded_query(result)
    return result


def translate_symptom_query(
    query: str,
    llm=None,
    force_llm: bool = False,
    candidate_terms: list[str] | None = None,
) -> dict[str, Any]:
    candidates = [
        term
        for term in unique(candidate_terms or [])
        if (1 < len(term) <= 16 or term in SYMPTOM_TERMS) and not INTERVENTION_CANDIDATE_RE.search(term)
    ][: getattr(config, "SYNDROME_TRANSLATOR_MAX_TERMS", 80)]
    local = _local_translation(query, candidates)
    if llm is None or not candidates:
        return local
    if not force_llm and not looks_like_symptom_query(query):
        return local

    candidate_text = "\n".join(f"- {term}" for term in candidates)
    prompt = f"""你是中医知识库的受约束查询解析器，不是诊断或处方 Agent。

你的唯一任务是把用户原话映射到本地知识库已有的规范检索词。

硬性规则：
1. canonical_term 只能逐字选自下方“本地候选规范词”，不得创造新术语。
2. 每个映射必须给出 source_start 和 source_end（Python 风格左闭右开字符下标）；系统会用用户原话[source_start:source_end]作为证据。source_phrase 必须与该切片逐字一致，不要复制规范词代替原话。
3. 用户否认的症状 polarity=absent；不确定的 polarity=uncertain，绝不能当成 present。
4. 不诊断，不推荐方剂、药物、穴位或剂量，不根据常识补充用户没有说的症状和病因。
5. 无法可靠映射的原话放入 unknown_phrases；信息不足时 needs_more_info=true。
6. pathogenesis_hints 只能从本地候选规范词中选，并且用户原话必须明确表达该病因；否则留空。
7. query_intent 只能是 clinical_symptom、acupuncture_principle、classical_theory、herb_indication、formula_knowledge、unknown。
8. 同一原话可对应多个候选词时，优先选择保留部位、方向、时序和症状性质的最具体候选词；不要用宽泛词替代更精确的本地词。
9. 不要为了凑足字段而映射。每个 present/absent 结论都必须能由对应 source_phrase 单独支持。
10. 否定表达优先选不附带额外症状的最小规范词；用户只否认喘时不得选择同时包含咳嗽的复合词。
11. 用户询问自己或患者的疾病、症状如何治疗时必须标记 clinical_symptom；即使原话出现“方子、方剂、中药”也不是 formula_knowledge。formula_knowledge 只用于询问一个已明确命名方剂的组成、功效、出处或用法。
12. 用户原话中每个独立的身体感受、症状、否定或不确定片段都必须处理：能够可靠对应本地词时建立 evidence_mapping，否则逐字放入 unknown_phrases；不得静默遗漏。
13. “无汗、不寐、不得卧、不渴”等规范词自身已经表示一种存在的临床表征：用户说“一点汗都没有”映射到“无汗”时 polarity=present。只有用户否认整个规范表征时才用 absent，例如“不怕冷”映射“恶寒”时 polarity=absent。

    本地候选规范词：
{candidate_text}
"""
    try:
        messages = [SystemMessage(content=prompt), HumanMessage(content=clean_text(query))]

        def invoke_structured():
            structured = llm.with_config(temperature=0).with_structured_output(SymptomQueryTranslation)
            return structured.invoke(messages)

        response = _invoke_with_hard_timeout(
            invoke_structured,
            timeout_seconds=float(getattr(config, "LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS", 12)),
        )
        return _merge_grounded_translation(query, local, response, candidates)
    except Exception as exc:
        local["translation_method"] = "local_fallback"
        detail = str(exc).replace(os.environ.get("DEEPSEEK_API_KEY", ""), "[REDACTED]")[:500]
        local["translation_errors"] = [f"llm_failure:{type(exc).__name__}:{detail}"]
        local["expanded_query"] = _build_expanded_query(local)
        return local
