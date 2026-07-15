from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPTS))

from build_syndrome_dictionary import (  # noqa: E402
    FormulaHerb,
    SyndromeEntry,
    extract_herb_indication_terms,
    make_search_text,
    modernize_symptoms,
    split_symptom_phrase,
)
from core.llm_factory import create_llm  # noqa: E402
from core.syndrome_terms import clean_text, unique  # noqa: E402


ALLOWED_SOURCE_TYPES = {"classical_clause", "classical_acupuncture", "herb_indication"}


class ExtractedClassicEntry(BaseModel):
    title: str = Field(default="", description="短标题，不要编造书外信息")
    source_type: str = Field(default="", description="classical_clause / classical_acupuncture / herb_indication")
    syndrome_name: str = ""
    ancient_symptoms: list[str] = Field(default_factory=list)
    modern_symptoms: list[str] = Field(default_factory=list)
    symptom_aliases: list[str] = Field(default_factory=list)
    pathogenesis: list[str] = Field(default_factory=list)
    intervention_type: str = Field(default="", description="formula / acupuncture / herb")
    intervention_name: str = ""
    treatment_method: str = ""
    acupoints_or_channels: list[str] = Field(default_factory=list)
    treatment_principle: str = ""
    formula: str = ""
    formula_composition: list[FormulaHerb] = Field(default_factory=list)
    herb_name: str = ""
    herb_aliases: list[str] = Field(default_factory=list)
    nature_flavor: list[str] = Field(default_factory=list)
    origin_habitat: str = ""
    indications: str = ""
    usage_original: str = ""
    contraindications: str = ""
    evidence: str = Field(default="", description="必须是输入原文中的短证据，不要改写")
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class ExtractedClassicBatch(BaseModel):
    entries: list[ExtractedClassicEntry] = Field(default_factory=list)


def chunk_markdown(markdown: str, chunk_chars: int) -> list[str]:
    lines = [clean_text(line) for line in markdown.splitlines()]
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        if not line:
            continue
        if size + len(line) > chunk_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            size = 0
        buffer.append(line)
        size += len(line)
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def infer_source_type(entry: ExtractedClassicEntry) -> tuple[str, str]:
    source_type = clean_text(entry.source_type)
    intervention_type = clean_text(entry.intervention_type)
    if source_type in ALLOWED_SOURCE_TYPES:
        if source_type == "classical_acupuncture":
            return source_type, "acupuncture"
        if source_type == "herb_indication":
            return source_type, "herb"
        return source_type, "formula"
    if entry.herb_name or entry.nature_flavor:
        return "herb_indication", "herb"
    if entry.acupoints_or_channels or "刺" in entry.treatment_method or "灸" in entry.treatment_method:
        return "classical_acupuncture", "acupuncture"
    if entry.formula:
        return "classical_clause", "formula"
    return "classical_clause", intervention_type or "formula"


def normalize_entry(raw: ExtractedClassicEntry, source_path: Path, chunk_index: int, item_index: int) -> SyndromeEntry | None:
    source_type, intervention_type = infer_source_type(raw)
    evidence = clean_text(raw.evidence or raw.indications)
    if not evidence:
        return None

    ancient_symptoms = unique(raw.ancient_symptoms)
    if not ancient_symptoms:
        if source_type == "herb_indication":
            ancient_symptoms = extract_herb_indication_terms(raw.indications or evidence)
        else:
            ancient_symptoms = split_symptom_phrase(raw.indications or evidence)
    if not ancient_symptoms:
        return None

    modern_symptoms = unique(raw.modern_symptoms)
    symptom_aliases = unique(raw.symptom_aliases)
    if not modern_symptoms:
        modern_symptoms, derived_aliases = modernize_symptoms(ancient_symptoms)
        symptom_aliases = unique(symptom_aliases + derived_aliases)

    formula = clean_text(raw.formula) if intervention_type == "formula" else ""
    herb_name = clean_text(raw.herb_name) if intervention_type == "herb" else ""
    intervention_name = clean_text(raw.intervention_name or formula or herb_name or raw.treatment_method)
    title = clean_text(raw.title or intervention_name or raw.syndrome_name or evidence[:24])
    entry_id_key = re.sub(r"\s+", "", f"{source_path.name}:{chunk_index}:{item_index}:{title}:{evidence[:40]}")

    entry = SyndromeEntry(
        entry_id=f"llm_candidate::{entry_id_key}",
        title=title,
        source_type=source_type,
        source_book=source_path.stem.replace("古籍_", ""),
        source_file=source_path.name,
        chapter="LLM候选抽取",
        syndrome_name=clean_text(raw.syndrome_name or title),
        ancient_symptoms=ancient_symptoms,
        modern_symptoms=modern_symptoms,
        symptom_aliases=symptom_aliases,
        diagnostic_keys=ancient_symptoms,
        pathogenesis=unique(raw.pathogenesis),
        intervention_type=intervention_type,
        intervention_name=intervention_name,
        treatment_method=clean_text(raw.treatment_method),
        acupoints_or_channels=unique(raw.acupoints_or_channels),
        treatment_principle=clean_text(raw.treatment_principle),
        formula=formula,
        formula_composition=raw.formula_composition if intervention_type == "formula" else [],
        herb_name=herb_name,
        herb_aliases=unique(raw.herb_aliases),
        nature_flavor=unique(raw.nature_flavor),
        origin_habitat=clean_text(raw.origin_habitat),
        property_text=clean_text("；".join(part for part in (raw.nature_flavor[0] if raw.nature_flavor else "", raw.indications) if part)),
        usage_original=clean_text(raw.usage_original),
        indications=clean_text(raw.indications or evidence),
        contraindications=clean_text(raw.contraindications),
        evidence=evidence,
        review_status="llm_extracted_needs_review",
        confidence=min(float(raw.confidence or 0.3), 0.5),
        raw_text=evidence,
        payload_version="syndrome_entry_v1",
    )
    entry.search_text = make_search_text(entry)
    return entry


def extract_chunk(llm, text: str, source_type_hint: str) -> ExtractedClassicBatch:
    prompt = f"""你是中医古籍结构化抽取器，只做证据抽取，不做诊断、不补现代剂量、不扩写。

目标 source_type_hint: {source_type_hint}

只输出输入文本中明确出现的“症状/病机/方剂/针刺灸法/单味药性味主治”映射。
规则：
1. evidence 必须逐字来自输入文本，可短摘录，但不得编造。
2. 没有明确方剂、针灸方法或本草单味药主治时，entries 返回空列表。
3. 不要把禁忌句抽成正向用方，例如“不可与某汤”不能变成可用某汤。
4. 不做跨时代剂量换算，不输出现代克数。
5. 单味药用 herb_indication；方剂条文用 classical_clause；针刺/灸法用 classical_acupuncture。
"""
    structured = llm.with_config(temperature=0).with_structured_output(ExtractedClassicBatch)
    return structured.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])


def main() -> None:
    parser = argparse.ArgumentParser(description="用 LLM 将新古籍 Markdown 抽成待审核 SyndromeEntry JSONL")
    parser.add_argument("--source", required=True, help="新古籍 Markdown 路径")
    parser.add_argument("--output", default="", help="输出 JSONL；默认 datasets/structured/llm_candidates_<stem>.jsonl")
    parser.add_argument("--chunk-chars", type=int, default=2400)
    parser.add_argument("--max-chunks", type=int, default=0, help="调试用；0 表示处理全部")
    parser.add_argument("--source-type-hint", default="auto", choices=["auto", "classical_clause", "classical_acupuncture", "herb_indication"])
    args = parser.parse_args()

    source_path = Path(args.source)
    markdown = source_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown, args.chunk_chars)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    llm = create_llm()
    entries: list[SyndromeEntry] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        batch = extract_chunk(llm, chunk, args.source_type_hint)
        for item_index, raw_entry in enumerate(batch.entries, start=1):
            entry = normalize_entry(raw_entry, source_path, chunk_index, item_index)
            if entry is not None:
                entries.append(entry)

    output_path = Path(args.output) if args.output else ROOT / "datasets" / "structured" / f"llm_candidates_{source_path.stem}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(json.dumps({"source": str(source_path), "chunks": len(chunks), "entries": len(entries), "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
