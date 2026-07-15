from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_chatmed_field_candidates import (
    EVIDENCE_ORIGIN,
    QUERY_CANDIDATE_STATUS,
    clean_query,
    extract_candidates,
    extract_from_record,
    is_negated,
    load_local_term_index,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def syndrome_entry(**overrides: object) -> dict:
    entry = {
        "entry_id": "formula::测试方",
        "title": "测试方证",
        "source_type": "formula_syndrome",
        "formula": "测试方",
        "herb_name": "",
        "diagnostic_keys": ["发热", "汗出"],
        "ancient_symptoms": ["恶寒"],
        "modern_symptoms": ["发烧", "出汗", "怕冷"],
        "symptom_aliases": ["身上发热", "冒汗", "冷得明显"],
        "pathogenesis": ["外感风寒"],
    }
    entry.update(overrides)
    return entry


class ExtractChatMedFieldCandidatesTests(unittest.TestCase):
    def test_clean_query_removes_model_meta_instruction(self) -> None:
        self.assertEqual(
            clean_query("我腹痛。要求：1. 请考虑所有症状。2. 请输出推理过程。"),
            "我腹痛",
        )
        self.assertEqual(
            clean_query("我头痛咳嗽怎么办？请帮我推荐中药或者方剂。2. 请根据中医知识输出一步步的推理过程"),
            "我头痛咳嗽怎么办？请帮我推荐中药或者方剂",
        )

    def test_negation_detection_handles_prefix(self) -> None:
        self.assertTrue(is_negated("没有发热，就是一直冒汗", "发热"))
        self.assertFalse(is_negated("有点发热，还一直冒汗", "发热"))
        self.assertFalse(is_negated("小便不利、肠鸣、里急后重怎么办？", "里急后重"))
        self.assertFalse(is_negated("我经常感觉烦躁不安，伴有潮热", "潮热"))

    def test_extract_uses_query_only_not_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            syndrome_path = Path(directory) / "syndrome.jsonl"
            write_jsonl(syndrome_path, [syndrome_entry()])
            index = load_local_term_index(syndrome_path)
            record = extract_from_record(
                source_line=1,
                raw_record={"query": "我肚子疼", "response": "这可能是外感风寒导致。"},
                term_index=index,
            )
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["pathogenesis_matches"], [])
            self.assertNotIn("response", record)

    def test_full_extraction_writes_isolated_unverified_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chatmed_path = root / "chatmed.jsonl"
            syndrome_path = root / "syndrome.jsonl"
            output_dir = root / "out"
            write_jsonl(
                chatmed_path,
                [
                    {"query": "没有发热，就是一直冒汗。", "response": "隐藏答案不应写出"},
                    {"query": "我这是外感风寒吗？", "response": "隐藏答案不应写出"},
                    {"query": "普通无关问题", "response": "隐藏答案不应写出"},
                ],
            )
            write_jsonl(syndrome_path, [syndrome_entry()])

            summary = extract_candidates(
                chatmed_path=chatmed_path,
                syndrome_dictionary_path=syndrome_path,
                output_dir=output_dir,
                max_rows_per_file=100,
            )

            self.assertEqual(summary["records_seen"], 3)
            self.assertEqual(summary["review_status"], QUERY_CANDIDATE_STATUS)
            self.assertEqual(summary["evidence_origin"], EVIDENCE_ORIGIN)

            diagnostic = (output_dir / "diagnostic_key_candidates.jsonl").read_text(encoding="utf-8")
            negation = (output_dir / "negation_forbidden_candidates.jsonl").read_text(encoding="utf-8")
            pathogenesis = (output_dir / "pathogenesis_candidates.jsonl").read_text(encoding="utf-8")
            self.assertIn('"review_status": "candidate_unverified"', diagnostic)
            self.assertIn("发热", negation)
            self.assertIn("外感风寒", pathogenesis)
            self.assertNotIn("隐藏答案", diagnostic + negation + pathogenesis)


if __name__ == "__main__":
    unittest.main()
