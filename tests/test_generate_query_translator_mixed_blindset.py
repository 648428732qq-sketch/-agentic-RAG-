from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.generate_query_translator_mixed_blindset import (
    STYLE_KEYS,
    build_private_record,
    build_one_per_seed_plans,
    build_plans,
    effective_groups,
    export_mixed,
    read_jsonl,
    write_generation_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "tests" / "evals" / "query_translator_cases.jsonl"


class MixedQueryTranslatorGenerationTests(unittest.TestCase):
    def test_plan_has_exactly_sixty_cases_per_style(self) -> None:
        plans = build_plans(read_jsonl(SEEDS), per_style=60)

        self.assertEqual(len(plans), 300)
        self.assertEqual(Counter(plan.style for plan in plans), Counter({style: 60 for style in STYLE_KEYS}))
        self.assertEqual(len({plan.case_id for plan in plans}), 300)

    def test_hard_negative_removes_one_expected_group(self) -> None:
        plan = next(plan for plan in build_plans(read_jsonl(SEEDS), 1) if plan.style == "hard_negative_clarify")

        self.assertIsNotNone(plan.omitted_group_index)
        self.assertEqual(len(effective_groups(plan)), len(plan.seed["expected_term_groups"]) - 1)

    def test_public_export_contains_no_style_or_gold_fields(self) -> None:
        records = [
            {
                "id": "case_1",
                "query": "脑壳昏得很",
                "generation_style": "asr_homophone_typos",
                "expected_term_groups": [["头晕"]],
            },
            {
                "id": "case_2",
                "query": "肚子像搅拌机一样",
                "generation_style": "metaphor_incomplete",
                "expected_term_groups": [["腹痛"]],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            export_mixed(records, output_dir, shuffle_seed=7)
            public = read_jsonl(output_dir / "questions_mixed.jsonl")
            private = read_jsonl(output_dir / "private" / "gold_keys.jsonl")

        self.assertEqual({key for record in public for key in record}, {"id", "query"})
        self.assertEqual(len(private), 2)
        self.assertIn("generation_style", private[0])

    def test_private_gold_preserves_retrieval_and_gate_targets(self) -> None:
        plan = next(
            plan
            for plan in build_plans(read_jsonl(SEEDS), 1)
            if plan.style == "hard_negative_clarify"
        )
        mappings = [
            {
                "source_phrase": group[0],
                "canonical_term": group[0],
                "polarity": "present",
            }
            for group in effective_groups(plan)
        ]
        record = build_private_record(
            plan,
            {
                "query": "测试问题",
                "evidence_mappings": mappings,
                "typo_pairs": [],
            },
            "qwen3.6-flash",
        )

        self.assertFalse(record["expected_gate"])
        self.assertTrue(record["expected_needs_more_info"])
        self.assertEqual(record["expected_decision"], "clarify")

    def test_one_per_seed_uses_each_planned_style_once(self) -> None:
        seeds = [
            {
                "id": f"seed_{index}",
                "planned_style": style,
                "planned_variant_index": index,
                "planned_region": "",
                "planned_omitted_group_index": 0,
                "query": "测试",
                "expected_term_groups": [["腹痛"], ["呕吐"]],
                "expected_negative_terms": ["发热"] if style == "negation_uncertainty" else [],
            }
            for index, style in enumerate(STYLE_KEYS)
        ]

        plans = build_one_per_seed_plans(seeds)

        self.assertEqual(len(plans), len(seeds))
        self.assertEqual(Counter(plan.style for plan in plans), Counter(STYLE_KEYS))

    def test_generation_manifest_preserves_seed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "seed_pool_frozen_pending_qwen_generation",
                        "count": 100,
                        "integrity_checks": {"answer_leaks": 0},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            write_generation_manifest(output_dir, {"complete": True, "generated_count": 100})
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["count"], 100)
        self.assertEqual(manifest["integrity_checks"]["answer_leaks"], 0)
        self.assertEqual(manifest["generation"]["generated_count"], 100)
        self.assertEqual(manifest["status"], "final_holdout_generated_pending_system_evaluation")


if __name__ == "__main__":
    unittest.main()
