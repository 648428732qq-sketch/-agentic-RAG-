from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "datasets" / "external" / "supervision"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "splits"
DEFAULT_REPORT = ROOT / "datasets" / "external" / "reports" / "supervision_leakage_report.json"
NORMALIZE_PATTERN = re.compile(r"[^0-9a-z\u3400-\u9fff]+", re.IGNORECASE)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: Counter[str] = Counter()

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def stable_split(component_key: str, dev_fraction: float, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}\x1f{component_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "dev" if value < dev_fraction else "train"


def mapping_nodes(record: dict[str, Any]) -> set[str]:
    nodes: set[str] = set()
    for mapping in record.get("mappings") or []:
        term = str(mapping.get("canonical_term", "")).strip()
        if term:
            nodes.add(f"term:{term}")
        for entry_id in mapping.get("evidence_entry_ids") or []:
            if entry_id:
                nodes.add(f"entry:{entry_id}")
    return nodes


def split_mapping_records(
    records: list[dict[str, Any]],
    dev_fraction: float,
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    record_nodes: list[set[str]] = []
    terms: set[str] = set()
    for record in records:
        nodes = {node for node in mapping_nodes(record) if node.startswith("term:")}
        record_nodes.append(nodes)
        terms.update(nodes)
    term_splits = {term: stable_split(term, dev_fraction, seed) for term in terms}
    outputs = {"train": [], "dev": [], "excluded": []}
    for record, nodes in zip(records, record_nodes):
        if not nodes:
            outputs["excluded"].append({**record, "exclusion_reason": "missing_mapping_nodes"})
            continue
        assigned = {term_splits[node] for node in nodes}
        if len(assigned) != 1:
            outputs["excluded"].append({**record, "exclusion_reason": "mixed_target_term_splits"})
            continue
        outputs[next(iter(assigned))].append(record)

    train_terms = {node for record in outputs["train"] for node in mapping_nodes(record) if node.startswith("term:")}
    dev_terms = {node for record in outputs["dev"] for node in mapping_nodes(record) if node.startswith("term:")}
    train_entries = {node for record in outputs["train"] for node in mapping_nodes(record) if node.startswith("entry:")}
    dev_entries = {node for record in outputs["dev"] for node in mapping_nodes(record) if node.startswith("entry:")}
    report = {
        "split_policy": "canonical_term_holdout",
        "target_term_count": len(terms),
        "train_target_terms": sum(split == "train" for split in term_splits.values()),
        "dev_target_terms": sum(split == "dev" for split in term_splits.values()),
        "train_records": len(outputs["train"]),
        "dev_records": len(outputs["dev"]),
        "excluded_records": len(outputs["excluded"]),
        "term_overlap": sorted(train_terms & dev_terms),
        "entry_overlap_count": len(train_entries & dev_entries),
        "entry_overlap_examples": sorted(train_entries & dev_entries)[:50],
        "entry_overlap_policy": "reported_not_used_for_translator_target_split; retrieval_uses_separate_entry_hard_negatives",
    }
    return outputs, report


def normalized_text(value: Any) -> str:
    return NORMALIZE_PATTERN.sub("", str(value).casefold())


def split_semantic_pairs(
    records: list[dict[str, Any]],
    dev_fraction: float,
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    dsu = DisjointSet()
    pair_nodes: list[tuple[str, str]] = []
    for record in records:
        left = f"text:{normalized_text(record.get('text_a', ''))}"
        right = f"text:{normalized_text(record.get('text_b', ''))}"
        pair_nodes.append((left, right))
        dsu.union(left, right)
    component_members: dict[str, set[str]] = defaultdict(set)
    for node in dsu.parent:
        component_members[dsu.find(node)].add(node)
    component_splits = {
        root: stable_split(min(members), dev_fraction, seed)
        for root, members in component_members.items()
    }
    outputs = {"train": [], "dev": []}
    for record, (left, _) in zip(records, pair_nodes):
        outputs[component_splits[dsu.find(left)]].append(record)
    train_texts = {node for record in outputs["train"] for node in (normalized_text(record.get("text_a")), normalized_text(record.get("text_b")))}
    dev_texts = {node for record in outputs["dev"] for node in (normalized_text(record.get("text_a")), normalized_text(record.get("text_b")))}
    component_sizes = sorted((len(members) for members in component_members.values()), reverse=True)
    return outputs, {
        "component_count": len(component_members),
        "largest_component_texts": component_sizes[0] if component_sizes else 0,
        "train_records": len(outputs["train"]),
        "dev_records": len(outputs["dev"]),
        "text_overlap_count": len(train_texts & dev_texts),
    }


def run_split(
    input_root: Path,
    output_root: Path,
    report_path: Path,
    dev_fraction: float = 0.15,
    seed: str = "query-translator-v1",
) -> dict[str, Any]:
    mappings = list(iter_jsonl(input_root / "query_term_pairs.jsonl"))
    semantic_pairs = list(iter_jsonl(input_root / "semantic_pair_supervision.jsonl"))
    hard_negatives = list(iter_jsonl(input_root / "hard_negatives.jsonl"))
    mapping_splits, mapping_report = split_mapping_records(mappings, dev_fraction, seed)
    semantic_splits, semantic_report = split_semantic_pairs(semantic_pairs, dev_fraction, seed)

    output_counts = {
        "train_query_term_pairs": write_jsonl(output_root / "train" / "query_term_pairs.jsonl", mapping_splits["train"]),
        "dev_query_term_pairs": write_jsonl(output_root / "dev" / "query_term_pairs.jsonl", mapping_splits["dev"]),
        "excluded_query_term_pairs": write_jsonl(output_root / "excluded" / "query_term_pairs.jsonl", mapping_splits["excluded"]),
        "train_semantic_pairs": write_jsonl(output_root / "train" / "semantic_pair_supervision.jsonl", semantic_splits["train"]),
        "dev_semantic_pairs": write_jsonl(output_root / "dev" / "semantic_pair_supervision.jsonl", semantic_splits["dev"]),
        "dev_hard_negatives": write_jsonl(output_root / "dev" / "hard_negatives.jsonl", hard_negatives),
    }
    report = {
        "report_version": 1,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "mapping_split": mapping_report,
        "semantic_pair_split": semantic_report,
        "output_counts": output_counts,
        "external_validation_policy": "mtcmb_tcm_ladder_and_v2_are_not_read_by_this_splitter",
        "micro_model_training_ready": (
            output_counts["train_query_term_pairs"] >= 20000
            and output_counts["dev_query_term_pairs"] >= 1000
            and not mapping_report["term_overlap"]
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-safe Query Translator train/dev splits")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", default="query-translator-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_split(args.input_root, args.output_root, args.report, args.dev_fraction, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
