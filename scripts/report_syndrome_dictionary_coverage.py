from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "datasets" / "structured" / "syndrome_dictionary.jsonl"
DEFAULT_MARKDOWN = ROOT / "datasets" / "structured" / "syndrome_dictionary_coverage.md"
DEFAULT_JSON = ROOT / "datasets" / "structured" / "syndrome_dictionary_coverage.json"


def load_payloads(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def active_classic_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "markdown_docs").glob("古籍_*.md")
        if "_术语通俗释义" not in path.name
    )


def report_status(source_file: str, count: int) -> str:
    if count:
        return "已进入结构化库"
    if "神农本草经" in source_file:
        return "未进入：未抽取到 herb_indication payload，需要复核本草解析规则"
    if "难经" in source_file:
        return "未进入：需补充 classical_theory 抽取规则"
    return "未进入：需要补充对应抽取规则或人工校验"


def build_report(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(payload.get("source_type", "") for payload in payloads)
    by_book = Counter(payload.get("source_book", "") for payload in payloads)
    by_file = Counter(payload.get("source_file", "") for payload in payloads)
    by_file_type: dict[str, Counter] = defaultdict(Counter)
    for payload in payloads:
        by_file_type[payload.get("source_file", "")][payload.get("source_type", "")] += 1

    classic_rows = []
    for path in active_classic_files():
        count = by_file.get(path.name, 0)
        classic_rows.append(
            {
                "file": path.name,
                "count": count,
                "source_types": dict(by_file_type.get(path.name, Counter())),
                "status": report_status(path.name, count),
            }
        )

    return {
        "total_entries": len(payloads),
        "source_type_counts": dict(by_type),
        "source_book_counts": dict(by_book),
        "classic_file_coverage": classic_rows,
        "limitations": [
            "当前结构化库是严格规则抽取的可验证子集，不宣称覆盖所有古籍中的所有症状。",
            "《神农本草经》已按 herb_indication 抽取单味药性味和主治，但仍需人工校验毒性、炮制、剂量和现代禁忌。",
            "《难经》已按 classical_theory 抽取理论问答；该类型只作为诊法/理论依据，不输出方药或针刺建议。",
            "classical_acupuncture_principle 只作为针刺/灸法原则依据，不输出现代临床操作建议。",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 结构化方证/古籍/针刺字典覆盖报告",
        "",
        f"- 总条目：{report['total_entries']}",
        "",
        "## 按类型统计",
        "",
    ]
    for source_type, count in report["source_type_counts"].items():
        lines.append(f"- {source_type}: {count}")

    lines.extend(["", "## 活动古籍覆盖", ""])
    lines.append("| 古籍文件 | 结构化条目数 | 类型分布 | 状态 |")
    lines.append("|---|---:|---|---|")
    for row in report["classic_file_coverage"]:
        type_summary = "；".join(f"{key}: {value}" for key, value in row["source_types"].items()) or "-"
        lines.append(f"| {row['file']} | {row['count']} | {type_summary} | {row['status']} |")

    lines.extend(["", "## 限制", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成结构化方证/古籍/针刺字典覆盖报告")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    payloads = load_payloads(Path(args.jsonl))
    report = build_report(payloads)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))
    print(json.dumps({"markdown": args.markdown, "json": args.json, "total_entries": report["total_entries"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
