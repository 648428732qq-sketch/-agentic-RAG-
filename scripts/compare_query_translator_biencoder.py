from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

from core.query_translator_biencoder import fuse_prediction_lists, score_predictions  # noqa: E402


DEFAULT_EXCLUDED_STYLES = {"regional_dialect"}


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError(f"invalid evaluation report: {path}")
    return value


def fuse_reports(
    paths: list[Path],
    output: Path,
    limit: int,
    rank_constant: int,
    excluded_styles: set[str] | None = None,
) -> dict[str, Any]:
    reports = [load_report(path) for path in paths]
    excluded_styles = excluded_styles or set()
    for report in reports:
        report["cases"] = [
            case for case in report["cases"] if str(case.get("style", "")) not in excluded_styles
        ]
    case_maps = [{str(case["id"]): case for case in report["cases"]} for report in reports]
    ids = [str(case["id"]) for case in reports[0]["cases"]]
    for case_map in case_maps[1:]:
        if set(case_map) != set(ids):
            raise ValueError("evaluation reports do not contain the same case ids")
    cases: list[dict[str, Any]] = []
    for case_id in ids:
        source = case_maps[0][case_id]
        cases.append(
            {
                "id": case_id,
                "query": source.get("query", ""),
                "style": source.get("style", ""),
                "expected_groups": source.get("expected_groups", []),
                "predictions": fuse_prediction_lists(
                    [case_map[case_id].get("predictions", []) for case_map in case_maps],
                    limit,
                    rank_constant,
                ),
            }
        )
    metrics = score_predictions(cases, (1, 3, 5, 10, limit))
    by_style: dict[str, dict[str, Any]] = {}
    for style in sorted({str(case.get("style", "")) for case in cases if case.get("style")}):
        by_style[style] = score_predictions([case for case in cases if case.get("style") == style], (5, 10, limit))
    report = {
        "report_version": 1,
        "mode": "equal_weight_reciprocal_rank_fusion",
        "inputs": [str(path.resolve()) for path in paths],
        "rank_constant": rank_constant,
        "excluded_styles": sorted(excluded_styles),
        "metrics": metrics,
        "by_style": by_style,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse base and trained Query Translator term rankings")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument(
        "--include-dialect",
        action="store_true",
        help="Include archived regional_dialect cases; excluded by current project policy",
    )
    args = parser.parse_args()
    report = fuse_reports(
        args.input,
        args.output,
        args.limit,
        args.rank_constant,
        set() if args.include_dialect else DEFAULT_EXCLUDED_STYLES,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
