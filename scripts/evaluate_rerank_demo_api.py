from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gpu_rerank_ab import merge_questions_gold, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_QUESTIONS = ROOT / "tests" / "evals" / "rerank_locked_v1" / "questions.jsonl"
DEFAULT_GOLD = ROOT / "tests" / "evals" / "rerank_locked_v1" / "private" / "gold_keys.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "structured" / "rerank_locked_v1_api_results"
SUMMARY_PATTERN = re.compile(
    r"- (?P<name>基线|Rerank)：(?P<latency>[0-9.]+) ms；status=(?P<status>[^;]+); "
    r"gate=(?P<gate>True|False); reasons=(?P<reasons>[^\n]+)"
)


def parse_summary(summary: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for match in SUMMARY_PATTERN.finditer(summary):
        key = "baseline" if match.group("name") == "基线" else "rerank"
        reasons_text = match.group("reasons").strip()
        parsed[key] = {
            "latency_ms": float(match.group("latency")),
            "status": match.group("status").strip(),
            "gate": match.group("gate") == "True",
            "reasons": [] if reasons_text == "-" else [item.strip() for item in reasons_text.split(",")],
        }
    if set(parsed) != {"baseline", "rerank"}:
        raise ValueError(f"unable to parse demo summary: {summary}")
    return parsed


def table_rows(value: dict[str, Any]) -> list[list[Any]]:
    rows = value.get("data", []) if isinstance(value, dict) else []
    return [list(row) for row in rows]


def evaluate_output(
    case: dict[str, Any],
    rows: list[list[Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    formulas = [str(row[1]) for row in rows]
    expected_formula = str(case.get("expected_formula", ""))
    target_rows = [row for row in rows if str(row[1]) == expected_formula]
    target_topk = expected_formula in formulas
    target_top1 = bool(formulas and formulas[0] == expected_formula)
    missing_required = any(str(row[5]).strip() not in {"", "[]", "-"} for row in target_rows)
    forbidden = str(case.get("forbidden_term", ""))
    forbidden_conflict = any(forbidden and forbidden in str(row[6]) for row in target_rows)
    status = str(state["status"])
    gate = bool(state["gate"])
    reasons = list(state.get("reasons", []))
    style = str(case["style"])
    checks: dict[str, bool] = {"route_ok": True}
    if style == "full_signature_topk":
        checks.update({"target_in_topk": target_topk, "not_no_match": status != "no_match"})
    elif style == "missing_required_clarify":
        checks.update(
            {
                "target_in_topk": target_topk,
                "not_grounded": status != "grounded_answer" and not gate,
                "missing_required_detected": missing_required or "missing_required_symptom_groups" in reasons,
            }
        )
    elif style == "forbidden_conflict":
        unsafe_target_grounded = gate and status == "grounded_answer" and target_top1
        checks.update(
            {
                "no_unsafe_target_grounded": not unsafe_target_grounded,
                "target_not_top1": not target_top1,
                "target_filtered_or_conflicted": (not target_topk) or forbidden_conflict,
            }
        )
    elif style == "shared_terms_clarify":
        contrast = str(case.get("contrast_formula", ""))
        checks.update(
            {
                "not_grounded": status != "grounded_answer" and not gate,
                "has_competing_context": bool(contrast)
                and (
                    contrast in formulas
                    or target_topk
                    or "competing_indistinguishable_matches" in reasons
                    or status in {"clarify", "no_match"}
                ),
            }
        )
    else:
        raise ValueError(f"unknown style: {style}")
    return {
        "id": case["id"],
        "style": style,
        "ok": all(checks.values()),
        "checks": checks,
        "latency_ms": round(float(state["latency_ms"]), 2),
        "status": status,
        "gate": gate,
        "reasons": reasons,
        "expected_formula": expected_formula,
        "target_rank": formulas.index(expected_formula) + 1 if target_topk else None,
        "top_formula": formulas[0] if formulas else "",
        "top_k_formulas": formulas,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    styles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    for row in rows:
        styles[str(row["style"])].append(row)
        if not row["ok"]:
            for name, ok in row["checks"].items():
                if not ok:
                    failure_counts[f"{row['style']}.{name}"] += 1
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    return {
        "case_count": len(rows),
        "passed": sum(bool(row["ok"]) for row in rows),
        "pass_rate": round(sum(bool(row["ok"]) for row in rows) / len(rows), 4),
        "by_style": {
            style: {
                "count": len(items),
                "passed": sum(bool(item["ok"]) for item in items),
                "pass_rate": round(sum(bool(item["ok"]) for item in items) / len(items), 4),
            }
            for style, items in sorted(styles.items())
        },
        "failure_counts": dict(failure_counts.most_common()),
        "latency_ms": {
            "average": round(statistics.fmean(latencies), 2),
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": max(latencies),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black-box regression test for the rerank Gradio demo API.")
    parser.add_argument("--url", default="http://127.0.0.1:17860/")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = merge_questions_gold(read_jsonl(args.questions), read_jsonl(args.gold))
    if args.limit:
        cases = cases[: max(1, args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline_predictions.jsonl"
    rerank_path = args.output_dir / "rerank_predictions.jsonl"
    if args.no_resume:
        baseline_path.unlink(missing_ok=True)
        rerank_path.unlink(missing_ok=True)
    baseline_by_id = {
        str(row["id"]): row for row in (read_jsonl(baseline_path) if baseline_path.exists() else [])
    }
    rerank_by_id = {
        str(row["id"]): row for row in (read_jsonl(rerank_path) if rerank_path.exists() else [])
    }
    completed_ids = set(baseline_by_id) & set(rerank_by_id)
    client = Client(
        args.url,
        verbose=False,
        httpx_kwargs={"timeout": args.request_timeout},
        download_files=False,
    )
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in completed_ids:
            if index % 10 == 0 or index == len(cases):
                print(f"done={index}/{len(cases)} (resumed)", flush=True)
            continue
        last_error: Exception | None = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                summary, baseline_table, rerank_table, _debug = client.predict(
                    str(case["query"]),
                    int(args.top_k),
                    api_name="/compare",
                )
                break
            except Exception as exc:  # Network/tunnel failures are retriable.
                last_error = exc
                print(
                    f"retry id={case_id} attempt={attempt}/{args.retries} error={type(exc).__name__}",
                    flush=True,
                )
                if attempt >= max(1, args.retries):
                    raise
                time.sleep(min(10.0, attempt * 2.0))
                client = Client(
                    args.url,
                    verbose=False,
                    httpx_kwargs={"timeout": args.request_timeout},
                    download_files=False,
                )
        else:  # pragma: no cover - loop either breaks or raises.
            raise RuntimeError(last_error)
        states = parse_summary(str(summary))
        baseline_row = evaluate_output(case, table_rows(baseline_table), states["baseline"])
        rerank_row = evaluate_output(case, table_rows(rerank_table), states["rerank"])
        baseline_by_id[case_id] = baseline_row
        rerank_by_id[case_id] = rerank_row
        with baseline_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(baseline_row, ensure_ascii=False, sort_keys=True) + "\n")
        with rerank_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(rerank_row, ensure_ascii=False, sort_keys=True) + "\n")
        if index % 10 == 0 or index == len(cases):
            print(f"done={index}/{len(cases)}", flush=True)
    baseline_rows = [baseline_by_id[str(case["id"])] for case in cases]
    rerank_rows = [rerank_by_id[str(case["id"])] for case in cases]
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(rerank_path, rerank_rows)
    baseline = summarize(baseline_rows)
    rerank = summarize(rerank_rows)
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    comparison = {
        "pass_gained": sum((not baseline_by_id[str(row["id"])]["ok"]) and row["ok"] for row in rerank_rows),
        "pass_lost": sum(baseline_by_id[str(row["id"])]["ok"] and (not row["ok"]) for row in rerank_rows),
        "top1_changed": sum(
            baseline_by_id[str(row["id"])]["top_formula"] != row["top_formula"] for row in rerank_rows
        ),
    }
    report = {
        "schema_version": 1,
        "mode": "gradio_api_black_box_locked_eval",
        "url": args.url,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "baseline": baseline,
        "rerank": rerank,
        "comparison": comparison,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
