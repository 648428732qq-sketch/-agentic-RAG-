from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "datasets" / "external" / ".query_translator_pipeline.lock"
DEFAULT_REPORT = ROOT / "datasets" / "external" / "reports" / "pipeline_execution_report.json"


def project_venv_ok(executable: Path) -> bool:
    # Keep the lexical venv path. Linux venv/bin/python is commonly a symlink
    # to /usr/bin/python, and resolving it would incorrectly escape the project.
    absolute = Path(os.path.abspath(executable))
    return any(parent.name in {".venv", ".venv-linux"} and parent.parent == ROOT for parent in absolute.parents)


def pipeline_steps() -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        ("raw_manifest", [python, "scripts/audit_external_datasets.py", "--write"]),
        ("clean_external_data", [python, "scripts/prepare_external_query_data.py"]),
        ("validate_processed_data", [python, "scripts/validate_external_processed_data.py"]),
        ("audit_payload_evidence", [python, "scripts/audit_syndrome_payload_evidence.py"]),
        ("build_supervision", [python, "scripts/build_query_translator_supervision.py"]),
        ("split_supervision", [python, "scripts/split_query_translator_supervision.py"]),
        ("evaluate_local_hard_negatives", [python, "scripts/evaluate_local_hard_negative_ranking.py"]),
        (
            "data_pipeline_tests",
            [
                python,
                "-m",
                "unittest",
                "tests.test_audit_external_datasets",
                "tests.test_prepare_external_query_data",
                "tests.test_validate_external_processed_data",
                "tests.test_audit_syndrome_payload_evidence",
                "tests.test_build_query_translator_supervision",
                "tests.test_split_query_translator_supervision",
                "tests.test_evaluate_local_hard_negative_ranking",
                "tests.test_evidence_gate",
                "-q",
            ],
        ),
    ]


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = path.read_text(encoding="utf-8", errors="replace")[:500]
        raise RuntimeError(f"pipeline lock already exists: {path}\n{owner}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "python": sys.executable,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def run_pipeline(report_path: Path) -> dict[str, Any]:
    if not project_venv_ok(Path(sys.executable)):
        raise RuntimeError(f"refusing non-project Python executable: {sys.executable}")
    acquire_lock(LOCK_PATH)
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": sys.executable,
        "steps": [],
        "ok": False,
    }
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        for name, command in pipeline_steps():
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            step = {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "output_tail": completed.stdout[-8000:],
            }
            report["steps"].append(step)
            print(completed.stdout, end="", flush=True)
            if completed.returncode != 0:
                raise RuntimeError(f"pipeline step failed: {name} ({completed.returncode})")
        report["ok"] = True
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated Query Translator data pipeline")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(args.report)
    print(json.dumps({"ok": report["ok"], "step_count": len(report["steps"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
