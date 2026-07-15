from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "datasets" / "external" / "raw"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "manifests" / "raw_dataset_manifest.json"

SOURCE_METADATA = {
    "cblue_mirror_unverified": {
        "source_url": "https://github.com/CBLUEbenchmark/CBLUE",
        "license": "unverified community mirror; upstream repository Apache-2.0",
        "usage": "development_only_no_redistribution",
    },
    "huatuo26m_lite": {
        "source_url": "https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite",
        "license": "Apache-2.0",
        "usage": "question_text_only",
    },
    "mtcmb": {
        "source_url": "https://zenodo.org/records/20465629",
        "license": "CC-BY-4.0",
        "usage": "validation_only",
    },
    "tcm_ladder": {
        "source_url": "https://huggingface.co/datasets/timzzyus/TCM-Ladder",
        "license": "treat_as_CC-BY-4.0_due_to_metadata_discrepancy",
        "usage": "validation_only_text_tables",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def update_schema(
    value: Any,
    key_counts: Counter[str],
    field_types: dict[str, Counter[str]],
) -> None:
    if not isinstance(value, dict):
        key_counts["<non_object_record>"] += 1
        return
    for key, field_value in value.items():
        name = str(key)
        key_counts[name] += 1
        field_types.setdefault(name, Counter())[value_type(field_value)] += 1


def inspect_json_records(records: Iterable[Any]) -> dict[str, Any]:
    count = 0
    key_counts: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = {}
    for record in records:
        count += 1
        update_schema(record, key_counts, field_types)
    return {
        "record_count": count,
        "fields": {
            key: {
                "present_count": key_counts[key],
                "types": dict(sorted(field_types.get(key, Counter()).items())),
            }
            for key in sorted(key_counts)
        },
    }


def inspect_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value if isinstance(value, list) else [value]
    return {"format": "json", **inspect_json_records(records)}


def iter_jsonl(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def inspect_jsonl(path: Path) -> dict[str, Any]:
    return {"format": "jsonl", **inspect_json_records(iter_jsonl(path))}


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow_not_installed") from exc
    metadata = parquet.read_metadata(path)
    schema = parquet.read_schema(path)
    return {
        "format": "parquet",
        "record_count": metadata.num_rows,
        "row_group_count": metadata.num_row_groups,
        "fields": {
            field.name: {"arrow_type": str(field.type), "nullable": field.nullable}
            for field in schema
        },
    }


def inspect_data_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return inspect_json(path)
    if suffix == ".jsonl":
        return inspect_jsonl(path)
    if suffix == ".parquet":
        return inspect_parquet(path)
    return {"format": suffix.removeprefix(".") or "unknown", "record_count": None}


def discover_files(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def build_manifest(raw_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for path in discover_files(raw_root):
        relative = path.relative_to(raw_root)
        dataset = relative.parts[0] if relative.parts else "unknown"
        dataset_counts[dataset] += 1
        record: dict[str, Any] = {
            "dataset": dataset,
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        try:
            record.update(inspect_data_file(path))
            record["parse_status"] = "ok"
        except Exception as exc:
            record["parse_status"] = "error"
            record["parse_error"] = f"{type(exc).__name__}: {exc}"[:500]
            errors.append({"path": relative.as_posix(), "error": record["parse_error"]})
        files.append(record)
    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root.resolve()),
        "dataset_metadata": SOURCE_METADATA,
        "dataset_file_counts": dict(sorted(dataset_counts.items())),
        "file_count": len(files),
        "parse_error_count": len(errors),
        "parse_errors": errors,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit external raw datasets without modifying them")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_root.is_dir():
        raise SystemExit(f"raw dataset directory not found: {args.raw_root}")
    manifest = build_manifest(args.raw_root)
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = {
        "file_count": manifest["file_count"],
        "dataset_file_counts": manifest["dataset_file_counts"],
        "parse_error_count": manifest["parse_error_count"],
        "output": str(args.output) if args.write else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if manifest["parse_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
