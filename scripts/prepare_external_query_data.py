from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "datasets" / "external" / "raw"
DEFAULT_MANIFEST = ROOT / "datasets" / "external" / "manifests" / "raw_dataset_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "datasets" / "external" / "processed"
DEFAULT_REPORT = ROOT / "datasets" / "external" / "reports" / "data_cleaning_report.json"

DATASET_ORDER = {
    "mtcmb": 0,
    "tcm_ladder": 1,
    "cblue_mirror_unverified": 2,
    "huatuo26m_lite": 3,
}

DATASET_ROLE = {
    "mtcmb": "validation_only",
    "tcm_ladder": "validation_only",
    "cblue_mirror_unverified": "development_only",
    "huatuo26m_lite": "development_only",
}

PII_PATTERNS = (
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)"), "[EMAIL]"),
    ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    ("id_card", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[ID_CARD]"),
    (
        "account",
        re.compile(r"(?i)(?:微信|wechat|QQ|账号)\s*(?:号|[:：])?\s*[A-Za-z0-9_-]{5,}"),
        "[ACCOUNT]",
    ),
    (
        "name",
        re.compile(r"(?:我叫|姓名\s*[:：]?\s*)([\u3400-\u9fff·]{2,8})"),
        "[NAME]",
    ),
    (
        "address",
        re.compile(r"(?:家住|住在|地址\s*(?:是|[:：])?)\s*[^，。；;\n]{3,50}"),
        "[ADDRESS]",
    ),
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_PATTERN = re.compile(r"\s+")
DEDUPE_DROP_PATTERN = re.compile(r"[^0-9a-z\u3400-\u9fff]+", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        return "".join(parser.parts)
    except Exception:
        return re.sub(r"<[^>]+>", " ", value)


def clean_text(value: Any) -> tuple[str, Counter[str]]:
    text = "" if value is None else str(value)
    stats: Counter[str] = Counter()
    # Preserve full-width Chinese punctuation because it can encode clause boundaries.
    normalized = unicodedata.normalize("NFC", html.unescape(text))
    if "<" in normalized and ">" in normalized:
        cleaned_html = strip_html(normalized)
        if cleaned_html != normalized:
            stats["html_removed"] += 1
        normalized = cleaned_html
    normalized, url_count = URL_PATTERN.subn("[URL]", normalized)
    stats["url_masked"] += url_count
    normalized, control_count = CONTROL_PATTERN.subn(" ", normalized)
    stats["control_removed"] += control_count
    for name, pattern, replacement in PII_PATTERNS:
        normalized, count = pattern.subn(replacement, normalized)
        stats[f"pii_{name}_masked"] += count
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized, stats


def dedupe_key(text: str) -> str:
    return DEDUPE_DROP_PATTERN.sub("", text.casefold())


def char_ngrams(text: str, size: int = 3) -> set[str]:
    value = dedupe_key(text)
    if len(value) <= size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def simhash64(features: set[str]) -> int:
    weights = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if number & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class NearDuplicateIndex:
    def __init__(self, threshold: float = 0.93, max_candidates: int = 128) -> None:
        self.threshold = threshold
        self.max_candidates = max_candidates
        self._texts: list[str] = []
        self._signatures: list[int] = []
        self._buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def find_or_add(self, text: str) -> int | None:
        features = char_ngrams(text)
        signature = simhash64(features)
        candidate_ids: set[int] = set()
        for band in range(4):
            bucket = (band, (signature >> (band * 16)) & 0xFFFF)
            candidate_ids.update(self._buckets.get(bucket, ()))
        if len(candidate_ids) > self.max_candidates:
            candidate_ids = set(
                sorted(
                    candidate_ids,
                    key=lambda candidate_id: (signature ^ self._signatures[candidate_id]).bit_count(),
                )[: self.max_candidates]
            )
        for candidate_id in candidate_ids:
            if jaccard(features, char_ngrams(self._texts[candidate_id])) >= self.threshold:
                return candidate_id
        record_id = len(self._texts)
        self._texts.append(text)
        self._signatures.append(signature)
        for band in range(4):
            bucket = (band, (signature >> (band * 16)) & 0xFFFF)
            self._buckets[bucket].append(record_id)
        return None


def stable_id(*parts: Any) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def iter_json(path: Path) -> Iterator[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    records = value if isinstance(value, list) else [value]
    for record in records:
        if isinstance(record, dict):
            yield record


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to process parquet datasets") from exc
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        yield from iter_json(path)
    elif path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
    elif path.suffix.lower() == ".parquet":
        yield from iter_parquet(path)


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(filter(None, (flatten_text(item) for item in value)))
    if isinstance(value, dict):
        return " ".join(filter(None, (flatten_text(item) for item in value.values())))
    return ""


@dataclass(frozen=True)
class ExtractedQuestion:
    text: str
    source_id: str
    metadata: dict[str, Any]
    pair: dict[str, Any] | None = None
    validation_label: dict[str, Any] | None = None


def extract_cblue(record: dict[str, Any], source_file: str) -> Iterable[ExtractedQuestion]:
    record_id = str(record.get("id", ""))
    name = Path(source_file).name
    if "CHIP-STS" in name:
        fields = ("text1", "text2")
        extra = {"category": record.get("category")}
    elif "KUAKE-QQR" in name:
        fields = ("query1", "query2")
        extra = {}
    elif "KUAKE-QTR" in name:
        fields = ("query", "title")
        extra = {}
    else:
        return
    raw_texts = [flatten_text(record.get(field)) for field in fields]
    pair = {
        "pair_id": stable_id("cblue", source_file, record_id),
        "source_id": record_id,
        "text_a": raw_texts[0],
        "text_b": raw_texts[1],
        "label": record.get("label"),
        **extra,
    }
    for role, raw_text in zip(("a", "b"), raw_texts):
        yield ExtractedQuestion(
            text=raw_text,
            source_id=f"{record_id}:{role}",
            metadata={"pair_id": pair["pair_id"], "pair_role": role, **extra},
            pair=pair if role == "a" else None,
        )


def extract_huatuo(record: dict[str, Any]) -> Iterable[ExtractedQuestion]:
    yield ExtractedQuestion(
        text=flatten_text(record.get("question")),
        source_id=str(record.get("id", "")),
        metadata={
            "department_label": record.get("label"),
            "related_diseases": record.get("related_diseases"),
        },
    )


def extract_mtcmb(record: dict[str, Any], source_file: str) -> Iterable[ExtractedQuestion]:
    record_id = str(record.get("id", ""))
    text_fields = ("question", "Medical_case", "dialogue", "text", "disease_case")
    selected_field = next((field for field in text_fields if record.get(field)), "")
    raw_text = flatten_text(record.get(selected_field))
    label_payload = {
        key: record[key]
        for key in ("answer", "options", "points", "annotations")
        if key in record
    }
    yield ExtractedQuestion(
        text=raw_text,
        source_id=record_id,
        metadata={"task_file": Path(source_file).name, "text_field": selected_field},
        validation_label={
            "source_id": record_id,
            "task_file": Path(source_file).name,
            "labels": label_payload,
        },
    )


def extract_tcm_ladder(record: dict[str, Any], source_file: str, index: int) -> Iterable[ExtractedQuestion]:
    source_id = str(record.get("id", index))
    options = {key: record[key] for key in ("A", "B", "C", "D", "E") if record.get(key) is not None}
    yield ExtractedQuestion(
        text=flatten_text(record.get("question")),
        source_id=source_id,
        metadata={
            "task_file": Path(source_file).name,
            "category": record.get("category"),
            "lang": record.get("lang"),
            "type": record.get("type"),
        },
        validation_label={
            "source_id": source_id,
            "task_file": Path(source_file).name,
            "answer": record.get("answer"),
            "options": options,
        },
    )


def extract_questions(dataset: str, record: dict[str, Any], source_file: str, index: int) -> Iterable[ExtractedQuestion]:
    if dataset == "cblue_mirror_unverified":
        yield from extract_cblue(record, source_file)
    elif dataset == "huatuo26m_lite":
        yield from extract_huatuo(record)
    elif dataset == "mtcmb":
        yield from extract_mtcmb(record, source_file)
    elif dataset == "tcm_ladder":
        yield from extract_tcm_ladder(record, source_file, index)


def load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    file_map = {entry["path"]: entry for entry in manifest.get("files", [])}
    return file_map, manifest.get("dataset_metadata", {})


class JsonlWriters:
    def __init__(self) -> None:
        self._handles: dict[Path, Any] = {}

    def write(self, path: Path, record: dict[str, Any]) -> None:
        if path not in self._handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[path] = path.open("w", encoding="utf-8", newline="\n")
        self._handles[path].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def reset_managed_outputs(output_root: Path) -> None:
    managed = (
        output_root / "questions" / "development" / "cblue_mirror_unverified.jsonl",
        output_root / "questions" / "development" / "huatuo26m_lite.jsonl",
        output_root / "questions" / "validation" / "mtcmb.jsonl",
        output_root / "questions" / "validation" / "tcm_ladder.jsonl",
        output_root / "pairs" / "cblue_semantic_pairs.jsonl",
        output_root / "validation_labels" / "mtcmb.jsonl",
        output_root / "validation_labels" / "tcm_ladder.jsonl",
    )
    for path in managed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def process_datasets(
    raw_root: Path,
    manifest_path: Path,
    output_root: Path,
    report_path: Path,
    near_duplicate_threshold: float = 0.93,
) -> dict[str, Any]:
    file_map, metadata = load_manifest(manifest_path)
    reset_managed_outputs(output_root)
    data_files = [
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".parquet"} and ".git" not in path.parts
    ]
    data_files.sort(key=lambda path: (DATASET_ORDER.get(path.relative_to(raw_root).parts[0], 99), path.as_posix()))

    writers = JsonlWriters()
    exact_seen: dict[str, str] = {}
    pair_seen: set[str] = set()
    near_index = NearDuplicateIndex(threshold=near_duplicate_threshold)
    counters: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        for path in data_files:
            relative = path.relative_to(raw_root).as_posix()
            dataset = path.relative_to(raw_root).parts[0]
            if dataset not in DATASET_ROLE:
                counters["unsupported_dataset_files"] += 1
                continue
            file_manifest = file_map.get(relative)
            if not file_manifest or file_manifest.get("parse_status") != "ok":
                counters["missing_or_invalid_manifest"] += 1
                continue
            source_metadata = metadata.get(dataset, {})
            role = DATASET_ROLE[dataset]
            for index, raw_record in enumerate(iter_records(path)):
                counters["raw_records"] += 1
                by_dataset[dataset]["raw_records"] += 1
                for extracted in extract_questions(dataset, raw_record, relative, index):
                    counters["candidate_questions"] += 1
                    by_dataset[dataset]["candidate_questions"] += 1
                    text, text_stats = clean_text(extracted.text)
                    counters.update(text_stats)
                    by_dataset[dataset].update(text_stats)

                    # Semantic-pair supervision has its own lifecycle. A repeated
                    # question may still participate in a distinct labeled pair.
                    if extracted.pair is not None:
                        counters["candidate_pairs"] += 1
                        by_dataset[dataset]["candidate_pairs"] += 1
                        text_a, stats_a = clean_text(extracted.pair["text_a"])
                        text_b, stats_b = clean_text(extracted.pair["text_b"])
                        counters.update(stats_a)
                        counters.update(stats_b)
                        pair_key = stable_id(
                            relative,
                            dedupe_key(text_a),
                            dedupe_key(text_b),
                            extracted.pair.get("label"),
                            extracted.pair.get("category"),
                        )
                        if len(dedupe_key(text_a)) < 4 or len(dedupe_key(text_b)) < 4:
                            counters["dropped_invalid_pair"] += 1
                            by_dataset[dataset]["dropped_invalid_pair"] += 1
                        elif pair_key in pair_seen:
                            counters["dropped_duplicate_pair"] += 1
                            by_dataset[dataset]["dropped_duplicate_pair"] += 1
                        else:
                            pair_seen.add(pair_key)
                            pair_record = {
                                **{
                                    key: value
                                    for key, value in extracted.pair.items()
                                    if key not in {"text_a", "text_b"}
                                },
                                "text_a": text_a,
                                "text_b": text_b,
                                "source_dataset": dataset,
                                "source_file": relative,
                                "source_hash": file_manifest["sha256"],
                                "usage": source_metadata.get("usage", role),
                            }
                            writers.write(output_root / "pairs" / "cblue_semantic_pairs.jsonl", pair_record)
                            counters["kept_pairs"] += 1
                            by_dataset[dataset]["kept_pairs"] += 1

                    key = dedupe_key(text)
                    if len(key) < 4:
                        counters["dropped_too_short"] += 1
                        by_dataset[dataset]["dropped_too_short"] += 1
                        continue
                    if len(text) > 2000:
                        counters["dropped_too_long"] += 1
                        by_dataset[dataset]["dropped_too_long"] += 1
                        continue
                    if key in exact_seen:
                        counters["dropped_exact_duplicate"] += 1
                        by_dataset[dataset]["dropped_exact_duplicate"] += 1
                        continue
                    near_duplicate_of = near_index.find_or_add(text)
                    if near_duplicate_of is not None:
                        counters["dropped_near_duplicate"] += 1
                        by_dataset[dataset]["dropped_near_duplicate"] += 1
                        continue
                    record_id = stable_id(dataset, relative, extracted.source_id, key)
                    exact_seen[key] = record_id
                    question_record = {
                        "record_id": record_id,
                        "text": text,
                        "source_dataset": dataset,
                        "source_id": extracted.source_id,
                        "source_file": relative,
                        "source_hash": file_manifest["sha256"],
                        "license": source_metadata.get("license", "unknown"),
                        "usage": source_metadata.get("usage", role),
                        "split_role": role,
                        "metadata": extracted.metadata,
                    }
                    output_partition = "validation" if role == "validation_only" else "development"
                    writers.write(output_root / "questions" / output_partition / f"{dataset}.jsonl", question_record)
                    counters["kept_questions"] += 1
                    by_dataset[dataset]["kept_questions"] += 1

                    if extracted.validation_label is not None:
                        label_record = {
                            "question_record_id": record_id,
                            "source_dataset": dataset,
                            "source_file": relative,
                            "source_hash": file_manifest["sha256"],
                            **extracted.validation_label,
                        }
                        writers.write(output_root / "validation_labels" / f"{dataset}.jsonl", label_record)
                        counters["kept_validation_labels"] += 1
                        by_dataset[dataset]["kept_validation_labels"] += 1
    finally:
        writers.close()

    report = {
        "report_version": 1,
        "raw_root": str(raw_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root.resolve()),
        "near_duplicate_threshold": near_duplicate_threshold,
        "processing_order": [name for name, _ in sorted(DATASET_ORDER.items(), key=lambda item: item[1])],
        "validation_precedes_development_for_cross_split_dedup": True,
        "answer_policy": {
            "huatuo26m_lite": "answer_not_read_or_exported",
            "cblue_mirror_unverified": "relation_labels_only",
            "mtcmb": "answers_separated_into_validation_labels",
            "tcm_ladder": "answers_separated_into_validation_labels",
        },
        "counts": dict(sorted(counters.items())),
        "by_dataset": {name: dict(sorted(values.items())) for name, values in sorted(by_dataset.items())},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and isolate external Query Translator datasets")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.93)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_root.is_dir():
        raise SystemExit(f"raw root not found: {args.raw_root}")
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    report = process_datasets(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        report_path=args.report,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
