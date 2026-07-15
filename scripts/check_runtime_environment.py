from __future__ import annotations

import argparse
import importlib.metadata
import json
import locale
import os
import platform
import sys
import tempfile
import codecs
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402


def is_utf8_encoding(value: str | None) -> bool:
    normalized = (value or "").lower().replace("-", "").replace("_", "")
    return normalized in {"utf8", "utf8sig"}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _model_cached(repo_id: str) -> tuple[bool, str]:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, local_files_only=True)
        return True, str(path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: local snapshot unavailable"


def inspect_utf8_file(path: Path) -> dict[str, Any]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    replacement_characters = 0
    size = 0
    has_bom = False
    try:
        with path.open("rb") as handle:
            first = handle.read(3)
            size += len(first)
            has_bom = first == codecs.BOM_UTF8
            text = decoder.decode(first)
            replacement_characters += text.count("\ufffd")
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                text = decoder.decode(chunk)
                replacement_characters += text.count("\ufffd")
            text = decoder.decode(b"", final=True)
            replacement_characters += text.count("\ufffd")
        return {
            "ok": replacement_characters == 0,
            "bytes": size,
            "bom": has_bom,
            "replacement_characters": replacement_characters,
        }
    except UnicodeDecodeError as exc:
        return {
            "ok": False,
            "bytes": size,
            "bom": has_bom,
            "replacement_characters": replacement_characters,
            "error": f"UnicodeDecodeError at byte {exc.start}",
        }
    except OSError as exc:
        return {
            "ok": False,
            "bytes": size,
            "bom": has_bom,
            "replacement_characters": replacement_characters,
            "error": f"{type(exc).__name__}: file read failed",
        }


def inspect_data_encoding() -> dict[str, Any]:
    roots = [Path(config.MARKDOWN_DIR), ROOT / "datasets" / "structured"]
    suffixes = {".md", ".json", ".jsonl", ".txt", ".csv"}
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    files = sorted(set(files))
    failures: list[dict[str, Any]] = []
    bom_files: list[str] = []
    total_bytes = 0
    for path in files:
        result = inspect_utf8_file(path)
        total_bytes += int(result.get("bytes", 0))
        relative = str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)
        if result.get("bom"):
            bom_files.append(relative)
        if not result.get("ok"):
            failures.append({"path": relative, **result})
    return {
        "ok": not failures,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "failure_count": len(failures),
        "failures": failures[:20],
        "bom_count": len(bom_files),
        "bom_files": bom_files[:20],
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    detail: Any,
    *,
    severity: str = "error",
) -> None:
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "detail": detail,
        }
    )


def collect_report(
    *,
    require_cuda: bool,
    require_models: bool,
    check_qdrant: bool,
    check_data_encoding: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stdout_encoding = getattr(sys.stdout, "encoding", None)
    preferred_encoding = locale.getpreferredencoding(False)
    filesystem_encoding = sys.getfilesystemencoding()
    encoding_ok = all(
        is_utf8_encoding(value)
        for value in (stdout_encoding, preferred_encoding, filesystem_encoding)
    )
    _add_check(
        checks,
        "utf8_runtime",
        encoding_ok,
        {
            "stdout": stdout_encoding,
            "preferred": preferred_encoding,
            "filesystem": filesystem_encoding,
            "utf8_mode": sys.flags.utf8_mode,
            "LANG": os.environ.get("LANG", ""),
            "LC_ALL": os.environ.get("LC_ALL", ""),
        },
    )

    try:
        with tempfile.TemporaryDirectory(prefix="tcm_utf8_") as temporary:
            probe = Path(temporary) / "中医编码检查.txt"
            expected = "中医方证：恶寒、无汗、身痛。"
            probe.write_text(expected, encoding="utf-8", newline="\n")
            roundtrip_ok = probe.read_text(encoding="utf-8") == expected
    except Exception as exc:
        roundtrip_ok = False
        expected = f"{type(exc).__name__}: filesystem probe failed"
    _add_check(checks, "utf8_file_roundtrip", roundtrip_ok, expected)

    torch_info: dict[str, Any] = {"installed": False}
    cuda_available = False
    cuda_compute_ok = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ] if cuda_available else [],
        }
        if cuda_available:
            try:
                probe = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
                result = probe @ probe.T
                torch.cuda.synchronize()
                cuda_compute_ok = bool(torch.isfinite(result).all().item())
                torch_info["compute_smoke"] = "ok" if cuda_compute_ok else "non-finite-result"
            except Exception as exc:
                torch_info["compute_smoke"] = f"{type(exc).__name__}: CUDA compute failed"
        else:
            torch_info["compute_smoke"] = "not-run"
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: torch unavailable"
    _add_check(checks, "torch", torch_info["installed"], torch_info)
    _add_check(
        checks,
        "cuda",
        (cuda_available and cuda_compute_ok) or not require_cuda,
        torch_info,
        severity="error" if require_cuda else "warning",
    )

    configured_cuda = {
        "EMBEDDING_DEVICE": config.EMBEDDING_DEVICE,
        "SYNDROME_RERANK_DEVICE": config.SYNDROME_RERANK_DEVICE,
    }
    invalid_cuda_config = [
        name for name, value in configured_cuda.items()
        if value.startswith("cuda") and not (cuda_available and cuda_compute_ok)
    ]
    _add_check(
        checks,
        "configured_devices",
        not invalid_cuda_config,
        {"configured": configured_cuda, "unavailable": invalid_cuda_config},
    )

    model_status: dict[str, Any] = {}
    required_models = [config.DENSE_MODEL]
    if config.ENABLE_SYNDROME_RERANK:
        required_models.append(config.SYNDROME_RERANK_MODEL)
    for repo_id in dict.fromkeys(required_models):
        cached, detail = _model_cached(repo_id)
        model_status[repo_id] = {"cached": cached, "detail": detail}
        _add_check(
            checks,
            f"model_cache:{repo_id}",
            cached or not require_models,
            model_status[repo_id],
            severity="error" if require_models else "warning",
        )

    paths = {
        "markdown": config.MARKDOWN_DIR,
        "parent_store": config.PARENT_STORE_PATH,
        "qdrant_local": config.QDRANT_DB_PATH,
    }
    for name, raw_path in paths.items():
        path = Path(raw_path)
        required = name == "markdown" or (name == "qdrant_local" and not config.QDRANT_URL)
        _add_check(
            checks,
            f"path:{name}",
            path.exists() or not required,
            {"path": str(path), "exists": path.exists(), "absolute": path.is_absolute()},
            severity="error" if required else "warning",
        )

    qdrant_info: dict[str, Any] = {
        "backend": "remote" if config.QDRANT_URL else "local",
        "location": config.QDRANT_URL or config.QDRANT_DB_PATH,
        "prefer_grpc": config.QDRANT_PREFER_GRPC,
    }
    if check_qdrant:
        try:
            from db.qdrant_client_factory import create_qdrant_client

            client = create_qdrant_client()
            try:
                collections = sorted(item.name for item in client.get_collections().collections)
                qdrant_info["collections"] = collections
            finally:
                client.close()
            qdrant_ok = True
        except Exception as exc:
            qdrant_ok = False
            qdrant_info["error"] = f"{type(exc).__name__}: Qdrant check failed"
        _add_check(checks, "qdrant_connection", qdrant_ok, qdrant_info)

    data_encoding: dict[str, Any] | None = None
    if check_data_encoding:
        data_encoding = inspect_data_encoding()
        _add_check(checks, "data_files_utf8", data_encoding["ok"], data_encoding)

    errors = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warning"]
    return {
        "ok": not errors,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": _package_version("torch"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "qdrant-client": _package_version("qdrant-client"),
            "gradio": _package_version("gradio"),
        },
        "qdrant": qdrant_info,
        "models": model_status,
        "data_encoding": data_encoding,
        "checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查中医 RAG 的跨平台、UTF-8、CUDA 和模型运行环境")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--check-qdrant", action="store_true")
    parser.add_argument("--check-data-encoding", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = collect_report(
        require_cuda=args.require_cuda,
        require_models=args.require_models,
        check_qdrant=args.check_qdrant,
        check_data_encoding=args.check_data_encoding,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    if args.json_output:
        print(serialized)
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"Runtime preflight: {status}")
        for item in report["checks"]:
            marker = "OK" if item["ok"] else item["severity"].upper()
            print(f"[{marker}] {item['name']}: {item['detail']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
