#!/usr/bin/env python3
"""Report formal official-baseline readiness without launching experiments.

This command validates the immutable central protocol, commit-pinned author
checkouts, the locked Python environment, CUDA availability, and the HerbHG
source-tree identity. It is a readiness report only: a successful preflight
does not turn a smoke run into a formal result and does not certify that the
five-fold experiments have completed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from packaging.version import InvalidVersion, Version

from astra.benchmark.protocol import load_protocol_manifest

from .common import (
    FORMAL_RUNTIME_VERSIONS,
    OFFICIAL_BASELINES_LOCK_PATH,
    PROJECT_DIR,
    formal_runtime_module_origins,
    formal_runtime_versions,
    official_baseline_layout,
    project_git_identity,
    resolve_official_source_root,
    validate_frozen_input_manifest,
    verify_checkout_commit,
)


LOCK_PATH = OFFICIAL_BASELINES_LOCK_PATH


def _error_text(error: BaseException) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        stderr = (error.stderr or "").strip()
        if stderr:
            return stderr
    return str(error)


def _protocol_check(manifest_path: Path) -> tuple[dict[str, Any], object | None]:
    try:
        protocol = load_protocol_manifest(manifest_path)
        input_manifest = protocol.input_manifest()
        return (
            {
                "ready": True,
                "manifest": str(manifest_path),
                "dataset": protocol.dataset,
                "seed": int(protocol.seed),
                "folds": int(protocol.n_folds),
                "counts": input_manifest["counts"],
                "input_manifest_sha256": validate_frozen_input_manifest(
                    input_manifest
                ),
            },
            protocol,
        )
    except Exception as error:  # Keep the report machine-readable.
        return (
            {
                "ready": False,
                "manifest": str(manifest_path),
                "error": _error_text(error),
            },
            None,
        )


def _upstream_check(source_root: Path | None) -> dict[str, Any]:
    try:
        canonical_root, source_directories, lock = official_baseline_layout(LOCK_PATH)
        resolved_source_root = resolve_official_source_root(
            source_root, lock_path=LOCK_PATH
        )
        sources = lock["sources"]
    except Exception as error:
        return {
            "ready": False,
            "lock_file": str(LOCK_PATH),
            "error": f"cannot load official-baseline lock: {_error_text(error)}",
            "sources": {},
        }

    records: dict[str, Any] = {}
    for name, directory in source_directories.items():
        source_dir = (resolved_source_root / directory).resolve()
        record = sources.get(name)
        if not isinstance(record, dict) or not record.get("commit"):
            records[name] = {
                "ready": False,
                "path": str(source_dir),
                "error": "missing pinned commit in lock file",
            }
            continue
        expected = str(record["commit"])
        try:
            actual = verify_checkout_commit(source_dir, expected)
            records[name] = {
                "ready": True,
                "path": str(source_dir),
                "url": record.get("url"),
                "expected_commit": expected,
                "actual_commit": actual,
            }
        except Exception as error:
            records[name] = {
                "ready": False,
                "path": str(source_dir),
                "url": record.get("url"),
                "expected_commit": expected,
                "actual_commit": None,
                "error": _error_text(error),
            }
    return {
        "ready": bool(records) and all(item["ready"] for item in records.values()),
        "lock_file": str(LOCK_PATH),
        "source_root": str(resolved_source_root),
        "canonical_source_root": str(canonical_root),
        "requested_source_root": (
            None if source_root is None else str(Path(source_root).resolve())
        ),
        "sources": records,
    }


def _runtime_check() -> dict[str, Any]:
    versions = formal_runtime_versions(require_expected=False)

    def version_matches(name: str, expected: str, actual: str | None) -> bool:
        if name != "dgl":
            return actual == expected
        try:
            return Version(str(actual)).public == Version(expected).public
        except InvalidVersion:
            return False

    dgl_raw = versions.get("dgl")
    try:
        dgl_version = Version(str(dgl_raw))
        dgl_identity = {
            "raw": dgl_raw,
            "public": dgl_version.public,
            "local": dgl_version.local,
        }
    except InvalidVersion:
        dgl_identity = {"raw": dgl_raw, "public": None, "local": None}
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in FORMAL_RUNTIME_VERSIONS.items()
        if not version_matches(name, expected, versions.get(name))
    }
    origins = formal_runtime_module_origins(require_environment=False)
    environment_error = None
    try:
        formal_runtime_module_origins(require_environment=True)
    except Exception as error:
        environment_error = _error_text(error)
    return {
        "ready": not mismatches and environment_error is None,
        "versions": versions,
        "dgl_version_identity": dgl_identity,
        "expected_versions": FORMAL_RUNTIME_VERSIONS,
        "version_mismatches": mismatches,
        "module_origins": origins,
        "isolated_environment": environment_error is None,
        "environment_error": environment_error,
    }


def _cuda_check() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
        devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
        dgl_probe: dict[str, Any] = {"attempted": False, "ready": False}
        if available:
            dgl_probe["attempted"] = True
            try:
                dgl = importlib.import_module("dgl")
                device = torch.device("cuda:0")
                graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2).to(device)
                graph.ndata["h"] = torch.tensor(
                    [[1.0], [2.0]], dtype=torch.float32, device=device
                )
                graph.update_all(
                    dgl.function.copy_u("h", "m"),
                    dgl.function.sum("m", "h_out"),
                )
                output = graph.ndata["h_out"]
                torch.cuda.synchronize(device)
                graph_device = str(graph.device)
                output_device = str(output.device)
                if not graph_device.startswith("cuda") or not output_device.startswith(
                    "cuda"
                ):
                    raise RuntimeError(
                        "DGL CUDA probe returned non-CUDA state: "
                        f"graph={graph_device}, output={output_device}"
                    )
                dgl_probe.update(
                    {
                        "ready": True,
                        "device": str(device),
                        "graph_device": graph_device,
                        "output_device": output_device,
                    }
                )
            except Exception as error:
                dgl_probe["error"] = _error_text(error)
        ready = available and bool(dgl_probe["ready"])
        return {
            "ready": ready,
            "available": available,
            "torch_cuda_version": torch.version.cuda,
            "device_count": device_count,
            "devices": devices,
            "dgl_cuda_probe": dgl_probe,
            "note": (
                "PyTorch and DGL CUDA execution are ready for the full five-fold workload."
                if ready
                else (
                    "PyTorch CUDA is available, but the DGL CUDA execution probe failed."
                    if available
                    else "CUDA is unavailable; no completed formal run is inferred."
                )
            ),
        }
    except Exception as error:
        return {
            "ready": False,
            "available": False,
            "error": _error_text(error),
        }


def _framework_check() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in ("dgl", "torch_geometric"):
        try:
            module = importlib.import_module(name)
            record: dict[str, Any] = {
                "ready": True,
                "version": getattr(module, "__version__", "unknown"),
            }
            if name == "dgl":
                record["backend"] = module.backend.backend_name
            records[name] = record
        except Exception as error:
            records[name] = {"ready": False, "error": _error_text(error)}
    return {
        "ready": all(record["ready"] for record in records.values()),
        "modules": records,
    }


def _project_source_check() -> dict[str, Any]:
    try:
        identity = project_git_identity(require_clean=False)
        ready = bool(
            identity["all_protected_paths_tracked"]
            and identity["all_protected_paths_in_head"]
            and identity["protected_paths_clean"]
        )
        return {"ready": ready, "identity": identity}
    except Exception as error:
        head_process = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
        )
        no_head = head_process.returncode != 0
        return {
            "ready": False,
            "identity": {
                "repository_root": str(PROJECT_DIR),
                "head": None if no_head else head_process.stdout.strip(),
            },
            "error": (
                "HerbHG has no resolvable Git HEAD; formal source identity "
                "cannot be certified until the relevant code is committed."
                if no_head
                else _error_text(error)
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Parent of the lock-pinned author checkouts. By default this is "
            "read from third_party/baselines.lock.json. Passing the HerbHG "
            "project root is accepted as a convenience alias."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally write the same JSON report that is printed to stdout.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return a nonzero status when any formal-readiness check is blocked.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("DGLBACKEND", "pytorch")
    manifest_path = args.protocol_manifest.resolve()

    protocol, _ = _protocol_check(manifest_path)
    upstream = _upstream_check(args.source_root)
    runtime = _runtime_check()
    frameworks = _framework_check()
    cuda = _cuda_check()
    project_source = _project_source_check()

    checks = {
        "protocol": bool(protocol["ready"]),
        "upstream": bool(upstream["ready"]),
        "runtime": bool(runtime["ready"]),
        "framework_imports": bool(frameworks["ready"]),
        "project_source": bool(project_source["ready"]),
        "cuda": bool(cuda["ready"]),
    }
    blockers = [name for name, ready in checks.items() if not ready]
    formal_run_ready = not blockers
    return {
        "schema": "astra-official-baseline-preflight/v1",
        "status": (
            "formal-run-ready" if formal_run_ready else "formal-preflight-blocked"
        ),
        "formal_run_ready": formal_run_ready,
        "formal_source_ready": bool(project_source["ready"]),
        "blockers": blockers,
        "scope": (
            "Readiness only. This report does not execute or certify five-fold "
            "experiments, and smoke artifacts are never formal results."
        ),
        "protocol": protocol,
        "upstream": upstream,
        "runtime": runtime,
        "frameworks": frameworks,
        "cuda": cuda,
        "project_source": project_source,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        destination = args.output_json.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    return int(bool(args.require_ready and not payload["formal_run_ready"]))


if __name__ == "__main__":
    raise SystemExit(main())
