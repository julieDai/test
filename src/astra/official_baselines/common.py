"""Shared, auditable utilities for commit-pinned official baseline adapters.

The frozen benchmark uses one-based integer identifiers.  This module is the
single conversion boundary: every relation is validated and converted to
zero-based indices before it is exposed to a model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]
OFFICIAL_BASELINES_LOCK_PATH = (
    PROJECT_DIR / "third_party" / "baselines.lock.json"
)
DEFAULT_DATA_DIR = PROJECT_DIR / "artifacts" / "datasets" / "ethnobotany" / "relations"
DEFAULT_EMBEDDING_DIR = PROJECT_DIR / "artifacts" / "datasets" / "ethnobotany" / "embeddings"
DEFAULT_THIRD_PARTY_DIR = PROJECT_DIR / "third_party" / "checkouts"
DEFAULT_CACHE_DIR = PROJECT_DIR / "outputs" / "official_baselines" / "_cache"
FORMAL_PROJECT_SOURCE_ROOTS = (
    "src/astra/official_baselines",
    "src/astra/benchmark",
    "src/herbhg/baselines",
    "src/herbhg/models/backends",
)


def _formal_tree_files() -> tuple[str, ...]:
    files: list[str] = []
    for relative_root in FORMAL_PROJECT_SOURCE_ROOTS:
        root = PROJECT_DIR / relative_root
        if not root.is_dir():
            continue
        files.extend(
            str(path.relative_to(PROJECT_DIR))
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    return tuple(sorted(files))


FORMAL_PROJECT_SOURCE_PATHS = (
    "third_party/baselines.lock.json",
    "third_party/fetch_baselines.sh",
    *_formal_tree_files(),
)
FORMAL_PROJECT_TREE_AUXILIARY_PATHS = ()
FORMAL_RUNTIME_VERSIONS = {
    "python": "3.10.20",
    "numpy": "1.23.5",
    "pandas": "1.5.3",
    "scipy": "1.10.1",
    "scikit-learn": "1.2.2",
    "torch": "2.7.1+cu118",
    "torch-geometric": "2.7.0",
    # The locked CUDA runtime reports DGL's local CUDA build tag.  Requiring
    # the CPU-style base version here would reject the very GPU environment
    # used by formal author-code adapters.
    "dgl": "1.1.3+cu118",
}
FORMAL_RUNTIME_DISTRIBUTIONS = tuple(
    name for name in FORMAL_RUNTIME_VERSIONS if name != "python"
)
FORMAL_RUNTIME_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "torch-geometric": "torch_geometric",
    "dgl": "dgl",
}
FORMAL_PYTHON_ENV_POLICY = {
    "removed": ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT"],
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "42",
}


def official_baseline_layout(
    lock_path: Path | str = OFFICIAL_BASELINES_LOCK_PATH,
) -> tuple[Path, Dict[str, str], Dict[str, object]]:
    """Load the lock-owned checkout root and per-project directory names.

    The lock file is the single source of truth shared by the fetch script,
    preflight, and launcher. Paths must be repository-relative and cannot
    escape the declared checkout root.
    """

    path = Path(lock_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative_root = Path(str(payload["checkout_root"]))
        sources = payload["sources"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot load official-baseline layout from {path}: {error}"
        ) from error
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise RuntimeError(
            "official-baseline checkout_root must be a repository-relative path"
        )
    checkout_root = (PROJECT_DIR / relative_root).resolve()
    if PROJECT_DIR.resolve() not in checkout_root.parents:
        raise RuntimeError("official-baseline checkout_root escapes the HerbHG tree")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("official-baseline lock has no sources")

    directories: Dict[str, str] = {}
    resolved_directories: set[Path] = set()
    for name, record in sources.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise RuntimeError("official-baseline source records must be objects")
        directory = Path(str(record.get("directory", "")))
        if (
            not directory.parts
            or directory == Path(".")
            or directory.is_absolute()
            or ".." in directory.parts
        ):
            raise RuntimeError(
                f"invalid official-baseline directory for {name!r}: {directory}"
            )
        resolved = (checkout_root / directory).resolve()
        if checkout_root != resolved.parent and checkout_root not in resolved.parents:
            raise RuntimeError(
                f"official-baseline directory for {name!r} escapes checkout_root"
            )
        if resolved in resolved_directories:
            raise RuntimeError(
                f"duplicate official-baseline checkout directory: {directory}"
            )
        resolved_directories.add(resolved)
        directories[name] = directory.as_posix()
    return checkout_root, directories, payload


def resolve_official_source_root(
    source_root: Path | str | None,
    *,
    lock_path: Path | str = OFFICIAL_BASELINES_LOCK_PATH,
) -> Path:
    """Resolve the checkout parent used by preflight and model launchers.

    Passing the HerbHG project root is accepted as a convenience alias and is
    canonicalized to the lock-declared ``third_party/checkouts`` directory.
    An explicit alternate checkout parent remains supported for controlled
    mirrors.
    """

    canonical_root, _, _ = official_baseline_layout(lock_path)
    if source_root is None:
        return canonical_root
    requested = Path(source_root).resolve()
    if requested == PROJECT_DIR.resolve():
        return canonical_root
    return requested


def diagnostic_execution_context(device: str) -> Dict[str, object]:
    """Describe why an explicitly requested run is not formally eligible.

    Diagnostic mode is intentionally independent of the training schedule: it
    may use every frozen row and the complete epoch budget, but it must never
    be promoted to a formal result.  Keeping the reasons machine-readable in
    every runner prevents an output-directory name or a later aggregation
    command from silently changing that status.
    """

    device_policy = str(device).strip().lower()
    reasons = ["explicit diagnostic mode"]
    if device_policy == "cpu":
        reasons.append("CPU execution requested; formal execution requires CUDA")
    elif device_policy == "auto":
        reasons.append(
            "automatic device selection does not certify formal CUDA execution"
        )

    source: Dict[str, object]
    try:
        identity = project_git_identity(require_clean=False)
        source_ready = bool(
            identity.get("all_protected_paths_tracked")
            and identity.get("all_protected_paths_in_head")
            and identity.get("protected_paths_clean")
        )
        source = {
            "head": identity.get("head"),
            "ready": source_ready,
            "all_protected_paths_tracked": identity.get(
                "all_protected_paths_tracked"
            ),
            "all_protected_paths_in_head": identity.get(
                "all_protected_paths_in_head"
            ),
            "protected_paths_clean": identity.get("protected_paths_clean"),
        }
        if not source_ready:
            reasons.append(
                "ASTRA protected source is uncommitted, untracked, or differs from HEAD"
            )
    except Exception as error:  # A diagnostic must record, not hide, uncertainty.
        source = {"head": None, "ready": False, "error": str(error)}
        reasons.append("ASTRA protected source identity could not be certified")

    return {
        "enabled": True,
        "device_policy": device_policy,
        "project_source": source,
        "reasons": reasons,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Return the protocol's canonical dtype/shape/content array digest.

    Including the dtype and shape prevents differently interpreted byte
    buffers from sharing a provenance identifier.  This intentionally mirrors
    :func:`astra.benchmark.protocol._array_digest` without importing a private
    benchmark helper into the official adapters.
    """

    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _working_tree_bytes(path: Path) -> bytes | None:
    """Read the bytes Git would store for a regular file or symbolic link."""
    if path.is_symlink():
        return os.fsencode(os.readlink(path))
    if path.is_file():
        return path.read_bytes()
    return None


def _head_blob(source_dir: Path, relative_path: str) -> tuple[str | None, bytes | None]:
    """Return a HEAD blob id and payload without trusting index stat flags."""
    blob_process = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", f"HEAD:{relative_path}"],
        capture_output=True,
        text=True,
    )
    if blob_process.returncode != 0:
        return None, None
    blob = blob_process.stdout.strip()
    payload = subprocess.run(
        ["git", "-C", str(source_dir), "cat-file", "blob", blob],
        check=True,
        capture_output=True,
    ).stdout
    return blob, payload


def _tracked_head_drift(source_dir: Path) -> list[str]:
    """List tracked paths whose worktree bytes differ from the HEAD tree.

    Git's ordinary status/diff machinery may intentionally skip paths marked
    ``assume-unchanged`` or ``skip-worktree``.  Formal provenance cannot rely
    on those hints, so every HEAD blob is compared to the bytes on disk.
    """
    tree = subprocess.run(
        ["git", "-C", str(source_dir), "ls-tree", "-r", "-z", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    drift = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata_bytes, path_bytes = entry.split(b"\t", 1)
        _mode, object_type, object_id = metadata_bytes.split(b" ", 2)
        relative_path = os.fsdecode(path_bytes)
        if object_type != b"blob":
            drift.append(relative_path + " (non-blob tree entry)")
            continue
        expected = subprocess.run(
            [
                "git",
                "-C",
                str(source_dir),
                "cat-file",
                "blob",
                object_id.decode("ascii"),
            ],
            check=True,
            capture_output=True,
        ).stdout
        actual = _working_tree_bytes(source_dir / relative_path)
        if actual != expected:
            drift.append(relative_path)
    return drift


def _unexpected_formal_tree_entries(
    project_dir: Path,
    protected_paths: Iterable[str],
    protected_roots: Iterable[str],
    auxiliary_paths: Iterable[str],
) -> list[str]:
    """Reject files that could shadow imports outside the committed allowlist."""
    expected = set(protected_paths) | set(auxiliary_paths)
    unexpected = []
    for root_name in protected_roots:
        root = project_dir / root_name
        if not root.is_dir():
            unexpected.append(root_name + " (missing directory)")
            continue
        for path in root.rglob("*"):
            relative_parts = path.relative_to(root).parts
            if "__pycache__" in relative_parts:
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            relative = str(path.relative_to(project_dir))
            if relative not in expected:
                unexpected.append(relative)
    return sorted(unexpected)


def validate_frozen_input_manifest(manifest: Dict[str, object]) -> str:
    """Return a stable digest for a central protocol's verified input manifest.

    File validation and ID compaction belong to :mod:`astra.benchmark.protocol`.
    Official adapters deliberately do not maintain a second, dataset-specific
    lock file: the same central manifest is used by every method and dataset.
    Calling this helper before and after a run detects in-process input drift.
    """
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("central benchmark input manifest is empty or invalid")
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_git_identity(
    project_dir: Path | str = PROJECT_DIR,
    protected_paths: Iterable[str] = FORMAL_PROJECT_SOURCE_PATHS,
    protected_roots: Iterable[str] | None = None,
    tree_auxiliary_paths: Iterable[str] | None = None,
    *,
    require_clean: bool = True,
) -> Dict[str, object]:
    """Return and optionally enforce the trusted Git identity of formal code.

    Only the release-critical adapter/orchestrator paths are required to be
    clean, so unrelated work in the shared research repository does not block
    a formal run.
    """
    project_dir = Path(project_dir).resolve()
    protected = tuple(dict.fromkeys(str(path) for path in protected_paths))
    if protected_roots is None:
        protected_roots = (
            FORMAL_PROJECT_SOURCE_ROOTS
            if project_dir == PROJECT_DIR.resolve()
            and protected == FORMAL_PROJECT_SOURCE_PATHS
            else ()
        )
    if tree_auxiliary_paths is None:
        tree_auxiliary_paths = (
            FORMAL_PROJECT_TREE_AUXILIARY_PATHS
            if project_dir == PROJECT_DIR.resolve()
            and protected == FORMAL_PROJECT_SOURCE_PATHS
            else ()
        )
    protected_roots = tuple(
        dict.fromkeys(str(path) for path in protected_roots)
    )
    tree_auxiliary_paths = tuple(
        dict.fromkeys(str(path) for path in tree_auxiliary_paths)
    )
    repository_root = Path(
        subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    if repository_root != project_dir:
        raise RuntimeError(
            f"formal project root is {project_dir}, but Git resolves to {repository_root}"
        )
    head = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    listed = subprocess.run(
        ["git", "-C", str(project_dir), "ls-files", "--", *protected],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked = set(listed)
    untracked = [path for path in protected if path not in tracked]
    status_lines = subprocess.run(
        [
            "git",
            "-C",
            str(project_dir),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *protected,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    protected_files = {}
    head_missing = []
    head_mismatches = []
    for path in protected:
        disk_path = project_dir / path
        blob, head_payload = _head_blob(project_dir, path)
        disk_payload = _working_tree_bytes(disk_path)
        if head_payload is None:
            head_missing.append(path)
        elif disk_payload != head_payload:
            head_mismatches.append(path)
        protected_files[path] = {
            "sha256": (
                hashlib.sha256(disk_payload).hexdigest()
                if disk_payload is not None
                else None
            ),
            "git_blob": blob,
            "head_sha256": (
                hashlib.sha256(head_payload).hexdigest()
                if head_payload is not None
                else None
            ),
            "matches_head": bool(
                head_payload is not None and disk_payload == head_payload
            ),
        }
    unexpected_tree_entries = _unexpected_formal_tree_entries(
        project_dir, protected, protected_roots, tree_auxiliary_paths
    )
    protected_paths_clean = (
        not status_lines
        and not head_missing
        and not head_mismatches
        and not unexpected_tree_entries
    )
    identity = {
        "repository_root": str(repository_root),
        "head": head,
        "protected_paths": list(protected),
        "protected_files": protected_files,
        "all_protected_paths_tracked": not untracked,
        "all_protected_paths_in_head": not head_missing,
        "protected_paths_clean": protected_paths_clean,
        "untracked_protected_paths": untracked,
        "protected_paths_missing_from_head": head_missing,
        "protected_paths_differing_from_head": head_mismatches,
        "unexpected_protected_tree_entries": unexpected_tree_entries,
        "protected_status": status_lines,
    }
    if require_clean and (
        untracked
        or status_lines
        or head_missing
        or head_mismatches
        or unexpected_tree_entries
    ):
        details = []
        if untracked:
            details.append("not tracked: " + ", ".join(untracked))
        if head_missing:
            details.append("not present in HEAD: " + ", ".join(head_missing))
        if head_mismatches:
            details.append(
                "working-tree bytes differ from HEAD: "
                + ", ".join(head_mismatches)
            )
        if unexpected_tree_entries:
            details.append(
                "unexpected files in protected source tree: "
                + ", ".join(unexpected_tree_entries)
            )
        if status_lines:
            details.append("not clean: " + "; ".join(status_lines))
        raise RuntimeError(
            "formal official-baseline code must be committed and clean ("
            + " | ".join(details)
            + ")"
        )
    return identity


def verify_checkout_commit(source_dir: Path | str, expected_commit: str) -> str:
    """Fail closed when an upstream checkout is not the clean pinned revision."""
    source_dir = Path(source_dir).resolve()
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"{source_dir} is not a Git checkout; provenance cannot be verified")
    actual = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"unexpected upstream revision at {source_dir}: {actual}; "
            f"expected {expected_commit}"
        )
    content_drift = _tracked_head_drift(source_dir)
    status = subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    material_changes = []
    for line in status:
        path = line[3:].strip()
        if "__pycache__/" in path or path.endswith(".pyc"):
            continue
        material_changes.append(line)
    if material_changes:
        raise RuntimeError(
            f"upstream checkout at {source_dir} has material local changes: "
            + "; ".join(material_changes[:10])
        )
    if content_drift:
        raise RuntimeError(
            f"upstream checkout at {source_dir} differs byte-for-byte from HEAD: "
            + ", ".join(content_drift[:10])
        )
    return actual


def runtime_versions(*distributions: str) -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def formal_runtime_versions(*, require_expected: bool = False) -> Dict[str, str]:
    """Return the dependency identity validated for the formal adapters."""
    actual = runtime_versions(*FORMAL_RUNTIME_DISTRIBUTIONS)
    if require_expected and actual != FORMAL_RUNTIME_VERSIONS:
        mismatches = {
            name: {
                "actual": actual.get(name),
                "expected": expected,
            }
            for name, expected in FORMAL_RUNTIME_VERSIONS.items()
            if actual.get(name) != expected
        }
        raise RuntimeError(
            "formal official-baseline runtime differs from the validated lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def formal_runtime_module_origins(*, require_environment: bool = False) -> Dict[str, str]:
    """Resolve actual import origins and reject project/PYTHONPATH shadows."""
    origins = {}
    prefix = Path(sys.prefix).resolve()
    project = PROJECT_DIR.resolve()
    invalid = {}
    for distribution, module_name in FORMAL_RUNTIME_MODULES.items():
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
        if not origin or origin in {"built-in", "frozen"}:
            resolved = None
        else:
            resolved = Path(origin).resolve()
        origins[distribution] = str(resolved) if resolved is not None else str(origin)
        if (
            resolved is None
            or resolved == project
            or project in resolved.parents
            or (resolved != prefix and prefix not in resolved.parents)
        ):
            invalid[distribution] = origins[distribution]
    inherited = {
        name: os.environ[name]
        for name in FORMAL_PYTHON_ENV_POLICY["removed"]
        if os.environ.get(name)
    }
    incorrect = {
        name: {"actual": os.environ.get(name), "expected": expected}
        for name, expected in FORMAL_PYTHON_ENV_POLICY.items()
        if name != "removed" and os.environ.get(name) != expected
    }
    if require_environment and (invalid or inherited or incorrect):
        raise RuntimeError(
            "formal Python import environment is not isolated: "
            + json.dumps(
                {
                    "invalid_module_origins": invalid,
                    "inherited": inherited,
                    "incorrect_fixed_values": incorrect,
                },
                sort_keys=True,
            )
        )
    return origins


def _load_relation(path: Path, min_columns: int = 2) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if values.ndim != 2 or values.shape[1] < min_columns:
        raise ValueError(f"{path} must have at least {min_columns} columns")
    return values


def _zero_based(values: np.ndarray, sizes: Tuple[int, int], name: str) -> np.ndarray:
    pairs = np.asarray(values[:, :2], dtype=np.int64) - 1
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    lows = pairs.min(axis=0)
    highs = pairs.max(axis=0)
    if np.any(lows < 0) or highs[0] >= sizes[0] or highs[1] >= sizes[1]:
        raise ValueError(
            f"{name} identifiers are outside one-based ranges "
            f"1..{sizes[0]} and 1..{sizes[1]}: min={lows + 1}, max={highs + 1}"
        )
    return pairs


@dataclass(frozen=True)
class FoldRows:
    fold: int
    train_pairs: np.ndarray
    train_labels: np.ndarray
    test_pairs: np.ndarray
    test_labels: np.ndarray
    train_row_indices: np.ndarray
    test_row_indices: np.ndarray

    @property
    def train_positive_pairs(self) -> np.ndarray:
        return np.unique(self.train_pairs[self.train_labels == 1], axis=0)


def balanced_prefix_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    """Select a deterministic, ordered prefix from each binary class.

    This is used only by smoke tests.  Taking the first rows directly would
    select positives only because the frozen protocol concatenates all
    positive rows before all stored negatives.
    """
    labels = np.asarray(labels).reshape(-1)
    if per_class < 1:
        raise ValueError("per_class must be positive")
    positive = np.flatnonzero(labels == 1)[:per_class]
    negative = np.flatnonzero(labels == 0)[:per_class]
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required")
    return np.sort(np.concatenate((positive, negative))).astype(np.int64, copy=False)


def fold_split(fold_rows: object, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pairs, labels, row_id)`` from a central fold object.

    The adapters accept both the explicit ``*_row_id`` spelling used by the
    fair protocol and the older ``*_row_indices`` spelling during migration.
    A row identity is mandatory: silently using array position would sever the
    prediction artifact from the frozen fold manifest.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported fold split: {split}")
    pairs = np.asarray(getattr(fold_rows, f"{split}_pairs"), dtype=np.int64)
    labels = np.asarray(getattr(fold_rows, f"{split}_labels"), dtype=np.float32)
    row_id = None
    for suffix in ("row_id", "row_ids", "row_indices"):
        candidate = f"{split}_{suffix}"
        if hasattr(fold_rows, candidate):
            row_id = np.asarray(getattr(fold_rows, candidate))
            break
    if row_id is None:
        raise AttributeError(
            f"central fold object has no explicit row identity for {split}"
        )
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{split}_pairs must have shape (n, 2), got {pairs.shape}")
    if labels.ndim != 1 or row_id.ndim != 1:
        raise ValueError(f"{split} labels and row identities must be one-dimensional")
    if not len(pairs) == len(labels) == len(row_id):
        raise ValueError(f"{split} pairs, labels, and row identities differ in length")
    return pairs, labels, row_id


def outer_training_split(
    fold_rows: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore the complete outer-training fold in stable source-row order.

    The retained legacy comparison trained on all four non-test folds and
    selected threshold-dependent diagnostics on the test fold.  The central
    protocol still records an inner validation partition for auditability;
    this helper reunites the two disjoint outer-training pieces without
    changing row identity.  New fair runs should continue to use
    ``fold_split(fold_rows, "train")`` instead.
    """

    train_pairs, train_labels, train_row_id = fold_split(fold_rows, "train")
    validation_pairs, validation_labels, validation_row_id = fold_split(
        fold_rows, "validation"
    )
    pairs = np.concatenate((train_pairs, validation_pairs), axis=0)
    labels = np.concatenate((train_labels, validation_labels), axis=0)
    row_id = np.concatenate((train_row_id, validation_row_id), axis=0)
    order = np.argsort(row_id, kind="stable")
    if np.unique(row_id).size != len(row_id):
        raise ValueError("outer-training row identities are not unique")
    return pairs[order], labels[order], row_id[order]


def legacy_protocol_provenance(
    protocol: object,
    fold_rows: object,
    protocol_manifest: Path | str,
    *,
    project_identity: Dict[str, object] | None = None,
    require_clean_project: bool = True,
) -> Dict[str, object]:
    """Build the fail-closed provenance record for a retained legacy run.

    The legacy manuscript comparison is deliberately narrower than the fair
    protocol: it is the source-order TCM-suite benchmark trained on the whole
    outer-training partition.  Keeping this record construction in one place
    prevents the three author-code adapters from hashing subtly different row
    orders or payloads.

    ``project_identity`` is injectable so the independent auditor can compute
    the Git identity once and reuse it for all methods/folds.  Normal runner
    calls should omit it and therefore verify the protected project tree
    directly.
    """

    dataset = str(getattr(protocol, "dataset", ""))
    pair_order = str(getattr(protocol, "pair_order", ""))
    n_folds = int(getattr(protocol, "n_folds", 0))
    if dataset != "tcm-suite":
        raise ValueError("legacy protocol provenance is restricted to TCM-suite")
    if pair_order != "source":
        raise ValueError("legacy protocol provenance requires pair_order='source'")
    positive_pairs = np.asarray(getattr(protocol, "positive_pairs"), dtype=np.int64)
    if n_folds < 1 or len(positive_pairs) % n_folds != 0:
        raise ValueError(
            "legacy class-local modulo folds require the TCM positive-row count "
            "to be divisible by n_folds"
        )

    pairs, labels, row_id = outer_training_split(fold_rows)
    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels)
    row_id = np.asarray(row_id, dtype=np.int64)
    outer_positive_pairs = pairs[labels == 1]
    positive_rows = int(np.count_nonzero(labels == 1))
    negative_rows = int(np.count_nonzero(labels == 0))
    if positive_rows + negative_rows != len(labels):
        raise ValueError("legacy outer-training labels must be binary")

    protocol_manifest = Path(protocol_manifest).resolve()
    if not protocol_manifest.is_file():
        raise FileNotFoundError(
            f"central protocol manifest does not exist: {protocol_manifest}"
        )
    input_manifest = protocol.input_manifest()
    if project_identity is None:
        project_identity = project_git_identity(require_clean=require_clean_project)
    head = project_identity.get("head")
    protected_paths_clean = project_identity.get("protected_paths_clean")
    if not isinstance(head, str) or len(head) != 40:
        raise RuntimeError("formal project Git identity has no valid HEAD")
    if require_clean_project and protected_paths_clean is not True:
        raise RuntimeError("formal project protected paths are not clean")

    return {
        "schema_version": 1,
        "dataset": dataset,
        "pair_order": pair_order,
        "outer_training_rows": int(len(row_id)),
        "outer_training_positive_rows": positive_rows,
        "outer_training_negative_rows": negative_rows,
        "outer_training_row_id_sha256": array_sha256(row_id),
        "outer_training_positive_pairs_sha256": array_sha256(
            outer_positive_pairs
        ),
        "central_input_manifest_sha256": validate_frozen_input_manifest(
            input_manifest
        ),
        "protocol_manifest_sha256": sha256(protocol_manifest),
        "project_git_head": head,
        "project_git_protected_paths_clean": bool(protected_paths_clean),
    }


@dataclass(frozen=True)
class EthnobotanyProtocol:
    data_dir: Path
    embedding_dir: Path
    n_herbs: int
    n_compounds: int
    n_proteins: int
    n_diseases: int
    positive_pairs: np.ndarray
    negative_pairs: np.ndarray
    herb_compound_pairs: np.ndarray
    protein_disease_pairs: np.ndarray

    @property
    def all_pairs(self) -> np.ndarray:
        return np.concatenate((self.positive_pairs, self.negative_pairs), axis=0)

    @property
    def all_labels(self) -> np.ndarray:
        return np.concatenate(
            (
                np.ones(len(self.positive_pairs), dtype=np.float32),
                np.zeros(len(self.negative_pairs), dtype=np.float32),
            )
        )

    def fold_rows(self, fold: int, n_folds: int = 5) -> FoldRows:
        if not 1 <= fold <= n_folds:
            raise ValueError(f"fold must be in 1..{n_folds}, got {fold}")
        pairs = self.all_pairs
        labels = self.all_labels
        rows = np.arange(len(pairs), dtype=np.int64)
        test_mask = rows % n_folds == fold - 1
        return FoldRows(
            fold=fold,
            train_pairs=pairs[~test_mask],
            train_labels=labels[~test_mask],
            test_pairs=pairs[test_mask],
            test_labels=labels[test_mask],
            train_row_indices=rows[~test_mask],
            test_row_indices=rows[test_mask],
        )

    def incidence_matrices(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        hc = _binary_incidence(
            self.herb_compound_pairs[:, 0],
            self.herb_compound_pairs[:, 1],
            (self.n_herbs, self.n_compounds),
        )
        pd = _binary_incidence(
            self.protein_disease_pairs[:, 1],
            self.protein_disease_pairs[:, 0],
            (self.n_diseases, self.n_proteins),
        )
        return hc, pd

    def input_manifest(self) -> Dict[str, object]:
        names = (
            "H_D.txt",
            "negativehdlist.txt",
            "H_C.txt",
            "C_P.txt",
            "P_D.txt",
            "herb_id_mapping.csv",
            "compound_id_mapping.csv",
            "protein_id_mapping.csv",
            "disease_id_mapping.csv",
        )
        files = {}
        for name in names:
            path = self.data_dir / name
            if path.exists():
                files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name in ("compound_embedding_kpgt.txt", "protein_embedding_esm2.txt"):
            path = self.embedding_dir / name
            files[f"embeddings/{name}"] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        unique_negatives = np.unique(self.negative_pairs, axis=0)
        negative_pair_rows: Dict[Tuple[int, int], list[int]] = {}
        for row, pair in enumerate(self.negative_pairs):
            negative_pair_rows.setdefault(tuple(pair), []).append(row)
        cross_fold_negative_pairs = sum(
            len({row % 5 for row in rows}) > 1
            for rows in negative_pair_rows.values()
        )
        return {
            "id_convention_on_disk": "one-based",
            "id_convention_in_adapters": "zero-based",
            "fold_rule": "concatenate positive then stored negative rows; row_index modulo 5",
            "counts": {
                "herbs": self.n_herbs,
                "compounds": self.n_compounds,
                "proteins": self.n_proteins,
                "diseases": self.n_diseases,
                "positive_rows": len(self.positive_pairs),
                "negative_rows": len(self.negative_pairs),
                "unique_negative_pairs": len(unique_negatives),
                "repeated_negative_rows": len(self.negative_pairs) - len(unique_negatives),
                "negative_pairs_present_in_multiple_folds": cross_fold_negative_pairs,
            },
            "files": files,
        }


def load_protocol(protocol_manifest: Path | str):
    """Load the only protocol accepted by official baseline adapters.

    The manifest path is intentionally mandatory.  Falling back to a runner's
    private dataset discovery would make it possible for two methods to use
    different pair rows, negative samples, or compact-ID orders.
    """
    try:
        from astra.benchmark.protocol import load_protocol_manifest
    except ImportError:  # Package-relative execution.
        from ..benchmark.protocol import load_protocol_manifest

    path = Path(protocol_manifest).resolve()
    protocol = load_protocol_manifest(path)
    required = (
        "n_herbs",
        "n_compounds",
        "n_proteins",
        "n_diseases",
        "positive_pairs",
        "negative_pairs",
        "herb_compound_pairs",
        "protein_disease_pairs",
        "fold_rows",
        "incidence_matrices",
        "input_manifest",
    )
    missing = [name for name in required if not hasattr(protocol, name)]
    if missing:
        raise RuntimeError(
            "central benchmark protocol is missing adapter contract fields: "
            + ", ".join(missing)
        )
    return protocol


def _binary_incidence(rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> sparse.csr_matrix:
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data.fill(1.0)
    return matrix


def jaccard_similarity(incidence: sparse.csr_matrix) -> np.ndarray:
    """Dense Jaccard matrix over binary incidence rows."""
    incidence = incidence.astype(np.float32, copy=False)
    intersections = (incidence @ incidence.T).toarray().astype(np.float32, copy=False)
    degrees = np.asarray(incidence.sum(axis=1), dtype=np.float32).reshape(-1)
    unions = degrees[:, None] + degrees[None, :] - intersections
    similarity = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    nonempty = degrees > 0
    similarity[np.arange(len(degrees))[nonempty], np.arange(len(degrees))[nonempty]] = 1.0
    return similarity


def load_or_build_similarities(
    protocol: EthnobotanyProtocol,
    cache_dir: Path | str,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(
        getattr(protocol, "dataset", getattr(protocol, "dataset_name", "benchmark"))
    ).lower().replace("-", "_")
    source_hashes = {
        "herb_compound_pairs": hashlib.sha256(
            np.ascontiguousarray(protocol.herb_compound_pairs, dtype=np.int64).view(np.uint8)
        ).hexdigest(),
        "protein_disease_pairs": hashlib.sha256(
            np.ascontiguousarray(protocol.protein_disease_pairs, dtype=np.int64).view(np.uint8)
        ).hexdigest(),
        "shape": [int(protocol.n_herbs), int(protocol.n_diseases)],
    }
    cache_identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = (
        cache_dir / f"{dataset_name}_incidence_jaccard_{cache_identity}.npz"
    )
    # A sidecar hash is not a trust anchor because the cache and its sidecar
    # can be replaced together.  Derive the expected matrices from the frozen
    # incidence rows on every invocation (about one second for Ethnobotany),
    # then accept cached bytes only when their arrays are exactly equal.
    hc, pd = protocol.incidence_matrices()
    expected_herb = jaccard_similarity(hc)
    expected_disease = jaccard_similarity(pd)
    _validate_similarity_arrays(protocol, expected_herb, expected_disease)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("source_sha256") == source_hashes
            and metadata.get("cache_sha256") == sha256(cache_path)
        ):
            try:
                with np.load(cache_path) as cached:
                    herb = np.asarray(cached["herb"], dtype=np.float32)
                    disease = np.asarray(cached["disease"], dtype=np.float32)
                _validate_similarity_arrays(protocol, herb, disease)
                if np.array_equal(herb, expected_herb) and np.array_equal(
                    disease, expected_disease
                ):
                    return herb, disease, cache_path
            except (KeyError, OSError, ValueError):
                pass
    herb = expected_herb
    disease = expected_disease
    temporary_cache = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(temporary_cache, herb=herb, disease=disease)
    temporary_cache.replace(cache_path)
    write_json(
        metadata_path,
        {
            "definition": "Jaccard similarity over unique H-C and P-D incidence sets",
            "validation": "exact array equality to recomputation from frozen incidence rows",
            "source_sha256": source_hashes,
            "cache_sha256": sha256(cache_path),
            "herb_shape": list(herb.shape),
            "disease_shape": list(disease.shape),
        },
    )
    return herb, disease, cache_path


def _validate_similarity_arrays(
    protocol: EthnobotanyProtocol,
    herb: np.ndarray,
    disease: np.ndarray,
) -> None:
    expected = (
        (protocol.n_herbs, protocol.n_herbs),
        (protocol.n_diseases, protocol.n_diseases),
    )
    for name, matrix, shape in (
        ("herb", herb, expected[0]),
        ("disease", disease, expected[1]),
    ):
        if matrix.shape != shape:
            raise ValueError(f"cached {name} similarity has shape {matrix.shape}, expected {shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"cached {name} similarity contains NaN or infinity")
        if float(matrix.min()) < 0.0 or float(matrix.max()) > 1.0:
            raise ValueError(f"cached {name} similarity is outside [0, 1]")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6):
            raise ValueError(f"cached {name} similarity is not symmetric")


def deterministic_topk(similarity: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return non-self top-k neighbours with stable index tie-breaking."""
    similarity = np.asarray(similarity, dtype=np.float32)
    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity must be square")
    if not 1 <= k < n:
        raise ValueError(f"k must be in 1..{n - 1}")
    indices = np.arange(n, dtype=np.int64)
    neighbors = np.empty((n, k), dtype=np.int64)
    weights = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        values = similarity[row].copy()
        values[row] = -np.inf
        order = np.lexsort((indices, -values))[:k]
        neighbors[row] = order
        weights[row] = similarity[row, order]
    return neighbors, weights


def similarity_edge_tuple(similarity: np.ndarray, k: int):
    neighbors, weights = deterministic_topk(similarity, k)
    rows = np.repeat(np.arange(len(neighbors), dtype=np.int64), k)
    cols = neighbors.reshape(-1)
    values = weights.reshape(-1)
    return np.vstack((rows, cols)), values, similarity.shape


def load_pretrained_features(protocol: EthnobotanyProtocol) -> Tuple[np.ndarray, np.ndarray]:
    compounds = np.loadtxt(
        protocol.embedding_dir / "compound_embedding_kpgt.txt", dtype=np.float32
    )
    proteins = np.loadtxt(
        protocol.embedding_dir / "protein_embedding_esm2.txt", dtype=np.float32
    )
    expected = ((protocol.n_compounds, 64), (protocol.n_proteins, 64))
    if compounds.shape != expected[0] or proteins.shape != expected[1]:
        raise ValueError(
            f"feature shapes are {compounds.shape} and {proteins.shape}; expected {expected}"
        )
    if not np.isfinite(compounds).all() or not np.isfinite(proteins).all():
        raise ValueError("pretrained features contain NaN or infinity")
    return compounds, proteins


def compact_compound_protein_pairs(protocol: EthnobotanyProtocol) -> np.ndarray:
    """Map C_P PubChem identifiers to compact compound rows when possible."""
    if hasattr(protocol, "compound_protein_pairs"):
        pairs = np.asarray(protocol.compound_protein_pairs, dtype=np.int64)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.unique(pairs.reshape(-1, 2), axis=0)
    mapping = pd.read_csv(protocol.data_dir / "compound_id_mapping.csv")
    pubchem_to_compact = {
        int(pubchem): int(compact) - 1
        for compact, pubchem in mapping.iloc[:, :2].itertuples(index=False, name=None)
    }
    raw = _load_relation(protocol.data_dir / "C_P.txt")
    pairs = []
    for pubchem, protein in raw[:, :2]:
        compound = pubchem_to_compact.get(int(pubchem))
        protein_index = int(protein) - 1
        if compound is not None and 0 <= protein_index < protocol.n_proteins:
            pairs.append((compound, protein_index))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)


def evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    """Fail closed for the removed evaluated-fold threshold implementation.

    Historical adapters selected a maximum-F1 threshold on the test labels.
    Keeping that helper executable would make it too easy to accidentally
    publish test-oracle metrics again.  New code must call
    :func:`astra.benchmark.metrics.evaluate_validation_and_test`.
    """
    del labels, scores
    raise RuntimeError(
        "evaluate_scores was retired because it selected thresholds on test "
        "labels; use astra.benchmark.metrics.evaluate_validation_and_test"
    )


def write_json(path: Path | str, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
