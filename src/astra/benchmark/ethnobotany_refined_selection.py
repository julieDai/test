#!/usr/bin/env python3
"""Freeze validation-only ASTRA-v2 selection before outer-test scoring.

This module intentionally does not import or modify an ASTRA training runner.
Training code may emit candidate manifests with this shape::

    {
      "schema": "astra-ethnobotany-refined-candidate/v1",
      "fold": 1,
      "config": {
        "id": "epoch_budget__30",
        "stage_id": "S1_epoch_budget",
        "candidate_value": 30,
        "chain_index": 0,
        "incumbent_id": null,
        "base_config_id": "epoch_budget__30"
      },
      "snapshots": [{
        "epoch": 30,
        "validation_auprc": 0.87,
        "validation_predictions": {"path": "validation.npz", "sha256": "..."},
        "u_embedding": {"path": "u.npy", "sha256": "..."},
        "i_embedding": {"path": "i.npy", "sha256": "..."}
      }]
    }

Every referenced path must be relative to the manifest directory.  Candidate
directories are validation-only: any file whose name contains ``test`` makes
the candidate ineligible.  All candidates are audited before an immutable
a canonical JSON selection plus SHA-256 sidecar is atomically created.

Outer-test helpers first verify the selection sidecar, locked plan, protocol,
source-code closure, candidate manifest, validation prediction, and embedding
hashes.  The F1 threshold is committed during freeze from the selected
validation predictions.  Scoring reserves all folds before invoking exactly
one test-score operation per fold; partial or repeated scoring is rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics import (
    evaluate_test_scores,
    evaluate_validation_scores,
    load_predictions,
    save_predictions,
    select_f1_threshold,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .backend_paths import ETHNOBOTANY_BACKEND_ROOT


CANDIDATE_SCHEMA = "astra-ethnobotany-refined-candidate/v1"
SELECTION_SCHEMA = "astra-ethnobotany-refined-selection/v1"
FREEZE_CONTRACT_SCHEMA = "astra-ethnobotany-refined-freeze-contract/v1"
SELECTION_HASH_SUFFIX = ".sha256"
VALIDATION_METRIC_TOLERANCE = 1e-12
_SELECTOR_SOURCE = Path(__file__).resolve()
_SOURCE_ROOT = _SELECTOR_SOURCE.parents[2]
EXPECTED_EXECUTION_CODE_PATHS = {
    "model": ETHNOBOTANY_BACKEND_ROOT / "ASTRARefined.py",
    "runner": _SELECTOR_SOURCE.with_name("ethnobotany_refined_runner.py"),
    "selector": _SELECTOR_SOURCE,
    "metrics": _SELECTOR_SOURCE.with_name("metrics.py"),
    "protocol": _SELECTOR_SOURCE.with_name("protocol.py"),
    "astra_runner": _SELECTOR_SOURCE.with_name("astra_runner.py"),
    "semantic_features": _SELECTOR_SOURCE.with_name(
        "ethnobotany_semantic_features.py"
    ),
    "semantic_controls": _SELECTOR_SOURCE.with_name(
        "ethnobotany_semantic_controls.py"
    ),
    "feature_receipt": _SELECTOR_SOURCE.with_name(
        "ethnobotany_refined_feature_receipt.py"
    ),
    "features": _SELECTOR_SOURCE.with_name("features.py"),
    "data_validation": _SOURCE_ROOT / "astra" / "data_validation.py",
    "tf_initialization": _SOURCE_ROOT / "astra" / "tf_initialization.py",
    "herb_recommender": (
        ETHNOBOTANY_BACKEND_ROOT / "base" / "herbRecommender.py"
    ),
    "disease_recommender": (
        ETHNOBOTANY_BACKEND_ROOT / "base" / "diseaseRecommender.py"
    ),
    "recommender": ETHNOBOTANY_BACKEND_ROOT / "base" / "recommender.py",
    "rating": ETHNOBOTANY_BACKEND_ROOT / "rating.py",
    "model_config": ETHNOBOTANY_BACKEND_ROOT / "util" / "config.py",
    "feature_paths": ETHNOBOTANY_BACKEND_ROOT / "ethnobotany_feature_paths.py",
    "astra_paths": _SOURCE_ROOT / "astra" / "paths.py",
    "model_io": ETHNOBOTANY_BACKEND_ROOT / "util" / "io.py",
    "model_log": ETHNOBOTANY_BACKEND_ROOT / "util" / "log.py",
    "feature_bundle": _SELECTOR_SOURCE.with_name(
        "ethnobotany_feature_bundle.py"
    ),
    "tcm_feature_bundle": _SELECTOR_SOURCE.with_name("tcm_feature_bundle.py"),
    "official_baseline_common": (
        _SOURCE_ROOT / "astra" / "official_baselines" / "common.py"
    ),
}
REQUIRED_EXECUTION_CODE = frozenset(EXPECTED_EXECUTION_CODE_PATHS)
RUNTIME_RESOLVED_FIELDS = frozenset(
    {"learning_rate", "batch_size", "embedding_dimension", "reg_u"}
)
LEGACY_RUNTIME_DERIVATION = "hashed_config_and_locked_runner/v1"
SELECTION_METRIC = "inner-validation trapezoidal AUPRC"
SELECTION_TIE_BREAK = (
    "within each candidate choose highest validation AUPRC then earlier "
    "epoch; process locked-plan chain order and retain the lower-complexity "
    "incumbent unless the candidate reaches the practical margin"
)
OUTER_TEST_POLICY = (
    "all candidates audited before freeze; exactly one selected "
    "embedding pair may score each fold after hash verification"
)
FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA = (
    "astra-ethnobotany-refined-feature-deep-audit-receipt/v1"
)
STAGE_EVIDENCE_SCHEMA = "astra-ethnobotany-refined-stage-evidence/v1"


class RefinedSelectionError(RuntimeError):
    """Raised when selection or test isolation cannot be proven."""


@dataclass(frozen=True)
class _ValidationRows:
    train_pairs: np.ndarray
    train_labels: np.ndarray
    train_row_indices: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray
    validation_row_indices: np.ndarray


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    path = Path(path)
    if not path.is_file():
        raise RefinedSelectionError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RefinedSelectionError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RefinedSelectionError(f"JSON artifact must be an object: {path}")
    return payload


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _resolve_plan_artifact(plan_path: Path, raw_path: object, label: str) -> Path:
    """Resolve a plan path without depending on the caller's working directory."""

    if not isinstance(raw_path, str) or not raw_path:
        raise RefinedSelectionError(f"{label} must be a non-empty path string")
    supplied = Path(raw_path)
    if supplied.is_absolute():
        resolved = supplied.resolve(strict=False)
        if not resolved.is_file() and label.endswith("root"):
            return resolved
        return resolved
    matches: list[Path] = []
    for ancestor in (plan_path.parent, *plan_path.parents):
        candidate = (ancestor / supplied).resolve(strict=False)
        if candidate.exists():
            matches.append(candidate)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        # Canonical output paths commonly do not exist before freeze/scoring.
        if label.endswith("root") or label.endswith("path"):
            for ancestor in (plan_path.parent, *plan_path.parents):
                if (ancestor / "pyproject.toml").is_file():
                    return (ancestor / supplied).resolve(strict=False)
        raise RefinedSelectionError(f"cannot resolve {label} from locked plan: {raw_path}")
    raise RefinedSelectionError(
        f"{label} is ambiguous relative to locked plan: {raw_path}"
    )


def _verify_hash_record(
    plan_path: Path, record: object, label: str
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise RefinedSelectionError(f"{label} must contain path and sha256")
    path = _resolve_plan_artifact(plan_path, record.get("path"), f"{label} path")
    recorded = record.get("sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RefinedSelectionError(f"{label}.sha256 must be a SHA-256 string")
    if sha256(path) != recorded:
        raise RefinedSelectionError(f"{label} hash mismatch: {path}")
    return path, recorded


def _plan_protocol_record(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, str]:
    raw_path = plan.get("protocol_manifest")
    recorded = plan.get("protocol_manifest_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RefinedSelectionError(
            "locked plan protocol_manifest_sha256 must be a SHA-256 string"
        )
    path = _resolve_plan_artifact(plan_path, raw_path, "protocol manifest path")
    if sha256(path) != recorded:
        raise RefinedSelectionError("locked plan protocol manifest hash mismatch")
    return {"path": str(path), "sha256": recorded}


def _plan_feature_bundle_record(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, str]:
    raw_path = plan.get("feature_bundle")
    path = _resolve_plan_artifact(plan_path, raw_path, "feature bundle path")
    if path.is_dir():
        path = path / "bundle_manifest.json"
    recorded = plan.get("feature_bundle_manifest_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RefinedSelectionError(
            "locked plan feature_bundle_manifest_sha256 must be a SHA-256 string"
        )
    if sha256(path) != recorded:
        raise RefinedSelectionError("locked plan feature-bundle manifest hash mismatch")
    return {"path": str(path.resolve()), "sha256": recorded}


def _plan_core_sha256(plan: Mapping[str, Any]) -> str:
    """Hash immutable plan content while allowing a terminal stage ledger."""

    payload = json.loads(_canonical(plan))
    freeze = payload.get("selection_freeze")
    if isinstance(freeze, dict):
        freeze.pop("stage_ledger", None)
    return _canonical_digest(payload)


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validation_identity_digest(
    row_id: np.ndarray, pair: np.ndarray, label: np.ndarray
) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "row_id": _array_digest(row_id),
                "pair": _array_digest(pair),
                "label": _array_digest(label),
            }
        ).encode("ascii")
    ).hexdigest()


def _validation_rows_without_outer_materialisation(
    protocol: BenchmarkProtocol, fold: int
) -> _ValidationRows:
    """Derive inner rows without asking the protocol to construct test rows."""

    required = (
        "positive_pairs",
        "negative_pairs",
        "n_folds",
        "validation_fraction",
        "seed",
    )
    if not all(hasattr(protocol, name) for name in required):
        # Kept only for small protocol test doubles.  The formal
        # BenchmarkProtocol always takes the branch below.
        rows = protocol.fold_rows(fold)
        return _ValidationRows(
            train_pairs=np.asarray(rows.train_pairs, dtype=np.int64),
            train_labels=np.asarray(rows.train_labels, dtype=np.uint8),
            train_row_indices=np.asarray(rows.train_row_indices, dtype=np.int64),
            validation_pairs=np.asarray(rows.validation_pairs, dtype=np.int64),
            validation_labels=np.asarray(rows.validation_labels, dtype=np.uint8),
            validation_row_indices=np.asarray(
                rows.validation_row_indices, dtype=np.int64
            ),
        )

    fold = int(fold)
    n_folds = int(protocol.n_folds)
    if not 1 <= fold <= n_folds:
        raise RefinedSelectionError(
            f"fold must be in 1..{n_folds}, got {fold}"
        )
    positives = np.asarray(protocol.positive_pairs, dtype=np.int64)
    negatives = np.asarray(protocol.negative_pairs, dtype=np.int64)
    positive_count = len(positives)
    negative_count = len(negatives)
    positive_rows = np.arange(positive_count, dtype=np.int64)
    negative_local = np.arange(negative_count, dtype=np.int64)
    negative_rows = positive_count + negative_local
    positive_outer = positive_rows[positive_rows % n_folds != fold - 1]
    negative_outer = negative_rows[negative_local % n_folds != fold - 1]
    positive_rng = np.random.default_rng(
        np.random.SeedSequence((int(protocol.seed), fold, 1))
    )
    negative_rng = np.random.default_rng(
        np.random.SeedSequence((int(protocol.seed), fold, 0))
    )
    positive_outer = positive_outer[
        positive_rng.permutation(len(positive_outer))
    ]
    negative_outer = negative_outer[
        negative_rng.permutation(len(negative_outer))
    ]

    def validation_size(count: int) -> int:
        if count < 2:
            raise RefinedSelectionError(
                "each outer-training class needs at least two rows"
            )
        size = int(math.floor(count * float(protocol.validation_fraction)))
        return min(max(size, 1), count - 1)

    positive_validation_count = validation_size(len(positive_outer))
    negative_validation_count = validation_size(len(negative_outer))
    validation_rows = np.concatenate(
        (
            positive_outer[:positive_validation_count],
            negative_outer[:negative_validation_count],
        )
    )
    train_rows = np.concatenate(
        (
            positive_outer[positive_validation_count:],
            negative_outer[negative_validation_count:],
        )
    )

    def pairs_for(rows: np.ndarray) -> np.ndarray:
        positive_mask = rows < positive_count
        pairs = np.empty((len(rows), 2), dtype=np.int64)
        pairs[positive_mask] = positives[rows[positive_mask]]
        pairs[~positive_mask] = negatives[rows[~positive_mask] - positive_count]
        return pairs

    return _ValidationRows(
        train_pairs=pairs_for(train_rows),
        train_labels=np.asarray(train_rows < positive_count, dtype=np.uint8),
        train_row_indices=train_rows,
        validation_pairs=pairs_for(validation_rows),
        validation_labels=np.asarray(
            validation_rows < positive_count, dtype=np.uint8
        ),
        validation_row_indices=validation_rows,
    )


def _atomic_create(path: Path, content: bytes) -> None:
    """Create a complete file atomically and refuse to replace a frozen file."""

    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RefinedSelectionError(f"refusing to replace frozen artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RefinedSelectionError(
                f"another process froze artifact first: {path}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> str:
    if path.suffix.casefold() != ".json":
        raise RefinedSelectionError("the frozen decision file must be JSON")
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_name(path.name + SELECTION_HASH_SUFFIX)
    if sidecar.exists():
        raise RefinedSelectionError(f"selection hash sidecar already exists: {sidecar}")
    _atomic_create(path, encoded)
    _atomic_create(sidecar, f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _load_verified_selection(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve(strict=False)
    sidecar = path.with_name(path.name + SELECTION_HASH_SUFFIX)
    if not path.is_file() or not sidecar.is_file():
        raise RefinedSelectionError(
            f"frozen selection and hash sidecar must both exist: {path}"
        )
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except OSError as error:
        raise RefinedSelectionError(f"cannot read selection hash sidecar: {sidecar}") from error
    if len(fields) != 2 or fields[1] != path.name:
        raise RefinedSelectionError(f"invalid selection hash sidecar: {sidecar}")
    observed = sha256(path)
    if fields[0] != observed:
        raise RefinedSelectionError(
            f"selection hash mismatch: recorded={fields[0]}, observed={observed}"
        )
    payload = _load_json(path)
    if payload.get("schema") != SELECTION_SCHEMA:
        raise RefinedSelectionError(f"unexpected selection schema in {path}")
    return payload, observed


def _resolve_artifact(root: Path, record: object, label: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise RefinedSelectionError(f"{label} must contain path and sha256")
    raw_path = record.get("path")
    recorded_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise RefinedSelectionError(f"{label}.path must be a non-empty string")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise RefinedSelectionError(f"{label}.sha256 must be a SHA-256 string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise RefinedSelectionError(f"{label}.path must be relative: {raw_path}")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RefinedSelectionError(
            f"{label}.path escapes candidate root: {raw_path}"
        ) from error
    if sha256(resolved) != recorded_hash:
        raise RefinedSelectionError(f"{label} hash mismatch: {resolved}")
    return resolved, recorded_hash


def _reject_test_files(candidate_root: Path) -> None:
    forbidden = sorted(
        path
        for path in candidate_root.rglob("*")
        if path.is_file()
        and any(
            "test" in part.casefold()
            for part in path.relative_to(candidate_root).parts
        )
    )
    if forbidden:
        raise RefinedSelectionError(
            "validation-only candidate contains forbidden test file(s): "
            + ", ".join(str(path) for path in forbidden)
        )


def _load_embedding(path: Path) -> np.ndarray:
    suffix = path.suffix.casefold()
    try:
        if suffix == ".npy":
            matrix = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if len(payload.files) != 1:
                    raise RefinedSelectionError(
                        f"embedding NPZ must contain exactly one array: {path}"
                    )
                matrix = np.asarray(payload[payload.files[0]])
        else:
            matrix = np.loadtxt(path)
    except (OSError, ValueError) as error:
        raise RefinedSelectionError(f"cannot load embedding matrix {path}: {error}") from error
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise RefinedSelectionError(
            f"embedding must be a non-empty two-dimensional matrix: {path} {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise RefinedSelectionError(f"embedding contains NaN or infinity: {path}")
    return matrix


def _audit_snapshot_training_trace(
    snapshot: Mapping[str, Any],
    *,
    epoch: int,
    manifest: Mapping[str, Any],
    expected_rows: object,
) -> dict[str, object]:
    """Verify the technical-stop telemetry emitted for one checkpoint."""

    trace = snapshot.get("training")
    if not isinstance(trace, Mapping):
        raise RefinedSelectionError(
            f"snapshot {epoch} has no training diagnostics"
        )
    diagnostics = manifest.get("training_diagnostics")
    resolved = (
        diagnostics.get("resolved_hyperparameters")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(resolved, Mapping):
        raise RefinedSelectionError(
            f"snapshot {epoch} cannot bind training diagnostics"
        )
    batch_value = resolved.get("batch_size")
    if batch_value is None:
        config_record = manifest.get("config_source")
        if isinstance(config_record, Mapping):
            config_path = Path(str(config_record.get("path", ""))).resolve(
                strict=False
            )
            config_source = _load_json(config_path)
            training = config_source.get("training")
            if isinstance(training, Mapping):
                batch_value = training.get("batch_size")
    try:
        batch_size = int(batch_value)
    except (TypeError, ValueError) as error:
        raise RefinedSelectionError(
            f"snapshot {epoch} has no valid resolved batch size"
        ) from error
    if batch_size <= 0:
        raise RefinedSelectionError(
            f"snapshot {epoch} resolved batch size must be positive"
        )
    train_count = len(np.asarray(expected_rows.train_row_indices))
    expected_batches = int(math.ceil(train_count / batch_size))
    integer_fields = {
        "gradient_trace_batches": expected_batches,
        "optimizer_steps": epoch * expected_batches,
    }
    audited: dict[str, object] = {}
    for key, expected in integer_fields.items():
        value = trace.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} mismatch: {value!r} vs {expected}"
            )
        audited[key] = value

    finite_fields = (
        "mean_training_loss",
        "preclip_gradient_norm",
        "mean_preclip_gradient_norm",
        "mean_pre_global_clip_gradient_norm",
        "mean_postclip_gradient_norm",
        "preclip_exceeds_clip_norm_fraction",
        "global_clip_trigger_fraction",
    )
    for key in finite_fields:
        value = trace.get(key)
        if isinstance(value, bool):
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} must be finite"
            )
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} must be finite"
            ) from error
        if not math.isfinite(parsed):
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} must be finite"
            )
        if parsed < 0.0:
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} must be non-negative"
            )
        audited[key] = parsed
    for key in (
        "preclip_exceeds_clip_norm_fraction",
        "global_clip_trigger_fraction",
    ):
        if not 0.0 <= float(audited[key]) <= 1.0:
            raise RefinedSelectionError(
                f"snapshot {epoch} training {key} must lie in [0, 1]"
            )
    if not math.isclose(
        float(audited["preclip_gradient_norm"]),
        float(audited["mean_preclip_gradient_norm"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RefinedSelectionError(
            f"snapshot {epoch} preclip gradient aliases disagree"
        )
    clip_norm = resolved.get("gradient_clip_norm")
    if clip_norm is not None:
        try:
            clip_value = float(clip_norm)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"snapshot {epoch} resolved clip norm is invalid"
            ) from error
        tolerance = max(1e-6, abs(clip_value) * 1e-6)
        if float(audited["mean_postclip_gradient_norm"]) > clip_value + tolerance:
            raise RefinedSelectionError(
                f"snapshot {epoch} post-clip gradient exceeds the resolved clip norm"
            )
    return audited


def _candidate_values(stage: Mapping[str, Any]) -> list[object]:
    values = stage.get("candidate_values")
    if isinstance(values, list) and values:
        return list(values)
    fallback = [stage[key] for key in ("incumbent", "candidate") if key in stage]
    if not fallback:
        raise RefinedSelectionError(
            f"plan stage has no ordered candidates: {stage.get('id')!r}"
        )
    return fallback


def _plan_candidate_rank(
    plan: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[int, int]:
    config_id = config.get("id")
    if not isinstance(config_id, str) or not config_id:
        raise RefinedSelectionError("candidate config.id must be a non-empty string")
    explicit_order = plan.get("candidate_config_order")
    if isinstance(explicit_order, list):
        if config_id not in explicit_order:
            raise RefinedSelectionError(
                f"candidate config {config_id!r} is absent from plan order"
            )
        return 0, explicit_order.index(config_id)

    stage_id = config.get("stage_id")
    if not isinstance(stage_id, str) or "candidate_value" not in config:
        raise RefinedSelectionError(
            "candidate config requires stage_id and candidate_value when the "
            "plan has no candidate_config_order"
        )
    stages = plan.get("controlled_stages")
    if not isinstance(stages, list):
        raise RefinedSelectionError("plan controlled_stages must be a list")
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or stage.get("id") != stage_id:
            continue
        target = _canonical(config["candidate_value"])
        values = _candidate_values(stage)
        for candidate_index, value in enumerate(values):
            if _canonical(value) == target:
                return stage_index, candidate_index
        raise RefinedSelectionError(
            f"candidate value is absent from plan stage {stage_id}: "
            f"{config['candidate_value']!r}"
        )
    raise RefinedSelectionError(f"candidate stage is absent from plan: {stage_id}")


def _control_leaf_values(
    value: object, *, prefix: str = ""
) -> dict[str, object]:
    """Flatten the executable part of a candidate config for exact diffs."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_control_leaf_values(value[key], prefix=child))
        return result
    if isinstance(value, list):
        return {prefix: json.loads(_canonical(value))}
    return {prefix: value}


def _candidate_control_payload(source: Mapping[str, Any]) -> dict[str, object]:
    option_defaults: dict[str, object] = {
        "astra.n_layers": 5,
        "astra.loss.kind": "legacy",
        "astra.loss.reduction": "sum",
        "astra.grad.clip_value": 10.0,
        "astra.grad.clip_norm": 0.5,
        "astra.protein_gate": False,
        "astra.dead_variables": "preserve",
        "astra.clique.sampling": "legacy_prefix",
        "astra.clique.max_pairs_per_hyperedge": 100,
        "astra.clique.max_directed_edges": 200_000,
        "astra.semantic.adapter": "identity",
        "astra.semantic.input_dim": 64,
        "astra.semantic.freeze_input": False,
        # The controlled runner forces telemetry on for every formal candidate.
        "astra.training.trace_gradients": True,
    }
    raw_options = source.get("model_options", {})
    if not isinstance(raw_options, Mapping):
        raise RefinedSelectionError("candidate model_options must be an object")
    options = {**option_defaults, **dict(raw_options)}
    options["astra.training.trace_gradients"] = True
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"schema", "config", "model_options"}
    }
    payload["model_options"] = options
    return _control_leaf_values(payload)


def _control_values_equal(left: object, right: object) -> bool:
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-15
        )
    return _canonical(left) == _canonical(right)


def _validate_fixed_config_controls(
    source: Mapping[str, Any], plan: Mapping[str, Any], *, config_id: str
) -> None:
    fixed = plan.get("fixed_controls")
    if not isinstance(fixed, Mapping):
        raise RefinedSelectionError("locked plan fixed_controls must be an object")
    training = source.get("training")
    if not isinstance(training, Mapping):
        raise RefinedSelectionError(
            f"candidate config {config_id} has no training controls"
        )
    bindings = {
        "batch_size": ("batch_size", int),
        "latent_dimension": ("embedding_dimension", int),
        "regularization_u": ("reg_u", float),
        "text_fusion_alpha": ("text_fusion_alpha", float),
    }
    for plan_key, (training_key, parser) in bindings.items():
        if plan_key not in fixed:
            raise RefinedSelectionError(
                f"locked plan fixed_controls is missing {plan_key}"
            )
        observed_raw = training.get(training_key)
        expected_raw = fixed[plan_key]
        if isinstance(observed_raw, bool) or isinstance(expected_raw, bool):
            raise RefinedSelectionError(
                f"candidate config {config_id} has an invalid fixed control "
                f"{plan_key}"
            )
        try:
            observed = parser(observed_raw)
            expected = parser(expected_raw)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"candidate config {config_id} has an invalid fixed control "
                f"{plan_key}"
            ) from error
        if isinstance(observed, float):
            matches = math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=1e-15
            )
        else:
            matches = observed == expected
        if not matches:
            raise RefinedSelectionError(
                f"candidate config {config_id} violates fixed control {plan_key}"
            )
    required_epochs = list(_required_epoch_grid(plan))
    observed_checkpoints = training.get("checkpoint_epochs")
    if observed_checkpoints != required_epochs:
        raise RefinedSelectionError(
            f"candidate config {config_id} checkpoint grid differs from plan"
        )
    hard_limit = plan.get("promotion_and_stop_rules", {}).get(
        "hard_epoch_limit"
    )
    if hard_limit is None or training.get("epochs") != hard_limit:
        raise RefinedSelectionError(
            f"candidate config {config_id} epoch limit differs from plan"
        )


def _registry_fold_sources(
    *,
    plan_path: Path,
    record: Mapping[str, Any],
    expected_folds: Sequence[int],
    label: str,
) -> dict[int, tuple[Path, str]]:
    path, digest = _verify_hash_record(
        plan_path,
        {"path": record.get("config_path"), "sha256": record.get("config_sha256")},
        label,
    )
    folds_raw = record.get("folds")
    if (
        not isinstance(folds_raw, list)
        or any(isinstance(fold, bool) or not isinstance(fold, int) for fold in folds_raw)
    ):
        raise RefinedSelectionError(f"{label} folds must be an integer list")
    folds = list(folds_raw)
    if (
        not folds
        or len(folds) != len(set(folds))
        or not set(folds).issubset(set(expected_folds))
    ):
        raise RefinedSelectionError(
            f"{label} folds must be a non-empty subset of the outer folds: "
            f"{sorted(folds)} vs {sorted(expected_folds)}"
        )
    overrides = record.get("fold_config_sources", {})
    if not isinstance(overrides, Mapping):
        raise RefinedSelectionError(f"{label} fold_config_sources must be an object")
    result = {fold: (path, digest) for fold in folds}
    observed_override_folds: set[int] = set()
    for raw_fold, override in overrides.items():
        if isinstance(raw_fold, bool):
            raise RefinedSelectionError(f"{label} has an invalid fold override")
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"{label} has an invalid fold override"
            ) from error
        if fold in observed_override_folds or fold not in result:
            raise RefinedSelectionError(f"{label} has an invalid fold override")
        observed_override_folds.add(fold)
        result[fold] = _verify_hash_record(
            plan_path, override, f"{label} fold {fold} config"
        )
    return result


def _load_candidate_config_registry(
    plan_path: Path,
    plan: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    expected_folds: Sequence[int],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Load the precommitted config registry and prove one-factor diffs."""

    raw = freeze.get("candidate_config_registry")
    if not isinstance(raw, list) or not raw:
        raise RefinedSelectionError(
            "selection_freeze.candidate_config_registry must be non-empty"
        )
    registry: dict[tuple[int, str], dict[str, Any]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise RefinedSelectionError(
                f"candidate_config_registry[{index}] must be an object"
            )
        config_id = record.get("config_id")
        stage_id = record.get("stage_id")
        if not isinstance(config_id, str) or not config_id:
            raise RefinedSelectionError(
                f"candidate_config_registry[{index}] has no config_id"
            )
        if not isinstance(stage_id, str) or not stage_id:
            raise RefinedSelectionError(
                f"candidate config registry {config_id} has no stage_id"
            )
        allowed_raw = record.get("allowed_changes_from_incumbent")
        if (
            not isinstance(allowed_raw, list)
            or any(not isinstance(path, str) or not path for path in allowed_raw)
            or len(allowed_raw) != len(set(allowed_raw))
        ):
            raise RefinedSelectionError(
                f"candidate config registry {config_id} requires unique "
                "allowed_changes_from_incumbent paths"
            )
        if any(path.startswith("config") for path in allowed_raw):
            raise RefinedSelectionError(
                f"candidate config registry {config_id} cannot whitelist identity fields"
            )
        sources = _registry_fold_sources(
            plan_path=plan_path,
            record=record,
            expected_folds=expected_folds,
            label=f"candidate config registry {config_id}",
        )
        for fold, (path, digest) in sources.items():
            key = (fold, config_id)
            if key in registry:
                raise RefinedSelectionError(
                    f"duplicate candidate config registry entry: fold {fold} {config_id}"
                )
            source = _load_json(path)
            identity = source.get("config")
            if (
                not isinstance(identity, Mapping)
                or identity.get("id") != config_id
                or identity.get("stage_id") != stage_id
            ):
                raise RefinedSelectionError(
                    f"candidate config registry identity mismatch: fold {fold} {config_id}"
                )
            _plan_candidate_rank(plan, identity)
            _validate_fixed_config_controls(source, plan, config_id=config_id)
            registry[key] = {
                "stage_id": stage_id,
                "config_id": config_id,
                "config_path": str(path),
                "config_sha256": digest,
                "allowed_changes_from_incumbent": sorted(allowed_raw),
                "source": source,
            }

    for (fold, config_id), record in registry.items():
        identity = record["source"]["config"]
        incumbent_id = identity.get("incumbent_id")
        allowed = record["allowed_changes_from_incumbent"]
        if incumbent_id is None:
            if allowed:
                raise RefinedSelectionError(
                    f"base candidate {config_id} cannot declare incumbent changes"
                )
            continue
        incumbent = registry.get((fold, str(incumbent_id)))
        if incumbent is None:
            raise RefinedSelectionError(
                f"candidate {config_id} fold {fold} incumbent is absent from "
                "candidate_config_registry"
            )
        current_controls = _candidate_control_payload(record["source"])
        incumbent_controls = _candidate_control_payload(incumbent["source"])
        observed = sorted(
            key
            for key in set(current_controls) | set(incumbent_controls)
            if key not in current_controls
            or key not in incumbent_controls
            or not _control_values_equal(
                current_controls[key], incumbent_controls[key]
            )
        )
        if observed != allowed:
            raise RefinedSelectionError(
                f"candidate {config_id} fold {fold} violates its single-factor "
                f"config contract: observed {observed}, allowed {allowed}"
            )
    return registry


def _verify_feature_dependency_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RefinedSelectionError(f"{label} must contain path/sha256/bytes")
    path = Path(str(record.get("path", ""))).resolve(strict=False)
    digest = record.get("sha256")
    size = record.get("bytes")
    if (
        not path.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or path.stat().st_size != size
        or sha256(path) != digest
    ):
        raise RefinedSelectionError(f"{label} differs from its deep-audit receipt")
    return path


def _load_semantic_feature_registry(
    plan_path: Path,
    freeze: Mapping[str, Any],
    *,
    feature_bundle_record: Mapping[str, str],
) -> dict[tuple[str, str], dict[str, str]]:
    raw = freeze.get("semantic_feature_registry")
    if not isinstance(raw, list) or not raw:
        raise RefinedSelectionError(
            "selection_freeze.semantic_feature_registry must be non-empty"
        )
    registry: dict[tuple[str, str], dict[str, str]] = {}
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise RefinedSelectionError(
                f"semantic_feature_registry[{index}] must be an object"
            )
        semantic_path, semantic_hash = _verify_hash_record(
            plan_path,
            {
                "path": record.get("semantic_manifest_path"),
                "sha256": record.get("semantic_manifest_sha256"),
            },
            f"semantic feature registry[{index}] manifest",
        )
        receipt_path, receipt_hash = _verify_hash_record(
            plan_path,
            {
                "path": record.get("deep_audit_receipt_path"),
                "sha256": record.get("deep_audit_receipt_sha256"),
            },
            f"semantic feature registry[{index}] deep receipt",
        )
        receipt = _load_json(receipt_path)
        if (
            receipt.get("schema") != FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA
            or receipt.get("dataset") != "ethnobotany"
            or receipt.get("deep_audit") is not True
            or receipt.get("path_bound") is not True
            or receipt.get("all_dependencies_rehashed_on_reuse") is not True
        ):
            raise RefinedSelectionError(
                f"semantic feature registry[{index}] deep receipt is invalid"
            )
        bundle_identity = _verify_feature_dependency_record(
            receipt.get("bundle_manifest"), label="deep receipt bundle manifest"
        )
        semantic_identity = _verify_feature_dependency_record(
            receipt.get("semantic_reduction_manifest"),
            label="deep receipt semantic manifest",
        )
        if (
            bundle_identity != Path(feature_bundle_record["path"])
            or receipt["bundle_manifest"].get("sha256")
            != feature_bundle_record["sha256"]
            or semantic_identity != semantic_path
            or receipt["semantic_reduction_manifest"].get("sha256")
            != semantic_hash
        ):
            raise RefinedSelectionError(
                f"semantic feature registry[{index}] receipt identity mismatch"
            )
        dependencies = receipt.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise RefinedSelectionError(
                f"semantic feature registry[{index}] has no dependency closure"
            )
        dependency_paths = [
            _verify_feature_dependency_record(
                dependency,
                label=f"semantic feature registry[{index}] dependency[{dep_index}]",
            )
            for dep_index, dependency in enumerate(dependencies)
        ]
        if len(dependency_paths) != len(set(dependency_paths)):
            raise RefinedSelectionError(
                f"semantic feature registry[{index}] dependency closure is duplicated"
            )
        required_paths = {bundle_identity, semantic_identity}
        if not required_paths.issubset(set(dependency_paths)):
            raise RefinedSelectionError(
                f"semantic feature registry[{index}] dependency closure is incomplete"
            )
        key = (str(semantic_path), semantic_hash)
        if key in registry:
            raise RefinedSelectionError("duplicate semantic feature registry entry")
        registry[key] = {
            "path": str(semantic_path),
            "sha256": semantic_hash,
            "deep_audit_receipt_path": str(receipt_path),
            "deep_audit_receipt_sha256": receipt_hash,
        }
    return registry


def _load_freeze_contract(
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    require_terminal: bool,
) -> dict[str, Any]:
    """Validate the explicit completed/skipped stage ledger.

    The ledger is intentionally terminal metadata and is excluded from the
    immutable plan-core digest.  This permits a conditionally stopped stage to
    be declared ``skipped`` without changing the protocol, candidate grid,
    fixed controls, code contract, or promotion rule used during training.
    """

    freeze = plan.get("selection_freeze")
    if not isinstance(freeze, Mapping):
        raise RefinedSelectionError("locked plan has no selection_freeze contract")
    if freeze.get("schema") != FREEZE_CONTRACT_SCHEMA:
        raise RefinedSelectionError(
            f"selection_freeze.schema must be {FREEZE_CONTRACT_SCHEMA!r}"
        )
    selection_path = _resolve_plan_artifact(
        plan_path,
        freeze.get("canonical_selection_path"),
        "canonical selection path",
    )
    if selection_path.suffix.casefold() != ".json":
        raise RefinedSelectionError(
            "canonical_selection_path must end in .json"
        )
    outer_test_root = _resolve_plan_artifact(
        plan_path,
        freeze.get("outer_test_output_root"),
        "outer test root",
    )
    code = freeze.get("execution_code")
    if not isinstance(code, Mapping) or not code:
        raise RefinedSelectionError(
            "selection_freeze.execution_code must be a non-empty object"
        )
    missing_code = sorted(REQUIRED_EXECUTION_CODE.difference(code))
    if missing_code:
        raise RefinedSelectionError(
            "selection_freeze.execution_code is missing required source(s): "
            + ", ".join(missing_code)
        )
    code_records: dict[str, dict[str, str]] = {}
    for name, record in code.items():
        path, digest = _verify_hash_record(
            plan_path, record, f"execution_code.{name}"
        )
        expected_path = EXPECTED_EXECUTION_CODE_PATHS.get(str(name))
        if expected_path is not None and path != expected_path.resolve():
            raise RefinedSelectionError(
                f"execution_code.{name} points to the wrong source: {path}"
            )
        code_records[str(name)] = {"path": str(path), "sha256": digest}

    raw_stages = plan.get("controlled_stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise RefinedSelectionError("locked plan controlled_stages must be non-empty")
    stage_ids: list[str] = []
    for stage in raw_stages:
        stage_id = stage.get("id") if isinstance(stage, Mapping) else None
        if not isinstance(stage_id, str) or not stage_id:
            raise RefinedSelectionError("every controlled stage needs a non-empty id")
        stage_ids.append(stage_id)
    if len(stage_ids) != len(set(stage_ids)):
        raise RefinedSelectionError("locked plan contains duplicate controlled stages")

    ledger = freeze.get("stage_ledger")
    if not isinstance(ledger, list):
        raise RefinedSelectionError("selection_freeze.stage_ledger must be a list")
    ledger_ids = [
        entry.get("stage_id") if isinstance(entry, Mapping) else None
        for entry in ledger
    ]
    if ledger_ids != stage_ids:
        raise RefinedSelectionError(
            "stage ledger must list every controlled stage exactly once in plan order"
        )
    expected_folds = plan.get("outer_folds")
    if not isinstance(expected_folds, list) or not expected_folds:
        raise RefinedSelectionError("locked plan outer_folds must be non-empty")
    folds = [int(fold) for fold in expected_folds]
    if len(folds) != len(set(folds)):
        raise RefinedSelectionError("locked plan outer_folds contains duplicates")
    config_registry = _load_candidate_config_registry(
        plan_path, plan, freeze, expected_folds=folds
    )
    feature_bundle_record = _plan_feature_bundle_record(plan_path, plan)
    semantic_registry = _load_semantic_feature_registry(
        plan_path,
        freeze,
        feature_bundle_record=feature_bundle_record,
    )

    candidate_contracts: dict[tuple[int, str], dict[str, Any]] = {}
    stage_status: dict[str, str] = {}
    evidence_records: list[dict[str, str]] = []
    for stage_index, entry in enumerate(ledger):
        if not isinstance(entry, Mapping):
            raise RefinedSelectionError("stage ledger entries must be objects")
        stage_id = str(entry["stage_id"])
        plan_stage = raw_stages[stage_index]
        assert isinstance(plan_stage, Mapping)
        is_gate_stage = "pass_rule" in plan_stage
        status = entry.get("status")
        if status not in {"completed", "skipped", "pending"}:
            raise RefinedSelectionError(
                f"stage {stage_id} status must be completed, skipped, or pending"
            )
        stage_status[stage_id] = str(status)
        if require_terminal and status == "pending":
            raise RefinedSelectionError(
                f"stage {stage_id} is still pending; selection cannot be frozen"
            )
        candidates = entry.get("candidates", [])
        if not isinstance(candidates, list):
            raise RefinedSelectionError(
                f"stage {stage_id} candidates must be a list"
            )
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list):
            raise RefinedSelectionError(
                f"stage {stage_id} evidence must be a list"
            )
        gate_only_raw = entry.get("gate_only", False)
        if not isinstance(gate_only_raw, bool):
            raise RefinedSelectionError(
                f"stage {stage_id} gate_only must be boolean"
            )
        gate_only = gate_only_raw
        if is_gate_stage and (status != "completed" or gate_only is not True):
            raise RefinedSelectionError(
                f"plan gate stage {stage_id} must be completed as gate_only"
            )
        if not is_gate_stage and gate_only:
            raise RefinedSelectionError(
                f"stage {stage_id} is not declared as a plan gate"
            )

        def record_evidence(*, expected_outcome: str) -> None:
            for index, record in enumerate(evidence):
                path, digest = _verify_hash_record(
                    plan_path, record, f"stage {stage_id} evidence[{index}]"
                )
                payload = _load_json(path)
                if (
                    payload.get("schema") != STAGE_EVIDENCE_SCHEMA
                    or payload.get("stage_id") != stage_id
                    or payload.get("plan_stage_sha256")
                    != _canonical_digest(plan_stage)
                    or payload.get("outcome") != expected_outcome
                ):
                    raise RefinedSelectionError(
                        f"stage {stage_id} evidence[{index}] does not prove "
                        f"the {expected_outcome} outcome"
                    )
                evidence_records.append(
                    {"path": str(path), "sha256": digest, "stage_id": stage_id}
                )

        if status == "skipped":
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise RefinedSelectionError(
                    f"skipped stage {stage_id} requires a non-empty reason"
                )
            if candidates:
                raise RefinedSelectionError(
                    f"skipped stage {stage_id} cannot declare candidates"
                )
            if not evidence:
                raise RefinedSelectionError(
                    f"skipped stage {stage_id} requires hashed evidence"
                )
            record_evidence(expected_outcome="skipped")
            continue
        if status == "pending":
            continue
        if gate_only:
            if candidates:
                raise RefinedSelectionError(
                    f"gate-only stage {stage_id} cannot declare candidates"
                )
            if not isinstance(evidence, list) or not evidence:
                raise RefinedSelectionError(
                    f"completed gate-only stage {stage_id} requires evidence"
                )
            record_evidence(expected_outcome="passed")
            continue
        if not candidates:
            raise RefinedSelectionError(
                f"completed stage {stage_id} requires candidate contracts"
            )
        stage_value_folds: dict[str, set[int]] = {}
        for index, contract in enumerate(candidates):
            if not isinstance(contract, Mapping):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate[{index}] must be an object"
                )
            config_id = contract.get("config_id")
            if not isinstance(config_id, str) or not config_id:
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate[{index}] lacks config_id"
                )
            config_path, config_hash = _verify_hash_record(
                plan_path,
                {
                    "path": contract.get("config_path"),
                    "sha256": contract.get("config_sha256"),
                },
                f"stage {stage_id} candidate {config_id} config",
            )
            fold_config_sources = contract.get("fold_config_sources", {})
            if not isinstance(fold_config_sources, Mapping):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} "
                    "fold_config_sources must be an object"
                )
            semantic_path, semantic_hash = _verify_hash_record(
                plan_path,
                {
                    "path": contract.get("semantic_manifest_path"),
                    "sha256": contract.get("semantic_manifest_sha256"),
                },
                f"stage {stage_id} candidate {config_id} semantics",
            )
            semantic_key = (str(semantic_path), semantic_hash)
            semantic_contract = semantic_registry.get(semantic_key)
            if semantic_contract is None:
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} semantic manifest "
                    "is absent from the precommitted semantic_feature_registry"
                )
            contract_folds = contract.get("folds")
            if not isinstance(contract_folds, list) or not contract_folds:
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} folds must be non-empty"
                )
            parsed_folds = [int(fold) for fold in contract_folds]
            if (
                not parsed_folds
                or len(parsed_folds) != len(set(parsed_folds))
                or not set(parsed_folds).issubset(set(folds))
            ):
                raise RefinedSelectionError(
                    f"completed stage {stage_id} candidate {config_id} folds "
                    f"must be a non-empty subset: {sorted(parsed_folds)} vs {folds}"
                )
            legacy_runtime_folds_raw = contract.get(
                "legacy_resolved_runtime_folds", []
            )
            if not isinstance(legacy_runtime_folds_raw, list):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} "
                    "legacy_resolved_runtime_folds must be a list"
                )
            if any(
                isinstance(fold, bool)
                or not isinstance(fold, int)
                for fold in legacy_runtime_folds_raw
            ):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} "
                    "legacy_resolved_runtime_folds must contain integer folds"
                )
            legacy_runtime_folds = set(legacy_runtime_folds_raw)
            if (
                len(legacy_runtime_folds) != len(legacy_runtime_folds_raw)
                or not legacy_runtime_folds.issubset(set(parsed_folds))
            ):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} has invalid "
                    "legacy_resolved_runtime_folds"
                )
            try:
                override_folds = {int(fold) for fold in fold_config_sources}
            except (TypeError, ValueError) as error:
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} has invalid "
                    "fold_config_sources keys"
                ) from error
            if (
                len(override_folds) != len(fold_config_sources)
                or not override_folds.issubset(set(parsed_folds))
            ):
                raise RefinedSelectionError(
                    f"stage {stage_id} candidate {config_id} has invalid "
                    "fold_config_sources keys"
                )
            normalized = {
                "stage_id": stage_id,
                "config_id": config_id,
                "semantic_manifest_path": str(semantic_path),
                "semantic_manifest_sha256": semantic_hash,
            }
            for fold in parsed_folds:
                key = (fold, config_id)
                if key in candidate_contracts:
                    raise RefinedSelectionError(
                        f"duplicate candidate contract for fold {fold}: {config_id}"
                    )
                fold_config_path = config_path
                fold_config_hash = config_hash
                override = fold_config_sources.get(str(fold))
                if override is None:
                    override = fold_config_sources.get(fold)
                if override is not None:
                    fold_config_path, fold_config_hash = _verify_hash_record(
                        plan_path,
                        override,
                        (
                            f"stage {stage_id} candidate {config_id} "
                            f"fold {fold} config"
                        ),
                    )
                candidate_contracts[key] = {
                    **normalized,
                    "config_path": str(fold_config_path),
                    "config_sha256": fold_config_hash,
                    "feature_deep_audit_receipt_path": semantic_contract[
                        "deep_audit_receipt_path"
                    ],
                    "feature_deep_audit_receipt_sha256": semantic_contract[
                        "deep_audit_receipt_sha256"
                    ],
                    "legacy_resolved_runtime_source": (
                        LEGACY_RUNTIME_DERIVATION
                        if fold in legacy_runtime_folds
                        else None
                    ),
                }
                registered = config_registry.get(key)
                if registered is None:
                    raise RefinedSelectionError(
                        f"stage {stage_id} candidate {config_id} fold {fold} is "
                        "absent from candidate_config_registry"
                    )
                if (
                    registered["stage_id"] != stage_id
                    or registered["config_path"] != str(fold_config_path)
                    or registered["config_sha256"] != fold_config_hash
                ):
                    raise RefinedSelectionError(
                        f"stage {stage_id} candidate {config_id} fold {fold} "
                        "differs from candidate_config_registry"
                    )
                candidate_value_key = _canonical(
                    registered["source"]["config"]["candidate_value"]
                )
                value_folds = stage_value_folds.setdefault(
                    candidate_value_key, set()
                )
                if fold in value_folds:
                    raise RefinedSelectionError(
                        f"stage {stage_id} candidate value "
                        f"{candidate_value_key} duplicates fold {fold}"
                    )
                value_folds.add(fold)
        for candidate_value, covered_folds in stage_value_folds.items():
            if covered_folds != set(folds):
                raise RefinedSelectionError(
                    f"completed stage {stage_id} candidate value "
                    f"{candidate_value} must cover every outer fold: "
                    f"{sorted(covered_folds)} vs {folds}"
                )

    if require_terminal and not candidate_contracts:
        raise RefinedSelectionError(
            "terminal stage ledger contains no selectable candidate contracts"
        )
    return {
        "selection_path": selection_path,
        "outer_test_output_root": outer_test_root,
        "execution_code": code_records,
        "candidate_config_registry": [
            {
                "fold": fold,
                "config_id": config_id,
                "stage_id": record["stage_id"],
                "config_path": record["config_path"],
                "config_sha256": record["config_sha256"],
                "allowed_changes_from_incumbent": record[
                    "allowed_changes_from_incumbent"
                ],
            }
            for (fold, config_id), record in sorted(config_registry.items())
        ],
        "semantic_feature_registry": [
            dict(record)
            for _, record in sorted(semantic_registry.items())
        ],
        "candidate_contracts": candidate_contracts,
        "stage_status": stage_status,
        "evidence": evidence_records,
    }


def _freeze_contract_receipt(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": FREEZE_CONTRACT_SCHEMA,
        "canonical_selection_path": str(contract["selection_path"]),
        "outer_test_output_root": str(contract["outer_test_output_root"]),
        "stage_status": dict(contract["stage_status"]),
        "execution_code": dict(contract["execution_code"]),
        "candidate_config_registry": list(contract["candidate_config_registry"]),
        "semantic_feature_registry": list(contract["semantic_feature_registry"]),
        "evidence": list(contract["evidence"]),
    }


def _validate_chain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fields needed to replay the lower-complexity chain."""

    required = {
        "id",
        "stage_id",
        "candidate_value",
        "chain_index",
        "incumbent_id",
        "base_config_id",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise RefinedSelectionError(
            "candidate config lacks promotion-chain field(s): " + ", ".join(missing)
        )
    config_id = config.get("id")
    stage_id = config.get("stage_id")
    base_id = config.get("base_config_id")
    incumbent_id = config.get("incumbent_id")
    chain_index = config.get("chain_index")
    if any(
        not isinstance(value, str) or not value
        for value in (config_id, stage_id, base_id)
    ):
        raise RefinedSelectionError(
            "config id, stage_id, and base_config_id must be non-empty strings"
        )
    if (
        isinstance(chain_index, bool)
        or not isinstance(chain_index, int)
        or chain_index < 0
    ):
        raise RefinedSelectionError(
            "candidate config.chain_index must be a non-negative integer"
        )
    if chain_index == 0:
        if incumbent_id is not None or base_id != config_id:
            raise RefinedSelectionError(
                "promotion-chain root requires incumbent_id=null and "
                "base_config_id=id"
            )
    elif not isinstance(incumbent_id, str) or not incumbent_id:
        raise RefinedSelectionError(
            "non-root promotion candidate requires a non-empty incumbent_id"
        )
    return dict(config)


def _parse_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RefinedSelectionError(f"{label} must be boolean")


def _optional_float(value: object, *, label: str) -> float | None:
    if value is None or str(value).strip().lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RefinedSelectionError(f"{label} must be numeric or none") from error
    if not math.isfinite(parsed):
        raise RefinedSelectionError(f"{label} must be finite")
    return parsed


def _expected_resolved_hyperparameters(
    candidate_source: Mapping[str, Any], *, seed: int
) -> dict[str, object]:
    training = candidate_source.get("training")
    options = candidate_source.get("model_options", {})
    if not isinstance(training, Mapping) or not isinstance(options, Mapping):
        raise RefinedSelectionError(
            "candidate config source requires training and model_options objects"
        )

    def integer(name: str, default: int) -> int:
        value = options.get(name, default)
        if isinstance(value, bool):
            raise RefinedSelectionError(f"candidate option {name} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"candidate option {name} must be an integer"
            ) from error
        return parsed

    try:
        epochs = int(training["epochs"])
        batch_size = int(training["batch_size"])
        embedding_dimension = int(training["embedding_dimension"])
        learning_rate = float(training["learning_rate"])
        reg_u = float(training["reg_u"])
        text_alpha = float(training["text_fusion_alpha"])
        checkpoints = [int(value) for value in training["checkpoint_epochs"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RefinedSelectionError(
            "candidate config has invalid epoch/alpha/checkpoint settings"
        ) from error
    expected: dict[str, object] = {
        "model_class": "ASTRARefined",
        "seed": int(seed),
        "max_epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "embedding_dimension": embedding_dimension,
        "reg_u": reg_u,
        "n_layers": integer("astra.n_layers", 5),
        "loss_kind": str(options.get("astra.loss.kind", "legacy")).strip().lower(),
        "loss_reduction": str(
            options.get("astra.loss.reduction", "sum")
        ).strip().lower(),
        "gradient_clip_value": _optional_float(
            options.get("astra.grad.clip_value", 10.0),
            label="astra.grad.clip_value",
        ),
        "gradient_clip_norm": _optional_float(
            options.get("astra.grad.clip_norm", 0.5),
            label="astra.grad.clip_norm",
        ),
        "protein_gate": _parse_bool(
            options.get("astra.protein_gate", False),
            label="astra.protein_gate",
        ),
        "dead_variables": str(
            options.get("astra.dead_variables", "preserve")
        ).strip().lower(),
        "clique_sampling": str(
            options.get("astra.clique.sampling", "legacy_prefix")
        ).strip().lower(),
        "clique_max_pairs_per_hyperedge": integer(
            "astra.clique.max_pairs_per_hyperedge", 100
        ),
        "clique_max_directed_edges": integer(
            "astra.clique.max_directed_edges", 200_000
        ),
        "checkpoint_epochs": checkpoints,
        "semantic_adapter": str(
            options.get("astra.semantic.adapter", "identity")
        ).strip().lower(),
        "semantic_input_dim": integer("astra.semantic.input_dim", 64),
        "semantic_freeze_input": _parse_bool(
            options.get("astra.semantic.freeze_input", False),
            label="astra.semantic.freeze_input",
        ),
        # The controlled runner deliberately forces identical telemetry for all
        # candidates, even if a source config attempts to disable it.
        "trace_gradients": True,
        "text_fusion_alpha": text_alpha,
    }
    expected["c0_compatible"] = bool(
        expected["n_layers"] == 5
        and expected["loss_kind"] == "legacy"
        and expected["loss_reduction"] == "sum"
        and expected["gradient_clip_value"] == 10.0
        and expected["gradient_clip_norm"] == 0.5
        and expected["protein_gate"] is False
        and expected["dead_variables"] == "preserve"
        and expected["clique_sampling"] == "legacy_prefix"
        and expected["clique_max_pairs_per_hyperedge"] == 100
        and expected["clique_max_directed_edges"] == 200_000
        and expected["semantic_adapter"] == "identity"
        and expected["semantic_input_dim"] == 64
        and expected["semantic_freeze_input"] is False
    )
    return expected


def _validate_candidate_provenance(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    protocol_record: Mapping[str, str],
    feature_bundle_record: Mapping[str, str],
    expected_rows: object,
) -> None:
    """Bind one candidate to the locked protocol, config, features, and runtime."""

    candidate_protocol = manifest.get("protocol")
    if not isinstance(candidate_protocol, Mapping):
        raise RefinedSelectionError("candidate manifest has no protocol binding")
    try:
        candidate_seed = int(candidate_protocol.get("seed"))
        plan_seed = int(plan.get("seed"))
    except (TypeError, ValueError) as error:
        raise RefinedSelectionError(
            "candidate or locked-plan seed is invalid"
        ) from error
    candidate_protocol_path = Path(
        str(candidate_protocol.get("path", ""))
    ).resolve(strict=False)
    if (
        candidate_protocol_path != Path(protocol_record["path"])
        or candidate_protocol.get("sha256") != protocol_record["sha256"]
        or sha256(candidate_protocol_path) != protocol_record["sha256"]
        or candidate_seed != plan_seed
    ):
        raise RefinedSelectionError("candidate protocol differs from locked plan")
    if manifest.get("validation_only") is not True:
        raise RefinedSelectionError("candidate is not declared validation-only")
    if manifest.get("outer_evaluation_rows_loaded") is not False:
        raise RefinedSelectionError(
            "candidate reports that outer-evaluation rows were loaded"
        )

    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise RefinedSelectionError("candidate manifest has no split binding")
    train_rows = np.asarray(expected_rows.train_row_indices, dtype=np.int64)
    validation_rows = np.asarray(
        expected_rows.validation_row_indices, dtype=np.int64
    )
    expected_split = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_row_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(train_rows).tobytes()
        ).hexdigest(),
        "validation_row_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(validation_rows).tobytes()
        ).hexdigest(),
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise RefinedSelectionError(
                f"candidate split binding differs from protocol at {key}"
            )

    config_record = manifest.get("config_source")
    if not isinstance(config_record, Mapping):
        raise RefinedSelectionError("candidate manifest has no config_source")
    config_path = Path(str(config_record.get("path", ""))).resolve(strict=False)
    if (
        config_path != Path(str(contract["config_path"]))
        or config_record.get("sha256") != contract["config_sha256"]
        or sha256(config_path) != contract["config_sha256"]
    ):
        raise RefinedSelectionError("candidate config source differs from stage ledger")
    config_source = _load_json(config_path)
    if config_source.get("config") != manifest.get("config"):
        raise RefinedSelectionError(
            "candidate identity differs from its hashed config source"
        )
    if config_source.get("schema") != config_record.get("schema"):
        raise RefinedSelectionError("candidate config-source schema mismatch")

    features = manifest.get("features")
    if not isinstance(features, Mapping):
        raise RefinedSelectionError("candidate manifest has no feature binding")
    bundle = features.get("base_bundle")
    if not isinstance(bundle, Mapping):
        raise RefinedSelectionError(
            "candidate base feature bundle differs from plan"
        )
    bundle_path = Path(str(bundle.get("path", ""))).resolve(strict=False)
    if (
        bundle_path != Path(feature_bundle_record["path"])
        or bundle.get("sha256") != feature_bundle_record["sha256"]
        or sha256(bundle_path) != feature_bundle_record["sha256"]
    ):
        raise RefinedSelectionError("candidate base feature bundle differs from plan")
    semantic = features.get("semantic_reduction")
    if not isinstance(semantic, Mapping):
        raise RefinedSelectionError("candidate has no semantic-reduction binding")
    semantic_path = Path(str(semantic.get("path", ""))).resolve(strict=False)
    if (
        semantic_path != Path(str(contract["semantic_manifest_path"]))
        or semantic.get("sha256") != contract["semantic_manifest_sha256"]
        or sha256(semantic_path) != contract["semantic_manifest_sha256"]
    ):
        raise RefinedSelectionError(
            "candidate semantic manifest differs from stage ledger"
        )

    diagnostics = manifest.get("training_diagnostics")
    resolved = (
        diagnostics.get("resolved_hyperparameters")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(resolved, Mapping):
        raise RefinedSelectionError(
            "candidate has no resolved_hyperparameters execution receipt"
        )
    expected_resolved = _expected_resolved_hyperparameters(
        config_source, seed=int(plan.get("seed"))
    )
    missing_runtime = RUNTIME_RESOLVED_FIELDS.difference(resolved)
    legacy_runtime_source = contract.get("legacy_resolved_runtime_source")
    if missing_runtime:
        if (
            missing_runtime != RUNTIME_RESOLVED_FIELDS
            or legacy_runtime_source != LEGACY_RUNTIME_DERIVATION
        ):
            raise RefinedSelectionError(
                "candidate is missing resolved runtime hyperparameters: "
                + ", ".join(sorted(missing_runtime))
            )
    elif legacy_runtime_source is not None:
        raise RefinedSelectionError(
            "legacy resolved-runtime compatibility was declared for a candidate "
            "that records the runtime fields directly"
        )
    for key, expected in expected_resolved.items():
        if key in missing_runtime:
            continue
        actual = resolved.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise RefinedSelectionError(
                f"candidate resolved hyperparameter mismatch at {key}: "
                f"expected={expected!r}, observed={actual!r}"
            )


def _practical_margin(plan: Mapping[str, Any]) -> float:
    rules = plan.get("promotion_and_stop_rules")
    raw = (
        rules.get("per_fold_practical_margin_auprc")
        if isinstance(rules, Mapping)
        else None
    )
    try:
        margin = float(raw)
    except (TypeError, ValueError) as error:
        raise RefinedSelectionError(
            "locked plan must define a finite non-negative per-fold AUPRC margin"
        ) from error
    if not math.isfinite(margin) or margin < 0.0:
        raise RefinedSelectionError(
            "locked plan per-fold AUPRC margin must be finite and non-negative"
        )
    return margin


def _required_epoch_grid(plan: Mapping[str, Any]) -> tuple[int, ...]:
    epoch_plan = plan.get("epoch_selection")
    raw_epochs = (
        epoch_plan.get("candidate_epochs")
        if isinstance(epoch_plan, Mapping)
        else None
    )
    if not isinstance(raw_epochs, list) or not raw_epochs:
        raise RefinedSelectionError(
            "locked plan must declare a non-empty candidate epoch grid"
        )
    epochs: list[int] = []
    for raw_epoch in raw_epochs:
        if isinstance(raw_epoch, bool):
            raise RefinedSelectionError(
                "locked candidate epochs must be positive integers"
            )
        try:
            epoch = int(raw_epoch)
        except (TypeError, ValueError) as error:
            raise RefinedSelectionError(
                "locked candidate epochs must be positive integers"
            ) from error
        if epoch <= 0 or epoch != raw_epoch:
            raise RefinedSelectionError(
                "locked candidate epochs must be positive integers"
            )
        epochs.append(epoch)
    if len(epochs) != len(set(epochs)):
        raise RefinedSelectionError("locked candidate epoch grid has duplicates")
    return tuple(epochs)


def _validate_epoch(plan: Mapping[str, Any], epoch: object) -> int:
    if isinstance(epoch, bool):
        raise RefinedSelectionError("snapshot epoch must be a positive integer")
    try:
        parsed = int(epoch)
    except (TypeError, ValueError) as error:
        raise RefinedSelectionError("snapshot epoch must be a positive integer") from error
    if parsed <= 0 or parsed != epoch:
        raise RefinedSelectionError("snapshot epoch must be a positive integer")
    allowed = _required_epoch_grid(plan)
    if parsed not in allowed:
        raise RefinedSelectionError(
            "snapshot epoch "
            f"{parsed} is absent from locked candidate epochs {list(allowed)}"
        )
    return parsed


def audit_candidate_manifest(
    manifest_path: str | Path,
    *,
    plan: Mapping[str, Any],
    plan_path: Path | None = None,
    contract: Mapping[str, Any] | None = None,
    protocol_record: Mapping[str, str] | None = None,
    feature_bundle_record: Mapping[str, str] | None = None,
    expected_rows: object | None = None,
) -> dict[str, Any]:
    """Validate one candidate and recompute every snapshot's validation AUPRC."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise RefinedSelectionError(f"unexpected candidate schema: {manifest_path}")
    try:
        fold = int(manifest["fold"])
    except (KeyError, TypeError, ValueError) as error:
        raise RefinedSelectionError(f"candidate fold is invalid: {manifest_path}") from error
    expected_folds = plan.get("outer_folds")
    if not isinstance(expected_folds, list) or fold not in expected_folds:
        raise RefinedSelectionError(f"candidate fold {fold} is absent from locked plan")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping):
        raise RefinedSelectionError(f"candidate config is missing: {manifest_path}")
    config = _validate_chain_config(raw_config)
    rank = _plan_candidate_rank(plan, config)
    candidate_root = manifest_path.parent.resolve()
    _reject_test_files(candidate_root)
    strict_arguments = (
        plan_path,
        contract,
        protocol_record,
        feature_bundle_record,
        expected_rows,
    )
    if any(value is not None for value in strict_arguments):
        if any(value is None for value in strict_arguments):
            raise RefinedSelectionError(
                "strict candidate audit requires plan, contract, protocol, and rows"
            )
        if contract.get("stage_id") != config.get("stage_id") or contract.get(
            "config_id"
        ) != config.get("id"):
            raise RefinedSelectionError(
                "candidate identity differs from terminal stage ledger"
            )
        _validate_candidate_provenance(
            manifest,
            manifest_path=manifest_path,
            plan_path=plan_path,
            plan=plan,
            contract=contract,
            protocol_record=protocol_record,
            feature_bundle_record=feature_bundle_record,
            expected_rows=expected_rows,
        )

    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise RefinedSelectionError(f"candidate snapshots must be non-empty: {manifest_path}")
    observed_epochs: set[int] = set()
    validation_identity: str | None = None
    audited_snapshots: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise RefinedSelectionError(f"snapshot {index} is not an object")
        epoch = _validate_epoch(plan, snapshot.get("epoch"))
        if epoch in observed_epochs:
            raise RefinedSelectionError(
                f"candidate has duplicate snapshot epoch {epoch}: {manifest_path}"
            )
        observed_epochs.add(epoch)
        try:
            recorded_auprc = float(snapshot["validation_auprc"])
        except (KeyError, TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"snapshot {epoch} validation_auprc is invalid"
            ) from error
        if not math.isfinite(recorded_auprc):
            raise RefinedSelectionError(
                f"snapshot {epoch} validation_auprc is not finite"
            )
        training_trace = None
        if expected_rows is not None:
            training_trace = _audit_snapshot_training_trace(
                snapshot,
                epoch=epoch,
                manifest=manifest,
                expected_rows=expected_rows,
            )
        validation_path, validation_hash = _resolve_artifact(
            candidate_root,
            snapshot.get("validation_predictions"),
            f"snapshot[{epoch}].validation_predictions",
        )
        u_path, u_hash = _resolve_artifact(
            candidate_root,
            snapshot.get("u_embedding"),
            f"snapshot[{epoch}].u_embedding",
        )
        i_path, i_hash = _resolve_artifact(
            candidate_root,
            snapshot.get("i_embedding"),
            f"snapshot[{epoch}].i_embedding",
        )
        predictions = load_predictions(validation_path)
        if expected_rows is not None:
            if not np.array_equal(
                predictions.row_id,
                np.asarray(expected_rows.validation_row_indices, dtype=np.int64),
            ):
                raise RefinedSelectionError(
                    f"snapshot {epoch} validation row IDs differ from protocol"
                )
            if not np.array_equal(
                predictions.pair,
                np.asarray(expected_rows.validation_pairs, dtype=np.int64),
            ):
                raise RefinedSelectionError(
                    f"snapshot {epoch} validation pairs differ from protocol"
                )
            if not np.array_equal(
                predictions.label,
                np.asarray(expected_rows.validation_labels, dtype=np.uint8),
            ):
                raise RefinedSelectionError(
                    f"snapshot {epoch} validation labels differ from protocol"
                )
        metrics = evaluate_validation_scores(predictions.label, predictions.score)
        recomputed_auprc = float(metrics["Validation_AUPRC"])
        if not math.isclose(
            recorded_auprc,
            recomputed_auprc,
            rel_tol=0.0,
            abs_tol=VALIDATION_METRIC_TOLERANCE,
        ):
            raise RefinedSelectionError(
                f"snapshot {epoch} validation AUPRC mismatch: "
                f"recorded={recorded_auprc}, recomputed={recomputed_auprc}"
            )
        u_matrix = _load_embedding(u_path)
        i_matrix = _load_embedding(i_path)
        if u_matrix.shape[1] != i_matrix.shape[1]:
            raise RefinedSelectionError(
                f"snapshot {epoch} u/i latent dimensions differ: "
                f"{u_matrix.shape[1]} vs {i_matrix.shape[1]}"
            )
        recomputed_scores = _score_pairs_from_embeddings(
            u_matrix, i_matrix, predictions.pair
        )
        if not np.allclose(
            recomputed_scores,
            predictions.score,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RefinedSelectionError(
                f"snapshot {epoch} validation predictions do not match its "
                "u/i embeddings"
            )
        snapshot_identity = _validation_identity_digest(
            predictions.row_id, predictions.pair, predictions.label
        )
        if validation_identity is None:
            validation_identity = snapshot_identity
        elif snapshot_identity != validation_identity:
            raise RefinedSelectionError(
                "candidate checkpoints use different validation rows/pairs/labels"
            )
        audited_snapshots.append(
            {
                "epoch": epoch,
                "validation_auprc": recomputed_auprc,
                "validation_predictions": {
                    "path": str(validation_path),
                    "sha256": validation_hash,
                },
                "u_embedding": {
                    "path": str(u_path),
                    "sha256": u_hash,
                    "shape": list(u_matrix.shape),
                },
                "i_embedding": {
                    "path": str(i_path),
                    "sha256": i_hash,
                    "shape": list(i_matrix.shape),
                },
                **(
                    {"training": training_trace}
                    if training_trace is not None
                    else {}
                ),
            }
        )
    required_epochs = _required_epoch_grid(plan)
    if observed_epochs != set(required_epochs):
        raise RefinedSelectionError(
            "candidate snapshot epochs are incomplete: "
            f"{sorted(observed_epochs)} vs {list(required_epochs)}"
        )
    return {
        "fold": fold,
        "config": dict(config),
        "plan_rank": list(rank),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "candidate_root": str(candidate_root),
        "validation_identity_sha256": validation_identity,
        "snapshots": audited_snapshots,
    }


def _audit_terminal_candidate_set(
    *,
    candidate_manifests: Sequence[str | Path],
    plan_path: Path,
    plan: Mapping[str, Any],
    freeze_contract: Mapping[str, Any],
    protocol_record: Mapping[str, str],
    feature_bundle_record: Mapping[str, str],
    protocol: BenchmarkProtocol,
) -> list[dict[str, Any]]:
    """Re-audit the exact terminal candidate set without touching test rows."""

    audited: list[dict[str, Any]] = []
    for raw_manifest_path in candidate_manifests:
        candidate_path = Path(raw_manifest_path).resolve(strict=True)
        candidate_payload = _load_json(candidate_path)
        raw_config = candidate_payload.get("config")
        try:
            identity = (
                int(candidate_payload["fold"]),
                str(raw_config["id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"candidate identity is invalid: {candidate_path}"
            ) from error
        contract = freeze_contract["candidate_contracts"].get(identity)
        if contract is None:
            raise RefinedSelectionError(
                f"candidate is absent from terminal stage ledger: {identity}"
            )
        validation_rows = _validation_rows_without_outer_materialisation(
            protocol, identity[0]
        )
        audited.append(
            audit_candidate_manifest(
                candidate_path,
                plan=plan,
                plan_path=plan_path,
                contract=contract,
                protocol_record=protocol_record,
                feature_bundle_record=feature_bundle_record,
                expected_rows=validation_rows,
            )
        )
    identities = [
        (record["fold"], str(record["config"]["id"])) for record in audited
    ]
    if len(identities) != len(set(identities)):
        raise RefinedSelectionError("duplicate fold/config candidate manifests")
    expected_identities = set(freeze_contract["candidate_contracts"])
    if set(identities) != expected_identities:
        missing = sorted(expected_identities.difference(identities))
        extra = sorted(set(identities).difference(expected_identities))
        raise RefinedSelectionError(
            "candidate manifests differ from terminal stage ledger: "
            f"missing={missing}, extra={extra}"
        )
    return audited


def _best_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = list(record["snapshots"])
    if not snapshots:
        raise RefinedSelectionError("cannot select from an empty candidate")
    snapshot = min(
        snapshots,
        key=lambda item: (
            -float(item["validation_auprc"]),
            int(item["epoch"]),
        ),
    )
    return {
        "fold": int(record["fold"]),
        "config": dict(record["config"]),
        "plan_rank": list(record["plan_rank"]),
        "manifest": dict(record["manifest"]),
        "candidate_root": str(record["candidate_root"]),
        "snapshot": dict(snapshot),
    }


def _select_by_promotion_chain(
    records: Sequence[Mapping[str, Any]], *, margin: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one fold's locked candidate order and practical-margin rule."""

    if not records:
        raise RefinedSelectionError("cannot select from an empty candidate set")
    ordered = sorted(
        records,
        key=lambda record: (
            int(record["plan_rank"][0]),
            int(record["plan_rank"][1]),
            int(record["config"]["chain_index"]),
            str(record["config"]["id"]),
        ),
    )
    chain_indices = [int(record["config"]["chain_index"]) for record in ordered]
    if len(chain_indices) != len(set(chain_indices)):
        raise RefinedSelectionError("promotion chain has duplicate chain_index values")
    if chain_indices != list(range(len(chain_indices))):
        raise RefinedSelectionError(
            "promotion chain indices must be contiguous from 0 and follow "
            "locked-plan order"
        )

    selected = _best_snapshot(ordered[0])
    root_id = str(selected["config"]["id"])
    trace: list[dict[str, Any]] = [
        {
            "chain_index": 0,
            "candidate_config_id": root_id,
            "declared_incumbent_id": None,
            "candidate_epoch": int(selected["snapshot"]["epoch"]),
            "candidate_validation_auprc": float(
                selected["snapshot"]["validation_auprc"]
            ),
            "improvement_over_incumbent": None,
            "promoted": True,
            "decision": "initialize_lower_complexity_incumbent",
        }
    ]
    if selected["config"].get("base_config_id") != root_id:
        raise RefinedSelectionError("promotion-chain root has an inconsistent base ID")

    for record in ordered[1:]:
        candidate = _best_snapshot(record)
        config = candidate["config"]
        current_id = str(selected["config"]["id"])
        if config.get("base_config_id") != root_id:
            raise RefinedSelectionError(
                f"candidate {config['id']!r} has a different promotion-chain base"
            )
        if config.get("incumbent_id") != current_id:
            raise RefinedSelectionError(
                f"candidate {config['id']!r} declares incumbent "
                f"{config.get('incumbent_id')!r}, but the retained incumbent is "
                f"{current_id!r}"
            )
        candidate_metric = float(candidate["snapshot"]["validation_auprc"])
        incumbent_metric = float(selected["snapshot"]["validation_auprc"])
        improvement = candidate_metric - incumbent_metric
        promoted = bool(improvement >= margin)
        trace.append(
            {
                "chain_index": int(config["chain_index"]),
                "candidate_config_id": str(config["id"]),
                "declared_incumbent_id": str(config["incumbent_id"]),
                "candidate_epoch": int(candidate["snapshot"]["epoch"]),
                "candidate_validation_auprc": candidate_metric,
                "incumbent_validation_auprc": incumbent_metric,
                "improvement_over_incumbent": improvement,
                "required_improvement": margin,
                "promoted": promoted,
                "decision": (
                    "replace_incumbent"
                    if promoted
                    else "retain_lower_complexity_incumbent"
                ),
            }
        )
        if promoted:
            selected = candidate
    return selected, trace


def _build_fold_selections(
    audited: Sequence[Mapping[str, Any]], *, practical_margin: float
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    observed_folds = sorted({int(record["fold"]) for record in audited})
    for fold in observed_folds:
        fold_records = [record for record in audited if record["fold"] == fold]
        identities_for_fold = {
            str(record["validation_identity_sha256"]) for record in fold_records
        }
        if len(identities_for_fold) != 1:
            raise RefinedSelectionError(
                f"fold {fold} candidates use different validation rows/pairs/labels"
            )
        selected, promotion_trace = _select_by_promotion_chain(
            fold_records, margin=practical_margin
        )
        selected_validation = load_predictions(
            Path(selected["snapshot"]["validation_predictions"]["path"])
        )
        frozen_threshold = select_f1_threshold(
            selected_validation.label, selected_validation.score
        )
        selections[str(fold)] = {
            "selected_config": selected["config"],
            "selected_plan_rank": selected["plan_rank"],
            "selected_epoch": selected["snapshot"]["epoch"],
            "selected_validation_auprc": selected["snapshot"][
                "validation_auprc"
            ],
            "selected_manifest": selected["manifest"],
            "selected_candidate_root": selected["candidate_root"],
            "selected_snapshot": selected["snapshot"],
            "frozen_f1_threshold": frozen_threshold.as_dict(),
            "promotion_trace": promotion_trace,
            "test_score_calls_before_freeze": 0,
        }
    return selections


def _audit_and_freeze_selection_with_protocol_loader(
    *,
    plan_path: str | Path,
    candidate_manifests: Sequence[str | Path],
    selection_path: str | Path,
    protocol_loader: Callable[[Path], BenchmarkProtocol] = load_protocol_manifest,
) -> dict[str, Any]:
    """Audit all candidates, select per fold, and atomically freeze selection."""

    plan_path = Path(plan_path).resolve(strict=True)
    selection_path = Path(selection_path).resolve(strict=False)
    if not candidate_manifests:
        raise RefinedSelectionError("at least one candidate manifest is required")
    if selection_path.exists() or selection_path.with_name(
        selection_path.name + SELECTION_HASH_SUFFIX
    ).exists():
        raise RefinedSelectionError(f"selection is already frozen: {selection_path}")
    plan = _load_json(plan_path)
    freeze_contract = _load_freeze_contract(
        plan_path, plan, require_terminal=True
    )
    if selection_path != freeze_contract["selection_path"]:
        raise RefinedSelectionError(
            "selection path differs from locked canonical_selection_path"
        )
    protocol_record = _plan_protocol_record(plan_path, plan)
    feature_bundle_record = _plan_feature_bundle_record(plan_path, plan)
    protocol = protocol_loader(Path(protocol_record["path"]))
    if (
        protocol.dataset != plan.get("dataset")
        or int(protocol.seed) != int(plan.get("seed"))
    ):
        raise RefinedSelectionError(
            "loaded protocol dataset/seed differs from locked plan"
        )
    expected_folds = plan.get("outer_folds")
    if not isinstance(expected_folds, list) or not expected_folds:
        raise RefinedSelectionError("locked plan outer_folds must be non-empty")
    practical_margin = _practical_margin(plan)

    # Do not create the selection directory/file until every candidate passes.
    audited = _audit_terminal_candidate_set(
        candidate_manifests=candidate_manifests,
        plan_path=plan_path,
        plan=plan,
        freeze_contract=freeze_contract,
        protocol_record=protocol_record,
        feature_bundle_record=feature_bundle_record,
        protocol=protocol,
    )
    observed_folds = sorted({int(record["fold"]) for record in audited})
    if observed_folds != sorted(int(fold) for fold in expected_folds):
        raise RefinedSelectionError(
            f"candidate folds differ from plan: {observed_folds} vs {expected_folds}"
        )

    selections = _build_fold_selections(
        audited, practical_margin=practical_margin
    )

    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": plan.get("dataset", "ethnobotany"),
        "seed": plan.get("seed"),
        "plan": {
            "path": str(plan_path),
            "sha256": sha256(plan_path),
            "core_sha256": _plan_core_sha256(plan),
            "schema": plan.get("schema"),
        },
        "protocol": dict(protocol_record),
        "feature_bundle": dict(feature_bundle_record),
        "freeze_contract": _freeze_contract_receipt(freeze_contract),
        "selector_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "metric_and_protocol_sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "metrics": Path(__file__).resolve().with_name("metrics.py"),
                "protocol": Path(__file__).resolve().with_name("protocol.py"),
            }.items()
        },
        "selection_metric": SELECTION_METRIC,
        "per_fold_practical_margin_auprc": practical_margin,
        "tie_break": SELECTION_TIE_BREAK,
        "outer_test_policy": OUTER_TEST_POLICY,
        "outer_test_pristine": False,
        "candidate_manifests": [record["manifest"] for record in audited],
        "folds": selections,
    }
    digest = _write_frozen_json(selection_path, payload)
    return {**payload, "selection_sha256": digest}


def audit_and_freeze_selection(
    *,
    plan_path: str | Path,
    candidate_manifests: Sequence[str | Path],
    selection_path: str | Path,
) -> dict[str, Any]:
    """Formal freeze entrypoint with the hashed protocol's fixed loader."""

    return _audit_and_freeze_selection_with_protocol_loader(
        plan_path=plan_path,
        candidate_manifests=candidate_manifests,
        selection_path=selection_path,
        protocol_loader=load_protocol_manifest,
    )


def _validate_frozen_selection_with_protocol(
    selection_path: str | Path,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol] = load_protocol_manifest,
) -> tuple[dict[str, Any], str, BenchmarkProtocol]:
    """Re-audit and replay a frozen selection from its locked inputs."""

    selection, selection_hash = _load_verified_selection(selection_path)
    plan_record = selection.get("plan")
    if not isinstance(plan_record, Mapping):
        raise RefinedSelectionError("selection has no locked-plan record")
    plan_path = Path(str(plan_record.get("path", ""))).resolve(strict=False)
    if sha256(plan_path) != plan_record.get("sha256"):
        raise RefinedSelectionError("locked plan hash changed after selection")
    plan = _load_json(plan_path)
    expected_plan_record = {
        "path": str(plan_path),
        "sha256": sha256(plan_path),
        "core_sha256": _plan_core_sha256(plan),
        "schema": plan.get("schema"),
    }
    if dict(plan_record) != expected_plan_record:
        raise RefinedSelectionError("selection locked-plan receipt is inconsistent")
    if _plan_core_sha256(plan) != plan_record.get("core_sha256"):
        raise RefinedSelectionError("locked plan core changed after selection")
    freeze_contract = _load_freeze_contract(
        plan_path, plan, require_terminal=True
    )
    if Path(selection_path).resolve() != freeze_contract["selection_path"]:
        raise RefinedSelectionError(
            "frozen selection is outside the locked canonical path"
        )
    protocol_record = _plan_protocol_record(plan_path, plan)
    if selection.get("protocol") != protocol_record:
        raise RefinedSelectionError(
            "selection protocol binding differs from locked plan"
        )
    feature_bundle_record = _plan_feature_bundle_record(plan_path, plan)
    if selection.get("feature_bundle") != feature_bundle_record:
        raise RefinedSelectionError(
            "selection feature-bundle binding differs from locked plan"
        )
    if (
        selection.get("dataset") != plan.get("dataset")
        or selection.get("seed") != plan.get("seed")
    ):
        raise RefinedSelectionError(
            "selection dataset/seed differs from locked plan"
        )
    source_record = selection.get("selector_source")
    if not isinstance(source_record, Mapping):
        raise RefinedSelectionError("selection has no selector-source record")
    source_path = Path(str(source_record.get("path", ""))).resolve(strict=False)
    if source_path != Path(__file__).resolve() or sha256(source_path) != source_record.get(
        "sha256"
    ):
        raise RefinedSelectionError("selector source changed after selection")
    dependency_sources = selection.get("metric_and_protocol_sources")
    expected_sources = {
        "metrics": Path(__file__).resolve().with_name("metrics.py"),
        "protocol": Path(__file__).resolve().with_name("protocol.py"),
    }
    if not isinstance(dependency_sources, Mapping):
        raise RefinedSelectionError(
            "selection has no metric/protocol source closure"
        )
    for name, expected_path in expected_sources.items():
        record = dependency_sources.get(name)
        if not isinstance(record, Mapping):
            raise RefinedSelectionError(f"selection source {name} is missing")
        observed_path = Path(str(record.get("path", ""))).resolve(strict=False)
        if (
            observed_path != expected_path
            or sha256(observed_path) != record.get("sha256")
        ):
            raise RefinedSelectionError(
                f"selection dependency source changed: {name}"
            )
    recorded_contract = selection.get("freeze_contract")
    if not isinstance(recorded_contract, Mapping):
        raise RefinedSelectionError("selection has no freeze-contract receipt")
    if dict(recorded_contract) != _freeze_contract_receipt(freeze_contract):
        raise RefinedSelectionError(
            "selection freeze-contract receipt differs from locked plan"
        )
    manifests = selection.get("candidate_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise RefinedSelectionError("selection has no candidate-manifest records")
    manifest_paths: list[Path] = []
    for record in manifests:
        if not isinstance(record, Mapping):
            raise RefinedSelectionError("candidate-manifest record is invalid")
        path = Path(str(record.get("path", ""))).resolve(strict=False)
        if sha256(path) != record.get("sha256"):
            raise RefinedSelectionError(f"candidate manifest changed after freeze: {path}")
        _reject_test_files(path.parent)
        manifest_paths.append(path)

    protocol = protocol_loader(Path(protocol_record["path"]))
    if (
        protocol.dataset != plan.get("dataset")
        or int(protocol.seed) != int(plan.get("seed"))
    ):
        raise RefinedSelectionError(
            "loaded protocol dataset/seed differs from locked plan"
        )
    audited = _audit_terminal_candidate_set(
        candidate_manifests=manifest_paths,
        plan_path=plan_path,
        plan=plan,
        freeze_contract=freeze_contract,
        protocol_record=protocol_record,
        feature_bundle_record=feature_bundle_record,
        protocol=protocol,
    )
    audited_manifest_records = [record["manifest"] for record in audited]
    if manifests != audited_manifest_records:
        raise RefinedSelectionError(
            "selection candidate-manifest receipt is inconsistent"
        )
    expected_folds = sorted(int(fold) for fold in plan.get("outer_folds", []))
    observed_folds = sorted({int(record["fold"]) for record in audited})
    if observed_folds != expected_folds:
        raise RefinedSelectionError(
            f"candidate folds differ from plan: {observed_folds} vs {expected_folds}"
        )
    practical_margin = _practical_margin(plan)
    replayed = _build_fold_selections(
        audited, practical_margin=practical_margin
    )
    if selection.get("folds") != replayed:
        raise RefinedSelectionError(
            "frozen fold selection differs from deterministic validation replay"
        )
    if selection.get("per_fold_practical_margin_auprc") != practical_margin:
        raise RefinedSelectionError(
            "selection practical margin differs from locked plan"
        )
    expected_claims = {
        "selection_metric": SELECTION_METRIC,
        "tie_break": SELECTION_TIE_BREAK,
        "outer_test_policy": OUTER_TEST_POLICY,
        "outer_test_pristine": False,
    }
    for key, expected in expected_claims.items():
        if selection.get(key) != expected:
            raise RefinedSelectionError(
                f"selection audit claim is inconsistent at {key}"
            )
    return selection, selection_hash, protocol


def validate_frozen_selection(
    selection_path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Public validation API; outer rows are never constructed."""

    selection, selection_hash, _ = _validate_frozen_selection_with_protocol(
        selection_path, protocol_loader=load_protocol_manifest
    )
    return selection, selection_hash


def _selected_artifact(record: object, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RefinedSelectionError(f"selected {label} record is missing")
    path = Path(str(record.get("path", ""))).resolve(strict=False)
    if sha256(path) != record.get("sha256"):
        raise RefinedSelectionError(f"selected {label} hash mismatch: {path}")
    return path


def _score_pairs_from_embeddings(
    u_embedding: np.ndarray,
    i_embedding: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RefinedSelectionError(f"test pairs must have shape (n, 2): {pairs.shape}")
    if not len(pairs):
        raise RefinedSelectionError("test pairs must not be empty")
    if pairs.min() < 0:
        raise RefinedSelectionError("test pairs contain negative compact indices")
    if int(pairs[:, 0].max()) >= len(u_embedding):
        raise RefinedSelectionError("test herb index exceeds selected u embedding")
    if int(pairs[:, 1].max()) >= len(i_embedding):
        raise RefinedSelectionError("test disease index exceeds selected i embedding")
    logits = np.einsum(
        "ij,ij->i",
        u_embedding[pairs[:, 0]],
        i_embedding[pairs[:, 1]],
    )
    logits = np.clip(np.asarray(logits, dtype=np.float64), -50.0, 50.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    if not np.isfinite(scores).all():
        raise RefinedSelectionError("selected embeddings emitted non-finite scores")
    return scores


def _score_outer_test_once(
    u_embedding: np.ndarray,
    i_embedding: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Dedicated, non-injectable outer-test model invocation boundary."""

    return _score_pairs_from_embeddings(u_embedding, i_embedding, pairs)


def _evaluate_selected_folds_with_protocol_loader(
    *,
    selection_path: str | Path,
    protocol_manifest: str | Path,
    output_root: str | Path,
    folds: Sequence[int] | None = None,
    protocol_loader: Callable[[Path], BenchmarkProtocol] = load_protocol_manifest,
) -> dict[str, Any]:
    """Test helper for scoring with an explicitly supplied protocol loader."""

    # Selection replay completes before any outer-test fold rows are built.
    selection, selection_hash, locked_protocol = _validate_frozen_selection_with_protocol(
        selection_path, protocol_loader=protocol_loader
    )
    selection_folds = selection.get("folds")
    if not isinstance(selection_folds, Mapping):
        raise RefinedSelectionError("selection folds are missing")
    expected_protocol = selection.get("protocol")
    if not isinstance(expected_protocol, Mapping):
        raise RefinedSelectionError("selection protocol binding is missing")
    supplied_protocol_path = Path(protocol_manifest).resolve(strict=True)
    if (
        supplied_protocol_path
        != Path(str(expected_protocol.get("path", ""))).resolve(strict=False)
        or sha256(supplied_protocol_path) != expected_protocol.get("sha256")
    ):
        raise RefinedSelectionError(
            "outer-test protocol manifest differs from frozen selection"
        )
    if (
        locked_protocol.dataset != selection.get("dataset")
        or int(locked_protocol.seed) != int(selection.get("seed"))
    ):
        raise RefinedSelectionError(
            "outer-test protocol dataset/seed differs from frozen selection"
        )
    requested = (
        [int(fold) for fold in folds]
        if folds is not None
        else sorted(int(fold) for fold in selection_folds)
    )
    if len(requested) != len(set(requested)):
        raise RefinedSelectionError("requested folds contain duplicates")
    all_selected_folds = sorted(int(fold) for fold in selection_folds)
    if sorted(requested) != all_selected_folds:
        raise RefinedSelectionError(
            "outer-test scoring must atomically reserve every selected fold"
        )
    requested = all_selected_folds
    output_root = Path(output_root).resolve(strict=False)
    freeze_receipt = selection.get("freeze_contract")
    if not isinstance(freeze_receipt, Mapping):
        raise RefinedSelectionError("selection freeze-contract receipt is missing")
    expected_output_root = Path(
        str(freeze_receipt.get("outer_test_output_root", ""))
    ).resolve(strict=False)
    if output_root != expected_output_root:
        raise RefinedSelectionError(
            "outer-test output root differs from frozen canonical root"
        )
    candidate_roots = [
        Path(str(record.get("path", ""))).resolve(strict=False).parent
        for record in selection.get("candidate_manifests", [])
        if isinstance(record, Mapping)
    ]
    for candidate_root in candidate_roots:
        try:
            output_root.relative_to(candidate_root)
        except ValueError:
            continue
        raise RefinedSelectionError(
            "outer-test output root must not be inside a candidate directory"
        )

    # Reserve every output fold before the first test callback.  This prevents
    # a partially rerun command from rescoring already completed folds.
    output_paths: dict[int, Path] = {}
    for fold in requested:
        if str(fold) not in selection_folds:
            raise RefinedSelectionError(f"fold {fold} is absent from selection")
        fold_output = output_root / f"fold_{fold}"
        if fold_output.is_symlink():
            raise RefinedSelectionError(
                f"outer-test fold output must not be a symlink: {fold_output}"
            )
        try:
            fold_output.resolve(strict=False).relative_to(output_root)
        except ValueError as error:
            raise RefinedSelectionError(
                f"outer-test fold output escapes the canonical root: {fold_output}"
            ) from error
        if fold_output.exists() and not fold_output.is_dir():
            raise RefinedSelectionError(
                f"outer-test fold output is not a directory: {fold_output}"
            )
        if fold_output.exists() and any(fold_output.iterdir()):
            raise RefinedSelectionError(
                f"refusing to repeat outer-test scoring for fold {fold}: {fold_output}"
            )
        output_paths[fold] = fold_output

    # A single root lock prevents opposite fold orders from partially reserving
    # disjoint folds in concurrent processes.
    _atomic_create(
        output_root / ".outer_test_all_folds_reserved.json",
        (
            json.dumps(
                {
                    "folds": requested,
                    "selection_sha256": selection_hash,
                    "state": "all-folds-reserved-before-test-score-call",
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )

    # Atomically reserve every fold before any outer-test pairs are built.
    for fold, fold_output in output_paths.items():
        _atomic_create(
            fold_output / ".outer_test_reserved.json",
            (
                json.dumps(
                    {
                        "fold": fold,
                        "selection_sha256": selection_hash,
                        "state": "reserved-before-test-score-call",
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )

    summaries: dict[str, Any] = {}
    for fold in requested:
        selected = selection_folds[str(fold)]
        if not isinstance(selected, Mapping):
            raise RefinedSelectionError(f"selected fold record is invalid: {fold}")
        snapshot = selected.get("selected_snapshot")
        if not isinstance(snapshot, Mapping):
            raise RefinedSelectionError(f"selected snapshot is missing: fold {fold}")
        validation_path = _selected_artifact(
            snapshot.get("validation_predictions"), "validation predictions"
        )
        u_path = _selected_artifact(snapshot.get("u_embedding"), "u embedding")
        i_path = _selected_artifact(snapshot.get("i_embedding"), "i embedding")
        validation = load_predictions(validation_path)
        threshold_record = selected.get("frozen_f1_threshold")
        if not isinstance(threshold_record, Mapping):
            raise RefinedSelectionError(
                f"selected fold {fold} has no frozen F1 threshold"
            )
        try:
            threshold_value = float(
                threshold_record["Validation_Selected_Threshold"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RefinedSelectionError(
                f"selected fold {fold} frozen F1 threshold is invalid"
            ) from error
        # Recompute validation diagnostics only as an integrity check; the
        # threshold used below is the value committed before any test row was
        # materialised.
        recomputed_threshold = select_f1_threshold(
            validation.label, validation.score
        ).as_dict()
        if recomputed_threshold != dict(threshold_record):
            raise RefinedSelectionError(
                f"selected fold {fold} frozen threshold receipt is inconsistent"
            )
        u_embedding = _load_embedding(u_path)
        i_embedding = _load_embedding(i_path)
        if u_embedding.shape[1] != i_embedding.shape[1]:
            raise RefinedSelectionError(
                f"selected fold {fold} u/i latent dimensions differ"
            )
        if len(u_embedding) != int(locked_protocol.n_herbs):
            raise RefinedSelectionError(
                f"fold {fold} u rows differ from protocol: "
                f"{len(u_embedding)} vs {locked_protocol.n_herbs}"
            )
        if len(i_embedding) != int(locked_protocol.n_diseases):
            raise RefinedSelectionError(
                f"fold {fold} i rows differ from protocol: "
                f"{len(i_embedding)} vs {locked_protocol.n_diseases}"
            )
        fold_rows = locked_protocol.fold_rows(fold)
        # This is the sole outer-test model invocation in the workflow.
        test_scores = np.asarray(
            _score_outer_test_once(
                u_embedding, i_embedding, fold_rows.test_pairs
            ),
            dtype=np.float64,
        )
        metrics = {
            **dict(threshold_record),
            **evaluate_test_scores(
                fold_rows.test_labels,
                test_scores,
                threshold=threshold_value,
            ),
            "Dataset": selection.get("dataset"),
            "Fold": fold,
            "Selected_Config": selected.get("selected_config"),
            "Selected_Epoch": selected.get("selected_epoch"),
            "Selected_Validation_AUPRC": selected.get(
                "selected_validation_auprc"
            ),
            "Selection_SHA256": selection_hash,
            "Decision_Frozen_Before_Test": True,
            "Test_Models_Evaluated": 1,
            "Unselected_Test_Models_Evaluated": 0,
        }
        fold_output = output_paths[fold]
        predictions_path = fold_output / "test_predictions.npz"
        save_predictions(
            predictions_path,
            row_id=fold_rows.test_row_indices,
            pair=fold_rows.test_pairs,
            label=fold_rows.test_labels,
            score=test_scores,
        )
        _atomic_create(
            fold_output / "metrics.json",
            (json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        binding = {
            "selection_path": str(Path(selection_path).resolve()),
            "selection_sha256": selection_hash,
            "fold": fold,
            "selected_manifest": selected.get("selected_manifest"),
            "selected_epoch": selected.get("selected_epoch"),
            "validation_predictions_sha256": sha256(validation_path),
            "u_embedding_sha256": sha256(u_path),
            "i_embedding_sha256": sha256(i_path),
            "test_predictions_sha256": sha256(predictions_path),
            "test_score_calls": 1,
        }
        _atomic_create(
            fold_output / "selection_binding.json",
            (json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        summaries[str(fold)] = metrics
    return {
        "schema": "astra-ethnobotany-refined-test-summary/v1",
        "selection_sha256": selection_hash,
        "test_score_calls": len(requested),
        "folds": summaries,
    }


def evaluate_selected_folds(
    *,
    selection_path: str | Path,
    protocol_manifest: str | Path,
    output_root: str | Path,
    folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score one frozen snapshot per fold using the locked protocol loader."""

    return _evaluate_selected_folds_with_protocol_loader(
        selection_path=selection_path,
        protocol_manifest=protocol_manifest,
        output_root=output_root,
        folds=folds,
        protocol_loader=load_protocol_manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="audit candidates and freeze selection")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument(
        "--candidate-manifest", type=Path, action="append", required=True
    )
    freeze.add_argument("--selection", type=Path, required=True)

    score = subparsers.add_parser("score", help="score selected outer-test folds once")
    score.add_argument("--selection", type=Path, required=True)
    score.add_argument("--protocol-manifest", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--folds", type=int, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = audit_and_freeze_selection(
            plan_path=args.plan,
            candidate_manifests=args.candidate_manifest,
            selection_path=args.selection,
        )
    else:
        payload = evaluate_selected_folds(
            selection_path=args.selection,
            protocol_manifest=args.protocol_manifest,
            output_root=args.output_root,
            folds=args.folds,
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
