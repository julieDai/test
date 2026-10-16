#!/usr/bin/env python3
"""One-shot outer evaluation of the validation-selected TCM adapter.

The evaluator is intentionally separate from candidate training and selection.
It first replays the complete two-arm, five-fold validation decision without
opening the frozen protocol arrays or deriving any protocol rows.  Only after
that preflight succeeds does it durably burn a content-addressed global ledger
key and reserve all five output folds.  The official protocol loader then
verifies all validation/test folds before only the saved winner embeddings are
scored with each winner's already-frozen validation threshold.

This remains a seed-42 post-baseline diagnostic.  It is never a formal result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics import (
    AUPRC_DEFINITION,
    evaluate_test_scores,
    evaluate_validation_scores,
    load_predictions,
    save_predictions,
    validate_prediction_coverage,
)
from .protocol import (
    ARRAY_FILENAME,
    MAPPING_FILENAME,
    PROTOCOL_VERSION,
    BenchmarkProtocol,
    load_protocol_manifest,
    _array_digest as _protocol_array_digest,
    _json_bytes as _protocol_json_bytes,
)
from .tcm_astra_hparam_selection import (
    ARRAY_HASH_SCHEMA,
    _canonical_array_sha256 as _stage1_array_digest,
    _embedding_scores as _stage1_embedding_scores,
)
from .tcm_astra_semantic_adapter_plan import PLAN_SCHEMA
from .tcm_astra_semantic_adapter_selection import (
    SELECTION_SCHEMA,
    _candidate_map as _selection_candidate_map,
    _reject_outer_artifacts as _selection_reject_outer_artifacts,
    _validate_candidate as _selection_validate_candidate,
    _validate_sidecar as _selection_validate_plan_sidecar,
    _verify_artifact as _selection_verify_artifact,
    _winner as _selection_winner,
    sha256,
)


EVALUATION_SCHEMA = "astra-tcm-semantic-adapter-outer-test-summary/v1"
PREFLIGHT_SCHEMA = "astra-tcm-semantic-adapter-outer-preflight/v1"
ACCESS_SCHEMA = "astra-tcm-semantic-adapter-outer-access/v1"
COMPLETION_SCHEMA = "astra-tcm-semantic-adapter-outer-completion/v1"
EXPECTED_CANDIDATES = {
    "identity_control": "identity",
    "residual_mlp_h32": "residual_mlp",
}
EXPECTED_FIXED = {
    "batch_size": 2000,
    "embedding_dimension": 64,
    "reg_u": 0.001,
    "text_fusion_alpha": 1.0,
    "semantic_input_dim": 64,
    "semantic_mlp_hidden_dim": 32,
    "semantic_freeze_input": False,
}
ARRAY_NAMES = (
    "positive_pairs",
    "negative_pairs",
    "herb_compound_pairs",
    "compound_protein_pairs",
    "protein_disease_pairs",
)
ENTITY_NAMES = ("herb", "compound", "protein", "disease")
SUMMARY_METRICS = (
    "AUROC",
    "AUPRC",
    "AveragePrecision",
    "F1",
    "Precision",
    "Recall",
    "Accuracy",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_LEDGER_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "tcm_suite"
    / "semantic_adapter_outer_access_ledger"
)


class TCMAdapterEvaluationError(RuntimeError):
    """Raised when a one-shot adapter evaluation cannot be audited."""


@dataclass(frozen=True)
class ProtocolMetadataPackage:
    """Frozen protocol metadata inspected without opening protocol arrays."""

    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class FileSnapshot:
    """A post-burn artifact identity that must survive until publication."""

    label: str
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AuditedOuterFold:
    """One fully audited outer fold, captured before any fold is scored."""

    winner: "WinnerPreflight"
    row_id: np.ndarray
    pair: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class WinnerPreflight:
    fold: int
    candidate_id: str
    adapter: str
    epochs: int
    learning_rate: float
    threshold: float
    validation_auprc: float
    validation_auroc: float
    validation_path: Path
    validation_sha256: str
    herb_path: Path
    herb_sha256: str
    disease_path: Path
    disease_sha256: str
    herb: np.ndarray
    disease: np.ndarray
    validation_identity: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationPreflight:
    selection_path: Path
    selection_sha256: str
    plan_path: Path
    plan_sha256: str
    candidate_root: Path
    selection: Mapping[str, Any]
    plan: Mapping[str, Any]
    protocol_package: ProtocolMetadataPackage
    winners: tuple[WinnerPreflight, ...]
    candidate_records: tuple[Mapping[str, Any], ...]


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMAdapterEvaluationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise TCMAdapterEvaluationError(f"JSON root must be an object: {path}")
    return value


def _json_document(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Persist a directory-entry mutation, not only the file contents."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    """Create missing directories one at a time and fsync every parent."""

    path = path.resolve()
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:  # pragma: no cover - defensive guard
            break
        cursor = cursor.parent
    if cursor.exists() and not cursor.is_dir():
        raise TCMAdapterEvaluationError(f"directory parent is not a directory: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory.parent)


def _durable_mkdir(path: Path) -> None:
    path = path.resolve()
    _ensure_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(path.parent)


def _fsync_file_and_parent(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink():
        raise TCMAdapterEvaluationError(f"refusing to overwrite JSON: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise TCMAdapterEvaluationError(
            f"refusing interrupted JSON residue: {temporary}"
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_document(payload))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        # Deliberately retain any partial file as an interruption witness.
        raise


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable one-shot receipt and never overwrite a residue."""

    _ensure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise TCMAdapterEvaluationError(
            f"refusing existing one-shot receipt or interrupted residue: {path}"
        ) from error
    # The existence of the ledger is the irreversible burn.  Persist its
    # directory entry immediately, even if payload writing later fails.
    with os.fdopen(descriptor, "wb") as handle:
        _fsync_directory(path.parent)
        handle.write(_json_document(payload))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _canonical_array_sha256(values: np.ndarray) -> str:
    """Use and cross-check the repaired protocol/stage-1 canonical digest."""

    array = np.asarray(values)
    protocol_digest = _protocol_array_digest(array)
    stage1_digest = _stage1_array_digest(array)
    if protocol_digest != stage1_digest:  # pragma: no cover - invariant guard
        raise TCMAdapterEvaluationError(
            "protocol and stage-1 canonical array digests disagree"
        )
    return protocol_digest


def _verify_file(path: Path, record: object, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise TCMAdapterEvaluationError(f"{label} record is absent")
    if not path.is_file():
        raise TCMAdapterEvaluationError(f"{label} is absent: {path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise TCMAdapterEvaluationError(f"{label} byte count differs")
    if str(record.get("sha256", "")) != sha256(path):
        raise TCMAdapterEvaluationError(f"{label} SHA-256 differs")


def _sorted_pair_keys(values: np.ndarray, width: int, *, label: str) -> np.ndarray:
    """Return compact sorted keys with bounded memory and reject duplicates."""

    if width <= 0:
        raise TCMAdapterEvaluationError(f"{label} key width is invalid")
    maximum = int(values[:, 0].max()) * int(width) + int(values[:, 1].max())
    if maximum > np.iinfo(np.uint64).max:
        raise TCMAdapterEvaluationError(f"{label} pair keys overflow uint64")
    keys = np.asarray(values[:, 0], dtype=np.uint64) * np.uint64(width)
    keys += np.asarray(values[:, 1], dtype=np.uint64)
    keys.sort()
    if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
        raise TCMAdapterEvaluationError(f"{label} contains duplicate pairs")
    return keys


def _assert_disjoint_sorted(
    first: np.ndarray, second: np.ndarray, *, label: str
) -> None:
    """Check sorted uint64 arrays in chunks without building Python tuple sets."""

    chunk_size = 1_000_000
    for start in range(0, len(second), chunk_size):
        chunk = second[start : start + chunk_size]
        positions = np.searchsorted(first, chunk)
        valid = positions < len(first)
        if np.any(valid) and np.any(first[positions[valid]] == chunk[valid]):
            raise TCMAdapterEvaluationError(f"{label} overlap")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_file_metadata(record: object, *, label: str) -> None:
    """Validate a manifest file record without touching its target path."""

    if not isinstance(record, Mapping):
        raise TCMAdapterEvaluationError(f"{label} metadata is absent")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise TCMAdapterEvaluationError(f"{label} byte metadata is invalid")
    if not _is_sha256(record.get("sha256")):
        raise TCMAdapterEvaluationError(f"{label} SHA-256 metadata is invalid")


def _read_protocol_metadata(plan: Mapping[str, Any]) -> ProtocolMetadataPackage:
    """Audit only the frozen receipt/manifest metadata before outer burn.

    In particular this function must never stat, hash, or load
    ``protocol_arrays.npz``.  Its bytes and all fold-derived arrays remain
    unavailable until the global one-shot ledger has been durably created.
    """

    record = plan.get("protocol_manifest")
    try:
        manifest_path = _selection_verify_artifact(
            record, label="protocol manifest"
        )
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    if isinstance(record, Mapping) and "bytes" in record:
        if int(record["bytes"]) != manifest_path.stat().st_size:
            raise TCMAdapterEvaluationError("protocol manifest byte count differs")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("dataset") != "tcm-suite"
        or int(manifest.get("seed", -1)) != 42
        or int(manifest.get("n_folds", -1)) != 5
    ):
        raise TCMAdapterEvaluationError("frozen protocol identity differs")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise TCMAdapterEvaluationError("protocol validation policy is absent")
    try:
        validation_fraction = float(validation["fraction_per_class"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMAdapterEvaluationError(
            "protocol validation fraction is invalid"
        ) from error
    if not 0.0 < validation_fraction < 1.0:
        raise TCMAdapterEvaluationError("protocol validation fraction is invalid")
    if manifest.get("pair_order") not in {"seeded", "source"}:
        raise TCMAdapterEvaluationError("protocol pair-order policy is invalid")
    if not isinstance(manifest.get("mapping_strategy"), Mapping) or not manifest.get(
        "mapping_strategy"
    ):
        raise TCMAdapterEvaluationError("protocol mapping strategy is absent")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        ARRAY_FILENAME,
        MAPPING_FILENAME,
    }:
        raise TCMAdapterEvaluationError("protocol artifact metadata differs")
    _validate_file_metadata(
        artifacts[ARRAY_FILENAME], label="protocol arrays"
    )
    _validate_file_metadata(
        artifacts[MAPPING_FILENAME], label="protocol mappings"
    )

    count_names = {
        "herbs",
        "compounds",
        "proteins",
        "diseases",
        "positive_rows",
        "negative_rows",
        "herb_compound_rows",
        "compound_protein_rows",
        "protein_disease_rows",
    }
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != count_names:
        raise TCMAdapterEvaluationError("protocol count metadata differs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise TCMAdapterEvaluationError("protocol count metadata is invalid")
    if counts["positive_rows"] != counts["negative_rows"]:
        raise TCMAdapterEvaluationError("protocol classes are not balanced")

    expected_shapes = {
        "positive_pairs": [counts["positive_rows"], 2],
        "negative_pairs": [counts["negative_rows"], 2],
        "herb_compound_pairs": [counts["herb_compound_rows"], 2],
        "compound_protein_pairs": [counts["compound_protein_rows"], 2],
        "protein_disease_pairs": [counts["protein_disease_rows"], 2],
    }
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(ARRAY_NAMES):
        raise TCMAdapterEvaluationError("protocol array metadata differs")
    for name, expected_shape in expected_shapes.items():
        array_record = arrays.get(name)
        if not isinstance(array_record, Mapping):
            raise TCMAdapterEvaluationError(
                f"protocol array metadata is absent for {name}"
            )
        try:
            dtype = np.dtype(str(array_record.get("dtype", "")))
        except (TypeError, ValueError) as error:
            raise TCMAdapterEvaluationError(
                f"protocol array dtype metadata is invalid for {name}"
            ) from error
        if (
            array_record.get("shape") != expected_shape
            or not np.issubdtype(dtype, np.integer)
            or not _is_sha256(array_record.get("sha256"))
        ):
            raise TCMAdapterEvaluationError(
                f"protocol array metadata is invalid for {name}"
            )

    entity_counts = {
        "herb": counts["herbs"],
        "compound": counts["compounds"],
        "protein": counts["proteins"],
        "disease": counts["diseases"],
    }
    mappings = manifest.get("mappings")
    if not isinstance(mappings, Mapping) or set(mappings) != set(ENTITY_NAMES):
        raise TCMAdapterEvaluationError("protocol mapping metadata differs")
    for entity, expected_count in entity_counts.items():
        mapping_record = mappings.get(entity)
        if (
            not isinstance(mapping_record, Mapping)
            or mapping_record.get("count") != expected_count
            or not _is_sha256(mapping_record.get("sha256"))
        ):
            raise TCMAdapterEvaluationError(
                f"protocol mapping metadata is invalid for {entity}"
            )

    source_files = manifest.get("files")
    if not isinstance(source_files, Mapping):
        raise TCMAdapterEvaluationError("protocol source-file metadata is invalid")
    for name, source in source_files.items():
        _validate_file_metadata(source, label=f"protocol source {name}")
        if not isinstance(source, Mapping) or not str(source.get("path", "")):
            raise TCMAdapterEvaluationError(
                f"protocol source path metadata is invalid: {name}"
            )

    total_rows = counts["positive_rows"] + counts["negative_rows"]
    folds = manifest.get("folds")
    if (
        not isinstance(folds, list)
        or not all(isinstance(item, Mapping) for item in folds)
        or [item.get("fold") for item in folds] != list(range(1, 6))
    ):
        raise TCMAdapterEvaluationError("protocol must contain five ordered folds")
    test_totals = {"rows": 0, "positive_rows": 0, "negative_rows": 0}
    for fold_record in folds:
        split_totals = {"rows": 0, "positive_rows": 0, "negative_rows": 0}
        for split in ("train", "validation", "test"):
            split_record = fold_record.get(split)
            if not isinstance(split_record, Mapping):
                raise TCMAdapterEvaluationError(
                    f"protocol fold {fold_record['fold']} {split} metadata is absent"
                )
            rows = split_record.get("rows")
            positives = split_record.get("positive_rows")
            negatives = split_record.get("negative_rows")
            digest = split_record.get("row_indices_sha256")
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or isinstance(positives, bool)
                or not isinstance(positives, int)
                or positives <= 0
                or isinstance(negatives, bool)
                or not isinstance(negatives, int)
                or negatives <= 0
                or positives + negatives != rows
                or not _is_sha256(digest)
            ):
                raise TCMAdapterEvaluationError(
                    f"protocol fold {fold_record['fold']} {split} metadata is invalid"
                )
            for key in split_totals:
                split_totals[key] += int(split_record[key])
                if split == "test":
                    test_totals[key] += int(split_record[key])
        if split_totals != {
            "rows": total_rows,
            "positive_rows": counts["positive_rows"],
            "negative_rows": counts["negative_rows"],
        }:
            raise TCMAdapterEvaluationError(
                f"protocol fold {fold_record['fold']} split totals differ"
            )
    if test_totals != {
        "rows": total_rows,
        "positive_rows": counts["positive_rows"],
        "negative_rows": counts["negative_rows"],
    }:
        raise TCMAdapterEvaluationError("protocol outer-test metadata coverage differs")
    return ProtocolMetadataPackage(
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_path),
        manifest=manifest,
    )


def _manifest_fold(
    package: ProtocolMetadataPackage, fold: int
) -> Mapping[str, Any]:
    matches = [
        item
        for item in package.manifest["folds"]
        if isinstance(item, Mapping) and item.get("fold") == fold
    ]
    if len(matches) != 1:
        raise TCMAdapterEvaluationError(
            f"protocol has {len(matches)} records for fold {fold}"
        )
    return matches[0]


def _canonical_prediction_identity(table: object) -> dict[str, str]:
    row_id = np.asarray(getattr(table, "row_id"))
    pair = np.asarray(getattr(table, "pair"))
    label = np.asarray(getattr(table, "label"))
    score = np.asarray(getattr(table, "score"))
    if row_id.dtype != np.dtype(np.int64):
        raise TCMAdapterEvaluationError("prediction row_id must use int64")
    if pair.dtype != np.dtype(np.int64):
        raise TCMAdapterEvaluationError("prediction pair must use int64")
    if label.dtype != np.dtype(np.int64):
        raise TCMAdapterEvaluationError("prediction label must canonicalise to int64")
    if score.dtype != np.dtype(np.float64):
        raise TCMAdapterEvaluationError("prediction score must canonicalise to float64")
    return {
        "row_id_sha256": _canonical_array_sha256(row_id),
        "pair_sha256": _canonical_array_sha256(pair),
        "label_sha256": _canonical_array_sha256(label),
        "score_sha256": _canonical_array_sha256(score),
    }


def _equal_metric(left: object, right: object) -> bool:
    if isinstance(right, (float, int)) and not isinstance(right, bool):
        if isinstance(left, bool):
            return False
        try:
            return math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return left == right


def _audit_candidate_validation(record: Mapping[str, Any]) -> object:
    """Recompute every saved validation metric, including its threshold."""

    prediction_path = Path(str(record["validation_predictions"])).resolve(
        strict=True
    )
    table = load_predictions(prediction_path)
    _canonical_prediction_identity(table)
    recomputed = evaluate_validation_scores(table.label, table.score)
    metrics = _load_json(Path(str(record["metrics"])).resolve(strict=True))
    for key, expected in recomputed.items():
        if key not in metrics or not _equal_metric(metrics[key], expected):
            raise TCMAdapterEvaluationError(
                f"{record['id']} fold {record['fold']} validation metric {key} "
                "cannot be reproduced"
            )
    record_bindings = {
        "validation_auprc": recomputed["Validation_AUPRC"],
        "validation_auroc": recomputed["Validation_AUROC"],
        "validation_f1": recomputed["Validation_F1"],
        "validation_threshold": recomputed["Validation_Selected_Threshold"],
    }
    for key, expected in record_bindings.items():
        if not _equal_metric(record.get(key), expected):
            raise TCMAdapterEvaluationError(
                f"{record['id']} fold {record['fold']} replay field {key} differs"
            )
    return table


def _safe_embedding_scores(
    herb: np.ndarray, disease: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    pair_values = np.asarray(pairs)
    if (
        pair_values.dtype != np.dtype(np.int64)
        or pair_values.ndim != 2
        or pair_values.shape[1] != 2
        or not len(pair_values)
        or int(pair_values.min()) < 0
        or np.any(
            pair_values.max(axis=0)
            >= np.asarray([len(herb), len(disease)], dtype=np.int64)
        )
    ):
        raise TCMAdapterEvaluationError("prediction pairs are non-canonical or out of bounds")
    scores = np.asarray(
        _stage1_embedding_scores(herb, disease, pair_values), dtype=np.float64
    )
    if scores.shape != (len(pair_values),) or not np.isfinite(scores).all():
        raise TCMAdapterEvaluationError("embedding scores are invalid")
    return scores


def _validate_selection_sidecar(selection_path: Path) -> str:
    digest = sha256(selection_path)
    sidecar = selection_path.with_suffix(selection_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMAdapterEvaluationError("stage-2 selection SHA sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields != [digest, selection_path.name]:
        raise TCMAdapterEvaluationError("stage-2 selection SHA sidecar differs")
    return digest


def _replay_selection(
    selection_path: Path, *, output_root: Path
) -> EvaluationPreflight:
    selection_path = selection_path.resolve(strict=True)
    selection_sha = _validate_selection_sidecar(selection_path)
    selection = _load_json(selection_path)
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("dataset") != "tcm-suite"
        or int(selection.get("seed", -1)) != 42
        or selection.get("outer_test_accessed") is not False
        or selection.get("outer_test_evaluator_in_this_module") is not False
    ):
        raise TCMAdapterEvaluationError("stage-2 selection envelope differs")

    selector = selection.get("selector")
    if not isinstance(selector, Mapping):
        raise TCMAdapterEvaluationError("stage-2 selector binding is absent")
    selector_path = Path(str(selector.get("path", ""))).resolve(strict=True)
    expected_selector = Path(
        sys.modules[_selection_winner.__module__].__file__ or ""
    ).resolve(strict=True)
    if (
        selector_path != expected_selector
        or selector.get("sha256") != sha256(selector_path)
    ):
        raise TCMAdapterEvaluationError("stage-2 selector changed after selection")

    plan_record = selection.get("plan")
    if not isinstance(plan_record, Mapping):
        raise TCMAdapterEvaluationError("stage-2 plan binding is absent")
    plan_path = Path(str(plan_record.get("path", ""))).resolve(strict=True)
    try:
        plan_sha = _selection_validate_plan_sidecar(plan_path)
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    if plan_record.get("sha256") != plan_sha:
        raise TCMAdapterEvaluationError("selection is bound to another stage-2 plan")
    plan = _load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen-before-stage2-candidate-testing"
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != 42
        or dict(plan.get("fixed", {})) != EXPECTED_FIXED
    ):
        raise TCMAdapterEvaluationError("stage-2 plan contract differs")
    for artifact_name in ("stage1_selection", "stage1_plan"):
        try:
            _selection_verify_artifact(
                plan.get(artifact_name), label=artifact_name.replace("_", "-")
            )
        except Exception as error:
            raise TCMAdapterEvaluationError(str(error)) from error
    try:
        candidate_by_id = _selection_candidate_map(plan)
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    observed_candidates = {
        key: str(value.get("semantic_adapter"))
        for key, value in candidate_by_id.items()
    }
    if observed_candidates != EXPECTED_CANDIDATES:
        raise TCMAdapterEvaluationError("stage-2 candidate arms differ")
    folds = plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping) or set(folds) != {str(i) for i in range(1, 6)}:
        raise TCMAdapterEvaluationError("stage-2 plan does not contain five folds")

    candidate_root = Path(str(selection.get("candidate_root", ""))).resolve(
        strict=True
    )
    output_root = output_root.resolve()
    try:
        output_root.relative_to(candidate_root)
    except ValueError:
        pass
    else:
        raise TCMAdapterEvaluationError(
            "outer output must not be inside the frozen candidate tree"
        )

    package = _read_protocol_metadata(plan)
    feature_record = plan.get("feature_bundle")
    if not isinstance(feature_record, Mapping):
        raise TCMAdapterEvaluationError("feature-bundle binding is absent")
    try:
        bundle_manifest_path = _selection_verify_artifact(
            feature_record.get("manifest"), label="feature bundle manifest"
        )
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    bundle_manifest = _load_json(bundle_manifest_path)
    try:
        herb_shape = tuple(
            int(value) for value in bundle_manifest["entities"]["herb"]["shape"]
        )
        disease_shape = tuple(
            int(value)
            for value in bundle_manifest["entities"]["disease"]["shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TCMAdapterEvaluationError(
            "feature bundle lacks herb/disease shapes"
        ) from error
    protocol_counts = package.manifest["counts"]
    if herb_shape != (protocol_counts["herbs"], 64) or disease_shape != (
        protocol_counts["diseases"],
        64,
    ):
        raise TCMAdapterEvaluationError(
            "feature-bundle dimensions differ from the protocol"
        )

    fold_payload: dict[str, Any] = {}
    all_records: list[Mapping[str, Any]] = []
    winner_preflights: list[WinnerPreflight] = []
    for fold in range(1, 6):
        records: list[Mapping[str, Any]] = []
        validation_tables: dict[str, object] = {}
        for candidate in candidate_by_id.values():
            try:
                record = _selection_validate_candidate(
                    plan=plan,
                    plan_sha256=plan_sha,
                    candidate_root=candidate_root,
                    candidate=candidate,
                    fold=fold,
                    herb_shape=herb_shape,
                    disease_shape=disease_shape,
                )
            except Exception as error:
                raise TCMAdapterEvaluationError(
                    f"candidate replay failed for fold {fold}: {error}"
                ) from error
            records.append(record)
            all_records.append(record)
            validation_tables[str(record["id"])] = _audit_candidate_validation(
                record
            )

        shared_hashes = {
            json.dumps(
                record["shared_initial_variable_sha256"], sort_keys=True
            )
            for record in records
        }
        if len(shared_hashes) != 1:
            raise TCMAdapterEvaluationError(
                f"fold {fold} adapter arms have unequal shared initialisation"
            )
        row_identities = {
            (
                record["validation_row_id_sha256"],
                record["validation_pair_sha256"],
                record["validation_label_sha256"],
            )
            for record in records
        }
        if len(row_identities) != 1:
            raise TCMAdapterEvaluationError(
                f"fold {fold} adapter arms use different validation rows"
            )
        selected = dict(_selection_winner(records))
        reconstructed_fold = {
            "selected_candidate": selected["id"],
            "selected_adapter": selected["semantic_adapter"],
            "selected_validation_auprc": selected["validation_auprc"],
            "selected_validation_auroc": selected["validation_auroc"],
            "selected_validation_threshold": selected["validation_threshold"],
            "winner": selected,
            "candidates": records,
        }
        fold_payload[str(fold)] = reconstructed_fold

        validation_path = Path(str(selected["validation_predictions"])).resolve(
            strict=True
        )
        table = validation_tables[str(selected["id"])]
        validation_identity = _canonical_prediction_identity(table)
        herb_path = Path(str(selected["herb_embedding"])).resolve(strict=True)
        disease_path = Path(str(selected["disease_embedding"])).resolve(
            strict=True
        )
        expected_candidate_dir = (
            candidate_root / str(selected["id"]) / f"fold_{fold}"
        ).resolve(strict=True)
        try:
            expected_candidate_dir.relative_to(candidate_root)
        except ValueError as error:
            raise TCMAdapterEvaluationError(
                f"fold {fold} candidate directory escapes the candidate root"
            ) from error
        for path, label in (
            (herb_path, "herb embedding"),
            (disease_path, "disease embedding"),
            (validation_path, "validation predictions"),
        ):
            try:
                path.relative_to(expected_candidate_dir)
            except ValueError as error:
                raise TCMAdapterEvaluationError(
                    f"selected {label} escapes its candidate directory"
                ) from error
        if (
            sha256(herb_path) != selected["herb_embedding_sha256"]
            or sha256(disease_path) != selected["disease_embedding_sha256"]
            or sha256(validation_path)
            != selected["validation_predictions_sha256"]
        ):
            raise TCMAdapterEvaluationError(
                f"fold {fold} winner artifact changed after selection"
            )
        herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
        disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
        if (
            herb.shape != herb_shape
            or disease.shape != disease_shape
            or not np.isfinite(herb).all()
            or not np.isfinite(disease).all()
        ):
            raise TCMAdapterEvaluationError(
                f"fold {fold} winner embeddings are invalid"
            )
        reproduced = _safe_embedding_scores(herb, disease, table.pair)
        if not np.allclose(reproduced, table.score, rtol=0.0, atol=1e-12):
            raise TCMAdapterEvaluationError(
                f"fold {fold} winner validation scores are not reproducible"
            )
        validation_metrics = evaluate_validation_scores(table.label, table.score)
        threshold = float(selected["validation_threshold"])
        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or not math.isclose(
                threshold,
                float(validation_metrics["Validation_Selected_Threshold"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(selected["validation_auprc"]),
                float(validation_metrics["Validation_AUPRC"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(selected["validation_auroc"]),
                float(validation_metrics["Validation_AUROC"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise TCMAdapterEvaluationError(
                f"fold {fold} frozen validation threshold/metrics differ"
            )
        herb.setflags(write=False)
        disease.setflags(write=False)
        winner_preflights.append(
            WinnerPreflight(
                fold=fold,
                candidate_id=str(selected["id"]),
                adapter=str(selected["semantic_adapter"]),
                epochs=int(selected["epochs"]),
                learning_rate=float(selected["learning_rate"]),
                threshold=threshold,
                validation_auprc=float(selected["validation_auprc"]),
                validation_auroc=float(selected["validation_auroc"]),
                validation_path=validation_path,
                validation_sha256=sha256(validation_path),
                herb_path=herb_path,
                herb_sha256=sha256(herb_path),
                disease_path=disease_path,
                disease_sha256=sha256(disease_path),
                herb=herb,
                disease=disease,
                validation_identity=validation_identity,
            )
        )

    for field in ("runner_sha256", "model_sha256"):
        observed = {str(record[field]) for record in all_records}
        if len(observed) != 1 or not next(iter(observed)):
            raise TCMAdapterEvaluationError(
                f"candidate replay does not share one {field}"
            )
    reconstructed = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "selector": {
            "path": str(selector_path),
            "sha256": sha256(selector_path),
        },
        "selection_metric": "Validation_AUPRC",
        "tie_break": plan.get("tie_break"),
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "interpretation": "seed-42 same-split post-baseline development diagnostic",
        "folds": fold_payload,
    }
    if dict(selection) != reconstructed:
        raise TCMAdapterEvaluationError(
            "stage-2 selection differs from deterministic candidate replay"
        )
    if len(winner_preflights) != 5 or len(all_records) != 10:
        raise TCMAdapterEvaluationError("preflight did not audit five folds/two arms")
    return EvaluationPreflight(
        selection_path=selection_path,
        selection_sha256=selection_sha,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        candidate_root=candidate_root,
        selection=selection,
        plan=plan,
        protocol_package=package,
        winners=tuple(winner_preflights),
        candidate_records=tuple(all_records),
    )


def _summary(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(fold_metrics) != 5:
        raise TCMAdapterEvaluationError("outer summary requires exactly five folds")
    result: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [float(record[metric]) for record in fold_metrics], dtype=np.float64
        )
        if values.shape != (5,) or not np.isfinite(values).all():
            raise TCMAdapterEvaluationError(
                f"outer metric {metric} is not five finite values"
            )
        result[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "folds": 5,
        }
    return result


def _require_path_hash(path_value: object, digest_value: object, *, label: str) -> None:
    try:
        path = Path(str(path_value)).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise TCMAdapterEvaluationError(f"{label} path is unavailable") from error
    if not isinstance(digest_value, str) or sha256(path) != digest_value:
        raise TCMAdapterEvaluationError(f"{label} changed after preflight")


def _verify_preflight_snapshot(preflight: EvaluationPreflight) -> None:
    """Close the long-preflight drift window before burning outer access."""

    if (
        _validate_selection_sidecar(preflight.selection_path)
        != preflight.selection_sha256
    ):
        raise TCMAdapterEvaluationError("stage-2 selection changed after replay")
    try:
        current_plan_sha = _selection_validate_plan_sidecar(preflight.plan_path)
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    if current_plan_sha != preflight.plan_sha256:
        raise TCMAdapterEvaluationError("stage-2 plan changed after replay")
    if sha256(preflight.protocol_package.manifest_path) != (
        preflight.protocol_package.manifest_sha256
    ):
        raise TCMAdapterEvaluationError("protocol manifest changed after preflight")

    for name in ("stage1_selection", "stage1_plan"):
        try:
            _selection_verify_artifact(preflight.plan.get(name), label=name)
        except Exception as error:
            raise TCMAdapterEvaluationError(str(error)) from error
    feature_bundle = preflight.plan.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping):
        raise TCMAdapterEvaluationError("feature-bundle binding disappeared")
    try:
        _selection_verify_artifact(
            feature_bundle.get("manifest"), label="feature bundle manifest"
        )
    except Exception as error:
        raise TCMAdapterEvaluationError(str(error)) from error
    source_freeze = preflight.plan.get("source_freeze")
    if not isinstance(source_freeze, Mapping):
        raise TCMAdapterEvaluationError("source freeze disappeared")
    for name, record in source_freeze.items():
        try:
            _selection_verify_artifact(record, label=f"source {name}")
        except Exception as error:
            raise TCMAdapterEvaluationError(str(error)) from error
    folds = preflight.plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping):
        raise TCMAdapterEvaluationError("fold hyperparameters disappeared")
    for fold, fold_record in folds.items():
        if not isinstance(fold_record, Mapping):
            raise TCMAdapterEvaluationError(f"fold {fold} binding disappeared")
        stage1_artifacts = fold_record.get("stage1_winner_artifacts")
        if not isinstance(stage1_artifacts, Mapping):
            raise TCMAdapterEvaluationError(
                f"fold {fold} stage-1 artifacts disappeared"
            )
        for name, record in stage1_artifacts.items():
            try:
                _selection_verify_artifact(
                    record, label=f"fold {fold} stage-1 {name}"
                )
            except Exception as error:
                raise TCMAdapterEvaluationError(str(error)) from error

    artifact_fields = (
        ("validation_predictions", "validation_predictions_sha256"),
        ("herb_embedding", "herb_embedding_sha256"),
        ("disease_embedding", "disease_embedding_sha256"),
        ("metrics", "metrics_sha256"),
        ("protocol", "protocol_sha256"),
        ("adapter_audit", "adapter_audit_sha256"),
        ("runner_path", "runner_sha256"),
        ("model_path", "model_sha256"),
    )
    audited_roots: set[Path] = set()
    for record in preflight.candidate_records:
        candidate_id = str(record["id"])
        fold = int(record["fold"])
        root = (
            preflight.candidate_root / candidate_id / f"fold_{fold}"
        ).resolve(strict=True)
        try:
            root.relative_to(preflight.candidate_root)
        except ValueError as error:
            raise TCMAdapterEvaluationError(
                f"{candidate_id} fold {fold} escapes the candidate root"
            ) from error
        audited_roots.add(root)
        for path_key, hash_key in artifact_fields:
            _require_path_hash(
                record.get(path_key),
                record.get(hash_key),
                label=f"{candidate_id} fold {fold} {path_key}",
            )
    for root in audited_roots:
        try:
            _selection_reject_outer_artifacts(root)
        except Exception as error:
            raise TCMAdapterEvaluationError(str(error)) from error


def _verify_winner_snapshot(preflight: EvaluationPreflight) -> None:
    """Require every scored in-memory winner to retain its frozen backing file."""

    if sha256(preflight.selection_path) != preflight.selection_sha256:
        raise TCMAdapterEvaluationError("selection changed during outer evaluation")
    if sha256(preflight.plan_path) != preflight.plan_sha256:
        raise TCMAdapterEvaluationError("plan changed during outer evaluation")
    for winner in preflight.winners:
        _require_path_hash(
            winner.validation_path,
            winner.validation_sha256,
            label=f"fold {winner.fold} validation predictions",
        )
        _require_path_hash(
            winner.herb_path,
            winner.herb_sha256,
            label=f"fold {winner.fold} herb embedding",
        )
        _require_path_hash(
            winner.disease_path,
            winner.disease_sha256,
            label=f"fold {winner.fold} disease embedding",
        )


def _staging_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}.incomplete")


def _ledger_key(plan_sha256: str, selection_sha256: str) -> str:
    if not _is_sha256(plan_sha256) or not _is_sha256(selection_sha256):
        raise TCMAdapterEvaluationError("outer-access ledger identity is invalid")
    material = f"{plan_sha256}\0{selection_sha256}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _access_path(
    plan_sha256: str,
    selection_sha256: str,
) -> Path:
    root = GLOBAL_LEDGER_ROOT.resolve()
    return root / f"{_ledger_key(plan_sha256, selection_sha256)}.json"


def _assert_output_available(
    output_root: Path, *, access_path: Path | None = None
) -> None:
    staging = _staging_path(output_root)
    checks: list[tuple[Path, str]] = [
        (output_root, "outer output"),
        (staging, "interrupted outer staging directory"),
    ]
    if access_path is not None:
        checks.append((access_path, "global outer-access ledger receipt"))
    for path, label in checks:
        if path.exists() or path.is_symlink():
            raise TCMAdapterEvaluationError(f"refusing existing {label}: {path}")


def _reserve_outer_access(
    preflight: EvaluationPreflight,
    output_root: Path,
) -> tuple[Path, Path]:
    """Reserve the manifest and all five folds before any official test load."""

    access_path = _access_path(
        preflight.plan_sha256,
        preflight.selection_sha256,
    )
    _assert_output_available(output_root, access_path=access_path)
    _ensure_directory(output_root.parent)
    staging = _staging_path(output_root)
    evaluator_path = Path(__file__).resolve()
    # This content-addressed global O_EXCL ledger is deliberately the first
    # mutable reservation.  Copying the selection, changing its path, or
    # choosing another output directory resolves to the same durable key.
    _exclusive_json(
        access_path,
        {
            "schema": ACCESS_SCHEMA,
            "state": "plan-and-selection-content-burned-before-outer-access",
            "ledger_key": _ledger_key(
                preflight.plan_sha256, preflight.selection_sha256
            ),
            "selection": str(preflight.selection_path),
            "selection_sha256": preflight.selection_sha256,
            "plan": str(preflight.plan_path),
            "plan_sha256": preflight.plan_sha256,
            "output_root": str(output_root),
            "staging_root": str(staging),
            "candidate_artifacts_audited": len(preflight.candidate_records),
            "fold_reservations_expected": 5,
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": sha256(evaluator_path),
            },
        },
    )
    _durable_mkdir(staging)
    markers: list[dict[str, Any]] = []
    for winner in preflight.winners:
        marker = {
            "schema": ACCESS_SCHEMA,
            "state": "reserved-before-outer-access",
            "selection_sha256": preflight.selection_sha256,
            "output_root": str(output_root),
            "fold": winner.fold,
            "winner": winner.candidate_id,
            "threshold": winner.threshold,
        }
        marker_path = staging / f"fold_{winner.fold}.reserved.json"
        _exclusive_json(marker_path, marker)
        markers.append(
            {
                "path": str(output_root / marker_path.name),
                "sha256": sha256(marker_path),
            }
        )
    preflight_payload = {
        "schema": PREFLIGHT_SCHEMA,
        "state": "all-five-folds-preflight-complete",
        "selection": str(preflight.selection_path),
        "selection_sha256": preflight.selection_sha256,
        "plan": str(preflight.plan_path),
        "plan_sha256": preflight.plan_sha256,
        "protocol_manifest": str(preflight.protocol_package.manifest_path),
        "protocol_manifest_sha256": preflight.protocol_package.manifest_sha256,
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "candidate_artifacts_audited": len(preflight.candidate_records),
        "folds_preflighted": 5,
        "outer_rows_accessed": False,
        "output_root": str(output_root),
        "global_access_ledger": str(access_path),
        "ledger_key": _ledger_key(
            preflight.plan_sha256, preflight.selection_sha256
        ),
        "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
        "reservations": markers,
        "winners": {
            str(item.fold): {
                "candidate": item.candidate_id,
                "adapter": item.adapter,
                "validation_threshold": item.threshold,
                "herb_embedding_sha256": item.herb_sha256,
                "disease_embedding_sha256": item.disease_sha256,
                "validation_predictions_sha256": item.validation_sha256,
                "validation_identity": dict(item.validation_identity),
            }
            for item in preflight.winners
        },
    }
    preflight_path = staging / "preflight.json"
    _atomic_json(preflight_path, preflight_payload)
    return staging, access_path


def _compare_loaded_protocol(
    loaded: BenchmarkProtocol, package: ProtocolMetadataPackage
) -> None:
    manifest = package.manifest
    if (
        loaded.dataset != "tcm-suite"
        or loaded.seed != 42
        or loaded.n_folds != 5
        or not math.isclose(
            float(loaded.validation_fraction),
            float(manifest["validation"]["fraction_per_class"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or loaded.pair_order != manifest.get("pair_order")
        or dict(loaded.mapping_strategy) != dict(manifest["mapping_strategy"])
    ):
        raise TCMAdapterEvaluationError("official protocol reload identity differs")
    counts = manifest["counts"]
    observed_counts = {
        "herbs": loaded.n_herbs,
        "compounds": loaded.n_compounds,
        "proteins": loaded.n_proteins,
        "diseases": loaded.n_diseases,
        "positive_rows": len(loaded.positive_pairs),
        "negative_rows": len(loaded.negative_pairs),
        "herb_compound_rows": len(loaded.herb_compound_pairs),
        "compound_protein_rows": len(loaded.compound_protein_pairs),
        "protein_disease_rows": len(loaded.protein_disease_pairs),
    }
    if observed_counts != dict(counts):
        raise TCMAdapterEvaluationError("official protocol counts differ")
    for name in ARRAY_NAMES:
        values = np.asarray(getattr(loaded, name))
        record = manifest["arrays"][name]
        if dict(record) != {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": _canonical_array_sha256(values),
        }:
            raise TCMAdapterEvaluationError(
                f"official protocol reload differs for {name}"
            )
    if set(loaded.entity_ids) != set(ENTITY_NAMES):
        raise TCMAdapterEvaluationError("official protocol entity mappings differ")
    for entity in ENTITY_NAMES:
        identifiers = list(loaded.entity_ids[entity])
        expected = {
            "count": len(identifiers),
            "sha256": hashlib.sha256(
                _protocol_json_bytes(identifiers)
            ).hexdigest(),
        }
        if dict(manifest["mappings"][entity]) != expected:
            raise TCMAdapterEvaluationError(
                f"official protocol mapping differs for {entity}"
            )


def _capture_protocol_snapshot(
    package: ProtocolMetadataPackage,
) -> tuple[FileSnapshot, ...]:
    """After burn, verify every protocol file and freeze its publish identity."""

    if (
        not package.manifest_path.is_file()
        or sha256(package.manifest_path) != package.manifest_sha256
    ):
        raise TCMAdapterEvaluationError(
            "protocol manifest changed after the outer-access burn"
        )
    snapshots = [
        FileSnapshot(
            label="protocol manifest",
            path=package.manifest_path,
            bytes=package.manifest_path.stat().st_size,
            sha256=package.manifest_sha256,
        )
    ]
    root = package.manifest_path.parent
    artifacts = package.manifest["artifacts"]
    for name, label in (
        (ARRAY_FILENAME, "protocol arrays"),
        (MAPPING_FILENAME, "protocol mappings"),
    ):
        path = root / name
        record = artifacts[name]
        _verify_file(path, record, label=label)
        snapshots.append(
            FileSnapshot(
                label=label,
                path=path,
                bytes=int(record["bytes"]),
                sha256=str(record["sha256"]),
            )
        )
    for name, record in package.manifest["files"].items():
        path = Path(str(record["path"])).resolve()
        _verify_file(path, record, label=f"protocol source {name}")
        snapshots.append(
            FileSnapshot(
                label=f"protocol source {name}",
                path=path,
                bytes=int(record["bytes"]),
                sha256=str(record["sha256"]),
            )
        )
    return tuple(snapshots)


def _verify_protocol_snapshot(snapshots: Sequence[FileSnapshot]) -> None:
    for snapshot in snapshots:
        if (
            not snapshot.path.is_file()
            or snapshot.path.stat().st_size != snapshot.bytes
            or sha256(snapshot.path) != snapshot.sha256
        ):
            raise TCMAdapterEvaluationError(
                f"{snapshot.label} changed before atomic publication"
            )


def _canonical_split_arrays(
    *,
    row_id_value: object,
    pair_value: object,
    label_value: object,
    total_rows: int,
    n_herbs: int,
    n_diseases: int,
    split_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_row_id = np.asarray(row_id_value)
    raw_pair = np.asarray(pair_value)
    raw_label = np.asarray(label_value)
    if (
        raw_row_id.ndim != 1
        or not np.issubdtype(raw_row_id.dtype, np.integer)
        or raw_pair.ndim != 2
        or raw_pair.shape[1] != 2
        or not np.issubdtype(raw_pair.dtype, np.integer)
        or raw_label.ndim != 1
        or not np.issubdtype(raw_label.dtype, np.integer)
    ):
        raise TCMAdapterEvaluationError(
            f"{split_label} row/pair/label arrays are non-canonical"
        )
    row_id = np.asarray(raw_row_id, dtype=np.int64)
    pair = np.asarray(raw_pair, dtype=np.int64)
    label = np.asarray(raw_label, dtype=np.int64)
    if (
        len(row_id) == 0
        or len(pair) != len(row_id)
        or len(label) != len(row_id)
        or len(np.unique(row_id)) != len(row_id)
        or int(row_id.min()) < 0
        or int(row_id.max()) >= total_rows
        or int(pair.min()) < 0
        or np.any(
            pair.max(axis=0)
            >= np.asarray([n_herbs, n_diseases], dtype=np.int64)
        )
        or not np.isin(label, np.asarray([0, 1], dtype=np.int64)).all()
    ):
        raise TCMAdapterEvaluationError(
            f"{split_label} row/pair/label coverage is invalid"
        )
    for value in (row_id, pair, label):
        value.setflags(write=False)
    return row_id, pair, label


def _verify_split_manifest(
    *,
    record: Mapping[str, Any],
    row_id: np.ndarray,
    label: np.ndarray,
    split_label: str,
) -> None:
    if (
        record.get("rows") != len(row_id)
        or record.get("positive_rows") != int(label.sum())
        or record.get("negative_rows") != int(len(label) - label.sum())
        or record.get("row_indices_sha256")
        != _canonical_array_sha256(row_id)
    ):
        raise TCMAdapterEvaluationError(
            f"{split_label} rows differ from frozen protocol manifest"
        )


def _audit_loaded_folds(
    *,
    protocol: BenchmarkProtocol,
    preflight: EvaluationPreflight,
) -> tuple[AuditedOuterFold, ...]:
    """Audit all five validation/test folds before the first test score."""

    all_pairs = np.asarray(protocol.all_pairs, dtype=np.int64)
    all_labels = np.asarray(protocol.all_labels, dtype=np.int64)
    total_rows = len(all_pairs)
    audited: list[AuditedOuterFold] = []
    all_test_rows: list[np.ndarray] = []
    for winner in preflight.winners:
        rows = protocol.fold_rows(winner.fold)
        manifest_fold = _manifest_fold(preflight.protocol_package, winner.fold)
        split_values = {
            "train": (
                rows.train_row_indices,
                rows.train_pairs,
                rows.train_labels,
            ),
            "validation": (
                rows.validation_row_indices,
                rows.validation_pairs,
                rows.validation_labels,
            ),
            "test": (rows.test_row_indices, rows.test_pairs, rows.test_labels),
        }
        canonical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for split, values in split_values.items():
            record = manifest_fold.get(split)
            if not isinstance(record, Mapping):
                raise TCMAdapterEvaluationError(
                    f"fold {winner.fold} {split} manifest is absent"
                )
            row_id, pair, label = _canonical_split_arrays(
                row_id_value=values[0],
                pair_value=values[1],
                label_value=values[2],
                total_rows=total_rows,
                n_herbs=protocol.n_herbs,
                n_diseases=protocol.n_diseases,
                split_label=f"fold {winner.fold} {split}",
            )
            _verify_split_manifest(
                record=record,
                row_id=row_id,
                label=label,
                split_label=f"fold {winner.fold} {split}",
            )
            if not np.array_equal(pair, all_pairs[row_id]) or not np.array_equal(
                label, all_labels[row_id]
            ):
                raise TCMAdapterEvaluationError(
                    f"fold {winner.fold} {split} pair/label identity differs"
                )
            canonical[split] = (row_id, pair, label)

        combined = np.concatenate(
            [canonical[name][0] for name in ("train", "validation", "test")]
        )
        if not np.array_equal(np.sort(combined), np.arange(total_rows)):
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} split row coverage differs"
            )

        validation_row, validation_pair, validation_label = canonical[
            "validation"
        ]
        try:
            table = validate_prediction_coverage(
                winner.validation_path,
                expected_row_id=validation_row,
                expected_pair=validation_pair,
                expected_label=validation_label,
            )
        except Exception as error:
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} validation row/pair/label identity differs"
            ) from error
        if _canonical_prediction_identity(table) != dict(
            winner.validation_identity
        ):
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} validation identity changed after burn"
            )
        reproduced = _safe_embedding_scores(
            winner.herb, winner.disease, validation_pair
        )
        if not np.allclose(reproduced, table.score, rtol=0.0, atol=1e-12):
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} validation scores changed after burn"
            )
        validation_metrics = evaluate_validation_scores(table.label, table.score)
        if (
            not _equal_metric(
                validation_metrics["Validation_Selected_Threshold"],
                winner.threshold,
            )
            or not _equal_metric(
                validation_metrics["Validation_AUPRC"], winner.validation_auprc
            )
            or not _equal_metric(
                validation_metrics["Validation_AUROC"], winner.validation_auroc
            )
        ):
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} frozen validation decision differs after burn"
            )

        test_row, test_pair, test_label = canonical["test"]
        all_test_rows.append(test_row)
        audited.append(
            AuditedOuterFold(
                winner=winner,
                row_id=test_row,
                pair=test_pair,
                label=test_label,
            )
        )
    if len(audited) != 5 or not np.array_equal(
        np.sort(np.concatenate(all_test_rows)), np.arange(total_rows)
    ):
        raise TCMAdapterEvaluationError(
            "five-fold outer test rows do not cover each association exactly once"
        )
    return tuple(audited)


def _validated_test_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    test_evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    recomputed = dict(evaluate_test_scores(labels, scores, threshold=threshold))
    callback_labels = np.asarray(labels).copy()
    callback_scores = np.asarray(scores).copy()
    callback_labels.setflags(write=False)
    callback_scores.setflags(write=False)
    reported = dict(
        test_evaluator(
            callback_labels,
            callback_scores,
            threshold=threshold,
        )
    )
    if set(reported) != set(recomputed):
        raise TCMAdapterEvaluationError("outer metric field set is not reproducible")
    for key, expected in recomputed.items():
        if not _equal_metric(reported.get(key), expected):
            raise TCMAdapterEvaluationError(
                f"outer metric {key} cannot be independently reproduced"
            )
    return recomputed


def _evaluate_reserved(
    *,
    preflight: EvaluationPreflight,
    output_root: Path,
    staging: Path,
    access_path: Path,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    test_evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """The only phase permitted to load fold test rows."""

    _verify_winner_snapshot(preflight)
    # The official loader itself constructs folds while validating its
    # manifest.  It is therefore deliberately called only after all-five
    # preflight and reservation receipts exist.
    protocol = protocol_loader(preflight.protocol_package.manifest_path)
    _compare_loaded_protocol(protocol, preflight.protocol_package)
    protocol_snapshots = _capture_protocol_snapshot(preflight.protocol_package)
    audited_folds = _audit_loaded_folds(
        protocol=protocol,
        preflight=preflight,
    )
    fold_metrics: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    score_call_count = 0
    for audited in audited_folds:
        winner = audited.winner
        row_id = audited.row_id
        pair = audited.pair
        label = audited.label
        scores = _safe_embedding_scores(winner.herb, winner.disease, pair)
        score_call_count += 1
        metrics = _validated_test_metrics(
            labels=label,
            scores=scores,
            threshold=winner.threshold,
            test_evaluator=test_evaluator,
        )
        if (
            not math.isclose(
                float(metrics.get("Operating_Threshold", math.nan)),
                winner.threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or metrics.get("AUPRC_Definition") != AUPRC_DEFINITION
        ):
            raise TCMAdapterEvaluationError(
                f"fold {winner.fold} did not apply the frozen threshold"
            )
        final_fold_dir = output_root / f"fold_{winner.fold}"
        staged_fold_dir = staging / f"fold_{winner.fold}"
        _durable_mkdir(staged_fold_dir)
        prediction_path = save_predictions(
            staged_fold_dir / "test_predictions.npz",
            row_id=row_id,
            pair=pair,
            label=label,
            score=scores,
        )
        _fsync_file_and_parent(prediction_path)
        table = validate_prediction_coverage(
            prediction_path,
            expected_row_id=row_id,
            expected_pair=pair,
            expected_label=label,
        )
        test_identity = _canonical_prediction_identity(table)
        record = {
            "Dataset": "tcm-suite",
            "Method": "ASTRA-TCM-semantic-adapter-inner-selected",
            "Fold": winner.fold,
            "Seed": 42,
            "Diagnostic": True,
            "Formal": False,
            "Formal_Eligible": False,
            "Seed42_Already_Observed": True,
            "Formal_Ineligibility_Reasons": [
                "seed-42 outer folds were observed before this development stage"
            ],
            "Selected_Candidate": winner.candidate_id,
            "Selected_Adapter": winner.adapter,
            "Selected_Epochs": winner.epochs,
            "Selected_Learning_Rate": winner.learning_rate,
            "Selected_Validation_AUPRC": winner.validation_auprc,
            "Selected_Validation_AUROC": winner.validation_auroc,
            "Selected_Validation_Threshold": winner.threshold,
            "Selection_Manifest": str(preflight.selection_path),
            "Selection_Manifest_SHA256": preflight.selection_sha256,
            "Validation_Predictions": str(winner.validation_path),
            "Validation_Predictions_SHA256": winner.validation_sha256,
            "Winner_Herb_Embedding": str(winner.herb_path),
            "Winner_Herb_Embedding_SHA256": winner.herb_sha256,
            "Winner_Disease_Embedding": str(winner.disease_path),
            "Winner_Disease_Embedding_SHA256": winner.disease_sha256,
            "Test_Predictions": str(final_fold_dir / "test_predictions.npz"),
            "Test_Predictions_SHA256": sha256(prediction_path),
            "Test_Row_ID_SHA256": test_identity["row_id_sha256"],
            "Test_Pair_SHA256": test_identity["pair_sha256"],
            "Test_Label_SHA256": test_identity["label_sha256"],
            "Test_Score_SHA256": test_identity["score_sha256"],
            "Test_Array_Hash_Schema": ARRAY_HASH_SCHEMA,
            "Test_Models_Evaluated": 1,
            "Unselected_Test_Models_Evaluated": 0,
            **metrics,
        }
        _atomic_json(staged_fold_dir / "metrics.json", record)
        _atomic_json(
            staged_fold_dir / "selection_binding.json",
            {
                "schema": "astra-tcm-semantic-adapter-outer-fold-binding/v1",
                "selection": str(preflight.selection_path),
                "selection_sha256": preflight.selection_sha256,
                "plan": str(preflight.plan_path),
                "plan_sha256": preflight.plan_sha256,
                "fold": winner.fold,
                "winner": winner.candidate_id,
                "adapter": winner.adapter,
                "validation_threshold": winner.threshold,
                "threshold_source": "frozen stage-2 winner validation maximum F1",
                "winner_embeddings_only": True,
                "test_models_evaluated": 1,
                "unselected_test_models_evaluated": 0,
                "test_predictions_sha256": sha256(prediction_path),
                "array_hash_schema": ARRAY_HASH_SCHEMA,
                "test_identity": test_identity,
            },
        )
        fold_metrics.append(record)
        selected[str(winner.fold)] = winner.candidate_id

    if score_call_count != 5:
        raise TCMAdapterEvaluationError(
            f"outer scoring count is {score_call_count}, expected 5"
        )
    _verify_winner_snapshot(preflight)
    evaluator_path = Path(__file__).resolve()
    payload = {
        "schema": EVALUATION_SCHEMA,
        "status": "diagnostic-complete",
        "dataset": "tcm-suite",
        "method": "ASTRA-TCM-semantic-adapter-inner-selected",
        "seed": 42,
        "Diagnostic": True,
        "Formal": False,
        "Formal_Eligible": False,
        "Seed42_Already_Observed": True,
        "formal_ineligibility_reasons": [
            "seed-42 outer folds were observed before this development stage"
        ],
        "selection_manifest": str(preflight.selection_path),
        "selection_manifest_sha256": preflight.selection_sha256,
        "plan": str(preflight.plan_path),
        "plan_sha256": preflight.plan_sha256,
        "protocol_manifest": str(preflight.protocol_package.manifest_path),
        "protocol_manifest_sha256": preflight.protocol_package.manifest_sha256,
        "global_access_ledger": str(access_path),
        "ledger_key": _ledger_key(
            preflight.plan_sha256, preflight.selection_sha256
        ),
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
        "candidate_artifacts_preflighted": len(preflight.candidate_records),
        "folds_preflighted_before_outer_access": 5,
        "outer_score_calls": score_call_count,
        "test_models_evaluated": 5,
        "unselected_test_models_evaluated": 0,
        "selected_candidates": selected,
        "fold_metrics": fold_metrics,
        "summary": _summary(fold_metrics),
        "interpretation": (
            "Seed-42 same-split post-baseline development diagnostic. Adapter, "
            "epoch, learning rate, and threshold were selected only from inner "
            "validation, but the seed-42 outer folds had already been observed; "
            "therefore this is not an independently confirmed formal result."
        ),
    }
    summary_path = staging / "summary.json"
    _atomic_json(summary_path, payload)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "state": "complete-before-atomic-publish",
        "selection_sha256": preflight.selection_sha256,
        "output_root": str(output_root),
        "summary": str(output_root / "summary.json"),
        "summary_sha256": sha256(summary_path),
        "folds": 5,
        "outer_score_calls": score_call_count,
        "Diagnostic": True,
        "Formal": False,
    }
    _atomic_json(staging / "completion_receipt.json", completion)
    # This is deliberately the final read-only gate: every one of the ten
    # candidates and every protocol file is re-hashed after scoring and before
    # the directory can become a published result.
    _verify_preflight_snapshot(preflight)
    _verify_protocol_snapshot(protocol_snapshots)
    if output_root.exists() or output_root.is_symlink():
        raise TCMAdapterEvaluationError(
            f"outer output appeared during evaluation: {output_root}"
        )
    _fsync_directory(staging)
    os.rename(staging, output_root)
    _fsync_directory(output_root.parent)
    return payload


def _evaluate_selected_with(
    *,
    selection_path: Path,
    output_root: Path,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    test_evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output_root = output_root.resolve()
    _assert_output_available(output_root)
    preflight = _replay_selection(selection_path, output_root=output_root)
    access_path = _access_path(
        preflight.plan_sha256,
        preflight.selection_sha256,
    )
    # Re-check after the potentially long preflight to close ordinary races.
    _assert_output_available(output_root, access_path=access_path)
    _verify_preflight_snapshot(preflight)
    _assert_output_available(output_root, access_path=access_path)
    staging, access_path = _reserve_outer_access(preflight, output_root)
    return _evaluate_reserved(
        preflight=preflight,
        output_root=output_root,
        staging=staging,
        access_path=access_path,
        protocol_loader=protocol_loader,
        test_evaluator=test_evaluator,
    )


def evaluate_selected(*, selection_path: Path, output_root: Path) -> dict[str, Any]:
    """Evaluate the five frozen winners exactly once as a diagnostic."""

    return _evaluate_selected_with(
        selection_path=selection_path,
        output_root=output_root,
        protocol_loader=load_protocol_manifest,
        test_evaluator=evaluate_test_scores,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_selected(
            selection_path=args.selection,
            output_root=args.output_root,
        )
    except Exception as error:
        print(f"TCM semantic-adapter outer evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
