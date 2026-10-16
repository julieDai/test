#!/usr/bin/env python3
"""Run ASTRA/HerbHG on one fold of the central fair benchmark protocol.

The numerical implementations are isolated by dataset under
``src/herbhg/models/backends/``. Their original entry points perform their own split
and choose a threshold on the test fold.  This adapter deliberately bypasses
that orchestration: it supplies only the frozen training rows to the model,
scores the frozen validation and test rows after training, and delegates all
metric computation to :mod:`astra.benchmark.metrics`.

Structural H--C, C--P, and P--D files are materialised from the protocol's
de-duplicated arrays.  The legacy model's entity dictionaries are then bound
to the protocol's compact row order before TensorFlow graph construction.
This makes biological and semantic feature row identity explicit even when
the raw relation files use non-consecutive identifiers or a different first-
occurrence order.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .features import (
    FeatureContractError,
    load_aligned_feature_matrix,
)
from .ethnobotany_feature_bundle import (
    FeatureBundleError,
    validate_feature_bundle,
)
from .tcm_feature_bundle import (
    TCMFeatureBundleError,
    validate_tcm_feature_bundle,
)
from .metrics import (
    evaluate_validation_scores,
    evaluate_validation_and_test,
    save_predictions,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, FoldRows, load_protocol_manifest
from astra.official_baselines.common import diagnostic_execution_context
from astra.tf_initialization import initialize_uninitialized_variables
from .backend_paths import (
    ETHNOBOTANY_BACKEND_ROOT,
    TCM_BACKEND_ROOT,
    activate_backend,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SOURCE_ROOT.parent

_DATASET_ALIASES = {
    "ethno": "ethnobotany",
    "ethnobotany": "ethnobotany",
    "tcm": "tcm-suite",
    "tcmsuite": "tcm-suite",
    "tcm-suite": "tcm-suite",
}

_HYPERPARAMETERS: Mapping[str, Mapping[str, Any]] = {
    "tcm-suite": {
        "epochs": 16,
        "batch_size": 2000,
        "learning_rate": 0.005,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "config": TCM_BACKEND_ROOT / "ASTRA.conf",
        "legacy_root": TCM_BACKEND_ROOT,
    },
    "ethnobotany": {
        "epochs": 30,
        "batch_size": 1000,
        "learning_rate": 0.0001,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "config": ETHNOBOTANY_BACKEND_ROOT / "ethnobotany.conf",
        "legacy_root": ETHNOBOTANY_BACKEND_ROOT,
    },
}

_FEATURE_FILENAMES = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
    "herb": "herb_semantic_embeddings_64d.npy",
    "disease": "disease_semantic_embeddings_64d.npy",
}

class ASTRARunnerError(RuntimeError):
    """Raised when a run cannot satisfy the central benchmark contract."""


@contextmanager
def _exclusive_run_lock(output_root: Path):
    """Fail closed if another process is writing the same fold directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".astra_run.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ASTRARunnerError(
                f"another ASTRA process holds the run lock {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parse_unit_float(value: str | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"expected a finite number in [0, 1], got {value!r}"
        ) from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite number in [0, 1], got {value!r}"
        )
    return parsed


def _normalise_dataset(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    try:
        return _DATASET_ALIASES[key]
    except KeyError as error:
        raise ASTRARunnerError(
            f"unsupported dataset {value!r}; expected ethnobotany or tcm-suite"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validated_alpha_selection(
    manifest_path: Path,
    *,
    dataset: str,
    seed: int,
    protocol_manifest: Path,
    embedding_dir: Path,
    fold: int,
    alpha: float,
) -> dict[str, Any]:
    """Audit all candidates, then bind one selected formal run to its decision."""

    path = manifest_path.resolve()
    try:
        from .ethnobotany_alpha_selection import validate_selection_decision

        _, payload, fold_records = validate_selection_decision(
            selection_path=path,
            protocol_manifest=protocol_manifest,
            output_root=path.parent,
            embedding_dir=embedding_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ASTRARunnerError(
            f"alpha-selection decision audit failed before test scoring: {error}"
        ) from error
    if payload.get("dataset") != dataset or payload.get("seed") != seed:
        raise ASTRARunnerError(
            "alpha-selection manifest dataset/seed differs from this run"
        )
    fold_record = fold_records.get(str(fold))
    if not isinstance(fold_record, Mapping):
        raise ASTRARunnerError(
            f"alpha-selection manifest has no decision for fold {fold}"
        )
    try:
        selected_alpha = float(fold_record["selected_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ASTRARunnerError(
            f"alpha-selection manifest fold {fold} has no valid selected alpha"
        ) from error
    if not math.isclose(selected_alpha, alpha, rel_tol=0.0, abs_tol=1e-15):
        raise ASTRARunnerError(
            f"fold {fold} selected alpha={selected_alpha}, not requested alpha={alpha}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema": payload["schema"],
        "selection_metric": payload.get("selection_metric"),
        "tie_break": payload.get("tie_break"),
        "candidate_alphas": payload.get("candidate_alphas"),
        "selected_alpha": selected_alpha,
        "selected_validation_auprc": fold_record.get(
            "selected_validation_auprc"
        ),
    }


def _write_lines(path: Path, rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write("\t".join(str(value) for value in row) + "\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _raw_pair_rows(
    protocol: BenchmarkProtocol,
    pairs: np.ndarray,
    left_entity: str,
    right_entity: str,
) -> list[tuple[str, str]]:
    return [
        (
            protocol.compact_to_raw(left_entity, int(left)),
            protocol.compact_to_raw(right_entity, int(right)),
        )
        for left, right in np.asarray(pairs, dtype=np.int64)
    ]


def materialise_protocol_relations(
    protocol: BenchmarkProtocol, output_dir: Path
) -> dict[str, Path]:
    """Write a header-free, de-duplicated legacy relation view."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "H_D": output_dir / "H_D.txt",
        "H_C": output_dir / "H_C.txt",
        "C_P": output_dir / "C_P.txt",
        "P_D": output_dir / "P_D.txt",
    }
    hd = [
        (*pair, 1)
        for pair in _raw_pair_rows(
            protocol, protocol.positive_pairs, "herb", "disease"
        )
    ]
    _write_lines(paths["H_D"], hd)
    _write_lines(
        paths["H_C"],
        _raw_pair_rows(
            protocol, protocol.herb_compound_pairs, "herb", "compound"
        ),
    )
    _write_lines(
        paths["C_P"],
        _raw_pair_rows(
            protocol, protocol.compound_protein_pairs, "compound", "protein"
        ),
    )
    _write_lines(
        paths["P_D"],
        _raw_pair_rows(
            protocol, protocol.protein_disease_pairs, "protein", "disease"
        ),
    )
    return paths


def compact_pairs_to_training_rows(
    protocol: BenchmarkProtocol, pairs: np.ndarray, labels: np.ndarray
) -> list[list[object]]:
    """Convert protocol compact pairs to the raw IDs expected by Rating."""

    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise ASTRARunnerError("training pairs/labels do not have compatible shapes")
    return [
        [
            protocol.compact_to_raw("herb", int(pair[0])),
            protocol.compact_to_raw("disease", int(pair[1])),
            float(label),
        ]
        for pair, label in zip(pairs, labels)
    ]


def _rebind_model_mappings(model: object, protocol: BenchmarkProtocol) -> None:
    """Bind all legacy integer lookups to the frozen protocol row order."""

    data = model.data
    specifications = (
        ("herb", "Hlist", "num_herbs"),
        ("disease", "Dlist", "num_diseases"),
        ("compound", "Clist", "num_compounds"),
        ("protein", "Plist", "num_proteins"),
    )
    for entity, list_attribute, count_attribute in specifications:
        raw_ids = tuple(str(value) for value in protocol.entity_ids[entity])
        mapping = {raw_id: index for index, raw_id in enumerate(raw_ids)}
        inverse = {index: raw_id for raw_id, index in mapping.items()}
        setattr(data, entity, mapping)
        setattr(data, f"id2{entity}", inverse)
        setattr(data, list_attribute, np.asarray(raw_ids, dtype=str))
        setattr(model, count_attribute, len(raw_ids))

    relation_specs = (
        (data.hcassociation, data.herb, data.compound, "H--C"),
        (data.cpassociation, data.compound, data.protein, "C--P"),
        (data.pdassociation, data.protein, data.disease, "P--D"),
    )
    for relation_rows, left, right, name in relation_specs:
        missing = [
            (str(row[0]), str(row[1]))
            for row in relation_rows
            if str(row[0]) not in left or str(row[1]) not in right
        ]
        if missing:
            raise ASTRARunnerError(
                f"{name} contains {len(missing)} identifiers outside the protocol mapping"
            )

    for raw_herb, raw_disease, _ in data.trainingData:
        if str(raw_herb) not in data.herb or str(raw_disease) not in data.disease:
            raise ASTRARunnerError(
                "a frozen training pair is outside the rebound legacy mappings"
            )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def score_compact_pairs(
    model: object, protocol: BenchmarkProtocol, pairs: np.ndarray
) -> np.ndarray:
    """Score compact pairs through raw IDs and the model's rebound mappings."""

    pairs = np.asarray(pairs, dtype=np.int64)
    herb_indices = np.empty(len(pairs), dtype=np.int64)
    disease_indices = np.empty(len(pairs), dtype=np.int64)
    for index, (herb, disease) in enumerate(pairs):
        raw_herb = protocol.compact_to_raw("herb", int(herb))
        raw_disease = protocol.compact_to_raw("disease", int(disease))
        try:
            herb_indices[index] = model.data.herb[str(raw_herb)]
            disease_indices[index] = model.data.disease[str(raw_disease)]
        except KeyError as error:
            raise ASTRARunnerError(
                f"prediction identifier {error.args[0]!r} is absent from model.data"
            ) from error
    herb_embeddings = np.asarray(model.u, dtype=np.float64)[herb_indices]
    disease_embeddings = np.asarray(model.i, dtype=np.float64)[disease_indices]
    logits = np.einsum("ij,ij->i", herb_embeddings, disease_embeddings)
    scores = _stable_sigmoid(logits)
    if not np.isfinite(scores).all():
        raise ASTRARunnerError("ASTRA emitted non-finite prediction scores")
    return scores


def _smoke_training_rows(
    fold_rows: FoldRows, *, per_class: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(fold_rows.train_labels, dtype=np.uint8)
    selected: list[np.ndarray] = []
    for label in (1, 0):
        indices = np.flatnonzero(labels == label)
        if not len(indices):
            raise ASTRARunnerError(
                f"smoke training split has no examples for class {label}"
            )
        selected.append(indices[: min(per_class, len(indices))])
    indices = np.concatenate(selected)
    return fold_rows.train_pairs[indices], fold_rows.train_labels[indices]


def _smoke_evaluation_rows(
    pairs: np.ndarray,
    labels: np.ndarray,
    row_id: np.ndarray,
    *,
    per_class: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the common deterministic smoke subset used by every method."""

    labels = np.asarray(labels, dtype=np.uint8)
    selected: list[np.ndarray] = []
    for label in (1, 0):
        positions = np.flatnonzero(labels == label)
        if not len(positions):
            raise ASTRARunnerError(
                f"smoke evaluation split has no examples for class {label}"
            )
        selected.append(positions[: min(per_class, len(positions))])
    indices = np.concatenate(selected)
    return pairs[indices], labels[indices], row_id[indices]


def _matrix_diagnostics(
    path: Path, *, expected_rows: int, expected_columns: int = 64
) -> dict[str, Any]:
    if not path.is_file():
        raise ASTRARunnerError(f"required feature matrix does not exist: {path}")
    matrix = np.load(path) if path.suffix.lower() == ".npy" else np.loadtxt(path)
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape != (expected_rows, expected_columns):
        raise ASTRARunnerError(
            f"{path} has shape {matrix.shape}; expected "
            f"({expected_rows}, {expected_columns})"
        )
    if not np.isfinite(matrix).all():
        raise ASTRARunnerError(f"{path} contains NaN or infinity")
    varying_columns = int(np.count_nonzero(np.ptp(matrix, axis=0) > 0.0))
    unique_rows = int(np.unique(matrix, axis=0).shape[0])
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "unique_rows": unique_rows,
        "varying_columns": varying_columns,
        "zero_variance_columns": int(matrix.shape[1] - varying_columns),
        "degenerate": bool(unique_rows < 2 or varying_columns == 0),
    }


def _materialise_aligned_feature(
    source_path: Path,
    matrix: np.ndarray,
    output_dir: Path,
) -> Path:
    """Atomically write a canonical-order matrix for the legacy loader."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source_path.name
    temporary: Path | None = None
    binary = source_path.suffix.lower() == ".npy"
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w",
            encoding=None if binary else "utf-8",
            dir=output_dir,
            prefix=f".{source_path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            if binary:
                np.save(handle, matrix, allow_pickle=False)
            else:
                np.savetxt(handle, matrix, fmt="%.18e")
            handle.flush()
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _validate_and_align_feature_paths(
    dataset: str,
    protocol: BenchmarkProtocol,
    feature_paths: Mapping[str, Path],
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Validate and canonicalise feature row identity for either dataset."""

    aligned_paths: dict[str, Path] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    mapping_manifest: Mapping[str, Mapping[str, Any]] | None = None
    for entity, path in feature_paths.items():
        mapping_source = protocol.source_files.get(f"mapping/{entity}")
        if mapping_source is None and mapping_manifest is None:
            mapping_manifest = protocol.input_manifest()["mappings"]
        mapping_sha256 = (
            str(mapping_source["sha256"])
            if mapping_source is not None
            else str(mapping_manifest[entity]["sha256"])  # type: ignore[index]
        )
        try:
            aligned = load_aligned_feature_matrix(
                path,
                entity=entity,
                canonical_ids=protocol.entity_ids[entity],
                mapping_sha256=mapping_sha256,
            )
        except FeatureContractError as error:
            raise ASTRARunnerError(str(error)) from error
        details = dict(aligned.diagnostics)
        if details["source_order_is_canonical"]:
            used_path = path
        else:
            used_path = _materialise_aligned_feature(
                path,
                aligned.matrix,
                output_root / "aligned_features",
            )
        details["used_path"] = str(used_path.resolve())
        details["used_bytes"] = used_path.stat().st_size
        details["used_sha256"] = _sha256(used_path)
        aligned_paths[entity] = used_path
        diagnostics[entity] = details
    return aligned_paths, diagnostics


def _validate_tcm_feature_bundle(
    protocol: BenchmarkProtocol,
    embedding_dir: Path,
    feature_paths: Mapping[str, Path],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Validate the four-entity TCM feature/provenance bundle."""

    manifest_path = embedding_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        if smoke:
            return {
                "present": False,
                "formal_eligible": False,
                "warnings": ["bundle_manifest.json is absent in smoke mode"],
            }
        raise ASTRARunnerError(
            "formal TCM-suite runs require embedding-dir/bundle_manifest.json"
        )
    try:
        validated = validate_tcm_feature_bundle(
            manifest_path,
            protocol=protocol,
            expected_feature_paths=feature_paths,
        )
    except TCMFeatureBundleError as error:
        raise ASTRARunnerError(
            f"TCM-suite feature bundle failed validation: {error}"
        ) from error
    return {
        "present": True,
        "path": str(manifest_path.resolve()),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
        "schema": validated["manifest"]["schema"],
        "profile": validated["manifest"]["profile"],
        "formal_eligible": True,
        "recursive_validation": True,
        "entities": {
            entity: {
                "matrix_sha256": details["matrix_sha256"],
                "row_ids_sidecar_sha256": details["sidecar_sha256"],
            }
            for entity, details in validated["entities"].items()
        },
        "evidence_artifact_count": len(validated["evidence_paths"]),
    }


def _validate_ethnobotany_feature_bundle(
    protocol: BenchmarkProtocol,
    embedding_dir: Path,
    feature_paths: Mapping[str, Path],
    *,
    smoke: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Validate the portable bundle, missing masks, and zero-UNK declarations."""

    manifest_path = embedding_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        if smoke:
            return {}, {
                "present": False,
                "formal_eligible": False,
                "warnings": ["bundle_manifest.json is absent in smoke mode"],
            }
        raise ASTRARunnerError(
            "formal Ethnobotany runs require embedding-dir/bundle_manifest.json"
        )
    # A present bundle is never trusted through the historical shallow smoke
    # checks.  Smoke mode may omit the bundle entirely, but once a manifest is
    # supplied it must satisfy the same recursive provenance contract as a
    # formal run.  This is especially important for the distinct
    # target-not-protein and audited-missing protein masks.
    selected_entities = set(feature_paths)
    biological_entities = {"compound", "protein"}
    all_entities = {"herb", "compound", "protein", "disease"}
    if selected_entities != biological_entities and selected_entities != all_entities:
        raise ASTRARunnerError(
            "Ethnobotany feature consumer must request either c/p or all four entities"
        )
    try:
        profile_hint = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "profile"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as error:
        raise ASTRARunnerError(
            f"cannot read Ethnobotany feature bundle profile: {error}"
        ) from error
    if selected_entities == all_entities:
        recursive_feature_paths = dict(feature_paths)
    elif profile_hint == "herbhg-four-entity":
        recursive_feature_paths = {
            entity: embedding_dir / _FEATURE_FILENAMES[entity]
            for entity in ("herb", "compound", "protein", "disease")
        }
    else:
        recursive_feature_paths = dict(feature_paths)
    expected_mapping_sha256 = {
        entity: str(protocol.source_files[f"mapping/{entity}"]["sha256"])
        for entity in recursive_feature_paths
    }
    try:
        validated = validate_feature_bundle(
            manifest_path,
            expected_mapping_sha256=expected_mapping_sha256,
            expected_row_ids=protocol.entity_ids,
            expected_feature_paths=recursive_feature_paths,
        )
    except FeatureBundleError as error:
        raise ASTRARunnerError(
            f"Ethnobotany feature bundle failed recursive validation: {error}"
        ) from error
    mask_paths = {
        entity: details["mask_path"]
        for entity, details in validated["entities"].items()
        if entity in selected_entities
    }
    entity_diagnostics = {
        entity: {
            "mask_path": str(details["mask_path"]),
            "mask_sha256": _sha256(details["mask_path"]),
            "missing_count": int(details["mask"].sum()),
            "declared_zero_unk_count": int(
                details["component"]["manifest"].get(
                    "declared_zero_unk_count", 0
                )
            ),
            "declared_audited_missing_count": int(
                details["component"]["manifest"].get(
                    "declared_audited_missing_count", 0
                )
            ),
        }
        for entity, details in validated["entities"].items()
    }
    return mask_paths, {
        "present": True,
        "path": str(manifest_path.resolve()),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
        "schema": validated["manifest"]["schema"],
        "formal_eligible": True,
        "recursive_validation": True,
        "entities": entity_diagnostics,
    }


def _assert_formal_feature_integrity(
    feature_diagnostics: Mapping[str, Mapping[str, Any]], *, smoke: bool
) -> list[str]:
    """Return warnings for smoke runs and fail closed for formal runs."""

    degenerate = [
        entity
        for entity, diagnostics in feature_diagnostics.items()
        if bool(diagnostics.get("degenerate"))
    ]
    if degenerate and not smoke:
        raise ASTRARunnerError(
            "formal runs reject degenerate feature matrices (fewer than two "
            "unique rows or no varying columns): "
            + ", ".join(degenerate)
        )
    return degenerate


def _validate_manifest_sources(
    protocol: BenchmarkProtocol, data_dir: Path, embedding_dir: Path
) -> dict[str, dict[str, Any]]:
    """Re-hash the source files frozen into the protocol manifest."""

    checked: dict[str, dict[str, Any]] = {}
    for name, record in protocol.source_files.items():
        recorded_path = Path(str(record.get("path", "")))
        candidate = recorded_path
        if name in {"H_D", "H_C", "C_P", "P_D"}:
            candidate = data_dir / recorded_path.name
        elif name.startswith("mapping/"):
            candidate = data_dir / recorded_path.name
        elif name.startswith("embedding/"):
            candidate = embedding_dir / recorded_path.name
        if not candidate.is_file():
            raise ASTRARunnerError(
                f"protocol source {name} is unavailable at {candidate}"
            )
        observed = {
            "path": str(candidate.resolve()),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        if observed["bytes"] != record.get("bytes") or observed["sha256"] != record.get(
            "sha256"
        ):
            raise ASTRARunnerError(
                f"{name} differs from the file frozen in the protocol manifest"
            )
        checked[name] = observed
    return checked


def _feature_paths(
    dataset: str,
    embedding_dir: Path,
    herb_semantic: Path | None,
    disease_semantic: Path | None,
) -> dict[str, Path]:
    if (herb_semantic is None) != (disease_semantic is None):
        raise ASTRARunnerError(
            "--herb-semantic and --disease-semantic must be supplied together"
        )
    if dataset == "ethnobotany" and (
        herb_semantic is not None or disease_semantic is not None
    ):
        raise ASTRARunnerError(
            "Ethnobotany semantic arrays are fixed inside --embedding-dir; "
            "TCM-suite alone accepts explicit semantic paths"
        )
    return {
        "compound": embedding_dir / _FEATURE_FILENAMES["compound"],
        "protein": embedding_dir / _FEATURE_FILENAMES["protein"],
        "herb": (
            herb_semantic.resolve()
            if herb_semantic is not None
            else embedding_dir / _FEATURE_FILENAMES["herb"]
        ),
        "disease": (
            disease_semantic.resolve()
            if disease_semantic is not None
            else embedding_dir / _FEATURE_FILENAMES["disease"]
        ),
    }


def _legacy_ethnobotany_semantic_paths(
    legacy_semantic_dir: Path,
    protocol: BenchmarkProtocol,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Load historical semantic arrays for a clearly ineligible diagnostic.

    The historical Ethnobotany herb/disease arrays have no trustworthy row-ID
    sidecars.  They must therefore never enter the feature-bundle contract or
    a formal run.  This narrow helper exists only to quantify the effect of
    switching semantic conditions while retaining the repaired biological
    features and frozen protocol.  It deliberately records that the legacy
    rows are consumed under the historical model-index assumption.
    """

    paths = {
        "herb": (
            legacy_semantic_dir / _FEATURE_FILENAMES["herb"]
        ).resolve(strict=False),
        "disease": (
            legacy_semantic_dir / _FEATURE_FILENAMES["disease"]
        ).resolve(strict=False),
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for entity, path in paths.items():
        expected_rows = (
            protocol.n_herbs if entity == "herb" else protocol.n_diseases
        )
        details = _matrix_diagnostics(
            path,
            expected_rows=expected_rows,
            expected_columns=64,
        )
        if bool(details["degenerate"]):
            raise ASTRARunnerError(
                f"legacy Ethnobotany {entity} semantic matrix is degenerate"
            )
        details.update(
            {
                "semantic_condition": "legacy_unverified_historical_matrix",
                "semantic_identity_status": (
                    "unverified_assumed_historical_model_index_order"
                ),
                "formal_eligible": False,
                "warning": (
                    "No row-ID sidecar exists; rows are loaded only under the "
                    "historical model-index-order assumption."
                ),
            }
        )
        diagnostics[entity] = details
    return paths, diagnostics


def _configure_runtime(seed: int, device: str, output_root: Path) -> None:
    os.environ["ASTRA_SEED"] = str(seed)
    os.environ["CDRUG_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["ASTRA_OUTPUT_DIR"] = str(output_root)
    os.environ["CDRUG_OUTPUT_DIR"] = str(output_root)
    os.environ["ASTRA_LOG_DIR"] = str(output_root / "logs")
    os.environ["CDRUG_LOG_DIR"] = str(output_root / "logs")
    os.environ.pop("ASTRA_ALLOW_MISSING_SEMANTIC", None)
    normalised = device.strip().lower()
    if normalised == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif normalised == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif normalised != "auto":
        raise ASTRARunnerError("device must be auto, cpu, or cuda")
    random.seed(seed)
    np.random.seed(seed)


def _load_legacy_components(dataset: str) -> tuple[type, type]:
    """Import the selected legacy model only after runtime configuration."""

    legacy_root = activate_backend(dataset)
    candidate_names = tuple(
        name
        for name in sys.modules
        if name in {"ASTRA", "HGHDA", "rating", "base", "util"}
        or name.startswith("base.")
        or name.startswith("util.")
    )
    for name in candidate_names:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is None or legacy_root not in Path(origin).resolve().parents:
            sys.modules.pop(name, None)
    try:
        model_class = getattr(importlib.import_module("ASTRA"), "ASTRA")
        config_class = getattr(importlib.import_module("util.config"), "ModelConf")
    except (ImportError, AttributeError) as error:
        raise ASTRARunnerError(
            f"cannot import legacy ASTRA implementation from {legacy_root}: {error}"
        ) from error
    for label, value in (("model", model_class), ("configuration", config_class)):
        module = sys.modules.get(value.__module__)
        origin = getattr(module, "__file__", None)
        if origin is None or legacy_root not in Path(origin).resolve().parents:
            raise ASTRARunnerError(
                f"legacy ASTRA {label} resolved outside {legacy_root}: {origin}"
            )
    return model_class, config_class


def _runtime_device_diagnostics(device: str) -> dict[str, Any]:
    """Record TensorFlow's observed device and reject a silent CUDA fallback."""

    try:
        import tensorflow.compat.v1 as tf
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise ASTRARunnerError("TensorFlow is unavailable") from error
    cuda_available = bool(tf.test.is_gpu_available(cuda_only=True))
    gpu_device_name = str(tf.test.gpu_device_name() or "")
    if device.strip().lower() == "cuda" and not cuda_available:
        raise ASTRARunnerError(
            "--device cuda was requested but TensorFlow found no usable CUDA GPU"
        )
    return {
        "tensorflow_cuda_available": cuda_available,
        "tensorflow_gpu_device_name": gpu_device_name or None,
    }


def _configure_model(
    dataset: str,
    config_class: type,
    relation_paths: Mapping[str, Path],
    feature_paths: Mapping[str, Path],
    embedding_dir: Path,
    output_root: Path,
    *,
    text_fusion_alpha: float = 1.0,
    feature_mask_paths: Mapping[str, Path] | None = None,
    smoke: bool,
) -> tuple[object, dict[str, Any]]:
    defaults = _HYPERPARAMETERS[dataset]
    config = config_class(str(defaults["config"]))
    epochs = 1 if smoke else int(defaults["epochs"])
    values = {
        "model.name": "ASTRA",
        "datapath": str(relation_paths["H_D"]),
        "ratings.setup": "-columns 0 1 2",
        "evaluation.setup": "-cv 5",
        "num.factors": str(defaults["embedding_dimension"]),
        "num.max.epoch": str(epochs),
        "batch_size": str(defaults["batch_size"]),
        "learnRate": f'-init {defaults["learning_rate"]} -max 1',
        "reg.lambda": "-u 0.001 -i 0.001 -b 0.2 -s 0.2",
        "output.setup": f'on -dir "{output_root}"',
        "data.dir": str(relation_paths["H_D"].parent),
        "embedding.dir": str(embedding_dir),
        "hc.path": str(relation_paths["H_C"]),
        "cp.path": str(relation_paths["C_P"]),
        "pd.path": str(relation_paths["P_D"]),
        "node.hc.path": str(relation_paths["H_C"]),
        "node.pd.path": str(relation_paths["P_D"]),
        "compound.embedding": str(feature_paths["compound"]),
        "protein.embedding": str(feature_paths["protein"]),
        "herb.text.embedding": str(feature_paths["herb"]),
        "disease.text.embedding": str(feature_paths["disease"]),
        "text.fusion.alpha": format(text_fusion_alpha, ".17g"),
    }
    if dataset == "ethnobotany" and feature_mask_paths:
        protein_mask = feature_mask_paths.get("protein")
        if protein_mask is not None:
            values["protein.embedding.missing_mask"] = str(protein_mask)
    config.config.update(values)
    recorded = {
        "epochs": epochs,
        "formal_epochs": int(defaults["epochs"]),
        "batch_size": int(defaults["batch_size"]),
        "learning_rate": float(defaults["learning_rate"]),
        "embedding_dimension": int(defaults["embedding_dimension"]),
        "reg_u": float(defaults["reg_u"]),
        "text_fusion_alpha": float(text_fusion_alpha),
        "smoke_override": bool(smoke),
    }
    return config, recorded


def _assert_unused_evaluation_rows(
    model: object, fold_rows: FoldRows, training_rows: Sequence[Sequence[object]]
) -> None:
    if model.data.testData:
        raise ASTRARunnerError("validation/test rows were unexpectedly passed to the model")
    if len(model.data.trainingData) != len(training_rows):
        raise ASTRARunnerError("legacy model changed the supplied training-row count")
    training_pairs = {
        (str(row[0]), str(row[1])) for row in model.data.trainingData
    }
    for split_name, pairs in (
        ("validation", fold_rows.validation_pairs),
        ("test", fold_rows.test_pairs),
    ):
        raw_pairs = set(
            _raw_pair_rows(model._benchmark_protocol, pairs, "herb", "disease")
        )
        overlap = training_pairs.intersection(raw_pairs)
        if overlap:
            raise ASTRARunnerError(
                f"{len(overlap)} {split_name} H-D pairs leaked into training rows"
            )


def _probe_astra_feature_dependencies(
    model: object,
    tensor_attributes: Mapping[str, str],
    *,
    expected_active: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, object]]:
    """Prove that inputs have exactly the dependencies implied by the alpha."""

    try:
        import tensorflow.compat.v1 as tf
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise ASTRARunnerError(
            "TensorFlow is unavailable for the feature dependency probe"
        ) from error
    outputs = {
        "compound": getattr(model, "final_hcedge", None),
        "herb": getattr(model, "final_hcedge", None),
        "protein": getattr(model, "final_pdedge", None),
        "disease": getattr(model, "final_pdedge", None),
    }
    session = getattr(model, "sess", None)
    diagnostics: dict[str, dict[str, object]] = {}
    for entity, output in outputs.items():
        should_be_active = bool(
            True if expected_active is None else expected_active.get(entity, True)
        )
        variable = getattr(model, tensor_attributes[entity], None)
        if output is None or variable is None:
            raise ASTRARunnerError(
                f"legacy {entity} feature dependency endpoints are unavailable"
            )
        gradient = tf.gradients(tf.reduce_sum(output), variable)[0]
        if gradient is None:
            if should_be_active:
                raise ASTRARunnerError(
                    f"legacy {entity} feature variable is disconnected from final prediction state"
                )
            norm = 0.0
        else:
            values = np.asarray(session.run(gradient), dtype=np.float64)
            norm = float(np.linalg.norm(values))
        if not np.isfinite(norm):
            raise ASTRARunnerError(
                f"legacy {entity} feature dependency gradient is non-finite"
            )
        active = bool(norm > 0.0)
        if active != should_be_active:
            expectation = "nonzero" if should_be_active else "zero"
            raise ASTRARunnerError(
                f"legacy {entity} feature dependency gradient must be {expectation}"
            )
        diagnostics[entity] = {
            "connected": active,
            "expected_active": should_be_active,
            "gradient_l2_norm": norm,
        }
    return diagnostics


def _initialize_model_variables(model: object) -> tuple[str, ...]:
    """Initialize graph state before executing fail-closed feature probes.

    A blanket TensorFlow initializer would reset variables that a caller may
    already have trained. The model-specific method is preferred because it
    also ensures the HGHDA optimizer graph exists before initialization.
    """

    initializer = getattr(model, "initialize_uninitialized_variables", None)
    if callable(initializer):
        try:
            return tuple(initializer())
        except RuntimeError as error:
            raise ASTRARunnerError(str(error)) from error
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise ASTRARunnerError(
            "TensorFlow is unavailable for model-variable initialization"
        ) from error
    session = getattr(model, "sess", None)
    if session is None:
        raise ASTRARunnerError("legacy model did not expose its TensorFlow session")
    try:
        return initialize_uninitialized_variables(session, tf)
    except RuntimeError as error:
        raise ASTRARunnerError(str(error)) from error


def _assert_canonical_model_mappings(
    model: object, protocol: BenchmarkProtocol
) -> None:
    for entity in ("herb", "compound", "protein", "disease"):
        expected_mapping = {
            str(raw_id): index
            for index, raw_id in enumerate(protocol.entity_ids[entity])
        }
        if getattr(model.data, entity, None) != expected_mapping:
            raise ASTRARunnerError(
                f"legacy {entity} mapping is not in canonical protocol row order"
            )


def _assert_ethnobotany_feature_binding(
    model: object,
    protocol: BenchmarkProtocol,
    feature_paths: Mapping[str, Path],
    feature_mask_paths: Mapping[str, Path] | None = None,
    *,
    clip_values: bool = True,
) -> dict[str, Any]:
    """Fail closed unless mappings, rows, transforms, and paths agree.

    Ethnobotany inputs can arrive in arbitrary sidecar-declared row order.
    TCM-suite inputs are independently bound to the strict protocol mapping.
    In either case, the runner first materialises canonical-order matrices and
    rebinds every legacy entity dictionary to that same order. This assertion
    closes the last link in the chain by confirming that the legacy loader
    consumed those exact four materialised paths and TensorFlow values.
    """

    counts = {
        "herb": protocol.n_herbs,
        "disease": protocol.n_diseases,
        "compound": protocol.n_compounds,
        "protein": protocol.n_proteins,
    }
    tensor_attributes = {
        "herb": "herb_text_embeddings",
        "disease": "disease_text_embeddings",
        "compound": "compound_embeddings",
        "protein": "protein_embeddings",
    }
    loaded_paths = getattr(model, "loaded_feature_paths", None)
    if not isinstance(loaded_paths, Mapping):
        raise ASTRARunnerError(
            "legacy loader did not report its loaded feature paths"
        )

    _assert_canonical_model_mappings(model, protocol)
    for entity, expected_path in feature_paths.items():
        observed_path = loaded_paths.get(entity)
        if observed_path is None or Path(str(observed_path)).resolve() != expected_path.resolve():
            raise ASTRARunnerError(
                f"legacy {entity} feature path does not match the aligned runner path"
            )

        value = getattr(model, tensor_attributes[entity], None)
        shape = getattr(value, "shape", None)
        if hasattr(shape, "as_list"):
            shape = shape.as_list()
        if shape is None or len(shape) != 2 or int(shape[0]) != counts[entity]:
            raise ASTRARunnerError(
                f"legacy {entity} feature rows do not match the canonical mapping"
            )

    protein_mask = np.zeros(protocol.n_proteins, dtype=np.bool_)
    if feature_mask_paths:
        expected_mask_path = feature_mask_paths.get("protein")
        if expected_mask_path is None:
            raise ASTRARunnerError(
                "audited Ethnobotany bundle did not supply a protein missing mask"
            )
        loaded_masks = getattr(model, "loaded_feature_masks", None)
        if not isinstance(loaded_masks, Mapping):
            raise ASTRARunnerError(
                "Ethnobotany legacy loader did not report its loaded feature masks"
            )
        observed_mask_path = loaded_masks.get("protein")
        if observed_mask_path is None or Path(str(observed_mask_path)).resolve() != expected_mask_path.resolve():
            raise ASTRARunnerError(
                "legacy protein missing-mask path does not match the audited bundle"
            )
        mask = np.load(expected_mask_path, allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != (protocol.n_proteins,):
            raise ASTRARunnerError(
                "legacy protein missing mask has a wrong dtype/shape"
            )
        protein_mask = np.asarray(mask, dtype=np.bool_)

    expected_loaded: dict[str, np.ndarray] = {}
    for entity in ("compound", "protein"):
        disk = np.asarray(np.loadtxt(feature_paths[entity]))
        mask = protein_mask if entity == "protein" else None
        if mask is not None and mask.any() and not np.array_equal(
            disk[mask], np.zeros((int(mask.sum()), disk.shape[1]), dtype=disk.dtype)
        ):
            raise ASTRARunnerError(
                "audited protein rows are nonzero before legacy preprocessing"
            )
        transformed = (disk - disk.mean(axis=0)) / (disk.std(axis=0) + 1e-8)
        if clip_values:
            transformed = np.clip(transformed, -10, 10)
        if mask is not None and mask.any():
            transformed[mask] = 0
        expected_loaded[entity] = transformed
    expected_loaded["herb"] = np.asarray(
        np.load(feature_paths["herb"], allow_pickle=False)
    )
    expected_loaded["disease"] = np.asarray(
        np.load(feature_paths["disease"], allow_pickle=False)
    )
    if clip_values:
        expected_loaded["herb"] = np.clip(expected_loaded["herb"], -10, 10)
        expected_loaded["disease"] = np.clip(expected_loaded["disease"], -10, 10)
    loaded_matrices = getattr(model, "loaded_feature_matrices", None)
    if not isinstance(loaded_matrices, Mapping) or set(loaded_matrices) != set(
        expected_loaded
    ):
        raise ASTRARunnerError(
            "legacy loader did not expose all four transformed arrays"
        )
    session = getattr(model, "sess", None)
    if session is None:
        raise ASTRARunnerError("legacy loader did not expose its session")
    for entity, expected in expected_loaded.items():
        entity_mask = protein_mask if entity == "protein" else np.zeros(
            expected.shape[0], dtype=np.bool_
        )
        audit_values = np.asarray(expected)[~entity_mask]
        if (
            audit_values.shape[0] < 2
            or np.unique(audit_values, axis=0).shape[0] < 2
            or np.count_nonzero(np.ptp(audit_values, axis=0) > 0.0) == 0
        ):
            raise ASTRARunnerError(
                f"legacy {entity} post-transform feature matrix is degenerate"
            )
        observed = np.asarray(loaded_matrices[entity])
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ASTRARunnerError(
                f"legacy {entity} post-transform array differs from the audited input"
            )
        variable = getattr(model, tensor_attributes[entity], None)
        initial_value = getattr(variable, "initial_value", None)
        if initial_value is None:
            raise ASTRARunnerError(
                f"legacy {entity} TensorFlow initialization is unavailable"
            )
        tensor_value = np.asarray(session.run(initial_value), dtype=np.float32)
        expected_tensor = np.asarray(expected, dtype=np.float32)
        if tensor_value.shape != expected_tensor.shape or not np.array_equal(
            tensor_value, expected_tensor
        ):
            raise ASTRARunnerError(
                f"legacy {entity} TensorFlow initialization differs from its "
                "post-transform audit array"
            )
        expected_loaded[entity] = expected_tensor
    if protein_mask.any() and not np.array_equal(
        np.asarray(loaded_matrices["protein"])[protein_mask],
        np.zeros(
            (int(protein_mask.sum()), 64),
            dtype=np.asarray(loaded_matrices["protein"]).dtype,
        ),
    ):
        raise ASTRARunnerError(
            "protein audited missing rows became nonzero during legacy loading"
        )
    text_fusion_alpha = float(getattr(model, "text_fusion_alpha", 1.0))
    semantic_branch_active = bool(text_fusion_alpha > 0.0)
    dependencies = _probe_astra_feature_dependencies(
        model,
        tensor_attributes,
        expected_active={
            "compound": True,
            "protein": True,
            "herb": semantic_branch_active,
            "disease": semantic_branch_active,
        },
    )
    return {
        entity: {
            "post_transform_array_sha256": _array_sha256(expected_loaded[entity]),
            "tensorflow_initial_array_sha256": _array_sha256(
                np.asarray(
                    session.run(
                        getattr(model, tensor_attributes[entity]).initial_value
                    ),
                    dtype=np.float32,
                )
            ),
            **dependencies[entity],
        }
        for entity in ("herb", "compound", "protein", "disease")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("ethnobotany", "ethno", "tcm-suite", "tcm") , required=True
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--text-fusion-alpha",
        type=_parse_unit_float,
        default=1.0,
        help=(
            "Fixed interpolation between structural and text-enhanced final "
            "branches. Zero is structural-only; one matches the previous "
            "text-enhanced implementation."
        ),
    )
    parser.add_argument(
        "--alpha-selection-manifest",
        type=Path,
        help=(
            "Validation-only alpha decision manifest. For a selected formal "
            "run, the manifest must name this fold and --text-fusion-alpha."
        ),
    )
    parser.add_argument(
        "--herb-semantic",
        type=Path,
        help="TCM-suite herb semantic .npy; defaults to --embedding-dir filename.",
    )
    parser.add_argument(
        "--disease-semantic",
        type=Path,
        help="TCM-suite disease semantic .npy; defaults to --embedding-dir filename.",
    )
    parser.add_argument(
        "--legacy-ethnobotany-semantic-dir",
        type=Path,
        help=(
            "Historical herb/disease semantic matrices for an Ethnobotany "
            "diagnostic only. Requires --diagnostic and can never produce a "
            "formal result because those matrices lack verified row IDs."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--smoke",
        action="store_true",
        help="Use a balanced training subset and one epoch; never a formal result.",
    )
    execution_mode.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Use all frozen rows and the complete epoch budget, but force every "
            "artifact to remain diagnostic-only and formally ineligible."
        ),
    )
    execution_mode.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train and score only the frozen inner-validation rows for alpha "
            "selection. No outer-test scores or predictions are materialised."
        ),
    )
    return parser


def _run_locked(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _normalise_dataset(args.dataset)
    smoke = bool(getattr(args, "smoke", False))
    diagnostic = bool(getattr(args, "diagnostic", False))
    validation_only = bool(getattr(args, "validation_only", False))
    if sum((smoke, diagnostic, validation_only)) > 1:
        raise ASTRARunnerError(
            "--smoke, --diagnostic, and --validation-only are mutually exclusive"
        )
    diagnostic_context = (
        diagnostic_execution_context(args.device) if diagnostic else None
    )
    diagnostic_reasons = (
        list(diagnostic_context["reasons"]) if diagnostic_context else []
    )
    protocol_manifest = args.protocol_manifest.resolve()
    data_dir = args.data_dir.resolve()
    embedding_dir = args.embedding_dir.resolve()
    output_root = args.output_root.resolve()
    text_fusion_alpha = _parse_unit_float(args.text_fusion_alpha)
    alpha_selection_manifest = getattr(args, "alpha_selection_manifest", None)
    if dataset != "ethnobotany" and not math.isclose(
        text_fusion_alpha, 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ASTRARunnerError(
            "--text-fusion-alpha is currently implemented only for Ethnobotany"
        )
    if validation_only and alpha_selection_manifest is not None:
        raise ASTRARunnerError(
            "--alpha-selection-manifest is valid only after validation selection"
        )
    if alpha_selection_manifest is not None and (smoke or diagnostic):
        raise ASTRARunnerError(
            "--alpha-selection-manifest cannot be combined with smoke/diagnostic mode"
        )
    if (
        dataset == "ethnobotany"
        and not smoke
        and not diagnostic
        and not validation_only
        and not math.isclose(text_fusion_alpha, 1.0, rel_tol=0.0, abs_tol=1e-15)
        and alpha_selection_manifest is None
    ):
        raise ASTRARunnerError(
            "a formal Ethnobotany alpha other than 1.0 requires a validated "
            "--alpha-selection-manifest"
        )
    if dataset != "ethnobotany" and alpha_selection_manifest is not None:
        raise ASTRARunnerError(
            "--alpha-selection-manifest is currently restricted to Ethnobotany"
        )
    legacy_semantic_dir = getattr(args, "legacy_ethnobotany_semantic_dir", None)
    if legacy_semantic_dir is not None:
        legacy_semantic_dir = legacy_semantic_dir.resolve()
        if dataset != "ethnobotany":
            raise ASTRARunnerError(
                "--legacy-ethnobotany-semantic-dir is valid only for Ethnobotany"
            )
        if not diagnostic:
            raise ASTRARunnerError(
                "legacy Ethnobotany semantic matrices require --diagnostic"
            )
        if not legacy_semantic_dir.is_dir():
            raise ASTRARunnerError(
                "legacy Ethnobotany semantic directory does not exist: "
                f"{legacy_semantic_dir}"
            )
    if args.seed < 0:
        raise ASTRARunnerError("seed must be non-negative")
    if not data_dir.is_dir():
        raise ASTRARunnerError(f"data directory does not exist: {data_dir}")
    if not embedding_dir.is_dir():
        raise ASTRARunnerError(f"embedding directory does not exist: {embedding_dir}")

    protocol = load_protocol_manifest(protocol_manifest)
    if protocol.dataset != dataset:
        raise ASTRARunnerError(
            f"CLI dataset {dataset} disagrees with manifest dataset {protocol.dataset}"
        )
    if protocol.seed != args.seed:
        raise ASTRARunnerError(
            f"CLI seed {args.seed} disagrees with manifest seed {protocol.seed}"
        )
    alpha_selection = None
    if alpha_selection_manifest is not None:
        alpha_selection = _validated_alpha_selection(
            alpha_selection_manifest,
            dataset=dataset,
            seed=int(args.seed),
            protocol_manifest=protocol_manifest,
            embedding_dir=embedding_dir,
            fold=int(args.fold),
            alpha=text_fusion_alpha,
        )
    initial_input_manifest = protocol.input_manifest()
    frozen_input_lock_sha256 = _json_sha256(initial_input_manifest)
    fold_rows = protocol.fold_rows(args.fold)

    result_dir = output_root / "results"
    protected_outputs = [
        result_dir / "metrics.json",
        result_dir / "validation_predictions.npz",
        output_root / "protocol.json",
    ]
    if not validation_only:
        protected_outputs.append(result_dir / "test_predictions.npz")
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise ASTRARunnerError(
            "refusing to overwrite an existing run artifact: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)

    source_inputs = _validate_manifest_sources(protocol, data_dir, embedding_dir)
    feature_paths = _feature_paths(
        dataset,
        embedding_dir,
        args.herb_semantic,
        args.disease_semantic,
    )
    if dataset == "ethnobotany":
        feature_mask_paths, feature_bundle_diagnostics = (
            _validate_ethnobotany_feature_bundle(
                protocol,
                embedding_dir,
                feature_paths,
                smoke=smoke,
            )
        )
    else:
        feature_mask_paths = {}
        feature_bundle_diagnostics = _validate_tcm_feature_bundle(
            protocol,
            embedding_dir,
            feature_paths,
            smoke=smoke,
        )
    feature_paths, feature_diagnostics = _validate_and_align_feature_paths(
        dataset,
        protocol,
        feature_paths,
        output_root,
    )
    if legacy_semantic_dir is not None:
        legacy_paths, legacy_diagnostics = _legacy_ethnobotany_semantic_paths(
            legacy_semantic_dir,
            protocol,
        )
        feature_paths.update(legacy_paths)
        feature_diagnostics.update(legacy_diagnostics)
        feature_bundle_diagnostics = {
            **feature_bundle_diagnostics,
            "formal_eligible": False,
            "legacy_semantic_override": {
                "enabled": True,
                "directory": str(legacy_semantic_dir),
                "identity_status": (
                    "unverified_assumed_historical_model_index_order"
                ),
                "warning": (
                    "Historical semantic arrays are a diagnostic condition only; "
                    "they do not satisfy the ID-bound feature contract."
                ),
            },
        }
        diagnostic_reasons.append(
            "legacy Ethnobotany semantic matrices use unverified historical "
            "model-index order"
        )
    degenerate_features = _assert_formal_feature_integrity(
        feature_diagnostics,
        smoke=smoke,
    )

    relation_paths = materialise_protocol_relations(
        protocol, output_root / "protocol_relations"
    )
    relation_view = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in relation_paths.items()
    }

    if smoke:
        train_pairs, train_labels = _smoke_training_rows(fold_rows)
        validation_pairs, validation_labels, validation_row_id = (
            _smoke_evaluation_rows(
                fold_rows.validation_pairs,
                fold_rows.validation_labels,
                fold_rows.validation_row_indices,
            )
        )
        test_pairs, test_labels, test_row_id = _smoke_evaluation_rows(
            fold_rows.test_pairs,
            fold_rows.test_labels,
            fold_rows.test_row_indices,
        )
    else:
        train_pairs, train_labels = fold_rows.train_pairs, fold_rows.train_labels
        validation_pairs = fold_rows.validation_pairs
        validation_labels = fold_rows.validation_labels
        validation_row_id = fold_rows.validation_row_indices
        test_pairs = fold_rows.test_pairs
        test_labels = fold_rows.test_labels
        test_row_id = fold_rows.test_row_indices
    training_rows = compact_pairs_to_training_rows(protocol, train_pairs, train_labels)

    _configure_runtime(args.seed, args.device, output_root)
    model_class, config_class = _load_legacy_components(dataset)
    runtime_device = _runtime_device_diagnostics(args.device)
    config, hyperparameters = _configure_model(
        dataset,
        config_class,
        relation_paths,
        feature_paths,
        embedding_dir,
        output_root,
        text_fusion_alpha=text_fusion_alpha,
        feature_mask_paths=feature_mask_paths,
        smoke=smoke,
    )
    hyperparameters["diagnostic_mode"] = diagnostic

    started = time.time()
    model = model_class(config, training_rows, [], f"[{args.fold}]")
    observed_model_alpha = getattr(model, "text_fusion_alpha", None)
    if dataset == "ethnobotany" and (
        observed_model_alpha is None
        or not math.isclose(
            float(observed_model_alpha),
            text_fusion_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ASTRARunnerError(
            "legacy Ethnobotany model did not consume the requested text-fusion alpha"
        )
    model._benchmark_protocol = protocol
    _rebind_model_mappings(model, protocol)
    _assert_unused_evaluation_rows(model, fold_rows, training_rows)
    model.readConfiguration()
    model.initModel()
    feature_binding_diagnostics: dict[str, Any] = {}
    if dataset in {"ethnobotany", "tcm-suite"}:
        _initialize_model_variables(model)
        feature_binding_diagnostics = _assert_ethnobotany_feature_binding(
            model,
            protocol,
            feature_paths,
            feature_mask_paths,
            clip_values=dataset == "ethnobotany",
        )
    model.trainModel()
    if dataset in {"ethnobotany", "tcm-suite"}:
        _assert_canonical_model_mappings(model, protocol)
    runtime_seconds = time.time() - started

    expected_steps = int(hyperparameters["epochs"]) * int(
        math.ceil(len(training_rows) / int(hyperparameters["batch_size"]))
    )
    if expected_steps <= 0:
        raise ASTRARunnerError("ASTRA completed without a calculable optimizer step")
    validation_scores = score_compact_pairs(model, protocol, validation_pairs)

    validation_path = save_predictions(
        result_dir / "validation_predictions.npz",
        row_id=validation_row_id,
        pair=validation_pairs,
        label=validation_labels,
        score=validation_scores,
    )
    validate_prediction_coverage(
        validation_path,
        expected_row_id=validation_row_id,
        expected_pair=validation_pairs,
        expected_label=validation_labels,
    )
    test_path: Path | None = None
    if validation_only:
        metrics = evaluate_validation_scores(validation_labels, validation_scores)
    else:
        test_scores = score_compact_pairs(model, protocol, test_pairs)
        test_path = save_predictions(
            result_dir / "test_predictions.npz",
            row_id=test_row_id,
            pair=test_pairs,
            label=test_labels,
            score=test_scores,
        )
        validate_prediction_coverage(
            test_path,
            expected_row_id=test_row_id,
            expected_pair=test_pairs,
            expected_label=test_labels,
        )
        metrics = evaluate_validation_and_test(
            validation_labels,
            validation_scores,
            test_labels,
            test_scores,
        )
    execution_mode = (
        "smoke"
        if smoke
        else "diagnostic"
        if diagnostic
        else "alpha-selection-candidate"
        if validation_only
        else "formal-candidate"
    )
    formal_ineligibility_reasons = (
        (["smoke mode"] if smoke else [])
        + diagnostic_reasons
        + (["validation-only alpha selection candidate"] if validation_only else [])
        + (
            [f"degenerate feature matrices: {', '.join(degenerate_features)}"]
            if degenerate_features
            else []
        )
    )
    formal_eligible = bool(
        not smoke
        and not diagnostic
        and not validation_only
        and not degenerate_features
    )
    metrics.update(
        {
            "Method": "ASTRA",
            "Dataset": dataset,
            "Fold": int(args.fold),
            "Seed": int(args.seed),
            "Optimizer_Steps": expected_steps,
            "Smoke": smoke,
            "Diagnostic": diagnostic,
            "Validation_Only": validation_only,
            "Evaluation_Mode": (
                "validation-only-alpha-selection"
                if validation_only
                else "validation"
            ),
            "Execution_Mode": execution_mode,
            "Formal_Eligible": formal_eligible,
            "Formal_Ineligibility_Reasons": formal_ineligibility_reasons,
            "Feature_Integrity_Warnings": degenerate_features,
            "Text_Fusion_Alpha": text_fusion_alpha,
            "Alpha_Selected_On_Validation": bool(alpha_selection is not None),
            "Decision_Frozen_Before_Test": bool(alpha_selection is not None),
            "Test_Models_Evaluated": 0 if validation_only else 1,
            "Unselected_Test_Models_Evaluated": 0,
        }
    )
    _write_json(result_dir / "metrics.json", metrics)

    if dataset == "ethnobotany":
        model_sources = {
            "entrypoint": ETHNOBOTANY_BACKEND_ROOT / "ASTRA.py",
            "model": ETHNOBOTANY_BACKEND_ROOT / "HGHDA.py",
            "herb_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/herbRecommender.py",
            "disease_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/diseaseRecommender.py",
            "base_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/recommender.py",
            "rating": ETHNOBOTANY_BACKEND_ROOT / "rating.py",
            "feature_paths": ETHNOBOTANY_BACKEND_ROOT
            / "ethnobotany_feature_paths.py",
            "configuration_loader": ETHNOBOTANY_BACKEND_ROOT / "util/config.py",
            "configuration": ETHNOBOTANY_BACKEND_ROOT / "ethnobotany.conf",
        }
    else:
        model_sources = {
            "model": TCM_BACKEND_ROOT / "HGHDA.py",
            "herb_recommender": TCM_BACKEND_ROOT / "base/herbRecommender.py",
            "rating": TCM_BACKEND_ROOT / "rating.py",
            "configuration": TCM_BACKEND_ROOT / "ASTRA.conf",
        }
    provenance = {
        "schema": "astra-central-fair-run/v1",
        "method": "ASTRA",
        "dataset": dataset,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "smoke": smoke,
        "diagnostic": diagnostic,
        "validation_only": validation_only,
        "execution_mode": execution_mode,
        "diagnostic_context": diagnostic_context,
        "formal_eligible": formal_eligible,
        "formal_ineligibility_reasons": formal_ineligibility_reasons,
        "protocol_manifest": {
            "path": str(protocol_manifest),
            "bytes": protocol_manifest.stat().st_size,
            "sha256": _sha256(protocol_manifest),
        },
        "protocol_input_manifest": initial_input_manifest,
        "frozen_input_lock_sha256": frozen_input_lock_sha256,
        "source_inputs_verified": source_inputs,
        "deduplicated_relation_view": relation_view,
        "feature_matrices": feature_diagnostics,
        "feature_bundle": feature_bundle_diagnostics,
        "feature_binding": feature_binding_diagnostics,
        "model_sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in model_sources.items()
        },
        "adapter_sources": {
            "astra_runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "metrics": {
                "path": str((SOURCE_ROOT / "astra/benchmark/metrics.py").resolve()),
                "sha256": _sha256(
                    (SOURCE_ROOT / "astra/benchmark/metrics.py").resolve()
                ),
            },
        },
        "hyperparameters": hyperparameters,
        "alpha_selection": alpha_selection,
        "decision_frozen_before_test": bool(alpha_selection is not None),
        "split_rows": {
            "train_full": int(len(fold_rows.train_pairs)),
            "train_used": int(len(training_rows)),
            "train_positive_graph_edges_full": int(len(fold_rows.train_positive_pairs)),
            "validation_full": int(len(fold_rows.validation_pairs)),
            "validation_used": int(len(validation_pairs)),
            "test_full": int(len(fold_rows.test_pairs)),
            "test_used": 0 if validation_only else int(len(test_pairs)),
        },
        "split_array_sha256": {
            "train_pairs_used": _array_sha256(np.asarray(train_pairs)),
            "train_labels_used": _array_sha256(np.asarray(train_labels)),
            "train_positive_pairs_full": _array_sha256(fold_rows.train_positive_pairs),
            "validation_pairs_used": _array_sha256(validation_pairs),
            "test_pairs_used": (
                None if validation_only else _array_sha256(test_pairs)
            ),
        },
        "optimizer_steps": expected_steps,
        "runtime_seconds": runtime_seconds,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device_policy": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            **runtime_device,
        },
        "evaluation": {
            "validation_only": validation_only,
            "threshold_source": "validation maximum F1 over 100 thresholds",
            "test_threshold_search": False,
            "test_models_evaluated": 0 if validation_only else 1,
            "unselected_test_models_evaluated": 0,
            "decision_frozen_before_test": bool(alpha_selection is not None),
            "auprc_definition": (
                "trapezoidal area: sklearn.metrics.auc(recall, precision) "
                "after precision_recall_curve"
            ),
            "average_precision_definition": "sklearn.metrics.average_precision_score",
            "validation_predictions": str(validation_path),
            "test_predictions": None if test_path is None else str(test_path),
        },
    }
    if protocol.input_manifest() != initial_input_manifest:
        raise ASTRARunnerError("central benchmark inputs changed during the ASTRA run")
    _write_json(output_root / "protocol.json", provenance)

    session = getattr(model, "sess", None)
    if session is not None:
        session.close()
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fold while holding an advisory lock on its artifact directory."""

    output_root = Path(args.output_root).resolve()
    with _exclusive_run_lock(output_root):
        return _run_locked(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run(args)
    except Exception as error:
        print(f"ASTRA fair runner failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
