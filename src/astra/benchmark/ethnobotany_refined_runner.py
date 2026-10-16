#!/usr/bin/env python3
"""Train validation-only ASTRARefined candidates on one frozen outer fold.

This runner is deliberately separate from :mod:`astra_runner`.  It derives
only the inner-training and inner-validation rows, trains one uninterrupted
trajectory, and materialises the locked checkpoint grid.  No outer-fold
evaluation rows are requested or passed to the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .astra_runner import (
    ASTRARunnerError,
    _assert_ethnobotany_feature_binding,
    _assert_formal_feature_integrity,
    _configure_runtime,
    _initialize_model_variables,
    _materialise_aligned_feature,
    _rebind_model_mappings,
    _runtime_device_diagnostics,
    _validate_and_align_feature_paths,
    _validate_ethnobotany_feature_bundle,
    _validate_manifest_sources,
    _write_lines,
    compact_pairs_to_training_rows,
)
from .ethnobotany_semantic_features import (
    REDUCTION_MANIFEST_SCHEMA,
    SemanticFeatureError,
    validate_semantic_deep_audit_receipt,
    validate_shared_semantic_reduction,
)
from .ethnobotany_semantic_controls import ZERO_PAD_CONTROL_MANIFEST_SCHEMA
from .ethnobotany_refined_feature_receipt import (
    FeatureDeepAuditReceiptError,
    validate_feature_deep_audit_receipt,
)
from .features import FeatureContractError, load_aligned_feature_matrix
from .metrics import (
    evaluate_validation_scores,
    save_predictions,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .ethnobotany_refined_selection import CANDIDATE_SCHEMA, sha256
from .backend_paths import ETHNOBOTANY_BACKEND_ROOT, activate_backend


CONFIG_SCHEMA = "astra-ethnobotany-refined-candidate-config/v1"
CHECKPOINT_EPOCHS = (30, 60, 100, 200)
SOURCE_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ETHNOBOTANY_BACKEND_ROOT
BASE_FEATURE_FILENAMES = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
    "herb": "herb_semantic_embeddings_64d.npy",
    "disease": "disease_semantic_embeddings_64d.npy",
}


class RefinedCandidateRunnerError(RuntimeError):
    """Raised when a validation-only candidate cannot be audited."""


@dataclass(frozen=True)
class TrainingValidationRows:
    """The only supervised rows made available to candidate training."""

    fold: int
    train_pairs: np.ndarray
    train_labels: np.ndarray
    train_row_indices: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray
    validation_row_indices: np.ndarray


@dataclass(frozen=True)
class PreparedFeatures:
    paths: Mapping[str, Path]
    mask_paths: Mapping[str, Path]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    snapshots: Mapping[int, Mapping[str, Any]]
    diagnostics: Mapping[str, Any]


def _freeze(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _validation_size(count: int, fraction: float) -> int:
    if count < 2:
        raise RefinedCandidateRunnerError(
            "each outer-training class needs at least two rows"
        )
    size = int(math.floor(count * fraction))
    return min(max(size, 1), count - 1)


def derive_training_validation_rows(
    protocol: BenchmarkProtocol, fold: int
) -> TrainingValidationRows:
    """Reproduce the protocol's nested split without constructing outer rows."""

    fold = int(fold)
    if not 1 <= fold <= int(protocol.n_folds):
        raise RefinedCandidateRunnerError(
            f"fold must be in 1..{protocol.n_folds}, got {fold}"
        )
    positives = np.asarray(protocol.positive_pairs, dtype=np.int64)
    negatives = np.asarray(protocol.negative_pairs, dtype=np.int64)
    positive_count = len(positives)
    negative_count = len(negatives)
    positive_rows = np.arange(positive_count, dtype=np.int64)
    negative_local = np.arange(negative_count, dtype=np.int64)
    negative_rows = positive_count + negative_local

    positive_outer = positive_rows[positive_rows % protocol.n_folds != fold - 1]
    negative_outer = negative_rows[negative_local % protocol.n_folds != fold - 1]
    positive_rng = np.random.default_rng(
        np.random.SeedSequence((protocol.seed, fold, 1))
    )
    negative_rng = np.random.default_rng(
        np.random.SeedSequence((protocol.seed, fold, 0))
    )
    positive_outer = positive_outer[positive_rng.permutation(len(positive_outer))]
    negative_outer = negative_outer[negative_rng.permutation(len(negative_outer))]
    positive_count_validation = _validation_size(
        len(positive_outer), protocol.validation_fraction
    )
    negative_count_validation = _validation_size(
        len(negative_outer), protocol.validation_fraction
    )
    validation_rows = np.concatenate(
        (
            positive_outer[:positive_count_validation],
            negative_outer[:negative_count_validation],
        )
    )
    train_rows = np.concatenate(
        (
            positive_outer[positive_count_validation:],
            negative_outer[negative_count_validation:],
        )
    )

    def pairs_for(rows: np.ndarray) -> np.ndarray:
        positive_mask = rows < positive_count
        output = np.empty((len(rows), 2), dtype=np.int64)
        output[positive_mask] = positives[rows[positive_mask]]
        output[~positive_mask] = negatives[rows[~positive_mask] - positive_count]
        return output

    def labels_for(rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows < positive_count, dtype=np.uint8)

    overlap = np.intersect1d(train_rows, validation_rows)
    if overlap.size:
        raise RefinedCandidateRunnerError("inner training/validation rows overlap")
    return TrainingValidationRows(
        fold=fold,
        train_pairs=_freeze(pairs_for(train_rows), np.dtype(np.int64)),
        train_labels=_freeze(labels_for(train_rows), np.dtype(np.uint8)),
        train_row_indices=_freeze(train_rows, np.dtype(np.int64)),
        validation_pairs=_freeze(
            pairs_for(validation_rows), np.dtype(np.int64)
        ),
        validation_labels=_freeze(
            labels_for(validation_rows), np.dtype(np.uint8)
        ),
        validation_row_indices=_freeze(
            validation_rows, np.dtype(np.int64)
        ),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RefinedCandidateRunnerError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RefinedCandidateRunnerError(f"JSON must contain an object: {path}")
    return payload


def validate_candidate_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen candidate identity, chain, and training budget."""

    if payload.get("schema") != CONFIG_SCHEMA:
        raise RefinedCandidateRunnerError(
            f"candidate config schema must be {CONFIG_SCHEMA!r}"
        )
    identity = payload.get("config")
    if not isinstance(identity, Mapping):
        raise RefinedCandidateRunnerError("candidate config identity is absent")
    required = {
        "id",
        "stage_id",
        "candidate_value",
        "chain_index",
        "incumbent_id",
        "base_config_id",
    }
    if set(identity) != required:
        raise RefinedCandidateRunnerError(
            "config identity fields must be exactly " + ", ".join(sorted(required))
        )
    config_id = identity.get("id")
    stage_id = identity.get("stage_id")
    base_id = identity.get("base_config_id")
    incumbent_id = identity.get("incumbent_id")
    chain_index = identity.get("chain_index")
    if any(not isinstance(value, str) or not value for value in (config_id, stage_id, base_id)):
        raise RefinedCandidateRunnerError(
            "config id, stage_id, and base_config_id must be non-empty strings"
        )
    if isinstance(chain_index, bool) or not isinstance(chain_index, int) or chain_index < 0:
        raise RefinedCandidateRunnerError("config chain_index must be a non-negative integer")
    if chain_index == 0:
        if incumbent_id is not None or base_id != config_id:
            raise RefinedCandidateRunnerError(
                "chain root requires incumbent_id=null and base_config_id=id"
            )
    elif not isinstance(incumbent_id, str) or not incumbent_id:
        raise RefinedCandidateRunnerError(
            "non-root candidate requires a non-empty incumbent_id"
        )

    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise RefinedCandidateRunnerError("candidate training settings are absent")
    checkpoint_epochs = training.get("checkpoint_epochs")
    if checkpoint_epochs != list(CHECKPOINT_EPOCHS):
        raise RefinedCandidateRunnerError(
            f"checkpoint_epochs must be exactly {list(CHECKPOINT_EPOCHS)}"
        )
    integer_fields = {"epochs": 200, "batch_size": None, "embedding_dimension": 64}
    normalized_training = dict(training)
    for name, exact in integer_fields.items():
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RefinedCandidateRunnerError(f"training.{name} must be a positive integer")
        if exact is not None and value != exact:
            raise RefinedCandidateRunnerError(f"training.{name} must equal {exact}")
    for name in ("learning_rate", "reg_u", "text_fusion_alpha"):
        try:
            value = float(training[name])
        except (KeyError, TypeError, ValueError) as error:
            raise RefinedCandidateRunnerError(
                f"training.{name} must be a finite number"
            ) from error
        if not math.isfinite(value):
            raise RefinedCandidateRunnerError(f"training.{name} must be finite")
        normalized_training[name] = value
    if normalized_training["learning_rate"] <= 0 or normalized_training["reg_u"] < 0:
        raise RefinedCandidateRunnerError("learning_rate must be positive and reg_u non-negative")
    if not 0 <= normalized_training["text_fusion_alpha"] <= 1:
        raise RefinedCandidateRunnerError("text_fusion_alpha must be in [0, 1]")
    model_options = payload.get("model_options", {})
    if not isinstance(model_options, Mapping):
        raise RefinedCandidateRunnerError("model_options must be an object")
    return {
        "schema": CONFIG_SCHEMA,
        "config": dict(identity),
        "training": normalized_training,
        "model_options": dict(model_options),
    }


def validate_semantic_reduction_manifest(
    manifest_path: Path | str,
    *,
    protocol: BenchmarkProtocol,
    deep_audit_receipt: Path | str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Audit a formal shared PCA64/PCA128 manifest and bind its row IDs."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    raw = _load_json(manifest_path)
    if raw.get("schema") not in {
        REDUCTION_MANIFEST_SCHEMA,
        ZERO_PAD_CONTROL_MANIFEST_SCHEMA,
    }:
        raise RefinedCandidateRunnerError("semantic reduction schema is invalid")
    dimension = raw.get("output_dimension")
    if dimension not in (64, 128):
        raise RefinedCandidateRunnerError(
            "semantic reduction output_dimension must be 64 or 128"
        )
    try:
        if deep_audit_receipt is None:
            audited = validate_shared_semantic_reduction(manifest_path)
        else:
            audited = validate_semantic_deep_audit_receipt(
                semantic_reduction_manifest=manifest_path,
                receipt_path=deep_audit_receipt,
            )
    except SemanticFeatureError as error:
        raise RefinedCandidateRunnerError(
            f"semantic reduction provenance failed: {error}"
        ) from error
    entities = raw.get("entities")
    if not isinstance(entities, Mapping):
        raise RefinedCandidateRunnerError("semantic reduction entities are absent")
    paths: dict[str, Path] = {}
    diagnostics: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        record = entities.get(entity)
        if not isinstance(record, Mapping):
            raise RefinedCandidateRunnerError(f"semantic {entity} record is absent")
        mapping = protocol.source_files.get(f"mapping/{entity}")
        if not isinstance(mapping, Mapping):
            raise RefinedCandidateRunnerError(
                f"protocol has no authoritative {entity} mapping digest"
            )
        if record.get("mapping_sha256") != mapping.get("sha256"):
            raise RefinedCandidateRunnerError(
                f"semantic {entity} mapping digest differs from the protocol"
            )
        matrix_path = Path(str(record.get("matrix", {}).get("path", ""))).resolve()
        if not matrix_path.is_file() or sha256(matrix_path) != record.get("matrix", {}).get("sha256"):
            raise RefinedCandidateRunnerError(f"semantic {entity} matrix hash is invalid")
        output_record = audited["outputs"][entity]
        if Path(output_record["matrix_path"]).resolve() != matrix_path:
            raise RefinedCandidateRunnerError(
                f"semantic {entity} audited output path differs from its manifest"
            )
        paths[entity] = matrix_path
        diagnostics[entity] = {
            "path": str(matrix_path),
            "sha256": sha256(matrix_path),
            "shape": list(np.asarray(output_record["matrix"]).shape),
            "mapping_sha256": str(mapping["sha256"]),
        }
    receipt_record = None
    if deep_audit_receipt is not None:
        receipt_path = Path(deep_audit_receipt).resolve(strict=True)
        receipt_record = {
            "path": str(receipt_path),
            "sha256": sha256(receipt_path),
        }
    return paths, {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "schema": str(raw["schema"]),
        "output_dimension": int(dimension),
        "formal_eligible": True,
        "deep_audit": True,
        "deep_audit_receipt": receipt_record,
        "controlled_experiment_only": bool(
            raw.get("controlled_experiment_only", False)
        ),
        "consumer_contract": raw.get("consumer_contract"),
        "entities": diagnostics,
    }


def _boolean_option(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RefinedCandidateRunnerError(f"{name} must be a boolean")


def _validate_semantic_model_contract(
    candidate: Mapping[str, Any], semantic: Mapping[str, Any]
) -> None:
    """Bind each selected semantic artifact to its intended model frontend."""

    options = candidate["model_options"]
    expected_dimension = int(semantic["output_dimension"])
    configured_dimension = int(options.get("astra.semantic.input_dim", 64))
    if configured_dimension != expected_dimension:
        raise RefinedCandidateRunnerError(
            "model semantic input dimension differs from the reduction manifest"
        )
    schema = semantic.get("schema")
    if schema == ZERO_PAD_CONTROL_MANIFEST_SCHEMA:
        if (
            configured_dimension != 128
            or str(options.get("astra.semantic.adapter", "")).strip().lower()
            != "linear"
            or not _boolean_option(
                options.get("astra.semantic.freeze_input", False),
                name="astra.semantic.freeze_input",
            )
        ):
            raise RefinedCandidateRunnerError(
                "zero-pad128 control requires frozen 128-dimensional semantics "
                "and a linear 128-to-64 adapter"
            )
    identity = candidate["config"]
    if identity.get("stage_id") == "S4b_semantic_retained_dimension":
        value = identity.get("candidate_value")
        expected_schema = {
            "shared_pca64_zeropad128_frozen_linear128to64": (
                ZERO_PAD_CONTROL_MANIFEST_SCHEMA
            ),
            "shared_pca128_frozen_linear128to64": REDUCTION_MANIFEST_SCHEMA,
        }.get(value)
        if expected_schema is None or schema != expected_schema:
            raise RefinedCandidateRunnerError(
                "S4b candidate value does not match its semantic manifest schema"
            )
        if (
            configured_dimension != 128
            or str(options.get("astra.semantic.adapter", "")).strip().lower()
            != "linear"
            or not _boolean_option(
                options.get("astra.semantic.freeze_input", False),
                name="astra.semantic.freeze_input",
            )
        ):
            raise RefinedCandidateRunnerError(
                "S4b requires the same frozen 128-to-64 linear semantic frontend"
            )


def prepare_candidate_features(
    *,
    protocol: BenchmarkProtocol,
    data_dir: Path,
    embedding_dir: Path,
    semantic_reduction_manifest: Path,
    output_root: Path,
    semantic_deep_audit_receipt: Path | None = None,
) -> PreparedFeatures:
    """Run the existing source/bundle audits and bind selected semantics."""

    source_inputs = _validate_manifest_sources(protocol, data_dir, embedding_dir)
    base_paths = {
        entity: embedding_dir / filename
        for entity, filename in BASE_FEATURE_FILENAMES.items()
    }
    mask_paths, base_bundle = _validate_ethnobotany_feature_bundle(
        protocol, embedding_dir, base_paths, smoke=False
    )
    semantic_paths, semantic_audit = validate_semantic_reduction_manifest(
        semantic_reduction_manifest,
        protocol=protocol,
        deep_audit_receipt=semantic_deep_audit_receipt,
    )
    aligned_paths, feature_diagnostics = _validate_and_align_feature_paths(
        "ethnobotany",
        protocol,
        {entity: base_paths[entity] for entity in ("compound", "protein")},
        output_root,
    )
    semantic_dimension = int(semantic_audit["output_dimension"])
    for entity, path in semantic_paths.items():
        mapping = protocol.source_files.get(f"mapping/{entity}")
        if not isinstance(mapping, Mapping):
            raise RefinedCandidateRunnerError(
                f"protocol has no authoritative {entity} mapping digest"
            )
        try:
            aligned = load_aligned_feature_matrix(
                path,
                entity=entity,
                canonical_ids=protocol.entity_ids[entity],
                mapping_sha256=str(mapping["sha256"]),
                expected_columns=semantic_dimension,
            )
        except FeatureContractError as error:
            raise RefinedCandidateRunnerError(str(error)) from error
        details = dict(aligned.diagnostics)
        if details["source_order_is_canonical"]:
            used_path = path
        else:
            used_path = _materialise_aligned_feature(
                path,
                aligned.matrix,
                output_root / "aligned_features",
            )
        details.update(
            {
                "used_path": str(used_path.resolve()),
                "used_bytes": used_path.stat().st_size,
                "used_sha256": sha256(used_path),
            }
        )
        aligned_paths[entity] = used_path
        feature_diagnostics[entity] = details
    _assert_formal_feature_integrity(feature_diagnostics, smoke=False)
    return PreparedFeatures(
        paths=aligned_paths,
        mask_paths=mask_paths,
        diagnostics={
            "protocol_sources": source_inputs,
            "base_bundle": base_bundle,
            "semantic_reduction": semantic_audit,
            "selected_features": feature_diagnostics,
        },
    )


def materialise_training_relations(
    protocol: BenchmarkProtocol,
    rows: TrainingValidationRows,
    output_dir: Path,
) -> dict[str, Path]:
    """Write legacy relations while limiting H--D to inner-training positives."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "H_D": output_dir / "H_D.txt",
        "H_C": output_dir / "H_C.txt",
        "C_P": output_dir / "C_P.txt",
        "P_D": output_dir / "P_D.txt",
    }

    def raw_pairs(pairs: np.ndarray, left: str, right: str) -> list[tuple[str, str]]:
        return [
            (
                protocol.compact_to_raw(left, int(pair[0])),
                protocol.compact_to_raw(right, int(pair[1])),
            )
            for pair in np.asarray(pairs, dtype=np.int64)
        ]

    positive_training = rows.train_pairs[rows.train_labels == 1]
    _write_lines(
        paths["H_D"],
        [(*pair, 1) for pair in raw_pairs(positive_training, "herb", "disease")],
    )
    _write_lines(
        paths["H_C"], raw_pairs(protocol.herb_compound_pairs, "herb", "compound")
    )
    _write_lines(
        paths["C_P"], raw_pairs(protocol.compound_protein_pairs, "compound", "protein")
    )
    _write_lines(
        paths["P_D"], raw_pairs(protocol.protein_disease_pairs, "protein", "disease")
    )
    return paths


def _load_refined_components() -> tuple[type, type]:
    activate_backend("ethnobotany")
    candidates = tuple(
        name
        for name in sys.modules
        if name in {"ASTRARefined", "rating", "base", "util", "ethnobotany_feature_paths"}
        or name.startswith("base.")
        or name.startswith("util.")
    )
    for name in candidates:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is None or LEGACY_ROOT not in Path(origin).resolve().parents:
            sys.modules.pop(name, None)
    try:
        model_class = getattr(importlib.import_module("ASTRARefined"), "ASTRARefined")
        config_class = getattr(importlib.import_module("util.config"), "ModelConf")
    except (ImportError, AttributeError) as error:
        raise RefinedCandidateRunnerError(
            f"cannot import ASTRARefined from {LEGACY_ROOT}: {error}"
        ) from error
    return model_class, config_class


def _configure_model(
    config_class: type,
    *,
    candidate: Mapping[str, Any],
    relation_paths: Mapping[str, Path],
    features: PreparedFeatures,
    embedding_dir: Path,
    output_root: Path,
) -> object:
    training = candidate["training"]
    config = config_class(str(LEGACY_ROOT / "ethnobotany.conf"))
    values: dict[str, str] = {
        "model.name": "ASTRARefined",
        "datapath": str(relation_paths["H_D"]),
        "ratings.setup": "-columns 0 1 2",
        "evaluation.setup": "-cv 5",
        "num.factors": str(training["embedding_dimension"]),
        "num.max.epoch": str(training["epochs"]),
        "batch_size": str(training["batch_size"]),
        "learnRate": f"-init {training['learning_rate']:.17g} -max 1",
        "reg.lambda": (
            f"-u {training['reg_u']:.17g} -i {training['reg_u']:.17g} "
            "-b 0.2 -s 0.2"
        ),
        "output.setup": f'on -dir "{output_root}"',
        "data.dir": str(relation_paths["H_D"].parent),
        "embedding.dir": str(embedding_dir),
        "hc.path": str(relation_paths["H_C"]),
        "cp.path": str(relation_paths["C_P"]),
        "pd.path": str(relation_paths["P_D"]),
        "node.hc.path": str(relation_paths["H_C"]),
        "node.pd.path": str(relation_paths["P_D"]),
        "compound.embedding": str(features.paths["compound"]),
        "protein.embedding": str(features.paths["protein"]),
        "herb.text.embedding": str(features.paths["herb"]),
        "disease.text.embedding": str(features.paths["disease"]),
        "text.fusion.alpha": f"{training['text_fusion_alpha']:.17g}",
        "astra.checkpoint.epochs": ",".join(map(str, CHECKPOINT_EPOCHS)),
    }
    protein_mask = features.mask_paths.get("protein")
    if protein_mask is not None:
        values["protein.embedding.missing_mask"] = str(protein_mask)
    values.update({str(key): str(value) for key, value in candidate["model_options"].items()})
    # Gradient telemetry is an audit observation, not a tunable candidate
    # factor, and is therefore enabled identically for every trajectory.
    values["astra.training.trace_gradients"] = "true"
    config.config.update(values)
    return config


def _score_embeddings(
    u_embedding: np.ndarray, i_embedding: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    u = np.asarray(u_embedding, dtype=np.float64)
    i = np.asarray(i_embedding, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if u.ndim != 2 or i.ndim != 2 or u.shape[1] != i.shape[1]:
        raise RefinedCandidateRunnerError("checkpoint u/i embeddings are incompatible")
    logits = np.einsum("ij,ij->i", u[pairs[:, 0]], i[pairs[:, 1]])
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    if not np.isfinite(scores).all():
        raise RefinedCandidateRunnerError("checkpoint emitted non-finite validation scores")
    return scores


def train_refined_candidate(
    *,
    protocol: BenchmarkProtocol,
    rows: TrainingValidationRows,
    candidate: Mapping[str, Any],
    relation_paths: Mapping[str, Path],
    features: PreparedFeatures,
    embedding_dir: Path,
    output_root: Path,
    device: str,
    seed: int,
) -> TrainingResult:
    """Run one ASTRARefined trajectory and copy all requested snapshots."""

    _configure_runtime(seed, device, output_root / "runtime")
    model_class, config_class = _load_refined_components()
    runtime_device = _runtime_device_diagnostics(device)
    config = _configure_model(
        config_class,
        candidate=candidate,
        relation_paths=relation_paths,
        features=features,
        embedding_dir=embedding_dir,
        output_root=output_root / "runtime",
    )
    training_rows = compact_pairs_to_training_rows(
        protocol, rows.train_pairs, rows.train_labels
    )
    model = model_class(config, training_rows, [], f"[{rows.fold}]")
    model._benchmark_protocol = protocol
    _rebind_model_mappings(model, protocol)
    if getattr(model.data, "testData", None):
        raise RefinedCandidateRunnerError("evaluation rows were passed to the model")
    if len(model.data.trainingData) != len(training_rows):
        raise RefinedCandidateRunnerError("model changed the supplied training-row count")
    train_keys = {
        (int(left), int(right)) for left, right in np.asarray(rows.train_pairs)
    }
    validation_keys = {
        (int(left), int(right)) for left, right in np.asarray(rows.validation_pairs)
    }
    if train_keys.intersection(validation_keys):
        raise RefinedCandidateRunnerError("validation pairs leaked into training")

    started = time.time()
    model.readConfiguration()
    model.initModel()
    _initialize_model_variables(model)
    semantic_input_frozen = bool(model.options.semantic_freeze_input)
    if semantic_input_frozen:
        # The shared audit interprets a zero gradient to the input tensor as an
        # inactive feature.  A frozen semantic tensor is intentionally
        # stop_gradient-bound but still feeds the trainable adapter.  Preserve
        # every path/value/tensor check while asking that audit for the expected
        # zero input gradient, then record the refined meaning explicitly.
        observed_alpha = model.text_fusion_alpha
        model.text_fusion_alpha = 0.0
        try:
            binding = _assert_ethnobotany_feature_binding(
                model,
                protocol,
                features.paths,
                features.mask_paths,
                clip_values=True,
            )
        finally:
            model.text_fusion_alpha = observed_alpha
        for entity in ("herb", "disease"):
            binding[entity]["semantic_input_frozen"] = True
            binding[entity]["input_gradient_expected"] = False
            binding[entity]["consumer"] = "trainable_semantic_adapter"
    else:
        binding = _assert_ethnobotany_feature_binding(
            model,
            protocol,
            features.paths,
            features.mask_paths,
            clip_values=True,
        )
    model.trainModel()
    snapshots: dict[int, Mapping[str, Any]] = {}
    for epoch in CHECKPOINT_EPOCHS:
        try:
            snapshot = model.epoch_snapshots[epoch]
        except (AttributeError, KeyError) as error:
            raise RefinedCandidateRunnerError(
                f"ASTRARefined did not retain epoch {epoch}"
            ) from error
        snapshots[epoch] = {
            **dict(snapshot),
            "u": np.asarray(snapshot["u"]).copy(),
            "i": np.asarray(snapshot["i"]).copy(),
        }
    session = getattr(model, "sess", None)
    if session is not None:
        session.close()
    return TrainingResult(
        snapshots=snapshots,
        diagnostics={
            "runtime_seconds": time.time() - started,
            "device": runtime_device,
            "feature_binding": binding,
            "resolved_hyperparameters": dict(model.resolved_hyperparameters),
        },
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RefinedCandidateRunnerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": sha256(path),
    }


def run_validation_candidate(
    *,
    protocol_manifest: Path,
    data_dir: Path,
    embedding_dir: Path,
    semantic_reduction_manifest: Path,
    semantic_deep_audit_receipt: Path | None = None,
    feature_deep_audit_receipt: Path | None = None,
    config_json: Path,
    fold: int,
    output_root: Path,
    device: str = "auto",
    protocol_loader: Callable[[Path], BenchmarkProtocol] = load_protocol_manifest,
    feature_preparer: Callable[..., PreparedFeatures] = prepare_candidate_features,
    feature_receipt_validator: Callable[..., Mapping[str, Any]] = (
        validate_feature_deep_audit_receipt
    ),
    trainer: Callable[..., TrainingResult] = train_refined_candidate,
) -> dict[str, Any]:
    """Create one candidate manifest; the outer fold is never evaluated."""

    output_root = Path(output_root).resolve(strict=False)
    manifest_path = output_root / "candidate.json"
    if output_root.exists() and any(output_root.iterdir()):
        raise RefinedCandidateRunnerError(
            f"candidate output directory must be new or empty: {output_root}"
        )
    protocol_manifest = Path(protocol_manifest).resolve(strict=True)
    config_json = Path(config_json).resolve(strict=True)
    candidate = validate_candidate_config(_load_json(config_json))
    protocol = protocol_loader(protocol_manifest)
    if protocol.dataset != "ethnobotany":
        raise RefinedCandidateRunnerError("refined runner supports only Ethnobotany")
    rows = derive_training_validation_rows(protocol, fold)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_data_dir = Path(data_dir).resolve(strict=True)
    resolved_embedding_dir = Path(embedding_dir).resolve(strict=True)
    resolved_semantic_manifest = Path(semantic_reduction_manifest).resolve(
        strict=True
    )
    feature_arguments = {
        "protocol": protocol,
        "data_dir": resolved_data_dir,
        "embedding_dir": resolved_embedding_dir,
        "semantic_reduction_manifest": resolved_semantic_manifest,
        "output_root": output_root,
    }
    resolved_semantic_receipt: Path | None = None
    if semantic_deep_audit_receipt is not None:
        resolved_semantic_receipt = Path(
            semantic_deep_audit_receipt
        ).resolve(strict=True)
        feature_arguments["semantic_deep_audit_receipt"] = resolved_semantic_receipt
    if feature_deep_audit_receipt is None:
        features = feature_preparer(**feature_arguments)
    else:
        if resolved_semantic_receipt is None:
            raise RefinedCandidateRunnerError(
                "--feature-deep-audit-receipt requires "
                "--semantic-deep-audit-receipt"
            )
        try:
            restored = feature_receipt_validator(
                receipt_path=Path(feature_deep_audit_receipt).resolve(strict=True),
                protocol_manifest=protocol_manifest,
                protocol=protocol,
                data_dir=resolved_data_dir,
                embedding_dir=resolved_embedding_dir,
                semantic_reduction_manifest=resolved_semantic_manifest,
                semantic_deep_audit_receipt=resolved_semantic_receipt,
            )
        except FeatureDeepAuditReceiptError as error:
            raise RefinedCandidateRunnerError(
                f"feature deep-audit receipt failed: {error}"
            ) from error
        try:
            features = PreparedFeatures(
                paths=restored["paths"],
                mask_paths=restored["mask_paths"],
                diagnostics=restored["diagnostics"],
            )
        except (KeyError, TypeError) as error:
            raise RefinedCandidateRunnerError(
                "feature deep-audit receipt validator returned a bad contract"
            ) from error
    _validate_semantic_model_contract(
        candidate, features.diagnostics["semantic_reduction"]
    )
    relation_paths = materialise_training_relations(
        protocol, rows, output_root / "protocol_relations"
    )
    trained = trainer(
        protocol=protocol,
        rows=rows,
        candidate=candidate,
        relation_paths=relation_paths,
        features=features,
        embedding_dir=resolved_embedding_dir,
        output_root=output_root,
        device=device,
        seed=int(protocol.seed),
    )
    if set(trained.snapshots) != set(CHECKPOINT_EPOCHS):
        raise RefinedCandidateRunnerError(
            f"trainer snapshots must be exactly {list(CHECKPOINT_EPOCHS)}"
        )

    snapshot_records: list[dict[str, Any]] = []
    for epoch in CHECKPOINT_EPOCHS:
        snapshot = trained.snapshots[epoch]
        epoch_root = output_root / "snapshots" / f"epoch_{epoch}"
        epoch_root.mkdir(parents=True, exist_ok=False)
        u = np.asarray(snapshot.get("u"))
        i = np.asarray(snapshot.get("i"))
        if not np.issubdtype(u.dtype, np.number) or not np.issubdtype(i.dtype, np.number):
            raise RefinedCandidateRunnerError("checkpoint embeddings must be numeric")
        if not np.isfinite(u).all() or not np.isfinite(i).all():
            raise RefinedCandidateRunnerError("checkpoint embeddings contain NaN/Inf")
        u_path = epoch_root / "u_embedding.npy"
        i_path = epoch_root / "i_embedding.npy"
        np.save(u_path, u, allow_pickle=False)
        np.save(i_path, i, allow_pickle=False)
        scores = _score_embeddings(u, i, rows.validation_pairs)
        prediction_path = save_predictions(
            epoch_root / "validation_predictions.npz",
            row_id=rows.validation_row_indices,
            pair=rows.validation_pairs,
            label=rows.validation_labels,
            score=scores,
        )
        validate_prediction_coverage(
            prediction_path,
            expected_row_id=rows.validation_row_indices,
            expected_pair=rows.validation_pairs,
            expected_label=rows.validation_labels,
        )
        metrics = evaluate_validation_scores(rows.validation_labels, scores)
        training_trace: dict[str, Any] = {}
        for key, value in snapshot.items():
            if key in {"epoch", "u", "i"}:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise RefinedCandidateRunnerError(
                    f"snapshot {epoch} training diagnostic {key!r} is not scalar"
                )
            training_trace[str(key)] = value
        snapshot_records.append(
            {
                "epoch": epoch,
                "validation_auprc": float(metrics["Validation_AUPRC"]),
                "validation_predictions": _relative_artifact(
                    prediction_path, output_root
                ),
                "u_embedding": _relative_artifact(u_path, output_root),
                "i_embedding": _relative_artifact(i_path, output_root),
                "training": training_trace,
            }
        )

    relation_audit = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in relation_paths.items()
    }
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "fold": int(fold),
        "config": candidate["config"],
        "snapshots": snapshot_records,
        "validation_only": True,
        "outer_evaluation_rows_loaded": False,
        "config_source": {
            "path": str(config_json),
            "sha256": sha256(config_json),
            "schema": CONFIG_SCHEMA,
        },
        "protocol": {
            "path": str(protocol_manifest),
            "sha256": sha256(protocol_manifest),
            "seed": int(protocol.seed),
        },
        "split": {
            "train_rows": len(rows.train_row_indices),
            "validation_rows": len(rows.validation_row_indices),
            "train_row_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(rows.train_row_indices).tobytes()
            ).hexdigest(),
            "validation_row_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(rows.validation_row_indices).tobytes()
            ).hexdigest(),
        },
        "features": features.diagnostics,
        "relations": relation_audit,
        "training_diagnostics": dict(trained.diagnostics),
    }
    _atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--semantic-reduction-manifest", type=Path, required=True)
    parser.add_argument("--semantic-deep-audit-receipt", type=Path)
    parser.add_argument("--feature-deep-audit-receipt", type=Path)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_validation_candidate(
            protocol_manifest=args.protocol_manifest,
            data_dir=args.data_dir,
            embedding_dir=args.embedding_dir,
            semantic_reduction_manifest=args.semantic_reduction_manifest,
            semantic_deep_audit_receipt=args.semantic_deep_audit_receipt,
            feature_deep_audit_receipt=args.feature_deep_audit_receipt,
            config_json=args.config_json,
            fold=args.fold,
            output_root=args.output_root,
            device=args.device,
        )
    except (RefinedCandidateRunnerError, ASTRARunnerError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
