#!/usr/bin/env python3
"""Run commit-pinned HGHDA on the central validation-based fair protocol.

The runner deliberately keeps the author's model and optimizer while replacing
only dataset plumbing and evaluation.  It consumes the exact ``train`` split
from :class:`astra.benchmark.protocol.BenchmarkProtocol`; validation and outer
test rows are never passed to the author model.  The epoch budget is fixed
before training (upstream-default 4 or matched-budget 16), and validation is
used only to select the reported operating threshold.

For Ethnobotany, compound and protein matrices must carry the row-identity
sidecars required by the central feature contract.  They are reordered to the
frozen mapping, column-standardised, clipped to ``[-10, 10]``, and supplied to
the untouched upstream HGHDA implementation.  For TCM-suite, the runner accepts
only the two exact biological matrices released with the locked HGHDA commit
and only when the frozen structural-relation hashes establish the corresponding
author row order.  The TCM matrices are consumed without the Ethnobotany-only
standardisation transform.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from astra.benchmark.features import FeatureContractError, load_aligned_feature_matrix
from astra.benchmark.astra_runner import (
    ASTRARunnerError,
    _validate_ethnobotany_feature_bundle,
)
from astra.benchmark.metrics import (
    evaluate_validation_and_test,
    save_predictions,
    validate_prediction_coverage,
)
from astra.benchmark.protocol import BenchmarkProtocol, FoldRows, load_protocol_manifest
from astra.tf_initialization import initialize_uninitialized_variables

from .common import (
    DEFAULT_THIRD_PARTY_DIR,
    array_sha256,
    balanced_prefix_indices,
    diagnostic_execution_context,
    fold_split,
    formal_runtime_module_origins,
    runtime_versions,
    sha256,
    validate_frozen_input_manifest,
    verify_checkout_commit,
    write_json,
)


UPSTREAM_COMMIT = "c176902f4206f45c2d23bc1c684eeb0c229984cb"
UPSTREAM_SOURCE = "https://github.com/bioxjz/HGHDA.git"
SUPPORTED_EPOCHS = (4, 16)
BATCH_SIZE = 2_000
LEARNING_RATE = 0.005
EMBEDDING_DIMENSION = 64
SMOKE_ROWS_PER_CLASS = 64
FEATURE_FILENAMES = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
}
UPSTREAM_FEATURE_FILENAMES = {
    "compound": "compound_embedding-64.txt",
    "protein": "protein_low.txt",
}
TCM_AUTHOR_FEATURE_SHA256 = {
    "compound": "1f27f3b0d1d0e6f5e940584b42d5991cddacbe2a750ebdb89047a7df96a6c51d",
    "protein": "7d5de593bd70a8334c65e9ff2f58a6c39ebebfafc2ac248ac1c8d99683d1ad66",
}
TCM_AUTHOR_RELATION_SHA256 = {
    "H_C": "f2d9c2f2ac3f07ebefe08645a9c8bfedf73d221fa16d7cb4a256521157ec14c1",
    "C_P": "e108b43d5c60c4dbb0d373ec657f634645992ab28a60d0f975a02866f8fb4154",
    "P_D": "91bc0e235e500849f18d9efc0fb15fc8148142f8d5333eb212538f378ef8e7b3",
}


class HGHDAFairRunnerError(RuntimeError):
    """Raised when an HGHDA Scheme-B run cannot be audited safely."""


def _atomic_savetxt(path: Path, values: np.ndarray) -> None:
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
            np.savetxt(handle, values, fmt="%.18e")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_relation(path: Path, rows: Sequence[Sequence[object]]) -> None:
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


def _raw_relation_rows(
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
    protocol: BenchmarkProtocol, dataset_dir: Path
) -> dict[str, Path]:
    """Create the exact de-duplicated relation view frozen by the manifest."""

    paths = {
        "H_C": dataset_dir / "H_C.txt",
        "C_P": dataset_dir / "C_P.txt",
        "P_D": dataset_dir / "P_D.txt",
    }
    _atomic_relation(
        paths["H_C"],
        _raw_relation_rows(
            protocol, protocol.herb_compound_pairs, "herb", "compound"
        ),
    )
    _atomic_relation(
        paths["C_P"],
        _raw_relation_rows(
            protocol, protocol.compound_protein_pairs, "compound", "protein"
        ),
    )
    _atomic_relation(
        paths["P_D"],
        _raw_relation_rows(
            protocol, protocol.protein_disease_pairs, "protein", "disease"
        ),
    )
    return paths


def _standardise_and_clip(
    values: np.ndarray, *, missing_mask: np.ndarray | None = None
) -> np.ndarray:
    """Apply the fixed Ethnobotany biological-feature transform."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise HGHDAFairRunnerError(
            f"feature matrix must have shape (n, {EMBEDDING_DIMENSION}), got "
            f"{matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise HGHDAFairRunnerError("feature matrix contains NaN or infinity")
    mask = (
        np.zeros(matrix.shape[0], dtype=bool)
        if missing_mask is None
        else np.asarray(missing_mask)
    )
    if mask.dtype != np.bool_ or mask.shape != (matrix.shape[0],):
        raise HGHDAFairRunnerError(
            "protein missing mask must be boolean with one entry per row"
        )
    if mask.any() and not np.array_equal(
        matrix[mask],
        np.zeros((int(mask.sum()), matrix.shape[1]), dtype=matrix.dtype),
    ):
        raise HGHDAFairRunnerError(
            "masked zero-UNK feature rows are not exactly zero on disk"
        )
    audit_matrix = matrix[~mask]
    if audit_matrix.shape[0] < 2:
        raise HGHDAFairRunnerError(
            "formal HGHDA requires at least two non-masked feature rows"
        )
    varying_columns = int(np.count_nonzero(np.ptp(audit_matrix, axis=0) > 0.0))
    unique_rows = int(np.unique(audit_matrix, axis=0).shape[0])
    if unique_rows < 2 or varying_columns == 0:
        raise HGHDAFairRunnerError(
            "formal HGHDA requires a non-degenerate feature matrix"
        )
    transformed = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-8)
    transformed = np.clip(transformed, -10.0, 10.0).astype(np.float32)
    if mask.any():
        transformed[mask] = 0.0
    if not np.isfinite(transformed).all():
        raise HGHDAFairRunnerError("feature transform emitted NaN or infinity")
    transformed_audit = transformed[~mask]
    if (
        transformed_audit.shape[0] < 2
        or np.unique(transformed_audit, axis=0).shape[0] < 2
        or np.count_nonzero(np.ptp(transformed_audit, axis=0) > 0.0) == 0
    ):
        raise HGHDAFairRunnerError(
            "feature matrix is degenerate after standardisation/clipping"
        )
    return transformed


def _load_tcm_author_feature(
    protocol: BenchmarkProtocol, embedding_dir: Path, entity: str
) -> tuple[np.ndarray, Path]:
    """Load one exact, commit-locked TCM feature matrix.

    TCM has no independent mapping CSV.  Row identity is therefore accepted
    only for the byte-exact author matrix together with the byte-exact relation
    files from which the protocol's first-occurrence entity order is derived.
    """

    source = embedding_dir / UPSTREAM_FEATURE_FILENAMES[entity]
    if not source.is_file():
        raise HGHDAFairRunnerError(
            f"missing locked TCM HGHDA {entity} matrix: {source}"
        )
    actual_hash = sha256(source)
    expected_hash = TCM_AUTHOR_FEATURE_SHA256[entity]
    if actual_hash != expected_hash:
        raise HGHDAFairRunnerError(
            f"TCM HGHDA {entity} matrix hash differs from the locked author "
            f"asset: {actual_hash} != {expected_hash}"
        )
    try:
        matrix = np.asarray(np.loadtxt(source), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise HGHDAFairRunnerError(
            f"cannot read locked TCM HGHDA {entity} matrix {source}: {error}"
        ) from error
    expected_shape = (len(protocol.entity_ids[entity]), EMBEDDING_DIMENSION)
    if matrix.shape != expected_shape:
        raise HGHDAFairRunnerError(
            f"locked TCM HGHDA {entity} matrix has shape {matrix.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(matrix).all():
        raise HGHDAFairRunnerError(
            f"locked TCM HGHDA {entity} matrix contains NaN or infinity"
        )
    if (
        np.unique(matrix, axis=0).shape[0] < 2
        or np.count_nonzero(np.ptp(matrix, axis=0) > 0.0) == 0
    ):
        raise HGHDAFairRunnerError(
            f"locked TCM HGHDA {entity} matrix is degenerate"
        )
    return matrix, source


def _validate_tcm_author_feature_bundle(
    protocol: BenchmarkProtocol, embedding_dir: Path
) -> dict[str, Any]:
    """Validate the closed TCM author-feature and relation-order contract."""

    if protocol.dataset != "tcm-suite":
        raise HGHDAFairRunnerError(
            "the locked TCM HGHDA feature contract requires TCM-suite"
        )
    relation_records: dict[str, dict[str, object]] = {}
    for relation, expected_hash in TCM_AUTHOR_RELATION_SHA256.items():
        record = protocol.source_files.get(relation)
        actual_hash = record.get("sha256") if isinstance(record, Mapping) else None
        if actual_hash != expected_hash:
            raise HGHDAFairRunnerError(
                f"TCM protocol {relation} hash does not establish the locked "
                f"HGHDA row order: {actual_hash!r} != {expected_hash!r}"
            )
        relation_records[relation] = {
            "sha256": actual_hash,
            "path": record.get("path"),
        }

    entities: dict[str, dict[str, object]] = {}
    for entity in ("compound", "protein"):
        matrix, source = _load_tcm_author_feature(protocol, embedding_dir, entity)
        entities[entity] = {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "shape": list(matrix.shape),
            "row_identity": (
                "locked author matrix plus locked first-occurrence relation order"
            ),
        }
    return {
        "schema": "astra-tcm-hghda-author-feature-bundle/v1",
        "dataset": "tcm-suite",
        "upstream_commit": UPSTREAM_COMMIT,
        "formal_eligible": True,
        "relations": relation_records,
        "entities": entities,
    }


def prepare_features(
    protocol: BenchmarkProtocol,
    embedding_dir: Path,
    dataset_dir: Path,
    feature_mask_paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Align feature rows to protocol IDs and materialise author input files."""

    diagnostics: dict[str, dict[str, Any]] = {}
    for entity in ("compound", "protein"):
        if protocol.dataset == "tcm-suite":
            matrix, source = _load_tcm_author_feature(
                protocol, embedding_dir, entity
            )
            consumed = np.asarray(matrix, dtype=np.float32)
            destination = dataset_dir / UPSTREAM_FEATURE_FILENAMES[entity]
            _atomic_savetxt(destination, consumed)
            diagnostics[entity] = {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
                "shape": list(consumed.shape),
                "dtype": str(consumed.dtype),
                "preprocessing": (
                    "no value transform; float32 cast matches the locked author loader"
                ),
                "row_identity": (
                    "locked author matrix plus locked first-occurrence relation order"
                ),
                "consumed_path": str(destination.resolve()),
                "consumed_bytes": destination.stat().st_size,
                "consumed_sha256": sha256(destination),
                "consumed_array_sha256": array_sha256(consumed),
                "missing_mask_path": None,
                "missing_count": 0,
            }
            continue

        source = embedding_dir / FEATURE_FILENAMES[entity]
        mapping_record = protocol.source_files.get(f"mapping/{entity}")
        if not isinstance(mapping_record, Mapping) or not mapping_record.get("sha256"):
            raise HGHDAFairRunnerError(
                f"protocol does not freeze the {entity} mapping checksum"
            )
        try:
            aligned = load_aligned_feature_matrix(
                source,
                entity=entity,
                canonical_ids=protocol.entity_ids[entity],
                mapping_sha256=str(mapping_record["sha256"]),
                expected_columns=EMBEDDING_DIMENSION,
            )
        except FeatureContractError as error:
            raise HGHDAFairRunnerError(str(error)) from error
        if bool(aligned.diagnostics.get("degenerate")):
            raise HGHDAFairRunnerError(
                f"formal HGHDA rejects degenerate {entity} features"
            )
        missing_mask = None
        mask_path = None
        if feature_mask_paths and entity in feature_mask_paths:
            mask_path = feature_mask_paths[entity]
            mask_raw = np.load(mask_path, allow_pickle=False)
            if mask_raw.dtype != np.bool_:
                raise HGHDAFairRunnerError(
                    f"{entity} missing mask must have boolean dtype"
                )
            missing_mask = np.asarray(mask_raw, dtype=bool)
        consumed = _standardise_and_clip(
            aligned.matrix,
            missing_mask=missing_mask if entity == "protein" else None,
        )
        destination = dataset_dir / UPSTREAM_FEATURE_FILENAMES[entity]
        _atomic_savetxt(destination, consumed)
        diagnostics[entity] = {
            **dict(aligned.diagnostics),
            "preprocessing": (
                "column z-score with denominator std + 1e-8, then clip to [-10, 10]"
            ),
            "consumed_path": str(destination.resolve()),
            "consumed_bytes": destination.stat().st_size,
            "consumed_sha256": sha256(destination),
            "consumed_array_sha256": array_sha256(consumed),
            "missing_mask_path": str(mask_path.resolve()) if mask_path else None,
            "missing_count": (
                int(missing_mask.sum()) if missing_mask is not None else 0
            ),
        }
    return diagnostics


def _probe_hghda_feature_dependencies(
    model: object,
) -> dict[str, dict[str, object]]:
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise HGHDAFairRunnerError(
            "TensorFlow is unavailable for the HGHDA feature dependency probe"
        ) from error
    session = getattr(model, "sess", None)
    endpoints = {
        "compound": ("compound_embeddings", "final_hcedge"),
        "protein": ("protein_embeddings", "final_pdedge"),
    }
    diagnostics: dict[str, dict[str, object]] = {}
    for entity, (variable_name, output_name) in endpoints.items():
        variable = getattr(model, variable_name, None)
        output = getattr(model, output_name, None)
        if variable is None or output is None:
            raise HGHDAFairRunnerError(
                f"author HGHDA {entity} dependency endpoints are unavailable"
            )
        gradient = tf.gradients(tf.reduce_sum(output), variable)[0]
        if gradient is None:
            raise HGHDAFairRunnerError(
                f"author HGHDA {entity} input is disconnected from its final state"
            )
        values = np.asarray(session.run(gradient), dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not np.isfinite(norm) or norm <= 0.0:
            raise HGHDAFairRunnerError(
                f"author HGHDA {entity} dependency gradient is zero/non-finite"
            )
        diagnostics[entity] = {"connected": True, "gradient_l2_norm": norm}
    return diagnostics


def assert_author_feature_binding(
    model: object,
    dataset_dir: Path,
    protein_mask_path: Path | None,
) -> dict[str, dict[str, object]]:
    """Verify both author biological initializations and the audited mask."""

    session = getattr(model, "sess", None)
    if session is None:
        raise HGHDAFairRunnerError(
            "author HGHDA did not expose its feature-loading session"
        )
    loaded_by_entity: dict[str, np.ndarray] = {}
    expected_by_entity: dict[str, np.ndarray] = {}
    for entity, attribute in (
        ("compound", "compound_embeddings"),
        ("protein", "protein_embeddings"),
    ):
        expected = np.asarray(
            np.loadtxt(dataset_dir / UPSTREAM_FEATURE_FILENAMES[entity]),
            dtype=np.float32,
        )
        variable = getattr(model, attribute, None)
        initial_value = getattr(variable, "initial_value", None)
        if initial_value is None:
            raise HGHDAFairRunnerError(
                f"author HGHDA did not expose its loaded {entity} initialization"
            )
        loaded = np.asarray(session.run(initial_value), dtype=np.float32)
        if loaded.shape != expected.shape or not np.array_equal(loaded, expected):
            raise HGHDAFairRunnerError(
                f"author HGHDA {entity} initialization differs from the audited matrix"
            )
        loaded_by_entity[entity] = loaded
        expected_by_entity[entity] = expected
    if protein_mask_path is not None:
        mask = np.load(protein_mask_path, allow_pickle=False)
        loaded = loaded_by_entity["protein"]
        if mask.dtype != np.bool_ or mask.shape != (loaded.shape[0],):
            raise HGHDAFairRunnerError("audited protein mask has a wrong dtype/shape")
        if np.asarray(mask, dtype=bool).any() and not np.array_equal(
            loaded[np.asarray(mask, dtype=bool)],
            np.zeros(
                (int(np.asarray(mask, dtype=bool).sum()), loaded.shape[1]),
                dtype=loaded.dtype,
            ),
        ):
            raise HGHDAFairRunnerError(
                "author HGHDA changed zero-UNK protein rows during loading"
            )
    dependencies = _probe_hghda_feature_dependencies(model)
    return {
        entity: {
            "post_transform_array_sha256": array_sha256(expected_by_entity[entity]),
            "tensorflow_initial_array_sha256": array_sha256(loaded_by_entity[entity]),
            **dependencies[entity],
        }
        for entity in ("compound", "protein")
    }


def training_rows(
    protocol: BenchmarkProtocol, pairs: np.ndarray, labels: np.ndarray
) -> list[list[object]]:
    """Convert only the selected compact training split to author raw IDs."""

    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise HGHDAFairRunnerError("training pairs and labels are not aligned")
    if not np.all(np.isin(labels, (0, 1))):
        raise HGHDAFairRunnerError("training labels must be binary")
    return [
        [
            protocol.compact_to_raw("herb", int(pair[0])),
            protocol.compact_to_raw("disease", int(pair[1])),
            float(label),
        ]
        for pair, label in zip(pairs, labels)
    ]


def assert_author_training_input(
    actual_rows: Sequence[Sequence[object]],
    expected_rows: Sequence[Sequence[object]],
) -> None:
    """Require the author data object to retain the exact inner-train rows."""

    normalised_actual = [
        (str(row[0]), str(row[1]), float(row[2])) for row in actual_rows
    ]
    normalised_expected = [
        (str(row[0]), str(row[1]), float(row[2])) for row in expected_rows
    ]
    if normalised_actual != normalised_expected:
        raise HGHDAFairRunnerError(
            "the author data loader changed or repartitioned the central "
            "inner-training rows"
        )


def assert_split_isolation(fold_rows: FoldRows) -> None:
    """Fail closed if central train/validation/test identities overlap."""

    split_values = {
        name: fold_split(fold_rows, name)
        for name in ("train", "validation", "test")
    }
    row_sets = {
        name: set(np.asarray(values[2]).tolist())
        for name, values in split_values.items()
    }
    pair_sets = {
        name: {tuple(map(int, pair)) for pair in np.asarray(values[0])}
        for name, values in split_values.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if row_sets[left].intersection(row_sets[right]):
            raise HGHDAFairRunnerError(f"{left}/{right} row identities overlap")
        if pair_sets[left].intersection(pair_sets[right]):
            raise HGHDAFairRunnerError(f"{left}/{right} H-D pairs overlap")


def _rebind_model_mappings(model: object, protocol: BenchmarkProtocol) -> None:
    """Bind the author's lookups and tensor dimensions to protocol row order."""

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
        setattr(data, entity, mapping)
        setattr(data, f"id2{entity}", {index: value for value, index in mapping.items()})
        setattr(data, list_attribute, np.asarray(raw_ids, dtype=str))
        setattr(model, count_attribute, len(raw_ids))

    relation_specs = (
        (data.hcassociation, data.herb, data.compound, "H--C"),
        (data.cpassociation, data.compound, data.protein, "C--P"),
        (data.pdassociation, data.protein, data.disease, "P--D"),
    )
    for rows, left_mapping, right_mapping, name in relation_specs:
        missing = [
            (str(row[0]), str(row[1]))
            for row in rows
            if str(row[0]) not in left_mapping or str(row[1]) not in right_mapping
        ]
        if missing:
            raise HGHDAFairRunnerError(
                f"{name} contains identifiers outside the frozen mappings: {missing[:5]}"
            )


def assert_author_mapping_binding(
    model: object, protocol: BenchmarkProtocol
) -> None:
    """Recheck all canonical author mappings before and after optimization."""

    specifications = (
        ("herb", "Hlist", "num_herbs"),
        ("disease", "Dlist", "num_diseases"),
        ("compound", "Clist", "num_compounds"),
        ("protein", "Plist", "num_proteins"),
    )
    for entity, list_attribute, count_attribute in specifications:
        raw_ids = tuple(str(value) for value in protocol.entity_ids[entity])
        expected = {value: index for index, value in enumerate(raw_ids)}
        inverse = {index: value for value, index in expected.items()}
        observed_list = tuple(
            str(value) for value in np.asarray(getattr(model.data, list_attribute)).tolist()
        )
        if (
            getattr(model.data, entity, None) != expected
            or getattr(model.data, f"id2{entity}", None) != inverse
            or observed_list != raw_ids
            or type(getattr(model, count_attribute, None)) is not int
            or getattr(model, count_attribute) != len(raw_ids)
        ):
            raise HGHDAFairRunnerError(
                f"author HGHDA {entity} mapping drifted from canonical protocol order"
            )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def score_pairs(model: object, pairs: np.ndarray) -> np.ndarray:
    """Return continuous probabilities for compact H-D pairs."""

    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise HGHDAFairRunnerError(f"prediction pairs must have shape (n, 2), got {pairs.shape}")
    herb_embeddings = np.asarray(model.u, dtype=np.float64)[pairs[:, 0]]
    disease_embeddings = np.asarray(model.i, dtype=np.float64)[pairs[:, 1]]
    scores = _stable_sigmoid(
        np.einsum("ij,ij->i", herb_embeddings, disease_embeddings, optimize=True)
    )
    if not np.isfinite(scores).all():
        raise HGHDAFairRunnerError("HGHDA emitted non-finite scores")
    return scores


def _smoke_subset(
    pairs: np.ndarray,
    labels: np.ndarray,
    row_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = balanced_prefix_indices(labels, SMOKE_ROWS_PER_CLASS)
    return pairs[selected], labels[selected], row_id[selected]


def _configure_device(device: str) -> None:
    normalised = str(device).strip().lower()
    if normalised == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif normalised == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif normalised != "auto":
        raise HGHDAFairRunnerError("device must be auto, cpu, or cuda")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def _load_upstream_components(official_src: Path):
    if str(official_src) not in sys.path:
        sys.path.insert(0, str(official_src))
    # The runner is a fresh child process in formal use.  Still fail closed on
    # accidental imports from another legacy tree during direct invocation.
    legacy_names = tuple(
        name
        for name in sys.modules
        if name in {"HGHDA", "rating", "base", "util"}
        or name.startswith("base.")
        or name.startswith("util.")
    )
    for name in legacy_names:
        module = sys.modules.get(name)
        origin_value = getattr(module, "__file__", None)
        if origin_value is None or official_src not in Path(origin_value).resolve().parents:
            sys.modules.pop(name, None)
    try:
        model_class = getattr(importlib.import_module("HGHDA"), "HGHDA")
        config_class = getattr(importlib.import_module("util.config"), "ModelConf")
    except (ImportError, AttributeError) as error:
        raise HGHDAFairRunnerError(
            f"cannot import pinned HGHDA from {official_src}: {error}"
        ) from error
    model_origin = Path(sys.modules[model_class.__module__].__file__).resolve()
    if official_src not in model_origin.parents:
        raise HGHDAFairRunnerError(
            f"HGHDA resolved outside pinned checkout: {model_origin}"
        )
    return model_class, config_class


def _save_training_provenance(
    path: Path,
    *,
    row_id: np.ndarray,
    pair: np.ndarray,
    label: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        row_id=np.asarray(row_id),
        pair=np.asarray(pair, dtype=np.int64),
        label=np.asarray(label, dtype=np.uint8),
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_THIRD_PARTY_DIR / "HGHDA_official",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, choices=SUPPORTED_EPOCHS, default=4)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--smoke", action="store_true")
    execution_mode.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Use the complete frozen splits and requested 4/16-epoch budget, "
            "but force all artifacts to remain diagnostic-only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and provenance without importing TensorFlow or training.",
    )
    parser.add_argument("--run-generation-id")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    smoke = bool(getattr(args, "smoke", False))
    diagnostic = bool(getattr(args, "diagnostic", False))
    if smoke and diagnostic:
        raise HGHDAFairRunnerError(
            "--smoke and --diagnostic are mutually exclusive"
        )
    diagnostic_context = (
        diagnostic_execution_context(args.device) if diagnostic else None
    )
    diagnostic_reasons = (
        list(diagnostic_context["reasons"]) if diagnostic_context else []
    )
    manifest_path = args.protocol_manifest.resolve()
    embedding_dir = args.embedding_dir.resolve()
    output_root = args.output_root.resolve()
    source_dir = args.source_dir.resolve()
    if not embedding_dir.is_dir():
        raise HGHDAFairRunnerError(
            f"embedding directory does not exist: {embedding_dir}"
        )
    if args.epochs not in SUPPORTED_EPOCHS:
        raise HGHDAFairRunnerError(
            f"epochs must be one of {SUPPORTED_EPOCHS}, got {args.epochs}"
        )

    protocol = load_protocol_manifest(manifest_path)
    if protocol.dataset not in {"ethnobotany", "tcm-suite"}:
        raise HGHDAFairRunnerError(
            "the Scheme-B HGHDA integration supports Ethnobotany and TCM-suite only"
        )
    if int(args.seed) != int(protocol.seed):
        raise HGHDAFairRunnerError(
            f"runner seed {args.seed} differs from protocol seed {protocol.seed}"
        )
    checkout_head = verify_checkout_commit(source_dir, UPSTREAM_COMMIT)
    official_src = source_dir / "src"
    required_sources = (
        official_src / "HGHDA.py",
        official_src / "rating.py",
        official_src / "base" / "recommender.py",
        official_src / "base" / "diseaseRecommender.py",
        official_src / "base" / "herbRecommender.py",
        official_src / "util" / "config.py",
        official_src / "util" / "io.py",
        official_src / "util" / "log.py",
        official_src / "HGHDA.conf",
    )
    missing = [str(path) for path in required_sources if not path.is_file()]
    if missing:
        raise HGHDAFairRunnerError(
            "pinned HGHDA checkout is incomplete: " + ", ".join(missing)
        )

    protected_outputs = (
        output_root / "results" / "metrics.json",
        output_root / "results" / "validation_predictions.npz",
        output_root / "results" / "test_predictions.npz",
        output_root / "protocol.json",
    )
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing:
        raise HGHDAFairRunnerError(
            "refusing to overwrite existing run artifacts: " + ", ".join(existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)

    initial_input_manifest = protocol.input_manifest()
    frozen_input_lock_sha256 = validate_frozen_input_manifest(initial_input_manifest)
    fold_rows = protocol.fold_rows(args.fold)
    assert_split_isolation(fold_rows)
    train_pairs, train_labels, train_row_id = fold_split(fold_rows, "train")
    validation_pairs, validation_labels, validation_row_id = fold_split(
        fold_rows, "validation"
    )
    test_pairs, test_labels, test_row_id = fold_split(fold_rows, "test")
    if smoke:
        train_pairs, train_labels, train_row_id = _smoke_subset(
            train_pairs, train_labels, train_row_id
        )
        validation_pairs, validation_labels, validation_row_id = _smoke_subset(
            validation_pairs, validation_labels, validation_row_id
        )
        test_pairs, test_labels, test_row_id = _smoke_subset(
            test_pairs, test_labels, test_row_id
        )

    work_dir = output_root / "work"
    dataset_dir = work_dir / "dataset"
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    relation_paths = materialise_protocol_relations(protocol, dataset_dir)
    if protocol.dataset == "ethnobotany":
        selected_feature_paths = {
            entity: embedding_dir / FEATURE_FILENAMES[entity]
            for entity in ("compound", "protein")
        }
        try:
            feature_mask_paths, feature_bundle_diagnostics = (
                _validate_ethnobotany_feature_bundle(
                    protocol,
                    embedding_dir,
                    selected_feature_paths,
                    smoke=smoke,
                )
            )
        except ASTRARunnerError as error:
            raise HGHDAFairRunnerError(str(error)) from error
        compatibility_adaptations = [
            "materialised the central de-duplicated H-C, C-P and P-D arrays",
            "rebound author entity dictionaries to frozen manifest row order",
            "aligned compound/protein rows by explicit sidecar IDs",
            "applied the fixed Ethnobotany z-score and [-10, 10] clipping transform",
            "re-zeroed manifest-declared target-not-protein rows after preprocessing",
            "passed no validation or outer-test rows to the author model",
            "used central continuous-score metrics and validation-only threshold selection",
        ]
    else:
        feature_mask_paths = {}
        feature_bundle_diagnostics = _validate_tcm_author_feature_bundle(
            protocol, embedding_dir
        )
        compatibility_adaptations = [
            "materialised the central de-duplicated H-C, C-P and P-D arrays",
            "rebound author entity dictionaries to frozen manifest row order",
            "required byte-exact commit-locked TCM author biological matrices",
            "required byte-exact TCM structural relations establishing author row order",
            "preserved author TCM feature values without Ethnobotany preprocessing",
            "passed no validation or outer-test rows to the author model",
            "used central continuous-score metrics and validation-only threshold selection",
        ]
    feature_diagnostics = prepare_features(
        protocol,
        embedding_dir,
        dataset_dir,
        feature_mask_paths,
    )
    training_data = training_rows(protocol, train_pairs, train_labels)

    adapter_sources = {
        "common.py": sha256(Path(__file__).with_name("common.py")),
        "hghda_runner.py": sha256(Path(__file__)),
    }
    upstream_sources = {
        str(path.relative_to(source_dir)): sha256(path) for path in required_sources
    }
    base_provenance: dict[str, Any] = {
        "schema": "astra-central-fair-run/v1",
        "method": "HGHDA",
        "dataset": protocol.dataset,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "smoke": smoke,
        "diagnostic": diagnostic,
        "execution_mode": (
            "smoke"
            if smoke
            else "diagnostic"
            if diagnostic
            else "dry-run"
            if args.dry_run
            else "formal-candidate"
        ),
        "diagnostic_context": diagnostic_context,
        "dry_run": bool(args.dry_run),
        "formal_eligible": bool(
            not smoke and not diagnostic and not args.dry_run
        ),
        "formal_ineligibility_reasons": (
            (["smoke mode"] if smoke else [])
            + diagnostic_reasons
            + (["dry-run validation only"] if args.dry_run else [])
        ),
        "protocol_manifest": str(manifest_path),
        "protocol_manifest_sha256": sha256(manifest_path),
        "protocol_input_manifest": initial_input_manifest,
        "frozen_input_lock_sha256": frozen_input_lock_sha256,
        "run_generation_id": args.run_generation_id,
        "adapter_sources": adapter_sources,
        "upstream": {
            "repository": UPSTREAM_SOURCE,
            "commit": checkout_head,
            "source_sha256": upstream_sources,
            "config_sha256": sha256(official_src / "HGHDA.conf"),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "epoch_budget_provenance": (
                "upstream default" if args.epochs == 4 else "predeclared matched budget"
            ),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
        "split_rows": {
            "inner_train_full": int(len(fold_rows.train_pairs)),
            "inner_train_used": int(len(train_pairs)),
            "validation_full": int(len(fold_rows.validation_pairs)),
            "validation_used": int(len(validation_pairs)),
            "outer_test_full": int(len(fold_rows.test_pairs)),
            "outer_test_used": int(len(test_pairs)),
        },
        "split_array_sha256": {
            "inner_train_row_id": array_sha256(np.asarray(train_row_id)),
            "inner_train_pairs": array_sha256(np.asarray(train_pairs)),
            "inner_train_labels": array_sha256(np.asarray(train_labels)),
            "validation_row_id": array_sha256(np.asarray(validation_row_id)),
            "test_row_id": array_sha256(np.asarray(test_row_id)),
        },
        "data_use_policy": {
            "training_rows": "central inner_train only",
            "validation_used_for_training": False,
            "validation_used_for_threshold_selection": True,
            "outer_test_used_for_training": False,
            "outer_test_used_for_checkpoint_or_hyperparameter_selection": False,
            "checkpoint_selection": "none; fixed final epoch",
        },
        "relations": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in relation_paths.items()
        },
        "feature_matrices": feature_diagnostics,
        "feature_bundle": feature_bundle_diagnostics,
        "compatibility_adaptations": compatibility_adaptations,
    }
    if args.dry_run:
        payload = {
            **base_provenance,
            "status": "validated-dry-run",
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "protocol.json", payload)
        return payload

    _configure_device(args.device)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    previous_cwd = Path.cwd()
    os.chdir(work_dir)
    model = None
    try:
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        if args.device == "cuda" and not tf.config.list_physical_devices("GPU"):
            raise HGHDAFairRunnerError(
                "--device cuda was requested, but TensorFlow exposes no GPU"
            )
        UpstreamHGHDA, ModelConf = _load_upstream_components(official_src)
        max_batches = 1 if smoke else None
        optimizer_step_counter = {"count": 0}

        class ControlledHGHDA(UpstreamHGHDA):
            def next_batch_pairwise(self):
                for batch_index, batch in enumerate(super().next_batch_pairwise()):
                    if max_batches is not None and batch_index >= max_batches:
                        return
                    optimizer_step_counter["count"] += 1
                    yield batch

        config = ModelConf(str(official_src / "HGHDA.conf"))
        config.config["num.max.epoch"] = str(args.epochs)
        config.config["batch_size"] = str(BATCH_SIZE)
        config.config["learnRate"] = f"-init {LEARNING_RATE} -max 1"
        config.config["output.setup"] = f"on -dir {results_dir}/"
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)

        training_log = output_root / "training.log"
        with training_log.open("w", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                model = ControlledHGHDA(
                    config, training_data, [], f"[{args.fold}]"
                )
                assert_author_training_input(model.data.trainingData, training_data)
                _rebind_model_mappings(model, protocol)
                assert_author_mapping_binding(model, protocol)
                if model.data.testData:
                    raise HGHDAFairRunnerError(
                        "validation/test rows were unexpectedly passed to HGHDA"
                    )
                model.readConfiguration()
                if (
                    int(model.maxEpoch) != int(args.epochs)
                    or int(model.batch_size) != BATCH_SIZE
                    or not math.isclose(
                        float(model.lRate), LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise HGHDAFairRunnerError(
                        "author configuration differs from the locked Scheme-B "
                        "epoch, batch-size, or learning-rate setting"
                    )
                if int(model.train_size) != len(training_data):
                    raise HGHDAFairRunnerError(
                        "author model training size differs from central inner train"
                )
                model.initModel()
                # The released implementation creates its layer variables in
                # ``initModel`` but initializes them only inside
                # ``trainModel``.  The fair adapter performs its read-only
                # feature-dependency probe between those calls, so initialize
                # only the variables that currently exist.  No optimizer step
                # has run at this point, and the author's own training setup
                # remains otherwise unchanged.
                initialize_uninitialized_variables(model.sess, tf)
                feature_binding_diagnostics = assert_author_feature_binding(
                    model,
                    dataset_dir,
                    feature_mask_paths.get("protein"),
                )
                model.trainModel()
                assert_author_mapping_binding(model, protocol)
        log_text = training_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bloss:\s*[+-]?(?:nan|inf)\b", log_text, re.IGNORECASE):
            raise HGHDAFairRunnerError("training log contains a non-finite loss")

        validation_scores = score_pairs(model, validation_pairs)
        test_scores = score_pairs(model, test_pairs)
        metrics = evaluate_validation_and_test(
            validation_labels,
            validation_scores,
            test_labels,
            test_scores,
        )
        expected_steps = int(args.epochs) * (
            1 if smoke else int(math.ceil(len(train_pairs) / BATCH_SIZE))
        )
        if optimizer_step_counter["count"] != expected_steps:
            raise HGHDAFairRunnerError(
                "observed optimizer-step count differs from the locked schedule: "
                f"{optimizer_step_counter['count']} != {expected_steps}"
            )
        metrics.update(
            {
                "Method": "HGHDA",
                "Model": "HGHDA-commit-pinned-Scheme-B",
                "Dataset": protocol.dataset,
                "Fold": int(args.fold),
                "Seed": int(args.seed),
                "Epochs_Requested": int(args.epochs),
                "Epochs_Executed": int(args.epochs),
                "Batch_Size": BATCH_SIZE,
                "Learning_Rate": LEARNING_RATE,
                "Optimizer_Steps": int(optimizer_step_counter["count"]),
                "Smoke": smoke,
                "Diagnostic": diagnostic,
                "Execution_Mode": (
                    "smoke"
                    if smoke
                    else "diagnostic"
                    if diagnostic
                    else "formal-candidate"
                ),
                "Formal_Eligible": bool(not smoke and not diagnostic),
                "Formal_Ineligibility_Reasons": (
                    (["smoke mode"] if smoke else []) + diagnostic_reasons
                ),
                "Evaluation_Mode": "validation",
                "Training_Rows": int(len(train_pairs)),
                "Validation_Rows": int(len(validation_pairs)),
                "Test_Rows": int(len(test_pairs)),
                "Checkpoint_Source": "fixed final epoch; no checkpoint selection",
                "Upstream_Commit": UPSTREAM_COMMIT,
                "Upstream_Source": UPSTREAM_SOURCE,
                "Feature_Preprocessing": "aligned zscore_clip",
                "Runtime_Seconds": time.time() - started,
            }
        )

        result_dir = output_root / "results"
        validation_path = save_predictions(
            result_dir / "validation_predictions.npz",
            row_id=validation_row_id,
            pair=validation_pairs,
            label=validation_labels,
            score=validation_scores,
        )
        test_path = save_predictions(
            result_dir / "test_predictions.npz",
            row_id=test_row_id,
            pair=test_pairs,
            label=test_labels,
            score=test_scores,
        )
        validate_prediction_coverage(
            validation_path,
            expected_row_id=validation_row_id,
            expected_pair=validation_pairs,
            expected_label=validation_labels,
        )
        validate_prediction_coverage(
            test_path,
            expected_row_id=test_row_id,
            expected_pair=test_pairs,
            expected_label=test_labels,
        )
        _save_training_provenance(
            result_dir / "training_rows.npz",
            row_id=train_row_id,
            pair=train_pairs,
            label=train_labels,
        )
        write_json(result_dir / "metrics.json", metrics)

        verify_checkout_commit(source_dir, UPSTREAM_COMMIT)
        final_input_manifest = protocol.input_manifest()
        if final_input_manifest != initial_input_manifest:
            raise HGHDAFairRunnerError(
                "central benchmark inputs changed during the HGHDA run"
            )
        validate_frozen_input_manifest(final_input_manifest)
        provenance = {
            **base_provenance,
            "model": metrics,
            "feature_binding": feature_binding_diagnostics,
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "training_provenance": "results/training_rows.npz",
                "prediction_fields": ["row_id", "pair", "label", "score"],
            },
            "training_log": {
                "path": str(training_log.resolve()),
                "bytes": training_log.stat().st_size,
                "sha256": sha256(training_log),
            },
            "runtime_versions": runtime_versions("tensorflow"),
            "runtime_module_origins": formal_runtime_module_origins(
                require_environment=False
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "protocol.json", provenance)
        return metrics
    finally:
        if model is not None and getattr(model, "sess", None) is not None:
            model.sess.close()
        os.chdir(previous_cwd)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
