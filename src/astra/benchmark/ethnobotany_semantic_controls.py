#!/usr/bin/env python3
"""Label-free semantic input controls for Ethnobotany ASTRA experiments.

The PCA64 zero-padding control preserves every audited PCA64 value and appends
64 exact zero columns.  It exists so PCA64 and genuine PCA128 inputs can enter
the same frozen-input, trainable 128-to-64 semantic adapter.  No association
file, split, label, prediction, or metric is accepted by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .ethnobotany_semantic_features import (
    REDUCTION_MANIFEST_SCHEMA,
    SemanticFeatureError,
    _artifact,
    _atomic_json,
    _atomic_npy,
    _atomic_npz,
    _read_json,
    _sha256,
    _utc_now,
    _verify_artifact,
    validate_shared_semantic_reduction,
)
from .features import feature_row_ids_path


ZERO_PAD_CONTROL_MANIFEST_SCHEMA = (
    "astra-ethnobotany-semantic-pca64-zero-pad128-control/v1"
)
ZERO_PAD_CONTROL_ALGORITHM = "append_64_exact_float32_zero_columns_to_pca64"
SOURCE_DIMENSION = 64
OUTPUT_DIMENSION = 128
ADAPTER_OUTPUT_DIMENSION = 64


def _control_condition(source_condition: object) -> str:
    if not isinstance(source_condition, str) or not source_condition.strip():
        raise SemanticFeatureError("source semantic condition is absent")
    return f"{source_condition}::pca64-zero-pad128-control/v1"


def _consumer_contract() -> dict[str, object]:
    return {
        "semantic_input_dimension": OUTPUT_DIMENSION,
        "semantic_adapter": "trainable_linear_128_to_64",
        "semantic_adapter_output_dimension": ADAPTER_OUTPUT_DIMENSION,
        "semantic_input_trainability": "frozen",
        "same_adapter_parameterization_as_genuine_pca128_required": True,
    }


def _checkpoint_contract(
    *,
    source_manifest_sha256: str,
    source_checkpoint_sha256: str,
    source_semantic_condition: str,
) -> dict[str, object]:
    return {
        "schema": ZERO_PAD_CONTROL_MANIFEST_SCHEMA,
        "algorithm": ZERO_PAD_CONTROL_ALGORITHM,
        "source_reduction_manifest_sha256": source_manifest_sha256,
        "source_reducer_checkpoint_sha256": source_checkpoint_sha256,
        "source_semantic_condition": source_semantic_condition,
        "source_dimension": SOURCE_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "appended_zero_columns": OUTPUT_DIMENSION - SOURCE_DIMENSION,
        "fit_required": False,
        "uses_association_labels": False,
    }


def _padded(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != SOURCE_DIMENSION:
        raise SemanticFeatureError(
            f"PCA64 control source must have width {SOURCE_DIMENSION}, "
            f"got {source.shape}"
        )
    result = np.zeros((source.shape[0], OUTPUT_DIMENSION), dtype=np.float32)
    result[:, :SOURCE_DIMENSION] = source
    return result


def build_pca64_zero_pad128_control(
    *,
    source_reduction_manifest: Path | str,
    output_dir: Path | str,
) -> Path:
    """Build a new audited 128-D control without modifying its PCA64 source."""

    source_path = Path(source_reduction_manifest).resolve()
    source = validate_shared_semantic_reduction(source_path)
    source_manifest = source["manifest"]
    if (
        source_manifest.get("schema") != REDUCTION_MANIFEST_SCHEMA
        or source_manifest.get("output_dimension") != SOURCE_DIMENSION
    ):
        raise SemanticFeatureError(
            "zero-pad128 control requires a formal shared PCA64 source"
        )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "semantic_zero_pad_control_manifest.json"
    if manifest_path.exists():
        checked = validate_pca64_zero_pad128_control(manifest_path)
        if _sha256(checked["source_reduction_manifest_path"]) != _sha256(
            source_path
        ):
            raise SemanticFeatureError(
                "existing zero-pad128 control was built from another PCA64 source"
            )
        return manifest_path

    checkpoint_path = output / "shared_semantic_pca64_zero_pad128_control.npz"
    matrix_paths = {
        entity: output
        / f"{entity}_semantic_embeddings_pca64_zero_pad_128d.npy"
        for entity in ("herb", "disease")
    }
    prospective_paths = [
        checkpoint_path,
        *matrix_paths.values(),
        *(feature_row_ids_path(path) for path in matrix_paths.values()),
    ]
    existing = [path for path in prospective_paths if path.exists()]
    if existing:
        raise SemanticFeatureError(
            "refusing to overwrite an incomplete zero-pad128 control: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        with np.load(source["checkpoint_path"], allow_pickle=False) as checkpoint:
            source_mean = np.asarray(checkpoint["mean"], dtype=np.float64)
            source_components = np.asarray(
                checkpoint["components"], dtype=np.float64
            )
            source_singular_values = np.asarray(
                checkpoint["singular_values"], dtype=np.float64
            )
    except (OSError, ValueError, KeyError) as error:
        raise SemanticFeatureError(
            f"cannot load PCA64 source checkpoint: {error}"
        ) from error
    if source_components.shape != (SOURCE_DIMENSION, source_mean.size):
        raise SemanticFeatureError("PCA64 source checkpoint has an invalid shape")

    control_components = np.zeros(
        (OUTPUT_DIMENSION, source_mean.size), dtype=np.float64
    )
    control_components[:SOURCE_DIMENSION] = source_components
    checkpoint_contract = _checkpoint_contract(
        source_manifest_sha256=_sha256(source_path),
        source_checkpoint_sha256=_sha256(source["checkpoint_path"]),
        source_semantic_condition=str(source_manifest["semantic_condition"]),
    )
    _atomic_npz(
        checkpoint_path,
        mean=source_mean,
        components=control_components,
        singular_values=source_singular_values,
        astra_zero_pad_control_contract_json=np.asarray(
            json.dumps(checkpoint_contract, sort_keys=True, separators=(",", ":"))
        ),
    )

    outputs: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        source_matrix = np.asarray(source["outputs"][entity]["matrix"])
        matrix = _padded(source_matrix)
        matrix_path = matrix_paths[entity]
        sidecar_path = feature_row_ids_path(matrix_path)
        source_record = source_manifest["entities"][entity]
        source_sidecar_path = _verify_artifact(
            source_record.get("row_ids_sidecar"),
            f"source PCA64 {entity} row IDs",
        )
        _atomic_npy(matrix_path, matrix)
        _atomic_json(sidecar_path, _read_json(source_sidecar_path))
        outputs[entity] = {
            "matrix": _artifact(matrix_path),
            "row_ids_sidecar": _artifact(sidecar_path),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "mapping_sha256": source_record["mapping_sha256"],
            "unique_rows": int(np.unique(matrix, axis=0).shape[0]),
            "varying_columns": int(
                np.count_nonzero(np.ptp(matrix, axis=0) > 0.0)
            ),
            "preserved_source_columns": [0, SOURCE_DIMENSION],
            "exact_zero_columns": [SOURCE_DIMENSION, OUTPUT_DIMENSION],
        }

    manifest = {
        **checkpoint_contract,
        "created_at": _utc_now(),
        "formal_eligible": True,
        "controlled_experiment_only": True,
        "random_imputation": False,
        "association_label_free": True,
        "label_inputs": [],
        "semantic_condition": _control_condition(
            source_manifest["semantic_condition"]
        ),
        "semantic_condition_status": (
            "formal_label_free_pca64_zero_pad128_control"
        ),
        "source_reduction_manifest": _artifact(source_path),
        "reducer_checkpoint": _artifact(checkpoint_path),
        "consumer_contract": _consumer_contract(),
        "entities": outputs,
    }
    _atomic_json(manifest_path, manifest)
    validate_pca64_zero_pad128_control(manifest_path)
    return manifest_path


def validate_pca64_zero_pad128_control(
    path: Path | str,
    *,
    resolver: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Recursively prove that a control is PCA64 followed only by zero padding."""

    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != ZERO_PAD_CONTROL_MANIFEST_SCHEMA:
        raise SemanticFeatureError("zero-pad128 control manifest has a wrong schema")
    if (
        payload.get("algorithm") != ZERO_PAD_CONTROL_ALGORITHM
        or payload.get("source_dimension") != SOURCE_DIMENSION
        or payload.get("output_dimension") != OUTPUT_DIMENSION
        or payload.get("appended_zero_columns")
        != OUTPUT_DIMENSION - SOURCE_DIMENSION
        or payload.get("fit_required") is not False
        or payload.get("uses_association_labels") is not False
        or payload.get("formal_eligible") is not True
        or payload.get("controlled_experiment_only") is not True
        or payload.get("random_imputation") is not False
        or payload.get("association_label_free") is not True
        or payload.get("label_inputs") != []
        or payload.get("consumer_contract") != _consumer_contract()
    ):
        raise SemanticFeatureError("zero-pad128 control eligibility contract is false")

    source_path = _verify_artifact(
        payload.get("source_reduction_manifest"),
        "source PCA64 reduction manifest",
        resolver=resolver,
    )
    source = validate_shared_semantic_reduction(source_path, resolver=resolver)
    source_manifest = source["manifest"]
    if (
        source_manifest.get("schema") != REDUCTION_MANIFEST_SCHEMA
        or source_manifest.get("output_dimension") != SOURCE_DIMENSION
    ):
        raise SemanticFeatureError(
            "zero-pad128 control source is not a formal shared PCA64 reduction"
        )
    source_condition = str(source_manifest.get("semantic_condition", ""))
    if (
        payload.get("source_reduction_manifest_sha256") != _sha256(source_path)
        or payload.get("source_reducer_checkpoint_sha256")
        != _sha256(source["checkpoint_path"])
        or payload.get("source_semantic_condition") != source_condition
        or payload.get("semantic_condition") != _control_condition(source_condition)
        or payload.get("semantic_condition_status")
        != "formal_label_free_pca64_zero_pad128_control"
    ):
        raise SemanticFeatureError("zero-pad128 control differs from its PCA64 source")

    checkpoint_path = _verify_artifact(
        payload.get("reducer_checkpoint"),
        "zero-pad128 control checkpoint",
        resolver=resolver,
    )
    try:
        with np.load(source["checkpoint_path"], allow_pickle=False) as checkpoint:
            source_mean = np.asarray(checkpoint["mean"], dtype=np.float64)
            source_components = np.asarray(
                checkpoint["components"], dtype=np.float64
            )
            source_singular_values = np.asarray(
                checkpoint["singular_values"], dtype=np.float64
            )
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            mean = np.asarray(checkpoint["mean"], dtype=np.float64)
            components = np.asarray(checkpoint["components"], dtype=np.float64)
            singular_values = np.asarray(
                checkpoint["singular_values"], dtype=np.float64
            )
            embedded_contract = json.loads(
                str(checkpoint["astra_zero_pad_control_contract_json"].item())
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SemanticFeatureError(
            f"cannot validate zero-pad128 control checkpoint: {error}"
        ) from error
    expected_checkpoint_contract = _checkpoint_contract(
        source_manifest_sha256=_sha256(source_path),
        source_checkpoint_sha256=_sha256(source["checkpoint_path"]),
        source_semantic_condition=source_condition,
    )
    expected_components = np.zeros(
        (OUTPUT_DIMENSION, source_components.shape[1]), dtype=np.float64
    )
    expected_components[:SOURCE_DIMENSION] = source_components
    if (
        embedded_contract != expected_checkpoint_contract
        or not np.array_equal(mean, source_mean)
        or not np.array_equal(components, expected_components)
        or not np.array_equal(singular_values, source_singular_values)
    ):
        raise SemanticFeatureError(
            "zero-pad128 checkpoint is not an exact extension of PCA64"
        )

    entities = payload.get("entities")
    if not isinstance(entities, Mapping) or set(entities) != {"herb", "disease"}:
        raise SemanticFeatureError("zero-pad128 control entity outputs are incomplete")
    outputs: dict[str, Any] = {}
    dependencies = [manifest_path, checkpoint_path, *source["dependencies"]]
    for entity in ("herb", "disease"):
        record = entities.get(entity)
        if not isinstance(record, Mapping):
            raise SemanticFeatureError(f"zero-pad128 {entity} output is malformed")
        matrix_path = _verify_artifact(
            record.get("matrix"),
            f"zero-pad128 {entity} matrix",
            resolver=resolver,
        )
        sidecar_path = _verify_artifact(
            record.get("row_ids_sidecar"),
            f"zero-pad128 {entity} row IDs",
            resolver=resolver,
        )
        try:
            observed = np.asarray(
                np.load(matrix_path, allow_pickle=False), dtype=np.float32
            )
        except (OSError, ValueError) as error:
            raise SemanticFeatureError(
                f"cannot load zero-pad128 {entity} matrix: {error}"
            ) from error
        expected = _padded(source["outputs"][entity]["matrix"])
        source_record = source_manifest["entities"][entity]
        source_sidecar_path = _verify_artifact(
            source_record.get("row_ids_sidecar"),
            f"source PCA64 {entity} row IDs",
            resolver=resolver,
        )
        if not np.array_equal(observed, expected):
            raise SemanticFeatureError(
                f"zero-pad128 {entity} matrix is not exact PCA64 plus zeros"
            )
        if _read_json(sidecar_path) != _read_json(source_sidecar_path):
            raise SemanticFeatureError(
                f"zero-pad128 {entity} row IDs differ from PCA64"
            )
        if (
            record.get("shape") != list(expected.shape)
            or record.get("dtype") != str(expected.dtype)
            or record.get("mapping_sha256")
            != source_record.get("mapping_sha256")
            or record.get("unique_rows")
            != int(np.unique(expected, axis=0).shape[0])
            or record.get("varying_columns")
            != int(np.count_nonzero(np.ptp(expected, axis=0) > 0.0))
            or record.get("preserved_source_columns") != [0, SOURCE_DIMENSION]
            or record.get("exact_zero_columns")
            != [SOURCE_DIMENSION, OUTPUT_DIMENSION]
        ):
            raise SemanticFeatureError(
                f"zero-pad128 {entity} diagnostics are false"
            )
        outputs[entity] = {"matrix_path": matrix_path, "matrix": observed}
        dependencies.extend([matrix_path, sidecar_path])

    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "source_reduction_manifest_path": source_path,
        "embedding_manifest_path": source["embedding_manifest_path"],
        "checkpoint_path": checkpoint_path,
        "outputs": outputs,
        "dependencies": list(
            dict.fromkeys(Path(item).resolve() for item in dependencies)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the PCA64 zero-pad128 control")
    build.add_argument("--source-reduction-manifest", required=True)
    build.add_argument("--output-dir", required=True)
    validate = subparsers.add_parser("validate", help="validate a saved control")
    validate.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        path = build_pca64_zero_pad128_control(
            source_reduction_manifest=args.source_reduction_manifest,
            output_dir=args.output_dir,
        )
    else:
        validate_pca64_zero_pad128_control(args.manifest)
        path = Path(args.manifest).resolve()
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
