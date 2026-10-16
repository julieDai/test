#!/usr/bin/env python3
"""Build and audit a versioned deterministic shared-PCA64 TCM bundle.

The builder consumes the already identity-audited AE64 feature bundle, not an
unlabelled matrix directory.  It takes the two target-neutral 768-D semantic
matrices from that bundle's recursive provenance, concatenates herb rows before
disease rows, centres the joint matrix, and runs NumPy's exact (non-randomized)
SVD.  Component signs are canonicalised by forcing the largest-absolute loading
of every component to be positive.  Scores are *not* whitened.

The output is a new immutable bundle.  Existing directories and the reserved
name ``final_bundle`` are rejected so this control can never overwrite the AE64
bundle used by the current strict experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .tcm_feature_bundle import (
    FEATURE_FILENAMES,
    TCM_FEATURE_BUNDLE_SCHEMA,
    TCM_SEMANTIC_CONDITION,
    TCMFeatureBundleError,
    validate_tcm_feature_bundle,
)


PCA64_BUNDLE_SCHEMA = "astra-tcm-pca64-feature-bundle/v1"
PCA64_BUNDLE_PROFILE = "herbhg-four-entity-target-neutral-shared-pca64-v1"
PCA64_SEMANTIC_CONDITION = (
    "llm_clean_target_neutral_qwen_scibert_deterministic_shared_pca64"
)
PCA64_COMPONENTS = 64
PCA64_INPUT_DIMENSION = 768
PCA64_ALGORITHM = "centered-exact-svd-canonical-sign-no-whitening"


class TCMPCA64BundleError(ValueError):
    """Raised when the PCA64 transform or its identity evidence is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TCMPCA64BundleError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TCMPCA64BundleError(f"JSON artifact must contain an object: {path}")
    return payload


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TCMPCA64BundleError(
            f"bundle artifact is outside the bundle root: {resolved}"
        ) from error
    return {
        "relative_path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _copy(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise TCMPCA64BundleError(f"required source artifact is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise TCMPCA64BundleError(f"copied artifact hash changed: {source}")
    return destination


def _load_matrix(path: Path, rows: int, columns: int) -> np.ndarray:
    try:
        values = (
            np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npy"
            else np.loadtxt(path)
        )
    except (OSError, ValueError) as error:
        raise TCMPCA64BundleError(f"cannot read feature matrix {path}: {error}") from error
    matrix = np.asarray(values)
    if matrix.shape != (rows, columns):
        raise TCMPCA64BundleError(
            f"{path} has shape {matrix.shape}; expected ({rows}, {columns})"
        )
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise TCMPCA64BundleError(f"{path} must contain finite numeric values")
    if (
        np.unique(matrix, axis=0).shape[0] < 2
        or np.count_nonzero(np.ptp(matrix, axis=0) > 0.0) == 0
    ):
        raise TCMPCA64BundleError(f"{path} is a degenerate feature matrix")
    return matrix


def canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    """Return SVD components with a deterministic, row-local sign convention."""

    result = np.asarray(components, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise TCMPCA64BundleError("PCA components must be a non-empty matrix")
    for index in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[index])))
        if result[index, pivot] == 0.0:
            raise TCMPCA64BundleError("PCA contains an all-zero component")
        if result[index, pivot] < 0.0:
            result[index] *= -1.0
    return result


def fit_centered_shared_pca64(
    herb: np.ndarray,
    disease: np.ndarray,
    *,
    n_components: int = PCA64_COMPONENTS,
) -> dict[str, np.ndarray]:
    """Fit one exact PCA basis jointly and return unwhitened entity scores."""

    herb64 = np.asarray(herb, dtype=np.float64)
    disease64 = np.asarray(disease, dtype=np.float64)
    if (
        herb64.ndim != 2
        or disease64.ndim != 2
        or herb64.shape[1] != disease64.shape[1]
        or herb64.shape[1] < n_components
        or herb64.shape[0] + disease64.shape[0] <= n_components
        or not np.isfinite(herb64).all()
        or not np.isfinite(disease64).all()
    ):
        raise TCMPCA64BundleError("joint semantic matrices cannot support PCA64")
    joint = np.concatenate((herb64, disease64), axis=0)
    mean = np.mean(joint, axis=0, dtype=np.float64)
    centered = joint - mean
    # full_matrices=False still computes the complete exact rank decomposition;
    # it merely omits mathematically unused null-space rows/columns.
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if right.shape[0] < n_components:
        raise TCMPCA64BundleError("exact SVD emitted fewer than 64 components")
    components = canonicalize_component_signs(right[:n_components])
    scores = centered @ components.T
    variance = np.square(singular_values) / float(joint.shape[0] - 1)
    total_variance = float(np.sum(np.var(joint, axis=0, ddof=1)))
    if total_variance <= 0.0 or not np.isfinite(total_variance):
        raise TCMPCA64BundleError("joint semantic matrix has no finite variance")
    explained_ratio = variance[:n_components] / total_variance
    split = herb64.shape[0]
    return {
        "mean": mean,
        "components": components,
        "singular_values": singular_values[:n_components],
        "explained_variance_ratio": explained_ratio,
        "herb_scores": scores[:split],
        "disease_scores": scores[split:],
    }


def _protocol_mapping_sha256(protocol: BenchmarkProtocol, entity: str) -> str:
    return str(protocol.input_manifest()["mappings"][entity]["sha256"])


def _resolve_artifact(
    root: Path, record: object, *, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise TCMPCA64BundleError(f"{label} artifact record is invalid")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise TCMPCA64BundleError(f"{label} has no relative_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise TCMPCA64BundleError(f"{label} escapes the bundle root") from error
    if (
        not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or sha256(path) != record.get("sha256")
    ):
        raise TCMPCA64BundleError(f"{label} bytes or SHA-256 do not match")
    return path


def _audit_transform(
    *,
    herb_raw: Path,
    disease_raw: Path,
    herb_rows: int,
    disease_rows: int,
    model_path: Path,
    herb_output: Path,
    disease_output: Path,
) -> dict[str, Any]:
    herb = _load_matrix(herb_raw, herb_rows, PCA64_INPUT_DIMENSION).astype(np.float64)
    disease = _load_matrix(
        disease_raw, disease_rows, PCA64_INPUT_DIMENSION
    ).astype(np.float64)
    with np.load(model_path, allow_pickle=False) as model:
        if set(model.files) != {
            "components",
            "explained_variance_ratio",
            "mean",
            "singular_values",
        }:
            raise TCMPCA64BundleError("PCA model fields differ from the v1 contract")
        mean = np.asarray(model["mean"], dtype=np.float64)
        components = np.asarray(model["components"], dtype=np.float64)
        singular_values = np.asarray(model["singular_values"], dtype=np.float64)
        explained_ratio = np.asarray(
            model["explained_variance_ratio"], dtype=np.float64
        )
    if mean.shape != (PCA64_INPUT_DIMENSION,) or components.shape != (
        PCA64_COMPONENTS,
        PCA64_INPUT_DIMENSION,
    ) or singular_values.shape != (PCA64_COMPONENTS,) or explained_ratio.shape != (
        PCA64_COMPONENTS,
    ):
        raise TCMPCA64BundleError("PCA model shape differs from the v1 contract")
    expected_mean = np.mean(np.concatenate((herb, disease), axis=0), axis=0)
    if not np.array_equal(mean, expected_mean):
        raise TCMPCA64BundleError("saved PCA centre is not the joint row mean")
    if not np.array_equal(components, canonicalize_component_signs(components)):
        raise TCMPCA64BundleError("saved PCA component signs are not canonical")
    joint = np.concatenate((herb, disease), axis=0)
    centered = joint - expected_mean
    _, recomputed_singular_values, recomputed_right = np.linalg.svd(
        centered, full_matrices=False
    )
    recomputed_components = canonicalize_component_signs(
        recomputed_right[:PCA64_COMPONENTS]
    )
    recomputed_variance = np.square(recomputed_singular_values) / float(
        joint.shape[0] - 1
    )
    recomputed_total_variance = float(np.sum(np.var(joint, axis=0, ddof=1)))
    recomputed_ratio = (
        recomputed_variance[:PCA64_COMPONENTS] / recomputed_total_variance
    )
    if not np.allclose(
        components, recomputed_components, rtol=0.0, atol=1e-12
    ):
        raise TCMPCA64BundleError("saved components are not the exact top-64 SVD basis")
    if not np.allclose(
        singular_values,
        recomputed_singular_values[:PCA64_COMPONENTS],
        rtol=0.0,
        atol=1e-12,
    ):
        raise TCMPCA64BundleError("saved singular values differ from exact SVD")
    if not np.allclose(
        explained_ratio, recomputed_ratio, rtol=0.0, atol=1e-15
    ):
        raise TCMPCA64BundleError("saved explained variance differs from exact PCA")
    gram = components @ components.T
    if not np.allclose(gram, np.eye(PCA64_COMPONENTS), rtol=0.0, atol=1e-10):
        raise TCMPCA64BundleError("saved PCA components are not orthonormal")
    expected_herb = ((herb - mean) @ components.T).astype(np.float32)
    expected_disease = ((disease - mean) @ components.T).astype(np.float32)
    actual_herb = _load_matrix(herb_output, herb_rows, PCA64_COMPONENTS)
    actual_disease = _load_matrix(disease_output, disease_rows, PCA64_COMPONENTS)
    if not np.array_equal(expected_herb, actual_herb):
        raise TCMPCA64BundleError("herb PCA64 output cannot be reproduced")
    if not np.array_equal(expected_disease, actual_disease):
        raise TCMPCA64BundleError("disease PCA64 output cannot be reproduced")
    return {
        "status": "verified",
        "joint_center_recomputed": True,
        "exact_top64_svd_recomputed": True,
        "canonical_signs_verified": True,
        "orthonormal_components_verified": True,
        "unwhitened_projection_reproduced": True,
        "herb_output_sha256": sha256(herb_output),
        "disease_output_sha256": sha256(disease_output),
    }


def build_tcm_pca64_feature_bundle(
    *,
    protocol_manifest: Path,
    source_bundle_manifest: Path,
    output_dir: Path,
) -> Path:
    """Build a new PCA64 bundle without mutating the source AE64 bundle."""

    protocol_manifest = protocol_manifest.resolve(strict=True)
    source_bundle_manifest = source_bundle_manifest.resolve(strict=True)
    source_root = source_bundle_manifest.parent
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise TCMPCA64BundleError(f"refusing to overwrite existing path: {output_dir}")
    if output_dir == source_root or output_dir.name.lower() == "final_bundle":
        raise TCMPCA64BundleError("PCA64 output cannot replace the AE final_bundle")
    lowered_name = output_dir.name.lower()
    if "pca64" not in lowered_name or "v1" not in lowered_name:
        raise TCMPCA64BundleError("PCA64 output directory name must include pca64 and v1")

    protocol = load_protocol_manifest(protocol_manifest)
    if protocol.dataset != "tcm-suite":
        raise TCMPCA64BundleError("PCA64 bundle requires a tcm-suite protocol")
    try:
        source = validate_tcm_feature_bundle(
            source_bundle_manifest,
            protocol=protocol,
        )
    except TCMFeatureBundleError as error:
        raise TCMPCA64BundleError(f"source AE64 bundle failed validation: {error}") from error
    source_manifest = source["manifest"]
    if source_manifest.get("schema") != TCM_FEATURE_BUNDLE_SCHEMA:
        raise TCMPCA64BundleError("source bundle is not the audited AE64 schema")
    evidence_paths = source["evidence_paths"]
    required_raw = {
        "herb": evidence_paths.get("herb_scibert_768d"),
        "disease": evidence_paths.get("disease_scibert_768d"),
    }
    if not all(isinstance(path, Path) for path in required_raw.values()):
        raise TCMPCA64BundleError("source bundle lacks the two audited 768-D matrices")
    for entity, path in required_raw.items():
        semantic = source_manifest["entities"][entity].get("semantic_evidence", {})
        if semantic.get("raw_768d_sha256") != sha256(path):
            raise TCMPCA64BundleError(
                f"source {entity} raw 768-D hash is not bound by semantic evidence"
            )

    herb_rows = len(protocol.entity_ids["herb"])
    disease_rows = len(protocol.entity_ids["disease"])
    herb_raw = _load_matrix(required_raw["herb"], herb_rows, PCA64_INPUT_DIMENSION)
    disease_raw = _load_matrix(
        required_raw["disease"], disease_rows, PCA64_INPUT_DIMENSION
    )
    fitted = fit_centered_shared_pca64(herb_raw, disease_raw)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        destinations: dict[str, Path] = {}
        for entity in ("compound", "protein"):
            destinations[entity] = _copy(
                source["entities"][entity]["matrix_path"],
                staging / FEATURE_FILENAMES[entity],
            )
        destinations["herb"] = staging / FEATURE_FILENAMES["herb"]
        destinations["disease"] = staging / FEATURE_FILENAMES["disease"]
        np.save(destinations["herb"], fitted["herb_scores"].astype(np.float32))
        np.save(destinations["disease"], fitted["disease_scores"].astype(np.float32))

        provenance_root = staging / "provenance"
        copied_source_manifest = _copy(
            source_bundle_manifest,
            provenance_root / "source_ae64_bundle_manifest.json",
        )
        copied_evidence: dict[str, Path] = {}
        for name, source_path in sorted(evidence_paths.items()):
            suffix = "".join(source_path.suffixes) or ".bin"
            copied_evidence[name] = _copy(
                source_path,
                provenance_root / "source_bundle_evidence" / f"{name}{suffix}",
            )

        copied_raw = {
            entity: copied_evidence[f"{entity}_scibert_768d"]
            for entity in ("herb", "disease")
        }
        model_path = provenance_root / "shared_pca64_model_v1.npz"
        np.savez(
            model_path,
            mean=fitted["mean"],
            components=fitted["components"],
            singular_values=fitted["singular_values"],
            explained_variance_ratio=fitted["explained_variance_ratio"],
        )
        joint_row_ids_path = provenance_root / "joint_semantic_row_ids.json"
        joint_row_payload = {
            "order": "all protocol herb rows, then all protocol disease rows",
            "herb_row_ids": list(protocol.entity_ids["herb"]),
            "disease_row_ids": list(protocol.entity_ids["disease"]),
            "herb_mapping_sha256": _protocol_mapping_sha256(protocol, "herb"),
            "disease_mapping_sha256": _protocol_mapping_sha256(protocol, "disease"),
        }
        joint_row_payload["joint_row_identity_sha256"] = _json_sha256(
            {
                "herb": joint_row_payload["herb_row_ids"],
                "disease": joint_row_payload["disease_row_ids"],
            }
        )
        _atomic_json(joint_row_ids_path, joint_row_payload)

        transform_audit = _audit_transform(
            herb_raw=copied_raw["herb"],
            disease_raw=copied_raw["disease"],
            herb_rows=herb_rows,
            disease_rows=disease_rows,
            model_path=model_path,
            herb_output=destinations["herb"],
            disease_output=destinations["disease"],
        )
        audit_path = provenance_root / "shared_pca64_audit_v1.json"
        _atomic_json(
            audit_path,
            {
                "schema": "astra-tcm-shared-pca64-audit/v1",
                "algorithm": PCA64_ALGORITHM,
                "input_order": "herb_then_disease",
                "input_dimension": PCA64_INPUT_DIMENSION,
                "output_dimension": PCA64_COMPONENTS,
                "whitening": False,
                "fit_uses_labels_or_graph_edges": False,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                **transform_audit,
            },
        )
        metadata_path = provenance_root / "shared_pca64_metadata_v1.json"
        _atomic_json(
            metadata_path,
            {
                "schema": "astra-tcm-shared-pca64-metadata/v1",
                "algorithm": PCA64_ALGORITHM,
                "svd": "numpy.linalg.svd(full_matrices=False), exact/non-randomized",
                "canonical_sign": (
                    "largest-absolute loading per component is positive; first index "
                    "wins an exact absolute-value tie"
                ),
                "whitening": False,
                "input_order": "herb_then_disease",
                "input_shape": [herb_rows + disease_rows, PCA64_INPUT_DIMENSION],
                "herb_input_sha256": sha256(copied_raw["herb"]),
                "disease_input_sha256": sha256(copied_raw["disease"]),
                "model_sha256": sha256(model_path),
                "joint_row_ids_sha256": sha256(joint_row_ids_path),
                "explained_variance_ratio_sum": float(
                    np.sum(fitted["explained_variance_ratio"])
                ),
            },
        )

        entity_records: dict[str, Any] = {}
        for entity, matrix_path in destinations.items():
            canonical = list(protocol.entity_ids[entity])
            mapping_sha = _protocol_mapping_sha256(protocol, entity)
            sidecar_path = feature_row_ids_path(matrix_path)
            source_sha = (
                sha256(copied_raw[entity])
                if entity in copied_raw
                else sha256(source["entities"][entity]["matrix_path"])
            )
            _atomic_json(
                sidecar_path,
                {
                    "schema": FEATURE_ROW_IDS_SCHEMA,
                    "entity": entity,
                    "id_space": "protocol_raw_id",
                    "mapping_sha256": mapping_sha,
                    "row_ids": canonical,
                    "identity_evidence": (
                        "joint_pca64_protocol_rows"
                        if entity in {"herb", "disease"}
                        else "copied_from_validated_ae64_bundle"
                    ),
                    "source_matrix_sha256": source_sha,
                },
            )
            entity_records[entity] = {
                "shape": [len(canonical), PCA64_COMPONENTS],
                "protocol_mapping_sha256": mapping_sha,
                "matrix": _artifact(matrix_path, root=staging),
                "row_ids_sidecar": _artifact(sidecar_path, root=staging),
                "semantic_condition": (
                    PCA64_SEMANTIC_CONDITION
                    if entity in {"herb", "disease"}
                    else None
                ),
            }

        transform_records = {
            "model": _artifact(model_path, root=staging),
            "metadata": _artifact(metadata_path, root=staging),
            "audit": _artifact(audit_path, root=staging),
            "joint_row_ids": _artifact(joint_row_ids_path, root=staging),
        }
        manifest = {
            "schema": PCA64_BUNDLE_SCHEMA,
            "dataset": "tcm-suite",
            "profile": PCA64_BUNDLE_PROFILE,
            "formal_eligible": True,
            "formal_ineligibility_reasons": [],
            "protocol": {
                "input_manifest_sha256": _json_sha256(protocol.input_manifest()),
                "manifest_sha256": sha256(protocol_manifest),
                "entity_mapping_sha256": {
                    entity: _protocol_mapping_sha256(protocol, entity)
                    for entity in FEATURE_FILENAMES
                },
            },
            "source_ae64_bundle": {
                "original_path": str(source_bundle_manifest),
                "original_sha256": sha256(source_bundle_manifest),
                "copied_manifest": _artifact(copied_source_manifest, root=staging),
            },
            "entities": entity_records,
            "transform": {
                "algorithm": PCA64_ALGORITHM,
                "components": PCA64_COMPONENTS,
                "input_dimension": PCA64_INPUT_DIMENSION,
                "joint_fit_order": ["herb", "disease"],
                "centered": True,
                "whitening": False,
                **transform_records,
            },
            "source_evidence": {
                name: _artifact(path, root=staging)
                for name, path in copied_evidence.items()
            },
            "notes": {
                "scope": "target-neutral feature-basis control; no labels or graph edges used",
                "immutability": "separate v1 path; source AE64 final_bundle is never modified",
            },
        }
        _atomic_json(staging / "bundle_manifest.json", manifest)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    manifest_path = output_dir / "bundle_manifest.json"
    validate_tcm_pca64_feature_bundle(
        manifest_path,
        protocol=protocol,
        expected_feature_paths={
            entity: output_dir / filename
            for entity, filename in FEATURE_FILENAMES.items()
        },
        verify_transform=True,
    )
    return manifest_path


def validate_tcm_pca64_feature_bundle(
    manifest_path: Path,
    *,
    protocol: BenchmarkProtocol,
    expected_feature_paths: Mapping[str, Path] | None = None,
    verify_transform: bool = False,
) -> dict[str, Any]:
    """Recursively validate a PCA64 bundle and optionally replay its transform."""

    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != PCA64_BUNDLE_SCHEMA
        or manifest.get("profile") != PCA64_BUNDLE_PROFILE
        or manifest.get("dataset") != "tcm-suite"
        or protocol.dataset != "tcm-suite"
        or manifest.get("formal_eligible") is not True
    ):
        raise TCMPCA64BundleError("unsupported PCA64 bundle identity")
    protocol_record = manifest.get("protocol")
    if (
        not isinstance(protocol_record, Mapping)
        or protocol_record.get("input_manifest_sha256")
        != _json_sha256(protocol.input_manifest())
    ):
        raise TCMPCA64BundleError("PCA64 bundle targets another protocol")

    entities = manifest.get("entities")
    if not isinstance(entities, Mapping) or set(entities) != set(FEATURE_FILENAMES):
        raise TCMPCA64BundleError("PCA64 bundle must declare all four entities")
    validated_entities: dict[str, Any] = {}
    for entity, filename in FEATURE_FILENAMES.items():
        record = entities[entity]
        if not isinstance(record, Mapping):
            raise TCMPCA64BundleError(f"{entity} entity record is invalid")
        mapping_sha = _protocol_mapping_sha256(protocol, entity)
        if (
            record.get("protocol_mapping_sha256") != mapping_sha
            or protocol_record.get("entity_mapping_sha256", {}).get(entity)
            != mapping_sha
        ):
            raise TCMPCA64BundleError(f"{entity} mapping hash differs")
        matrix = _resolve_artifact(root, record.get("matrix"), label=f"{entity} matrix")
        sidecar = _resolve_artifact(
            root, record.get("row_ids_sidecar"), label=f"{entity} row IDs"
        )
        if matrix.name != filename or sidecar != feature_row_ids_path(matrix).resolve():
            raise TCMPCA64BundleError(f"{entity} canonical filename differs")
        if expected_feature_paths is not None and matrix != Path(
            expected_feature_paths[entity]
        ).resolve():
            raise TCMPCA64BundleError(f"{entity} selected feature path differs")
        _load_matrix(matrix, len(protocol.entity_ids[entity]), PCA64_COMPONENTS)
        sidecar_payload = _load_json(sidecar)
        if (
            sidecar_payload.get("schema") != FEATURE_ROW_IDS_SCHEMA
            or sidecar_payload.get("entity") != entity
            or sidecar_payload.get("mapping_sha256") != mapping_sha
            or sidecar_payload.get("row_ids") != list(protocol.entity_ids[entity])
        ):
            raise TCMPCA64BundleError(f"{entity} row-ID sidecar differs")
        if entity in {"herb", "disease"} and record.get(
            "semantic_condition"
        ) != PCA64_SEMANTIC_CONDITION:
            raise TCMPCA64BundleError(f"{entity} semantic condition is not PCA64")
        validated_entities[entity] = {
            "matrix_path": matrix,
            "sidecar_path": sidecar,
            "matrix_sha256": sha256(matrix),
            "sidecar_sha256": sha256(sidecar),
        }

    transform = manifest.get("transform")
    if (
        not isinstance(transform, Mapping)
        or transform.get("algorithm") != PCA64_ALGORITHM
        or transform.get("components") != PCA64_COMPONENTS
        or transform.get("input_dimension") != PCA64_INPUT_DIMENSION
        or transform.get("joint_fit_order") != ["herb", "disease"]
        or transform.get("centered") is not True
        or transform.get("whitening") is not False
    ):
        raise TCMPCA64BundleError("PCA64 transform contract differs")
    transform_paths = {
        name: _resolve_artifact(root, transform.get(name), label=f"transform/{name}")
        for name in ("model", "metadata", "audit", "joint_row_ids")
    }
    source_evidence = manifest.get("source_evidence")
    if not isinstance(source_evidence, Mapping) or not source_evidence:
        raise TCMPCA64BundleError("PCA64 source evidence is absent")
    evidence_paths = {
        str(name): _resolve_artifact(root, record, label=f"source/{name}")
        for name, record in source_evidence.items()
    }
    copied_source = _resolve_artifact(
        root,
        manifest.get("source_ae64_bundle", {}).get("copied_manifest"),
        label="source AE64 manifest",
    )
    source_record = manifest["source_ae64_bundle"]
    if sha256(copied_source) != source_record.get("original_sha256"):
        raise TCMPCA64BundleError("copied AE64 manifest hash differs from its source")
    copied_source_payload = _load_json(copied_source)
    if (
        copied_source_payload.get("schema") != TCM_FEATURE_BUNDLE_SCHEMA
        or copied_source_payload.get("dataset") != "tcm-suite"
        or copied_source_payload.get("protocol", {}).get("input_manifest_sha256")
        != _json_sha256(protocol.input_manifest())
    ):
        raise TCMPCA64BundleError("copied source is not the bound AE64 manifest")
    metadata = _load_json(transform_paths["metadata"])
    audit = _load_json(transform_paths["audit"])
    joint_rows = _load_json(transform_paths["joint_row_ids"])
    try:
        herb_raw_evidence = evidence_paths["herb_scibert_768d"]
        disease_raw_evidence = evidence_paths["disease_scibert_768d"]
    except KeyError as error:
        raise TCMPCA64BundleError("PCA64 raw semantic evidence is absent") from error
    if (
        metadata.get("schema") != "astra-tcm-shared-pca64-metadata/v1"
        or metadata.get("algorithm") != PCA64_ALGORITHM
        or metadata.get("whitening") is not False
        or metadata.get("model_sha256") != sha256(transform_paths["model"])
        or metadata.get("joint_row_ids_sha256")
        != sha256(transform_paths["joint_row_ids"])
        or metadata.get("herb_input_sha256") != sha256(herb_raw_evidence)
        or metadata.get("disease_input_sha256") != sha256(disease_raw_evidence)
        or audit.get("schema") != "astra-tcm-shared-pca64-audit/v1"
        or audit.get("algorithm") != PCA64_ALGORITHM
        or audit.get("status") != "verified"
        or audit.get("exact_top64_svd_recomputed") is not True
        or audit.get("unwhitened_projection_reproduced") is not True
        or audit.get("herb_output_sha256")
        != validated_entities["herb"]["matrix_sha256"]
        or audit.get("disease_output_sha256")
        != validated_entities["disease"]["matrix_sha256"]
        or joint_rows.get("herb_row_ids") != list(protocol.entity_ids["herb"])
        or joint_rows.get("disease_row_ids")
        != list(protocol.entity_ids["disease"])
        or joint_rows.get("herb_mapping_sha256")
        != _protocol_mapping_sha256(protocol, "herb")
        or joint_rows.get("disease_mapping_sha256")
        != _protocol_mapping_sha256(protocol, "disease")
        or joint_rows.get("joint_row_identity_sha256")
        != _json_sha256(
            {
                "herb": list(protocol.entity_ids["herb"]),
                "disease": list(protocol.entity_ids["disease"]),
            }
        )
    ):
        raise TCMPCA64BundleError("PCA64 metadata or transform audit differs")

    if verify_transform:
        replay = _audit_transform(
            herb_raw=herb_raw_evidence,
            disease_raw=disease_raw_evidence,
            herb_rows=len(protocol.entity_ids["herb"]),
            disease_rows=len(protocol.entity_ids["disease"]),
            model_path=transform_paths["model"],
            herb_output=validated_entities["herb"]["matrix_path"],
            disease_output=validated_entities["disease"]["matrix_path"],
        )
        for key, value in replay.items():
            if audit.get(key) != value:
                raise TCMPCA64BundleError(f"saved PCA64 audit differs at {key}")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "entities": validated_entities,
        "transform_paths": transform_paths,
        "evidence_paths": evidence_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="audit --output-dir instead of building it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.audit_existing:
            protocol = load_protocol_manifest(args.protocol_manifest.resolve(strict=True))
            path = args.output_dir.resolve(strict=True) / "bundle_manifest.json"
            validate_tcm_pca64_feature_bundle(
                path,
                protocol=protocol,
                verify_transform=True,
            )
        else:
            if args.source_bundle_manifest is None:
                raise TCMPCA64BundleError(
                    "--source-bundle-manifest is required when building"
                )
            path = build_tcm_pca64_feature_bundle(
                protocol_manifest=args.protocol_manifest,
                source_bundle_manifest=args.source_bundle_manifest,
                output_dir=args.output_dir,
            )
    except Exception as error:
        print(f"TCM PCA64 bundle failed: {error}")
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
