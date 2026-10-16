#!/usr/bin/env python3
"""Reusable, path-bound deep-audit receipts for ASTRARefined features.

The Ethnobotany feature bundle validator intentionally walks its complete
portable provenance closure.  That is the right operation when a bundle is
first admitted, but repeating the recursive semantic/biological reconstruction
for every controlled candidate adds no evidence when all bytes are unchanged.

This module records the result of one full validation and the SHA-256 identity
of every unique dependency.  Reuse is fail-closed: every dependency and every
validator source is re-hashed, the receipt must be bound to the same protocol,
bundle, and semantic receipt, and only source-order-is-canonical feature paths
may be restored.  File mtimes are never trusted.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA = (
    "astra-ethnobotany-refined-feature-deep-audit-receipt/v1"
)
_SEMANTIC_DEEP_AUDIT_RECEIPT_SCHEMA = (
    "astra-ethnobotany-semantic-deep-audit-receipt/v1"
)
_ENTITIES = ("herb", "compound", "protein", "disease")
_BUNDLE_ENTITY_ARTIFACTS = (
    "matrix",
    "row_ids_sidecar",
    "missing_mask",
    "mapping",
    "component_manifest",
    "assembly_manifest",
    "reducer_checkpoint",
    "retrieval_cache",
)


class FeatureDeepAuditReceiptError(RuntimeError):
    """Raised when a feature receipt cannot be created or revalidated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeatureDeepAuditReceiptError(
            f"cannot read feature receipt dependency {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise FeatureDeepAuditReceiptError(f"JSON must contain an object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve(strict=False)
    if destination.exists():
        raise FeatureDeepAuditReceiptError(
            f"refusing to overwrite feature deep-audit receipt: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validator_source_paths() -> list[Path]:
    root = Path(__file__).resolve().parent
    names = (
        "ethnobotany_refined_feature_receipt.py",
        "ethnobotany_refined_runner.py",
        "astra_runner.py",
        "protocol.py",
        "features.py",
        "ethnobotany_feature_bundle.py",
        "ethnobotany_feature_extractors.py",
        "ethnobotany_qwen_semantics.py",
        "ethnobotany_semantic_features.py",
        "ethnobotany_semantic_controls.py",
    )
    paths = [(root / name).resolve(strict=True) for name in names]
    return paths


def _bundle_record_path(
    record: object, *, bundle_root: Path, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise FeatureDeepAuditReceiptError(f"{label} artifact record is invalid")
    raw_path = record.get("path")
    relative = record.get("relative_path")
    if not isinstance(raw_path, str) or not isinstance(relative, str):
        raise FeatureDeepAuditReceiptError(f"{label} artifact path is invalid")
    path = Path(raw_path).resolve(strict=True)
    try:
        observed_relative = path.relative_to(bundle_root)
    except ValueError as error:
        raise FeatureDeepAuditReceiptError(
            f"{label} artifact escapes the fixed feature bundle"
        ) from error
    if str(observed_relative) != relative:
        raise FeatureDeepAuditReceiptError(
            f"{label} bundle-relative path is inconsistent"
        )
    return path


def _bundle_dependency_paths(bundle_manifest: Path) -> list[Path]:
    manifest = bundle_manifest.resolve(strict=True)
    root = manifest.parent
    payload = _load_json(manifest)
    paths: list[Path] = [manifest]
    objects = payload.get("provenance_objects")
    if not isinstance(objects, list) or not objects:
        raise FeatureDeepAuditReceiptError(
            "feature bundle has no portable provenance closure"
        )
    for index, record in enumerate(objects):
        paths.append(
            _bundle_record_path(
                record, bundle_root=root, label=f"provenance object {index}"
            )
        )
    entities = payload.get("entities")
    if not isinstance(entities, Mapping) or set(entities) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError(
            "feature bundle does not contain the four required entities"
        )
    for entity in _ENTITIES:
        entity_record = entities.get(entity)
        if not isinstance(entity_record, Mapping):
            raise FeatureDeepAuditReceiptError(
                f"feature bundle {entity} record is invalid"
            )
        for name in _BUNDLE_ENTITY_ARTIFACTS:
            artifact = entity_record.get(name)
            if artifact is None and name == "retrieval_cache":
                continue
            paths.append(
                _bundle_record_path(
                    artifact,
                    bundle_root=root,
                    label=f"bundle {entity} {name}",
                )
            )
    return paths


def _protocol_source_paths(
    protocol: object, *, data_dir: Path, embedding_dir: Path
) -> list[Path]:
    source_files = getattr(protocol, "source_files", None)
    if not isinstance(source_files, Mapping):
        raise FeatureDeepAuditReceiptError("protocol has no frozen source files")
    paths: list[Path] = []
    for name, record in source_files.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise FeatureDeepAuditReceiptError("protocol source record is invalid")
        recorded = Path(str(record.get("path", "")))
        candidate = recorded
        if name in {"H_D", "H_C", "C_P", "P_D"} or name.startswith("mapping/"):
            candidate = data_dir / recorded.name
        elif name.startswith("embedding/"):
            candidate = embedding_dir / recorded.name
        paths.append(candidate.resolve(strict=True))
    return paths


def _semantic_receipt_dependency_paths(receipt_path: Path) -> list[Path]:
    receipt = receipt_path.resolve(strict=True)
    payload = _load_json(receipt)
    if (
        payload.get("schema") != _SEMANTIC_DEEP_AUDIT_RECEIPT_SCHEMA
        or payload.get("deep_audit") is not True
    ):
        raise FeatureDeepAuditReceiptError(
            "semantic deep-audit receipt contract is invalid"
        )
    dependencies = payload.get("dependencies")
    validator_sources = payload.get("validator_sources")
    if not isinstance(dependencies, list) or not dependencies:
        raise FeatureDeepAuditReceiptError(
            "semantic deep-audit receipt has no dependencies"
        )
    if not isinstance(validator_sources, list) or not validator_sources:
        raise FeatureDeepAuditReceiptError(
            "semantic deep-audit receipt has no validator sources"
        )
    paths = [receipt]
    for label, records in (
        ("semantic dependency", dependencies),
        ("semantic validator source", validator_sources),
    ):
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(
                record.get("path"), str
            ):
                raise FeatureDeepAuditReceiptError(f"{label} record is invalid")
            paths.append(Path(record["path"]).resolve(strict=True))
    return paths


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted(
        {path.resolve(strict=True) for path in paths}, key=lambda item: str(item)
    )


def _canonical_feature_records(prepared: object) -> dict[str, Any]:
    paths = getattr(prepared, "paths", None)
    diagnostics = getattr(prepared, "diagnostics", None)
    if not isinstance(paths, Mapping) or set(paths) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError(
            "deep feature audit did not return four selected feature paths"
        )
    if not isinstance(diagnostics, Mapping):
        raise FeatureDeepAuditReceiptError(
            "deep feature audit did not return diagnostics"
        )
    selected = diagnostics.get("selected_features")
    if not isinstance(selected, Mapping) or set(selected) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError(
            "deep feature audit omitted selected-feature diagnostics"
        )
    output: dict[str, Any] = {}
    for entity in _ENTITIES:
        details = selected.get(entity)
        if not isinstance(details, Mapping):
            raise FeatureDeepAuditReceiptError(
                f"deep feature audit omitted {entity} diagnostics"
            )
        selected_path = Path(paths[entity]).resolve(strict=True)
        source_path = Path(str(details.get("path", ""))).resolve(strict=True)
        used_path = Path(str(details.get("used_path", ""))).resolve(strict=True)
        if (
            details.get("source_order_is_canonical") is not True
            or selected_path != source_path
            or selected_path != used_path
        ):
            raise FeatureDeepAuditReceiptError(
                "feature receipts only reuse original source-order-is-canonical "
                f"paths; {entity} required materialised alignment"
            )
        sidecar_path = Path(str(details.get("sidecar_path", ""))).resolve(
            strict=True
        )
        output[entity] = {
            "matrix": _file_record(selected_path),
            "row_ids_sidecar": _file_record(sidecar_path),
            "shape": details.get("shape"),
            "dtype": details.get("dtype"),
            "canonical_array_sha256": details.get("canonical_array_sha256"),
            "source_order_is_canonical": True,
        }
    return output


def _mask_records(prepared: object) -> dict[str, Any]:
    paths = getattr(prepared, "mask_paths", None)
    if not isinstance(paths, Mapping) or set(paths) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError(
            "deep feature audit did not return all missing-mask paths"
        )
    return {entity: _file_record(Path(paths[entity])) for entity in _ENTITIES}


def create_feature_deep_audit_receipt(
    *,
    receipt_path: Path | str,
    protocol_manifest: Path | str,
    protocol: object,
    data_dir: Path | str,
    embedding_dir: Path | str,
    semantic_reduction_manifest: Path | str,
    semantic_deep_audit_receipt: Path | str,
    prepared: object,
) -> Path:
    """Attest one completed recursive audit and all unique trusted bytes."""

    destination = Path(receipt_path).resolve(strict=False)
    if destination.exists():
        raise FeatureDeepAuditReceiptError(
            f"refusing to overwrite feature deep-audit receipt: {destination}"
        )
    protocol_path = Path(protocol_manifest).resolve(strict=True)
    data_root = Path(data_dir).resolve(strict=True)
    embedding_root = Path(embedding_dir).resolve(strict=True)
    bundle_manifest = (embedding_root / "bundle_manifest.json").resolve(strict=True)
    semantic_manifest = Path(semantic_reduction_manifest).resolve(strict=True)
    semantic_receipt = Path(semantic_deep_audit_receipt).resolve(strict=True)
    canonical = _canonical_feature_records(prepared)
    masks = _mask_records(prepared)
    diagnostics = getattr(prepared, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise FeatureDeepAuditReceiptError("deep feature diagnostics are invalid")

    dependency_paths = _unique_paths(
        [protocol_path, semantic_manifest]
        + _protocol_source_paths(
            protocol, data_dir=data_root, embedding_dir=embedding_root
        )
        + _bundle_dependency_paths(bundle_manifest)
        + _semantic_receipt_dependency_paths(semantic_receipt)
        + [
            Path(record[key]["path"])
            for record in canonical.values()
            for key in ("matrix", "row_ids_sidecar")
        ]
        + [Path(record["path"]) for record in masks.values()]
    )
    if destination in dependency_paths:
        raise FeatureDeepAuditReceiptError(
            "feature receipt path collides with an audited dependency"
        )
    payload = {
        "schema": FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "ethnobotany",
        "deep_audit": True,
        "path_bound": True,
        "all_dependencies_rehashed_on_reuse": True,
        "protocol": _file_record(protocol_path),
        "data_dir": str(data_root),
        "embedding_dir": str(embedding_root),
        "bundle_manifest": _file_record(bundle_manifest),
        "semantic_reduction_manifest": _file_record(semantic_manifest),
        "semantic_deep_audit_receipt": _file_record(semantic_receipt),
        "validator_sources": [
            _file_record(path) for path in _validator_source_paths()
        ],
        "dependencies": [_file_record(path) for path in dependency_paths],
        "canonical_features": canonical,
        "missing_masks": masks,
        "diagnostics": copy.deepcopy(dict(diagnostics)),
    }
    _atomic_json(destination, payload)
    return destination


def validate_feature_deep_audit_receipt(
    *,
    receipt_path: Path | str,
    protocol_manifest: Path | str,
    protocol: object,
    data_dir: Path | str,
    embedding_dir: Path | str,
    semantic_reduction_manifest: Path | str,
    semantic_deep_audit_receipt: Path | str,
) -> dict[str, Any]:
    """Re-hash a prior deep audit and restore its canonical feature paths."""

    receipt = Path(receipt_path).resolve(strict=True)
    payload = _load_json(receipt)
    if (
        payload.get("schema") != FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA
        or payload.get("dataset") != "ethnobotany"
        or payload.get("deep_audit") is not True
        or payload.get("path_bound") is not True
        or payload.get("all_dependencies_rehashed_on_reuse") is not True
    ):
        raise FeatureDeepAuditReceiptError(
            "feature deep-audit receipt contract is invalid"
        )

    protocol_path = Path(protocol_manifest).resolve(strict=True)
    data_root = Path(data_dir).resolve(strict=True)
    embedding_root = Path(embedding_dir).resolve(strict=True)
    bundle_manifest = (embedding_root / "bundle_manifest.json").resolve(strict=True)
    semantic_manifest = Path(semantic_reduction_manifest).resolve(strict=True)
    semantic_receipt = Path(semantic_deep_audit_receipt).resolve(strict=True)
    if payload.get("data_dir") != str(data_root) or payload.get(
        "embedding_dir"
    ) != str(embedding_root):
        raise FeatureDeepAuditReceiptError(
            "feature receipt is bound to different data/embedding directories"
        )

    verified: dict[Path, dict[str, Any]] = {}

    def verify_record(record: object, label: str) -> Path:
        if not isinstance(record, Mapping):
            raise FeatureDeepAuditReceiptError(f"{label} receipt record is invalid")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file():
            raise FeatureDeepAuditReceiptError(f"{label} is absent: {path}")
        prior = verified.get(path)
        if prior is None:
            observed = _file_record(path)
            verified[path] = observed
        else:
            observed = prior
        if (
            type(record.get("bytes")) is not int
            or record.get("bytes") != observed["bytes"]
            or not isinstance(record.get("sha256"), str)
            or record.get("sha256") != observed["sha256"]
        ):
            raise FeatureDeepAuditReceiptError(
                f"{label} differs from its feature deep-audit receipt"
            )
        return path

    identity = (
        ("protocol", protocol_path),
        ("bundle_manifest", bundle_manifest),
        ("semantic_reduction_manifest", semantic_manifest),
        ("semantic_deep_audit_receipt", semantic_receipt),
    )
    for key, expected in identity:
        observed = verify_record(payload.get(key), key.replace("_", " "))
        if observed != expected:
            raise FeatureDeepAuditReceiptError(
                f"feature receipt {key} path differs from the requested input"
            )

    source_records = payload.get("validator_sources")
    if not isinstance(source_records, list):
        raise FeatureDeepAuditReceiptError(
            "feature receipt validator-source records are absent"
        )
    observed_sources = [
        verify_record(record, "feature validator source")
        for record in source_records
    ]
    expected_sources = _validator_source_paths()
    if observed_sources != expected_sources:
        raise FeatureDeepAuditReceiptError(
            "feature validator source set differs from the receipt"
        )

    dependency_records = payload.get("dependencies")
    if not isinstance(dependency_records, list) or not dependency_records:
        raise FeatureDeepAuditReceiptError(
            "feature receipt dependency records are absent"
        )
    observed_dependencies = [
        verify_record(record, "feature dependency")
        for record in dependency_records
    ]
    if observed_dependencies != _unique_paths(observed_dependencies):
        raise FeatureDeepAuditReceiptError(
            "feature receipt dependency set is duplicated or unsorted"
        )
    expected_dependencies = _unique_paths(
        [protocol_path, semantic_manifest]
        + _protocol_source_paths(
            protocol, data_dir=data_root, embedding_dir=embedding_root
        )
        + _bundle_dependency_paths(bundle_manifest)
        + _semantic_receipt_dependency_paths(semantic_receipt)
    )

    canonical = payload.get("canonical_features")
    masks = payload.get("missing_masks")
    if not isinstance(canonical, Mapping) or set(canonical) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError(
            "feature receipt canonical paths are incomplete"
        )
    if not isinstance(masks, Mapping) or set(masks) != set(_ENTITIES):
        raise FeatureDeepAuditReceiptError("feature receipt masks are incomplete")
    feature_paths: dict[str, Path] = {}
    mask_paths: dict[str, Path] = {}
    extra_dependencies: list[Path] = []
    for entity in _ENTITIES:
        record = canonical[entity]
        if (
            not isinstance(record, Mapping)
            or record.get("source_order_is_canonical") is not True
        ):
            raise FeatureDeepAuditReceiptError(
                f"feature receipt {entity} path is not canonical"
            )
        matrix_path = verify_record(
            record.get("matrix"), f"canonical {entity} feature"
        )
        sidecar_path = verify_record(
            record.get("row_ids_sidecar"), f"canonical {entity} row IDs"
        )
        mask_path = verify_record(masks[entity], f"canonical {entity} mask")
        feature_paths[entity] = matrix_path
        mask_paths[entity] = mask_path
        extra_dependencies.extend((matrix_path, sidecar_path, mask_path))
    expected_dependencies = _unique_paths(expected_dependencies + extra_dependencies)
    if observed_dependencies != expected_dependencies:
        raise FeatureDeepAuditReceiptError(
            "feature receipt dependency closure differs from current manifests"
        )

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise FeatureDeepAuditReceiptError(
            "feature receipt has no preserved deep-audit diagnostics"
        )
    restored = copy.deepcopy(dict(diagnostics))
    selected = restored.get("selected_features")
    if not isinstance(selected, Mapping):
        raise FeatureDeepAuditReceiptError(
            "preserved selected-feature diagnostics are absent"
        )
    for entity in _ENTITIES:
        details = selected.get(entity)
        if (
            not isinstance(details, Mapping)
            or details.get("source_order_is_canonical") is not True
            or Path(str(details.get("used_path", ""))).resolve()
            != feature_paths[entity]
            or details.get("used_sha256")
            != verified[feature_paths[entity]]["sha256"]
        ):
            raise FeatureDeepAuditReceiptError(
                f"preserved {entity} canonical diagnostics are inconsistent"
            )
    restored["feature_deep_audit_receipt"] = {
        "path": str(receipt),
        "sha256": _sha256(receipt),
        "schema": FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA,
        "deep_audit": True,
        "fast_revalidation": True,
        "all_dependencies_rehashed": True,
        "dependency_count": len(observed_dependencies),
        "canonical_paths_reused": True,
    }
    return {
        "paths": feature_paths,
        "mask_paths": mask_paths,
        "diagnostics": restored,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="run one full feature audit and write a reusable receipt"
    )
    create.add_argument("--protocol-manifest", type=Path, required=True)
    create.add_argument("--data-dir", type=Path, required=True)
    create.add_argument("--embedding-dir", type=Path, required=True)
    create.add_argument("--semantic-reduction-manifest", type=Path, required=True)
    create.add_argument("--semantic-deep-audit-receipt", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "create":  # pragma: no cover - argparse enforces this
        raise SystemExit(f"unsupported command: {args.command}")
    # Lazy imports avoid a module cycle: the runner imports the validator above.
    from .ethnobotany_refined_runner import prepare_candidate_features
    from .protocol import load_protocol_manifest

    protocol_path = args.protocol_manifest.resolve(strict=True)
    protocol = load_protocol_manifest(protocol_path)
    if protocol.dataset != "ethnobotany":
        raise SystemExit("feature receipts support only Ethnobotany")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="astra_feature_deep_audit_", dir=args.output.parent
        ) as temporary:
            prepared = prepare_candidate_features(
                protocol=protocol,
                data_dir=args.data_dir.resolve(strict=True),
                embedding_dir=args.embedding_dir.resolve(strict=True),
                semantic_reduction_manifest=args.semantic_reduction_manifest.resolve(
                    strict=True
                ),
                output_root=Path(temporary),
                semantic_deep_audit_receipt=args.semantic_deep_audit_receipt.resolve(
                    strict=True
                ),
            )
        receipt = create_feature_deep_audit_receipt(
            receipt_path=args.output,
            protocol_manifest=protocol_path,
            protocol=protocol,
            data_dir=args.data_dir,
            embedding_dir=args.embedding_dir,
            semantic_reduction_manifest=args.semantic_reduction_manifest,
            semantic_deep_audit_receipt=args.semantic_deep_audit_receipt,
            prepared=prepared,
        )
    except FeatureDeepAuditReceiptError as error:
        raise SystemExit(str(error)) from error
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
