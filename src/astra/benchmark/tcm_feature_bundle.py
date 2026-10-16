#!/usr/bin/env python3
"""Build and validate the identity-bound TCM-suite HerbHG feature bundle.

The historical TCM matrices are useful only when their row identities can be
proved.  This module does not infer identities from matrix shape.  Instead, it
requires the mapping artifacts emitted by the biological extractor and the
ID-keyed metadata emitted by the controlled semantic pipeline, verifies both
against a frozen :class:`~astra.benchmark.protocol.BenchmarkProtocol`, and only
then writes row-ID sidecars and a recursively hashed bundle manifest.

The validator is intentionally smaller than the Ethnobotany bundle validator:
TCM has no missing-row masks or retrieval caches.  It nevertheless re-hashes
every matrix, sidecar, and bundled evidence artifact and binds all four entity
orders to the protocol mapping digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path
from .protocol import BenchmarkProtocol, load_protocol_manifest


TCM_FEATURE_BUNDLE_SCHEMA = "astra-tcm-feature-bundle/v1"
TCM_FEATURE_BUNDLE_PROFILE = "herbhg-four-entity-target-neutral"
TCM_SEMANTIC_CONDITION = "llm_clean_target_neutral_qwen_scibert_shared_ae64"

FEATURE_FILENAMES = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
    "herb": "herb_semantic_embeddings_64d.npy",
    "disease": "disease_semantic_embeddings_64d.npy",
}


class TCMFeatureBundleError(ValueError):
    """Raised when TCM feature identity or provenance is not demonstrable."""


def _sha256(path: Path) -> str:
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


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TCMFeatureBundleError(
            f"bundle artifact is outside the bundle root: {resolved}"
        ) from error
    return {
        "relative_path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


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
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _copy(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise TCMFeatureBundleError(f"required source artifact is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise TCMFeatureBundleError(f"copied artifact hash changed: {source}")
    return destination


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TCMFeatureBundleError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TCMFeatureBundleError(f"JSON artifact must contain an object: {path}")
    return payload


def _load_matrix(path: Path, expected_rows: int, expected_columns: int) -> np.ndarray:
    try:
        matrix = (
            np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npy"
            else np.loadtxt(path)
        )
    except (OSError, ValueError) as error:
        raise TCMFeatureBundleError(f"cannot read feature matrix {path}: {error}") from error
    matrix = np.asarray(matrix)
    if matrix.shape != (expected_rows, expected_columns):
        raise TCMFeatureBundleError(
            f"{path} has shape {matrix.shape}; expected "
            f"({expected_rows}, {expected_columns})"
        )
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise TCMFeatureBundleError(f"{path} must contain only finite numeric values")
    if (
        np.unique(matrix, axis=0).shape[0] < 2
        or np.count_nonzero(np.ptp(matrix, axis=0) > 0.0) == 0
    ):
        raise TCMFeatureBundleError(f"{path} is a degenerate feature matrix")
    return matrix


def _protocol_mapping_sha256(protocol: BenchmarkProtocol, entity: str) -> str:
    return str(protocol.input_manifest()["mappings"][entity]["sha256"])


def _parse_index_mapping(path: Path) -> tuple[str, ...]:
    """Read the extractor's ``zero-based-index<TAB>raw-id`` evidence file."""

    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: expected index and raw ID"
                )
            try:
                index = int(parts[0])
            except ValueError as error:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: index is not an integer"
                ) from error
            raw_id = parts[1].strip()
            if index != len(rows) or not raw_id:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: mapping must use consecutive zero-based indices"
                )
            rows.append(raw_id)
    if not rows or len(rows) != len(set(rows)):
        raise TCMFeatureBundleError(f"{path} is empty or contains duplicate raw IDs")
    return tuple(rows)


def _parse_csv_mapping(path: Path, id_column: str) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
            raise TCMFeatureBundleError(f"{path} has no {id_column!r} column")
        rows = tuple(str(row[id_column]).strip() for row in reader)
    if not rows or any(not value for value in rows) or len(rows) != len(set(rows)):
        raise TCMFeatureBundleError(f"{path} has empty or duplicate {id_column} values")
    return rows


def _first_occurrence_column(path: Path, column: int) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= column:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: relation row has no column {column}"
                )
            value = parts[column].strip()
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return tuple(ordered)


def _verify_corpus(path: Path, entity: str, canonical_ids: Sequence[str]) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: malformed JSON"
                ) from error
            raw_id = str(record.get("entity_id", "")).strip()
            if not raw_id or raw_id in observed:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: missing or duplicate entity_id {raw_id!r}"
                )
            if record.get("entity") != entity:
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: wrong entity declaration"
                )
            if record.get("variant") != "clean_no_cross_association":
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: semantic variant is not target-neutral"
                )
            if not str(record.get("description", "")).strip():
                raise TCMFeatureBundleError(
                    f"{path}:{line_number}: description is empty"
                )
            observed[raw_id] = record
    unexpected = sorted(set(observed) - set(canonical_ids))
    if unexpected:
        raise TCMFeatureBundleError(
            f"{path} contains IDs outside the protocol mapping: {unexpected[:5]}"
        )
    return {
        "rows": len(observed),
        "missing_row_ids": [value for value in canonical_ids if value not in observed],
    }


def _assert_hash_field(record: Mapping[str, Any], key: str, path: Path) -> None:
    if record.get(key) != _sha256(path):
        raise TCMFeatureBundleError(
            f"declared {key} does not match {path}"
        )


def _verify_semantic_entity(
    *,
    entity: str,
    protocol: BenchmarkProtocol,
    matrix64: Path,
    matrix768: Path,
    metadata_path: Path,
    mapping_path: Path,
    corpus_path: Path,
    relation_path: Path,
    reducer_status: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = tuple(protocol.entity_ids[entity])
    id_column = "herb_id" if entity == "herb" else "disease_id"
    if _parse_csv_mapping(mapping_path, id_column) != canonical:
        raise TCMFeatureBundleError(
            f"{mapping_path} row order does not equal the frozen {entity} mapping"
        )
    relation_ids = _first_occurrence_column(relation_path, 0 if entity == "herb" else 1)
    if relation_ids != canonical:
        raise TCMFeatureBundleError(
            f"{relation_path} first-occurrence order does not equal the frozen "
            f"{entity} mapping"
        )
    _load_matrix(matrix64, len(canonical), 64)
    _load_matrix(matrix768, len(canonical), 768)
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "success" or metadata.get("entity") != entity:
        raise TCMFeatureBundleError(f"semantic metadata is not successful {entity}: {metadata_path}")
    if metadata.get("alignment") != "first-occurrence node order from node_file":
        raise TCMFeatureBundleError(f"unexpected semantic alignment declaration: {metadata_path}")
    if metadata.get("mapping_sha256") != _sha256(mapping_path):
        raise TCMFeatureBundleError(f"semantic metadata mapping hash mismatch: {metadata_path}")
    if metadata.get("node_file_sha256") != _sha256(relation_path):
        raise TCMFeatureBundleError(f"semantic metadata relation hash mismatch: {metadata_path}")
    if metadata.get("source_texts_sha256") != _sha256(corpus_path):
        raise TCMFeatureBundleError(f"semantic metadata corpus hash mismatch: {metadata_path}")
    if metadata.get("output_sha256") != _sha256(matrix768):
        raise TCMFeatureBundleError(f"semantic metadata 768-D hash mismatch: {metadata_path}")
    corpus = _verify_corpus(corpus_path, entity, canonical)
    if int(metadata.get("present_descriptions", -1)) != corpus["rows"]:
        raise TCMFeatureBundleError(f"semantic metadata present-row count mismatch: {metadata_path}")
    if list(metadata.get("missing_ids", [])) != corpus["missing_row_ids"]:
        raise TCMFeatureBundleError(f"semantic metadata missing IDs mismatch: {metadata_path}")
    candidates = [
        record
        for record in reducer_status.get("outputs", [])
        if record.get("condition") == "llm_clean" and record.get("entity") == entity
    ]
    if len(candidates) != 1:
        raise TCMFeatureBundleError(
            f"shared reducer status must contain one llm_clean/{entity} output"
        )
    reduced = candidates[0]
    if (
        reduced.get("input_sha256") != _sha256(matrix768)
        or reduced.get("output_sha256") != _sha256(matrix64)
        or reduced.get("shape") != [len(canonical), 64]
    ):
        raise TCMFeatureBundleError(
            f"shared reducer status does not bind the selected {entity} matrix"
        )
    return {
        "mapping_sha256": _sha256(mapping_path),
        "relation_sha256": _sha256(relation_path),
        "corpus_sha256": _sha256(corpus_path),
        "metadata_sha256": _sha256(metadata_path),
        "raw_768d_sha256": _sha256(matrix768),
        "present_descriptions": corpus["rows"],
        "missing_row_ids": corpus["missing_row_ids"],
        "pooling": metadata.get("pooling"),
        "encoder_model": metadata.get("model"),
    }


def build_tcm_feature_bundle(
    *,
    protocol_manifest: Path,
    data_dir: Path,
    semantic_root: Path,
    herb_mapping_csv: Path,
    disease_mapping_csv: Path,
    output_dir: Path,
) -> Path:
    """Verify source identity evidence and assemble a portable feature bundle."""

    protocol_manifest = protocol_manifest.resolve()
    data_dir = data_dir.resolve()
    semantic_root = semantic_root.resolve()
    herb_mapping_csv = herb_mapping_csv.resolve()
    disease_mapping_csv = disease_mapping_csv.resolve()
    output_dir = output_dir.resolve()
    protocol = load_protocol_manifest(protocol_manifest)
    if protocol.dataset != "tcm-suite":
        raise TCMFeatureBundleError("the TCM feature builder requires a tcm-suite protocol")

    source_matrices = {
        "compound": data_dir / FEATURE_FILENAMES["compound"],
        "protein": data_dir / FEATURE_FILENAMES["protein"],
        "herb": semantic_root / "embeddings_64d/llm_clean/herb_semantic_embeddings_64d.npy",
        "disease": semantic_root / "embeddings_64d/llm_clean/disease_semantic_embeddings_64d.npy",
    }
    biological_maps = {
        "compound": data_dir / "compound_id_mapping.txt",
        "protein": data_dir / "protein_id_mapping.txt",
    }
    for entity in ("compound", "protein"):
        observed = _parse_index_mapping(biological_maps[entity])
        canonical = tuple(protocol.entity_ids[entity])
        if observed != canonical:
            raise TCMFeatureBundleError(
                f"{biological_maps[entity]} row order does not equal the frozen "
                f"{entity} mapping"
            )
        _load_matrix(source_matrices[entity], len(canonical), 64)

    reducer_status_path = semantic_root / "shared_reducer/status.json"
    reducer_checkpoint_path = semantic_root / "shared_reducer/shared_reducer.pt"
    reducer_status = _load_json(reducer_status_path)
    if reducer_status.get("status") != "success":
        raise TCMFeatureBundleError("shared semantic reducer is not marked successful")
    _assert_hash_field(reducer_status, "checkpoint_sha256", reducer_checkpoint_path)

    semantic_specs = {
        "herb": {
            "mapping": herb_mapping_csv,
            "corpus": semantic_root / "corpora/clean_herb.jsonl",
            "raw": semantic_root / "embeddings_768d/llm_clean_herb.npy",
            "metadata": semantic_root / "embeddings_768d/llm_clean_herb.npy.metadata.json",
            "relation": data_dir / "H_C.txt",
        },
        "disease": {
            "mapping": disease_mapping_csv,
            "corpus": semantic_root / "corpora/clean_disease.jsonl",
            "raw": semantic_root / "embeddings_768d/llm_clean_disease.npy",
            "metadata": semantic_root / "embeddings_768d/llm_clean_disease.npy.metadata.json",
            "relation": data_dir / "P_D.txt",
        },
    }
    semantic_evidence = {
        entity: _verify_semantic_entity(
            entity=entity,
            protocol=protocol,
            matrix64=source_matrices[entity],
            matrix768=spec["raw"],
            metadata_path=spec["metadata"],
            mapping_path=spec["mapping"],
            corpus_path=spec["corpus"],
            relation_path=spec["relation"],
            reducer_status=reducer_status,
        )
        for entity, spec in semantic_specs.items()
    }

    if output_dir.exists() and any(output_dir.iterdir()):
        raise TCMFeatureBundleError(
            f"refusing to overwrite non-empty feature bundle directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        entity: _copy(source, output_dir / FEATURE_FILENAMES[entity])
        for entity, source in source_matrices.items()
    }

    evidence_sources: dict[str, Path] = {
        "compound_extractor_mapping": biological_maps["compound"],
        "protein_extractor_mapping": biological_maps["protein"],
        "herb_source_mapping": herb_mapping_csv,
        "disease_source_mapping": disease_mapping_csv,
        "herb_clean_corpus": semantic_specs["herb"]["corpus"],
        "disease_clean_corpus": semantic_specs["disease"]["corpus"],
        "herb_scibert_768d": semantic_specs["herb"]["raw"],
        "disease_scibert_768d": semantic_specs["disease"]["raw"],
        "herb_scibert_metadata": semantic_specs["herb"]["metadata"],
        "disease_scibert_metadata": semantic_specs["disease"]["metadata"],
        "shared_reducer_status": reducer_status_path,
        "shared_reducer_checkpoint": reducer_checkpoint_path,
        "clean_herb_prompt": semantic_root / "prompts/clean_herb_no_disease.txt",
        "clean_disease_prompt": semantic_root / "prompts/clean_disease_no_herb.txt",
        "api_generation_status": semantic_root / "api_generation_status.json",
    }
    evidence_destinations: dict[str, Path] = {}
    for name, source in evidence_sources.items():
        suffix = "".join(source.suffixes) or ".bin"
        evidence_destinations[name] = _copy(
            source,
            output_dir / "provenance" / f"{name}{suffix}",
        )

    entity_records: dict[str, Any] = {}
    for entity, matrix_path in destinations.items():
        canonical = tuple(protocol.entity_ids[entity])
        mapping_sha256 = _protocol_mapping_sha256(protocol, entity)
        sidecar_path = feature_row_ids_path(matrix_path)
        source_evidence = (
            "extractor_index_mapping"
            if entity in {"compound", "protein"}
            else "id_keyed_corpus_and_first_occurrence_relation_order"
        )
        _atomic_json(
            sidecar_path,
            {
                "schema": FEATURE_ROW_IDS_SCHEMA,
                "entity": entity,
                "id_space": "protocol_raw_id",
                "mapping_sha256": mapping_sha256,
                "row_ids": list(canonical),
                "identity_evidence": source_evidence,
                "source_matrix_sha256": _sha256(source_matrices[entity]),
            },
        )
        entity_records[entity] = {
            "shape": [len(canonical), 64],
            "protocol_mapping_sha256": mapping_sha256,
            "matrix": _artifact(matrix_path, root=output_dir),
            "row_ids_sidecar": _artifact(sidecar_path, root=output_dir),
            "source_identity_evidence": source_evidence,
            "semantic_condition": (
                TCM_SEMANTIC_CONDITION if entity in {"herb", "disease"} else None
            ),
            "semantic_evidence": semantic_evidence.get(entity),
        }

    manifest = {
        "schema": TCM_FEATURE_BUNDLE_SCHEMA,
        "dataset": "tcm-suite",
        "profile": TCM_FEATURE_BUNDLE_PROFILE,
        "formal_eligible": True,
        "formal_ineligibility_reasons": [],
        "protocol": {
            "input_manifest_sha256": _json_sha256(protocol.input_manifest()),
            "manifest_sha256": _sha256(protocol_manifest),
            "entity_mapping_sha256": {
                entity: _protocol_mapping_sha256(protocol, entity)
                for entity in FEATURE_FILENAMES
            },
        },
        "entities": entity_records,
        "evidence": {
            name: _artifact(path, root=output_dir)
            for name, path in evidence_destinations.items()
        },
        "notes": {
            "biological_identity": (
                "compound/protein row IDs are taken from the extractor's "
                "sequential index mappings and exactly equal the frozen protocol order"
            ),
            "semantic_identity": (
                "herb/disease rows were encoded by explicit entity ID and materialised "
                "in the H_C/P_D first-occurrence order, which exactly equals the frozen "
                "protocol order"
            ),
            "semantic_scope": (
                "selected corpora declare clean_no_cross_association; the shared "
                "unsupervised reducer status is retained verbatim in provenance"
            ),
        },
    }
    manifest_path = output_dir / "bundle_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_tcm_feature_bundle(
        manifest_path,
        protocol=protocol,
        expected_feature_paths=destinations,
    )
    return manifest_path


def _resolve_artifact(
    root: Path, record: Mapping[str, Any], *, label: str
) -> Path:
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise TCMFeatureBundleError(f"{label} has no relative_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise TCMFeatureBundleError(f"{label} escapes the bundle root") from error
    if not path.is_file():
        raise TCMFeatureBundleError(f"{label} is absent: {path}")
    if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
        raise TCMFeatureBundleError(f"{label} bytes or SHA-256 do not match the manifest")
    return path


def validate_tcm_feature_bundle(
    manifest_path: Path,
    *,
    protocol: BenchmarkProtocol,
    expected_feature_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Recursively validate the compact TCM bundle against a frozen protocol."""

    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != TCM_FEATURE_BUNDLE_SCHEMA:
        raise TCMFeatureBundleError(
            f"TCM bundle schema must be {TCM_FEATURE_BUNDLE_SCHEMA!r}"
        )
    if manifest.get("dataset") != "tcm-suite" or protocol.dataset != "tcm-suite":
        raise TCMFeatureBundleError("TCM feature bundle/protocol dataset mismatch")
    if manifest.get("profile") != TCM_FEATURE_BUNDLE_PROFILE:
        raise TCMFeatureBundleError("TCM feature bundle has an unsupported profile")
    if manifest.get("formal_eligible") is not True:
        raise TCMFeatureBundleError("TCM feature bundle is not formally eligible")
    protocol_record = manifest.get("protocol")
    if not isinstance(protocol_record, dict):
        raise TCMFeatureBundleError("TCM feature bundle has no protocol binding")
    if protocol_record.get("input_manifest_sha256") != _json_sha256(protocol.input_manifest()):
        raise TCMFeatureBundleError("TCM feature bundle targets a different protocol input")

    entities = manifest.get("entities")
    if not isinstance(entities, dict) or set(entities) != set(FEATURE_FILENAMES):
        raise TCMFeatureBundleError("TCM feature bundle must declare all four entities")
    validated_entities: dict[str, Any] = {}
    for entity, filename in FEATURE_FILENAMES.items():
        record = entities[entity]
        if not isinstance(record, dict):
            raise TCMFeatureBundleError(f"TCM {entity} record is not an object")
        mapping_sha256 = _protocol_mapping_sha256(protocol, entity)
        if (
            record.get("protocol_mapping_sha256") != mapping_sha256
            or protocol_record.get("entity_mapping_sha256", {}).get(entity)
            != mapping_sha256
        ):
            raise TCMFeatureBundleError(f"TCM {entity} mapping hash mismatch")
        matrix = _resolve_artifact(root, record.get("matrix", {}), label=f"{entity} matrix")
        sidecar = _resolve_artifact(
            root,
            record.get("row_ids_sidecar", {}),
            label=f"{entity} row-ID sidecar",
        )
        if matrix.name != filename or sidecar != feature_row_ids_path(matrix).resolve():
            raise TCMFeatureBundleError(f"TCM {entity} bundle filenames are not canonical")
        if expected_feature_paths is not None:
            expected = Path(expected_feature_paths[entity]).resolve()
            if matrix != expected:
                raise TCMFeatureBundleError(
                    f"TCM {entity} manifest matrix differs from the selected feature path"
                )
        _load_matrix(matrix, len(protocol.entity_ids[entity]), 64)
        sidecar_payload = _load_json(sidecar)
        if (
            sidecar_payload.get("schema") != FEATURE_ROW_IDS_SCHEMA
            or sidecar_payload.get("entity") != entity
            or sidecar_payload.get("id_space") != "protocol_raw_id"
            or sidecar_payload.get("mapping_sha256") != mapping_sha256
            or sidecar_payload.get("row_ids") != list(protocol.entity_ids[entity])
        ):
            raise TCMFeatureBundleError(f"TCM {entity} row-ID sidecar is inconsistent")
        if entity in {"herb", "disease"} and record.get("semantic_condition") != TCM_SEMANTIC_CONDITION:
            raise TCMFeatureBundleError(f"TCM {entity} is not the target-neutral semantic condition")
        validated_entities[entity] = {
            "matrix_path": matrix,
            "sidecar_path": sidecar,
            "matrix_sha256": _sha256(matrix),
            "sidecar_sha256": _sha256(sidecar),
        }

    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise TCMFeatureBundleError("TCM feature bundle has no identity evidence")
    evidence_paths = {
        name: _resolve_artifact(root, record, label=f"evidence/{name}")
        for name, record in evidence.items()
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "entities": validated_entities,
        "evidence_paths": evidence_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--herb-mapping-csv", type=Path, required=True)
    parser.add_argument("--disease-mapping-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = build_tcm_feature_bundle(
        protocol_manifest=args.protocol_manifest,
        data_dir=args.data_dir,
        semantic_root=args.semantic_root,
        herb_mapping_csv=args.herb_mapping_csv,
        disease_mapping_csv=args.disease_mapping_csv,
        output_dir=args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
