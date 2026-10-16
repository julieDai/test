#!/usr/bin/env python3
"""Auditable regeneration of Ethnobotany herb/disease semantic features.

This module defines a new semantic condition.  It deliberately does not load,
copy, or attest the historical 64-dimensional arrays.  Entity identity is
established before inference by joining authoritative source IDs to a
generation-time text manifest.  SciBERT is then run in deterministic inference
mode and one centered-SVD/PCA reducer is fitted jointly to herb and disease
embeddings.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import numpy as np

from .ethnobotany_feature_bundle import (
    EntityMapping,
    FeatureBundleError,
    read_entity_mapping,
)
from .features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path


TEXT_MANIFEST_SCHEMA = "astra-ethnobotany-semantic-text/v1"
EMBEDDING_MANIFEST_SCHEMA = "astra-ethnobotany-scibert-embeddings/v1"
REDUCTION_MANIFEST_SCHEMA = "astra-ethnobotany-shared-semantic-pca/v1"
DEEP_AUDIT_RECEIPT_SCHEMA = "astra-ethnobotany-semantic-deep-audit-receipt/v1"
SEMANTIC_CONDITION = (
    "ethnobotany-id-keyed-names-mondo-v2026-09-01-scibert-cls-shared-pca64/v1"
)

MONDO_RELEASE = "v2026-09-01"
MONDO_RELEASE_COMMIT = "d50a5d1"
MONDO_BASE_JSON_URL = (
    "https://github.com/monarch-initiative/mondo/releases/download/"
    "v2026-09-01/mondo-base.json"
)
MONDO_BASE_JSON_SHA256 = (
    "c1ff23f548e330972b5e8d3bf95b701b66346d91bb531f05b13dda370e3716e9"
)

HERB_COMPONENT_SHA256 = (
    "7d24fd85adaa2541f07a8bab43cd117ecb5e1e7ffe7df4102d84a8f2cfd29075"
)
SCIBERT_MODEL_ID = "allenai/scibert_scivocab_uncased"
SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
SCIBERT_FILE_SHA256 = {
    "config.json": "488e7dbd68e27a38337ecb16358810da6df4226ac3b8880a0fde44bfdc37ccf0",
    "pytorch_model.bin": (
        "e492944d88ac97dee6baa547671d3c526a3d067676883efb058311f4e5882e1a"
    ),
    "vocab.txt": "f0a650346e51ede8996710f79ba65d83fdb8da05b159f17037b74ba4e3a36c6f",
}
SCIBERT_DIMENSION = 768
SCIBERT_MAX_LENGTH = 512
SEMANTIC_OUTPUT_DIMENSION = 64
FORMAL_SEMANTIC_OUTPUT_DIMENSIONS = (64, 128)

_MONDO_CURIE = re.compile(r"^MONDO:(\d+)$")


class SemanticFeatureError(FeatureBundleError):
    """Raised when the semantic generation contract cannot be proved."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SemanticFeatureError(f"required artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_artifact(
    record: object,
    label: str,
    *,
    resolver: Mapping[str, Path] | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise SemanticFeatureError(f"{label} artifact record is absent")
    recorded_path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise SemanticFeatureError(f"{label} artifact path is absent")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SemanticFeatureError(f"{label} artifact SHA-256 is malformed")
    if resolver is None:
        path = Path(recorded_path).resolve()
    else:
        path = resolver.get(digest, Path("/__missing_semantic_artifact__")).resolve()
    if not path.is_file():
        raise SemanticFeatureError(f"{label} is absent at {path}")
    if type(record.get("bytes")) is not int or record["bytes"] != path.stat().st_size:
        raise SemanticFeatureError(f"{label} size differs from its manifest")
    if digest != _sha256(path):
        raise SemanticFeatureError(f"{label} SHA-256 differs from its manifest")
    return path


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
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemanticFeatureError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SemanticFeatureError(f"JSON {path} must contain an object")
    return payload


def _normalise_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split()).strip()


def _require_sha(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise SemanticFeatureError(
            f"{label} SHA-256 {observed} != reviewed pin {expected}"
        )


def _read_plant_names(
    path: Path,
    *,
    id_column: str,
    name_column: str,
    source_key: str,
    candidates: dict[str, dict[str, dict[str, int]]],
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {id_column, name_column}.issubset(
                reader.fieldnames
            ):
                raise SemanticFeatureError(
                    f"{path} lacks required columns {id_column!r}/{name_column!r}"
                )
            for row in reader:
                source_id = _normalise_text(row.get(id_column))
                name = _normalise_text(row.get(name_column))
                if not source_id or not name:
                    continue
                per_name = candidates.setdefault(source_id, {}).setdefault(name, {})
                per_name[source_key] = per_name.get(source_key, 0) + 1
    except (OSError, csv.Error) as error:
        raise SemanticFeatureError(f"cannot parse plant-name source {path}: {error}") from error


def _rank_plant_candidate(
    item: tuple[str, Mapping[str, int]],
) -> tuple[int, str, str]:
    name, counts = item
    # Only the reviewed herb-component source participates.  Recorded support
    # wins, followed by a fully deterministic Unicode lexical tie-break.  All
    # candidates and counts remain in the manifest for independent re-derivation.
    return (-sum(counts.values()), name.casefold(), name)


def _derive_herb_rows(
    mapping: EntityMapping,
    herb_component_csv: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: dict[str, dict[str, dict[str, int]]] = {}
    _read_plant_names(
        herb_component_csv,
        id_column="plant_curie",
        name_column="plant name",
        source_key="herb-component.csv",
        candidates=candidates,
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row_id, source_id in zip(mapping.row_ids, mapping.source_ids):
        entity_candidates = candidates.get(source_id, {})
        if not entity_candidates:
            missing.append(source_id)
            continue
        ordered = sorted(entity_candidates.items(), key=_rank_plant_candidate)
        canonical = ordered[0][0]
        alternatives = sorted(
            (name for name, _ in ordered[1:]), key=lambda name: (name.casefold(), name)
        )
        text = f"Medicinal plant: {canonical}."
        if alternatives:
            text += " Other recorded names: " + "; ".join(alternatives) + "."
        candidate_records = [
            {
                "name": name,
                "source_counts": dict(sorted(counts.items())),
                "source_table_count": len(counts),
                "total_occurrences": sum(counts.values()),
            }
            for name, counts in sorted(
                entity_candidates.items(), key=lambda item: (item[0].casefold(), item[0])
            )
        ]
        rows.append(
            {
                "row_id": row_id,
                "source_id": source_id,
                "canonical_name": canonical,
                "name_candidates": candidate_records,
                "text": text,
                "text_sha256": _text_sha256(text),
            }
        )
    return rows, missing


def _mondo_curie(value: object) -> str | None:
    raw = _normalise_text(value)
    prefix = "http://purl.obolibrary.org/obo/MONDO_"
    if raw.startswith(prefix) and raw[len(prefix) :].isdigit():
        return f"MONDO:{raw[len(prefix):]}"
    if _MONDO_CURIE.fullmatch(raw):
        return raw
    return None


def _parse_mondo(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise SemanticFeatureError("MONDO JSON has no OBO Graphs graph list")
    terms: dict[str, dict[str, Any]] = {}
    for graph in graphs:
        if not isinstance(graph, Mapping):
            continue
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            curie = _mondo_curie(node.get("id"))
            if curie is None:
                continue
            label = _normalise_text(node.get("lbl"))
            meta = node.get("meta")
            meta = meta if isinstance(meta, Mapping) else {}
            definition_record = meta.get("definition")
            definition = (
                _normalise_text(definition_record.get("val"))
                if isinstance(definition_record, Mapping)
                else ""
            )
            synonym_records = meta.get("synonyms")
            synonyms: list[str] = []
            if isinstance(synonym_records, list):
                for record in synonym_records:
                    value = (
                        _normalise_text(record.get("val"))
                        if isinstance(record, Mapping)
                        else ""
                    )
                    if value:
                        synonyms.append(value)
            synonyms = sorted(set(synonyms), key=lambda value: (value.casefold(), value))
            candidate = {
                "label": label,
                "definition": definition,
                "synonyms": synonyms,
            }
            prior = terms.get(curie)
            if prior is not None and prior != candidate:
                raise SemanticFeatureError(f"MONDO JSON contains conflicting nodes for {curie}")
            terms[curie] = candidate
    if not terms:
        raise SemanticFeatureError("MONDO JSON contains no MONDO nodes")
    return terms


def _derive_disease_rows(
    mapping: EntityMapping, mondo_json: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    terms = _parse_mondo(mondo_json)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row_id, source_id in zip(mapping.row_ids, mapping.source_ids):
        term = terms.get(source_id)
        if term is None:
            missing.append(source_id)
            continue
        synonyms = list(term["synonyms"])
        label = str(term["label"])
        label_source = "rdfs_label"
        if not label and synonyms:
            label = synonyms[0]
            label_source = "first_lexical_synonym_fallback"
        if not label:
            missing.append(source_id)
            continue
        definition = str(term["definition"])
        if definition:
            text_mode = "label_plus_definition"
            text = f"Disease: {label}. Definition: {definition}"
        else:
            alternatives = [value for value in synonyms if value != label]
            if alternatives:
                text_mode = "label_plus_synonym_fallback"
                text = f"Disease: {label}. Synonyms: " + "; ".join(alternatives) + "."
            else:
                text_mode = "label_only_fallback"
                text = f"Disease: {label}."
        rows.append(
            {
                "row_id": row_id,
                "source_id": source_id,
                "label": label,
                "label_source": label_source,
                "definition": definition,
                "synonyms": synonyms,
                "text_mode": text_mode,
                "text": text,
                "text_sha256": _text_sha256(text),
            }
        )
    return rows, missing


def _text_contract(
    *,
    herb_mapping: EntityMapping,
    disease_mapping: EntityMapping,
    herb_component_csv: Path,
    mondo_json: Path,
) -> dict[str, Any]:
    herb_rows, missing_herbs = _derive_herb_rows(herb_mapping, herb_component_csv)
    disease_rows, missing_diseases = _derive_disease_rows(disease_mapping, mondo_json)
    if missing_herbs or missing_diseases:
        raise SemanticFeatureError(
            "semantic text coverage is incomplete: "
            f"missing herbs={missing_herbs[:5]}, diseases={missing_diseases[:5]}"
        )
    return {
        "schema": TEXT_MANIFEST_SCHEMA,
        "semantic_condition": SEMANTIC_CONDITION,
        "semantic_condition_status": "new_regenerated_condition_not_legacy_reproduction",
        "formal_eligible": True,
        "random_imputation": False,
        "association_label_free": True,
        "row_identity_basis": "generation_time_id_keyed_join",
        "plant_name_selection_rule": (
            "max occurrence count in reviewed herb-component.csv, then Unicode-NFC "
            "whitespace-normalized casefold lexical order, then exact lexical order"
        ),
        "disease_text_rule": (
            "rdfs label plus definition; if definition is absent use sorted synonyms; "
            "if label is absent use the first sorted synonym; label-only is the final fallback"
        ),
        "mappings": {
            "herb": _artifact(herb_mapping.path),
            "disease": _artifact(disease_mapping.path),
        },
        "sources": {
            "herb_component": {
                **_artifact(herb_component_csv),
                "columns_read": ["plant_curie", "plant name"],
                "association_columns_read": [],
            },
            "mondo_base_json": {
                **_artifact(mondo_json),
                "release": MONDO_RELEASE,
                "release_commit": MONDO_RELEASE_COMMIT,
                "download_url": MONDO_BASE_JSON_URL,
            },
        },
        "entities": {
            "herb": {
                "row_count": len(herb_rows),
                "mapping_sha256": herb_mapping.sha256,
                "rows": herb_rows,
            },
            "disease": {
                "row_count": len(disease_rows),
                "mapping_sha256": disease_mapping.sha256,
                "rows": disease_rows,
                "definition_count": sum(bool(row["definition"]) for row in disease_rows),
                "synonym_fallback_count": sum(
                    row["text_mode"] == "label_plus_synonym_fallback"
                    for row in disease_rows
                ),
                "label_only_fallback_count": sum(
                    row["text_mode"] == "label_only_fallback" for row in disease_rows
                ),
                "label_from_synonym_count": sum(
                    row["label_source"] == "first_lexical_synonym_fallback"
                    for row in disease_rows
                ),
            },
        },
    }


def build_semantic_text_manifest(
    *,
    herb_mapping_path: Path | str,
    disease_mapping_path: Path | str,
    herb_component_csv: Path | str,
    mondo_json: Path | str,
    output_path: Path | str,
    expected_herb_component_sha256: str = HERB_COMPONENT_SHA256,
    expected_mondo_sha256: str = MONDO_BASE_JSON_SHA256,
) -> Path:
    """Build the complete ID-keyed text corpus and provenance manifest."""

    herb_mapping = read_entity_mapping(herb_mapping_path, "herb")
    disease_mapping = read_entity_mapping(disease_mapping_path, "disease")
    component = Path(herb_component_csv).resolve()
    mondo = Path(mondo_json).resolve()
    _require_sha(component, expected_herb_component_sha256, "herb-component.csv")
    _require_sha(mondo, expected_mondo_sha256, f"MONDO {MONDO_RELEASE} mondo-base.json")
    contract = _text_contract(
        herb_mapping=herb_mapping,
        disease_mapping=disease_mapping,
        herb_component_csv=component,
        mondo_json=mondo,
    )
    output = Path(output_path).resolve()
    payload = {**contract, "created_at": _utc_now()}
    _atomic_json(output, payload)
    validate_semantic_text_manifest(output)
    return output


def validate_semantic_text_manifest(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    """Validate a supported ID-keyed semantic-text condition.

    The original source-text condition is re-derived locally below.  A Qwen
    condition is validated lazily by its own module so the existing SciBERT
    and shared-PCA stages can consume either condition without duplicating
    their identity checks.
    """

    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != TEXT_MANIFEST_SCHEMA:
        # Lazy import avoids a module cycle: the Qwen validator recursively
        # validates the source-text manifest through this public dispatcher.
        from .ethnobotany_qwen_semantics import (  # noqa: PLC0415
            QWEN_TEXT_MANIFEST_SCHEMA,
            validate_qwen_text_manifest,
        )

        if payload.get("schema") == QWEN_TEXT_MANIFEST_SCHEMA:
            return validate_qwen_text_manifest(manifest_path, resolver=resolver)
        raise SemanticFeatureError("semantic text manifest has a wrong schema")
    if (
        payload.get("semantic_condition") != SEMANTIC_CONDITION
        or payload.get("semantic_condition_status")
        != "new_regenerated_condition_not_legacy_reproduction"
        or payload.get("formal_eligible") is not True
        or payload.get("random_imputation") is not False
        or payload.get("association_label_free") is not True
        or payload.get("row_identity_basis") != "generation_time_id_keyed_join"
    ):
        raise SemanticFeatureError("semantic text manifest eligibility contract is false")
    mappings = payload.get("mappings")
    sources = payload.get("sources")
    if not isinstance(mappings, Mapping) or not isinstance(sources, Mapping):
        raise SemanticFeatureError("semantic text manifest sources are incomplete")
    herb_mapping = read_entity_mapping(
        _verify_artifact(
            mappings.get("herb"), "herb mapping", resolver=resolver
        ),
        "herb",
    )
    disease_mapping = read_entity_mapping(
        _verify_artifact(
            mappings.get("disease"), "disease mapping", resolver=resolver
        ),
        "disease",
    )
    component = _verify_artifact(
        sources.get("herb_component"), "herb-component.csv", resolver=resolver
    )
    if (
        sources["herb_component"].get("columns_read")
        != ["plant_curie", "plant name"]
        or sources["herb_component"].get("association_columns_read") != []
    ):
        raise SemanticFeatureError("semantic text generation reads association fields")
    mondo_record = sources.get("mondo_base_json")
    mondo = _verify_artifact(
        mondo_record, "MONDO mondo-base.json", resolver=resolver
    )
    if not isinstance(mondo_record, Mapping) or (
        mondo_record.get("release") != MONDO_RELEASE
        or mondo_record.get("release_commit") != MONDO_RELEASE_COMMIT
        or mondo_record.get("download_url") != MONDO_BASE_JSON_URL
        or mondo_record.get("sha256") != MONDO_BASE_JSON_SHA256
    ):
        raise SemanticFeatureError("MONDO release identity differs from the reviewed pin")
    if (
        _sha256(component) != HERB_COMPONENT_SHA256
    ):
        raise SemanticFeatureError("plant-name source differs from the reviewed pin")
    expected = _text_contract(
        herb_mapping=herb_mapping,
        disease_mapping=disease_mapping,
        herb_component_csv=component,
        mondo_json=mondo,
    )
    if resolver is not None:
        # Portable bundles retain immutable generation-time path strings in
        # nested manifests.  Resolver-selected objects are authenticated by
        # bytes/hash above; path spelling is not an entity-identity claim.
        expected["mappings"]["herb"]["path"] = mappings["herb"]["path"]
        expected["mappings"]["disease"]["path"] = mappings["disease"]["path"]
        expected["sources"]["herb_component"]["path"] = sources[
            "herb_component"
        ]["path"]
        expected["sources"]["mondo_base_json"]["path"] = mondo_record["path"]
    observed = {key: value for key, value in payload.items() if key != "created_at"}
    if observed != expected:
        raise SemanticFeatureError(
            "semantic text manifest does not equal rows re-derived from its pinned sources"
        )
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "mappings": {"herb": herb_mapping, "disease": disease_mapping},
        "dependencies": [
            manifest_path,
            herb_mapping.path,
            disease_mapping.path,
            component,
            mondo,
        ],
    }


def _scibert_provenance(snapshot_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, expected in SCIBERT_FILE_SHA256.items():
        path = snapshot_dir / name
        _require_sha(path, expected, f"SciBERT {name}")
        files[name] = _artifact(path)
    return {
        "model_id": SCIBERT_MODEL_ID,
        "revision": SCIBERT_REVISION,
        "files": files,
        "pooling": "first_special_classification_token_last_hidden_state",
        "tokenization": {
            "truncation": True,
            "max_length": SCIBERT_MAX_LENGTH,
            "special_tokens": True,
        },
        "raw_dimension": SCIBERT_DIMENSION,
    }


def _formal_thread_environment() -> dict[str, str]:
    limits = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    }
    if any(value != "4" for value in limits.values()):
        raise SemanticFeatureError(
            "formal semantic generation requires OMP_NUM_THREADS=4, "
            "MKL_NUM_THREADS=4, and OPENBLAS_NUM_THREADS=4 before process start"
        )
    return limits


def _encode_scibert(
    texts: Sequence[str],
    *,
    snapshot_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    thread_environment = _formal_thread_environment()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise SemanticFeatureError(
            "SciBERT extraction requires torch and transformers in the selected interpreter"
        ) from error
    if batch_size < 1:
        raise SemanticFeatureError("batch_size must be positive")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before parallel work starts.  The
        # CLI launches a fresh process; this guard keeps library reuse explicit.
        pass
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir), local_files_only=True, use_fast=False
    )
    model = AutoModel.from_pretrained(str(snapshot_dir), local_files_only=True)
    model.eval()
    model.to("cpu")
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            selected = list(texts[start : start + batch_size])
            encoded = tokenizer(
                selected,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=SCIBERT_MAX_LENGTH,
            )
            output = model(**encoded)
            values = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            batches.append(np.asarray(values, dtype=np.float32))
    matrix = np.concatenate(batches, axis=0)
    if matrix.shape != (len(texts), SCIBERT_DIMENSION) or not np.isfinite(matrix).all():
        raise SemanticFeatureError(f"SciBERT emitted invalid matrix {matrix.shape}")
    runtime = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "device": "cpu",
        "seed": 42,
        "model_training_flag": bool(model.training),
        "deterministic_algorithms": True,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": thread_environment,
        "batch_size": batch_size,
    }
    return matrix, runtime


def extract_scibert_semantic_embeddings(
    *,
    text_manifest: Path | str,
    snapshot_dir: Path | str,
    output_dir: Path | str,
    batch_size: int = 16,
) -> Path:
    """Encode all ID-keyed texts with a pinned, local SciBERT snapshot."""

    validated = validate_semantic_text_manifest(text_manifest)
    payload = validated["manifest"]
    snapshot = Path(snapshot_dir).resolve()
    if snapshot.name != SCIBERT_REVISION:
        raise SemanticFeatureError(
            f"SciBERT snapshot directory must be the pinned revision {SCIBERT_REVISION}"
        )
    provenance = _scibert_provenance(snapshot)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entity_records: dict[str, Any] = {}
    runtime: dict[str, Any] | None = None
    for entity in ("herb", "disease"):
        rows = payload["entities"][entity]["rows"]
        matrix, observed_runtime = _encode_scibert(
            [str(row["text"]) for row in rows],
            snapshot_dir=snapshot,
            batch_size=batch_size,
        )
        if runtime is None:
            runtime = observed_runtime
        elif runtime != observed_runtime:
            raise SemanticFeatureError("SciBERT runtime changed between entity extractions")
        path = output / f"{entity}_semantic_embeddings_768d.npz"
        _atomic_npz(
            path,
            row_ids=np.asarray([row["row_id"] for row in rows], dtype=str),
            source_ids=np.asarray([row["source_id"] for row in rows], dtype=str),
            text_sha256=np.asarray([row["text_sha256"] for row in rows], dtype=str),
            embeddings=matrix,
        )
        entity_records[entity] = {
            "artifact": _artifact(path),
            "row_count": len(rows),
            "dimension": SCIBERT_DIMENSION,
            "mapping_sha256": payload["entities"][entity]["mapping_sha256"],
        }
    manifest = {
        "schema": EMBEDDING_MANIFEST_SCHEMA,
        "semantic_condition": payload["semantic_condition"],
        "semantic_condition_status": payload["semantic_condition_status"],
        "created_at": _utc_now(),
        "formal_eligible": True,
        "random_imputation": False,
        "text_manifest": _artifact(Path(text_manifest).resolve()),
        "model": provenance,
        "runtime": runtime,
        "entities": entity_records,
    }
    manifest_path = output / "semantic_embedding_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_scibert_embedding_manifest(manifest_path)
    return manifest_path


def validate_scibert_embedding_manifest(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != EMBEDDING_MANIFEST_SCHEMA:
        raise SemanticFeatureError("SciBERT embedding manifest has a wrong schema")
    text_path = _verify_artifact(
        payload.get("text_manifest"), "semantic text manifest", resolver=resolver
    )
    text_validation = validate_semantic_text_manifest(text_path, resolver=resolver)
    text = text_validation["manifest"]
    if (
        payload.get("semantic_condition") != text.get("semantic_condition")
        or payload.get("semantic_condition_status")
        != text.get("semantic_condition_status")
        or payload.get("formal_eligible") is not True
        or payload.get("random_imputation") is not False
    ):
        raise SemanticFeatureError("SciBERT embedding eligibility contract is false")
    model = payload.get("model")
    if not isinstance(model, Mapping) or (
        model.get("model_id") != SCIBERT_MODEL_ID
        or model.get("revision") != SCIBERT_REVISION
        or model.get("raw_dimension") != SCIBERT_DIMENSION
        or model.get("pooling")
        != "first_special_classification_token_last_hidden_state"
        or model.get("tokenization")
        != {
            "truncation": True,
            "max_length": SCIBERT_MAX_LENGTH,
            "special_tokens": True,
        }
    ):
        raise SemanticFeatureError("SciBERT model identity/pooling is false")
    files = model.get("files")
    if not isinstance(files, Mapping):
        raise SemanticFeatureError("SciBERT model file manifest is absent")
    for name, expected in SCIBERT_FILE_SHA256.items():
        model_path = _verify_artifact(
            files.get(name), f"SciBERT {name}", resolver=resolver
        )
        if _sha256(model_path) != expected:
            raise SemanticFeatureError(f"SciBERT {name} differs from its reviewed pin")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or (
        runtime.get("device") != "cpu"
        or runtime.get("seed") != 42
        or runtime.get("model_training_flag") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("torch_num_threads") != 4
        or runtime.get("torch_num_interop_threads") != 1
        or runtime.get("thread_environment")
        != {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
    ):
        raise SemanticFeatureError("SciBERT deterministic runtime contract is absent")
    outputs: dict[str, dict[str, Any]] = {}
    entities = payload.get("entities")
    if not isinstance(entities, Mapping):
        raise SemanticFeatureError("SciBERT entity outputs are absent")
    for entity in ("herb", "disease"):
        record = entities.get(entity)
        if not isinstance(record, Mapping):
            raise SemanticFeatureError(f"SciBERT {entity} output record is absent")
        artifact = _verify_artifact(
            record.get("artifact"), f"SciBERT {entity} output", resolver=resolver
        )
        rows = text["entities"][entity]["rows"]
        try:
            with np.load(artifact, allow_pickle=False) as arrays:
                row_ids = arrays["row_ids"].astype(str).tolist()
                source_ids = arrays["source_ids"].astype(str).tolist()
                text_hashes = arrays["text_sha256"].astype(str).tolist()
                matrix = np.asarray(arrays["embeddings"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as error:
            raise SemanticFeatureError(f"cannot validate {entity} embeddings: {error}") from error
        if row_ids != [row["row_id"] for row in rows]:
            raise SemanticFeatureError(f"{entity} embedding row IDs are not canonical")
        if source_ids != [row["source_id"] for row in rows]:
            raise SemanticFeatureError(f"{entity} embedding source IDs are false")
        if text_hashes != [row["text_sha256"] for row in rows]:
            raise SemanticFeatureError(f"{entity} embedding text hashes are false")
        if matrix.shape != (len(rows), SCIBERT_DIMENSION) or not np.isfinite(matrix).all():
            raise SemanticFeatureError(f"{entity} embedding matrix is invalid")
        if (
            record.get("row_count") != len(rows)
            or record.get("dimension") != SCIBERT_DIMENSION
            or record.get("mapping_sha256")
            != text["entities"][entity]["mapping_sha256"]
        ):
            raise SemanticFeatureError(f"{entity} embedding manifest counts are false")
        outputs[entity] = {"path": artifact, "matrix": matrix, "rows": rows}
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "text_manifest_path": text_path,
        "outputs": outputs,
        "dependencies": list(
            dict.fromkeys(
                [manifest_path, *text_validation["dependencies"]]
                + [
                    _verify_artifact(
                        files[name], f"SciBERT {name}", resolver=resolver
                    )
                    for name in sorted(SCIBERT_FILE_SHA256)
                ]
                + [outputs[entity]["path"] for entity in ("herb", "disease")]
            )
        ),
    }


def _canonical_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for index, row in enumerate(result):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            result[index] *= -1.0
    return result


def _row_sidecar(mapping: EntityMapping) -> dict[str, Any]:
    return {
        "schema": FEATURE_ROW_IDS_SCHEMA,
        "entity": mapping.entity,
        "id_space": "protocol_raw_id",
        "mapping_sha256": mapping.sha256,
        "row_ids": list(mapping.row_ids),
    }


def _formal_semantic_output_dimension(value: object) -> int:
    """Return a dimension admitted by the reviewed semantic-PCA contract."""

    if type(value) is not int or value not in FORMAL_SEMANTIC_OUTPUT_DIMENSIONS:
        supported = ", ".join(str(item) for item in FORMAL_SEMANTIC_OUTPUT_DIMENSIONS)
        raise SemanticFeatureError(
            f"formal semantic output dimension must be one of: {supported}"
        )
    return value


def reduce_shared_semantic_embeddings(
    *,
    embedding_manifest: Path | str,
    output_dir: Path | str,
    output_dimension: int = SEMANTIC_OUTPUT_DIMENSION,
) -> Path:
    """Fit one centered full-SVD reducer to herb+disease and apply it to both."""

    output_dimension = _formal_semantic_output_dimension(output_dimension)
    validated = validate_scibert_embedding_manifest(embedding_manifest)
    thread_environment = _formal_thread_environment()
    herb = validated["outputs"]["herb"]["matrix"].astype(np.float64)
    disease = validated["outputs"]["disease"]["matrix"].astype(np.float64)
    combined = np.vstack([herb, disease])
    rank_bound = min(combined.shape[0] - 1, combined.shape[1])
    if output_dimension > rank_bound:
        raise SemanticFeatureError(
            f"semantic output dimension {output_dimension} exceeds centered "
            f"fit rank bound {rank_bound}"
        )
    mean = combined.mean(axis=0)
    centered = combined - mean
    try:
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise SemanticFeatureError(f"shared semantic SVD failed: {error}") from error
    components = _canonical_component_signs(right[:output_dimension])
    reduced = {
        "herb": np.asarray((herb - mean) @ components.T, dtype=np.float32),
        "disease": np.asarray((disease - mean) @ components.T, dtype=np.float32),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / (
        f"shared_semantic_centered_svd_pca{output_dimension}.npz"
    )
    checkpoint_contract = {
        "schema": REDUCTION_MANIFEST_SCHEMA,
        "semantic_condition": validated["manifest"]["semantic_condition"],
        "algorithm": "deterministic_centered_full_svd_pca_with_canonical_component_signs",
        "fit_order": "herb rows in mapping order, followed by disease rows in mapping order",
        "fit_row_count": int(combined.shape[0]),
        "input_dimension": int(combined.shape[1]),
        "output_dimension": output_dimension,
        "embedding_manifest_sha256": _sha256(validated["manifest_path"]),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": thread_environment,
        },
    }
    _atomic_npz(
        checkpoint,
        mean=mean,
        components=components,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        astra_reducer_contract_json=np.asarray(
            json.dumps(checkpoint_contract, sort_keys=True, separators=(",", ":"))
        ),
    )
    text_validated = validate_semantic_text_manifest(
        validated["text_manifest_path"]
    )
    outputs: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        matrix = reduced[entity]
        if not np.isfinite(matrix).all() or np.unique(matrix, axis=0).shape[0] < 2:
            raise SemanticFeatureError(f"shared reducer emitted degenerate {entity} features")
        matrix_path = output / (
            f"{entity}_semantic_embeddings_{output_dimension}d.npy"
        )
        sidecar_path = feature_row_ids_path(matrix_path)
        mapping = text_validated["mappings"][entity]
        _atomic_npy(matrix_path, matrix)
        _atomic_json(sidecar_path, _row_sidecar(mapping))
        outputs[entity] = {
            "matrix": _artifact(matrix_path),
            "row_ids_sidecar": _artifact(sidecar_path),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "mapping_sha256": mapping.sha256,
            "unique_rows": int(np.unique(matrix, axis=0).shape[0]),
            "varying_columns": int(np.count_nonzero(np.ptp(matrix, axis=0) > 0.0)),
        }
    manifest = {
        **checkpoint_contract,
        "created_at": _utc_now(),
        "formal_eligible": True,
        "random_imputation": False,
        "semantic_condition_status": validated["manifest"][
            "semantic_condition_status"
        ],
        "embedding_manifest": _artifact(validated["manifest_path"]),
        "reducer_checkpoint": _artifact(checkpoint),
        "entities": outputs,
    }
    manifest_path = output / "semantic_reduction_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_shared_semantic_reduction(manifest_path)
    return manifest_path


def validate_shared_semantic_reduction(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != REDUCTION_MANIFEST_SCHEMA:
        # Keep the reviewed PCA reducer schema narrow while allowing a
        # separately identified, recursively audited S4 control to satisfy the
        # same consumer interface.  The lazy import avoids a module cycle.
        from .ethnobotany_semantic_controls import (  # noqa: PLC0415
            ZERO_PAD_CONTROL_MANIFEST_SCHEMA,
            validate_pca64_zero_pad128_control,
        )

        if payload.get("schema") == ZERO_PAD_CONTROL_MANIFEST_SCHEMA:
            return validate_pca64_zero_pad128_control(
                manifest_path, resolver=resolver
            )
        raise SemanticFeatureError("semantic reduction manifest has a wrong schema")
    output_dimension = _formal_semantic_output_dimension(
        payload.get("output_dimension")
    )
    if (
        payload.get("formal_eligible") is not True
        or payload.get("random_imputation") is not False
        or payload.get("algorithm")
        != "deterministic_centered_full_svd_pca_with_canonical_component_signs"
        or payload.get("fit_order")
        != "herb rows in mapping order, followed by disease rows in mapping order"
        or payload.get("input_dimension") != SCIBERT_DIMENSION
    ):
        raise SemanticFeatureError("shared semantic reducer eligibility contract is false")
    embeddings_path = _verify_artifact(
        payload.get("embedding_manifest"),
        "semantic embedding manifest",
        resolver=resolver,
    )
    embeddings = validate_scibert_embedding_manifest(
        embeddings_path, resolver=resolver
    )
    if (
        payload.get("semantic_condition")
        != embeddings["manifest"].get("semantic_condition")
        or payload.get("semantic_condition_status")
        != embeddings["manifest"].get("semantic_condition_status")
    ):
        raise SemanticFeatureError(
            "shared semantic reducer condition differs from its embeddings"
        )
    expected_fit_rows = sum(
        embeddings["outputs"][entity]["matrix"].shape[0]
        for entity in ("herb", "disease")
    )
    if payload.get("fit_row_count") != expected_fit_rows:
        raise SemanticFeatureError("shared reducer fit-row count is false")
    if payload.get("embedding_manifest_sha256") != _sha256(embeddings_path):
        raise SemanticFeatureError("shared reducer was fitted to another embedding manifest")
    checkpoint_path = _verify_artifact(
        payload.get("reducer_checkpoint"),
        "shared semantic reducer",
        resolver=resolver,
    )
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            mean = np.asarray(checkpoint["mean"], dtype=np.float64)
            components = np.asarray(checkpoint["components"], dtype=np.float64)
            singular_values = np.asarray(checkpoint["singular_values"], dtype=np.float64)
            embedded_contract = json.loads(
                str(checkpoint["astra_reducer_contract_json"].item())
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SemanticFeatureError(f"cannot validate shared semantic reducer: {error}") from error
    contract_keys = {
        "schema",
        "semantic_condition",
        "algorithm",
        "fit_order",
        "fit_row_count",
        "input_dimension",
        "output_dimension",
        "embedding_manifest_sha256",
        "runtime",
    }
    if embedded_contract != {key: payload[key] for key in contract_keys}:
        raise SemanticFeatureError("reducer checkpoint contract differs from its manifest")
    if mean.shape != (SCIBERT_DIMENSION,) or components.shape != (
        output_dimension,
        SCIBERT_DIMENSION,
    ):
        raise SemanticFeatureError("shared semantic reducer shapes are invalid")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("thread_environment") != {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }:
        raise SemanticFeatureError("shared semantic reducer thread contract is absent")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise SemanticFeatureError("shared semantic reducer contains NaN/Inf")
    if singular_values.ndim != 1 or singular_values.size < output_dimension:
        raise SemanticFeatureError("shared semantic reducer singular values are invalid")
    combined = np.vstack(
        [
            embeddings["outputs"]["herb"]["matrix"].astype(np.float64),
            embeddings["outputs"]["disease"]["matrix"].astype(np.float64),
        ]
    )
    rank_bound = min(combined.shape[0] - 1, combined.shape[1])
    if output_dimension > rank_bound:
        raise SemanticFeatureError(
            f"semantic output dimension {output_dimension} exceeds centered "
            f"fit rank bound {rank_bound}"
        )
    expected_mean = combined.mean(axis=0)
    try:
        _, expected_singular_values, expected_right = np.linalg.svd(
            combined - expected_mean, full_matrices=False
        )
    except np.linalg.LinAlgError as error:
        raise SemanticFeatureError(
            f"cannot re-derive shared semantic SVD from pinned embeddings: {error}"
        ) from error
    expected_components = _canonical_component_signs(
        expected_right[:output_dimension]
    )
    if not (
        np.array_equal(mean, expected_mean)
        and np.allclose(
            singular_values,
            expected_singular_values,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(
            components,
            expected_components,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise SemanticFeatureError(
            "shared semantic reducer is not re-derived from the pinned embeddings"
        )
    entities = payload.get("entities")
    if not isinstance(entities, Mapping):
        raise SemanticFeatureError("semantic reduction entity outputs are absent")
    outputs: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        record = entities.get(entity)
        if not isinstance(record, Mapping):
            raise SemanticFeatureError(f"reduced {entity} output record is absent")
        matrix_path = _verify_artifact(
            record.get("matrix"), f"reduced {entity} matrix", resolver=resolver
        )
        sidecar_path = _verify_artifact(
            record.get("row_ids_sidecar"),
            f"reduced {entity} row IDs",
            resolver=resolver,
        )
        raw = embeddings["outputs"][entity]["matrix"].astype(np.float64)
        expected = np.asarray((raw - mean) @ components.T, dtype=np.float32)
        try:
            observed = np.load(matrix_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise SemanticFeatureError(f"cannot read reduced {entity} matrix: {error}") from error
        if not np.array_equal(observed, expected):
            raise SemanticFeatureError(f"reduced {entity} matrix is not reproduced by checkpoint")
        text = validate_semantic_text_manifest(
            embeddings["text_manifest_path"], resolver=resolver
        )
        mapping = text["mappings"][entity]
        if _read_json(sidecar_path) != _row_sidecar(mapping):
            raise SemanticFeatureError(f"reduced {entity} row-ID sidecar is false")
        if (
            record.get("shape") != list(expected.shape)
            or record.get("dtype") != str(expected.dtype)
            or record.get("mapping_sha256") != mapping.sha256
            or record.get("unique_rows") != int(np.unique(expected, axis=0).shape[0])
            or record.get("varying_columns")
            != int(np.count_nonzero(np.ptp(expected, axis=0) > 0.0))
        ):
            raise SemanticFeatureError(f"reduced {entity} diagnostics are false")
        outputs[entity] = {"matrix_path": matrix_path, "matrix": observed}
    dependencies = [
        manifest_path,
        embeddings_path,
        embeddings["text_manifest_path"],
        checkpoint_path,
    ]
    text_dependencies = validate_semantic_text_manifest(
        embeddings["text_manifest_path"], resolver=resolver
    )["dependencies"]
    dependencies.extend(text_dependencies)
    model_files = embeddings["manifest"]["model"]["files"]
    dependencies.extend(
        _verify_artifact(
            model_files[name], f"SciBERT {name}", resolver=resolver
        )
        for name in sorted(SCIBERT_FILE_SHA256)
    )
    for entity in ("herb", "disease"):
        dependencies.append(embeddings["outputs"][entity]["path"])
        record = entities[entity]
        dependencies.append(
            _verify_artifact(
                record["matrix"], f"reduced {entity} matrix", resolver=resolver
            )
        )
        dependencies.append(
            _verify_artifact(
                record["row_ids_sidecar"],
                f"reduced {entity} row IDs",
                resolver=resolver,
            )
        )
    unique_dependencies = list(dict.fromkeys(Path(item).resolve() for item in dependencies))
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "embedding_manifest_path": embeddings_path,
        "checkpoint_path": checkpoint_path,
        "outputs": outputs,
        "dependencies": unique_dependencies,
    }


def _semantic_validator_source_paths(schema: str) -> list[Path]:
    """Return every in-repository source file trusted by the deep validator."""

    root = Path(__file__).resolve().parent
    names = [
        "ethnobotany_semantic_features.py",
        "ethnobotany_feature_bundle.py",
        "ethnobotany_qwen_semantics.py",
        "features.py",
    ]
    if schema != REDUCTION_MANIFEST_SCHEMA:
        names.append("ethnobotany_semantic_controls.py")
    paths = [(root / name).resolve() for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SemanticFeatureError(
            "semantic validator source is absent: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _receipt_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def create_semantic_deep_audit_receipt(
    *,
    semantic_reduction_manifest: Path | str,
    receipt_path: Path | str,
) -> Path:
    """Run the recursive validator once and atomically attest all its inputs.

    The receipt is deliberately path-bound.  It is an optimisation for repeated
    candidates on one reviewed workspace, not a portable replacement for the
    recursive validator.
    """

    manifest_path = Path(semantic_reduction_manifest).resolve(strict=True)
    destination = Path(receipt_path).resolve(strict=False)
    if destination.exists():
        raise SemanticFeatureError(
            f"refusing to overwrite semantic deep-audit receipt: {destination}"
        )
    audited = validate_shared_semantic_reduction(manifest_path)
    manifest = audited.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SemanticFeatureError("deep semantic validator returned no manifest")
    schema = manifest.get("schema")
    dimension = manifest.get("output_dimension")
    if not isinstance(schema, str) or type(dimension) is not int:
        raise SemanticFeatureError("deep semantic validator returned a bad contract")
    dependencies = list(
        dict.fromkeys(
            Path(item).resolve(strict=True)
            for item in audited.get("dependencies", [])
        )
    )
    if manifest_path not in dependencies:
        raise SemanticFeatureError(
            "deep semantic validator omitted its manifest from dependencies"
        )
    if destination in dependencies:
        raise SemanticFeatureError("receipt path collides with a semantic dependency")
    outputs = audited.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SemanticFeatureError("deep semantic validator returned no entity outputs")
    entity_records: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        output = outputs.get(entity)
        if not isinstance(output, Mapping):
            raise SemanticFeatureError(
                f"deep semantic validator omitted the {entity} output"
            )
        matrix_path = Path(output.get("matrix_path", "")).resolve(strict=True)
        try:
            shape = list(np.asarray(output["matrix"]).shape)
        except KeyError as error:
            raise SemanticFeatureError(
                f"deep semantic validator omitted the {entity} matrix"
            ) from error
        entity_records[entity] = {
            **_receipt_file_record(matrix_path),
            "shape": shape,
        }
    payload = {
        "schema": DEEP_AUDIT_RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "deep_audit": True,
        "semantic_schema": schema,
        "output_dimension": dimension,
        "manifest": _receipt_file_record(manifest_path),
        "validator_sources": [
            _receipt_file_record(path)
            for path in _semantic_validator_source_paths(schema)
        ],
        "dependencies": [
            _receipt_file_record(path) for path in dependencies
        ],
        "entities": entity_records,
    }
    _atomic_json(destination, payload)
    return destination


def validate_semantic_deep_audit_receipt(
    *,
    semantic_reduction_manifest: Path | str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Fast-path a prior deep audit while rehashing every trusted byte."""

    manifest_path = Path(semantic_reduction_manifest).resolve(strict=True)
    receipt = Path(receipt_path).resolve(strict=True)
    payload = _read_json(receipt)
    if (
        payload.get("schema") != DEEP_AUDIT_RECEIPT_SCHEMA
        or payload.get("deep_audit") is not True
    ):
        raise SemanticFeatureError("semantic deep-audit receipt contract is invalid")
    manifest_record = payload.get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise SemanticFeatureError("semantic deep-audit receipt has no manifest")
    if Path(str(manifest_record.get("path", ""))).resolve() != manifest_path:
        raise SemanticFeatureError("semantic deep-audit receipt is for another manifest")

    def verify_record(record: object, label: str) -> Path:
        if not isinstance(record, Mapping):
            raise SemanticFeatureError(f"{label} receipt record is invalid")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file():
            raise SemanticFeatureError(f"{label} receipt artifact is absent: {path}")
        if (
            type(record.get("bytes")) is not int
            or record["bytes"] != path.stat().st_size
            or not isinstance(record.get("sha256"), str)
            or record["sha256"] != _sha256(path)
        ):
            raise SemanticFeatureError(f"{label} differs from its deep-audit receipt")
        return path

    verify_record(manifest_record, "semantic manifest")
    manifest = _read_json(manifest_path)
    schema = manifest.get("schema")
    dimension = manifest.get("output_dimension")
    if (
        schema != payload.get("semantic_schema")
        or dimension != payload.get("output_dimension")
        or type(dimension) is not int
    ):
        raise SemanticFeatureError(
            "semantic manifest contract differs from its deep-audit receipt"
        )
    source_records = payload.get("validator_sources")
    if not isinstance(source_records, list):
        raise SemanticFeatureError("deep-audit validator source records are absent")
    expected_sources = _semantic_validator_source_paths(str(schema))
    observed_sources = [
        verify_record(record, "semantic validator source")
        for record in source_records
    ]
    if observed_sources != expected_sources:
        raise SemanticFeatureError(
            "semantic validator source set differs from its deep-audit receipt"
        )
    dependency_records = payload.get("dependencies")
    if not isinstance(dependency_records, list) or not dependency_records:
        raise SemanticFeatureError("deep-audit dependency records are absent")
    dependencies = [
        verify_record(record, "semantic dependency")
        for record in dependency_records
    ]
    if len(dependencies) != len(set(dependencies)) or manifest_path not in dependencies:
        raise SemanticFeatureError("deep-audit dependency set is invalid")

    entities = manifest.get("entities")
    receipt_entities = payload.get("entities")
    if not isinstance(entities, Mapping) or not isinstance(receipt_entities, Mapping):
        raise SemanticFeatureError("deep-audit entity records are absent")
    outputs: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        manifest_entity = entities.get(entity)
        receipt_entity = receipt_entities.get(entity)
        if not isinstance(manifest_entity, Mapping) or not isinstance(
            receipt_entity, Mapping
        ):
            raise SemanticFeatureError(f"deep-audit {entity} record is absent")
        matrix_path = verify_record(receipt_entity, f"deep-audit {entity} matrix")
        matrix_artifact = manifest_entity.get("matrix")
        if not isinstance(matrix_artifact, Mapping):
            raise SemanticFeatureError(f"semantic {entity} matrix record is absent")
        if (
            Path(str(matrix_artifact.get("path", ""))).resolve() != matrix_path
            or matrix_artifact.get("sha256") != receipt_entity.get("sha256")
            or manifest_entity.get("shape") != receipt_entity.get("shape")
            or matrix_path not in dependencies
        ):
            raise SemanticFeatureError(
                f"semantic {entity} output differs from its deep-audit receipt"
            )
        try:
            matrix = np.load(matrix_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise SemanticFeatureError(
                f"cannot load deep-audited {entity} matrix: {error}"
            ) from error
        if list(matrix.shape) != receipt_entity.get("shape"):
            raise SemanticFeatureError(
                f"semantic {entity} matrix shape differs from its deep-audit receipt"
            )
        outputs[entity] = {"matrix_path": matrix_path, "matrix": matrix}
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "outputs": outputs,
        "dependencies": dependencies,
        "deep_audit": True,
        "receipt_path": receipt,
        "receipt_sha256": _sha256(receipt),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    text_parser = subparsers.add_parser("build-text", help="build ID-keyed texts")
    text_parser.add_argument("--herb-mapping", required=True)
    text_parser.add_argument("--disease-mapping", required=True)
    text_parser.add_argument("--herb-component", required=True)
    text_parser.add_argument("--mondo-json", required=True)
    text_parser.add_argument("--output", required=True)
    embed_parser = subparsers.add_parser("embed", help="run pinned local SciBERT")
    embed_parser.add_argument("--text-manifest", required=True)
    embed_parser.add_argument("--snapshot", required=True)
    embed_parser.add_argument("--output-dir", required=True)
    embed_parser.add_argument("--batch-size", type=int, default=16)
    reduce_parser = subparsers.add_parser(
        "reduce", help="fit a shared formal PCA64 or PCA128 reducer"
    )
    reduce_parser.add_argument("--embedding-manifest", required=True)
    reduce_parser.add_argument("--output-dir", required=True)
    reduce_parser.add_argument(
        "--output-dimension",
        type=int,
        choices=FORMAL_SEMANTIC_OUTPUT_DIMENSIONS,
        default=SEMANTIC_OUTPUT_DIMENSION,
    )
    validate_parser = subparsers.add_parser("validate", help="validate a final reducer")
    validate_parser.add_argument("--manifest", required=True)
    receipt_parser = subparsers.add_parser(
        "create-receipt", help="deep-audit a reducer and write a reusable receipt"
    )
    receipt_parser.add_argument("--manifest", required=True)
    receipt_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-text":
        path = build_semantic_text_manifest(
            herb_mapping_path=args.herb_mapping,
            disease_mapping_path=args.disease_mapping,
            herb_component_csv=args.herb_component,
            mondo_json=args.mondo_json,
            output_path=args.output,
        )
    elif args.command == "embed":
        path = extract_scibert_semantic_embeddings(
            text_manifest=args.text_manifest,
            snapshot_dir=args.snapshot,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    elif args.command == "reduce":
        path = reduce_shared_semantic_embeddings(
            embedding_manifest=args.embedding_manifest,
            output_dir=args.output_dir,
            output_dimension=args.output_dimension,
        )
    elif args.command == "validate":
        validate_shared_semantic_reduction(args.manifest)
        path = Path(args.manifest).resolve()
    else:
        path = create_semantic_deep_audit_receipt(
            semantic_reduction_manifest=args.manifest,
            receipt_path=args.output,
        )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
