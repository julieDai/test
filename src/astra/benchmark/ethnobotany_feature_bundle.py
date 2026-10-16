#!/usr/bin/env python3
"""Build an identity-safe, resumable Ethnobotany feature bundle.

This command intentionally does not call PubChem, UniProt, KPGT, ESM-2, or a
language model.  It records ID-keyed retrieval caches, merges independently
generated embedding shards, applies a deterministic or supplied linear
reducer, and packages the four 64-dimensional matrices consumed by HerbHG.
Missing data are represented by explicit masks and failure records; random
feature fabrication is never used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path


CACHE_SCHEMA = "astra-ethnobotany-retrieval-cache/v2"
ASSEMBLY_SCHEMA = "astra-ethnobotany-raw-embedding-assembly/v1"
REDUCER_SCHEMA = "astra-ethnobotany-feature-component/v1"
BUNDLE_SCHEMA = "astra-ethnobotany-feature-bundle/v1"
EMBEDDING_SHARD_SCHEMA = "astra-ethnobotany-embedding-shard/v2"
EXTRACTION_MANIFEST_SCHEMA = "astra-ethnobotany-embedding-extraction/v2"
ORDER_CLAIM_IMPORT_SCHEMA = "astra-ethnobotany-order-claim-import/v1"
TARGET_CLASSIFICATION_SCHEMA = (
    "astra-ethnobotany-target-classification-evidence/v1"
)
AUDITED_MISSING_SEQUENCE_SCHEMA = (
    "astra-ethnobotany-audited-missing-sequence-evidence/v1"
)
SEQUENCE_SET_AGGREGATION = "equal_weight_mean_of_distinct_sequence_embeddings"
SEQUENCE_SET_RESOLUTION_SCHEMA = (
    "astra-ethnobotany-protein-sequence-set-resolution/v1"
)
FROZEN_SEQUENCE_TIERS = (
    "mane_select",
    "refseq_select",
    "reviewed_uniprot",
    "known_refseq_np",
    "unreviewed_uniprot",
    "complete_refseq",
)

_ZERO_UNK_GENE_TYPES = {
    "pseudogene": "PSEUDO",
    "ncrna": "ncRNA",
    "rrna": "rRNA",
}
_GENERIC_ZERO_UNK_REASONS = {
    "mapping target is not legally sequence-encodable",
    "target is not a protein",
    "target not protein",
    "not a protein",
    "non-protein",
    "non protein",
    "unknown",
}
_AUDITED_MISSING_CLASSES = {
    "unknown_target_class": ("UNKNOWN", "ineligible"),
    "protein_coding_no_valid_sequence": ("PROTEIN_CODING", "missing"),
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

ENTITY_NAMES = ("herb", "compound", "protein", "disease")
BIOLOGICAL_ENTITIES = ("compound", "protein")
STANDARD_FILENAMES = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
    "herb": "herb_semantic_embeddings_64d.npy",
    "disease": "disease_semantic_embeddings_64d.npy",
}


class FeatureBundleError(RuntimeError):
    """Raised when an artifact violates the feature-bundle contract."""


class IncompleteFeatureBundleError(FeatureBundleError):
    """Raised after incomplete state has been saved for later resumption."""


@dataclass(frozen=True)
class EntityMapping:
    entity: str
    path: Path
    sha256: str
    row_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    headers: tuple[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _bundle_artifact(path: Path, bundle_root: Path) -> dict[str, Any]:
    record = _artifact(path)
    try:
        record["relative_path"] = str(path.resolve().relative_to(bundle_root.resolve()))
    except ValueError as error:
        raise FeatureBundleError(
            f"bundle artifact {path} is outside bundle root {bundle_root}"
        ) from error
    return record


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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
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
            np.savez(handle, **arrays)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_matrix(path: Path, values: np.ndarray) -> None:
    if path.suffix.lower() == ".npy":
        _atomic_npy(path, np.asarray(values))
        return
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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle)
            handle.flush()
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeatureBundleError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FeatureBundleError(f"JSON artifact {path} must contain an object")
    return payload


def read_entity_mapping(path: Path | str, entity: str) -> EntityMapping:
    """Read the authoritative two-column mapping without positional guessing."""

    path = Path(path).resolve()
    entity = str(entity).strip().lower()
    if entity not in ENTITY_NAMES:
        raise FeatureBundleError(f"unsupported entity {entity!r}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, csv.Error) as error:
        raise FeatureBundleError(f"cannot read {entity} mapping {path}: {error}") from error
    if len(rows) < 2 or len(rows[0]) < 2:
        raise FeatureBundleError(f"{entity} mapping must have a header and two columns")
    row_ids: list[str] = []
    source_ids: list[str] = []
    for row_number, row in enumerate(rows[1:], start=1):
        if len(row) < 2:
            raise FeatureBundleError(
                f"{entity} mapping row {row_number + 1} has fewer than two columns"
            )
        row_id = str(row[0]).strip()
        source_id = str(row[1]).strip()
        if row_id != str(row_number):
            raise FeatureBundleError(
                f"{entity} mapping row IDs must be consecutive one-based strings; "
                f"expected {row_number}, got {row_id!r}"
            )
        if not source_id:
            raise FeatureBundleError(
                f"{entity} mapping row {row_number + 1} has an empty source ID"
            )
        row_ids.append(row_id)
        source_ids.append(source_id)
    return EntityMapping(
        entity=entity,
        path=path,
        sha256=_sha256(path),
        row_ids=tuple(row_ids),
        source_ids=tuple(source_ids),
        headers=(str(rows[0][0]).strip(), str(rows[0][1]).strip()),
    )


_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _normalise_sequence(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    sequence = "".join(value.split()).upper()
    if not sequence or any(residue not in _PROTEIN_ALPHABET for residue in sequence):
        return None
    return sequence


def _strict_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureBundleError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_sha256(value: object, label: str) -> str:
    digest = _strict_nonempty_string(value, label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise FeatureBundleError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_evidence_json_artifact(
    record: object,
    label: str,
    *,
    resolver: Mapping[str, Path] | None,
    artifact_cache: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] | None,
) -> tuple[Path, dict[str, Any]]:
    """Verify/load an evidence artifact once within one validation transaction."""

    if not isinstance(record, Mapping):
        raise FeatureBundleError(f"{label} has no artifact record")
    digest = _strict_sha256(record.get("sha256"), f"{label} sha256")
    size = _manifest_count(record.get("bytes"), f"{label} bytes")
    recorded_path = _strict_nonempty_string(record.get("path"), f"{label} path")
    key = (digest, size, recorded_path)
    if artifact_cache is not None and key in artifact_cache:
        return artifact_cache[key]
    path = _verify_artifact(record, label, resolver=resolver)
    payload = _read_json(path)
    result = (path, payload)
    if artifact_cache is not None:
        artifact_cache[key] = result
    return result


def _validate_audit_row_binding(
    evidence: Mapping[str, Any],
    *,
    source_id: str,
    resolver: Mapping[str, Path] | None = None,
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> Path:
    """Bind self-contained row evidence to the exact archived audit artifact."""

    audit_path, audit_payload = _load_evidence_json_artifact(
        evidence.get("audit_artifact", {}),
        "sequence-resolution audit artifact",
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    if evidence.get("audit_artifact_sha256") != _sha256(audit_path):
        raise FeatureBundleError(
            "audit evidence artifact SHA-256 differs from its verified artifact"
        )
    audit_row = evidence.get("audit_row")
    if not isinstance(audit_row, Mapping):
        raise FeatureBundleError("audit evidence lacks its canonical audit_row")
    if audit_row.get("gene_symbol") != source_id:
        raise FeatureBundleError(
            "audit evidence row gene_symbol differs from mapping source_id"
        )
    if evidence.get("audit_row_sha256") != _canonical_json_sha256(audit_row):
        raise FeatureBundleError("audit evidence row SHA-256 is false")
    section = _strict_nonempty_string(
        evidence.get("audit_section"), "audit evidence section"
    )
    index = _strict_manifest_integer(
        evidence.get("audit_row_index"), "audit evidence row index"
    )
    audit: object = audit_payload
    for key in section.split("."):
        if not isinstance(audit, Mapping) or key not in audit:
            raise FeatureBundleError(
                "audit evidence section does not exist in its artifact"
            )
        audit = audit[key]
    if (
        not isinstance(audit, list)
        or index >= len(audit)
        or audit[index] != dict(audit_row)
    ):
        raise FeatureBundleError(
            "audit evidence row does not equal its artifact section/index"
        )
    return audit_path


def _validate_candidate_import_binding(
    evidence: Mapping[str, Any],
    *,
    source_id: str,
    resolver: Mapping[str, Path] | None = None,
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> Mapping[str, Any]:
    """Verify the exact candidate-source record used by an evidence decision."""

    import_path, payload = _load_evidence_json_artifact(
        evidence.get("candidate_import_artifact", {}),
        "protein candidate import artifact",
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    record = evidence.get("candidate_import_record")
    if not isinstance(record, Mapping):
        raise FeatureBundleError("protein evidence lacks candidate_import_record")
    if evidence.get("candidate_import_record_sha256") != _canonical_json_sha256(
        record
    ):
        raise FeatureBundleError("protein candidate import record SHA-256 is false")
    records = payload.get("records")
    if not isinstance(records, Mapping) or records.get(source_id) != dict(record):
        raise FeatureBundleError(
            "protein evidence record differs from its candidate import artifact"
        )
    return record


def _validate_target_classification_evidence(
    row: Mapping[str, Any],
    *,
    source_id: str,
    mapping_sha256: str,
    resolver: Mapping[str, Path] | None = None,
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    """Validate authoritative evidence before permitting a zero-UNK target.

    A status label is deliberately insufficient: only current NCBI Gene rows
    whose audited gene type is explicitly non-protein-coding are eligible.
    Unknown, generic, and protein-coding targets remain failed/unresolved.
    """

    target_class = _strict_nonempty_string(
        row.get("target_class"), "target_not_protein target_class"
    ).lower()
    expected_gene_type = _ZERO_UNK_GENE_TYPES.get(target_class)
    if expected_gene_type is None:
        raise FeatureBundleError(
            "target_not_protein target_class must be one of "
            + ", ".join(sorted(_ZERO_UNK_GENE_TYPES))
        )
    reason = _strict_nonempty_string(
        row.get("reason"), "target_not_protein reason"
    )
    if reason.lower() in _GENERIC_ZERO_UNK_REASONS:
        raise FeatureBundleError(
            "target_not_protein reason is generic/default rather than audited evidence"
        )
    evidence = row.get("classification_evidence")
    if not isinstance(evidence, Mapping):
        raise FeatureBundleError(
            "target_not_protein row lacks classification_evidence"
        )
    if evidence.get("schema") != TARGET_CLASSIFICATION_SCHEMA:
        raise FeatureBundleError(
            "target_not_protein classification evidence has a wrong schema"
        )
    if evidence.get("mapping_sha256") != mapping_sha256:
        raise FeatureBundleError(
            "target_not_protein classification evidence has a wrong mapping hash"
        )
    if evidence.get("source") != "NCBI Datasets Gene report":
        raise FeatureBundleError(
            "target_not_protein classification source is not the NCBI Gene report"
        )
    if _strict_nonempty_string(
        evidence.get("gene_symbol"), "classification gene_symbol"
    ) != source_id:
        raise FeatureBundleError(
            "target_not_protein classification gene_symbol differs from mapping source_id"
        )
    for key in ("audit_id", "source_gene_id", "current_gene_id"):
        _strict_nonempty_string(evidence.get(key), f"classification {key}")
    taxid = _strict_nonempty_string(evidence.get("taxid"), "classification taxid")
    if not taxid.isdecimal() or int(taxid) <= 0:
        raise FeatureBundleError("classification taxid must be a positive decimal string")
    if evidence.get("gene_type") != expected_gene_type:
        raise FeatureBundleError(
            "target_not_protein gene_type does not match its whitelisted target_class"
        )
    release = evidence.get("annotation_release")
    queried_at = evidence.get("queried_at")
    if not (
        isinstance(release, str)
        and release.strip()
        or isinstance(queried_at, str)
        and queried_at.strip()
    ):
        raise FeatureBundleError(
            "classification evidence requires annotation_release or queried_at"
        )
    _strict_sha256(
        evidence.get("source_record_sha256"),
        "classification source_record_sha256",
    )
    _validate_audit_row_binding(
        evidence,
        source_id=source_id,
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    _validate_candidate_import_binding(
        evidence,
        source_id=source_id,
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    audit_row = evidence["audit_row"]
    if (
        audit_row.get("source_gene_id") != evidence.get("source_gene_id")
        or audit_row.get("current_gene_id") != evidence.get("current_gene_id")
        or str(audit_row.get("taxid")) != taxid
        or audit_row.get("gene_type") != expected_gene_type
    ):
        raise FeatureBundleError(
            "target_not_protein classification fields differ from the audit row"
        )
    normalized = dict(evidence)
    normalized["taxid"] = taxid
    return {
        "target_class": target_class,
        "reason": reason,
        "taxid": taxid,
        "annotation_release": (
            str(release).strip()
            if isinstance(release, str) and release.strip()
            else str(queried_at).strip()
        ),
        "classification_evidence": normalized,
    }


def _validate_audited_missing_sequence_evidence(
    row: Mapping[str, Any],
    *,
    source_id: str,
    mapping_sha256: str,
    resolver: Mapping[str, Path] | None = None,
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    """Validate the separately audited, non-ESM missing-sequence state.

    This state is intentionally distinct from ``target_not_protein``.  It is
    permitted only for an exact row in the frozen resolution audit and never
    turns an unknown or protein-coding target into a resolved ESM feature.
    """

    target_class = _strict_nonempty_string(
        row.get("target_class"), "audited-missing target_class"
    ).lower()
    expected = _AUDITED_MISSING_CLASSES.get(target_class)
    if expected is None:
        raise FeatureBundleError(
            "audited-missing target_class must be unknown_target_class or "
            "protein_coding_no_valid_sequence"
        )
    reason = _strict_nonempty_string(row.get("reason"), "audited-missing reason")
    if reason.lower() in _GENERIC_ZERO_UNK_REASONS:
        raise FeatureBundleError(
            "audited-missing reason is generic/default rather than row evidence"
        )
    evidence = row.get("missing_sequence_evidence")
    if not isinstance(evidence, Mapping):
        raise FeatureBundleError(
            "audited-missing row lacks missing_sequence_evidence"
        )
    if evidence.get("schema") != AUDITED_MISSING_SEQUENCE_SCHEMA:
        raise FeatureBundleError("audited-missing evidence has a wrong schema")
    if evidence.get("mapping_sha256") != mapping_sha256:
        raise FeatureBundleError("audited-missing evidence has a wrong mapping hash")
    if evidence.get("source") != "Ethnobotany sequence-resolution audit":
        raise FeatureBundleError("audited-missing evidence has a wrong source")
    if _strict_nonempty_string(
        evidence.get("gene_symbol"), "audited-missing gene_symbol"
    ) != source_id:
        raise FeatureBundleError(
            "audited-missing gene_symbol differs from mapping source_id"
        )
    for key in ("audit_id", "source_gene_id"):
        _strict_nonempty_string(evidence.get(key), f"audited-missing {key}")
    expected_gene_type, expected_resolution_status = expected
    if evidence.get("gene_type") != expected_gene_type:
        raise FeatureBundleError(
            "audited-missing gene_type does not match target_class"
        )
    if evidence.get("resolution_status") != expected_resolution_status:
        raise FeatureBundleError(
            "audited-missing resolution_status does not match target_class"
        )
    current_gene_id = evidence.get("current_gene_id")
    if target_class == "protein_coding_no_valid_sequence":
        _strict_nonempty_string(
            current_gene_id, "audited-missing current_gene_id"
        )
    elif current_gene_id is not None and (
        not isinstance(current_gene_id, str) or not current_gene_id.strip()
    ):
        raise FeatureBundleError(
            "audited-missing current_gene_id must be null or a non-empty string"
        )
    taxid = _strict_nonempty_string(evidence.get("taxid"), "audited-missing taxid")
    if not taxid.isdecimal() or int(taxid) <= 0:
        raise FeatureBundleError(
            "audited-missing taxid must be a positive decimal string"
        )
    release = evidence.get("annotation_release")
    queried_at = evidence.get("queried_at")
    if not (
        isinstance(release, str)
        and release.strip()
        or isinstance(queried_at, str)
        and queried_at.strip()
    ):
        raise FeatureBundleError(
            "audited-missing evidence requires annotation_release or queried_at"
        )
    _strict_sha256(
        evidence.get("audit_artifact_sha256"),
        "audited-missing audit_artifact_sha256",
    )
    _strict_sha256(
        evidence.get("audit_row_sha256"), "audited-missing audit_row_sha256"
    )
    _validate_audit_row_binding(
        evidence,
        source_id=source_id,
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    _validate_candidate_import_binding(
        evidence,
        source_id=source_id,
        resolver=resolver,
        artifact_cache=artifact_cache,
    )
    audit_row = evidence["audit_row"]
    normalized_current = (
        str(audit_row.get("current_gene_id")).strip()
        if audit_row.get("current_gene_id") is not None
        else None
    )
    if (
        str(audit_row.get("source_gene_id", "")).strip()
        != evidence.get("source_gene_id")
        or normalized_current != current_gene_id
        or str(audit_row.get("taxid")) != taxid
        or audit_row.get("gene_type") != expected_gene_type
        or audit_row.get("status") != expected_resolution_status
        or str(audit_row.get("reason", "")).strip() != reason
    ):
        raise FeatureBundleError(
            "audited-missing fields/reason differ from the exact audit row"
        )
    normalized = dict(evidence)
    normalized["taxid"] = taxid
    return {
        "target_class": target_class,
        "reason": reason,
        "taxid": taxid,
        "annotation_release": (
            str(release).strip()
            if isinstance(release, str) and release.strip()
            else str(queried_at).strip()
        ),
        "missing_sequence_evidence": normalized,
    }


def _validate_sequence_set_row(
    row: Mapping[str, Any],
    *,
    source_id: str | None = None,
    mapping_sha256: str | None = None,
    resolver: Mapping[str, Path] | None = None,
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate a deterministic equal-weight set of distinct protein sequences."""

    if row.get("value") is not None or row.get("accession") is not None:
        raise FeatureBundleError(
            "resolved_sequence_set rows must not contain a single value/accession"
        )
    tier = _strict_nonempty_string(
        row.get("sequence_set_tier"), "sequence_set_tier"
    )
    if row.get("sequence_set_aggregation") != SEQUENCE_SET_AGGREGATION:
        raise FeatureBundleError("sequence-set aggregation contract is invalid")
    raw_members = row.get("sequence_set")
    if not isinstance(raw_members, list) or not raw_members:
        raise FeatureBundleError("resolved_sequence_set row has no members")
    if _manifest_count(
        row.get("sequence_set_count"), "sequence_set_count"
    ) != len(raw_members):
        raise FeatureBundleError("sequence_set_count disagrees with its members")
    members: list[dict[str, Any]] = []
    for index, raw_member in enumerate(raw_members):
        if not isinstance(raw_member, Mapping):
            raise FeatureBundleError(f"sequence_set[{index}] is not an object")
        sequence = _normalise_sequence(raw_member.get("sequence"))
        digest = raw_member.get("sequence_sha256")
        if (
            sequence is None
            or sequence != raw_member.get("sequence")
            or digest != hashlib.sha256(sequence.encode("ascii")).hexdigest()
        ):
            raise FeatureBundleError(
                f"sequence_set[{index}] has a noncanonical sequence or false digest"
            )
        accessions = raw_member.get("accessions")
        if (
            not isinstance(accessions, list)
            or not accessions
            or any(not isinstance(value, str) or not value.strip() for value in accessions)
            or accessions != sorted(set(accessions))
        ):
            raise FeatureBundleError(
                f"sequence_set[{index}] has invalid accessions"
            )
        taxid = raw_member.get("taxid")
        if not isinstance(taxid, str) or not taxid.isdecimal() or int(taxid) <= 0:
            raise FeatureBundleError(f"sequence_set[{index}] has an invalid taxid")
        releases = raw_member.get("annotation_releases")
        if (
            not isinstance(releases, list)
            or any(not isinstance(value, str) or not value.strip() for value in releases)
            or releases != sorted(set(releases))
        ):
            raise FeatureBundleError(
                f"sequence_set[{index}] has invalid annotation releases"
            )
        members.append(
            {
                "sequence": sequence,
                "sequence_sha256": digest,
                "accessions": list(accessions),
                "taxid": taxid,
                "annotation_releases": list(releases),
            }
        )
    digests = [member["sequence_sha256"] for member in members]
    if digests != sorted(digests) or len(digests) != len(set(digests)):
        raise FeatureBundleError(
            "sequence_set members must be distinct and SHA-sorted"
        )
    if len({member["taxid"] for member in members}) != 1:
        raise FeatureBundleError("sequence_set members have inconsistent taxids")
    if source_id is not None or mapping_sha256 is not None:
        if source_id is None or mapping_sha256 is None:
            raise FeatureBundleError(
                "sequence-set identity validation requires source ID and mapping hash"
            )
        evidence = row.get("sequence_resolution_evidence")
        if not isinstance(evidence, Mapping):
            raise FeatureBundleError(
                "resolved_sequence_set lacks sequence_resolution_evidence"
            )
        if (
            evidence.get("schema") != SEQUENCE_SET_RESOLUTION_SCHEMA
            or evidence.get("mapping_sha256") != mapping_sha256
            or evidence.get("source") != "Ethnobotany sequence-resolution audit"
            or evidence.get("selected_tier") != tier
            or evidence.get("resolution_hierarchy") != list(FROZEN_SEQUENCE_TIERS)
        ):
            raise FeatureBundleError(
                "sequence-set resolution identity/hierarchy evidence is invalid"
            )
        if _strict_nonempty_string(
            evidence.get("gene_symbol"), "sequence-set gene_symbol"
        ) != source_id:
            raise FeatureBundleError(
                "sequence-set gene_symbol differs from mapping source_id"
            )
        for key in ("audit_id", "source_gene_id"):
            _strict_nonempty_string(evidence.get(key), f"sequence-set {key}")
        expected_taxid = _strict_nonempty_string(
            evidence.get("expected_taxid"), "sequence-set expected_taxid"
        )
        if not expected_taxid.isdecimal() or int(expected_taxid) <= 0:
            raise FeatureBundleError(
                "sequence-set expected_taxid must be a positive decimal string"
            )
        if {member["taxid"] for member in members} != {expected_taxid}:
            raise FeatureBundleError(
                "sequence-set member taxids differ from audited expected_taxid"
            )
        _strict_sha256(
            evidence.get("audit_artifact_sha256"),
            "sequence-set audit_artifact_sha256",
        )
        _strict_sha256(
            evidence.get("audit_row_sha256"), "sequence-set audit_row_sha256"
        )
        _validate_audit_row_binding(
            evidence,
            source_id=source_id,
            resolver=resolver,
            artifact_cache=artifact_cache,
        )
        candidate_record = _validate_candidate_import_binding(
            evidence,
            source_id=source_id,
            resolver=resolver,
            artifact_cache=artifact_cache,
        )
        audit_candidates = candidate_record.get("candidates")
        if not isinstance(audit_candidates, list) or any(
            not isinstance(candidate, Mapping) for candidate in audit_candidates
        ):
            raise FeatureBundleError(
                "sequence-set audit row lacks its validated candidate list"
            )
        tier_evidence = evidence.get("tier_evidence")
        selected_index = FROZEN_SEQUENCE_TIERS.index(tier) if tier in FROZEN_SEQUENCE_TIERS else -1
        if (
            selected_index < 0
            or not isinstance(tier_evidence, list)
            or len(tier_evidence) != selected_index + 1
        ):
            raise FeatureBundleError(
                "sequence-set tier evidence does not stop at the selected tier"
            )
        for index, tier_record in enumerate(tier_evidence):
            if not isinstance(tier_record, Mapping):
                raise FeatureBundleError("sequence-set tier evidence is malformed")
            digests_at_tier = tier_record.get("sequence_sha256s")
            accessions_at_tier = tier_record.get("accessions")
            if (
                tier_record.get("tier") != FROZEN_SEQUENCE_TIERS[index]
                or not isinstance(digests_at_tier, list)
                or digests_at_tier != sorted(set(digests_at_tier))
                or any(
                    not isinstance(value, str)
                    or _SHA256_PATTERN.fullmatch(value) is None
                    for value in digests_at_tier
                )
                or not isinstance(accessions_at_tier, list)
                or accessions_at_tier != sorted(set(accessions_at_tier))
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in accessions_at_tier
                )
            ):
                raise FeatureBundleError("sequence-set tier evidence is invalid")
            tier_name = FROZEN_SEQUENCE_TIERS[index]
            expected_candidates = []
            for candidate in audit_candidates:
                accession = str(candidate.get("accession", ""))
                source = str(candidate.get("source", "")).lower()
                select_category = str(
                    candidate.get("select_category", "") or ""
                ).upper()
                reviewed = candidate.get("reviewed") is True
                include = (
                    tier_name == "mane_select"
                    and select_category == "MANE_SELECT"
                    or tier_name == "refseq_select"
                    and select_category == "REFSEQ_SELECT"
                    or tier_name == "reviewed_uniprot"
                    and source == "uniprot"
                    and reviewed
                    or tier_name == "known_refseq_np"
                    and source == "refseq"
                    and accession.startswith("NP_")
                    or tier_name == "unreviewed_uniprot"
                    and source == "uniprot"
                    and not reviewed
                    or tier_name == "complete_refseq"
                    and source == "refseq"
                )
                if include:
                    expected_candidates.append(candidate)
            expected_digests = sorted(
                {
                    _strict_sha256(
                        candidate.get("sequence_sha256"),
                        "audit candidate sequence_sha256",
                    )
                    for candidate in expected_candidates
                }
            )
            expected_accessions = sorted(
                {
                    _strict_nonempty_string(
                        candidate.get("accession"), "audit candidate accession"
                    )
                    for candidate in expected_candidates
                }
            )
            if (
                digests_at_tier != expected_digests
                or accessions_at_tier != expected_accessions
                or _strict_manifest_integer(
                    tier_record.get("candidate_count"),
                    "sequence-set tier candidate_count",
                )
                != len(expected_candidates)
                or _strict_manifest_integer(
                    tier_record.get("sequence_group_count"),
                    "sequence-set tier sequence_group_count",
                )
                != len(expected_digests)
            ):
                raise FeatureBundleError(
                    "sequence-set tier evidence differs from the archived audit row"
                )
            if index < selected_index and (digests_at_tier or accessions_at_tier):
                raise FeatureBundleError(
                    "sequence-set selected a lower tier despite earlier evidence"
                )
        if tier_evidence[-1]["sequence_sha256s"] != digests:
            raise FeatureBundleError(
                "sequence-set members differ from the selected-tier audit evidence"
            )
        member_accessions = sorted(
            {
                accession
                for member in members
                for accession in member["accessions"]
            }
        )
        if tier_evidence[-1]["accessions"] != member_accessions:
            raise FeatureBundleError(
                "sequence-set accessions differ from selected-tier audit evidence"
            )
    return tuple(members)


def _truthy(value: object) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}
    )


def _protein_candidate(
    candidate: object, *, expected_taxid: str
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(candidate, Mapping):
        return None, "candidate is not a JSON object"
    accession = str(candidate.get("accession", candidate.get("uniprot_id", ""))).strip()
    sequence = _normalise_sequence(candidate.get("sequence"))
    if not accession:
        return None, "candidate has no accession"
    if sequence is None:
        return None, f"candidate {accession} has an empty or non-canonical FASTA sequence"
    reported_length = candidate.get(
        "reported_length", candidate.get("product_report_length")
    )
    try:
        reported_length = int(reported_length)
    except (TypeError, ValueError):
        return None, f"candidate {accession} has no valid product-report length"
    if reported_length != len(sequence):
        return None, (
            f"candidate {accession} report/FASTA length mismatch: "
            f"{reported_length} != {len(sequence)}"
        )
    if _truthy(candidate.get("partial", False)):
        return None, f"candidate {accession} is marked partial"
    if _truthy(candidate.get("low_quality", candidate.get("low-quality", False))):
        return None, f"candidate {accession} is marked low-quality"
    taxid = str(candidate.get("taxid", "")).strip()
    if not taxid:
        return None, f"candidate {accession} has no taxid"
    if taxid != expected_taxid:
        return None, (
            f"candidate {accession} taxid {taxid!r} does not match expected "
            f"taxid {expected_taxid!r}"
        )
    source = str(candidate.get("source", "")).strip().lower()
    if source not in {"refseq", "uniprot"}:
        return None, f"candidate {accession} source must be refseq or uniprot"
    select_category = str(
        candidate.get("select_category", candidate.get("selectCategory", ""))
    ).strip().upper()
    reviewed = _truthy(candidate.get("reviewed", False))
    sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    return {
        "accession": accession,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256,
        "reported_length": reported_length,
        "taxid": taxid,
        "source": source,
        "select_category": select_category or None,
        "reviewed": reviewed,
        "annotation_release": candidate.get("annotation_release"),
    }, None


def _resolved_protein(
    chosen: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    rule: str,
    candidate_failures: Sequence[str],
    resolution_attempts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    equivalent = sorted(
        str(candidate["accession"])
        for candidate in candidates
        if candidate["sequence_sha256"] == chosen["sequence_sha256"]
    )
    return {
        "status": "resolved",
        "value": chosen["sequence"],
        "error": None,
        "reason": None,
        "accession": chosen["accession"],
        "equivalent_accessions": equivalent,
        "sequence_sha256": chosen["sequence_sha256"],
        "resolution_rule": rule,
        "select_category": chosen.get("select_category"),
        "candidate_source": chosen.get("source"),
        "reviewed": bool(chosen.get("reviewed")),
        "taxid": chosen.get("taxid"),
        "annotation_release": chosen.get("annotation_release"),
        "candidate_failures": list(candidate_failures),
        "resolution_attempts": [dict(attempt) for attempt in resolution_attempts],
        "target_class": "protein_target",
    }


def _ambiguous_protein(
    *,
    source_id: str,
    rule: str,
    count: int,
    candidate_failures: Sequence[str],
    resolution_attempts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "status": "failed",
        "value": None,
        "error": f"ambiguous protein candidates for {source_id} at {rule}: count={count}",
        "reason": None,
        "accession": None,
        "equivalent_accessions": [],
        "sequence_sha256": None,
        "resolution_rule": rule,
        "select_category": None,
        "candidate_source": None,
        "reviewed": None,
        "taxid": None,
        "annotation_release": None,
        "candidate_failures": list(candidate_failures),
        "resolution_attempts": [dict(attempt) for attempt in resolution_attempts],
        "target_class": "protein_target",
    }


def _resolved_protein_sequence_set(
    selected: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    candidate_failures: Sequence[str],
    resolution_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in selected:
        groups.setdefault(str(candidate["sequence_sha256"]), []).append(candidate)
    members: list[dict[str, Any]] = []
    for digest in sorted(groups):
        group = groups[digest]
        sequences = {str(candidate["sequence"]) for candidate in group}
        taxids = {str(candidate["taxid"]) for candidate in group}
        if len(sequences) != 1 or len(taxids) != 1:
            raise FeatureBundleError(
                "sequence digest group has inconsistent sequence/taxid content"
            )
        releases = sorted(
            {
                str(candidate["annotation_release"]).strip()
                for candidate in group
                if isinstance(candidate.get("annotation_release"), str)
                and str(candidate["annotation_release"]).strip()
            }
        )
        members.append(
            {
                "sequence": next(iter(sequences)),
                "sequence_sha256": digest,
                "accessions": sorted(
                    {str(candidate["accession"]) for candidate in group}
                ),
                "taxid": next(iter(taxids)),
                "annotation_releases": releases,
            }
        )
    return {
        "status": "resolved_sequence_set",
        "value": None,
        "error": None,
        "reason": None,
        "accession": None,
        "equivalent_accessions": [],
        "sequence_sha256": None,
        "resolution_rule": "mean_distinct_sequences_at_first_available_tier",
        "select_category": None,
        "candidate_source": None,
        "reviewed": None,
        "taxid": members[0]["taxid"],
        "annotation_release": None,
        "candidate_failures": list(candidate_failures),
        "resolution_attempts": [dict(attempt) for attempt in resolution_attempts],
        "target_class": "protein_target",
        "sequence_set_tier": tier,
        "sequence_set_aggregation": SEQUENCE_SET_AGGREGATION,
        "sequence_set_count": len(members),
        "sequence_set": members,
    }


def _cache_value(
    entity: str,
    value: object,
    *,
    source_id: str,
    mapping_sha256: str,
    allowed_accession: str | None = None,
    ambiguous_sequence_policy: str = "strict",
    artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    if entity == "compound":
        if isinstance(value, Mapping):
            value = value.get("smiles", value.get("canonical_smiles"))
        if isinstance(value, str) and value.strip():
            return {
                "status": "resolved",
                "value": value.strip(),
                "error": None,
                "accession": None,
                "resolution_rule": "unique_source_record",
                "target_class": "chemical_compound",
            }
        return {
            "status": "failed",
            "value": None,
            "error": "cache value has no non-empty SMILES string",
            "accession": None,
            "resolution_rule": None,
            "target_class": "chemical_compound",
        }

    if isinstance(value, Mapping):
        target_class = str(
            value.get("target_class", value.get("target_type", ""))
        ).strip().lower().replace("_", "-")
        status = str(value.get("status", "")).strip().lower().replace("_", "-")
        if status == "audited-missing-sequence":
            try:
                missing = _validate_audited_missing_sequence_evidence(
                    value,
                    source_id=source_id,
                    mapping_sha256=mapping_sha256,
                    artifact_cache=artifact_cache,
                )
            except FeatureBundleError as error:
                return {
                    "status": "failed",
                    "value": None,
                    "error": str(error),
                    "reason": value.get("reason"),
                    "accession": None,
                    "equivalent_accessions": [],
                    "sequence_sha256": None,
                    "resolution_rule": None,
                    "select_category": None,
                    "candidate_source": None,
                    "reviewed": None,
                    "taxid": value.get("taxid"),
                    "annotation_release": value.get("annotation_release"),
                    "candidate_failures": [],
                    "resolution_attempts": [],
                    "target_class": target_class or None,
                    "missing_sequence_evidence": value.get(
                        "missing_sequence_evidence"
                    ),
                }
            return {
                "status": "audited_missing_sequence",
                "value": None,
                "error": missing["reason"],
                "reason": missing["reason"],
                "accession": None,
                "equivalent_accessions": [],
                "sequence_sha256": None,
                "resolution_rule": "explicit_audited_missing_sequence",
                "select_category": None,
                "candidate_source": None,
                "reviewed": None,
                "taxid": missing["taxid"],
                "annotation_release": missing["annotation_release"],
                "candidate_failures": [],
                "resolution_attempts": [],
                "target_class": missing["target_class"],
                "missing_sequence_evidence": missing[
                    "missing_sequence_evidence"
                ],
            }
        if status == "resolved-sequence-set":
            try:
                members = _validate_sequence_set_row(
                    value,
                    source_id=source_id,
                    mapping_sha256=mapping_sha256,
                    artifact_cache=artifact_cache,
                )
            except FeatureBundleError as error:
                return {
                    "status": "failed",
                    "value": None,
                    "error": str(error),
                    "reason": None,
                    "accession": None,
                    "equivalent_accessions": [],
                    "sequence_sha256": None,
                    "resolution_rule": None,
                    "target_class": "protein_target",
                }
            normalized = dict(value)
            normalized.update(
                {
                    "status": "resolved_sequence_set",
                    "value": None,
                    "error": None,
                    "accession": None,
                    "sequence_sha256": None,
                    "target_class": "protein_target",
                    "sequence_set": [dict(member) for member in members],
                }
            )
            return normalized
        if status == "target-not-protein" or target_class in _ZERO_UNK_GENE_TYPES:
            try:
                classification = _validate_target_classification_evidence(
                    value,
                    source_id=source_id,
                    mapping_sha256=mapping_sha256,
                    artifact_cache=artifact_cache,
                )
            except FeatureBundleError as error:
                return {
                    "status": "failed",
                    "value": None,
                    "error": str(error),
                    "reason": value.get("reason"),
                    "accession": None,
                    "equivalent_accessions": [],
                    "sequence_sha256": None,
                    "resolution_rule": None,
                    "select_category": None,
                    "candidate_source": None,
                    "reviewed": None,
                    "taxid": value.get("taxid"),
                    "annotation_release": value.get("annotation_release"),
                    "candidate_failures": [],
                    "resolution_attempts": [],
                    "target_class": target_class or None,
                    "classification_evidence": value.get(
                        "classification_evidence"
                    ),
                }
            return {
                "status": "target_not_protein",
                "value": None,
                "error": classification["reason"],
                "reason": classification["reason"],
                "accession": None,
                "equivalent_accessions": [],
                "sequence_sha256": None,
                "resolution_rule": "explicit_target_classification",
                "select_category": None,
                "candidate_source": None,
                "reviewed": None,
                "taxid": classification["taxid"],
                "annotation_release": classification["annotation_release"],
                "candidate_failures": [],
                "resolution_attempts": [],
                "target_class": classification["target_class"],
                "classification_evidence": classification[
                    "classification_evidence"
                ],
            }

        if "candidates" in value:
            expected_taxid = str(
                value.get("expected_taxid", value.get("taxid", ""))
            ).strip()
            if not expected_taxid:
                return _ambiguous_protein(
                    source_id=source_id,
                    rule="missing_expected_taxid",
                    count=0,
                    candidate_failures=["target record has no expected taxid"],
                )
            raw_candidates = value.get("candidates")
            if not isinstance(raw_candidates, list):
                raw_candidates = []
            candidates: list[dict[str, Any]] = []
            candidate_failures: list[str] = []
            for candidate in raw_candidates:
                parsed, failure = _protein_candidate(
                    candidate, expected_taxid=expected_taxid
                )
                if parsed is not None:
                    candidates.append(parsed)
                elif failure is not None:
                    candidate_failures.append(failure)
            if allowed_accession is not None:
                selected = [
                    candidate
                    for candidate in candidates
                    if candidate["accession"] == allowed_accession
                ]
                if len(selected) == 1:
                    return _resolved_protein(
                        selected[0],
                        candidates,
                        rule="accession_allowlist",
                        candidate_failures=candidate_failures,
                        resolution_attempts=[
                            {
                                "rule": "accession_allowlist",
                                "candidate_count": len(selected),
                                "accessions": sorted(
                                    candidate["accession"] for candidate in selected
                                ),
                            }
                        ],
                    )
                return _ambiguous_protein(
                    source_id=source_id,
                    rule="accession_allowlist",
                    count=len(selected),
                    candidate_failures=candidate_failures,
                    resolution_attempts=[
                        {
                            "rule": "accession_allowlist",
                            "candidate_count": len(selected),
                            "accessions": sorted(
                                candidate["accession"] for candidate in selected
                            ),
                        }
                    ],
                )

            hierarchy = (
                (
                    "mane_select",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["select_category"] == "MANE_SELECT"
                    ],
                ),
                (
                    "refseq_select",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["select_category"] == "REFSEQ_SELECT"
                    ],
                ),
                (
                    "reviewed_uniprot",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["source"] == "uniprot" and candidate["reviewed"]
                    ],
                ),
                (
                    "known_refseq_np",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["source"] == "refseq"
                        and str(candidate["accession"]).startswith("NP_")
                    ],
                ),
                (
                    "unreviewed_uniprot",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["source"] == "uniprot"
                        and not candidate["reviewed"]
                    ],
                ),
                (
                    "complete_refseq",
                    [
                        candidate
                        for candidate in candidates
                        if candidate["source"] == "refseq"
                    ],
                ),
            )
            resolution_attempts: list[dict[str, Any]] = []
            for rule, selected in hierarchy:
                groups: dict[str, list[dict[str, Any]]] = {}
                for candidate in selected:
                    groups.setdefault(candidate["sequence_sha256"], []).append(candidate)
                resolution_attempts.append(
                    {
                        "rule": rule,
                        "candidate_count": len(selected),
                        "sequence_group_count": len(groups),
                        "sequence_sha256s": sorted(groups),
                        "accessions": sorted(
                            candidate["accession"] for candidate in selected
                        ),
                    }
                )
                if not groups:
                    continue
                if len(groups) == 1:
                    equivalent_group = next(iter(groups.values()))
                    chosen = sorted(
                        equivalent_group, key=lambda item: item["accession"]
                    )[0]
                    return _resolved_protein(
                        chosen,
                        candidates,
                        rule=rule,
                        candidate_failures=candidate_failures,
                        resolution_attempts=resolution_attempts,
                    )
                # A multi-sequence first available tier is never inferred from
                # generic candidates.  The audit must import an explicit,
                # row-bound resolved_sequence_set record that the validator
                # checks against the frozen tier evidence.
                suffix = (
                    "explicit audited resolved_sequence_set required"
                    if ambiguous_sequence_policy == "mean-distinct-first-tier"
                    else "strict policy rejects multiple distinct sequences"
                )
                return _ambiguous_protein(
                    source_id=source_id,
                    rule=f"{rule}: {suffix}",
                    count=len(groups),
                    candidate_failures=candidate_failures,
                    resolution_attempts=resolution_attempts,
                )
            return _ambiguous_protein(
                source_id=source_id,
                rule="frozen_resolution_hierarchy_has_no_sequence",
                count=0,
                candidate_failures=candidate_failures,
                resolution_attempts=resolution_attempts,
            )

    return {
        "status": "failed",
        "value": None,
        "error": (
            "protein sequence lacks strict candidate-resolution provenance; "
            "provide candidates, a unique NCBI Select record, or an accession allowlist"
        ),
        "reason": None,
        "accession": None,
        "equivalent_accessions": [],
        "sequence_sha256": None,
        "resolution_rule": None,
        "select_category": None,
        "candidate_source": None,
        "reviewed": None,
        "taxid": None,
        "annotation_release": None,
        "candidate_failures": [],
        "target_class": "unknown",
        "classification_evidence": None,
    }


def update_retrieval_cache(
    *,
    entity: str,
    mapping_path: Path | str,
    output_path: Path | str,
    import_path: Path | str | None = None,
    missing_sequence_policy: str = "strict",
    ambiguous_sequence_policy: str = "strict",
    accession_allowlist: Path | str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Create or resume an ID-keyed SMILES/sequence cache."""

    entity = str(entity).strip().lower()
    if entity not in BIOLOGICAL_ENTITIES:
        raise FeatureBundleError("retrieval caches are defined for compound/protein")
    missing_sequence_policy = str(missing_sequence_policy).strip().lower()
    if entity == "compound" and missing_sequence_policy != "strict":
        raise FeatureBundleError(
            "non-strict missing-sequence policies are valid only for protein caches"
        )
    if missing_sequence_policy not in {
        "strict",
        "zero-unk",
        "audited-missing-zero",
    }:
        raise FeatureBundleError(
            "missing sequence policy must be strict, zero-unk, or audited-missing-zero"
        )
    ambiguous_sequence_policy = str(ambiguous_sequence_policy).strip().lower()
    if ambiguous_sequence_policy not in {"strict", "mean-distinct-first-tier"}:
        raise FeatureBundleError(
            "ambiguous sequence policy must be strict or mean-distinct-first-tier"
        )
    if entity == "compound" and ambiguous_sequence_policy != "strict":
        raise FeatureBundleError(
            "ambiguous sequence aggregation is valid only for protein caches"
        )
    mapping = read_entity_mapping(mapping_path, entity)
    output_path = Path(output_path).resolve()
    previous_rows: dict[str, Mapping[str, Any]] = {}
    if output_path.exists():
        previous = _read_json(output_path)
        if previous.get("schema") != CACHE_SCHEMA:
            raise FeatureBundleError(f"existing cache {output_path} has a wrong schema")
        if previous.get("entity") != entity:
            raise FeatureBundleError(f"existing cache {output_path} has a wrong entity")
        if previous.get("mapping_sha256") != mapping.sha256:
            raise FeatureBundleError(
                f"existing cache {output_path} belongs to a different mapping"
            )
        saved_rows = previous.get("rows", [])
        if not isinstance(saved_rows, list) or any(
            not isinstance(row, Mapping) for row in saved_rows
        ):
            raise FeatureBundleError(
                f"existing cache {output_path} rows must be a JSON array of objects"
            )
        raw_previous_rows = list(saved_rows)
        previous_pairs = [
            (row.get("row_id"), row.get("source_id"))
            for row in raw_previous_rows
        ]
        if previous_pairs != list(zip(mapping.row_ids, mapping.source_ids)) or any(
            not isinstance(row_id, str) or not isinstance(source_id, str)
            for row_id, source_id in previous_pairs
        ):
            raise FeatureBundleError(
                f"existing cache {output_path} row_id/source_id pairs do not "
                "exactly match the canonical mapping"
            )
        previous_rows = {row["row_id"]: row for row in raw_previous_rows}

    imported: Mapping[str, Any] = {}
    import_record: dict[str, Any] | None = None
    if import_path is not None:
        import_path = Path(import_path).resolve()
        imported_payload = _read_json(import_path)
        if "records" in imported_payload:
            records = imported_payload.get("records")
            if not isinstance(records, Mapping):
                raise FeatureBundleError("cache import records must be a JSON object")
            imported = records
            import_metadata = imported_payload.get("metadata", {})
            if not isinstance(import_metadata, Mapping):
                raise FeatureBundleError("cache import metadata must be a JSON object")
        else:
            imported = imported_payload
            import_metadata = {}
        import_record = _artifact(import_path)
        import_record["metadata"] = dict(import_metadata)
        unknown_imports = sorted(set(imported).difference(mapping.source_ids))
        if unknown_imports:
            raise FeatureBundleError(
                f"cache import has source IDs outside the mapping: {unknown_imports[:5]}"
            )

    allowlist: Mapping[str, Any] = {}
    allowlist_record: dict[str, Any] | None = None
    if accession_allowlist is not None:
        allowlist_path = Path(accession_allowlist).resolve()
        allowlist = _read_json(allowlist_path)
        allowlist_record = _artifact(allowlist_path)
        if any(not isinstance(value, str) or not value.strip() for value in allowlist.values()):
            raise FeatureBundleError("accession allowlist values must be non-empty strings")
        unknown_allowlist = sorted(set(allowlist).difference(mapping.source_ids))
        if unknown_allowlist:
            raise FeatureBundleError(
                f"accession allowlist has IDs outside the mapping: {unknown_allowlist[:5]}"
            )

    evidence_artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ] = {}
    rows: list[dict[str, Any]] = []
    for row_id, source_id in zip(mapping.row_ids, mapping.source_ids):
        old = previous_rows.get(row_id, {})
        row: dict[str, Any] = {
            "row_id": row_id,
            "source_id": source_id,
            "status": str(old.get("status", "pending")),
            "attempts": _manifest_count(
                old.get("attempts", 0), f"cache attempts for {entity} row {row_id}"
            ),
            "value": old.get("value"),
            "error": old.get("error"),
            "reason": old.get("reason"),
            "accession": old.get("accession"),
            "equivalent_accessions": old.get("equivalent_accessions", []),
            "sequence_sha256": old.get("sequence_sha256"),
            "resolution_rule": old.get("resolution_rule"),
            "select_category": old.get("select_category"),
            "candidate_source": old.get("candidate_source"),
            "reviewed": old.get("reviewed"),
            "taxid": old.get("taxid"),
            "annotation_release": old.get("annotation_release"),
            "candidate_failures": old.get("candidate_failures", []),
            "resolution_attempts": old.get("resolution_attempts", []),
            "target_class": old.get("target_class"),
            "classification_evidence": old.get("classification_evidence"),
            "missing_sequence_evidence": old.get("missing_sequence_evidence"),
            "sequence_set_tier": old.get("sequence_set_tier"),
            "sequence_set_aggregation": old.get("sequence_set_aggregation"),
            "sequence_set_count": old.get("sequence_set_count"),
            "sequence_set": old.get("sequence_set", []),
            "sequence_resolution_evidence": old.get(
                "sequence_resolution_evidence"
            ),
        }
        if old and old.get("source_id") != source_id:
            raise FeatureBundleError(
                f"cache source ID changed for {entity} row {row_id}"
            )
        if source_id in imported:
            row["attempts"] += 1
            resolved = _cache_value(
                entity,
                imported[source_id],
                source_id=source_id,
                mapping_sha256=mapping.sha256,
                allowed_accession=(
                    str(allowlist[source_id]).strip()
                    if source_id in allowlist
                    else None
                ),
                ambiguous_sequence_policy=ambiguous_sequence_policy,
                artifact_cache=evidence_artifact_cache,
            )
            row.update(resolved)
        rows.append(row)

    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "resolved",
            "resolved_sequence_set",
            "target_not_protein",
            "audited_missing_sequence",
            "pending",
            "failed",
        )
    }
    permitted = status_counts["resolved"] + status_counts["resolved_sequence_set"]
    if entity == "protein" and missing_sequence_policy in {
        "zero-unk",
        "audited-missing-zero",
    }:
        permitted += status_counts["target_not_protein"]
    if entity == "protein" and missing_sequence_policy == "audited-missing-zero":
        permitted += status_counts["audited_missing_sequence"]
    formal_eligible = permitted == len(rows)
    protein_denominator = (
        len(rows)
        - status_counts["target_not_protein"]
        - status_counts["audited_missing_sequence"]
    )
    resolved_sequence_count = (
        status_counts["resolved"] + status_counts["resolved_sequence_set"]
    )
    resolution_rule_counts: dict[str, int] = {}
    for row in rows:
        rule = row.get("resolution_rule")
        if rule:
            resolution_rule_counts[str(rule)] = resolution_rule_counts.get(str(rule), 0) + 1
    taxids = sorted(
        {str(row["taxid"]) for row in rows if row.get("taxid") not in {None, ""}}
    )
    annotation_releases = sorted(
        {
            str(row["annotation_release"])
            for row in rows
            if row.get("annotation_release") not in {None, ""}
        }
    )
    rejected_candidate_count = sum(
        len(row.get("candidate_failures", [])) for row in rows
    )
    payload: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "entity": entity,
        "kind": "pubchem_smiles" if entity == "compound" else "protein_sequence",
        "mapping": _artifact(mapping.path),
        "mapping_sha256": mapping.sha256,
        "row_id_space": "protocol_raw_id",
        "source_id_header": mapping.headers[1],
        "rows": rows,
        "status_counts": status_counts,
        "missing_sequence_policy": missing_sequence_policy,
        "ambiguous_sequence_policy": ambiguous_sequence_policy,
        "target_not_protein_count": status_counts["target_not_protein"],
        "audited_missing_sequence_count": status_counts[
            "audited_missing_sequence"
        ],
        "masked_missing_count": (
            status_counts["target_not_protein"]
            + status_counts["audited_missing_sequence"]
        ),
        "resolved_single_sequence_count": status_counts["resolved"],
        "resolved_sequence_set_count": status_counts["resolved_sequence_set"],
        "resolved_sequence_count": resolved_sequence_count,
        "overall_sequence_resolution_rate": (
            resolved_sequence_count / len(rows) if rows else 0.0
        ),
        "protein_target_resolution_rate": (
            resolved_sequence_count / protein_denominator
            if protein_denominator
            else 0.0
        ),
        "resolution_rule_counts": resolution_rule_counts,
        "taxids": taxids,
        "annotation_releases": annotation_releases,
        "rejected_candidate_count": rejected_candidate_count,
        "formal_eligible": formal_eligible,
        "accession_allowlist": allowlist_record,
        "imports": (
            list(_read_json(output_path).get("imports", []))
            if output_path.exists()
            else []
        ),
    }
    if entity == "protein":
        payload["candidate_validation"] = {
            "required": [
                "accession",
                "FASTA sequence",
                "matching expected taxid",
                "product-report length equal to FASTA length",
                "canonical 20-amino-acid alphabet",
                "not partial",
                "not low-quality",
            ],
            "resolution_hierarchy": [
                "explicit accession allowlist override",
                "MANE_SELECT distinct-sequence tier",
                "REFSEQ_SELECT distinct-sequence tier",
                "reviewed UniProt distinct-sequence tier",
                "known RefSeq NP_ distinct-sequence tier",
                "unreviewed UniProt distinct-sequence tier",
                "complete RefSeq distinct-sequence tier (XP_ only here)",
            ],
            "tier_rule": (
                "stop at the first non-empty tier after sequence-SHA deduplication; "
                "one SHA yields a single sequence and multiple SHAs require an "
                "explicit audit-bound equal-weight sequence set"
            ),
            "first_longest_or_lower_tier_fallback_after_ambiguity": False,
        }
    if import_record is not None and all(
        record.get("sha256") != import_record["sha256"]
        for record in payload["imports"]
        if isinstance(record, Mapping)
    ):
        payload["imports"].append(import_record)
    # Reuse the same strict content validator used downstream before a cache
    # may advertise formal eligibility. Pending/failed rows remain resumable,
    # but a malformed resolved/zero-UNK row is never hidden by status counts.
    derived_mask = np.asarray(
        [
            row["status"]
            in {"target_not_protein", "audited_missing_sequence"}
            for row in rows
        ],
        dtype=np.bool_,
    )
    _validate_retrieval_cache_for_mask(
        output_path,
        mapping,
        derived_mask,
        missing_sequence_policy=missing_sequence_policy,
        payload_override=payload,
        require_formal=False,
    )
    _atomic_json(output_path, payload)
    if require_complete and not payload["formal_eligible"]:
        raise IncompleteFeatureBundleError(
            f"{entity} retrieval cache remains incomplete at {output_path}: "
            f"{status_counts}"
        )
    return payload


def _row_sidecar(mapping: EntityMapping) -> dict[str, Any]:
    return {
        "schema": FEATURE_ROW_IDS_SCHEMA,
        "entity": mapping.entity,
        "id_space": "protocol_raw_id",
        "mapping_sha256": mapping.sha256,
        "row_ids": list(mapping.row_ids),
    }


def _load_retrieval_cache(path: Path, mapping: EntityMapping) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != CACHE_SCHEMA:
        raise FeatureBundleError(f"retrieval cache {path} has a wrong schema")
    if payload.get("entity") != mapping.entity:
        raise FeatureBundleError(f"retrieval cache {path} has a wrong entity")
    if payload.get("mapping_sha256") != mapping.sha256:
        raise FeatureBundleError(f"retrieval cache {path} has a wrong mapping hash")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise FeatureBundleError(f"retrieval cache {path} rows are malformed")
    cache_pairs = [(row.get("row_id"), row.get("source_id")) for row in rows]
    if cache_pairs != list(zip(mapping.row_ids, mapping.source_ids)) or any(
        not isinstance(row_id, str) or not isinstance(source_id, str)
        for row_id, source_id in cache_pairs
    ):
        raise FeatureBundleError(
            f"retrieval cache {path} row_id/source_id pairs are not canonical"
        )
    return payload


def _read_embedding_shard(
    path: Path, *, entity: str
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "row_ids" not in payload or "embeddings" not in payload:
                raise FeatureBundleError(
                    f"embedding shard {path} requires row_ids and embeddings arrays"
                )
            row_ids = tuple(str(value).strip() for value in payload["row_ids"].tolist())
            embeddings = np.asarray(payload["embeddings"])
            failures: dict[str, str] = {}
            has_failed_ids = "failed_row_ids" in payload
            has_failure_reasons = "failure_reasons" in payload
            if has_failed_ids != has_failure_reasons:
                raise FeatureBundleError(
                    f"embedding shard {path} must provide both failed_row_ids and "
                    "failure_reasons"
                )
            if has_failed_ids:
                failed_ids = [str(value).strip() for value in payload["failed_row_ids"]]
                reasons = [str(value).strip() for value in payload["failure_reasons"]]
                if len(failed_ids) != len(reasons):
                    raise FeatureBundleError(
                        f"embedding shard {path} failure arrays have different lengths"
                    )
                failures = dict(zip(failed_ids, reasons))
    except (OSError, ValueError) as error:
        raise FeatureBundleError(f"cannot read embedding shard {path}: {error}") from error
    if embeddings.ndim != 2 or len(row_ids) != embeddings.shape[0]:
        raise FeatureBundleError(
            f"embedding shard {path} row IDs and matrix shape disagree"
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise FeatureBundleError(f"embedding shard {path} must be numeric")
    if not np.isfinite(embeddings).all():
        raise FeatureBundleError(f"embedding shard {path} contains NaN or infinity")
    if any(not value for value in row_ids) or len(row_ids) != len(set(row_ids)):
        raise FeatureBundleError(f"embedding shard {path} has invalid/duplicate row IDs")
    if set(row_ids).intersection(failures):
        raise FeatureBundleError(
            f"embedding shard {path} marks the same {entity} row successful and failed"
        )
    if any(not reason for reason in failures.values()):
        raise FeatureBundleError(f"embedding shard {path} has an empty failure reason")
    return row_ids, embeddings.astype(np.float32, copy=False), failures


def _strict_manifest_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FeatureBundleError(
            f"{label} must be a JSON integer >= {minimum}"
        )
    return value


def _validate_biological_model_provenance(
    provenance: Mapping[str, Any],
    *,
    extractor: str,
    dimension: int,
    resolver: Mapping[str, Path] | None,
) -> list[Path]:
    """Validate the model/runtime closure shared by every shard in one run."""

    if _strict_manifest_integer(
        provenance.get("raw_dimension"), "provenance raw_dimension", minimum=1
    ) != dimension:
        raise FeatureBundleError("extractor raw dimension is inconsistent")
    checkpoint_path = _verify_artifact(
        provenance.get("checkpoint", {}), f"{extractor} checkpoint", resolver=resolver
    )
    if provenance.get("expected_checkpoint_sha256") != _sha256(checkpoint_path):
        raise FeatureBundleError("extractor checkpoint hash is inconsistent")
    dependencies = [checkpoint_path]
    if extractor == "kpgt-base-generate-fps":
        # The extractor owns the formal environment lock.  Import lazily to
        # avoid its module-level bundle import creating a cycle.
        from .ethnobotany_feature_extractors import (  # noqa: PLC0415
            KPGT_EXPECTED_DGL_VERSION,
            KPGT_EXPECTED_PYTHON_VERSION,
            KPGT_EXPECTED_RDKIT_VERSION,
            KPGT_EXPECTED_TORCH_VERSION,
        )

        commit = "47dc1646c70b2138a157de481d24a1ac35d174cd"
        if (
            provenance.get("implementation") != "official KPGT LiGhT generate_fps"
            or provenance.get("config") != "base"
            or provenance.get("repository_commit") != commit
            or provenance.get("expected_repository_commit") != commit
        ):
            raise FeatureBundleError("KPGT model/source pin provenance is invalid")
        _strict_nonempty_string(provenance.get("repository"), "KPGT repository")
        _strict_sha256(
            provenance.get("repository_tree_sha256"), "KPGT source-tree hash"
        )
        worker = _verify_artifact(
            provenance.get("worker", {}), "KPGT worker", resolver=resolver
        )
        dependencies.append(worker)
        runtime = provenance.get("runtime")
        expected_runtime = provenance.get("expected_runtime")
        packages = runtime.get("packages") if isinstance(runtime, Mapping) else None
        if (
            not isinstance(expected_runtime, Mapping)
            or not isinstance(packages, Mapping)
            or expected_runtime.get("python_version")
            != KPGT_EXPECTED_PYTHON_VERSION
            or expected_runtime.get("torch") != KPGT_EXPECTED_TORCH_VERSION
            or expected_runtime.get("dgl") != KPGT_EXPECTED_DGL_VERSION
            or expected_runtime.get("rdkit") != KPGT_EXPECTED_RDKIT_VERSION
            or runtime.get("python_version") != KPGT_EXPECTED_PYTHON_VERSION
            or packages.get("torch") != KPGT_EXPECTED_TORCH_VERSION
            or packages.get("dgl") != KPGT_EXPECTED_DGL_VERSION
            or packages.get("rdkit") != KPGT_EXPECTED_RDKIT_VERSION
            or runtime.get("worker_sha256") != _sha256(worker)
        ):
            raise FeatureBundleError("KPGT runtime provenance is invalid")
    else:
        if (
            provenance.get("implementation") != "fair-esm local model loader"
            or provenance.get("model_name") != "esm2_t30_150M_UR50D"
            or provenance.get("representation_layer") != 30
        ):
            raise FeatureBundleError("ESM-2 model provenance is invalid")
        _strict_sha256(
            provenance.get("source_tree_sha256"), "fair-esm source-tree hash"
        )
        _strict_nonempty_string(
            provenance.get("source_tree_root"), "fair-esm source-tree root"
        )
        package_version = _strict_nonempty_string(
            provenance.get("package_version"), "fair-esm package version"
        )
        runtime = provenance.get("runtime")
        versions = provenance.get("dependency_versions")
        packages = runtime.get("packages") if isinstance(runtime, Mapping) else None
        if (
            not isinstance(versions, Mapping)
            or not isinstance(packages, Mapping)
            or dict(packages) != dict(versions)
            or packages.get("fair-esm") != package_version
            or provenance.get("sequence_set_aggregation")
            != SEQUENCE_SET_AGGREGATION
        ):
            raise FeatureBundleError("ESM-2 runtime/aggregation provenance is invalid")
    return dependencies


def _validate_biological_shard_contract(
    sidecar: Mapping[str, Any],
    *,
    sidecar_path: Path,
    shard_path: Path,
    mapping: EntityMapping,
    cache_path: Path,
    resolver: Mapping[str, Path] | None,
    cache_sha256: str | None = None,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
    validate_shared_provenance: bool = True,
    validate_row_provenance: bool = True,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str], list[Path]]:
    observed_cache_sha256 = cache_sha256 or _sha256(cache_path)
    if sidecar.get("schema") != EMBEDDING_SHARD_SCHEMA:
        raise FeatureBundleError("embedding shard sidecar has a wrong schema")
    if (
        sidecar.get("entity") != mapping.entity
        or sidecar.get("mapping_sha256") != mapping.sha256
        or sidecar.get("retrieval_cache_sha256") != observed_cache_sha256
    ):
        raise FeatureBundleError("embedding shard sidecar identity is false")
    start = _strict_manifest_integer(sidecar.get("row_start"), "shard row_start")
    end = _strict_manifest_integer(
        sidecar.get("row_stop"), "shard row_stop", minimum=1
    )
    row_count = _strict_manifest_integer(
        sidecar.get("row_count"), "shard row_count", minimum=1
    )
    _strict_manifest_integer(sidecar.get("batch_size"), "shard batch_size", minimum=1)
    input_dimension = _strict_manifest_integer(
        sidecar.get("input_dimension"), "shard input_dimension", minimum=1
    )
    dimension = _strict_manifest_integer(
        sidecar.get("embedding_dimension"), "shard embedding_dimension", minimum=1
    )
    if not 0 <= start < end <= len(mapping.row_ids) or row_count != end - start:
        raise FeatureBundleError("embedding shard range/count is invalid")
    extractor = _strict_nonempty_string(sidecar.get("extractor"), "shard extractor")
    expected_dimension = {
        "kpgt-base-generate-fps": 2304,
        "esm2_t30_150M_UR50D": 640,
    }.get(extractor)
    if (
        expected_dimension is None
        or input_dimension != expected_dimension
        or dimension != expected_dimension
        or (mapping.entity, extractor)
        not in {
            ("compound", "kpgt-base-generate-fps"),
            ("protein", "esm2_t30_150M_UR50D"),
        }
    ):
        raise FeatureBundleError("formal shard extractor/entity/dimension is invalid")
    recorded_shard = _verify_artifact(
        sidecar.get("artifact", {}), "embedding shard", resolver=resolver
    )
    if _sha256(recorded_shard) != _sha256(shard_path):
        raise FeatureBundleError("shard sidecar points to different content")
    row_ids, values, failures = _read_embedding_shard(shard_path, entity=mapping.entity)
    successful = list(row_ids)
    failed = list(failures)
    expected_ids = list(mapping.row_ids[start:end])
    successful_set = set(successful)
    failed_set = set(failed)
    if (
        successful_set | failed_set != set(expected_ids)
        or successful_set & failed_set
        or successful != [row_id for row_id in expected_ids if row_id in successful_set]
        or failed != [row_id for row_id in expected_ids if row_id in failed_set]
        or sidecar.get("row_ids") != successful
        or sidecar.get("failure_row_ids") != failed
        or values.shape[1] != dimension
    ):
        raise FeatureBundleError("embedding shard coverage/arrays are inconsistent")
    if (
        _strict_manifest_integer(sidecar.get("success_count"), "shard success_count")
        != len(successful)
        or _strict_manifest_integer(sidecar.get("failure_count"), "shard failure_count")
        != len(failed)
    ):
        raise FeatureBundleError("embedding shard counts disagree with its arrays")
    provenance = sidecar.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FeatureBundleError("embedding shard lacks model provenance")
    dependencies = [sidecar_path, shard_path]
    if validate_shared_provenance:
        dependencies.extend(
            _validate_biological_model_provenance(
                provenance,
                extractor=extractor,
                dimension=dimension,
                resolver=resolver,
            )
        )
    if (
        extractor == "esm2_t30_150M_UR50D"
        and validate_row_provenance
    ):
        observed_cache_rows = cache_rows
        if observed_cache_rows is None:
            cache = _read_json(cache_path)
            raw_cache_rows = cache.get("rows")
            if not isinstance(raw_cache_rows, list) or any(
                not isinstance(row, Mapping) for row in raw_cache_rows
            ):
                raise FeatureBundleError("protein cache rows are malformed")
            observed_cache_rows = raw_cache_rows
        expected_sequence_rows: list[dict[str, Any]] = []
        for row in observed_cache_rows[start:end]:
            if row.get("status") != "resolved_sequence_set":
                continue
            members = _validate_sequence_set_row(
                row,
                source_id=str(row.get("source_id", "")),
                mapping_sha256=mapping.sha256,
                resolver=resolver,
            )
            expected_sequence_rows.append(
                {
                    "row_id": row["row_id"],
                    "source_id": row["source_id"],
                    "sequence_set_tier": row["sequence_set_tier"],
                    "aggregation": row["sequence_set_aggregation"],
                    "sequence_count": len(members),
                    "sequences": [
                        {
                            "sequence_sha256": member["sequence_sha256"],
                            "accessions": member["accessions"],
                            "length": len(member["sequence"]),
                            "taxid": member["taxid"],
                            "annotation_releases": member["annotation_releases"],
                        }
                        for member in members
                    ],
                }
            )
        if sidecar.get("sequence_set_rows") != expected_sequence_rows:
            raise FeatureBundleError("protein shard sequence-set provenance is false")
    return row_ids, values, failures, dependencies


def _validate_order_claim_sidecar(
    sidecar: Mapping[str, Any],
    *,
    sidecar_path: Path,
    shard_path: Path,
    mapping: EntityMapping,
    resolver: Mapping[str, Path] | None,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str], list[Path]]:
    if sidecar.get("schema") != ORDER_CLAIM_IMPORT_SCHEMA:
        raise FeatureBundleError("order-claim sidecar has a wrong schema")
    if sidecar.get("entity") != mapping.entity or mapping.entity not in {
        "herb",
        "disease",
    }:
        raise FeatureBundleError("order-claim sidecar has a wrong entity")
    if sidecar.get("mapping_sha256") != mapping.sha256:
        raise FeatureBundleError("order-claim sidecar has a wrong mapping hash")
    mapping_path = _verify_artifact(
        sidecar.get("mapping", {}), "order-claim mapping", resolver=resolver
    )
    if _sha256(mapping_path) != mapping.sha256:
        raise FeatureBundleError("order-claim mapping content differs")
    source_path = _verify_artifact(
        sidecar.get("source_matrix", {}),
        "order-claim source matrix",
        resolver=resolver,
    )
    recorded_shard = _verify_artifact(
        sidecar.get("shard", {}), "order-claim shard", resolver=resolver
    )
    if _sha256(recorded_shard) != _sha256(shard_path):
        raise FeatureBundleError("order-claim sidecar points to different shard content")
    reducer_path = _verify_artifact(
        sidecar.get("identity_reducer_checkpoint", {}),
        "order-claim identity reducer",
        resolver=resolver,
    )
    if (
        not isinstance(sidecar.get("identity_claim"), str)
        or not sidecar["identity_claim"].strip()
        or not isinstance(sidecar.get("source_provenance"), str)
        or not sidecar["source_provenance"].strip()
        or sidecar.get("row_identity_inferred_from_shape") is not False
        or sidecar.get("semantic_features_regenerated") is not False
        or sidecar.get("semantic_feature_treatment")
        != "frozen_existing_64d_condition"
        or sidecar.get("provenance_verified") is not False
        or sidecar.get("formal_eligible") is not False
        or sidecar.get("semantic_identity_status")
        != "legacy_self_attested_unverified"
        or sidecar.get("reducer_algorithm")
        != "explicit_identity_64_to_64_no_centering_no_rotation"
        or sidecar.get("expected_row_ids") != list(mapping.row_ids)
        or sidecar.get("embedding_dimension") != 64
    ):
        raise FeatureBundleError("order-claim identity/provenance contract is invalid")
    if _strict_manifest_integer(
        sidecar.get("success_count"), "order-claim success_count"
    ) != len(mapping.row_ids) or _strict_manifest_integer(
        sidecar.get("failure_count"), "order-claim failure_count"
    ) != 0:
        raise FeatureBundleError("order-claim counts are invalid")
    row_ids, values, failures = _read_embedding_shard(
        shard_path, entity=mapping.entity
    )
    if row_ids != mapping.row_ids or failures or values.shape != (
        len(mapping.row_ids),
        64,
    ):
        raise FeatureBundleError("order-claim shard coverage/dimension is invalid")
    source = np.asarray(_load_npy(source_path, "order-claim source matrix"), dtype=np.float32)
    if source.shape != values.shape or not np.array_equal(source, values):
        raise FeatureBundleError("order-claim shard differs from its source matrix")
    try:
        with np.load(reducer_path, allow_pickle=False) as reducer:
            mean = np.asarray(reducer["mean"])
            components = np.asarray(reducer["components"])
            reducer_metadata = str(reducer["astra_identity_provenance_json"].item())
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_metadata = str(shard["astra_identity_provenance_json"].item())
    except (OSError, ValueError, KeyError) as error:
        raise FeatureBundleError(f"cannot validate order-claim provenance: {error}") from error
    expected_metadata = json.dumps(
        {
            "schema": ORDER_CLAIM_IMPORT_SCHEMA,
            "entity": mapping.entity,
            "mapping_sha256": mapping.sha256,
            "source_matrix_sha256": _sha256(source_path),
            "identity_claim": sidecar["identity_claim"],
            "source_provenance": sidecar["source_provenance"],
            "semantic_features_regenerated": False,
            "semantic_feature_treatment": "frozen_existing_64d_condition",
            "provenance_verified": False,
            "formal_eligible": False,
            "semantic_identity_status": "legacy_self_attested_unverified",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if (
        not np.array_equal(mean, np.zeros(64))
        or not np.array_equal(components, np.eye(64))
        or reducer_metadata != expected_metadata
        or shard_metadata != expected_metadata
    ):
        raise FeatureBundleError("order-claim identity reducer/provenance is false")
    return row_ids, values, failures, [
        sidecar_path,
        shard_path,
        mapping_path,
        source_path,
        reducer_path,
    ]


def _validate_verified_semantic_shard(
    record: Mapping[str, Any],
    *,
    mapping: EntityMapping,
    resolver: Mapping[str, Path] | None,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str], list[Path]]:
    """Validate one raw SciBERT matrix through the ID-keyed semantic chain."""

    if mapping.entity not in {"herb", "disease"}:
        raise FeatureBundleError("verified semantic shards are herb/disease only")
    if set(record) != {"shard", "semantic_reduction_manifest"}:
        raise FeatureBundleError(
            "verified semantic shard records may contain only shard and "
            "semantic_reduction_manifest"
        )
    reduction_path = _verify_artifact(
        record.get("semantic_reduction_manifest", {}),
        "semantic reduction manifest",
        resolver=resolver,
    )
    shard_path = _verify_artifact(
        record.get("shard", {}), "semantic raw embedding shard", resolver=resolver
    )
    # Lazy imports avoid the semantic pipeline -> feature-bundle import cycle.
    from .ethnobotany_semantic_features import (  # noqa: PLC0415
        SemanticFeatureError,
        validate_scibert_embedding_manifest,
        validate_shared_semantic_reduction,
    )

    try:
        reduction = validate_shared_semantic_reduction(
            reduction_path, resolver=resolver
        )
        embeddings = validate_scibert_embedding_manifest(
            reduction["embedding_manifest_path"], resolver=resolver
        )
    except SemanticFeatureError as error:
        raise FeatureBundleError(
            f"verified semantic provenance is invalid: {error}"
        ) from error
    entity_output = embeddings["outputs"][mapping.entity]
    if _sha256(shard_path) != _sha256(entity_output["path"]):
        raise FeatureBundleError(
            "semantic assembly shard differs from the recursively validated "
            "SciBERT output"
        )
    rows = entity_output["rows"]
    row_ids = tuple(str(row.get("row_id", "")) for row in rows)
    source_ids = tuple(str(row.get("source_id", "")) for row in rows)
    if row_ids != mapping.row_ids or source_ids != mapping.source_ids:
        raise FeatureBundleError(
            "semantic embedding identities differ from the canonical mapping"
        )
    reduction_record = reduction["manifest"]["entities"][mapping.entity]
    if reduction_record.get("mapping_sha256") != mapping.sha256:
        raise FeatureBundleError(
            "semantic reduction mapping hash differs from the canonical mapping"
        )
    values = np.asarray(entity_output["matrix"], dtype=np.float32)
    if values.shape != (len(mapping.row_ids), 768) or not np.isfinite(values).all():
        raise FeatureBundleError("verified SciBERT semantic matrix is invalid")
    dependencies = [Path(value).resolve() for value in reduction["dependencies"]]
    return row_ids, values, {}, list(dict.fromkeys(dependencies))


def _validate_assembly_shard_record(
    record: Mapping[str, Any],
    *,
    mapping: EntityMapping,
    cache_path: Path | None,
    resolver: Mapping[str, Path] | None = None,
    live_source_validation: bool = False,
    cache_sha256: str | None = None,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
    extraction_validation_cache: dict[
        tuple[str, str, str, bool], dict[str, Any]
    ] | None = None,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str], list[Path]]:
    if not isinstance(record, Mapping):
        raise FeatureBundleError("assembly shard record must be a JSON object")
    if record.get("semantic_reduction_manifest") is not None:
        return _validate_verified_semantic_shard(
            record, mapping=mapping, resolver=resolver
        )
    shard_path = _verify_artifact(
        record.get("shard", {}), "assembly raw shard", resolver=resolver
    )
    sidecar_path = _verify_artifact(
        record.get("sidecar", {}), "assembly shard sidecar", resolver=resolver
    )
    sidecar = _read_json(sidecar_path)
    if mapping.entity in BIOLOGICAL_ENTITIES:
        if cache_path is None:
            raise FeatureBundleError("biological shard has no retrieval cache")
        extraction_path = _verify_artifact(
            record.get("extraction_manifest", {}),
            "embedding extraction manifest",
            resolver=resolver,
        )
        observed_cache_sha256 = cache_sha256 or _sha256(cache_path)
        extraction_key = (
            _sha256(extraction_path),
            mapping.sha256,
            observed_cache_sha256,
            live_source_validation,
        )
        cached_extraction = (
            extraction_validation_cache.get(extraction_key)
            if extraction_validation_cache is not None
            else None
        )
        if live_source_validation and cached_extraction is None:
            # Lazy import avoids the extractor -> bundle import cycle.
            from .ethnobotany_feature_extractors import (  # noqa: PLC0415
                ExtractionError,
                validate_formal_extractor_shard,
            )

            try:
                validate_formal_extractor_shard(
                    shard_path=shard_path,
                    sidecar_path=sidecar_path,
                    extraction_manifest=extraction_path,
                    mapping_path=mapping.path,
                    retrieval_cache=cache_path,
                )
            except ExtractionError as error:
                raise FeatureBundleError(str(error)) from error
        if cached_extraction is not None:
            selected_digest = _sha256(sidecar_path)
            if cached_extraction["sidecar_digest_counts"].get(selected_digest) != 1:
                raise FeatureBundleError(
                    "selected shard is absent or duplicated in its cached extraction manifest"
                )
            selected = _validate_biological_shard_contract(
                sidecar,
                sidecar_path=sidecar_path,
                shard_path=shard_path,
                mapping=mapping,
                cache_path=cache_path,
                resolver=resolver,
                cache_sha256=observed_cache_sha256,
                validate_shared_provenance=False,
                validate_row_provenance=False,
            )
            dependencies = list(selected[3]) + list(
                cached_extraction["dependencies"]
            )
            return selected[0], selected[1], selected[2], list(
                dict.fromkeys(dependencies)
            )
        selected = _validate_biological_shard_contract(
            sidecar,
            sidecar_path=sidecar_path,
            shard_path=shard_path,
            mapping=mapping,
            cache_path=cache_path,
            resolver=resolver,
            cache_sha256=observed_cache_sha256,
            cache_rows=cache_rows,
        )
        manifest = _read_json(extraction_path)
        if (
            manifest.get("schema") != EXTRACTION_MANIFEST_SCHEMA
            or manifest.get("entity") != mapping.entity
            or manifest.get("extractor") != sidecar.get("extractor")
            or manifest.get("mapping_sha256") != mapping.sha256
            or manifest.get("provenance") != sidecar.get("provenance")
            or manifest.get("retrieval_cache_formal_eligible") is not True
        ):
            raise FeatureBundleError(
                "embedding extraction manifest identity/provenance is invalid"
            )
        manifest_mapping = _verify_artifact(
            manifest.get("mapping", {}),
            "extraction mapping",
            resolver=resolver,
        )
        manifest_cache = _verify_artifact(
            manifest.get("retrieval_cache", {}),
            "extraction retrieval cache",
            resolver=resolver,
        )
        if (
            _sha256(manifest_mapping) != mapping.sha256
            or _sha256(manifest_cache) != observed_cache_sha256
        ):
            raise FeatureBundleError(
                "embedding extraction manifest mapping/cache content is invalid"
            )
        range_value = manifest.get("range")
        if (
            not isinstance(range_value, list)
            or len(range_value) != 2
            or type(range_value[0]) is not int
            or type(range_value[1]) is not int
            or not 0 <= range_value[0] < range_value[1] <= len(mapping.row_ids)
        ):
            raise FeatureBundleError("embedding extraction range is invalid")
        _strict_manifest_integer(
            manifest.get("shard_size"), "extraction shard_size", minimum=1
        )
        _strict_manifest_integer(
            manifest.get("batch_size"), "extraction batch_size", minimum=1
        )
        entries = manifest.get("shards")
        if not isinstance(entries, list) or not entries:
            raise FeatureBundleError("embedding extraction manifest has no shards")
        dependencies = list(selected[3]) + [
            extraction_path,
            manifest_mapping,
            manifest_cache,
        ]
        covered: list[str] = []
        success_total = 0
        failure_total = 0
        selected_matches = 0
        sidecar_digest_counts: dict[str, int] = {}
        sequence_set_rows: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("contract"), Mapping
            ):
                raise FeatureBundleError(
                    "embedding extraction manifest contains a malformed shard entry"
                )
            entry_sidecar = _verify_artifact(
                entry.get("sidecar", {}),
                "extraction shard sidecar",
                resolver=resolver,
            )
            entry_sidecar_digest = _sha256(entry_sidecar)
            sidecar_digest_counts[entry_sidecar_digest] = (
                sidecar_digest_counts.get(entry_sidecar_digest, 0) + 1
            )
            contract = _read_json(entry_sidecar)
            if contract != dict(entry["contract"]):
                raise FeatureBundleError(
                    "embedding extraction manifest embeds a false sidecar"
                )
            if (
                contract.get("extractor") != manifest.get("extractor")
                or contract.get("provenance") != manifest.get("provenance")
            ):
                raise FeatureBundleError(
                    "embedding extraction shard provenance differs from its manifest"
                )
            entry_shard = _verify_artifact(
                contract.get("artifact", {}),
                "extraction raw shard",
                resolver=resolver,
            )
            row_ids, _, failures, entry_dependencies = (
                _validate_biological_shard_contract(
                    contract,
                    sidecar_path=entry_sidecar,
                    shard_path=entry_shard,
                    mapping=mapping,
                    cache_path=cache_path,
                    resolver=resolver,
                    cache_sha256=observed_cache_sha256,
                    cache_rows=cache_rows,
                    validate_shared_provenance=False,
                )
            )
            start = _strict_manifest_integer(
                contract.get("row_start"), "extraction entry row_start"
            )
            stop = _strict_manifest_integer(
                contract.get("row_stop"), "extraction entry row_stop", minimum=1
            )
            expected_ids = list(mapping.row_ids[start:stop])
            covered.extend(expected_ids)
            success_total += len(row_ids)
            failure_total += len(failures)
            dependencies.extend(entry_dependencies)
            if mapping.entity == "protein":
                raw_sequence_rows = contract.get("sequence_set_rows")
                if (
                    not isinstance(raw_sequence_rows, list)
                    or any(not isinstance(item, Mapping) for item in raw_sequence_rows)
                ):
                    raise FeatureBundleError(
                        "protein extraction sidecar has malformed sequence_set_rows"
                    )
                sequence_set_rows.extend(dict(item) for item in raw_sequence_rows)
            if _sha256(entry_sidecar) == _sha256(sidecar_path):
                selected_matches += 1
        if covered != list(mapping.row_ids[range_value[0] : range_value[1]]):
            raise FeatureBundleError(
                "embedding extraction shard ranges do not exactly cover its range"
            )
        if selected_matches != 1:
            raise FeatureBundleError(
                "selected shard is absent or duplicated in its extraction manifest"
            )
        if _strict_manifest_integer(
            manifest.get("success_count"), "extraction success_count"
        ) != success_total or _strict_manifest_integer(
            manifest.get("failure_count"), "extraction failure_count"
        ) != failure_total:
            raise FeatureBundleError(
                "embedding extraction aggregate counts are inconsistent"
            )
        if mapping.entity == "protein" and manifest.get(
            "sequence_set_rows"
        ) != sequence_set_rows:
            raise FeatureBundleError(
                "protein extraction manifest sequence-set provenance is inconsistent"
            )
        if extraction_validation_cache is not None:
            extraction_validation_cache[extraction_key] = {
                "sidecar_digest_counts": sidecar_digest_counts,
                "dependencies": list(dict.fromkeys(dependencies)),
            }
        return selected[0], selected[1], selected[2], list(
            dict.fromkeys(dependencies)
        )

    if live_source_validation:
        from .ethnobotany_feature_extractors import (  # noqa: PLC0415
            ExtractionError,
            validate_order_claimed_shard,
        )

        try:
            validate_order_claimed_shard(
                shard_path=shard_path,
                manifest_path=sidecar_path,
                mapping_path=mapping.path,
            )
        except ExtractionError as error:
            raise FeatureBundleError(str(error)) from error
    return _validate_order_claim_sidecar(
        sidecar,
        sidecar_path=sidecar_path,
        shard_path=shard_path,
        mapping=mapping,
        resolver=resolver,
    )


def assemble_embedding_shards(
    *,
    entity: str,
    mapping_path: Path | str,
    shard_paths: Sequence[Path | str],
    output_dir: Path | str,
    retrieval_cache: Path | str | None = None,
    semantic_reduction_manifest: Path | str | None = None,
    allow_incomplete: bool = False,
) -> Path:
    """Merge ID-keyed raw embedding shards into canonical mapping order."""

    mapping = read_entity_mapping(mapping_path, entity)
    if semantic_reduction_manifest is not None:
        if mapping.entity not in {"herb", "disease"}:
            raise FeatureBundleError(
                "--semantic-reduction-manifest is valid only for herb/disease"
            )
        if shard_paths:
            raise FeatureBundleError(
                "verified semantic assembly cannot be mixed with raw shard arguments"
            )
        if retrieval_cache is not None:
            raise FeatureBundleError(
                "verified semantic assembly cannot use a retrieval cache"
            )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "assembly_manifest.json"
    raw_path = output_dir / "raw_embeddings.npy"
    mask_path = output_dir / "missing_mask.npy"
    sidecar_path = feature_row_ids_path(raw_path)

    cache_record: dict[str, Any] | None = None
    cache_reasons: list[str] = []
    cache_status_by_id: dict[str, str] = {}
    missing_sequence_policy = "strict"
    sequence_resolution: dict[str, Any] | None = None
    cache_path: Path | None = None
    cache_sha256: str | None = None
    cache_rows: tuple[Mapping[str, Any], ...] | None = None
    cache_valid = False
    if retrieval_cache is not None:
        cache_path = Path(retrieval_cache).resolve()
        cache = _load_retrieval_cache(cache_path, mapping)
        cache_record = _artifact(cache_path)
        cache_sha256 = str(cache_record["sha256"])
        cache_rows = tuple(cache["rows"])
        cache_status_by_id = {
            str(row.get("row_id")): str(row.get("status"))
            for row in cache.get("rows", [])
            if isinstance(row, Mapping)
        }
        missing_sequence_policy = str(
            cache.get("missing_sequence_policy", "strict")
        )
        derived_cache_mask = np.asarray(
            [
                status in {"target_not_protein", "audited_missing_sequence"}
                for status in cache_status_by_id.values()
            ],
            dtype=np.bool_,
        )
        try:
            _validate_retrieval_cache_for_mask(
                cache_path,
                mapping,
                derived_cache_mask,
                missing_sequence_policy=missing_sequence_policy,
            )
            cache_valid = True
        except FeatureBundleError as error:
            cache_reasons.append(f"retrieval_cache_invalid: {error}")
        status_counts = cache.get("status_counts")
        if not isinstance(status_counts, Mapping):
            raise FeatureBundleError("retrieval cache status_counts are malformed")
        resolved_count = _manifest_count(
            status_counts.get("resolved"), "retrieval-cache resolved count"
        )
        resolved_sequence_set_count = (
            _manifest_count(
                status_counts.get("resolved_sequence_set"),
                "retrieval-cache resolved_sequence_set count",
            )
            if mapping.entity == "protein"
            else 0
        )
        unresolved = (
            len(mapping.row_ids) - resolved_count - resolved_sequence_set_count
        )
        permitted_zero_unk = (
            _manifest_count(
                status_counts.get("target_not_protein"),
                "retrieval-cache target_not_protein count",
            )
            if mapping.entity == "protein"
            and missing_sequence_policy in {"zero-unk", "audited-missing-zero"}
            else 0
        )
        unresolved -= permitted_zero_unk
        permitted_audited_missing = (
            _manifest_count(
                status_counts.get("audited_missing_sequence"),
                "retrieval-cache audited_missing_sequence count",
            )
            if mapping.entity == "protein"
            and missing_sequence_policy == "audited-missing-zero"
            else 0
        )
        unresolved -= permitted_audited_missing
        if unresolved:
            cache_reasons.append(f"retrieval_cache_unresolved_rows={unresolved}")
        sequence_resolution = {
            "missing_sequence_policy": missing_sequence_policy,
            "target_not_protein_count": _manifest_count(
                cache.get("target_not_protein_count", 0)
                , "cache target_not_protein_count"
            ),
            "audited_missing_sequence_count": _manifest_count(
                cache.get("audited_missing_sequence_count", 0),
                "cache audited_missing_sequence_count",
            ),
            "masked_missing_count": _manifest_count(
                cache.get("masked_missing_count", 0),
                "cache masked_missing_count",
            ),
            "resolved_single_sequence_count": _manifest_count(
                cache.get("resolved_single_sequence_count", 0),
                "cache resolved_single_sequence_count",
            ),
            "resolved_sequence_set_count": _manifest_count(
                cache.get("resolved_sequence_set_count", 0),
                "cache resolved_sequence_set_count",
            ),
            "resolved_sequence_count": _manifest_count(
                cache.get("resolved_sequence_count", 0),
                "cache resolved_sequence_count",
            ),
            "overall_sequence_resolution_rate": float(
                cache.get("overall_sequence_resolution_rate", 0.0)
            ),
            "protein_target_resolution_rate": float(
                cache.get("protein_target_resolution_rate", 0.0)
            ),
        }
    elif mapping.entity in BIOLOGICAL_ENTITIES:
        cache_reasons.append("retrieval_cache_not_supplied")

    matrix: np.ndarray | None = None
    prior_failures: dict[str, str] = {}
    shard_records: list[dict[str, Any]] = []
    provenance_reasons: list[str] = []
    # One extraction manifest may enumerate dozens of shards.  Its complete
    # coverage/provenance closure is validated once; every selected shard is
    # still independently revalidated before it can use the cached closure.
    extraction_validation_cache: dict[
        tuple[str, str, str, bool], dict[str, Any]
    ] = {}
    if manifest_path.exists():
        prior_manifest = _read_json(manifest_path)
        if prior_manifest.get("schema") != ASSEMBLY_SCHEMA:
            raise FeatureBundleError(f"existing {manifest_path} has a wrong schema")
        if prior_manifest.get("entity") != mapping.entity:
            raise FeatureBundleError(f"existing {manifest_path} has a wrong entity")
        if prior_manifest.get("mapping_sha256") != mapping.sha256:
            raise FeatureBundleError(f"existing {manifest_path} has a wrong mapping")
        recorded_raw = _verify_artifact(
            prior_manifest.get("raw_matrix", {}), "existing raw embedding matrix"
        )
        recorded_mask = _verify_artifact(
            prior_manifest.get("missing_mask", {}), "existing missing mask"
        )
        recorded_sidecar = _verify_artifact(
            prior_manifest.get("row_ids_sidecar", {}),
            "existing raw row-ID sidecar",
        )
        if (
            recorded_raw != raw_path
            or recorded_mask != mask_path
            or recorded_sidecar != sidecar_path
            or _read_json(recorded_sidecar) != _row_sidecar(mapping)
        ):
            raise FeatureBundleError(
                "existing assembly manifest points outside/corrupts its output directory"
            )
        matrix = np.asarray(_load_npy(raw_path, "existing raw matrix"), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(mapping.row_ids):
            raise FeatureBundleError(f"existing raw matrix {raw_path} has wrong shape")
        raw_failures = prior_manifest.get("failures")
        if not isinstance(raw_failures, list) or any(
            not isinstance(record, Mapping) for record in raw_failures
        ):
            raise FeatureBundleError("existing assembly failures are malformed")
        prior_failures = {
            _strict_nonempty_string(record.get("row_id"), "failure row_id"):
            _strict_nonempty_string(record.get("reason"), "failure reason")
            for record in raw_failures
        }
        saved_shards = prior_manifest.get("shards")
        if not isinstance(saved_shards, list):
            raise FeatureBundleError("existing assembly shard records are malformed")
        for index, saved_record in enumerate(saved_shards):
            try:
                _validate_assembly_shard_record(
                    saved_record,
                    mapping=mapping,
                    cache_path=cache_path,
                    live_source_validation=True,
                    cache_sha256=cache_sha256,
                    cache_rows=cache_rows,
                    extraction_validation_cache=extraction_validation_cache,
                )
                if (
                    mapping.entity in {"herb", "disease"}
                    and saved_record.get("semantic_reduction_manifest") is None
                ):
                    provenance_reasons.append(
                        "semantic_identity_legacy_self_attested_unverified"
                    )
            except FeatureBundleError as error:
                provenance_reasons.append(
                    f"saved_shard_{index}_provenance_invalid: {error}"
                )
            if isinstance(saved_record, Mapping):
                shard_records.append(dict(saved_record))

    positions = {row_id: index for index, row_id in enumerate(mapping.row_ids)}
    if semantic_reduction_manifest is not None:
        reduction_path = Path(semantic_reduction_manifest).resolve()
        # Resolve the raw entity-specific SciBERT artifact only through the
        # recursive formal semantic validator; no caller-supplied row claim is
        # accepted here.
        from .ethnobotany_semantic_features import (  # noqa: PLC0415
            SemanticFeatureError,
            validate_scibert_embedding_manifest,
            validate_shared_semantic_reduction,
        )

        try:
            reduction = validate_shared_semantic_reduction(reduction_path)
            embeddings = validate_scibert_embedding_manifest(
                reduction["embedding_manifest_path"]
            )
        except SemanticFeatureError as error:
            raise FeatureBundleError(
                f"cannot assemble verified semantic features: {error}"
            ) from error
        semantic_shard_path = Path(
            embeddings["outputs"][mapping.entity]["path"]
        ).resolve()
        record = {
            "shard": _artifact(semantic_shard_path),
            "semantic_reduction_manifest": _artifact(reduction_path),
        }
        row_ids, values, failures, _ = _validate_assembly_shard_record(
            record,
            mapping=mapping,
            cache_path=None,
            live_source_validation=True,
        )
        if failures:
            raise FeatureBundleError(
                "verified semantic extraction unexpectedly contains failed rows"
            )
        matrix = np.full(
            (len(mapping.row_ids), values.shape[1]), np.nan, dtype=np.float32
        )
        for row_id, value in zip(row_ids, values):
            matrix[positions[row_id]] = value
        shard_records.append(record)

    for raw_shard_path in shard_paths:
        shard_path = Path(raw_shard_path).resolve()
        sidecar = shard_path.with_suffix(".manifest.json")
        record: dict[str, Any] = {"shard": _artifact(shard_path)}
        if sidecar.is_file():
            record["sidecar"] = _artifact(sidecar)
        if mapping.entity in BIOLOGICAL_ENTITIES:
            extraction = shard_path.parent / "extraction_manifest.json"
            if extraction.is_file():
                record["extraction_manifest"] = _artifact(extraction)
        try:
            row_ids, values, failures, _ = _validate_assembly_shard_record(
                record,
                mapping=mapping,
                cache_path=cache_path,
                live_source_validation=True,
                cache_sha256=cache_sha256,
                cache_rows=cache_rows,
                extraction_validation_cache=extraction_validation_cache,
            )
            if mapping.entity in {"herb", "disease"}:
                provenance_reasons.append(
                    "semantic_identity_legacy_self_attested_unverified"
                )
        except FeatureBundleError as error:
            provenance_reasons.append(
                f"shard_{shard_path.name}_provenance_invalid: {error}"
            )
            row_ids, values, failures = _read_embedding_shard(
                shard_path, entity=mapping.entity
            )
        unknown = sorted((set(row_ids) | set(failures)).difference(positions))
        if unknown:
            raise FeatureBundleError(
                f"embedding shard {shard_path} has IDs outside the mapping: {unknown[:5]}"
            )
        if matrix is None:
            matrix = np.full(
                (len(mapping.row_ids), values.shape[1]), np.nan, dtype=np.float32
            )
        if values.shape[1] != matrix.shape[1]:
            raise FeatureBundleError(
                f"embedding shard {shard_path} dimension {values.shape[1]} does not "
                f"match existing dimension {matrix.shape[1]}"
            )
        for row_id, value in zip(row_ids, values):
            index = positions[row_id]
            existing = matrix[index]
            if np.isfinite(existing).all() and not np.array_equal(existing, value):
                raise FeatureBundleError(
                    f"conflicting embeddings were supplied for {mapping.entity} row {row_id}"
                )
            matrix[index] = value
            prior_failures.pop(row_id, None)
        for row_id, reason in failures.items():
            index = positions[row_id]
            if not np.isfinite(matrix[index]).all():
                prior_failures[row_id] = reason
        record_digest = record["shard"]["sha256"]
        if all(
            not isinstance(existing, Mapping)
            or not isinstance(existing.get("shard"), Mapping)
            or existing["shard"].get("sha256") != record_digest
            for existing in shard_records
        ):
            shard_records.append(record)

    if matrix is None:
        raise FeatureBundleError("no existing assembly or embedding shard was supplied")

    declared_zero_unk = np.zeros(len(mapping.row_ids), dtype=bool)
    declared_audited_missing = np.zeros(len(mapping.row_ids), dtype=bool)
    if (
        cache_valid
        and mapping.entity == "protein"
        and missing_sequence_policy in {"zero-unk", "audited-missing-zero"}
    ):
        for index, row_id in enumerate(mapping.row_ids):
            if cache_status_by_id.get(row_id) == "target_not_protein":
                if np.isfinite(matrix[index]).all():
                    raise FeatureBundleError(
                        f"protein row {row_id} is target-not-protein but also has an "
                        "embedding; resolve the provenance conflict explicitly"
                    )
                matrix[index] = 0.0
                declared_zero_unk[index] = True
            elif (
                cache_status_by_id.get(row_id) == "audited_missing_sequence"
                and missing_sequence_policy == "audited-missing-zero"
            ):
                if np.isfinite(matrix[index]).all():
                    raise FeatureBundleError(
                        f"protein row {row_id} is audited-missing but also has an "
                        "embedding; resolve the provenance conflict explicitly"
                    )
                matrix[index] = 0.0
                declared_audited_missing[index] = True

    declared_mask = declared_zero_unk | declared_audited_missing
    missing_mask = ~np.isfinite(matrix).all(axis=1) | declared_mask
    unpermitted_missing = missing_mask & ~declared_mask
    failures: list[dict[str, str]] = []
    for index in np.flatnonzero(missing_mask):
        row_id = mapping.row_ids[int(index)]
        is_zero_unk = bool(declared_zero_unk[int(index)])
        is_audited_missing = bool(declared_audited_missing[int(index)])
        failures.append(
            {
                "row_id": row_id,
                "source_id": mapping.source_ids[int(index)],
                "failure_class": (
                    "target_not_protein"
                    if is_zero_unk
                    else (
                        "audited_missing_sequence"
                        if is_audited_missing
                        else "embedding_missing_or_failed"
                    )
                ),
                "reason": (
                    "explicit zero-UNK for a target classified as non-protein"
                    if is_zero_unk
                    else (
                        "explicit audited-missing zero; no ESM feature was produced"
                        if is_audited_missing
                        else prior_failures.get(
                            row_id,
                            "row was not present in any successful embedding shard",
                        )
                    )
                ),
            }
        )

    reasons = (
        [f"missing_embedding_rows={int(unpermitted_missing.sum())}"]
        if unpermitted_missing.any()
        else []
    )
    reasons.extend(cache_reasons)
    reasons.extend(provenance_reasons)
    semantic_identity_status: str | None = None
    semantic_condition: str | None = None
    if mapping.entity in {"herb", "disease"}:
        semantic_records = [
            record
            for record in shard_records
            if isinstance(record, Mapping)
            and record.get("semantic_reduction_manifest") is not None
        ]
        if len(semantic_records) == 1 and len(shard_records) == 1:
            semantic_identity_status = "independently_id_keyed_verified"
            reduction_path = _verify_artifact(
                semantic_records[0]["semantic_reduction_manifest"],
                "semantic reduction manifest",
            )
            semantic_payload = _read_json(reduction_path)
            semantic_condition = _strict_nonempty_string(
                semantic_payload.get("semantic_condition"), "semantic condition"
            )
            reasons = [
                reason
                for reason in reasons
                if reason != "semantic_identity_legacy_self_attested_unverified"
            ]
        elif not semantic_records:
            semantic_identity_status = "legacy_self_attested_unverified"
        else:
            semantic_identity_status = "mixed_or_duplicated_semantic_provenance"
            reasons.append("semantic_verified_and_legacy_shards_were_mixed")
    reasons = list(dict.fromkeys(reasons))
    _atomic_npy(raw_path, matrix)
    _atomic_npy(mask_path, missing_mask.astype(np.bool_))
    _atomic_json(sidecar_path, _row_sidecar(mapping))
    manifest: dict[str, Any] = {
        "schema": ASSEMBLY_SCHEMA,
        "entity": mapping.entity,
        "mapping": _artifact(mapping.path),
        "mapping_sha256": mapping.sha256,
        "row_count": len(mapping.row_ids),
        "raw_dimension": int(matrix.shape[1]),
        "raw_matrix": _artifact(raw_path),
        "row_ids_sidecar": _artifact(sidecar_path),
        "missing_mask": _artifact(mask_path),
        "missing_count": int(missing_mask.sum()),
        "unpermitted_missing_count": int(unpermitted_missing.sum()),
        "declared_zero_unk_count": int(declared_zero_unk.sum()),
        "declared_zero_unk_row_ids": [
            mapping.row_ids[int(index)] for index in np.flatnonzero(declared_zero_unk)
        ],
        "declared_audited_missing_count": int(declared_audited_missing.sum()),
        "declared_audited_missing_row_ids": [
            mapping.row_ids[int(index)]
            for index in np.flatnonzero(declared_audited_missing)
        ],
        "declared_masked_row_count": int(declared_mask.sum()),
        "declared_masked_row_ids": [
            mapping.row_ids[int(index)] for index in np.flatnonzero(declared_mask)
        ],
        "missing_sequence_policy": missing_sequence_policy,
        "sequence_resolution": sequence_resolution,
        "semantic_identity_status": semantic_identity_status,
        "semantic_condition": semantic_condition,
        "failures": failures,
        "retrieval_cache": cache_record,
        "shards": shard_records,
        "formal_eligible": not reasons,
        "formal_ineligibility_reasons": reasons,
        "random_imputation": False,
    }
    _atomic_json(manifest_path, manifest)
    if reasons and not allow_incomplete:
        raise IncompleteFeatureBundleError(
            f"{mapping.entity} assembly is incomplete at {manifest_path}: {reasons}"
        )
    return manifest_path


def _verify_artifact(
    record: Mapping[str, Any],
    label: str,
    *,
    resolver: Mapping[str, Path] | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise FeatureBundleError(f"{label} has no artifact record")
    recorded_path = _strict_nonempty_string(record.get("path"), f"{label} path")
    digest = _strict_sha256(record.get("sha256"), f"{label} sha256")
    size = _manifest_count(record.get("bytes"), f"{label} bytes")
    if resolver is not None:
        path = resolver.get(digest)
        if path is None:
            raise FeatureBundleError(
                f"{label} is absent from the portable provenance closure"
            )
        path = path.resolve()
    else:
        path = Path(recorded_path).resolve()
    if not path.is_file():
        raise FeatureBundleError(f"{label} is missing at {path}")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise FeatureBundleError(f"{label} hash/size differs from its manifest")
    return path


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise FeatureBundleError(f"cannot read {label} {path}: {error}") from error


def _load_numeric_matrix(path: Path, label: str) -> np.ndarray:
    try:
        values = (
            np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npy"
            else np.loadtxt(path)
        )
    except (OSError, ValueError) as error:
        raise FeatureBundleError(f"cannot read {label} {path}: {error}") from error
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number):
        raise FeatureBundleError(f"{label} must be a numeric two-dimensional matrix")
    return matrix


def _manifest_count(value: Any, label: str) -> int:
    if type(value) is not int:
        raise FeatureBundleError(f"{label} must be a JSON integer")
    if value < 0:
        raise FeatureBundleError(f"{label} must be nonnegative")
    return value


def _manifest_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise FeatureBundleError(f"{label} must be a JSON boolean")
    return value


def _manifest_string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise FeatureBundleError(f"{label} must be a list of non-empty strings")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise FeatureBundleError(f"{label} contains duplicate IDs")
    return normalized


def _validate_retrieval_cache_for_mask(
    cache_path: Path,
    mapping: EntityMapping,
    missing_mask: np.ndarray,
    *,
    missing_sequence_policy: str,
    payload_override: Mapping[str, Any] | None = None,
    require_formal: bool = True,
    resolver: Mapping[str, Path] | None = None,
) -> tuple[str, ...]:
    """Recompute cache completeness and its exact relationship to the mask."""

    evidence_artifact_cache: dict[
        tuple[str, int, str], tuple[Path, dict[str, Any]]
    ] = {}
    cache = dict(payload_override) if payload_override is not None else _read_json(cache_path)
    if (
        cache.get("schema") != CACHE_SCHEMA
        or cache.get("entity") != mapping.entity
        or cache.get("mapping_sha256") != mapping.sha256
        or cache.get("row_id_space") != "protocol_raw_id"
    ):
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache identity does not match its mapping"
        )
    mapping_record = cache.get("mapping")
    if isinstance(mapping_record, Mapping):
        recorded_mapping = _verify_artifact(
            mapping_record,
            f"{mapping.entity} retrieval-cache mapping",
            resolver=resolver,
        )
        if _sha256(recorded_mapping) != mapping.sha256:
            raise FeatureBundleError(
                f"{mapping.entity} retrieval cache points to another mapping"
            )
    else:
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache lacks its mapping artifact"
        )
    rows = cache.get("rows")
    if not isinstance(rows, list) or len(rows) != len(mapping.row_ids) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache rows are malformed or incomplete"
        )
    observed_pairs = [(row.get("row_id"), row.get("source_id")) for row in rows]
    if observed_pairs != list(zip(mapping.row_ids, mapping.source_ids)) or any(
        not isinstance(row_id, str) or not isinstance(source_id, str)
        for row_id, source_id in observed_pairs
    ):
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache row_id/source_id pairs are not canonical"
        )
    if missing_mask.dtype != np.bool_ or missing_mask.shape != (len(mapping.row_ids),):
        raise FeatureBundleError(
            f"{mapping.entity} retrieval-cache mask must be boolean and row-aligned"
        )
    statuses = tuple(row.get("status") for row in rows)
    status_names = (
        "resolved",
        "resolved_sequence_set",
        "target_not_protein",
        "audited_missing_sequence",
        "failed",
        "pending",
    )
    allowed_statuses = set(status_names)
    if any(not isinstance(status, str) or status not in allowed_statuses for status in statuses):
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache contains an unsupported status"
        )
    observed_counts = {
        status: statuses.count(status)
        for status in status_names
    }
    status_counts = cache.get("status_counts")
    protein_only_zero_statuses = {
        "resolved_sequence_set",
        "target_not_protein",
        "audited_missing_sequence",
    }
    status_counts_valid = isinstance(status_counts, Mapping) and not set(
        status_counts
    ).difference(observed_counts)
    if status_counts_valid:
        for status, count in observed_counts.items():
            if status not in status_counts:
                if (
                    mapping.entity == "compound"
                    and status in protein_only_zero_statuses
                    and count == 0
                ):
                    continue
                status_counts_valid = False
                break
            if (
                _manifest_count(status_counts[status], f"cache {status} count")
                != count
            ):
                status_counts_valid = False
                break
    if not status_counts_valid:
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache status counts are inconsistent"
        )
    mask_ids = tuple(
        mapping.row_ids[int(index)] for index in np.flatnonzero(missing_mask)
    )
    target_rows = [
        row for row in rows if row.get("status") == "target_not_protein"
    ]
    audited_missing_rows = [
        row for row in rows if row.get("status") == "audited_missing_sequence"
    ]
    resolved_rows = [row for row in rows if row.get("status") == "resolved"]
    sequence_set_rows = [
        row for row in rows if row.get("status") == "resolved_sequence_set"
    ]
    target_ids = tuple(row["row_id"] for row in target_rows)
    audited_missing_ids = tuple(row["row_id"] for row in audited_missing_rows)
    cache_policy = cache.get("missing_sequence_policy")
    if not isinstance(cache_policy, str) or cache_policy not in {
        "strict",
        "zero-unk",
        "audited-missing-zero",
    }:
        raise FeatureBundleError("retrieval cache has an invalid missing-sequence policy")
    if cache_policy != missing_sequence_policy:
        raise FeatureBundleError(
            f"{mapping.entity} missing-sequence policy disagrees with its cache"
        )
    if mapping.entity == "compound":
        if cache_policy != "strict":
            raise FeatureBundleError("compound retrieval cache must use strict policy")
        if mask_ids or target_ids:
            raise FeatureBundleError(
                "compound retrieval cache cannot contain masked/target_not_protein rows"
            )
        if any(
            not isinstance(row.get("value"), str) or not row["value"].strip()
            for row in resolved_rows
        ):
            raise FeatureBundleError(
                "compound retrieval cache has a resolved row without SMILES"
            )
        if sequence_set_rows or audited_missing_rows:
            raise FeatureBundleError(
                "compound retrieval cache contains protein-only row states"
            )
    else:
        expected_mask_ids = tuple(
            row_id
            for row_id, status in zip(mapping.row_ids, statuses)
            if status in {"target_not_protein", "audited_missing_sequence"}
        )
        if expected_mask_ids != mask_ids:
            raise FeatureBundleError(
                "protein missing mask does not exactly trace to its audited cache rows"
            )
        for row in target_rows:
            _validate_target_classification_evidence(
                row,
                source_id=row["source_id"],
                mapping_sha256=mapping.sha256,
                resolver=resolver,
                artifact_cache=evidence_artifact_cache,
            )
        for row in resolved_rows:
            sequence = _normalise_sequence(row.get("value"))
            accession = str(row.get("accession", "")).strip()
            digest = str(row.get("sequence_sha256", "")).strip()
            if (
                sequence is None
                or sequence != row.get("value")
                or not accession
                or digest != hashlib.sha256(sequence.encode("ascii")).hexdigest()
            ):
                raise FeatureBundleError(
                    "protein retrieval cache has a resolved row without a valid "
                    "accession, canonical sequence, and matching SHA-256"
                )
        if any(
            row.get("value") is not None
            or row.get("accession") is not None
            or row.get("sequence_sha256") is not None
            for row in target_rows
        ):
            raise FeatureBundleError(
                "target_not_protein cache rows must not contain a protein sequence"
            )
        for row in audited_missing_rows:
            _validate_audited_missing_sequence_evidence(
                row,
                source_id=row["source_id"],
                mapping_sha256=mapping.sha256,
                resolver=resolver,
                artifact_cache=evidence_artifact_cache,
            )
        if any(
            row.get("value") is not None
            or row.get("accession") is not None
            or row.get("sequence_sha256") is not None
            for row in audited_missing_rows
        ):
            raise FeatureBundleError(
                "audited_missing_sequence rows must not contain a protein sequence"
            )
        for row in sequence_set_rows:
            _validate_sequence_set_row(
                row,
                source_id=row["source_id"],
                mapping_sha256=mapping.sha256,
                resolver=resolver,
                artifact_cache=evidence_artifact_cache,
            )
        ambiguous_policy = cache.get("ambiguous_sequence_policy", "strict")
        if ambiguous_policy not in {"strict", "mean-distinct-first-tier"}:
            raise FeatureBundleError(
                "retrieval cache has an invalid ambiguous-sequence policy"
            )
        if sequence_set_rows and ambiguous_policy != "mean-distinct-first-tier":
            raise FeatureBundleError(
                "sequence-set rows require mean-distinct-first-tier policy"
            )
        if require_formal and any(
            status not in {"resolved", "resolved_sequence_set"}
            for status, row_id in zip(statuses, mapping.row_ids)
            if row_id not in set(mask_ids)
        ):
            raise FeatureBundleError(
                "every unmasked protein target must have a resolved sequence or "
                "audited sequence set"
            )
        if require_formal and missing_sequence_policy == "strict" and mask_ids:
            raise FeatureBundleError(
                "protein missing-sequence policy disagrees with its mask"
            )
        if (
            require_formal
            and missing_sequence_policy == "zero-unk"
            and audited_missing_ids
        ):
            raise FeatureBundleError(
                "zero-unk does not authorize audited missing protein/unknown rows"
            )
        if (
            require_formal
            and
            missing_sequence_policy == "audited-missing-zero"
            and set(mask_ids) != set(target_ids).union(audited_missing_ids)
        ):
            raise FeatureBundleError(
                "audited-missing-zero mask is not fully audit-backed"
            )
        resolved_single = observed_counts["resolved"]
        resolved_set = observed_counts["resolved_sequence_set"]
        resolved = resolved_single + resolved_set
        target_count = observed_counts["target_not_protein"]
        audited_missing_count = observed_counts["audited_missing_sequence"]
        protein_count = len(rows) - target_count - audited_missing_count
        expected_target_rate = resolved / protein_count if protein_count else 0.0
        expected_overall_rate = resolved / len(rows) if rows else 0.0
        try:
            statistics_valid = (
                _manifest_count(
                    cache.get("target_not_protein_count", -1),
                    "target_not_protein_count",
                )
                == target_count
                and _manifest_count(
                    cache.get("audited_missing_sequence_count", -1),
                    "audited_missing_sequence_count",
                )
                == audited_missing_count
                and _manifest_count(
                    cache.get("masked_missing_count", -1),
                    "masked_missing_count",
                )
                == target_count + audited_missing_count
                and _manifest_count(
                    cache.get("resolved_single_sequence_count", -1),
                    "resolved_single_sequence_count",
                )
                == resolved_single
                and _manifest_count(
                    cache.get("resolved_sequence_set_count", -1),
                    "resolved_sequence_set_count",
                )
                == resolved_set
                and _manifest_count(
                    cache.get("resolved_sequence_count", -1),
                    "resolved_sequence_count",
                )
                == resolved
                and np.isclose(
                    float(cache.get("protein_target_resolution_rate", -1.0)),
                    expected_target_rate,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.isclose(
                    float(cache.get("overall_sequence_resolution_rate", -1.0)),
                    expected_overall_rate,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        except (TypeError, ValueError):
            statistics_valid = False
        if not statistics_valid:
            raise FeatureBundleError(
                "protein retrieval cache resolution statistics are inconsistent"
            )
    resolved_statuses = {"resolved", "resolved_sequence_set"}
    if mapping.entity == "compound" or missing_sequence_policy == "strict":
        calculated_formal = all(status in resolved_statuses for status in statuses)
    elif missing_sequence_policy == "zero-unk":
        calculated_formal = all(
            status in resolved_statuses | {"target_not_protein"}
            for status in statuses
        )
    else:
        calculated_formal = all(
            status
            in resolved_statuses
            | {"target_not_protein", "audited_missing_sequence"}
            for status in statuses
        )
    declared_formal = _manifest_bool(
        cache.get("formal_eligible"), f"{mapping.entity} cache formal_eligible"
    )
    if declared_formal is not calculated_formal:
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache formal flag disagrees with its rows"
        )
    if require_formal and declared_formal is not True:
        raise FeatureBundleError(
            f"{mapping.entity} retrieval cache is not formal-eligible"
        )
    return tuple(mask_ids)


def validate_embedding_assembly(
    assembly_manifest: Path | str,
    *,
    resolver: Mapping[str, Path] | None = None,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Recursively revalidate an assembly and reconstruct its raw matrix.

    ``resolver`` is a SHA-256-to-path index used by portable bundles.  It makes
    copied manifests independently verifiable even when their original
    absolute paths no longer exist.
    """

    manifest_path = Path(assembly_manifest).resolve()
    assembly = _read_json(manifest_path)
    if assembly.get("schema") != ASSEMBLY_SCHEMA:
        raise FeatureBundleError(f"{manifest_path} has a wrong assembly schema")
    entity = _strict_nonempty_string(assembly.get("entity"), "assembly entity")
    if entity not in ENTITY_NAMES:
        raise FeatureBundleError(f"unsupported assembly entity {entity!r}")
    mapping_path = _verify_artifact(
        assembly.get("mapping", {}), "assembly mapping", resolver=resolver
    )
    mapping = read_entity_mapping(mapping_path, entity)
    if assembly.get("mapping_sha256") != mapping.sha256:
        raise FeatureBundleError("assembly mapping hash disagrees with mapping content")
    row_count = _manifest_count(assembly.get("row_count"), "assembly row_count")
    raw_dimension = _manifest_count(
        assembly.get("raw_dimension"), "assembly raw_dimension"
    )
    if row_count != len(mapping.row_ids) or raw_dimension < 1:
        raise FeatureBundleError("assembly row_count/raw_dimension is false")
    raw_path = _verify_artifact(
        assembly.get("raw_matrix", {}), "assembly raw matrix", resolver=resolver
    )
    sidecar_path = _verify_artifact(
        assembly.get("row_ids_sidecar", {}),
        "assembly row-ID sidecar",
        resolver=resolver,
    )
    mask_path = _verify_artifact(
        assembly.get("missing_mask", {}), "assembly missing mask", resolver=resolver
    )
    raw = np.asarray(_load_numeric_matrix(raw_path, "assembly raw matrix"), dtype=np.float32)
    mask_raw = _load_npy(mask_path, "assembly missing mask")
    if raw.shape != (row_count, raw_dimension):
        raise FeatureBundleError("assembly raw matrix shape disagrees with its manifest")
    if mask_raw.dtype != np.bool_ or mask_raw.shape != (row_count,):
        raise FeatureBundleError("assembly missing mask must be boolean and row-aligned")
    mask = np.asarray(mask_raw, dtype=np.bool_)
    if _read_json(sidecar_path) != _row_sidecar(mapping):
        raise FeatureBundleError("assembly row-ID sidecar is not canonical")
    missing_count = _manifest_count(
        assembly.get("missing_count"), "assembly missing_count"
    )
    unpermitted_count = _manifest_count(
        assembly.get("unpermitted_missing_count"),
        "assembly unpermitted_missing_count",
    )
    zero_unk_count = _manifest_count(
        assembly.get("declared_zero_unk_count"),
        "assembly declared_zero_unk_count",
    )
    zero_unk_ids = _manifest_string_list(
        assembly.get("declared_zero_unk_row_ids"),
        "assembly declared_zero_unk_row_ids",
    )
    audited_missing_count = _manifest_count(
        assembly.get("declared_audited_missing_count"),
        "assembly declared_audited_missing_count",
    )
    audited_missing_ids = _manifest_string_list(
        assembly.get("declared_audited_missing_row_ids"),
        "assembly declared_audited_missing_row_ids",
    )
    declared_count = _manifest_count(
        assembly.get("declared_masked_row_count"),
        "assembly declared_masked_row_count",
    )
    declared_ids = _manifest_string_list(
        assembly.get("declared_masked_row_ids"),
        "assembly declared_masked_row_ids",
    )
    observed_ids = tuple(
        mapping.row_ids[int(index)] for index in np.flatnonzero(mask)
    )
    if (
        missing_count != int(mask.sum())
        or declared_count != int(mask.sum())
        or declared_ids != observed_ids
        or zero_unk_count != len(zero_unk_ids)
        or audited_missing_count != len(audited_missing_ids)
        or set(zero_unk_ids) & set(audited_missing_ids)
        or tuple(
            row_id
            for row_id in mapping.row_ids
            if row_id in set(zero_unk_ids).union(audited_missing_ids)
        )
        != observed_ids
        or zero_unk_count + audited_missing_count != declared_count
    ):
        raise FeatureBundleError(
            "assembly mask and declared zero-row IDs/count do not exactly agree"
        )
    policy = assembly.get("missing_sequence_policy")
    if not isinstance(policy, str) or policy not in {
        "strict",
        "zero-unk",
        "audited-missing-zero",
    }:
        raise FeatureBundleError("assembly has an invalid missing-sequence policy")
    if entity != "protein" and (mask.any() or policy != "strict"):
        raise FeatureBundleError("only protein assemblies may contain zero rows")
    if entity == "protein" and mask.any() and policy == "strict":
        raise FeatureBundleError("protein zero rows require an explicit missing policy")
    if policy == "zero-unk" and audited_missing_ids:
        raise FeatureBundleError("zero-unk cannot authorize audited-missing rows")
    if policy != "audited-missing-zero" and audited_missing_ids:
        raise FeatureBundleError(
            "audited missing rows require audited-missing-zero policy"
        )
    if unpermitted_count != 0:
        raise FeatureBundleError("assembly contains unpermitted missing rows")
    if not np.isfinite(raw).all():
        raise FeatureBundleError("formal assembly raw matrix contains NaN/Inf")
    if mask.any() and not np.array_equal(
        raw[mask], np.zeros((int(mask.sum()), raw_dimension), dtype=np.float32)
    ):
        raise FeatureBundleError("assembly masked raw rows are not exact zeros")

    dependencies = [manifest_path, mapping_path, raw_path, sidecar_path, mask_path]
    cache_path: Path | None = None
    cache_sha256: str | None = None
    cache_rows: tuple[Mapping[str, Any], ...] | None = None
    if entity in BIOLOGICAL_ENTITIES:
        cache_path = _verify_artifact(
            assembly.get("retrieval_cache", {}),
            f"{entity} assembly retrieval cache",
            resolver=resolver,
        )
        dependencies.append(cache_path)
        _validate_retrieval_cache_for_mask(
            cache_path,
            mapping,
            mask,
            missing_sequence_policy=policy,
            resolver=resolver,
        )
        cache_payload = _read_json(cache_path)
        cache_sha256 = _sha256(cache_path)
        raw_cache_rows = cache_payload.get("rows")
        if not isinstance(raw_cache_rows, list) or any(
            not isinstance(row, Mapping) for row in raw_cache_rows
        ):
            raise FeatureBundleError("assembly retrieval cache rows are malformed")
        cache_rows = tuple(raw_cache_rows)
        for row in cache_payload.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            for evidence_key in (
                "classification_evidence",
                "missing_sequence_evidence",
                "sequence_resolution_evidence",
            ):
                evidence = row.get(evidence_key)
                if isinstance(evidence, Mapping):
                    for artifact_key, label in (
                        ("audit_artifact", "audit artifact"),
                        ("candidate_import_artifact", "candidate import artifact"),
                    ):
                        dependencies.append(
                            _verify_artifact(
                                evidence.get(artifact_key, {}),
                                f"{entity} {evidence_key} {label}",
                                resolver=resolver,
                            )
                        )
    elif assembly.get("retrieval_cache") is not None:
        raise FeatureBundleError("semantic assembly unexpectedly records a retrieval cache")
    semantic_identity_status = assembly.get("semantic_identity_status")
    if entity in {"herb", "disease"} and semantic_identity_status not in {
        "legacy_self_attested_unverified",
        "independently_id_keyed_verified",
    }:
        raise FeatureBundleError("semantic assembly identity status is absent/unsupported")
    if entity not in {"herb", "disease"} and semantic_identity_status is not None:
        raise FeatureBundleError("biological assembly has a semantic identity status")
    semantic_condition = assembly.get("semantic_condition")
    if entity in {"herb", "disease"}:
        if semantic_identity_status == "independently_id_keyed_verified":
            _strict_nonempty_string(semantic_condition, "semantic condition")
        elif semantic_condition is not None:
            raise FeatureBundleError(
                "legacy semantic assembly cannot declare a verified condition"
            )
    elif semantic_condition is not None:
        raise FeatureBundleError("biological assembly declares a semantic condition")

    shard_records = assembly.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise FeatureBundleError("formal assembly has no provenance-bound shards")
    reconstructed = np.full(raw.shape, np.nan, dtype=np.float32)
    covered: set[str] = set()
    shard_failures: dict[str, str] = {}
    positions = {row_id: index for index, row_id in enumerate(mapping.row_ids)}
    extraction_validation_cache: dict[
        tuple[str, str, str, bool], dict[str, Any]
    ] = {}
    for index, record in enumerate(shard_records):
        row_ids, values, failures, shard_dependencies = (
            _validate_assembly_shard_record(
                record,
                mapping=mapping,
                cache_path=cache_path,
                resolver=resolver,
                live_source_validation=False,
                cache_sha256=cache_sha256,
                cache_rows=cache_rows,
                extraction_validation_cache=extraction_validation_cache,
            )
        )
        all_ids = set(row_ids) | set(failures)
        overlap = covered & all_ids
        if overlap:
            raise FeatureBundleError(
                f"assembly shard {index} overlaps prior rows: {sorted(overlap)[:5]}"
            )
        covered.update(all_ids)
        for row_id, value in zip(row_ids, values):
            reconstructed[positions[row_id]] = value
        shard_failures.update(failures)
        dependencies.extend(shard_dependencies)
    semantic_record_count = sum(
        isinstance(record, Mapping)
        and record.get("semantic_reduction_manifest") is not None
        for record in shard_records
    )
    if entity in {"herb", "disease"}:
        if semantic_identity_status == "independently_id_keyed_verified" and (
            len(shard_records) != 1 or semantic_record_count != 1
        ):
            raise FeatureBundleError(
                "verified semantic assembly must contain exactly one recursively "
                "validated semantic record"
            )
        if semantic_identity_status == "independently_id_keyed_verified":
            reduction_path = _verify_artifact(
                shard_records[0].get("semantic_reduction_manifest", {}),
                "semantic reduction manifest",
                resolver=resolver,
            )
            if _read_json(reduction_path).get("semantic_condition") != semantic_condition:
                raise FeatureBundleError(
                    "semantic assembly condition differs from its reduction manifest"
                )
        if semantic_identity_status == "legacy_self_attested_unverified" and (
            semantic_record_count != 0
        ):
            raise FeatureBundleError(
                "legacy semantic identity contradicts a verified semantic record"
            )
    if covered != set(mapping.row_ids):
        raise FeatureBundleError("assembly shards do not cover every canonical row")
    for index in np.flatnonzero(mask):
        reconstructed[int(index)] = 0.0
    if not np.array_equal(reconstructed, raw):
        raise FeatureBundleError(
            "assembly raw matrix cannot be reconstructed from its audited shards"
        )
    if any(
        row_id not in set(observed_ids) for row_id in shard_failures
    ):
        raise FeatureBundleError(
            "assembly contains a failed shard row that was not explicitly masked"
        )
    failures = assembly.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(record, Mapping) for record in failures
    ):
        raise FeatureBundleError("assembly failure records are malformed")
    failure_ids = tuple(record.get("row_id") for record in failures)
    if failure_ids != observed_ids or any(
        not isinstance(record.get("source_id"), str)
        or record.get("source_id") != mapping.source_ids[positions[record["row_id"]]]
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
        for record in failures
    ):
        raise FeatureBundleError(
            "assembly failure rows do not exactly match its declared mask"
        )
    reasons = assembly.get("formal_ineligibility_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise FeatureBundleError("assembly formal-ineligibility reasons are malformed")
    semantic_reason = "semantic_identity_legacy_self_attested_unverified"
    if entity in {"herb", "disease"}:
        if semantic_identity_status == "legacy_self_attested_unverified":
            if semantic_reason not in reasons:
                raise FeatureBundleError(
                    "legacy semantic identity is missing its formal-ineligibility reason"
                )
        elif semantic_reason in reasons:
            raise FeatureBundleError(
                "verified semantic identity retains a stale unverified reason"
            )
    formal = _manifest_bool(
        assembly.get("formal_eligible"), "assembly formal_eligible"
    )
    if assembly.get("random_imputation") is not False:
        raise FeatureBundleError("assembly does not strictly prohibit random imputation")
    recomputed_formal = not reasons
    if formal is not recomputed_formal:
        raise FeatureBundleError("assembly formal flag disagrees with its reasons")
    if require_formal and (formal is not True or reasons != []):
        raise IncompleteFeatureBundleError(
            f"assembly is not formal-eligible: {reasons}"
        )
    return {
        "manifest": assembly,
        "manifest_path": manifest_path,
        "entity": entity,
        "mapping": mapping,
        "raw": raw,
        "raw_path": raw_path,
        "mask": mask,
        "mask_path": mask_path,
        "sidecar_path": sidecar_path,
        "cache_path": cache_path,
        "dependencies": list(dict.fromkeys(path.resolve() for path in dependencies)),
    }


def _canonical_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for index, row in enumerate(result):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            result[index] *= -1.0
    return result


def validate_feature_component(
    component_manifest: Path | str,
    *,
    resolver: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Recursively verify assembly, reducer, mask, and final 64-D values."""

    manifest_path = Path(component_manifest).resolve()
    component = _read_json(manifest_path)
    if component.get("schema") != REDUCER_SCHEMA:
        raise FeatureBundleError(f"{manifest_path} has a wrong component schema")
    entity = _strict_nonempty_string(component.get("entity"), "component entity")
    if entity not in ENTITY_NAMES:
        raise FeatureBundleError(f"unsupported component entity {entity!r}")
    formal = _manifest_bool(
        component.get("formal_eligible"), "component formal_eligible"
    )
    reasons = component.get("formal_ineligibility_reasons")
    if reasons != [] or formal is not True:
        raise IncompleteFeatureBundleError(
            f"{entity} component is not formal-eligible: {reasons!r}"
        )
    if component.get("random_imputation") is not False:
        raise FeatureBundleError("component does not strictly prohibit random imputation")
    if component.get("status") != "complete":
        raise FeatureBundleError("component status is not complete")
    assembly_path = _verify_artifact(
        component.get("assembly_manifest", {}),
        f"{entity} assembly manifest",
        resolver=resolver,
    )
    assembly = validate_embedding_assembly(
        assembly_path, resolver=resolver, require_formal=True
    )
    if assembly["entity"] != entity:
        raise FeatureBundleError("component and assembly entities differ")
    if (
        component.get("semantic_identity_status")
        != assembly["manifest"].get("semantic_identity_status")
        or component.get("semantic_condition")
        != assembly["manifest"].get("semantic_condition")
    ):
        raise FeatureBundleError(
            "component semantic identity differs from its assembly"
        )
    mapping_path = _verify_artifact(
        component.get("mapping", {}), f"{entity} component mapping", resolver=resolver
    )
    mapping = read_entity_mapping(mapping_path, entity)
    if (
        component.get("mapping_sha256") != mapping.sha256
        or mapping.sha256 != assembly["mapping"].sha256
    ):
        raise FeatureBundleError("component mapping identity differs from its assembly")
    matrix_path = _verify_artifact(
        component.get("matrix", {}), f"{entity} component matrix", resolver=resolver
    )
    sidecar_path = _verify_artifact(
        component.get("row_ids_sidecar", {}),
        f"{entity} component row-ID sidecar",
        resolver=resolver,
    )
    mask_path = _verify_artifact(
        component.get("missing_mask", {}),
        f"{entity} component missing mask",
        resolver=resolver,
    )
    reducer_path = _verify_artifact(
        component.get("reducer_checkpoint", {}),
        f"{entity} reducer checkpoint",
        resolver=resolver,
    )
    matrix = np.asarray(
        _load_numeric_matrix(matrix_path, f"{entity} component matrix"),
        dtype=np.float32,
    )
    mask_raw = _load_npy(mask_path, f"{entity} component missing mask")
    if mask_raw.dtype != np.bool_ or mask_raw.shape != (len(mapping.row_ids),):
        raise FeatureBundleError("component missing mask must be boolean and row-aligned")
    mask = np.asarray(mask_raw, dtype=np.bool_)
    if not np.array_equal(mask, assembly["mask"]):
        raise FeatureBundleError("component mask differs from its assembly mask")
    if _read_json(sidecar_path) != _row_sidecar(mapping):
        raise FeatureBundleError("component row-ID sidecar is not canonical")
    shape = component.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int for value in shape)
        or shape != list(matrix.shape)
        or matrix.shape != (len(mapping.row_ids), 64)
    ):
        raise FeatureBundleError("component shape does not equal its measured 64-D matrix")
    if component.get("dtype") != str(matrix.dtype):
        raise FeatureBundleError("component dtype differs from its measured matrix")
    if not np.isfinite(matrix).all():
        raise FeatureBundleError("component matrix contains NaN or infinity")
    zero_unk_count = _manifest_count(
        component.get("declared_zero_unk_count"),
        "component declared_zero_unk_count",
    )
    zero_unk_ids = _manifest_string_list(
        component.get("declared_zero_unk_row_ids"),
        "component declared_zero_unk_row_ids",
    )
    audited_missing_count = _manifest_count(
        component.get("declared_audited_missing_count"),
        "component declared_audited_missing_count",
    )
    audited_missing_ids = _manifest_string_list(
        component.get("declared_audited_missing_row_ids"),
        "component declared_audited_missing_row_ids",
    )
    declared_count = _manifest_count(
        component.get("declared_masked_row_count"),
        "component declared_masked_row_count",
    )
    declared_ids = _manifest_string_list(
        component.get("declared_masked_row_ids"),
        "component declared_masked_row_ids",
    )
    observed_ids = tuple(
        mapping.row_ids[int(index)] for index in np.flatnonzero(mask)
    )
    if (
        declared_count != int(mask.sum())
        or declared_ids != observed_ids
        or zero_unk_count != assembly["manifest"].get("declared_zero_unk_count")
        or zero_unk_ids
        != tuple(assembly["manifest"].get("declared_zero_unk_row_ids", []))
        or audited_missing_count
        != assembly["manifest"].get("declared_audited_missing_count")
        or audited_missing_ids
        != tuple(
            assembly["manifest"].get("declared_audited_missing_row_ids", [])
        )
        or zero_unk_count + audited_missing_count != declared_count
    ):
        raise FeatureBundleError(
            "component mask and declared zero-row IDs/count do not agree"
        )
    if component.get("missing_sequence_policy") != assembly["manifest"].get(
        "missing_sequence_policy"
    ):
        raise FeatureBundleError("component/assembly missing-sequence policies differ")
    component_cache = component.get("retrieval_cache")
    if entity in BIOLOGICAL_ENTITIES:
        cache_path = _verify_artifact(
            component_cache,
            f"{entity} component retrieval cache",
            resolver=resolver,
        )
        if assembly["cache_path"] is None or _sha256(cache_path) != _sha256(
            assembly["cache_path"]
        ):
            raise FeatureBundleError("component retrieval cache differs from assembly")
    elif component_cache is not None:
        raise FeatureBundleError("semantic component unexpectedly records a cache")
    try:
        with np.load(reducer_path, allow_pickle=False) as reducer:
            mean = np.asarray(reducer["mean"], dtype=np.float64)
            components = np.asarray(reducer["components"], dtype=np.float64)
            singular_values = np.asarray(
                reducer["singular_values"], dtype=np.float64
            )
    except (OSError, ValueError, KeyError) as error:
        raise FeatureBundleError(f"cannot load reducer checkpoint: {error}") from error
    raw = np.asarray(assembly["raw"], dtype=np.float64)
    if (
        mean.shape != (raw.shape[1],)
        or components.shape != (64, raw.shape[1])
        or singular_values.ndim != 1
        or not np.isfinite(mean).all()
        or not np.isfinite(components).all()
        or not np.isfinite(singular_values).all()
    ):
        raise FeatureBundleError("reducer checkpoint arrays have invalid shape/values")
    recomputed = np.zeros(matrix.shape, dtype=np.float32)
    recomputed[~mask] = np.asarray(
        (raw[~mask] - mean) @ components.T, dtype=np.float32
    )
    if not np.array_equal(recomputed, matrix):
        raise FeatureBundleError(
            "component matrix differs from (raw - mean) @ components.T"
        )
    if mask.any() and not np.array_equal(
        matrix[mask], np.zeros((int(mask.sum()), 64), dtype=np.float32)
    ):
        raise FeatureBundleError("component masked rows are not exact zeros")
    audit_matrix = matrix[~mask]
    unique_rows = int(np.unique(audit_matrix, axis=0).shape[0])
    varying_columns = int(np.count_nonzero(np.ptp(audit_matrix, axis=0) > 0.0))
    if unique_rows < 2 or varying_columns == 0:
        raise FeatureBundleError("component unmasked matrix is degenerate")
    if _manifest_count(component.get("unique_rows"), "component unique_rows") != unique_rows:
        raise FeatureBundleError("component unique_rows declaration is false")
    if _manifest_count(
        component.get("varying_columns"), "component varying_columns"
    ) != varying_columns:
        raise FeatureBundleError("component varying_columns declaration is false")
    dependencies = list(assembly["dependencies"]) + [
        manifest_path,
        mapping_path,
        matrix_path,
        sidecar_path,
        mask_path,
        reducer_path,
    ]
    reducer_source = component.get("reducer_source")
    if reducer_source is not None:
        dependencies.append(
            _verify_artifact(
                reducer_source,
                f"{entity} supplied reducer source",
                resolver=resolver,
            )
        )
    return {
        "manifest": component,
        "manifest_path": manifest_path,
        "entity": entity,
        "mapping": mapping,
        "matrix": matrix,
        "matrix_path": matrix_path,
        "sidecar_path": sidecar_path,
        "mask": mask,
        "mask_path": mask_path,
        "reducer_path": reducer_path,
        "assembly": assembly,
        "dependencies": list(dict.fromkeys(path.resolve() for path in dependencies)),
    }


def reduce_embedding_assembly(
    *,
    assembly_manifest: Path | str,
    output_dir: Path | str,
    output_dimension: int = 64,
    reducer_checkpoint: Path | str | None = None,
    allow_ineligible: bool = False,
) -> Path:
    """Fit/apply a deterministic centered-SVD reducer to one complete assembly."""

    assembly_manifest = Path(assembly_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "component_manifest.json"
    validated = validate_embedding_assembly(
        assembly_manifest, require_formal=True
    )
    assembly = validated["manifest"]
    entity = validated["entity"]
    mapping = validated["mapping"]
    raw = np.asarray(validated["raw"], dtype=np.float64)
    missing_mask = np.asarray(validated["mask"], dtype=np.bool_)
    zero_unk_count = _manifest_count(
        assembly.get("declared_zero_unk_count"),
        "assembly declared_zero_unk_count",
    )
    audited_missing_count = _manifest_count(
        assembly.get("declared_audited_missing_count"),
        "assembly declared_audited_missing_count",
    )
    masked_count = int(missing_mask.sum())
    reasons: list[str] = []
    if output_dimension != 64:
        reasons.append(f"output_dimension={output_dimension}, expected=64")
    fit_row_count = int((~missing_mask).sum())
    rank_bound = min(fit_row_count, raw.shape[1])
    if reducer_checkpoint is not None:
        # A supplied reducer may have been fitted jointly across entity types
        # (the formal semantic condition does exactly this).  Its local
        # application is bounded by input width, not by this entity's row
        # count.  The checkpoint is still shape-checked and recursively tied
        # to the shared semantic assembly below.
        rank_bound = raw.shape[1]
    if output_dimension <= 0 or output_dimension > rank_bound:
        reasons.append(
            f"output_dimension={output_dimension} exceeds reducer rank bound="
            f"{rank_bound}"
        )
    if reasons:
        blocked = {
            "schema": REDUCER_SCHEMA,
            "entity": entity,
            "assembly_manifest": _artifact(assembly_manifest),
            "formal_eligible": False,
            "formal_ineligibility_reasons": reasons,
            "random_imputation": False,
            "status": "blocked_before_reduction",
        }
        _atomic_json(manifest_path, blocked)
        if not allow_ineligible:
            raise IncompleteFeatureBundleError(
                f"cannot reduce incomplete/ineligible assembly: {reasons}"
            )
        return manifest_path

    reducer_source: dict[str, Any] | None = None
    fit_raw = raw[~missing_mask]
    if reducer_checkpoint is None:
        mean = fit_raw.mean(axis=0)
        centered = fit_raw - mean
        try:
            _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError as error:
            raise FeatureBundleError(f"reducer SVD failed: {error}") from error
        components = _canonical_component_signs(right[:output_dimension])
        algorithm = "deterministic_centered_svd_pca"
    else:
        reducer_path = Path(reducer_checkpoint).resolve()
        try:
            with np.load(reducer_path, allow_pickle=False) as checkpoint:
                mean = np.asarray(checkpoint["mean"], dtype=np.float64)
                components = np.asarray(checkpoint["components"], dtype=np.float64)
                singular_values = np.asarray(
                    checkpoint.get("singular_values", np.empty(0)), dtype=np.float64
                )
        except (OSError, ValueError, KeyError) as error:
            raise FeatureBundleError(
                f"cannot load reducer checkpoint {reducer_path}: {error}"
            ) from error
        reducer_source = _artifact(reducer_path)
        algorithm = "supplied_linear_reducer"
    if mean.shape != (raw.shape[1],) or components.shape != (
        output_dimension,
        raw.shape[1],
    ):
        raise FeatureBundleError(
            f"reducer shapes mean={mean.shape}, components={components.shape} do not "
            f"match raw shape {raw.shape} and output dimension {output_dimension}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise FeatureBundleError("reducer contains NaN or infinity")
    reduced = np.zeros((raw.shape[0], output_dimension), dtype=np.float32)
    transform_rows = ~missing_mask
    reduced[transform_rows] = np.asarray(
        (raw[transform_rows] - mean) @ components.T, dtype=np.float32
    )
    if not np.isfinite(reduced).all():
        raise FeatureBundleError("reducer emitted NaN or infinity")
    audit_rows = reduced[~missing_mask]
    unique_rows = int(np.unique(audit_rows, axis=0).shape[0])
    varying_columns = int(np.count_nonzero(np.ptp(audit_rows, axis=0) > 0.0))
    if unique_rows < 2 or varying_columns == 0:
        raise FeatureBundleError("reducer emitted a degenerate feature matrix")

    matrix_path = output_dir / STANDARD_FILENAMES[entity]
    sidecar_path = feature_row_ids_path(matrix_path)
    output_mask_path = output_dir / f"{matrix_path.name}.missing_mask.npy"
    checkpoint_path = output_dir / f"{entity}_reducer.npz"
    _atomic_matrix(matrix_path, reduced)
    _atomic_json(sidecar_path, _row_sidecar(mapping))
    _atomic_npy(output_mask_path, missing_mask)
    _atomic_npz(
        checkpoint_path,
        mean=np.asarray(mean, dtype=np.float64),
        components=np.asarray(components, dtype=np.float64),
        singular_values=np.asarray(singular_values, dtype=np.float64),
    )
    manifest = {
        "schema": REDUCER_SCHEMA,
        "entity": entity,
        "mapping": _artifact(mapping.path),
        "mapping_sha256": mapping.sha256,
        "assembly_manifest": _artifact(assembly_manifest),
        "retrieval_cache": assembly.get("retrieval_cache"),
        "missing_sequence_policy": assembly.get("missing_sequence_policy", "strict"),
        "sequence_resolution": assembly.get("sequence_resolution"),
        "semantic_identity_status": assembly.get("semantic_identity_status"),
        "semantic_condition": assembly.get("semantic_condition"),
        "declared_zero_unk_count": zero_unk_count,
        "declared_zero_unk_row_ids": assembly.get(
            "declared_zero_unk_row_ids", []
        ),
        "declared_audited_missing_count": audited_missing_count,
        "declared_audited_missing_row_ids": assembly.get(
            "declared_audited_missing_row_ids", []
        ),
        "declared_masked_row_count": masked_count,
        "declared_masked_row_ids": assembly.get("declared_masked_row_ids", []),
        "matrix": _artifact(matrix_path),
        "row_ids_sidecar": _artifact(sidecar_path),
        "missing_mask": _artifact(output_mask_path),
        "reducer_checkpoint": _artifact(checkpoint_path),
        "reducer_source": reducer_source,
        "reducer_algorithm": algorithm,
        "shape": list(reduced.shape),
        "dtype": str(reduced.dtype),
        "unique_rows": unique_rows,
        "varying_columns": varying_columns,
        "formal_eligible": True,
        "formal_ineligibility_reasons": [],
        "random_imputation": False,
        "status": "complete",
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def build_verified_semantic_components(
    *,
    semantic_reduction_manifest: Path | str,
    herb_mapping_path: Path | str,
    disease_mapping_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Build both semantic components from one recursively verified shared PCA.

    This is the only formal bridge from the independently ID-keyed semantic
    pipeline into a four-entity HerbHG bundle.  Historical order-claimed 64-D
    arrays remain diagnostic-only.
    """

    reduction_path = Path(semantic_reduction_manifest).resolve()
    from .ethnobotany_semantic_features import (  # noqa: PLC0415
        SemanticFeatureError,
        validate_shared_semantic_reduction,
    )

    try:
        reduction = validate_shared_semantic_reduction(reduction_path)
    except SemanticFeatureError as error:
        raise FeatureBundleError(
            f"cannot import formal semantic condition: {error}"
        ) from error
    mapping_paths = {
        "herb": Path(herb_mapping_path).resolve(),
        "disease": Path(disease_mapping_path).resolve(),
    }
    root = Path(output_dir).resolve()
    results: dict[str, Path] = {}
    for entity in ("herb", "disease"):
        mapping = read_entity_mapping(mapping_paths[entity], entity)
        reduction_record = reduction["manifest"]["entities"][entity]
        if reduction_record.get("mapping_sha256") != mapping.sha256:
            raise FeatureBundleError(
                f"semantic {entity} mapping differs from the requested mapping"
            )
        assembly = assemble_embedding_shards(
            entity=entity,
            mapping_path=mapping.path,
            shard_paths=[],
            output_dir=root / entity / "assembly",
            semantic_reduction_manifest=reduction_path,
        )
        component = reduce_embedding_assembly(
            assembly_manifest=assembly,
            output_dir=root / entity / "component",
            reducer_checkpoint=reduction["checkpoint_path"],
        )
        validated = validate_feature_component(component)
        expected = np.asarray(reduction["outputs"][entity]["matrix"], dtype=np.float32)
        if not np.array_equal(validated["matrix"], expected):
            raise FeatureBundleError(
                f"bundle {entity} component differs from the shared semantic output"
            )
        results[entity] = component
    return results


def finalize_feature_bundle(
    *,
    component_manifests: Sequence[Path | str],
    output_dir: Path | str,
    allow_ineligible: bool = False,
    profile: str = "herbhg-four-entity",
) -> Path:
    """Package canonical 64-D components and their complete provenance.

    ``hghda-biological-only`` exists solely for methods which consume no herb
    or disease semantic arrays.  A HerbHG runner supplies four expected feature
    paths and therefore cannot accidentally accept this reduced profile.
    """

    del allow_ineligible  # A portable formal bundle is always fail-closed.
    profiles = {
        "herbhg-four-entity": ENTITY_NAMES,
        "hghda-biological-only": BIOLOGICAL_ENTITIES,
    }
    required_entities = profiles.get(profile)
    if required_entities is None:
        raise FeatureBundleError(f"unsupported feature-bundle profile {profile!r}")
    if len(component_manifests) != len(required_entities):
        raise FeatureBundleError(
            f"{profile} finalization requires exactly {len(required_entities)} "
            "component manifests"
        )
    components: dict[str, dict[str, Any]] = {}
    for value in component_manifests:
        path = Path(value).resolve()
        validated = validate_feature_component(path)
        entity = validated["entity"]
        if entity not in required_entities or entity in components:
            raise FeatureBundleError(f"duplicate or unsupported component entity {entity!r}")
        components[entity] = validated
    missing_entities = sorted(set(required_entities).difference(components))
    if missing_entities:
        raise FeatureBundleError(f"feature bundle lacks entities: {missing_entities}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    object_records: dict[str, dict[str, Any]] = {}
    for validated in components.values():
        for source in validated["dependencies"]:
            source = Path(source).resolve()
            digest = _sha256(source)
            if digest in object_records:
                continue
            suffix = "".join(source.suffixes[-2:]) if source.name.endswith(
                ".manifest.json"
            ) else source.suffix
            destination = (
                output_dir
                / "provenance"
                / "objects"
                / digest[:2]
                / f"{digest}{suffix}"
            )
            _atomic_copy(source, destination)
            object_records[digest] = _bundle_artifact(destination, output_dir)

    entity_records: dict[str, Any] = {}
    for entity in required_entities:
        validated = components[entity]
        component = validated["manifest"]
        mapping = validated["mapping"]
        matrix = validated["matrix"]
        mask = validated["mask"]
        source_matrix = validated["matrix_path"]
        source_sidecar = validated["sidecar_path"]
        source_mask = validated["mask_path"]
        source_mapping = mapping.path
        source_reducer = validated["reducer_path"]
        component_path = validated["manifest_path"]
        source_assembly = validated["assembly"]["manifest_path"]
        observed_ids = tuple(
            mapping.row_ids[int(index)] for index in np.flatnonzero(mask)
        )
        zero_unk_ids = tuple(component.get("declared_zero_unk_row_ids", []))
        audited_missing_ids = tuple(
            component.get("declared_audited_missing_row_ids", [])
        )
        if tuple(
            row_id
            for row_id in mapping.row_ids
            if row_id in set(zero_unk_ids).union(audited_missing_ids)
        ) != observed_ids:
            raise FeatureBundleError(
                f"{entity} component mask declarations are not canonical"
            )
        source_cache = validated["assembly"]["cache_path"]
        matrix_path = output_dir / STANDARD_FILENAMES[entity]
        sidecar_path = feature_row_ids_path(matrix_path)
        mask_path = output_dir / f"{matrix_path.name}.missing_mask.npy"
        mapping_path = output_dir / "mappings" / source_mapping.name
        reducer_path = output_dir / "reducers" / f"{entity}_reducer.npz"
        cache_copy = (
            output_dir / "retrieval" / f"{entity}_retrieval_cache.json"
            if source_cache is not None
            else None
        )
        _atomic_copy(source_matrix, matrix_path)
        _atomic_copy(source_sidecar, sidecar_path)
        _atomic_copy(source_mask, mask_path)
        _atomic_copy(source_mapping, mapping_path)
        _atomic_copy(source_reducer, reducer_path)
        if source_cache is not None and cache_copy is not None:
            _atomic_copy(source_cache, cache_copy)
        component_object = object_records[_sha256(component_path)]
        assembly_object = object_records[_sha256(source_assembly)]
        entity_records[entity] = {
            "matrix": _bundle_artifact(matrix_path, output_dir),
            "row_ids_sidecar": _bundle_artifact(sidecar_path, output_dir),
            "missing_mask": _bundle_artifact(mask_path, output_dir),
            "mapping": _bundle_artifact(mapping_path, output_dir),
            "mapping_sha256": mapping.sha256,
            "component_manifest": component_object,
            "assembly_manifest": assembly_object,
            "reducer_checkpoint": _bundle_artifact(reducer_path, output_dir),
            "retrieval_cache": (
                _bundle_artifact(cache_copy, output_dir)
                if cache_copy is not None
                else None
            ),
            "missing_sequence_policy": component.get(
                "missing_sequence_policy", "strict"
            ),
            "sequence_resolution": component.get("sequence_resolution"),
            "semantic_identity_status": component.get("semantic_identity_status"),
            "semantic_condition": component.get("semantic_condition"),
            "declared_zero_unk_count": int(
                component.get("declared_zero_unk_count", 0)
            ),
            "declared_zero_unk_row_ids": list(zero_unk_ids),
            "declared_audited_missing_count": int(
                component.get("declared_audited_missing_count", 0)
            ),
            "declared_audited_missing_row_ids": list(audited_missing_ids),
            "declared_masked_row_count": int(mask.sum()),
            "declared_masked_row_ids": list(observed_ids),
            "shape": list(matrix.shape),
            "reducer_algorithm": component.get("reducer_algorithm"),
        }

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "dataset": "ethnobotany",
        "row_id_space": "protocol_raw_id",
        "profile": profile,
        "required_entities": list(required_entities),
        "entities": entity_records,
        "provenance_objects": [
            object_records[digest] for digest in sorted(object_records)
        ],
        "formal_eligible": True,
        "formal_ineligibility_reasons": [],
        "random_imputation": False,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    _atomic_json(manifest_path, bundle)
    validate_feature_bundle(manifest_path)
    return manifest_path


def _bundle_local_artifact(
    record: Mapping[str, Any], bundle_root: Path, label: str
) -> Path:
    path = _verify_artifact(record, label)
    try:
        relative = path.relative_to(bundle_root)
    except ValueError as error:
        raise FeatureBundleError(f"{label} escapes the bundle directory") from error
    if record.get("relative_path") != str(relative):
        raise FeatureBundleError(f"{label} has a false bundle-relative path")
    return path


def validate_feature_bundle(
    manifest_path: Path | str,
    *,
    expected_mapping_sha256: Mapping[str, str] | None = None,
    expected_row_ids: Mapping[str, Sequence[str]] | None = None,
    expected_feature_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Independently and recursively verify a portable feature bundle."""

    manifest_path = Path(manifest_path).resolve()
    bundle_root = manifest_path.parent
    bundle = _read_json(manifest_path)
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("dataset") != "ethnobotany"
        or bundle.get("row_id_space") != "protocol_raw_id"
    ):
        raise FeatureBundleError("feature bundle identity/schema is invalid")
    if _manifest_bool(bundle.get("formal_eligible"), "bundle formal_eligible") is not True:
        raise IncompleteFeatureBundleError("feature bundle is not formal-eligible")
    if bundle.get("formal_ineligibility_reasons") != []:
        raise FeatureBundleError("formal feature bundle contains ineligibility reasons")
    if bundle.get("random_imputation") is not False:
        raise FeatureBundleError("feature bundle does not prohibit random imputation")
    objects = bundle.get("provenance_objects")
    if not isinstance(objects, list) or not objects:
        raise FeatureBundleError("feature bundle has no portable provenance closure")
    resolver: dict[str, Path] = {}
    for index, record in enumerate(objects):
        path = _bundle_local_artifact(
            record, bundle_root, f"provenance object {index}"
        )
        digest = _strict_sha256(record.get("sha256"), "provenance object sha256")
        if digest in resolver:
            raise FeatureBundleError("feature bundle repeats a provenance object digest")
        resolver[digest] = path
    profiles = {
        "herbhg-four-entity": ENTITY_NAMES,
        "hghda-biological-only": BIOLOGICAL_ENTITIES,
    }
    profile = bundle.get("profile")
    required_entities = profiles.get(profile)
    if required_entities is None or bundle.get("required_entities") != list(
        required_entities
    ):
        raise FeatureBundleError("feature bundle profile/entity declaration is invalid")
    entities = bundle.get("entities")
    if not isinstance(entities, Mapping) or set(entities) != set(required_entities):
        raise FeatureBundleError(
            f"{profile} bundle must contain exactly {list(required_entities)}"
        )
    if expected_feature_paths is not None and set(expected_feature_paths) != set(
        required_entities
    ):
        raise FeatureBundleError(
            "runner-selected feature entities do not match the bundle profile"
        )
    diagnostics: dict[str, Any] = {}
    for entity in required_entities:
        record = entities[entity]
        if not isinstance(record, Mapping):
            raise FeatureBundleError(f"bundle {entity} record is malformed")
        matrix_path = _bundle_local_artifact(
            record.get("matrix", {}), bundle_root, f"bundle {entity} matrix"
        )
        sidecar_path = _bundle_local_artifact(
            record.get("row_ids_sidecar", {}),
            bundle_root,
            f"bundle {entity} row-ID sidecar",
        )
        mask_path = _bundle_local_artifact(
            record.get("missing_mask", {}),
            bundle_root,
            f"bundle {entity} missing mask",
        )
        mapping_path = _bundle_local_artifact(
            record.get("mapping", {}), bundle_root, f"bundle {entity} mapping"
        )
        reducer_path = _bundle_local_artifact(
            record.get("reducer_checkpoint", {}),
            bundle_root,
            f"bundle {entity} reducer",
        )
        component_path = _bundle_local_artifact(
            record.get("component_manifest", {}),
            bundle_root,
            f"bundle {entity} component manifest",
        )
        assembly_path = _bundle_local_artifact(
            record.get("assembly_manifest", {}),
            bundle_root,
            f"bundle {entity} assembly manifest",
        )
        validated = validate_feature_component(component_path, resolver=resolver)
        if validated["entity"] != entity:
            raise FeatureBundleError("bundle entity/component identity differs")
        if (
            record.get("semantic_identity_status")
            != validated["manifest"].get("semantic_identity_status")
            or record.get("semantic_condition")
            != validated["manifest"].get("semantic_condition")
        ):
            raise FeatureBundleError(
                f"bundle {entity} semantic identity differs from its component"
            )
        mapping = read_entity_mapping(mapping_path, entity)
        if (
            record.get("mapping_sha256") != mapping.sha256
            or mapping.sha256 != validated["mapping"].sha256
            or _sha256(assembly_path)
            != _sha256(validated["assembly"]["manifest_path"])
            or _sha256(reducer_path) != _sha256(validated["reducer_path"])
        ):
            raise FeatureBundleError(f"bundle {entity} provenance hash chain is broken")
        matrix = np.asarray(
            _load_numeric_matrix(matrix_path, f"bundle {entity} matrix"),
            dtype=np.float32,
        )
        mask_raw = _load_npy(mask_path, f"bundle {entity} missing mask")
        if (
            matrix.shape != validated["matrix"].shape
            or not np.array_equal(matrix, validated["matrix"])
            or mask_raw.dtype != np.bool_
            or not np.array_equal(mask_raw, validated["mask"])
            or _read_json(sidecar_path) != _row_sidecar(mapping)
        ):
            raise FeatureBundleError(
                f"bundle {entity} public artifacts differ from recursive provenance"
            )
        shape = record.get("shape")
        if (
            not isinstance(shape, list)
            or any(type(value) is not int for value in shape)
            or shape != list(matrix.shape)
        ):
            raise FeatureBundleError(f"bundle {entity} shape declaration is false")
        observed_ids = tuple(
            mapping.row_ids[int(index)] for index in np.flatnonzero(mask_raw)
        )
        zero_unk_count = _manifest_count(
            record.get("declared_zero_unk_count"),
            f"bundle {entity} declared_zero_unk_count",
        )
        zero_unk_ids = _manifest_string_list(
            record.get("declared_zero_unk_row_ids"),
            f"bundle {entity} declared_zero_unk_row_ids",
        )
        audited_missing_count = _manifest_count(
            record.get("declared_audited_missing_count"),
            f"bundle {entity} declared_audited_missing_count",
        )
        audited_missing_ids = _manifest_string_list(
            record.get("declared_audited_missing_row_ids"),
            f"bundle {entity} declared_audited_missing_row_ids",
        )
        if (
            _manifest_count(
                record.get("declared_masked_row_count"),
                f"bundle {entity} declared_masked_row_count",
            )
            != int(mask_raw.sum())
            or _manifest_string_list(
                record.get("declared_masked_row_ids"),
                f"bundle {entity} declared_masked_row_ids",
            )
            != observed_ids
            or zero_unk_count != len(zero_unk_ids)
            or audited_missing_count != len(audited_missing_ids)
            or zero_unk_count + audited_missing_count != int(mask_raw.sum())
            or tuple(
                row_id
                for row_id in mapping.row_ids
                if row_id in set(zero_unk_ids).union(audited_missing_ids)
            )
            != observed_ids
        ):
            raise FeatureBundleError(
                f"bundle {entity} mask and declared zero rows disagree"
            )
        if expected_mapping_sha256 is not None and (
            entity not in expected_mapping_sha256
            or record.get("mapping_sha256") != expected_mapping_sha256[entity]
        ):
            raise FeatureBundleError(f"bundle {entity} mapping differs from protocol")
        if expected_row_ids is not None and (
            entity not in expected_row_ids
            or mapping.row_ids
            != tuple(str(value) for value in expected_row_ids[entity])
        ):
            raise FeatureBundleError(f"bundle {entity} row order differs from protocol")
        if expected_feature_paths is not None and matrix_path != Path(
            expected_feature_paths[entity]
        ).resolve():
            raise FeatureBundleError(f"bundle {entity} matrix is not runner-selected")
        cache_path: Path | None = None
        if entity in BIOLOGICAL_ENTITIES:
            cache_path = _bundle_local_artifact(
                record.get("retrieval_cache", {}),
                bundle_root,
                f"bundle {entity} retrieval cache",
            )
            if validated["assembly"]["cache_path"] is None or _sha256(
                cache_path
            ) != _sha256(validated["assembly"]["cache_path"]):
                raise FeatureBundleError(
                    f"bundle {entity} public cache differs from recursive provenance"
                )
        elif record.get("retrieval_cache") is not None:
            raise FeatureBundleError("semantic bundle record unexpectedly has a cache")
        diagnostics[entity] = {
            "matrix_path": matrix_path,
            "sidecar_path": sidecar_path,
            "mask_path": mask_path,
            "mapping_path": mapping_path,
            "cache_path": cache_path,
            "mapping": mapping,
            "matrix": matrix,
            "mask": np.asarray(mask_raw, dtype=np.bool_),
            "component": validated,
        }
    return {
        "manifest": bundle,
        "manifest_path": manifest_path,
        "resolver": resolver,
        "entities": diagnostics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="create/resume a retrieval cache")
    cache.add_argument("--entity", choices=BIOLOGICAL_ENTITIES, required=True)
    cache.add_argument("--mapping", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--import-json", type=Path)
    cache.add_argument(
        "--missing-sequence-policy",
        choices=("strict", "zero-unk", "audited-missing-zero"),
        default="strict",
        help=(
            "Protein only: strict rejects every unresolved target; zero-unk permits "
            "only authoritative non-protein targets; audited-missing-zero additionally "
            "permits exact rows bound to the frozen missing-sequence audit."
        ),
    )
    cache.add_argument(
        "--ambiguous-sequence-policy",
        choices=("strict", "mean-distinct-first-tier"),
        default="strict",
        help=(
            "Protein only: accept explicit audit-backed sequence sets and average "
            "their per-sequence ESM embeddings at the first frozen evidence tier."
        ),
    )
    cache.add_argument(
        "--accession-allowlist",
        type=Path,
        help="JSON object mapping a source/gene ID to one approved accession.",
    )
    cache.add_argument("--require-complete", action="store_true")

    assemble = subparsers.add_parser("assemble", help="merge raw embedding shards")
    assemble.add_argument("--entity", choices=ENTITY_NAMES, required=True)
    assemble.add_argument("--mapping", type=Path, required=True)
    assemble.add_argument("--shard", action="append", type=Path, default=[])
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--retrieval-cache", type=Path)
    assemble.add_argument("--semantic-reduction-manifest", type=Path)
    assemble.add_argument("--allow-incomplete", action="store_true")

    reduce = subparsers.add_parser("reduce", help="reduce one raw assembly")
    reduce.add_argument("--assembly-manifest", type=Path, required=True)
    reduce.add_argument("--output-dir", type=Path, required=True)
    reduce.add_argument("--output-dimension", type=int, default=64)
    reduce.add_argument("--reducer-checkpoint", type=Path)
    reduce.add_argument("--allow-ineligible", action="store_true")

    semantic = subparsers.add_parser(
        "semantic-components",
        help="build herb/disease components from the verified shared semantic PCA",
    )
    semantic.add_argument("--semantic-reduction-manifest", type=Path, required=True)
    semantic.add_argument("--herb-mapping", type=Path, required=True)
    semantic.add_argument("--disease-mapping", type=Path, required=True)
    semantic.add_argument("--output-dir", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="package a complete method-specific component profile"
    )
    finalize.add_argument(
        "--component-manifest", action="append", type=Path, required=True
    )
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument(
        "--profile",
        choices=("herbhg-four-entity", "hghda-biological-only"),
        default="herbhg-four-entity",
    )
    finalize.add_argument("--allow-ineligible", action="store_true")

    status = subparsers.add_parser("status", help="print a saved JSON manifest")
    status.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "cache":
            result: Any = update_retrieval_cache(
                entity=args.entity,
                mapping_path=args.mapping,
                output_path=args.output,
                import_path=args.import_json,
                missing_sequence_policy=args.missing_sequence_policy,
                ambiguous_sequence_policy=args.ambiguous_sequence_policy,
                accession_allowlist=args.accession_allowlist,
                require_complete=args.require_complete,
            )
        elif args.command == "assemble":
            result = assemble_embedding_shards(
                entity=args.entity,
                mapping_path=args.mapping,
                shard_paths=args.shard,
                output_dir=args.output_dir,
                retrieval_cache=args.retrieval_cache,
                semantic_reduction_manifest=args.semantic_reduction_manifest,
                allow_incomplete=args.allow_incomplete,
            )
        elif args.command == "reduce":
            result = reduce_embedding_assembly(
                assembly_manifest=args.assembly_manifest,
                output_dir=args.output_dir,
                output_dimension=args.output_dimension,
                reducer_checkpoint=args.reducer_checkpoint,
                allow_ineligible=args.allow_ineligible,
            )
        elif args.command == "semantic-components":
            result = {
                entity: str(path)
                for entity, path in build_verified_semantic_components(
                    semantic_reduction_manifest=args.semantic_reduction_manifest,
                    herb_mapping_path=args.herb_mapping,
                    disease_mapping_path=args.disease_mapping,
                    output_dir=args.output_dir,
                ).items()
            }
        elif args.command == "finalize":
            result = finalize_feature_bundle(
                component_manifests=args.component_manifest,
                output_dir=args.output_dir,
                allow_ineligible=args.allow_ineligible,
                profile=args.profile,
            )
        else:
            result = _read_json(args.manifest.resolve())
    except FeatureBundleError as error:
        print(f"Ethnobotany feature-bundle command failed: {error}", file=os.sys.stderr)
        return 2
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
