"""Identity-safe feature loading for the central benchmark protocol.

Feature matrix shape alone cannot establish which entity owns each row.  A
matrix used by a fair runner therefore has a sibling JSON
sidecar named ``<matrix filename>.row_ids.json``.  The sidecar records the
protocol raw ID belonging to each source row and binds those IDs to a frozen
mapping digest.  For Ethnobotany this is the authoritative mapping CSV hash;
for TCM-suite it is the digest of the canonical raw-ID list recorded by the
frozen protocol. Loading joins on those IDs and returns the matrix in the
protocol's canonical entity order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_ROW_IDS_SCHEMA = "astra-feature-row-ids/v1"
FEATURE_ROW_IDS_SUFFIX = ".row_ids.json"


class FeatureContractError(ValueError):
    """Raised when feature rows cannot be joined safely to protocol IDs."""


@dataclass(frozen=True)
class AlignedFeatureMatrix:
    """A validated feature matrix reordered to canonical protocol IDs."""

    matrix: np.ndarray
    diagnostics: Mapping[str, Any]


def feature_row_ids_path(matrix_path: Path | str) -> Path:
    """Return the mandatory sibling row-identity sidecar path."""

    matrix_path = Path(matrix_path)
    return matrix_path.with_name(matrix_path.name + FEATURE_ROW_IDS_SUFFIX)


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


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FeatureContractError(f"required feature matrix does not exist: {path}")
    try:
        matrix = (
            np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npy"
            else np.loadtxt(path)
        )
    except (OSError, ValueError) as error:
        raise FeatureContractError(f"cannot read feature matrix {path}: {error}") from error
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise FeatureContractError(
            f"feature matrix {path} must be two-dimensional, got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise FeatureContractError(f"feature matrix {path} must be numeric")
    if not np.isfinite(matrix).all():
        raise FeatureContractError(f"feature matrix {path} contains NaN or infinity")
    return matrix


def _load_row_ids(
    path: Path, *, entity: str, mapping_sha256: str
) -> tuple[str, ...]:
    if not path.is_file():
        raise FeatureContractError(
            f"missing {entity} feature row-ID sidecar: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeatureContractError(
            f"cannot read feature sidecar {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise FeatureContractError(f"feature sidecar {path} must contain a JSON object")
    if payload.get("schema") != FEATURE_ROW_IDS_SCHEMA:
        raise FeatureContractError(
            f"feature sidecar {path} schema must be {FEATURE_ROW_IDS_SCHEMA!r}"
        )
    if payload.get("entity") != entity:
        raise FeatureContractError(
            f"feature sidecar {path} declares entity {payload.get('entity')!r}, "
            f"expected {entity!r}"
        )
    if payload.get("id_space") != "protocol_raw_id":
        raise FeatureContractError(
            f"feature sidecar {path} id_space must be 'protocol_raw_id'"
        )
    declared_mapping_sha256 = payload.get("mapping_sha256")
    if declared_mapping_sha256 != mapping_sha256:
        raise FeatureContractError(
            f"feature sidecar {path} mapping_sha256 does not match the "
            f"frozen {entity} mapping"
        )
    row_ids = payload.get("row_ids")
    if not isinstance(row_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in row_ids
    ):
        raise FeatureContractError(
            f"feature sidecar {path} row_ids must be a list of non-empty strings"
        )
    normalised = tuple(value.strip() for value in row_ids)
    if len(normalised) != len(set(normalised)):
        duplicates = sorted(
            value for value, count in Counter(normalised).items() if count > 1
        )
        raise FeatureContractError(
            f"feature sidecar {path} contains duplicate IDs: {duplicates[:5]}"
        )
    return normalised


def load_aligned_feature_matrix(
    matrix_path: Path | str,
    *,
    entity: str,
    canonical_ids: Sequence[str],
    mapping_sha256: str,
    expected_columns: int = 64,
    sidecar_path: Path | str | None = None,
) -> AlignedFeatureMatrix:
    """Validate a feature/ID pair and join it to canonical protocol order.

    ``canonical_ids`` are the raw IDs serialized by
    :class:`~astra.benchmark.protocol.BenchmarkProtocol`.  Missing, duplicate,
    or unexpected IDs fail closed; positional coincidence is never accepted as
    evidence of identity.
    """

    matrix_path = Path(matrix_path).resolve()
    entity = str(entity).strip().lower()
    canonical = tuple(str(value) for value in canonical_ids)
    mapping_sha256 = str(mapping_sha256).strip().lower()
    if len(mapping_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in mapping_sha256
    ):
        raise FeatureContractError(
            f"frozen {entity} mapping_sha256 must be a lowercase SHA-256 digest"
        )
    if not canonical:
        raise FeatureContractError(f"canonical {entity} mapping must not be empty")
    if len(canonical) != len(set(canonical)):
        raise FeatureContractError(f"canonical {entity} mapping contains duplicate IDs")

    matrix = _load_matrix(matrix_path)
    if matrix.shape[1] != int(expected_columns):
        raise FeatureContractError(
            f"{matrix_path} has {matrix.shape[1]} columns; expected {expected_columns}"
        )
    resolved_sidecar = (
        Path(sidecar_path).resolve()
        if sidecar_path is not None
        else feature_row_ids_path(matrix_path)
    )
    row_ids = _load_row_ids(
        resolved_sidecar,
        entity=entity,
        mapping_sha256=mapping_sha256,
    )
    if len(row_ids) != matrix.shape[0]:
        raise FeatureContractError(
            f"{resolved_sidecar} has {len(row_ids)} row IDs but {matrix_path} "
            f"has {matrix.shape[0]} rows"
        )

    source_ids = set(row_ids)
    canonical_set = set(canonical)
    missing = [value for value in canonical if value not in source_ids]
    unexpected = [value for value in row_ids if value not in canonical_set]
    if missing or unexpected:
        raise FeatureContractError(
            f"{entity} feature IDs disagree with the canonical mapping: "
            f"missing={len(missing)} {missing[:5]}, "
            f"unexpected={len(unexpected)} {unexpected[:5]}"
        )

    source_position = {raw_id: index for index, raw_id in enumerate(row_ids)}
    permutation = np.fromiter(
        (source_position[raw_id] for raw_id in canonical),
        dtype=np.int64,
        count=len(canonical),
    )
    aligned = np.ascontiguousarray(matrix[permutation])
    varying_columns = int(np.count_nonzero(np.ptp(aligned, axis=0) > 0.0))
    unique_rows = int(np.unique(aligned, axis=0).shape[0])
    degenerate = bool(unique_rows < 2 or varying_columns == 0)
    diagnostics: dict[str, Any] = {
        "path": str(matrix_path),
        "bytes": matrix_path.stat().st_size,
        "sha256": _sha256(matrix_path),
        "sidecar_path": str(resolved_sidecar),
        "sidecar_bytes": resolved_sidecar.stat().st_size,
        "sidecar_sha256": _sha256(resolved_sidecar),
        "sidecar_schema": FEATURE_ROW_IDS_SCHEMA,
        "id_space": "protocol_raw_id",
        "mapping_sha256": mapping_sha256,
        "shape": list(aligned.shape),
        "dtype": str(aligned.dtype),
        "source_order_is_canonical": bool(row_ids == canonical),
        "canonical_array_sha256": _array_sha256(aligned),
        "unique_rows": unique_rows,
        "varying_columns": varying_columns,
        "zero_variance_columns": int(aligned.shape[1] - varying_columns),
        "degenerate": degenerate,
    }
    return AlignedFeatureMatrix(matrix=aligned, diagnostics=diagnostics)
