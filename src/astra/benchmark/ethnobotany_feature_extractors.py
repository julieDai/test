#!/usr/bin/env python3
"""Resumable, identity-safe Ethnobotany molecular feature extraction.

This module performs the network/model stages intentionally kept out of
``ethnobotany_feature_bundle``.  It writes PubChem imports and ID-keyed NPZ
shards that can be consumed directly by ``astra-build-ethnobotany-features``.
No missing molecular feature is imputed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import select
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import parse, request

import numpy as np

from .ethnobotany_feature_bundle import (
    CACHE_SCHEMA,
    EntityMapping,
    FeatureBundleError,
    read_entity_mapping,
)


PUBCHEM_IMPORT_SCHEMA = "astra-ethnobotany-pubchem-import/v1"
EMBEDDING_SHARD_SCHEMA = "astra-ethnobotany-embedding-shard/v2"
EXTRACTION_MANIFEST_SCHEMA = "astra-ethnobotany-embedding-extraction/v2"
ORDER_CLAIM_IMPORT_SCHEMA = "astra-ethnobotany-order-claim-import/v1"
PUBCHEM_ENDPOINT = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/"
    "CanonicalSMILES,IsomericSMILES/JSON"
)
OFFICIAL_KPGT_COMMIT = "47dc1646c70b2138a157de481d24a1ac35d174cd"
KPGT_CONFIG = "base"
KPGT_DIMENSION = 2304
ESM2_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM2_LAYER = 30
ESM2_DIMENSION = 640
ESM2_MAX_RESIDUES = 1022
ESM2_SEQUENCE_CACHE_SCHEMA = "astra-esm2-sequence-embedding-cache/v1"
ESM2_SEQUENCE_SET_AGGREGATION = (
    "equal_weight_mean_of_distinct_sequence_embeddings"
)
KPGT_WORKER_INPUT_SCHEMA = "astra-kpgt-worker-input/v1"
KPGT_WORKER_PROTOCOL_SCHEMA = "astra-kpgt-worker-protocol/v1"
KPGT_EXPECTED_PYTHON_VERSION = "3.7.12"
KPGT_EXPECTED_TORCH_VERSION = "1.10.0"
KPGT_EXPECTED_DGL_VERSION = "0.7.2"
KPGT_EXPECTED_RDKIT_VERSION = "2021.09.5"


class ExtractionError(FeatureBundleError):
    """Raised when retrieval or extraction provenance is invalid."""


class EmbeddingBackend(Protocol):
    embedding_dimension: int
    provenance: Mapping[str, Any]

    def embed(
        self, values: Sequence[str], *, row_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return one finite embedding row for every input value."""


@dataclass(frozen=True)
class SequenceSetMember:
    sequence: str
    sequence_sha256: str
    accessions: tuple[str, ...]
    taxid: str
    annotation_releases: tuple[str, ...]


@dataclass(frozen=True)
class CacheRow:
    row_id: str
    source_id: str
    status: str
    value: str | None
    error: str | None
    sequence_set_tier: str | None = None
    sequence_set_aggregation: str | None = None
    sequence_set: tuple[SequenceSetMember, ...] = ()

    @property
    def embedding_values(self) -> tuple[str, ...]:
        if self.sequence_set:
            return tuple(member.sequence for member in self.sequence_set)
        return (self.value,) if self.value is not None else ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ExtractionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ExtractionError(f"JSON artifact {path} must contain an object")
    return payload


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ExtractionError(
            f"{label} must be a JSON integer, not {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise ExtractionError(f"{label} must be at least {minimum}")
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _normalise_cid(value: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ExtractionError(f"PubChem CID must be a positive decimal integer: {value!r}")
    return str(int(text))


def _default_post(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> bytes:
    req = request.Request(url, data=body, headers=dict(headers), method="POST")
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _parse_pubchem_response(raw: bytes, requested_cids: Sequence[str]) -> dict[str, str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
        properties = payload["PropertyTable"]["Properties"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ExtractionError(f"invalid PubChem property response: {error}") from error
    if not isinstance(properties, list):
        raise ExtractionError("PubChem Properties must be a JSON array")
    resolved: dict[str, str] = {}
    for record in properties:
        if not isinstance(record, Mapping) or "CID" not in record:
            raise ExtractionError("PubChem returned a property without CID")
        cid = _normalise_cid(str(record["CID"]))
        if cid in resolved:
            raise ExtractionError(f"PubChem returned CID {cid} more than once")
        smiles = next(
            (
                str(record[key]).strip()
                for key in (
                    "ConnectivitySMILES",
                    "CanonicalSMILES",
                    "IsomericSMILES",
                    "SMILES",
                )
                if record.get(key) is not None and str(record[key]).strip()
            ),
            None,
        )
        if smiles is None:
            raise ExtractionError(f"PubChem CID {cid} has no returned SMILES")
        resolved[cid] = smiles
    requested = set(requested_cids)
    returned = set(resolved)
    if returned != requested:
        missing = sorted(requested - returned, key=int)
        extra = sorted(returned - requested, key=int)
        raise ExtractionError(
            "PubChem CID set mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return resolved


def fetch_pubchem_smiles(
    *,
    mapping_path: Path | str,
    output_path: Path | str,
    batch_size: int = 100,
    timeout: float = 60.0,
    max_call_attempts: int = 2,
    endpoint: str = PUBCHEM_ENDPOINT,
    user_agent: str = "ASTRA-Scheme-B/1.0 (PubChem feature provenance)",
    post: Callable[[str, bytes, Mapping[str, str], float], bytes] | None = None,
) -> dict[str, Any]:
    """Fetch canonical SMILES by mapping-column CID with recursive isolation.

    Successful records are saved after every successful (sub-)batch.  A failed
    multi-CID request is bisected until individual unavailable CIDs carry an
    explicit failure reason.  Existing successful records are hash-bound to
    the same mapping and skipped on resume.
    """

    if batch_size < 1:
        raise ExtractionError("batch_size must be positive")
    if max_call_attempts < 1:
        raise ExtractionError("max_call_attempts must be positive")
    mapping = read_entity_mapping(mapping_path, "compound")
    output_path = Path(output_path).resolve()
    original_to_normalised: dict[str, str] = {}
    normalised_to_original: dict[str, str] = {}
    for source_id in mapping.source_ids:
        cid = _normalise_cid(source_id)
        original_to_normalised[source_id] = cid
        prior = normalised_to_original.setdefault(cid, source_id)
        if prior != source_id:
            raise ExtractionError(
                f"mapping has different source strings for the same CID: {prior!r}, "
                f"{source_id!r}"
            )

    records: dict[str, str] = {}
    failures: dict[str, str] = {}
    request_log: list[dict[str, Any]] = []
    created_at = _utc_now()
    if output_path.exists():
        prior = _read_json(output_path)
        metadata = prior.get("metadata", {})
        if prior.get("schema") != PUBCHEM_IMPORT_SCHEMA:
            raise ExtractionError(f"existing PubChem import {output_path} has wrong schema")
        if not isinstance(metadata, Mapping) or metadata.get("mapping_sha256") != mapping.sha256:
            raise ExtractionError(f"existing PubChem import {output_path} uses another mapping")
        if metadata.get("endpoint") != endpoint:
            raise ExtractionError(f"existing PubChem import {output_path} uses another endpoint")
        raw_records = prior.get("records", {})
        raw_failures = prior.get("failures", {})
        if not isinstance(raw_records, Mapping) or not isinstance(raw_failures, Mapping):
            raise ExtractionError("existing PubChem import records/failures must be objects")
        records = {str(key): str(value) for key, value in raw_records.items()}
        failures = {str(key): str(value) for key, value in raw_failures.items()}
        unknown = (set(records) | set(failures)).difference(mapping.source_ids)
        if unknown:
            raise ExtractionError(f"existing PubChem import has unknown IDs: {sorted(unknown)[:5]}")
        if any(not value.strip() for value in records.values()):
            raise ExtractionError("existing PubChem import contains an empty SMILES")
        request_log = [
            dict(item) for item in metadata.get("request_log", []) if isinstance(item, Mapping)
        ]
        created_at = str(metadata.get("created_at", created_at))

    def payload() -> dict[str, Any]:
        return {
            "schema": PUBCHEM_IMPORT_SCHEMA,
            "records": dict(sorted(records.items(), key=lambda item: int(item[0]))),
            "failures": dict(sorted(failures.items(), key=lambda item: int(item[0]))),
            "metadata": {
                "provider": "NIH PubChem PUG REST",
                "endpoint": endpoint,
                "method": "POST",
                "post_field": "cid",
                "requested_properties": ["CanonicalSMILES", "IsomericSMILES"],
                "response_smiles_preference": [
                    "ConnectivitySMILES",
                    "CanonicalSMILES",
                    "IsomericSMILES",
                    "SMILES",
                ],
                "strict_returned_cid_set": True,
                "recursive_failure_isolation": True,
                "mapping_path": str(mapping.path),
                "mapping_sha256": mapping.sha256,
                "batch_size": batch_size,
                "timeout_seconds": timeout,
                "max_call_attempts": max_call_attempts,
                "user_agent": user_agent,
                "created_at": created_at,
                "updated_at": _utc_now(),
                "resolved_count": len(records),
                "failed_count": len(failures),
                "pending_count": len(set(mapping.source_ids) - set(records) - set(failures)),
                "request_log": request_log,
            },
        }

    def save() -> None:
        _atomic_json(output_path, payload())

    transport = post or _default_post

    def call(cids: Sequence[str]) -> dict[str, str]:
        body = parse.urlencode({"cid": ",".join(cids)}).encode("ascii")
        cid_digest = hashlib.sha256(",".join(cids).encode("ascii")).hexdigest()
        body_digest = hashlib.sha256(body).hexdigest()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": user_agent,
        }
        last_error: Exception | None = None
        for attempt in range(1, max_call_attempts + 1):
            started = _utc_now()
            try:
                raw = transport(endpoint, body, headers, timeout)
                result = _parse_pubchem_response(raw, cids)
                request_log.append(
                    {
                        "at": started,
                        "cid_count": len(cids),
                        "ordered_cid_sha256": cid_digest,
                        "request_body_sha256": body_digest,
                        "response_sha256": hashlib.sha256(raw).hexdigest(),
                        "response_bytes": len(raw),
                        "attempt": attempt,
                        "status": "resolved",
                    }
                )
                return result
            except Exception as error:  # network/parser errors are isolated identically
                last_error = error
                request_log.append(
                    {
                        "at": started,
                        "cid_count": len(cids),
                        "ordered_cid_sha256": cid_digest,
                        "request_body_sha256": body_digest,
                        "attempt": attempt,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if attempt < max_call_attempts:
                    time.sleep(min(2.0 ** (attempt - 1), 4.0))
        assert last_error is not None
        raise last_error

    def resolve(cids: Sequence[str]) -> None:
        try:
            returned = call(cids)
        except Exception as error:
            if len(cids) > 1:
                middle = len(cids) // 2
                resolve(cids[:middle])
                resolve(cids[middle:])
                return
            source_id = normalised_to_original[cids[0]]
            failures[source_id] = f"{type(error).__name__}: {error}"
            save()
            return
        for cid, smiles in returned.items():
            source_id = normalised_to_original[cid]
            records[source_id] = smiles
            failures.pop(source_id, None)
        save()

    # Failed leaves are retried on every invocation; successful records are not.
    unresolved_cids = [
        normalised_to_original[cid]
        for cid in sorted(normalised_to_original, key=int)
        if normalised_to_original[cid] not in records
    ]
    for offset in range(0, len(unresolved_cids), batch_size):
        originals = unresolved_cids[offset : offset + batch_size]
        resolve([original_to_normalised[value] for value in originals])
    save()
    return _read_json(output_path)


def _sequence_set_members(raw: Mapping[str, Any], *, row_id: str) -> tuple[SequenceSetMember, ...]:
    if raw.get("value") is not None:
        raise ExtractionError(
            f"resolved_sequence_set protein row {row_id} must have value=null"
        )
    tier = raw.get("sequence_set_tier")
    aggregation = raw.get("sequence_set_aggregation")
    if not isinstance(tier, str) or not tier.strip():
        raise ExtractionError(f"protein row {row_id} has no sequence_set_tier")
    if aggregation != ESM2_SEQUENCE_SET_AGGREGATION:
        raise ExtractionError(f"protein row {row_id} has a wrong sequence-set aggregation")
    raw_members = raw.get("sequence_set")
    if not isinstance(raw_members, list) or not raw_members:
        raise ExtractionError(f"protein row {row_id} has an empty sequence_set")
    count = _strict_int(
        raw.get("sequence_set_count"),
        f"protein row {row_id} sequence_set_count",
        minimum=1,
    )
    if count != len(raw_members):
        raise ExtractionError(f"protein row {row_id} has a false sequence_set_count")
    members: list[SequenceSetMember] = []
    for index, member in enumerate(raw_members):
        if not isinstance(member, Mapping):
            raise ExtractionError(
                f"protein row {row_id} sequence_set[{index}] is not an object"
            )
        sequence = member.get("sequence")
        digest = member.get("sequence_sha256")
        accessions = member.get("accessions")
        taxid = member.get("taxid")
        releases = member.get("annotation_releases")
        if (
            not isinstance(sequence, str)
            or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence)
            or not isinstance(digest, str)
            or digest != hashlib.sha256(sequence.encode("ascii")).hexdigest()
        ):
            raise ExtractionError(
                f"protein row {row_id} sequence_set[{index}] has a false sequence/SHA"
            )
        if (
            not isinstance(accessions, list)
            or not accessions
            or any(not isinstance(value, str) or not value.strip() for value in accessions)
            or accessions != sorted(set(accessions))
        ):
            raise ExtractionError(
                f"protein row {row_id} sequence_set[{index}] has invalid accessions"
            )
        if not isinstance(taxid, str) or not taxid.isdecimal() or int(taxid) < 1:
            raise ExtractionError(
                f"protein row {row_id} sequence_set[{index}] has an invalid taxid"
            )
        if (
            not isinstance(releases, list)
            or any(not isinstance(value, str) or not value.strip() for value in releases)
            or releases != sorted(set(releases))
        ):
            raise ExtractionError(
                f"protein row {row_id} sequence_set[{index}] has invalid releases"
            )
        members.append(
            SequenceSetMember(
                sequence=sequence,
                sequence_sha256=digest,
                accessions=tuple(accessions),
                taxid=taxid,
                annotation_releases=tuple(releases),
            )
        )
    digests = [member.sequence_sha256 for member in members]
    if digests != sorted(digests) or len(set(digests)) != len(digests):
        raise ExtractionError(
            f"protein row {row_id} sequence_set must be distinct and SHA-sorted"
        )
    return tuple(members)


def _load_cache(
    cache_path: Path | str, mapping: EntityMapping, entity: str
) -> tuple[Path, list[CacheRow], dict[str, Any]]:
    path = Path(cache_path).resolve()
    cache = _read_json(path)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("entity") != entity:
        raise ExtractionError(f"retrieval cache {path} is not a {entity} cache")
    if cache.get("mapping_sha256") != mapping.sha256:
        raise ExtractionError(f"retrieval cache {path} belongs to another mapping")
    raw_rows = cache.get("rows")
    if not isinstance(raw_rows, list):
        raise ExtractionError(f"retrieval cache {path} has no row list")
    rows: list[CacheRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ExtractionError(f"retrieval cache {path} contains a non-object row")
        value = raw.get("value")
        status = str(raw.get("status", ""))
        sequence_set: tuple[SequenceSetMember, ...] = ()
        if status == "resolved_sequence_set":
            if entity != "protein":
                raise ExtractionError(
                    "resolved_sequence_set is valid only in a protein cache"
                )
            sequence_set = _sequence_set_members(
                raw, row_id=str(raw.get("row_id", ""))
            )
        rows.append(
            CacheRow(
                row_id=str(raw.get("row_id", "")),
                source_id=str(raw.get("source_id", "")),
                status=status,
                value=str(value) if value is not None else None,
                error=str(raw.get("error")) if raw.get("error") is not None else None,
                sequence_set_tier=(
                    str(raw["sequence_set_tier"]).strip() if sequence_set else None
                ),
                sequence_set_aggregation=(
                    str(raw["sequence_set_aggregation"]) if sequence_set else None
                ),
                sequence_set=sequence_set,
            )
        )
    expected = list(zip(mapping.row_ids, mapping.source_ids))
    actual = [(row.row_id, row.source_id) for row in rows]
    if actual != expected:
        raise ExtractionError(f"retrieval cache {path} rows are not in canonical mapping order")
    for row in rows:
        if row.status == "resolved" and (row.value is None or not row.value.strip()):
            raise ExtractionError(f"resolved {entity} cache row {row.row_id} has no value")
        if row.status == "resolved_sequence_set" and not row.sequence_set:
            raise ExtractionError(
                f"resolved_sequence_set protein row {row.row_id} has no sequences"
            )
    return path, rows, cache


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExtractionError(f"cannot inspect KPGT repository {repo}: {error}") from error
    return completed.stdout.strip()


def _source_tree_sha256(repo: Path) -> str:
    tracked = [line for line in _git_output(repo, "ls-files").splitlines() if line]
    if not tracked:
        raise ExtractionError(f"KPGT repository {repo} has no tracked files")
    digest = hashlib.sha256()
    for relative in sorted(tracked):
        path = repo / relative
        if not path.is_file():
            raise ExtractionError(f"tracked KPGT source is absent: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _python_source_tree_sha256(root: Path) -> str:
    root = root.resolve()
    sources = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not sources:
        raise ExtractionError(f"Python source tree has no .py files: {root}")
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def kpgt_source_provenance(
    repo_path: Path | str,
    checkpoint_path: Path | str,
    *,
    expected_commit: str = OFFICIAL_KPGT_COMMIT,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not (repo / "src/model/light.py").is_file():
        raise ExtractionError(f"KPGT source tree is incomplete: {repo}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ExtractionError(f"KPGT checkpoint is absent or empty: {checkpoint}")
    expected_checkpoint_sha256 = str(expected_checkpoint_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256):
        raise ExtractionError("expected KPGT checkpoint SHA-256 must have 64 hex digits")
    observed_checkpoint_sha256 = _sha256(checkpoint)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ExtractionError(
            "KPGT checkpoint SHA-256 does not match the reviewed expected hash"
        )
    commit = _git_output(repo, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ExtractionError(
            f"KPGT checkout commit {commit} != required pinned commit {expected_commit}"
        )
    tracked_dirty = _git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise ExtractionError("KPGT checkout has modified tracked files")
    return {
        "implementation": "official KPGT LiGhT generate_fps",
        "repository": str(repo),
        "repository_commit": commit,
        "expected_repository_commit": expected_commit,
        "repository_tree_sha256": _source_tree_sha256(repo),
        "checkpoint": _artifact(checkpoint),
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "config": KPGT_CONFIG,
        "raw_dimension": KPGT_DIMENSION,
    }


class _KPGTWorkerBackend:
    """Persistent bridge into the pinned Python-3.7 official KPGT environment."""

    embedding_dimension = KPGT_DIMENSION

    def __init__(
        self,
        *,
        extractor_python: Path,
        worker_script: Path,
        repo: Path,
        checkpoint: Path,
        device: str,
        seed: int,
        provenance: Mapping[str, Any],
        expected_environment_prefix: Path,
        expected_python_version: str,
        expected_torch_version: str,
        expected_dgl_version: str,
        expected_rdkit_version: str,
        startup_timeout: float = 600.0,
        response_timeout: float = 600.0,
    ) -> None:
        self._extractor_python = extractor_python.resolve()
        self._worker_script = worker_script.resolve()
        self._expected_environment_prefix = expected_environment_prefix.resolve()
        if not self._extractor_python.is_file() or not os.access(
            self._extractor_python, os.X_OK
        ):
            raise ExtractionError(
                f"KPGT extractor interpreter is not executable: {self._extractor_python}"
            )
        if not self._worker_script.is_file():
            raise ExtractionError(f"KPGT worker script is missing: {self._worker_script}")
        if not self._expected_environment_prefix.is_dir():
            raise ExtractionError(
                "expected KPGT environment prefix is not a directory: "
                f"{self._expected_environment_prefix}"
            )
        if not _path_is_within(
            self._extractor_python, self._expected_environment_prefix
        ):
            raise ExtractionError(
                "KPGT extractor interpreter is outside the explicitly reviewed "
                f"environment prefix {self._expected_environment_prefix}"
            )
        read_fd, write_fd = os.pipe()
        self._log = tempfile.TemporaryFile(mode="w+b")
        command = [
            str(self._extractor_python),
            str(self._worker_script),
            "--repo",
            str(repo.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--expected-repo-commit",
            str(provenance["expected_repository_commit"]),
            "--expected-checkpoint-sha256",
            str(provenance["expected_checkpoint_sha256"]),
            "--device",
            device,
            "--seed",
            str(seed),
            "--protocol-fd",
            str(write_fd),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=self._log,
                stderr=self._log,
                text=True,
                bufsize=1,
                pass_fds=(write_fd,),
            )
        except OSError as error:
            os.close(read_fd)
            os.close(write_fd)
            self._log.close()
            raise ExtractionError(f"cannot start KPGT worker: {error}") from error
        os.close(write_fd)
        self._protocol = os.fdopen(read_fd, "r", encoding="utf-8")
        self._response_timeout = response_timeout
        self._request_number = 0
        try:
            ready = self._read_response(timeout=startup_timeout)
            self._validate_ready(
                ready,
                provenance=provenance,
                expected_environment_prefix=self._expected_environment_prefix,
                expected_python_version=expected_python_version,
                expected_torch_version=expected_torch_version,
                expected_dgl_version=expected_dgl_version,
                expected_rdkit_version=expected_rdkit_version,
            )
        except Exception:
            self.close(force=True)
            raise
        self.provenance = {
            **dict(provenance),
            "worker": _artifact(self._worker_script),
            "extractor_interpreter": str(self._extractor_python),
            "runtime": ready,
            "expected_runtime": {
                "environment_prefix": str(self._expected_environment_prefix),
                "python_version": expected_python_version,
                "torch": expected_torch_version,
                "dgl": expected_dgl_version,
                "rdkit": expected_rdkit_version,
            },
            "seed": seed,
            "fingerprint": "RDKit RDKFingerprint minPath=1 maxPath=7 fpSize=512",
            "descriptors": "official KPGT RDKit2DNormalized; NaN replaced by zero",
            "graph": "official KPGT smiles_to_graph_tune path_length=5 virtual_nodes=2",
            "readout": "official LiGhTPredictor.generate_fps",
            "environment_isolation": (
                "persistent standalone worker; ASTRA package is not imported by the "
                "Python-3.7 extractor process"
            ),
        }

    def _log_tail(self) -> str:
        try:
            self._log.flush()
            position = self._log.tell()
            self._log.seek(max(0, position - 8192))
            value = self._log.read().decode("utf-8", errors="replace")
            self._log.seek(position)
            return value[-8192:]
        except (OSError, ValueError):
            return ""

    def _read_response(self, *, timeout: float) -> dict[str, Any]:
        ready, _, _ = select.select([self._protocol.fileno()], [], [], timeout)
        if not ready:
            raise ExtractionError(
                f"KPGT worker timed out after {timeout:g}s; log tail: {self._log_tail()}"
            )
        line = self._protocol.readline()
        if not line:
            code = self._process.poll()
            raise ExtractionError(
                f"KPGT worker exited before a response (code={code}); "
                f"log tail: {self._log_tail()}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExtractionError(f"KPGT worker emitted invalid control JSON: {error}") from error
        if not isinstance(payload, dict) or payload.get("schema") != KPGT_WORKER_PROTOCOL_SCHEMA:
            raise ExtractionError("KPGT worker emitted a response with the wrong schema")
        if payload.get("status") in {"error", "startup-error"}:
            raise ExtractionError(f"KPGT worker error: {payload.get('error')}")
        return payload

    def _validate_ready(
        self,
        ready: Mapping[str, Any],
        *,
        provenance: Mapping[str, Any],
        expected_environment_prefix: Path,
        expected_python_version: str,
        expected_torch_version: str,
        expected_dgl_version: str,
        expected_rdkit_version: str,
    ) -> None:
        if ready.get("status") != "ready":
            raise ExtractionError("KPGT worker did not emit a ready handshake")
        packages = ready.get("packages")
        if not isinstance(packages, Mapping):
            raise ExtractionError("KPGT worker handshake lacks dependency versions")
        actual_interpreter_path = Path(str(ready.get("interpreter", ""))).resolve()
        actual_interpreter = str(actual_interpreter_path)
        if not _path_is_within(actual_interpreter_path, expected_environment_prefix):
            raise ExtractionError(
                "KPGT worker interpreter is outside the explicitly reviewed "
                f"environment prefix {expected_environment_prefix}"
            )
        checks = {
            "interpreter": (actual_interpreter, str(self._extractor_python)),
            "python": (
                str(ready.get("python_version", "")),
                expected_python_version,
            ),
            "dependency-python": (
                str(packages.get("python", "")),
                expected_python_version,
            ),
            "torch": (str(packages.get("torch", "")), expected_torch_version),
            "dgl": (str(packages.get("dgl", "")), expected_dgl_version),
            "rdkit": (str(packages.get("rdkit", "")), expected_rdkit_version),
            "repo_commit": (
                str(ready.get("repo_commit", "")),
                str(provenance["expected_repository_commit"]),
            ),
            "checkpoint_sha256": (
                str(ready.get("checkpoint_sha256", "")),
                str(provenance["expected_checkpoint_sha256"]),
            ),
            "worker_sha256": (
                str(ready.get("worker_sha256", "")),
                _sha256(self._worker_script),
            ),
        }
        if type(ready.get("embedding_dimension")) is not int or ready.get(
            "embedding_dimension"
        ) != KPGT_DIMENSION:
            raise ExtractionError(
                "KPGT worker embedding_dimension must be the integer 2304"
            )
        mismatches: list[str] = []
        for label, (observed, expected) in checks.items():
            if observed != expected:
                mismatches.append(f"{label}={observed!r}, expected={expected!r}")
        if mismatches:
            raise ExtractionError(
                "KPGT worker is not the reviewed official environment: "
                + "; ".join(mismatches)
            )

    def embed(
        self, values: Sequence[str], *, row_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        if row_ids is None:
            raise ExtractionError("KPGT worker requires explicit protocol row IDs")
        if len(values) != len(row_ids) or not values:
            raise ExtractionError("KPGT worker values/row IDs are empty or unequal")
        if self._process.poll() is not None:
            raise ExtractionError(
                f"KPGT worker is no longer running; log tail: {self._log_tail()}"
            )
        self._request_number += 1
        request_id = f"request-{self._request_number}"
        with tempfile.TemporaryDirectory(prefix="astra_kpgt_ipc_") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "request.json"
            output_path = temporary_path / "response.npz"
            _atomic_json(
                input_path,
                {
                    "schema": KPGT_WORKER_INPUT_SCHEMA,
                    "items": [
                        {"row_id": str(row_id), "smiles": str(value)}
                        for row_id, value in zip(row_ids, values)
                    ],
                },
            )
            command = {
                "command": "embed",
                "request_id": request_id,
                "input_json": str(input_path),
                "output_npz": str(output_path),
            }
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps(command, sort_keys=True) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise ExtractionError(
                    f"cannot send request to KPGT worker: {error}; "
                    f"log tail: {self._log_tail()}"
                ) from error
            response = self._read_response(timeout=self._response_timeout)
            if response.get("status") != "resolved" or response.get("request_id") != request_id:
                raise ExtractionError("KPGT worker response does not match its request")
            if (
                type(response.get("row_count")) is not int
                or response.get("row_count") != len(row_ids)
                or type(response.get("embedding_dimension")) is not int
                or response.get("embedding_dimension") != KPGT_DIMENSION
            ):
                raise ExtractionError("KPGT worker response has false row/dimension counts")
            if not output_path.is_file() or response.get("output_sha256") != _sha256(
                output_path
            ):
                raise ExtractionError("KPGT worker output is absent or hash-mismatched")
            try:
                with np.load(output_path, allow_pickle=False) as output:
                    returned_ids = [str(value) for value in output["row_ids"].tolist()]
                    matrix = np.asarray(output["embeddings"], dtype=np.float32)
            except (OSError, ValueError, KeyError) as error:
                raise ExtractionError(f"cannot read KPGT worker output: {error}") from error
        if returned_ids != [str(value) for value in row_ids]:
            raise ExtractionError("KPGT worker returned different or reordered row IDs")
        return _validate_backend_output(matrix, len(row_ids), KPGT_DIMENSION)

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.poll() is None and not force and process.stdin is not None:
                self._request_number += 1
                request_id = f"request-{self._request_number}"
                process.stdin.write(
                    json.dumps({"command": "shutdown", "request_id": request_id}) + "\n"
                )
                process.stdin.flush()
                response = self._read_response(timeout=30.0)
                if response.get("status") != "shutdown":
                    force = True
            if process.poll() is None:
                if force:
                    process.terminate()
                process.wait(timeout=30.0)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired, ExtractionError):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10.0)
        finally:
            protocol = getattr(self, "_protocol", None)
            if protocol is not None and not protocol.closed:
                protocol.close()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            log = getattr(self, "_log", None)
            if log is not None and not log.closed:
                log.close()
            self._process = None


def esm_sequence_chunks(
    sequence: str, *, max_residues: int = ESM2_MAX_RESIDUES
) -> tuple[str, ...]:
    """Cover a sequence once with deterministic non-overlapping chunks."""

    if max_residues < 1:
        raise ExtractionError("max_residues must be positive")
    sequence = str(sequence).strip().upper()
    if not sequence:
        raise ExtractionError("cannot chunk an empty sequence")
    return tuple(
        sequence[offset : offset + max_residues]
        for offset in range(0, len(sequence), max_residues)
    )


def weighted_chunk_mean(means: Sequence[np.ndarray], lengths: Sequence[int]) -> np.ndarray:
    """Combine chunk means so each residue has equal weight exactly once."""

    if not means or len(means) != len(lengths) or any(length < 1 for length in lengths):
        raise ExtractionError("chunk means and positive lengths must have equal size")
    matrix = np.asarray(means, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ExtractionError("chunk means must form a finite matrix")
    weights = np.asarray(lengths, dtype=np.float64)
    return np.average(matrix, axis=0, weights=weights).astype(np.float32)


class _ESM2Backend:
    embedding_dimension = ESM2_DIMENSION

    def __init__(
        self,
        *,
        checkpoint: Path,
        sequence_cache_dir: Path,
        device: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> None:
        try:
            torch = importlib.import_module("torch")
            esm = importlib.import_module("esm")
        except (ImportError, OSError) as error:
            raise ExtractionError(f"ESM-2 extraction requires torch and fair-esm: {error}") from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ExtractionError("ESM-2 CUDA was requested but torch reports no CUDA")
        self._torch = torch
        self._device = torch.device(device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        expected_digest = str(provenance.get("expected_checkpoint_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or _sha256(
            checkpoint
        ) != expected_digest:
            raise ExtractionError(
                "ESM-2 backend refuses to deserialize a checkpoint before exact "
                "expected-SHA verification"
            )
        try:
            model, alphabet = _load_verified_fair_esm_checkpoint(
                torch=torch,
                esm=esm,
                checkpoint=checkpoint,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ExtractionError(f"cannot load local ESM-2 checkpoint: {error}") from error
        embed_dim = int(getattr(model, "embed_dim", -1))
        num_layers = int(getattr(model, "num_layers", -1))
        if embed_dim != ESM2_DIMENSION or num_layers != ESM2_LAYER:
            raise ExtractionError(
                "ESM checkpoint is not esm2_t30_150M_UR50D: "
                f"layers={num_layers}, embed_dim={embed_dim}"
            )
        self._model = model.to(self._device).eval()
        self._batch_converter = alphabet.get_batch_converter()
        self._sequence_cache_dir = sequence_cache_dir.resolve()
        self._sequence_cache_dir.mkdir(parents=True, exist_ok=True)
        versions: dict[str, str] = {}
        for package in ("torch", "fair-esm", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        esm_source_root = Path(str(esm.__file__)).resolve().parent
        self.provenance = {
            **dict(provenance),
            "device": str(self._device),
            "seed": seed,
            "dependency_versions": versions,
            "package_version": versions["fair-esm"],
            "source_tree_root": str(esm_source_root),
            "source_tree_sha256": _python_source_tree_sha256(esm_source_root),
            "runtime": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": platform.python_version(),
                "packages": versions,
            },
            "representation_layer": ESM2_LAYER,
            "pooling": "mean over residue states; BOS/EOS/padding excluded",
            "long_sequence_strategy": (
                "non-overlapping contiguous chunks cover every residue once; "
                "chunk means are weighted by residue count"
            ),
            "max_residues_per_chunk": ESM2_MAX_RESIDUES,
            "sequence_set_aggregation": ESM2_SEQUENCE_SET_AGGREGATION,
            "sequence_embedding_cache_schema": ESM2_SEQUENCE_CACHE_SCHEMA,
            "torch_2_6_deserialization": (
                "official fair-esm local loader inside argparse.Namespace safe_globals; "
                "enabled only after expected SHA-256 verification"
            ),
        }

    def _compute_one(self, sequence: str) -> np.ndarray:
        chunks = esm_sequence_chunks(sequence)
        means: list[np.ndarray] = []
        lengths: list[int] = []
        for chunk_index, chunk in enumerate(chunks):
            _, _, tokens = self._batch_converter(
                [(f"chunk-{chunk_index}", chunk)]
            )
            with self._torch.no_grad():
                output = self._model(
                    tokens.to(self._device),
                    repr_layers=[ESM2_LAYER],
                    return_contacts=False,
                )
            states = output["representations"][ESM2_LAYER][0, 1 : len(chunk) + 1]
            if states.shape[0] != len(chunk):
                raise ExtractionError("ESM-2 returned the wrong residue count")
            means.append(np.asarray(states.mean(0).detach().cpu().numpy()))
            lengths.append(len(chunk))
        return weighted_chunk_mean(means, lengths)

    def _sequence_cache_contract(self, sequence: str) -> dict[str, Any]:
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        return {
            "schema": ESM2_SEQUENCE_CACHE_SCHEMA,
            "model_name": ESM2_MODEL_NAME,
            "expected_checkpoint_sha256": self.provenance[
                "expected_checkpoint_sha256"
            ],
            "representation_layer": ESM2_LAYER,
            "raw_dimension": ESM2_DIMENSION,
            "pooling": "mean over residue states; BOS/EOS/padding excluded",
            "long_sequence_strategy": self.provenance["long_sequence_strategy"],
            "max_residues_per_chunk": ESM2_MAX_RESIDUES,
            "sequence_sha256": digest,
            "sequence_length": len(sequence),
        }

    def _embed_one(self, sequence: str) -> np.ndarray:
        sequence = str(sequence).strip().upper()
        if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence):
            raise ExtractionError("ESM-2 sequence contains non-canonical residues")
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        embedding_path = self._sequence_cache_dir / f"{digest}.npz"
        manifest_path = self._sequence_cache_dir / f"{digest}.manifest.json"
        contract = self._sequence_cache_contract(sequence)
        if embedding_path.exists() or manifest_path.exists():
            if not embedding_path.is_file() or not manifest_path.is_file():
                raise ExtractionError(
                    f"incomplete ESM-2 sequence cache pair for SHA {digest}"
                )
            manifest = _read_json(manifest_path)
            for key, value in contract.items():
                if manifest.get(key) != value:
                    raise ExtractionError(
                        f"ESM-2 sequence cache contract mismatch at {key} for {digest}"
                    )
            _verify_artifact_record(
                manifest.get("embedding"),
                label=f"ESM-2 sequence cache {digest}",
                expected_path=embedding_path,
            )
            try:
                with np.load(embedding_path, allow_pickle=False) as payload:
                    cached_digest = str(payload["sequence_sha256"].item())
                    vector = np.asarray(payload["embedding"], dtype=np.float32)
            except (OSError, ValueError, KeyError) as error:
                raise ExtractionError(
                    f"cannot read ESM-2 sequence cache {embedding_path}: {error}"
                ) from error
            if (
                cached_digest != digest
                or vector.shape != (ESM2_DIMENSION,)
                or not np.isfinite(vector).all()
            ):
                raise ExtractionError(f"invalid ESM-2 sequence cache for SHA {digest}")
            return vector
        vector = _validate_backend_output(
            self._compute_one(sequence)[None, :], 1, ESM2_DIMENSION
        )[0]
        _atomic_npz(
            embedding_path,
            sequence_sha256=np.asarray(digest),
            embedding=vector,
        )
        _atomic_json(
            manifest_path,
            {**contract, "created_at": _utc_now(), "embedding": _artifact(embedding_path)},
        )
        return vector

    def embed_sequence_sets(
        self,
        sequence_sets: Sequence[Sequence[str]],
        *,
        row_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        if row_ids is None or len(row_ids) != len(sequence_sets) or not sequence_sets:
            raise ExtractionError("ESM-2 sequence sets require one explicit row ID each")
        entity_vectors: list[np.ndarray] = []
        for row_id, sequences in zip(row_ids, sequence_sets):
            if not sequences:
                raise ExtractionError(f"ESM-2 row {row_id} has an empty sequence set")
            digests = [
                hashlib.sha256(str(sequence).strip().upper().encode("ascii")).hexdigest()
                for sequence in sequences
            ]
            if len(set(digests)) != len(digests):
                raise ExtractionError(
                    f"ESM-2 row {row_id} sequence set contains duplicate sequence SHAs"
                )
            vectors = [self._embed_one(sequence) for sequence in sequences]
            entity_vectors.append(
                np.mean(np.asarray(vectors, dtype=np.float64), axis=0).astype(np.float32)
            )
        return np.stack(entity_vectors).astype(np.float32)

    def embed(
        self, values: Sequence[str], *, row_ids: Sequence[str] | None = None
    ) -> np.ndarray:
        if row_ids is None:
            row_ids = [f"row-{index}" for index in range(len(values))]
        return self.embed_sequence_sets(
            [(value,) for value in values], row_ids=row_ids
        )


def _load_verified_fair_esm_checkpoint(
    *, torch: Any, esm: Any, checkpoint: Path
) -> tuple[Any, Any]:
    """Call fair-esm's loader with the one legacy safe global it requires.

    The caller must verify the checkpoint against an explicitly reviewed
    SHA-256 before reaching this function. PyTorch <2.6 has no ``safe_globals``
    context and retains its historical loader behavior.
    """

    loader = esm.pretrained.load_model_and_alphabet_local
    serialization = getattr(torch, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if callable(safe_globals):
        with safe_globals([argparse.Namespace]):
            return loader(str(checkpoint))
    return loader(str(checkpoint))


def esm2_checkpoint_provenance(
    checkpoint_path: Path | str, *, expected_checkpoint_sha256: str
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ExtractionError(f"ESM-2 checkpoint is absent or empty: {checkpoint}")
    expected_checkpoint_sha256 = str(expected_checkpoint_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256):
        raise ExtractionError("expected ESM-2 checkpoint SHA-256 must have 64 hex digits")
    if _sha256(checkpoint) != expected_checkpoint_sha256:
        raise ExtractionError(
            "ESM-2 checkpoint SHA-256 does not match the reviewed expected hash"
        )
    return {
        "implementation": "fair-esm local model loader",
        "model_name": ESM2_MODEL_NAME,
        "checkpoint": _artifact(checkpoint),
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "raw_dimension": ESM2_DIMENSION,
    }


def _validate_backend_output(
    values: np.ndarray, expected_rows: int, expected_dimension: int
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (expected_rows, expected_dimension):
        raise ExtractionError(
            f"backend returned shape {matrix.shape}, expected "
            f"({expected_rows}, {expected_dimension})"
        )
    if not np.isfinite(matrix).all():
        raise ExtractionError("backend returned NaN or infinity")
    return matrix


def _embed_with_isolation(
    backend: EmbeddingBackend,
    rows: Sequence[CacheRow],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    successes: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}

    def recurse(batch: Sequence[CacheRow]) -> None:
        try:
            sequence_sets = [row.embedding_values for row in batch]
            if any(len(values) != 1 for values in sequence_sets):
                embed_sequence_sets = getattr(backend, "embed_sequence_sets", None)
                if not callable(embed_sequence_sets):
                    raise ExtractionError(
                        "backend cannot aggregate a resolved protein sequence_set"
                    )
                raw_matrix = embed_sequence_sets(
                    sequence_sets,
                    row_ids=[row.row_id for row in batch],
                )
            else:
                raw_matrix = backend.embed(
                    [values[0] for values in sequence_sets],
                    row_ids=[row.row_id for row in batch],
                )
            matrix = _validate_backend_output(
                raw_matrix,
                len(batch),
                backend.embedding_dimension,
            )
        except Exception as error:
            if len(batch) > 1:
                middle = len(batch) // 2
                recurse(batch[:middle])
                recurse(batch[middle:])
                return
            failures[batch[0].row_id] = f"{type(error).__name__}: {error}"
            return
        for row, embedding in zip(batch, matrix):
            successes[row.row_id] = embedding

    for offset in range(0, len(rows), batch_size):
        recurse(rows[offset : offset + batch_size])
    return successes, failures


def _read_shard(path: Path, dimension: int) -> tuple[list[str], list[str]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            row_ids = [str(value) for value in payload["row_ids"].tolist()]
            matrix = np.asarray(payload["embeddings"])
            failed = [str(value) for value in payload["failed_row_ids"].tolist()]
            reasons = [str(value) for value in payload["failure_reasons"].tolist()]
    except (OSError, ValueError, KeyError) as error:
        raise ExtractionError(f"cannot validate embedding shard {path}: {error}") from error
    if matrix.shape != (len(row_ids), dimension) or not np.isfinite(matrix).all():
        raise ExtractionError(f"embedding shard {path} has invalid matrix shape/values")
    if len(failed) != len(reasons) or any(not value for value in reasons):
        raise ExtractionError(f"embedding shard {path} has invalid failure arrays")
    if len(set(row_ids + failed)) != len(row_ids) + len(failed):
        raise ExtractionError(f"embedding shard {path} contains duplicate/overlapping IDs")
    return row_ids, failed


def _verify_artifact_record(
    record: object, *, label: str, expected_path: Path | None = None
) -> Path:
    if not isinstance(record, Mapping):
        raise ExtractionError(f"{label} has no artifact record")
    path = Path(str(record.get("path", ""))).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ExtractionError(f"{label} points to a different path")
    if not path.is_file():
        raise ExtractionError(f"{label} is missing: {path}")
    byte_count = _strict_int(record.get("bytes"), f"{label}.bytes", minimum=0)
    digest = record.get("sha256")
    if (
        byte_count != path.stat().st_size
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or digest != _sha256(path)
    ):
        raise ExtractionError(f"{label} fails its size/SHA-256 contract")
    return path


def _validate_formal_sidecar_contract(
    *,
    contract: Mapping[str, Any],
    shard_path: Path,
    mapping: EntityMapping,
    cache_rows: Sequence[CacheRow],
    cache_sha256: str,
    extractor: str,
    provenance: Mapping[str, Any],
    required_dimension: int,
) -> tuple[int, int, int, int]:
    """Validate one complete shard contract and return range/success/failure."""

    if contract.get("schema") != EMBEDDING_SHARD_SCHEMA:
        raise ExtractionError("embedding shard sidecar has a wrong schema")
    if (
        contract.get("extractor") != extractor
        or contract.get("entity") != mapping.entity
        or contract.get("mapping_sha256") != mapping.sha256
        or contract.get("retrieval_cache_sha256") != cache_sha256
        or contract.get("provenance") != provenance
    ):
        raise ExtractionError("embedding shard identity/provenance contract is false")
    try:
        start = _strict_int(
            contract["row_start"],
            "embedding shard start index",
            minimum=0,
        )
        end = _strict_int(
            contract["row_stop"],
            "embedding shard end index",
            minimum=1,
        )
        row_count = _strict_int(
            contract["row_count"], "embedding shard row_count", minimum=1
        )
        input_dimension = _strict_int(
            contract["input_dimension"],
            "embedding shard input_dimension",
            minimum=1,
        )
        dimension = _strict_int(
            contract["embedding_dimension"],
            "embedding shard dimension",
            minimum=1,
        )
        _strict_int(contract["batch_size"], "embedding shard batch size", minimum=1)
    except KeyError as error:
        raise ExtractionError("embedding shard range/dimension metadata is absent") from error
    if (
        not start < end <= len(mapping.row_ids)
        or row_count != end - start
        or input_dimension != required_dimension
        or dimension != required_dimension
    ):
        raise ExtractionError("embedding shard range or dimension is invalid")
    expected_ids = list(mapping.row_ids[start:end])
    if mapping.entity == "protein":
        raw_sequence_rows = contract.get("sequence_set_rows")
        if not isinstance(raw_sequence_rows, list):
            raise ExtractionError("protein shard lacks sequence_set_rows provenance")
        for row_index, raw_row in enumerate(raw_sequence_rows):
            if not isinstance(raw_row, Mapping):
                raise ExtractionError("protein sequence_set_rows contains a non-object")
            _strict_int(
                raw_row.get("sequence_count"),
                f"protein sequence_set_rows[{row_index}].sequence_count",
                minimum=1,
            )
            raw_sequences = raw_row.get("sequences")
            if not isinstance(raw_sequences, list):
                raise ExtractionError("protein sequence-set sequences are not an array")
            for sequence_index, raw_sequence in enumerate(raw_sequences):
                if not isinstance(raw_sequence, Mapping):
                    raise ExtractionError("protein sequence-set entry is not an object")
                _strict_int(
                    raw_sequence.get("length"),
                    (
                        f"protein sequence_set_rows[{row_index}]"
                        f".sequences[{sequence_index}].length"
                    ),
                    minimum=1,
                )
        if raw_sequence_rows != _sequence_set_contract_rows(cache_rows[start:end]):
            raise ExtractionError("protein sequence_set_rows disagree with retrieval cache")
    _verify_artifact_record(
        contract.get("artifact"), label="embedding shard", expected_path=shard_path
    )
    successful, failed = _read_shard(shard_path, dimension)
    successful_set = set(successful)
    failed_set = set(failed)
    if (
        successful_set.union(failed_set) != set(expected_ids)
        or successful != [value for value in expected_ids if value in successful_set]
        or failed != [value for value in expected_ids if value in failed_set]
    ):
        raise ExtractionError("embedding shard arrays do not exactly cover their range")
    if contract.get("row_ids") != successful or contract.get(
        "failure_row_ids"
    ) != failed:
        raise ExtractionError("embedding shard sidecar row IDs disagree with its NPZ")
    success_count = _strict_int(
        contract.get("success_count"), "embedding shard success_count", minimum=0
    )
    failure_count = _strict_int(
        contract.get("failure_count"), "embedding shard failure_count", minimum=0
    )
    if success_count != len(successful) or failure_count != len(failed):
        raise ExtractionError("embedding shard counts disagree with its arrays")
    return start, end, success_count, failure_count


def validate_formal_extractor_shard(
    *,
    shard_path: Path | str,
    sidecar_path: Path | str,
    extraction_manifest: Path | str,
    mapping_path: Path | str,
    retrieval_cache: Path | str,
) -> dict[str, Any]:
    """Fail-closed validation for a formal KPGT or ESM-2 raw shard."""

    shard_path = Path(shard_path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    extraction_manifest = Path(extraction_manifest).resolve()
    sidecar = _read_json(sidecar_path)
    if sidecar.get("schema") != EMBEDDING_SHARD_SCHEMA:
        raise ExtractionError("embedding shard sidecar has a wrong schema")
    entity = str(sidecar.get("entity", ""))
    if entity not in {"compound", "protein"}:
        raise ExtractionError("formal extractor shard entity must be compound/protein")
    mapping = read_entity_mapping(mapping_path, entity)
    cache_path, cache_rows, cache = _load_cache(retrieval_cache, mapping, entity)
    if cache.get("formal_eligible") is not True:
        raise ExtractionError("formal extraction requires a formal-eligible retrieval cache")
    if sidecar.get("mapping_sha256") != mapping.sha256:
        raise ExtractionError("embedding shard sidecar mapping SHA-256 mismatch")
    cache_sha256 = _sha256(cache_path)
    if sidecar.get("retrieval_cache_sha256") != cache_sha256:
        raise ExtractionError("embedding shard retrieval-cache SHA-256 mismatch")
    extractor = str(sidecar.get("extractor", ""))
    required_dimension = {
        "kpgt-base-generate-fps": KPGT_DIMENSION,
        ESM2_MODEL_NAME: ESM2_DIMENSION,
    }.get(extractor)
    if required_dimension is None:
        raise ExtractionError("formal shard has an unsupported extractor")

    provenance = sidecar.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ExtractionError("embedding shard lacks model provenance")
    raw_dimension = _strict_int(
        provenance.get("raw_dimension"), "extractor raw_dimension", minimum=1
    )
    if raw_dimension != required_dimension:
        raise ExtractionError("extractor raw_dimension is false")
    _validate_formal_sidecar_contract(
        contract=sidecar,
        shard_path=shard_path,
        mapping=mapping,
        cache_rows=cache_rows,
        cache_sha256=cache_sha256,
        extractor=extractor,
        provenance=provenance,
        required_dimension=required_dimension,
    )
    checkpoint_record = provenance.get("checkpoint")
    checkpoint_path = _verify_artifact_record(
        checkpoint_record, label=f"{extractor} checkpoint"
    )
    expected_checkpoint = str(provenance.get("expected_checkpoint_sha256", ""))
    if expected_checkpoint != _sha256(checkpoint_path):
        raise ExtractionError("extractor expected checkpoint hash is inconsistent")
    if extractor == "kpgt-base-generate-fps":
        if entity != "compound" or provenance.get("config") != KPGT_CONFIG:
            raise ExtractionError("KPGT shard has a wrong entity/configuration")
        if provenance.get("implementation") != "official KPGT LiGhT generate_fps":
            raise ExtractionError("KPGT implementation provenance is not official")
        if provenance.get("expected_repository_commit") != OFFICIAL_KPGT_COMMIT:
            raise ExtractionError("KPGT shard does not use the pinned official commit")
        if provenance.get("repository_commit") != OFFICIAL_KPGT_COMMIT:
            raise ExtractionError("KPGT observed commit differs from its pin")
        tree_hash = str(provenance.get("repository_tree_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", tree_hash):
            raise ExtractionError("KPGT tracked source-tree SHA-256 is absent")
        repo = Path(str(provenance.get("repository", ""))).resolve()
        observed = kpgt_source_provenance(
            repo,
            checkpoint_path,
            expected_commit=OFFICIAL_KPGT_COMMIT,
            expected_checkpoint_sha256=expected_checkpoint,
        )
        if observed["repository_tree_sha256"] != tree_hash:
            raise ExtractionError("KPGT source tree changed after extraction")
        worker_path = _verify_artifact_record(
            provenance.get("worker"), label="KPGT worker"
        )
        runtime = provenance.get("runtime")
        expected_runtime = provenance.get("expected_runtime")
        if not isinstance(runtime, Mapping) or not isinstance(expected_runtime, Mapping):
            raise ExtractionError("KPGT worker runtime provenance is absent")
        if set(expected_runtime) != {
            "environment_prefix",
            "python_version",
            "torch",
            "dgl",
            "rdkit",
        }:
            raise ExtractionError("KPGT expected runtime fields are incomplete")
        environment_prefix = Path(
            str(expected_runtime.get("environment_prefix", ""))
        )
        if (
            not environment_prefix.is_absolute()
            or expected_runtime.get("python_version")
            != KPGT_EXPECTED_PYTHON_VERSION
            or expected_runtime.get("torch") != KPGT_EXPECTED_TORCH_VERSION
            or expected_runtime.get("dgl") != KPGT_EXPECTED_DGL_VERSION
            or expected_runtime.get("rdkit") != KPGT_EXPECTED_RDKIT_VERSION
        ):
            raise ExtractionError("KPGT formal runtime pins were overridden")
        packages = runtime.get("packages")
        if not isinstance(packages, Mapping):
            raise ExtractionError("KPGT runtime dependency versions are absent")
        runtime_checks = {
            "torch": KPGT_EXPECTED_TORCH_VERSION,
            "dgl": KPGT_EXPECTED_DGL_VERSION,
            "rdkit": KPGT_EXPECTED_RDKIT_VERSION,
        }
        if (
            runtime.get("schema") != KPGT_WORKER_PROTOCOL_SCHEMA
            or runtime.get("status") != "ready"
            or runtime.get("python_version") != KPGT_EXPECTED_PYTHON_VERSION
            or packages.get("python") != KPGT_EXPECTED_PYTHON_VERSION
            or any(packages.get(key) != value for key, value in runtime_checks.items())
        ):
            raise ExtractionError("KPGT worker did not use the formal pinned dependencies")
        if _strict_int(
            runtime.get("embedding_dimension"),
            "KPGT worker embedding_dimension",
            minimum=1,
        ) != KPGT_DIMENSION:
            raise ExtractionError("KPGT worker embedding dimension is false")
        _strict_int(provenance.get("seed"), "KPGT seed", minimum=0)
        if runtime.get("worker_sha256") != _sha256(worker_path):
            raise ExtractionError("KPGT worker runtime/source hashes disagree")
        interpreter = Path(str(runtime.get("interpreter", ""))).resolve()
        selected_interpreter = Path(
            str(provenance.get("extractor_interpreter", ""))
        ).resolve()
        if (
            not interpreter.is_absolute()
            or not interpreter.is_file()
            or interpreter != selected_interpreter
            or not _path_is_within(interpreter, environment_prefix)
        ):
            raise ExtractionError("KPGT extractor interpreter path is not absolute/current")
        if (
            runtime.get("repo_commit") != OFFICIAL_KPGT_COMMIT
            or runtime.get("checkpoint_sha256") != expected_checkpoint
            or Path(str(runtime.get("repo", ""))).resolve() != repo
            or Path(str(runtime.get("checkpoint", ""))).resolve() != checkpoint_path
        ):
            raise ExtractionError("KPGT worker handshake disagrees with model sources")
    else:
        if entity != "protein":
            raise ExtractionError("ESM-2 shard has a wrong entity")
        if provenance.get("model_name") != ESM2_MODEL_NAME:
            raise ExtractionError("ESM-2 model identity is absent")
        if provenance.get("implementation") != "fair-esm local model loader":
            raise ExtractionError("ESM-2 implementation provenance is absent")
        if _strict_int(
            provenance.get("representation_layer"),
            "ESM-2 representation_layer",
            minimum=1,
        ) != ESM2_LAYER:
            raise ExtractionError("ESM-2 representation layer is wrong")
        if _strict_int(
            provenance.get("max_residues_per_chunk"),
            "ESM-2 max_residues_per_chunk",
            minimum=1,
        ) != ESM2_MAX_RESIDUES:
            raise ExtractionError("ESM-2 long-sequence chunk size is wrong")
        _strict_int(provenance.get("seed"), "ESM-2 seed", minimum=0)
        dependency_versions = provenance.get("dependency_versions")
        runtime = provenance.get("runtime")
        if not isinstance(dependency_versions, Mapping) or not isinstance(
            runtime, Mapping
        ):
            raise ExtractionError("ESM-2 dependency provenance is absent")
        packages = runtime.get("packages")
        package_version = provenance.get("package_version")
        source_tree_hash = provenance.get("source_tree_sha256")
        source_tree_root = Path(str(provenance.get("source_tree_root", "")))
        runtime_interpreter = Path(str(runtime.get("python_executable", "")))
        if (
            not isinstance(packages, Mapping)
            or dict(packages) != dict(dependency_versions)
            or not isinstance(package_version, str)
            or not package_version
            or package_version != packages.get("fair-esm")
            or not isinstance(runtime.get("python_version"), str)
            or not runtime.get("python_version")
            or not runtime_interpreter.is_absolute()
            or not runtime_interpreter.is_file()
            or not source_tree_root.is_absolute()
            or not source_tree_root.is_dir()
            or not isinstance(source_tree_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_tree_hash)
            or _python_source_tree_sha256(source_tree_root) != source_tree_hash
        ):
            raise ExtractionError("ESM-2 runtime/source-tree provenance is false")
        if (
            provenance.get("sequence_set_aggregation")
            != ESM2_SEQUENCE_SET_AGGREGATION
            or provenance.get("sequence_embedding_cache_schema")
            != ESM2_SEQUENCE_CACHE_SCHEMA
        ):
            raise ExtractionError("ESM-2 sequence-set/cache provenance is absent")

    manifest = _read_json(extraction_manifest)
    if manifest.get("schema") != EXTRACTION_MANIFEST_SCHEMA:
        raise ExtractionError("extraction manifest has a wrong schema")
    if (
        manifest.get("extractor") != extractor
        or manifest.get("entity") != entity
        or manifest.get("mapping_sha256") != mapping.sha256
        or manifest.get("provenance") != provenance
    ):
        raise ExtractionError("extraction manifest disagrees with the shard sidecar")
    if manifest.get("retrieval_cache_formal_eligible") is not True:
        raise ExtractionError("extraction manifest is not formally cache-eligible")
    _verify_artifact_record(
        manifest.get("mapping"), label="extraction mapping", expected_path=mapping.path
    )
    _verify_artifact_record(
        manifest.get("retrieval_cache"),
        label="extraction retrieval cache",
        expected_path=cache_path,
    )
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ExtractionError("extraction manifest has no shard entries")
    raw_range = manifest.get("range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ExtractionError("extraction manifest range is malformed")
    manifest_start = _strict_int(
        raw_range[0], "extraction manifest range start", minimum=0
    )
    manifest_end = _strict_int(
        raw_range[1], "extraction manifest range end", minimum=1
    )
    shard_size = _strict_int(
        manifest.get("shard_size"), "extraction manifest shard_size", minimum=1
    )
    manifest_batch_size = _strict_int(
        manifest.get("batch_size"), "extraction manifest batch_size", minimum=1
    )
    if not manifest_start < manifest_end <= len(mapping.row_ids):
        raise ExtractionError("extraction manifest range is invalid")
    matching = []
    success_total = 0
    failure_total = 0
    cursor = manifest_start
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("contract"), Mapping):
            raise ExtractionError("extraction manifest contains a malformed shard entry")
        entry_sidecar = _verify_artifact_record(
            entry.get("sidecar"), label="extraction shard sidecar"
        )
        contract = dict(entry["contract"])
        if _read_json(entry_sidecar) != contract:
            raise ExtractionError("extraction manifest embeds a false sidecar contract")
        entry_shard = _verify_artifact_record(
            contract.get("artifact"), label="extraction entry shard"
        )
        start, end, successes, failures = _validate_formal_sidecar_contract(
            contract=contract,
            shard_path=entry_shard,
            mapping=mapping,
            cache_rows=cache_rows,
            cache_sha256=cache_sha256,
            extractor=extractor,
            provenance=provenance,
            required_dimension=required_dimension,
        )
        if (
            start != cursor
            or end - start > shard_size
            or contract.get("batch_size") != manifest_batch_size
        ):
            raise ExtractionError("extraction manifest shard ranges are not contiguous")
        cursor = end
        success_total += successes
        failure_total += failures
        if entry_sidecar == sidecar_path:
            matching.append(contract)
    if cursor != manifest_end:
        raise ExtractionError("extraction manifest shard ranges do not cover its range")
    if entity == "protein":
        manifest_sequence_rows = manifest.get("sequence_set_rows")
        if not isinstance(manifest_sequence_rows, list):
            raise ExtractionError("protein extraction manifest lacks sequence_set_rows")
        for row_index, raw_row in enumerate(manifest_sequence_rows):
            if not isinstance(raw_row, Mapping):
                raise ExtractionError(
                    "protein extraction sequence_set_rows contains a non-object"
                )
            _strict_int(
                raw_row.get("sequence_count"),
                f"extraction sequence_set_rows[{row_index}].sequence_count",
                minimum=1,
            )
            sequences = raw_row.get("sequences")
            if not isinstance(sequences, list):
                raise ExtractionError("protein extraction sequence list is malformed")
            for sequence_index, sequence in enumerate(sequences):
                if not isinstance(sequence, Mapping):
                    raise ExtractionError("protein extraction sequence is not an object")
                _strict_int(
                    sequence.get("length"),
                    (
                        f"extraction sequence_set_rows[{row_index}]"
                        f".sequences[{sequence_index}].length"
                    ),
                    minimum=1,
                )
        if manifest_sequence_rows != _sequence_set_contract_rows(
            cache_rows[manifest_start:manifest_end]
        ):
            raise ExtractionError(
                "protein extraction sequence_set_rows disagree with retrieval cache"
            )
    if matching != [sidecar]:
        raise ExtractionError("sidecar is absent or duplicated in its extraction manifest")
    manifest_successes = _strict_int(
        manifest.get("success_count"), "extraction manifest success_count", minimum=0
    )
    manifest_failures = _strict_int(
        manifest.get("failure_count"), "extraction manifest failure_count", minimum=0
    )
    if manifest_successes != success_total or manifest_failures != failure_total:
        raise ExtractionError("extraction manifest aggregate counts are false")
    return sidecar


def validate_order_claimed_shard(
    *,
    shard_path: Path | str,
    manifest_path: Path | str,
    mapping_path: Path | str,
) -> dict[str, Any]:
    """Validate manual semantic row identity and its no-op 64-D reducer."""

    shard_path = Path(shard_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != ORDER_CLAIM_IMPORT_SCHEMA:
        raise ExtractionError("order-claim manifest has a wrong schema")
    entity = str(manifest.get("entity", ""))
    if entity not in {"herb", "disease"}:
        raise ExtractionError("order-claim entity must be herb/disease")
    mapping = read_entity_mapping(mapping_path, entity)
    if manifest.get("mapping_sha256") != mapping.sha256:
        raise ExtractionError("order-claim mapping SHA-256 mismatch")
    _verify_artifact_record(
        manifest.get("mapping"), label="order-claim mapping", expected_path=mapping.path
    )
    source_path = _verify_artifact_record(
        manifest.get("source_matrix"), label="order-claim source matrix"
    )
    identity_claim = manifest.get("identity_claim")
    source_provenance = manifest.get("source_provenance")
    if (
        not isinstance(identity_claim, str)
        or not identity_claim.strip()
        or not isinstance(source_provenance, str)
        or not source_provenance.strip()
    ):
        raise ExtractionError("order-claim text/source provenance is empty")
    dimension = _strict_int(
        manifest.get("embedding_dimension"), "order-claim embedding_dimension", minimum=1
    )
    success_count = _strict_int(
        manifest.get("success_count"), "order-claim success_count", minimum=0
    )
    failure_count = _strict_int(
        manifest.get("failure_count"), "order-claim failure_count", minimum=0
    )
    if (
        manifest.get("row_identity_inferred_from_shape") is not False
        or manifest.get("provenance_verified") is not False
        or manifest.get("formal_eligible") is not False
        or manifest.get("semantic_identity_status")
        != "legacy_self_attested_unverified"
        or manifest.get("semantic_features_regenerated") is not False
        or manifest.get("semantic_feature_treatment") != "frozen_existing_64d_condition"
        or manifest.get("reducer_algorithm")
        != "explicit_identity_64_to_64_no_centering_no_rotation"
        or manifest.get("expected_row_ids") != list(mapping.row_ids)
        or dimension != 64
        or success_count != len(mapping.row_ids)
        or failure_count != 0
    ):
        raise ExtractionError("order-claim identity/no-op contract is false")
    _verify_artifact_record(
        manifest.get("shard"), label="order-claim shard", expected_path=shard_path
    )
    reducer_path = _verify_artifact_record(
        manifest.get("identity_reducer_checkpoint"), label="identity reducer"
    )
    successful, failed = _read_shard(shard_path, 64)
    if successful != list(mapping.row_ids) or failed:
        raise ExtractionError("order-claim shard row coverage is not canonical")
    with np.load(shard_path, allow_pickle=False) as shard:
        matrix = np.asarray(shard["embeddings"], dtype=np.float32)
        shard_metadata = str(shard["astra_identity_provenance_json"].item())
    source = np.asarray(np.load(source_path, allow_pickle=False), dtype=np.float32)
    if matrix.shape != source.shape or not np.array_equal(matrix, source):
        raise ExtractionError("order-claim shard values differ from the source matrix")
    with np.load(reducer_path, allow_pickle=False) as reducer:
        mean = np.asarray(reducer["mean"])
        components = np.asarray(reducer["components"])
        singular_values = np.asarray(reducer["singular_values"])
        reducer_metadata = str(reducer["astra_identity_provenance_json"].item())
    expected_metadata = json.dumps(
        {
            "schema": ORDER_CLAIM_IMPORT_SCHEMA,
            "entity": entity,
            "mapping_sha256": mapping.sha256,
            "source_matrix_sha256": _sha256(source_path),
            "identity_claim": manifest["identity_claim"],
            "source_provenance": manifest["source_provenance"],
            "provenance_verified": False,
            "formal_eligible": False,
            "semantic_identity_status": "legacy_self_attested_unverified",
            "semantic_features_regenerated": False,
            "semantic_feature_treatment": "frozen_existing_64d_condition",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if (
        not np.array_equal(mean, np.zeros(64))
        or not np.array_equal(components, np.eye(64))
        or singular_values.size != 0
        or shard_metadata != expected_metadata
        or reducer_metadata != expected_metadata
    ):
        raise ExtractionError("identity reducer or embedded order provenance is false")
    return manifest


def _sequence_set_contract_rows(rows: Sequence[CacheRow]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not row.sequence_set:
            continue
        records.append(
            {
                "row_id": row.row_id,
                "source_id": row.source_id,
                "sequence_set_tier": row.sequence_set_tier,
                "aggregation": row.sequence_set_aggregation,
                "sequence_count": len(row.sequence_set),
                "sequences": [
                    {
                        "sequence_sha256": member.sequence_sha256,
                        "accessions": list(member.accessions),
                        "length": len(member.sequence),
                        "taxid": member.taxid,
                        "annotation_releases": list(member.annotation_releases),
                    }
                    for member in row.sequence_set
                ],
            }
        )
    return records


def _shard_contract(
    *,
    extractor: str,
    mapping: EntityMapping,
    cache_path: Path,
    provenance: Mapping[str, Any],
    start: int,
    end: int,
    batch_size: int,
    cache_rows: Sequence[CacheRow] | None = None,
) -> dict[str, Any]:
    dimension = _strict_int(
        provenance.get("raw_dimension"), "backend raw_dimension", minimum=1
    )
    contract = {
        "schema": EMBEDDING_SHARD_SCHEMA,
        "extractor": extractor,
        "entity": mapping.entity,
        "mapping_sha256": mapping.sha256,
        "retrieval_cache_sha256": _sha256(cache_path),
        "row_start": start,
        "row_stop": end,
        "row_count": end - start,
        "batch_size": batch_size,
        "input_dimension": dimension,
        "embedding_dimension": dimension,
        "provenance": dict(provenance),
    }
    if mapping.entity == "protein":
        if cache_rows is None or len(cache_rows) != end - start:
            raise ExtractionError("protein shard contract lacks its canonical cache rows")
        contract["sequence_set_rows"] = _sequence_set_contract_rows(cache_rows)
    return contract


def extract_embedding_shards(
    *,
    extractor: str,
    entity: str,
    mapping_path: Path | str,
    retrieval_cache: Path | str,
    output_dir: Path | str,
    backend: EmbeddingBackend,
    start_index: int = 0,
    end_index: int | None = None,
    shard_size: int = 512,
    batch_size: int = 16,
) -> Path:
    """Extract verified ID-keyed shards with per-row failure isolation."""

    start_index = _strict_int(start_index, "start_index", minimum=0)
    if end_index is not None:
        end_index = _strict_int(end_index, "end_index", minimum=1)
    shard_size = _strict_int(shard_size, "shard_size", minimum=1)
    batch_size = _strict_int(batch_size, "batch_size", minimum=1)
    mapping = read_entity_mapping(mapping_path, entity)
    cache_path, cache_rows, cache = _load_cache(retrieval_cache, mapping, entity)
    end_index = len(cache_rows) if end_index is None else end_index
    if not 0 <= start_index < end_index <= len(cache_rows):
        raise ExtractionError(
            f"invalid extraction range [{start_index}, {end_index}) for {len(cache_rows)} rows"
        )
    backend_dimension = _strict_int(
        backend.provenance.get("raw_dimension"), "backend raw_dimension", minimum=1
    )
    if type(backend.embedding_dimension) is not int or (
        backend_dimension != backend.embedding_dimension
    ):
        raise ExtractionError("backend dimension disagrees with its provenance")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []

    for start in range(start_index, end_index, shard_size):
        end = min(start + shard_size, end_index)
        selected = cache_rows[start:end]
        stem = f"{entity}_{start:06d}_{end - 1:06d}"
        shard_path = output_dir / f"{stem}.npz"
        sidecar_path = output_dir / f"{stem}.manifest.json"
        contract = _shard_contract(
            extractor=extractor,
            mapping=mapping,
            cache_path=cache_path,
            provenance=backend.provenance,
            start=start,
            end=end,
            batch_size=batch_size,
            cache_rows=selected if entity == "protein" else None,
        )
        if shard_path.exists() or sidecar_path.exists():
            if not shard_path.is_file() or not sidecar_path.is_file():
                raise ExtractionError(f"incomplete existing shard pair for {stem}")
            sidecar = _read_json(sidecar_path)
            for key, value in contract.items():
                if sidecar.get(key) != value:
                    raise ExtractionError(
                        f"existing shard {shard_path} contract mismatch at {key}"
                    )
            artifact = sidecar.get("artifact")
            if (
                not isinstance(artifact, Mapping)
                or Path(str(artifact.get("path", ""))).resolve() != shard_path
                or artifact.get("bytes") != shard_path.stat().st_size
                or artifact.get("sha256") != _sha256(shard_path)
            ):
                raise ExtractionError(f"existing shard {shard_path} fails its recorded hash")
            successful, failed = _read_shard(shard_path, backend.embedding_dimension)
            expected_ids = list(mapping.row_ids[start:end])
            if (
                set(successful).union(failed) != set(expected_ids)
                or successful != [value for value in expected_ids if value in set(successful)]
                or failed != [value for value in expected_ids if value in set(failed)]
            ):
                raise ExtractionError(f"existing shard {shard_path} does not cover its range")
            if (
                sidecar.get("row_ids") != successful
                or sidecar.get("failure_row_ids") != failed
            ):
                raise ExtractionError(
                    f"existing shard {shard_path} sidecar row IDs are false"
                )
            success_count = _strict_int(
                sidecar.get("success_count"),
                "existing shard success_count",
                minimum=0,
            )
            failure_count = _strict_int(
                sidecar.get("failure_count"),
                "existing shard failure_count",
                minimum=0,
            )
            if success_count != len(successful) or failure_count != len(failed):
                raise ExtractionError(f"existing shard {shard_path} has false row counts")
            shard_records.append(
                {"sidecar": _artifact(sidecar_path), "contract": dict(sidecar)}
            )
            continue

        resolved: list[CacheRow] = []
        failures: dict[str, str] = {}
        for row in selected:
            if row.status in {"resolved", "resolved_sequence_set"}:
                resolved.append(row)
            elif row.status == "target_not_protein" and entity == "protein":
                failures[row.row_id] = (
                    "target_not_protein: no sequence embedding was generated; "
                    "the bundle assembler may apply only its declared zero-UNK policy"
                )
            elif row.status == "audited_missing_sequence" and entity == "protein":
                failures[row.row_id] = (
                    "audited_missing_sequence: no sequence embedding was generated; "
                    "the bundle assembler may apply only its declared "
                    "audited-missing-zero policy"
                )
            else:
                failures[row.row_id] = (
                    f"retrieval_cache_status={row.status}: "
                    f"{row.error or 'no resolved value'}"
                )
        embedded, extraction_failures = _embed_with_isolation(
            backend, resolved, batch_size=batch_size
        )
        failures.update(extraction_failures)
        successful_ids = [row.row_id for row in selected if row.row_id in embedded]
        failed_ids = [row.row_id for row in selected if row.row_id in failures]
        if successful_ids + failed_ids != [row.row_id for row in selected]:
            # Both lists must preserve canonical range order, not concatenate by class.
            covered = set(successful_ids) | set(failed_ids)
            expected = {row.row_id for row in selected}
            if covered != expected:
                raise ExtractionError(f"internal shard coverage error for {stem}")
        matrix = (
            np.stack([embedded[row_id] for row_id in successful_ids]).astype(np.float32)
            if successful_ids
            else np.empty((0, backend.embedding_dimension), dtype=np.float32)
        )
        _atomic_npz(
            shard_path,
            row_ids=np.asarray(successful_ids, dtype=str),
            embeddings=matrix,
            failed_row_ids=np.asarray(failed_ids, dtype=str),
            failure_reasons=np.asarray([failures[row_id] for row_id in failed_ids], dtype=str),
        )
        sidecar = {
            **contract,
            "created_at": _utc_now(),
            "success_count": len(successful_ids),
            "failure_count": len(failed_ids),
            "row_ids": successful_ids,
            "failure_row_ids": failed_ids,
            "artifact": _artifact(shard_path),
        }
        _atomic_json(sidecar_path, sidecar)
        shard_records.append(
            {"sidecar": _artifact(sidecar_path), "contract": dict(sidecar)}
        )

    manifest = {
        "schema": EXTRACTION_MANIFEST_SCHEMA,
        "extractor": extractor,
        "entity": entity,
        "mapping": _artifact(mapping.path),
        "mapping_sha256": mapping.sha256,
        "retrieval_cache": _artifact(cache_path),
        "retrieval_cache_formal_eligible": cache.get("formal_eligible") is True,
        "range": [start_index, end_index],
        "shard_size": shard_size,
        "batch_size": batch_size,
        "provenance": dict(backend.provenance),
        "shards": shard_records,
        "success_count": sum(
            record["contract"]["success_count"] for record in shard_records
        ),
        "failure_count": sum(
            record["contract"]["failure_count"] for record in shard_records
        ),
    }
    if entity == "protein":
        manifest["sequence_set_rows"] = [
            sequence_row
            for record in shard_records
            for sequence_row in record["contract"]["sequence_set_rows"]
        ]
    manifest_path = output_dir / "extraction_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def import_order_claimed_matrix(
    *,
    entity: str,
    mapping_path: Path | str,
    matrix_path: Path | str,
    output_path: Path | str,
    identity_claim: str,
    source_provenance: str,
) -> Path:
    """Package a legacy mapping-order claim as a formal-ineligible diagnostic.

    Shape is checked only for corruption detection.  The caller's explicit
    identity claim is the provenance basis and is retained verbatim; this
    function never infers row identity from shape.
    """

    entity = str(entity).strip().lower()
    if entity not in {"herb", "disease"}:
        raise ExtractionError("order-claimed imports are restricted to herb/disease")
    identity_claim = str(identity_claim).strip()
    source_provenance = str(source_provenance).strip()
    if not identity_claim:
        raise ExtractionError("--identity-claim must be explicit and non-empty")
    if not source_provenance:
        raise ExtractionError("--source-provenance must be explicit and non-empty")
    mapping = read_entity_mapping(mapping_path, entity)
    matrix_path = Path(matrix_path).resolve()
    if matrix_path.suffix.lower() != ".npy":
        raise ExtractionError("order-claimed matrix import accepts only .npy input")
    try:
        matrix = np.load(matrix_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ExtractionError(f"cannot read claimed matrix {matrix_path}: {error}") from error
    if not np.issubdtype(matrix.dtype, np.number):
        raise ExtractionError("claimed matrix must be numeric")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (len(mapping.row_ids), 64):
        raise ExtractionError(
            f"claimed matrix shape {matrix.shape} != ({len(mapping.row_ids)}, 64)"
        )
    if not np.isfinite(matrix).all():
        raise ExtractionError("claimed matrix contains NaN or infinity")
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ExtractionError("order-claimed shard output must end in .npz")
    manifest_path = output_path.with_suffix(".manifest.json")
    reducer_path = output_path.with_name(f"{output_path.stem}.identity_reducer.npz")
    embedded_provenance = {
        "schema": ORDER_CLAIM_IMPORT_SCHEMA,
        "entity": entity,
        "mapping_sha256": mapping.sha256,
        "source_matrix_sha256": _sha256(matrix_path),
        "identity_claim": identity_claim,
        "source_provenance": source_provenance,
        "provenance_verified": False,
        "formal_eligible": False,
        "semantic_identity_status": "legacy_self_attested_unverified",
        "semantic_features_regenerated": False,
        "semantic_feature_treatment": "frozen_existing_64d_condition",
    }
    embedded_provenance_json = json.dumps(
        embedded_provenance, sort_keys=True, separators=(",", ":")
    )
    contract = {
        "schema": ORDER_CLAIM_IMPORT_SCHEMA,
        "extractor": "manual-order-claim-import",
        "entity": entity,
        "mapping": _artifact(mapping.path),
        "mapping_sha256": mapping.sha256,
        "source_matrix": _artifact(matrix_path),
        "identity_claim": identity_claim,
        "source_provenance": source_provenance,
        "row_identity_inferred_from_shape": False,
        "provenance_verified": False,
        "formal_eligible": False,
        "semantic_identity_status": "legacy_self_attested_unverified",
        "semantic_features_regenerated": False,
        "semantic_feature_treatment": "frozen_existing_64d_condition",
        "reducer_algorithm": "explicit_identity_64_to_64_no_centering_no_rotation",
        "expected_row_ids": list(mapping.row_ids),
        "embedding_dimension": 64,
    }
    if output_path.exists() or manifest_path.exists() or reducer_path.exists():
        if (
            not output_path.is_file()
            or not manifest_path.is_file()
            or not reducer_path.is_file()
        ):
            raise ExtractionError("existing order-claimed import has an incomplete file pair")
        prior = _read_json(manifest_path)
        for key, value in contract.items():
            if prior.get(key) != value:
                raise ExtractionError(f"existing order-claimed import differs at {key}")
        artifact = prior.get("shard")
        if not isinstance(artifact, Mapping) or artifact.get("sha256") != _sha256(output_path):
            raise ExtractionError("existing order-claimed shard fails its recorded hash")
        reducer_artifact = prior.get("identity_reducer_checkpoint")
        if (
            not isinstance(reducer_artifact, Mapping)
            or reducer_artifact.get("sha256") != _sha256(reducer_path)
        ):
            raise ExtractionError("existing identity reducer fails its recorded hash")
        try:
            with np.load(reducer_path, allow_pickle=False) as reducer:
                mean = np.asarray(reducer["mean"])
                components = np.asarray(reducer["components"])
                reducer_provenance = str(
                    reducer["astra_identity_provenance_json"].item()
                )
        except (OSError, ValueError, KeyError) as error:
            raise ExtractionError(f"cannot validate identity reducer: {error}") from error
        if not np.array_equal(mean, np.zeros(64)) or not np.array_equal(
            components, np.eye(64)
        ):
            raise ExtractionError("existing reducer is not an exact 64-D identity")
        if reducer_provenance != embedded_provenance_json:
            raise ExtractionError("existing identity reducer provenance has changed")
        successful, failed = _read_shard(output_path, 64)
        if successful != list(mapping.row_ids) or failed:
            raise ExtractionError("existing order-claimed shard has invalid row coverage")
        return output_path
    _atomic_npz(
        reducer_path,
        mean=np.zeros(64, dtype=np.float64),
        components=np.eye(64, dtype=np.float64),
        singular_values=np.asarray([], dtype=np.float64),
        astra_identity_provenance_json=np.asarray(embedded_provenance_json),
    )
    _atomic_npz(
        output_path,
        row_ids=np.asarray(mapping.row_ids, dtype=str),
        embeddings=matrix,
        failed_row_ids=np.asarray([], dtype=str),
        failure_reasons=np.asarray([], dtype=str),
        astra_identity_provenance_json=np.asarray(embedded_provenance_json),
    )
    manifest = {
        **contract,
        "created_at": _utc_now(),
        "success_count": len(mapping.row_ids),
        "failure_count": 0,
        "shard": _artifact(output_path),
        "identity_reducer_checkpoint": _artifact(reducer_path),
    }
    _atomic_json(manifest_path, manifest)
    return output_path


def _default_kpgt_worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "kpgt_official_worker.py"


def extract_kpgt_shards(
    *,
    mapping_path: Path | str,
    retrieval_cache: Path | str,
    repo_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    extractor_python: Path | str,
    expected_checkpoint_sha256: str,
    worker_script: Path | str | None = None,
    expected_commit: str = OFFICIAL_KPGT_COMMIT,
    expected_environment_prefix: Path | str | None = None,
    expected_python_version: str = KPGT_EXPECTED_PYTHON_VERSION,
    expected_torch_version: str = KPGT_EXPECTED_TORCH_VERSION,
    expected_dgl_version: str = KPGT_EXPECTED_DGL_VERSION,
    expected_rdkit_version: str = KPGT_EXPECTED_RDKIT_VERSION,
    device: str = "auto",
    seed: int = 42,
    start_index: int = 0,
    end_index: int | None = None,
    shard_size: int = 512,
    batch_size: int = 16,
) -> Path:
    extractor_python = Path(extractor_python).resolve()
    environment_prefix = (
        Path(expected_environment_prefix).resolve()
        if expected_environment_prefix is not None
        else extractor_python.parent.parent.resolve()
    )
    provenance = kpgt_source_provenance(
        repo_path,
        checkpoint_path,
        expected_commit=expected_commit,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    backend = _KPGTWorkerBackend(
        extractor_python=extractor_python,
        worker_script=(
            Path(worker_script) if worker_script is not None else _default_kpgt_worker_script()
        ),
        repo=Path(repo_path).resolve(),
        checkpoint=Path(checkpoint_path).resolve(),
        device=device,
        seed=seed,
        provenance=provenance,
        expected_environment_prefix=environment_prefix,
        expected_python_version=expected_python_version,
        expected_torch_version=expected_torch_version,
        expected_dgl_version=expected_dgl_version,
        expected_rdkit_version=expected_rdkit_version,
    )
    try:
        return extract_embedding_shards(
            extractor="kpgt-base-generate-fps",
            entity="compound",
            mapping_path=mapping_path,
            retrieval_cache=retrieval_cache,
            output_dir=output_dir,
            backend=backend,
            start_index=start_index,
            end_index=end_index,
            shard_size=shard_size,
            batch_size=batch_size,
        )
    finally:
        backend.close()


def smoke_kpgt_worker(
    *,
    repo_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    extractor_python: Path | str,
    expected_checkpoint_sha256: str,
    smiles: str = "CCO",
    worker_script: Path | str | None = None,
    expected_commit: str = OFFICIAL_KPGT_COMMIT,
    expected_environment_prefix: Path | str | None = None,
    expected_python_version: str = KPGT_EXPECTED_PYTHON_VERSION,
    expected_torch_version: str = KPGT_EXPECTED_TORCH_VERSION,
    expected_dgl_version: str = KPGT_EXPECTED_DGL_VERSION,
    expected_rdkit_version: str = KPGT_EXPECTED_RDKIT_VERSION,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """Run one real molecule through official graph, descriptor and model code."""

    smiles = str(smiles).strip()
    if not smiles:
        raise ExtractionError("KPGT smoke SMILES must be non-empty")
    extractor_python = Path(extractor_python).resolve()
    environment_prefix = (
        Path(expected_environment_prefix).resolve()
        if expected_environment_prefix is not None
        else extractor_python.parent.parent.resolve()
    )
    provenance = kpgt_source_provenance(
        repo_path,
        checkpoint_path,
        expected_commit=expected_commit,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    backend = _KPGTWorkerBackend(
        extractor_python=extractor_python,
        worker_script=(
            Path(worker_script) if worker_script is not None else _default_kpgt_worker_script()
        ),
        repo=Path(repo_path).resolve(),
        checkpoint=Path(checkpoint_path).resolve(),
        device=device,
        seed=seed,
        provenance=provenance,
        expected_environment_prefix=environment_prefix,
        expected_python_version=expected_python_version,
        expected_torch_version=expected_torch_version,
        expected_dgl_version=expected_dgl_version,
        expected_rdkit_version=expected_rdkit_version,
    )
    try:
        matrix = backend.embed([smiles], row_ids=["smoke"])
        runtime_provenance = dict(backend.provenance)
    finally:
        backend.close()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kpgt_single_molecule_smoke.npz"
    manifest_path = output_dir / "kpgt_single_molecule_smoke.json"
    if output_path.exists() or manifest_path.exists():
        raise ExtractionError(
            "KPGT smoke output already exists; choose a new output directory"
        )
    _atomic_npz(
        output_path,
        row_ids=np.asarray(["smoke"], dtype=str),
        embeddings=matrix,
        failed_row_ids=np.asarray([], dtype=str),
        failure_reasons=np.asarray([], dtype=str),
    )
    _atomic_json(
        manifest_path,
        {
            "schema": "astra-kpgt-official-worker-smoke/v1",
            "diagnostic_only": True,
            "smiles": smiles,
            "shape": list(matrix.shape),
            "finite": bool(np.isfinite(matrix).all()),
            "nonzero": bool(np.count_nonzero(matrix)),
            "output": _artifact(output_path),
            "provenance": runtime_provenance,
        },
    )
    return manifest_path


def extract_esm2_shards(
    *,
    mapping_path: Path | str,
    retrieval_cache: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    expected_checkpoint_sha256: str,
    device: str = "auto",
    seed: int = 42,
    start_index: int = 0,
    end_index: int | None = None,
    shard_size: int = 128,
    batch_size: int = 4,
) -> Path:
    output_dir = Path(output_dir).resolve()
    provenance = esm2_checkpoint_provenance(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    backend = _ESM2Backend(
        checkpoint=Path(checkpoint_path).resolve(),
        sequence_cache_dir=output_dir / "sequence_embedding_cache",
        device=device,
        seed=seed,
        provenance=provenance,
    )
    return extract_embedding_shards(
        extractor=ESM2_MODEL_NAME,
        entity="protein",
        mapping_path=mapping_path,
        retrieval_cache=retrieval_cache,
        output_dir=output_dir,
        backend=backend,
        start_index=start_index,
        end_index=end_index,
        shard_size=shard_size,
        batch_size=batch_size,
    )


def _add_range_arguments(parser: argparse.ArgumentParser, *, shard_size: int, batch_size: int) -> None:
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--shard-size", type=int, default=shard_size)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)


def _add_kpgt_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extractor-python",
        type=Path,
        required=True,
        help="Python executable from the reviewed official KPGT environment.",
    )
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=_default_kpgt_worker_script(),
        help="Standalone Python-3.7-compatible worker (hash recorded).",
    )
    parser.add_argument(
        "--expected-extractor-prefix",
        type=Path,
        help=(
            "Reviewed environment prefix containing --extractor-python; defaults "
            "to the interpreter's bin/.. directory."
        ),
    )
    parser.add_argument(
        "--expected-python-version", default=KPGT_EXPECTED_PYTHON_VERSION
    )
    parser.add_argument(
        "--expected-torch-version", default=KPGT_EXPECTED_TORCH_VERSION
    )
    parser.add_argument(
        "--expected-dgl-version", default=KPGT_EXPECTED_DGL_VERSION
    )
    parser.add_argument(
        "--expected-rdkit-version", default=KPGT_EXPECTED_RDKIT_VERSION
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pubchem = subparsers.add_parser(
        "fetch-pubchem", help="fetch CID-keyed PubChem SMILES import JSON"
    )
    pubchem.add_argument("--mapping", type=Path, required=True)
    pubchem.add_argument("--output", type=Path, required=True)
    pubchem.add_argument("--batch-size", type=int, default=100)
    pubchem.add_argument("--timeout", type=float, default=60.0)
    pubchem.add_argument("--max-call-attempts", type=int, default=2)
    pubchem.add_argument("--endpoint", default=PUBCHEM_ENDPOINT)
    pubchem.add_argument(
        "--user-agent", default="ASTRA-Scheme-B/1.0 (PubChem feature provenance)"
    )

    kpgt = subparsers.add_parser(
        "extract-kpgt", help="extract official pinned KPGT compound shards"
    )
    _add_range_arguments(kpgt, shard_size=512, batch_size=16)
    kpgt.add_argument("--repo", type=Path, required=True)
    kpgt.add_argument("--checkpoint", type=Path, required=True)
    kpgt.add_argument("--expected-checkpoint-sha256", required=True)
    kpgt.add_argument("--expected-repo-commit", default=OFFICIAL_KPGT_COMMIT)
    _add_kpgt_runtime_arguments(kpgt)

    kpgt_smoke = subparsers.add_parser(
        "smoke-kpgt",
        help="run one molecule through the isolated official KPGT worker",
    )
    kpgt_smoke.add_argument("--repo", type=Path, required=True)
    kpgt_smoke.add_argument("--checkpoint", type=Path, required=True)
    kpgt_smoke.add_argument("--expected-checkpoint-sha256", required=True)
    kpgt_smoke.add_argument("--expected-repo-commit", default=OFFICIAL_KPGT_COMMIT)
    kpgt_smoke.add_argument("--output-dir", type=Path, required=True)
    kpgt_smoke.add_argument("--smiles", default="CCO")
    kpgt_smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    kpgt_smoke.add_argument("--seed", type=int, default=42)
    _add_kpgt_runtime_arguments(kpgt_smoke)

    esm2 = subparsers.add_parser(
        "extract-esm2", help="extract local ESM-2 t30 150M protein shards"
    )
    _add_range_arguments(esm2, shard_size=128, batch_size=4)
    esm2.add_argument("--checkpoint", type=Path, required=True)
    esm2.add_argument("--expected-checkpoint-sha256", required=True)

    claimed = subparsers.add_parser(
        "import-matrix",
        help=(
            "package a legacy mapping-order-claimed herb/disease 64-D .npy "
            "as a formal-ineligible diagnostic"
        ),
    )
    claimed.add_argument("--entity", choices=("herb", "disease"), required=True)
    claimed.add_argument("--mapping", type=Path, required=True)
    claimed.add_argument("--matrix", type=Path, required=True)
    claimed.add_argument("--output", type=Path, required=True)
    claimed.add_argument("--identity-claim", required=True)
    claimed.add_argument("--source-provenance", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fetch-pubchem":
            result: Any = fetch_pubchem_smiles(
                mapping_path=args.mapping,
                output_path=args.output,
                batch_size=args.batch_size,
                timeout=args.timeout,
                max_call_attempts=args.max_call_attempts,
                endpoint=args.endpoint,
                user_agent=args.user_agent,
            )
            print(json.dumps(result["metadata"], indent=2, sort_keys=True))
            return 0 if not result["failures"] else 2
        if args.command == "import-matrix":
            result = import_order_claimed_matrix(
                entity=args.entity,
                mapping_path=args.mapping,
                matrix_path=args.matrix,
                output_path=args.output,
                identity_claim=args.identity_claim,
                source_provenance=args.source_provenance,
            )
        elif args.command == "extract-kpgt":
            result = extract_kpgt_shards(
                mapping_path=args.mapping,
                retrieval_cache=args.retrieval_cache,
                repo_path=args.repo,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                extractor_python=args.extractor_python,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                worker_script=args.worker_script,
                expected_commit=args.expected_repo_commit,
                expected_environment_prefix=args.expected_extractor_prefix,
                expected_python_version=args.expected_python_version,
                expected_torch_version=args.expected_torch_version,
                expected_dgl_version=args.expected_dgl_version,
                expected_rdkit_version=args.expected_rdkit_version,
                device=args.device,
                seed=args.seed,
                start_index=args.start_index,
                end_index=args.end_index,
                shard_size=args.shard_size,
                batch_size=args.batch_size,
            )
        elif args.command == "smoke-kpgt":
            result = smoke_kpgt_worker(
                repo_path=args.repo,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                extractor_python=args.extractor_python,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                smiles=args.smiles,
                worker_script=args.worker_script,
                expected_commit=args.expected_repo_commit,
                expected_environment_prefix=args.expected_extractor_prefix,
                expected_python_version=args.expected_python_version,
                expected_torch_version=args.expected_torch_version,
                expected_dgl_version=args.expected_dgl_version,
                expected_rdkit_version=args.expected_rdkit_version,
                device=args.device,
                seed=args.seed,
            )
        else:
            result = extract_esm2_shards(
                mapping_path=args.mapping,
                retrieval_cache=args.retrieval_cache,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                device=args.device,
                seed=args.seed,
                start_index=args.start_index,
                end_index=args.end_index,
                shard_size=args.shard_size,
                batch_size=args.batch_size,
            )
    except (ExtractionError, OSError, ValueError) as error:
        print(f"Ethnobotany feature extraction failed: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
