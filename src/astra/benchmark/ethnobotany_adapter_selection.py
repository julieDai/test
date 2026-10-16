#!/usr/bin/env python3
"""Validation-only selection for the Ethnobotany S5 semantic adapters.

This module deliberately has no outer-fold evaluation function or command.  A
pre-training plan fixes candidate paths, configurations, feature provenance,
code provenance, and the selection rule.  ``freeze-plan`` verifies that every
new candidate manifest is absent before writing the plan digest sidecar.
``select`` later audits validation artifacts, recomputes their scores and
metrics from saved embeddings, and writes ``development_selection.json``.

The historical S4a control was produced by an older code snapshot.  Its code
contract is therefore verified through its frozen controlled-plan provenance;
new S5 candidates use direct source-file hashes.  This distinction prevents a
historical run from being falsely attributed to the current source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import (
    BenchmarkAuditError,
    PREDICTION_FIELDS,
    evaluate_validation_scores,
    load_predictions,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest


PLAN_SCHEMA = "astra-ethnobotany-adapter-development-plan/v1"
OUTPUT_SCHEMA = "astra-ethnobotany-adapter-development-selection/v1"
CANDIDATE_SCHEMA = "astra-ethnobotany-refined-candidate/v1"
CONFIG_SCHEMA = "astra-ethnobotany-refined-candidate-config/v1"
FROZEN_SELECTION_SCHEMA = "astra-ethnobotany-refined-selection/v1"
DATASET = "ethnobotany"
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
FEATURE_ENTITIES = ("compound", "protein", "herb", "disease")
S5_ADAPTERS = frozenset({"residual_linear", "gated_residual", "residual_mlp"})
METRIC = "Validation_AUPRC"
METRIC_ATOL = 1e-12
SCORE_ATOL = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The requested historical control is not an arbitrary S4a run.  These are
# the five manifests committed by the old frozen nested selection.
BASELINE_CONFIG_BY_FOLD = {
    1: "s4a_linear64_root",
    2: "s4a_linear64_root",
    3: "s4a_linear64_lr3e4",
    4: "s4a_linear64_lr3e4",
    5: "s4a_linear64_balanced_root",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ARTIFACT = re.compile(
    r"(^|[_.-])(outer(?:[_-]?test)?|test)([_.-]|$)", re.IGNORECASE
)


class AdapterSelectionError(RuntimeError):
    """Raised when the S5 development-only contract cannot be proven."""


@dataclass(frozen=True)
class ValidationRows:
    train_row_indices: np.ndarray
    validation_row_indices: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray


@dataclass(frozen=True)
class PlanContext:
    path: Path
    digest: str
    sidecar: Path | None
    payload: Mapping[str, Any]
    protocol_path: Path
    output_path: Path
    groups: Mapping[str, Mapping[int, Mapping[str, Any]]]
    baseline_frozen: Mapping[int, Mapping[str, Any]]


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterSelectionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdapterSelectionError(f"{label} must contain one JSON object: {path}")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise AdapterSelectionError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _resolve_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AdapterSelectionError(f"{label} path must be a non-empty string")
    supplied = Path(value)
    return (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()


def _validate_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise AdapterSelectionError(f"{label} sha256 must be 64 lowercase hex digits")
    return value


def _validate_artifact(
    record: object,
    *,
    label: str,
    exact: bool = True,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise AdapterSelectionError(f"{label} must be an artifact object")
    if exact:
        _exact_keys(record, {"path", "sha256"}, label=label)
    elif not {"path", "sha256"}.issubset(record):
        raise AdapterSelectionError(f"{label} must contain path and sha256")
    path = _resolve_path(record.get("path"), label=label)
    expected = _validate_digest(record.get("sha256"), label=label)
    if not path.is_file():
        raise AdapterSelectionError(f"{label} does not exist: {path}")
    observed = sha256(path)
    if observed != expected:
        raise AdapterSelectionError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return path, expected


def _artifact_identity(record: Mapping[str, Any], *, label: str) -> tuple[Path, str]:
    if not {"path", "sha256"}.issubset(record):
        raise AdapterSelectionError(f"{label} lacks path or sha256")
    return (
        _resolve_path(record["path"], label=label),
        _validate_digest(record["sha256"], label=label),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sidecar_path(plan_path: Path) -> Path:
    return Path(str(plan_path) + ".sha256")


def _read_plan_sidecar(plan_path: Path) -> tuple[Path, str]:
    sidecar = _sidecar_path(plan_path)
    if not sidecar.is_file():
        raise AdapterSelectionError(
            f"frozen plan sidecar is absent; run freeze-plan first: {sidecar}"
        )
    try:
        value = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise AdapterSelectionError(f"cannot read plan sidecar {sidecar}: {error}") from error
    expected = _validate_digest(value, label="plan sidecar")
    observed = sha256(plan_path)
    if observed != expected:
        raise AdapterSelectionError(
            f"frozen plan was modified: expected {expected}, observed {observed}"
        )
    if plan_path.stat().st_mtime_ns > sidecar.stat().st_mtime_ns:
        raise AdapterSelectionError("plan modification time is newer than its freeze sidecar")
    return sidecar, expected


def _validate_code_contracts(
    raw: object,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise AdapterSelectionError("code_contracts must be a non-empty object")
    output: dict[str, dict[str, Any]] = {}
    for contract_id, value in raw.items():
        if not isinstance(contract_id, str) or not contract_id:
            raise AdapterSelectionError("code contract ids must be non-empty strings")
        if not isinstance(value, Mapping):
            raise AdapterSelectionError(f"code contract {contract_id} must be an object")
        kind = value.get("kind")
        if kind == "files":
            roles = ("model", "runner", "parent_model", "parent_runner", "selector")
            _exact_keys(
                value,
                {"kind", *roles},
                label=f"code contract {contract_id}",
            )
            normalized: dict[str, Any] = {"kind": kind}
            for role in roles:
                source_path, source_hash = _validate_artifact(
                    value[role], label=f"code contract {contract_id} {role}"
                )
                if role == "selector" and source_path != Path(__file__).resolve():
                    raise AdapterSelectionError(
                        f"code contract {contract_id} selector must bind this module"
                    )
                normalized[role] = {
                    "path": str(source_path),
                    "sha256": source_hash,
                }
            output[contract_id] = normalized
        elif kind == "frozen_plan":
            _exact_keys(
                value,
                {"kind", "source_plan", "model", "runner"},
                label=f"code contract {contract_id}",
            )
            source_path, source_hash = _validate_artifact(
                value["source_plan"], label=f"code contract {contract_id} source_plan"
            )
            source = _load_json(source_path, label="frozen source plan")
            try:
                execution = source["selection_freeze"]["execution_code"]
            except (KeyError, TypeError) as error:
                raise AdapterSelectionError(
                    f"frozen source plan has no execution_code: {source_path}"
                ) from error
            for role in ("model", "runner"):
                declared = value[role]
                if not isinstance(declared, Mapping):
                    raise AdapterSelectionError(
                        f"code contract {contract_id} {role} must be an object"
                    )
                _exact_keys(
                    declared,
                    {"path", "sha256"},
                    label=f"code contract {contract_id} {role}",
                )
                recorded = execution.get(role)
                if not isinstance(recorded, Mapping):
                    raise AdapterSelectionError(
                        f"frozen source plan has no {role} code record"
                    )
                declared_path, declared_hash = _artifact_identity(
                    declared, label=f"code contract {contract_id} {role}"
                )
                recorded_path, recorded_hash = _artifact_identity(
                    recorded, label=f"frozen source plan {role}"
                )
                if declared_path != recorded_path or declared_hash != recorded_hash:
                    raise AdapterSelectionError(
                        f"code contract {contract_id} {role} differs from frozen source plan"
                    )
            output[contract_id] = {
                "kind": kind,
                "source_plan": {"path": str(source_path), "sha256": source_hash},
                "model": dict(value["model"]),
                "runner": dict(value["runner"]),
            }
        else:
            raise AdapterSelectionError(
                f"code contract {contract_id} kind must be files or frozen_plan"
            )
    return output


def _validate_feature_contract(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AdapterSelectionError("features must be an object")
    _exact_keys(
        raw,
        {"selected_features", "semantic_reduction", "deep_audit_receipt"},
        label="features",
    )
    selected = raw["selected_features"]
    if not isinstance(selected, Mapping) or set(selected) != set(FEATURE_ENTITIES):
        raise AdapterSelectionError(
            f"selected_features must contain exactly {list(FEATURE_ENTITIES)}"
        )
    normalized: dict[str, Any] = {"selected_features": {}}
    for entity in FEATURE_ENTITIES:
        path, digest = _validate_artifact(
            selected[entity], label=f"selected feature {entity}"
        )
        normalized["selected_features"][entity] = {
            "path": str(path),
            "sha256": digest,
        }
    for key in ("semantic_reduction", "deep_audit_receipt"):
        path, digest = _validate_artifact(raw[key], label=key)
        normalized[key] = {"path": str(path), "sha256": digest}
    return normalized


def _validate_selection_rule(raw: object, architecture_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AdapterSelectionError("selection must be an object")
    _exact_keys(
        raw,
        {
            "folds",
            "metric",
            "candidate_epochs",
            "epoch_tie_break",
            "architecture_tie_order",
            "minimum_mean_improvement",
            "minimum_improved_folds",
        },
        label="selection",
    )
    if raw["folds"] != list(EXPECTED_FOLDS):
        raise AdapterSelectionError(f"selection.folds must equal {list(EXPECTED_FOLDS)}")
    if raw["metric"] != METRIC:
        raise AdapterSelectionError(f"selection.metric must be {METRIC}")
    epochs = raw["candidate_epochs"]
    if (
        not isinstance(epochs, list)
        or not epochs
        or any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0 for epoch in epochs)
        or epochs != sorted(set(epochs))
    ):
        raise AdapterSelectionError(
            "selection.candidate_epochs must be sorted unique positive integers"
        )
    if raw["epoch_tie_break"] != "lowest_epoch":
        raise AdapterSelectionError("epoch_tie_break must be lowest_epoch")
    tie_order = raw["architecture_tie_order"]
    if (
        not isinstance(tie_order, list)
        or any(not isinstance(value, str) or not value for value in tie_order)
        or len(tie_order) != len(set(tie_order))
        or set(tie_order) != set(architecture_ids)
    ):
        raise AdapterSelectionError(
            "architecture_tie_order must contain every new architecture exactly once"
        )
    improvement = raw["minimum_mean_improvement"]
    if isinstance(improvement, bool) or not isinstance(improvement, (int, float)):
        raise AdapterSelectionError("minimum_mean_improvement must be numeric")
    if not math.isclose(float(improvement), 0.001, rel_tol=0.0, abs_tol=0.0):
        raise AdapterSelectionError("minimum_mean_improvement must equal 0.001")
    if raw["minimum_improved_folds"] != 3:
        raise AdapterSelectionError("minimum_improved_folds must equal 3")
    return {
        **dict(raw),
        "candidate_epochs": list(epochs),
        "architecture_tie_order": list(tie_order),
    }


def _validate_config(
    record: object,
    *,
    config_id: str,
    group: Mapping[str, Any],
) -> tuple[Path, str, Mapping[str, Any]]:
    path, digest = _validate_artifact(record, label=f"config {config_id}")
    payload = _load_json(path, label=f"config {config_id}")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise AdapterSelectionError(f"config {config_id} schema is not {CONFIG_SCHEMA}")
    identity = payload.get("config")
    if not isinstance(identity, Mapping) or identity.get("id") != config_id:
        raise AdapterSelectionError(f"config {config_id} identity mismatch")
    if identity.get("stage_id") != group["stage_id"]:
        raise AdapterSelectionError(f"config {config_id} stage differs from its group")
    if identity.get("candidate_value") != group["candidate_value"]:
        raise AdapterSelectionError(
            f"config {config_id} candidate_value differs from its group"
        )
    return path, digest, payload


def _flatten_group(
    group: object,
    *,
    baseline: bool,
    phase: str,
    code_contract_ids: set[str],
    sidecar_mtime_ns: int | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if not isinstance(group, Mapping):
        raise AdapterSelectionError("candidate group must be an object")
    expected = {"id", "candidate_value", "stage_id", "code_contract", "candidates"}
    if baseline:
        expected.add("frozen_selection")
    _exact_keys(group, expected, label="baseline group" if baseline else "architecture group")
    for field in ("id", "candidate_value", "stage_id", "code_contract"):
        if not isinstance(group[field], str) or not group[field]:
            raise AdapterSelectionError(f"group {field} must be a non-empty string")
    if group["code_contract"] not in code_contract_ids:
        raise AdapterSelectionError(
            f"group {group['id']} references unknown code contract {group['code_contract']}"
        )
    candidates = group["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise AdapterSelectionError(f"group {group['id']} candidates must be non-empty")
    folds: dict[int, dict[str, Any]] = {}
    config_ids: set[str] = set()
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise AdapterSelectionError(
                f"group {group['id']} candidate {candidate_index} must be an object"
            )
        _exact_keys(
            candidate,
            {"config_id", "config", "manifests"},
            label=f"group {group['id']} candidate {candidate_index}",
        )
        config_id = candidate["config_id"]
        if not isinstance(config_id, str) or not config_id or config_id in config_ids:
            raise AdapterSelectionError(f"group {group['id']} has invalid/duplicate config_id")
        config_ids.add(config_id)
        config_path, config_hash, config_payload = _validate_config(
            candidate["config"], config_id=config_id, group=group
        )
        manifests = candidate["manifests"]
        if not isinstance(manifests, Mapping) or not manifests:
            raise AdapterSelectionError(f"config {config_id} manifests must be an object")
        for fold_text, manifest_record in manifests.items():
            try:
                fold = int(fold_text)
            except (TypeError, ValueError) as error:
                raise AdapterSelectionError(
                    f"config {config_id} manifest fold is not an integer"
                ) from error
            if str(fold) != str(fold_text) or fold not in EXPECTED_FOLDS:
                raise AdapterSelectionError(f"config {config_id} has invalid fold {fold_text}")
            if fold in folds:
                raise AdapterSelectionError(
                    f"group {group['id']} defines fold {fold} more than once"
                )
            if not isinstance(manifest_record, Mapping):
                raise AdapterSelectionError(f"config {config_id} fold {fold} manifest is invalid")
            if baseline:
                _exact_keys(
                    manifest_record,
                    {"path", "sha256"},
                    label=f"baseline config {config_id} fold {fold} manifest",
                )
                manifest_path, manifest_hash = _validate_artifact(
                    manifest_record,
                    label=f"baseline config {config_id} fold {fold} manifest",
                )
            else:
                _exact_keys(
                    manifest_record,
                    {"path", "must_not_exist_at_freeze"},
                    label=f"architecture config {config_id} fold {fold} manifest",
                )
                if manifest_record["must_not_exist_at_freeze"] is not True:
                    raise AdapterSelectionError(
                        f"new manifest {config_id} fold {fold} must declare "
                        "must_not_exist_at_freeze=true"
                    )
                manifest_path = _resolve_path(
                    manifest_record["path"], label=f"config {config_id} fold {fold} manifest"
                )
                manifest_hash = None
                if phase == "freeze" and manifest_path.exists():
                    raise AdapterSelectionError(
                        f"new manifest already exists before plan freeze: {manifest_path}"
                    )
                if phase == "select":
                    if not manifest_path.is_file():
                        raise AdapterSelectionError(
                            f"expected new manifest is absent: {manifest_path}"
                        )
                    if (
                        sidecar_mtime_ns is not None
                        and manifest_path.stat().st_mtime_ns < sidecar_mtime_ns
                    ):
                        raise AdapterSelectionError(
                            f"new manifest predates frozen plan sidecar: {manifest_path}"
                        )
                    manifest_hash = sha256(manifest_path)
            if manifest_path.name != "candidate.json":
                raise AdapterSelectionError(
                    f"candidate manifest must be named candidate.json: {manifest_path}"
                )
            if any(_FORBIDDEN_ARTIFACT.search(part) for part in manifest_path.parts):
                raise AdapterSelectionError(
                    f"candidate manifest path contains forbidden outer/test component: "
                    f"{manifest_path}"
                )
            folds[fold] = {
                "group_id": group["id"],
                "candidate_value": group["candidate_value"],
                "stage_id": group["stage_id"],
                "code_contract": group["code_contract"],
                "config_id": config_id,
                "config_path": config_path,
                "config_sha256": config_hash,
                "config_payload": config_payload,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
            }
    if set(folds) != set(EXPECTED_FOLDS):
        raise AdapterSelectionError(
            f"group {group['id']} fold coverage differs: {sorted(folds)}"
        )
    return folds, dict(group)


def _validate_baseline_frozen_selection(
    baseline_group: Mapping[str, Any],
    baseline_folds: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    frozen_path, _ = _validate_artifact(
        baseline_group["frozen_selection"], label="baseline frozen_selection"
    )
    frozen = _load_json(frozen_path, label="baseline frozen selection")
    if frozen.get("schema") != FROZEN_SELECTION_SCHEMA:
        raise AdapterSelectionError("baseline frozen selection schema is invalid")
    raw_folds = frozen.get("folds")
    if not isinstance(raw_folds, Mapping) or set(raw_folds) != {
        str(fold) for fold in EXPECTED_FOLDS
    }:
        raise AdapterSelectionError("baseline frozen selection fold coverage is invalid")
    output: dict[int, dict[str, Any]] = {}
    for fold in EXPECTED_FOLDS:
        supplied = baseline_folds[fold]
        expected_config = BASELINE_CONFIG_BY_FOLD[fold]
        if supplied["config_id"] != expected_config:
            raise AdapterSelectionError(
                f"baseline fold {fold} must use {expected_config}, got {supplied['config_id']}"
            )
        record = raw_folds[str(fold)]
        if not isinstance(record, Mapping):
            raise AdapterSelectionError(f"baseline frozen fold {fold} is invalid")
        selected_config = record.get("selected_config")
        selected_manifest = record.get("selected_manifest")
        if (
            not isinstance(selected_config, Mapping)
            or selected_config.get("id") != expected_config
            or selected_config.get("stage_id") != "S4a_semantic_adapter"
        ):
            raise AdapterSelectionError(
                f"baseline frozen fold {fold} does not select the required S4a config"
            )
        if not isinstance(selected_manifest, Mapping):
            raise AdapterSelectionError(f"baseline frozen fold {fold} has no manifest record")
        selected_path, selected_hash = _artifact_identity(
            selected_manifest, label=f"baseline frozen fold {fold} selected manifest"
        )
        if (
            selected_path != supplied["manifest_path"]
            or selected_hash != supplied["manifest_sha256"]
        ):
            raise AdapterSelectionError(
                f"baseline fold {fold} manifest path/hash differs from old frozen selection"
            )
        epoch = record.get("selected_epoch")
        auprc = record.get("selected_validation_auprc")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise AdapterSelectionError(f"baseline frozen fold {fold} selected_epoch is invalid")
        if not isinstance(auprc, (int, float)) or not math.isfinite(float(auprc)):
            raise AdapterSelectionError(f"baseline frozen fold {fold} AUPRC is invalid")
        output[fold] = {"epoch": int(epoch), "auprc": float(auprc)}
    return output


def _validate_plan(
    plan_path: Path | str,
    *,
    phase: str,
) -> PlanContext:
    if phase not in {"freeze", "select"}:
        raise ValueError("phase must be freeze or select")
    path = Path(plan_path).resolve(strict=True)
    sidecar: Path | None = None
    digest = sha256(path)
    sidecar_mtime_ns: int | None = None
    if phase == "select":
        sidecar, digest = _read_plan_sidecar(path)
        sidecar_mtime_ns = sidecar.stat().st_mtime_ns
    payload = _load_json(path, label="adapter development plan")
    _exact_keys(
        payload,
        {
            "schema",
            "experiment_id",
            "dataset",
            "protocol",
            "features",
            "selection",
            "code_contracts",
            "baseline",
            "architectures",
            "output",
        },
        label="adapter development plan",
    )
    if payload["schema"] != PLAN_SCHEMA:
        raise AdapterSelectionError(f"plan schema must be {PLAN_SCHEMA}")
    if payload["dataset"] != DATASET:
        raise AdapterSelectionError(f"plan dataset must be {DATASET}")
    if not isinstance(payload["experiment_id"], str) or not payload["experiment_id"]:
        raise AdapterSelectionError("experiment_id must be a non-empty string")
    protocol_path, protocol_hash = _validate_artifact(
        payload["protocol"], label="protocol"
    )
    feature_contract = _validate_feature_contract(payload["features"])
    contracts = _validate_code_contracts(payload["code_contracts"])
    architectures = payload["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 3:
        raise AdapterSelectionError("architectures must contain exactly three groups")
    architecture_ids = [
        group.get("id") if isinstance(group, Mapping) else None for group in architectures
    ]
    if any(not isinstance(value, str) or not value for value in architecture_ids):
        raise AdapterSelectionError("each architecture requires a non-empty id")
    if len(set(architecture_ids)) != len(architecture_ids):
        raise AdapterSelectionError("architecture ids must be unique")
    rule = _validate_selection_rule(payload["selection"], architecture_ids)

    baseline_folds, baseline_group = _flatten_group(
        payload["baseline"],
        baseline=True,
        phase=phase,
        code_contract_ids=set(contracts),
        sidecar_mtime_ns=sidecar_mtime_ns,
    )
    if contracts[baseline_group["code_contract"]]["kind"] != "frozen_plan":
        raise AdapterSelectionError(
            "historical baseline must use a frozen_plan code contract"
        )
    if baseline_group["stage_id"] != "S4a_semantic_adapter":
        raise AdapterSelectionError("historical baseline stage must be S4a_semantic_adapter")
    groups: dict[str, Mapping[int, Mapping[str, Any]]] = {
        str(baseline_group["id"]): baseline_folds
    }
    if baseline_group["id"] in architecture_ids:
        raise AdapterSelectionError("baseline id cannot also be an architecture id")
    manifest_paths = {record["manifest_path"] for record in baseline_folds.values()}
    config_ids = {record["config_id"] for record in baseline_folds.values()}
    for raw_group in architectures:
        folds, group = _flatten_group(
            raw_group,
            baseline=False,
            phase=phase,
            code_contract_ids=set(contracts),
            sidecar_mtime_ns=sidecar_mtime_ns,
        )
        group_id = str(group["id"])
        if group["stage_id"] != "S5_semantic_adapter_architecture":
            raise AdapterSelectionError(
                f"new architecture {group_id} stage must be "
                "S5_semantic_adapter_architecture"
            )
        if group["candidate_value"] not in S5_ADAPTERS:
            raise AdapterSelectionError(
                f"architecture {group_id} candidate_value must be one of "
                + ", ".join(sorted(S5_ADAPTERS))
            )
        if contracts[group["code_contract"]]["kind"] != "files":
            raise AdapterSelectionError(
                f"new architecture {group_id} must use a files code contract"
            )
        group_config_ids = {record["config_id"] for record in folds.values()}
        duplicate_config_ids = config_ids.intersection(group_config_ids)
        if duplicate_config_ids:
            raise AdapterSelectionError(
                "config ids must be unique across architecture groups: "
                f"{sorted(duplicate_config_ids)}"
            )
        for record in folds.values():
            if record["manifest_path"] in manifest_paths:
                raise AdapterSelectionError("candidate manifest paths must be globally unique")
            manifest_paths.add(record["manifest_path"])
        config_ids.update(group_config_ids)
        groups[group_id] = folds
    baseline_frozen = _validate_baseline_frozen_selection(
        payload["baseline"], baseline_folds
    )

    output = payload["output"]
    if not isinstance(output, Mapping):
        raise AdapterSelectionError("output must be an object")
    _exact_keys(output, {"path"}, label="output")
    output_path = _resolve_path(output["path"], label="output")
    if output_path.name != "development_selection.json":
        raise AdapterSelectionError("output filename must be development_selection.json")
    if any(_FORBIDDEN_ARTIFACT.search(part) for part in output_path.parts[:-1]):
        raise AdapterSelectionError("output path contains a forbidden outer/test component")
    if output_path in manifest_paths:
        raise AdapterSelectionError("selection output cannot alias a candidate manifest")
    if output_path.exists():
        raise AdapterSelectionError(f"development selection already exists: {output_path}")

    # Retain normalized values inside a shallow copy used by the audit output.
    normalized = dict(payload)
    normalized["protocol"] = {
        "path": str(protocol_path),
        "sha256": protocol_hash,
    }
    normalized["features"] = feature_contract
    normalized["selection"] = rule
    normalized["code_contracts"] = contracts
    return PlanContext(
        path=path,
        digest=digest,
        sidecar=sidecar,
        payload=normalized,
        protocol_path=protocol_path,
        output_path=output_path,
        groups=groups,
        baseline_frozen=baseline_frozen,
    )


def validate_plan_before_training(plan_path: Path | str) -> PlanContext:
    """Validate a pre-training plan and require all S5 manifests to be absent."""

    return _validate_plan(plan_path, phase="freeze")


def freeze_plan(plan_path: Path | str) -> Path:
    """Freeze a valid pre-training plan by atomically writing its digest sidecar."""

    context = validate_plan_before_training(plan_path)
    sidecar = _sidecar_path(context.path)
    if sidecar.exists():
        raise AdapterSelectionError(f"plan sidecar already exists: {sidecar}")
    _atomic_text(sidecar, context.digest + "\n")
    return sidecar


def _validation_size(count: int, fraction: float) -> int:
    if count < 2:
        raise AdapterSelectionError("each outer-training class needs at least two rows")
    size = int(math.floor(count * fraction))
    return min(max(size, 1), count - 1)


def _derive_validation_rows(protocol: BenchmarkProtocol, fold: int) -> ValidationRows:
    """Derive inner rows without ever constructing an outer-test FoldRows."""

    positives = np.asarray(protocol.positive_pairs, dtype=np.int64)
    negatives = np.asarray(protocol.negative_pairs, dtype=np.int64)
    positive_count = len(positives)
    positive_rows = np.arange(positive_count, dtype=np.int64)
    negative_local = np.arange(len(negatives), dtype=np.int64)
    negative_rows = positive_count + negative_local
    positive_outer = positive_rows[positive_rows % protocol.n_folds != fold - 1]
    negative_outer = negative_rows[negative_local % protocol.n_folds != fold - 1]
    positive_rng = np.random.default_rng(np.random.SeedSequence((protocol.seed, fold, 1)))
    negative_rng = np.random.default_rng(np.random.SeedSequence((protocol.seed, fold, 0)))
    positive_outer = positive_outer[positive_rng.permutation(len(positive_outer))]
    negative_outer = negative_outer[negative_rng.permutation(len(negative_outer))]
    positive_validation = _validation_size(
        len(positive_outer), protocol.validation_fraction
    )
    negative_validation = _validation_size(
        len(negative_outer), protocol.validation_fraction
    )
    validation_rows = np.concatenate(
        (positive_outer[:positive_validation], negative_outer[:negative_validation])
    )
    train_rows = np.concatenate(
        (positive_outer[positive_validation:], negative_outer[negative_validation:])
    )
    positive_mask = validation_rows < positive_count
    pairs = np.empty((len(validation_rows), 2), dtype=np.int64)
    pairs[positive_mask] = positives[validation_rows[positive_mask]]
    pairs[~positive_mask] = negatives[
        validation_rows[~positive_mask] - positive_count
    ]
    labels = positive_mask.astype(np.uint8)
    return ValidationRows(train_rows, validation_rows, pairs, labels)


def _raw_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _relative_artifact(
    record: object, *, manifest_path: Path, label: str
) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or not {"path", "sha256"}.issubset(record):
        raise AdapterSelectionError(f"{label} must contain path and sha256")
    supplied = Path(str(record["path"]))
    path = (supplied if supplied.is_absolute() else manifest_path.parent / supplied).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise AdapterSelectionError(f"{label} escapes its candidate directory") from error
    expected = _validate_digest(record["sha256"], label=label)
    if not path.is_file() or sha256(path) != expected:
        raise AdapterSelectionError(f"{label} file/hash mismatch: {path}")
    return path, expected


def _reject_test_artifacts(candidate_root: Path) -> None:
    for path in candidate_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(candidate_root)
        if any(_FORBIDDEN_ARTIFACT.search(part) for part in relative.parts):
            raise AdapterSelectionError(
                f"candidate directory contains forbidden outer/test artifact: {path}"
            )


def _reject_manifest_test_references(value: object, *, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            name = str(child_key)
            if (
                name != "outer_evaluation_rows_loaded"
                and _FORBIDDEN_ARTIFACT.search(name)
            ):
                raise AdapterSelectionError(
                    f"candidate manifest contains forbidden outer/test field: {name}"
                )
            _reject_manifest_test_references(child, key=name)
    elif isinstance(value, list):
        for child in value:
            _reject_manifest_test_references(child, key=key)
    elif isinstance(value, str) and ("path" in key.lower() or key.lower().endswith("file")):
        path_parts = Path(value).parts
        if any(_FORBIDDEN_ARTIFACT.search(part) for part in path_parts):
            raise AdapterSelectionError(
                f"candidate manifest contains forbidden outer/test path in {key}"
            )


def _validate_candidate_features(
    manifest: Mapping[str, Any], plan_features: Mapping[str, Any]
) -> None:
    features = manifest.get("features")
    if not isinstance(features, Mapping):
        raise AdapterSelectionError("candidate has no feature provenance")
    selected = features.get("selected_features")
    if not isinstance(selected, Mapping):
        raise AdapterSelectionError("candidate selected_features are absent")
    for entity in FEATURE_ENTITIES:
        record = selected.get(entity)
        if not isinstance(record, Mapping):
            raise AdapterSelectionError(f"candidate selected feature {entity} is absent")
        candidate_path = _resolve_path(record.get("used_path"), label=f"candidate {entity}")
        candidate_hash = _validate_digest(
            record.get("used_sha256"), label=f"candidate {entity}"
        )
        expected = plan_features["selected_features"][entity]
        if candidate_path != Path(expected["path"]) or candidate_hash != expected["sha256"]:
            raise AdapterSelectionError(f"candidate selected feature {entity} differs from plan")
    for candidate_key, plan_key in (
        ("semantic_reduction", "semantic_reduction"),
        ("feature_deep_audit_receipt", "deep_audit_receipt"),
    ):
        record = features.get(candidate_key)
        if not isinstance(record, Mapping):
            raise AdapterSelectionError(f"candidate {candidate_key} is absent")
        candidate_path, candidate_hash = _artifact_identity(
            record, label=f"candidate {candidate_key}"
        )
        expected = plan_features[plan_key]
        if candidate_path != Path(expected["path"]) or candidate_hash != expected["sha256"]:
            raise AdapterSelectionError(f"candidate {candidate_key} differs from plan")


def _control_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=0.0)
        )
    if isinstance(expected, list):
        return isinstance(observed, (list, tuple)) and list(observed) == expected
    return observed == expected


def _validate_s5_runtime_receipt(
    manifest: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    """Prove that the runner consumed the frozen S5 configuration."""

    diagnostics = manifest.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise AdapterSelectionError("S5 candidate training_diagnostics are absent")
    resolved = diagnostics.get("resolved_hyperparameters")
    implementation = diagnostics.get("implementation")
    if not isinstance(resolved, Mapping):
        raise AdapterSelectionError("S5 resolved_hyperparameters are absent")
    if not isinstance(implementation, Mapping):
        raise AdapterSelectionError("S5 implementation receipt is absent")
    expected_implementation = {
        "model_module": "ASTRASemanticRefined",
        "runner_module": "astra.benchmark.ethnobotany_semantic_adapter_runner",
        "frozen_parent_model_module": "ASTRARefined",
    }
    for key, expected in expected_implementation.items():
        if implementation.get(key) != expected:
            raise AdapterSelectionError(f"S5 implementation receipt differs at {key}")

    config = record["config_payload"]
    training = config.get("training")
    options = config.get("model_options")
    if not isinstance(training, Mapping) or not isinstance(options, Mapping):
        raise AdapterSelectionError("S5 frozen config lacks training/model_options")
    required_training = {
        "epochs": "max_epochs",
        "checkpoint_epochs": "checkpoint_epochs",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "embedding_dimension": "embedding_dimension",
        "reg_u": "reg_u",
        "text_fusion_alpha": "text_fusion_alpha",
    }
    required_options = {
        "astra.n_layers": "n_layers",
        "astra.loss.kind": "loss_kind",
        "astra.loss.reduction": "loss_reduction",
        "astra.grad.clip_value": "gradient_clip_value",
        "astra.grad.clip_norm": "gradient_clip_norm",
        "astra.protein_gate": "protein_gate",
        "astra.dead_variables": "dead_variables",
        "astra.clique.sampling": "clique_sampling",
        "astra.clique.max_pairs_per_hyperedge": "clique_max_pairs_per_hyperedge",
        "astra.clique.max_directed_edges": "clique_max_directed_edges",
        "astra.semantic.adapter": "semantic_adapter",
        "astra.semantic.input_dim": "semantic_input_dim",
        "astra.semantic.freeze_input": "semantic_freeze_input",
        "astra.semantic.mlp_hidden_dim": "semantic_mlp_hidden_dim",
        "astra.training.trace_gradients": "trace_gradients",
    }
    for source_key, receipt_key in required_training.items():
        if source_key not in training:
            raise AdapterSelectionError(f"S5 config training.{source_key} is absent")
        if not _control_equal(resolved.get(receipt_key), training[source_key]):
            raise AdapterSelectionError(
                f"S5 resolved control differs from config at training.{source_key}"
            )
    for source_key, receipt_key in required_options.items():
        if source_key not in options:
            raise AdapterSelectionError(f"S5 config model_options.{source_key} is absent")
        if not _control_equal(resolved.get(receipt_key), options[source_key]):
            raise AdapterSelectionError(
                f"S5 resolved control differs from config at model_options.{source_key}"
            )
    if resolved.get("model_class") != "ASTRASemanticRefined":
        raise AdapterSelectionError("S5 model_class must be ASTRASemanticRefined")
    adapter = record["candidate_value"]
    if adapter not in S5_ADAPTERS or resolved.get("semantic_adapter") != adapter:
        raise AdapterSelectionError("S5 adapter does not match its candidate_value")
    if resolved.get("semantic_input_dim") != 64:
        raise AdapterSelectionError("S5 semantic_input_dim must equal 64")
    if resolved.get("semantic_freeze_input") is not False:
        raise AdapterSelectionError("S5 semantic_freeze_input must be false")
    if resolved.get("semantic_mlp_hidden_dim") != 32:
        raise AdapterSelectionError("S5 semantic_mlp_hidden_dim must equal 32")


def _score_embeddings(
    u: np.ndarray, i: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    herb = np.asarray(u, dtype=np.float64)
    disease = np.asarray(i, dtype=np.float64)
    if herb.ndim != 2 or disease.ndim != 2 or herb.shape[1] != disease.shape[1]:
        raise AdapterSelectionError("saved herb/disease embeddings are incompatible")
    if not np.isfinite(herb).all() or not np.isfinite(disease).all():
        raise AdapterSelectionError("saved embeddings contain NaN or infinity")
    if (
        np.any(pairs[:, 0] < 0)
        or np.any(pairs[:, 0] >= len(herb))
        or np.any(pairs[:, 1] < 0)
        or np.any(pairs[:, 1] >= len(disease))
    ):
        raise AdapterSelectionError("validation pair index exceeds saved embeddings")
    logits = np.einsum("ij,ij->i", herb[pairs[:, 0]], disease[pairs[:, 1]])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def _audit_candidate(
    record: Mapping[str, Any],
    *,
    fold: int,
    protocol: BenchmarkProtocol,
    validation: ValidationRows,
    plan: PlanContext,
) -> dict[str, Any]:
    manifest_path = Path(record["manifest_path"])
    manifest_hash = sha256(manifest_path)
    if record["manifest_sha256"] is not None and manifest_hash != record["manifest_sha256"]:
        raise AdapterSelectionError(f"candidate manifest hash changed: {manifest_path}")
    _reject_test_artifacts(manifest_path.parent)
    manifest = _load_json(manifest_path, label="candidate manifest")
    _reject_manifest_test_references(manifest)
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise AdapterSelectionError(f"candidate schema is not {CANDIDATE_SCHEMA}")
    if manifest.get("validation_only") is not True:
        raise AdapterSelectionError("candidate must declare validation_only=true")
    if manifest.get("outer_evaluation_rows_loaded") is not False:
        raise AdapterSelectionError(
            "candidate must declare outer_evaluation_rows_loaded=false"
        )
    if manifest.get("fold") != fold:
        raise AdapterSelectionError(f"candidate fold differs from plan fold {fold}")
    identity = manifest.get("config")
    expected_identity = record["config_payload"].get("config")
    if identity != expected_identity:
        raise AdapterSelectionError("candidate config identity differs from frozen config")
    source = manifest.get("config_source")
    if not isinstance(source, Mapping):
        raise AdapterSelectionError("candidate config_source is absent")
    source_path, source_hash = _artifact_identity(source, label="candidate config_source")
    if source_path != record["config_path"] or source_hash != record["config_sha256"]:
        raise AdapterSelectionError("candidate config source differs from plan")
    protocol_record = manifest.get("protocol")
    if not isinstance(protocol_record, Mapping):
        raise AdapterSelectionError("candidate protocol record is absent")
    protocol_path, protocol_hash = _artifact_identity(
        protocol_record, label="candidate protocol"
    )
    if protocol_path != plan.protocol_path or protocol_hash != plan.payload["protocol"]["sha256"]:
        raise AdapterSelectionError("candidate protocol differs from frozen plan")
    _validate_candidate_features(manifest, plan.payload["features"])
    if record["stage_id"] != "S4a_semantic_adapter":
        _validate_s5_runtime_receipt(manifest, record)
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise AdapterSelectionError("candidate split receipt is absent")
    expected_split = {
        "train_rows": len(validation.train_row_indices),
        "validation_rows": len(validation.validation_row_indices),
        "train_row_indices_sha256": _raw_array_sha256(validation.train_row_indices),
        "validation_row_indices_sha256": _raw_array_sha256(
            validation.validation_row_indices
        ),
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise AdapterSelectionError(f"candidate split receipt differs at {key}")

    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise AdapterSelectionError("candidate snapshots must be non-empty")
    epochs = [snapshot.get("epoch") if isinstance(snapshot, Mapping) else None for snapshot in snapshots]
    if epochs != plan.payload["selection"]["candidate_epochs"]:
        raise AdapterSelectionError(
            f"candidate checkpoint epochs differ from plan: {epochs}"
        )
    audited: list[dict[str, Any]] = []
    for snapshot in snapshots:
        assert isinstance(snapshot, Mapping)
        epoch = int(snapshot["epoch"])
        u_path, u_hash = _relative_artifact(
            snapshot.get("u_embedding"),
            manifest_path=manifest_path,
            label=f"fold {fold} epoch {epoch} herb embedding",
        )
        i_path, i_hash = _relative_artifact(
            snapshot.get("i_embedding"),
            manifest_path=manifest_path,
            label=f"fold {fold} epoch {epoch} disease embedding",
        )
        prediction_path, prediction_hash = _relative_artifact(
            snapshot.get("validation_predictions"),
            manifest_path=manifest_path,
            label=f"fold {fold} epoch {epoch} validation predictions",
        )
        try:
            with np.load(prediction_path, allow_pickle=False) as archive:
                if set(archive.files) != set(PREDICTION_FIELDS):
                    raise AdapterSelectionError(
                        f"validation NPZ fields differ at fold {fold} epoch {epoch}"
                    )
            predictions = load_predictions(prediction_path)
            aligned = validate_prediction_coverage(
                predictions,
                expected_row_id=validation.validation_row_indices,
                expected_pair=validation.validation_pairs,
                expected_label=validation.validation_labels,
            )
        except (BenchmarkAuditError, OSError, ValueError) as error:
            raise AdapterSelectionError(
                f"validation prediction audit failed at fold {fold} epoch {epoch}: {error}"
            ) from error
        try:
            u = np.load(u_path, allow_pickle=False)
            i = np.load(i_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AdapterSelectionError(
                f"cannot load embeddings at fold {fold} epoch {epoch}: {error}"
            ) from error
        if u.shape[0] != protocol.n_herbs or i.shape[0] != protocol.n_diseases:
            raise AdapterSelectionError(
                f"embedding row count differs from protocol at fold {fold} epoch {epoch}"
            )
        recomputed_scores = _score_embeddings(u, i, aligned.pair)
        max_score_drift = float(np.max(np.abs(recomputed_scores - aligned.score)))
        if not np.allclose(
            recomputed_scores, aligned.score, rtol=0.0, atol=SCORE_ATOL
        ):
            raise AdapterSelectionError(
                f"embedding-recomputed score mismatch at fold {fold} epoch {epoch}: "
                f"max_abs={max_score_drift}"
            )
        try:
            metrics = evaluate_validation_scores(aligned.label, recomputed_scores)
        except BenchmarkAuditError as error:
            raise AdapterSelectionError(
                f"validation metric recomputation failed at fold {fold} epoch {epoch}: {error}"
            ) from error
        auprc = float(metrics[METRIC])
        declared = snapshot.get("validation_auprc")
        if not isinstance(declared, (int, float)) or not math.isfinite(float(declared)):
            raise AdapterSelectionError("snapshot validation_auprc is invalid")
        if not math.isclose(auprc, float(declared), rel_tol=0.0, abs_tol=METRIC_ATOL):
            raise AdapterSelectionError(
                f"recomputed AUPRC differs at fold {fold} epoch {epoch}"
            )
        audited.append(
            {
                "epoch": epoch,
                "validation_auprc": auprc,
                "max_abs_embedding_score_drift": max_score_drift,
                "u_embedding_sha256": u_hash,
                "i_embedding_sha256": i_hash,
                "validation_predictions_sha256": prediction_hash,
            }
        )
    best = audited[0]
    for candidate in audited[1:]:
        gain = candidate["validation_auprc"] - best["validation_auprc"]
        if gain > METRIC_ATOL or (
            abs(gain) <= METRIC_ATOL and candidate["epoch"] < best["epoch"]
        ):
            best = candidate
    return {
        "fold": fold,
        "config_id": record["config_id"],
        "config_path": str(record["config_path"]),
        "config_sha256": record["config_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "code_contract": record["code_contract"],
        "selected_epoch": int(best["epoch"]),
        "validation_auprc": float(best["validation_auprc"]),
        "epoch_audit": audited,
    }


def _select_best_architecture(
    summaries: Mapping[str, Mapping[str, Any]], tie_order: Sequence[str]
) -> str:
    """Select maximum mean AUPRC, using the predeclared order for ties."""

    rank = {name: index for index, name in enumerate(tie_order)}
    best: str | None = None
    for name in tie_order:
        if name not in summaries:
            raise AdapterSelectionError(f"missing architecture summary {name}")
        if best is None:
            best = name
            continue
        gain = float(summaries[name]["mean_auprc"]) - float(
            summaries[best]["mean_auprc"]
        )
        if gain > METRIC_ATOL or (abs(gain) <= METRIC_ATOL and rank[name] < rank[best]):
            best = name
    assert best is not None
    return best


def _promotion_gate(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    minimum_mean_improvement: float = 0.001,
    minimum_improved_folds: int = 3,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.shape != (5,):
        raise AdapterSelectionError("promotion gate requires five paired folds")
    deltas = candidate - baseline
    mean_improvement = float(candidate.mean() - baseline.mean())
    improved = [index + 1 for index, value in enumerate(deltas) if value > METRIC_ATOL]
    mean_gate = mean_improvement >= float(minimum_mean_improvement)
    fold_gate = len(improved) >= int(minimum_improved_folds)
    return {
        "mean_improvement": mean_improvement,
        "required_mean_improvement": float(minimum_mean_improvement),
        "mean_gate_passed": bool(mean_gate),
        "improved_folds": improved,
        "improved_fold_count": len(improved),
        "required_improved_folds": int(minimum_improved_folds),
        "fold_gate_passed": bool(fold_gate),
        "promoted": bool(mean_gate and fold_gate),
    }


def _group_summary(
    group_id: str,
    folds: Sequence[Mapping[str, Any]],
    *,
    baseline_values: Sequence[float] | None,
) -> dict[str, Any]:
    ordered = sorted(folds, key=lambda value: int(value["fold"]))
    values = np.asarray([float(value["validation_auprc"]) for value in ordered])
    output: dict[str, Any] = {
        "id": group_id,
        "folds": ordered,
        "mean_auprc": float(values.mean()),
        "std_auprc": float(values.std(ddof=0)),
    }
    code_contracts = {str(value["code_contract"]) for value in ordered}
    if len(code_contracts) != 1:
        raise AdapterSelectionError(f"group {group_id} spans multiple code contracts")
    output["code_contract"] = next(iter(code_contracts))
    if baseline_values is not None:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        deltas = values - baseline
        for fold_record, delta in zip(ordered, deltas):
            fold_record["delta_vs_baseline"] = float(delta)
        output["gate"] = _promotion_gate(values, baseline)
    else:
        for fold_record in ordered:
            fold_record["delta_vs_baseline"] = 0.0
    return output


def select_development_model(
    plan_path: Path | str,
    *,
    protocol_loader=load_protocol_manifest,
) -> dict[str, Any]:
    """Audit all validation candidates and atomically commit development selection."""

    plan = _validate_plan(plan_path, phase="select")
    protocol = protocol_loader(plan.protocol_path)
    if protocol.dataset != DATASET or protocol.n_folds != len(EXPECTED_FOLDS):
        raise AdapterSelectionError("loaded protocol is not five-fold Ethnobotany")
    validations = {
        fold: _derive_validation_rows(protocol, fold) for fold in EXPECTED_FOLDS
    }
    audited_groups: dict[str, list[dict[str, Any]]] = {}
    for group_id, folds in plan.groups.items():
        audited_groups[group_id] = [
            _audit_candidate(
                folds[fold],
                fold=fold,
                protocol=protocol,
                validation=validations[fold],
                plan=plan,
            )
            for fold in EXPECTED_FOLDS
        ]
    baseline_id = str(plan.payload["baseline"]["id"])
    baseline_audits = audited_groups[baseline_id]
    for fold_record in baseline_audits:
        fold = int(fold_record["fold"])
        frozen = plan.baseline_frozen[fold]
        if fold_record["selected_epoch"] != frozen["epoch"] or not math.isclose(
            fold_record["validation_auprc"],
            frozen["auprc"],
            rel_tol=0.0,
            abs_tol=METRIC_ATOL,
        ):
            raise AdapterSelectionError(
                f"baseline fold {fold} replay differs from old frozen selection"
            )
    baseline_summary = _group_summary(
        baseline_id, baseline_audits, baseline_values=None
    )
    baseline_values = [
        float(record["validation_auprc"])
        for record in sorted(baseline_audits, key=lambda value: value["fold"])
    ]
    architecture_summaries = {
        group_id: _group_summary(
            group_id, audited_groups[group_id], baseline_values=baseline_values
        )
        for group_id in plan.payload["selection"]["architecture_tie_order"]
    }
    best_id = _select_best_architecture(
        architecture_summaries,
        plan.payload["selection"]["architecture_tie_order"],
    )
    best = architecture_summaries[best_id]
    promoted = bool(best["gate"]["promoted"])
    selected_id = best_id if promoted else baseline_id
    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "dataset": DATASET,
        "experiment_id": plan.payload["experiment_id"],
        "development_only": True,
        "validation_only": True,
        "outer_evaluation_permitted": False,
        "outer_score_command_available": False,
        "policy": (
            "Architecture and epoch selection use inner-validation artifacts only; "
            "this module provides no held-out evaluation operation."
        ),
        "plan": {
            "path": str(plan.path),
            "sha256": plan.digest,
            "sidecar": str(plan.sidecar),
            "sidecar_sha256": sha256(plan.sidecar) if plan.sidecar else None,
        },
        "protocol": {
            "path": str(plan.protocol_path),
            "sha256": plan.payload["protocol"]["sha256"],
        },
        "selection_rule": dict(plan.payload["selection"]),
        "code_contracts": dict(plan.payload["code_contracts"]),
        "baseline": baseline_summary,
        "architectures": architecture_summaries,
        "best_new_architecture": best_id,
        "best_new_architecture_mean_auprc": float(best["mean_auprc"]),
        "gate": dict(best["gate"]),
        "promoted": promoted,
        "selected_model": selected_id,
    }
    _atomic_json(plan.output_path, payload)
    _atomic_text(Path(str(plan.output_path) + ".sha256"), sha256(plan.output_path) + "\n")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or execute validation-only Ethnobotany S5 adapter selection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze-plan", help="validate the pre-training plan and write its SHA sidecar"
    )
    freeze.add_argument("--plan", type=Path, required=True)
    select = subparsers.add_parser(
        "select", help="audit validation manifests and write development_selection.json"
    )
    select.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "freeze-plan":
            path = freeze_plan(arguments.plan)
            print(path)
        elif arguments.command == "select":
            payload = select_development_model(arguments.plan)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:  # pragma: no cover - argparse enforces the command set.
            raise AdapterSelectionError("unsupported command")
    except AdapterSelectionError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AdapterSelectionError",
    "BASELINE_CONFIG_BY_FOLD",
    "OUTPUT_SCHEMA",
    "PLAN_SCHEMA",
    "freeze_plan",
    "select_development_model",
    "sha256",
    "validate_plan_before_training",
]
