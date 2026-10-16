#!/usr/bin/env python3
"""Outcome-blind receipt amendment for Ethnobotany S5 adapter selection.

The original frozen S5 plan correctly binds the current feature receipt for
new candidates, but its historical S4a baseline manifests bind the earlier
receipt under which they were produced.  This module leaves the frozen plan
and its selector untouched.  A post-training, pre-outcome amendment binds the
two receipts separately, binds both selector implementations, and freezes the
hashes of every S5 candidate manifest before any validation metric is read.

Only ``freeze-amendment`` and validation-only ``select`` are exposed.  There
is deliberately no outer/test evaluation operation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import ethnobotany_adapter_selection as base
from .protocol import load_protocol_manifest


AMENDMENT_SCHEMA = "astra-ethnobotany-adapter-selection-amendment/v1"
OUTPUT_SCHEMA = "astra-ethnobotany-adapter-development-selection-amended/v1"
RECEIPT_SCHEMA = "astra-ethnobotany-refined-feature-deep-audit-receipt/v1"
EXPECTED_ORIGINAL_SELECTOR_SHA256 = (
    "9d09690bee40fdf01bd989666482c342b3fa7101fb5649aeccf11a85052cdd36"
)
AMENDED_OUTPUT_NAME = "development_selection_amended.json"
FAILURE_STAGE = "feature_receipt_identity_before_snapshot_or_score_audit"
_ATTESTATION_FLAGS = (
    "new_s5_validation_metrics_reviewed_before_freeze",
    "selection_rule_changed",
    "candidate_set_changed",
    "epoch_grid_changed",
    "promotion_gate_changed",
)


@dataclass(frozen=True)
class OriginalPlanAudit:
    """Original plan metadata validated without opening candidate JSON."""

    path: Path
    digest: str
    sidecar: Path
    payload: Mapping[str, Any]
    protocol_path: Path
    groups: Mapping[str, Mapping[int, Mapping[str, Any]]]
    baseline_group: Mapping[str, Any]


@dataclass(frozen=True)
class AmendmentContext:
    path: Path
    digest: str
    sidecar: Path | None
    payload: Mapping[str, Any]
    original: OriginalPlanAudit
    baseline_receipt: Mapping[str, str]
    s5_receipt: Mapping[str, str]
    receipt_equivalence: Mapping[str, Any]
    output_path: Path


def _sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _read_frozen_sidecar(path: Path, *, label: str) -> tuple[Path, str]:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        raise base.AdapterSelectionError(
            f"frozen {label} sidecar is absent: {sidecar}"
        )
    try:
        value = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise base.AdapterSelectionError(
            f"cannot read {label} sidecar {sidecar}: {error}"
        ) from error
    expected = base._validate_digest(value, label=f"{label} sidecar")
    observed = base.sha256(path)
    if observed != expected:
        raise base.AdapterSelectionError(
            f"frozen {label} was modified: expected {expected}, observed {observed}"
        )
    if path.stat().st_mtime_ns > sidecar.stat().st_mtime_ns:
        raise base.AdapterSelectionError(
            f"{label} modification time is newer than its freeze sidecar"
        )
    return sidecar, expected


def _validate_original_plan_outcome_blind(record: object) -> OriginalPlanAudit:
    """Validate the v2 plan without parsing candidate or selection outcomes."""

    plan_path, declared_digest = base._validate_artifact(
        record, label="original frozen plan"
    )
    sidecar, frozen_digest = _read_frozen_sidecar(
        plan_path, label="original plan"
    )
    if declared_digest != frozen_digest:
        raise base.AdapterSelectionError(
            "amendment original_plan hash differs from the frozen plan sidecar"
        )
    payload = base._load_json(plan_path, label="original adapter development plan")
    base._exact_keys(
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
        label="original adapter development plan",
    )
    if payload.get("schema") != base.PLAN_SCHEMA:
        raise base.AdapterSelectionError(
            f"original plan schema must be {base.PLAN_SCHEMA}"
        )
    if payload.get("dataset") != base.DATASET:
        raise base.AdapterSelectionError(
            f"original plan dataset must be {base.DATASET}"
        )
    if not isinstance(payload.get("experiment_id"), str) or not payload["experiment_id"]:
        raise base.AdapterSelectionError("original plan experiment_id is invalid")

    protocol_path, protocol_hash = base._validate_artifact(
        payload["protocol"], label="protocol"
    )
    features = base._validate_feature_contract(payload["features"])
    contracts = base._validate_code_contracts(payload["code_contracts"])
    architectures = payload["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 3:
        raise base.AdapterSelectionError(
            "original plan architectures must contain exactly three groups"
        )
    architecture_ids = [
        group.get("id") if isinstance(group, Mapping) else None
        for group in architectures
    ]
    if any(not isinstance(value, str) or not value for value in architecture_ids):
        raise base.AdapterSelectionError("each architecture requires a non-empty id")
    if len(set(architecture_ids)) != len(architecture_ids):
        raise base.AdapterSelectionError("architecture ids must be unique")
    rule = base._validate_selection_rule(payload["selection"], architecture_ids)

    groups: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    baseline_folds, baseline_group = base._flatten_group(
        payload["baseline"],
        baseline=True,
        phase="select",
        code_contract_ids=set(contracts),
        sidecar_mtime_ns=sidecar.stat().st_mtime_ns,
    )
    baseline_id = str(baseline_group["id"])
    if contracts[baseline_group["code_contract"]]["kind"] != "frozen_plan":
        raise base.AdapterSelectionError(
            "historical baseline must use a frozen_plan code contract"
        )
    if baseline_group["stage_id"] != "S4a_semantic_adapter":
        raise base.AdapterSelectionError(
            "historical baseline stage must be S4a_semantic_adapter"
        )
    if baseline_id in architecture_ids:
        raise base.AdapterSelectionError(
            "baseline id cannot also be an architecture id"
        )
    groups[baseline_id] = baseline_folds
    manifest_paths = {record["manifest_path"] for record in baseline_folds.values()}
    config_ids = {record["config_id"] for record in baseline_folds.values()}
    for raw_group in architectures:
        folds, group = base._flatten_group(
            raw_group,
            baseline=False,
            phase="select",
            code_contract_ids=set(contracts),
            sidecar_mtime_ns=sidecar.stat().st_mtime_ns,
        )
        group_id = str(group["id"])
        if group["stage_id"] != "S5_semantic_adapter_architecture":
            raise base.AdapterSelectionError(
                f"new architecture {group_id} has the wrong stage"
            )
        if group["candidate_value"] not in base.S5_ADAPTERS:
            raise base.AdapterSelectionError(
                f"architecture {group_id} candidate_value is unsupported"
            )
        if contracts[group["code_contract"]]["kind"] != "files":
            raise base.AdapterSelectionError(
                f"new architecture {group_id} must use a files code contract"
            )
        group_config_ids = {record["config_id"] for record in folds.values()}
        duplicate_config_ids = config_ids.intersection(group_config_ids)
        if duplicate_config_ids:
            raise base.AdapterSelectionError(
                "config ids must be unique across architecture groups: "
                f"{sorted(duplicate_config_ids)}"
            )
        for candidate in folds.values():
            if candidate["manifest_path"] in manifest_paths:
                raise base.AdapterSelectionError(
                    "candidate manifest paths must be globally unique"
                )
            manifest_paths.add(candidate["manifest_path"])
        config_ids.update(group_config_ids)
        groups[group_id] = folds

    output = payload["output"]
    if not isinstance(output, Mapping):
        raise base.AdapterSelectionError("original plan output must be an object")
    base._exact_keys(output, {"path"}, label="original plan output")
    original_output = base._resolve_path(output["path"], label="original plan output")
    if original_output.name != "development_selection.json":
        raise base.AdapterSelectionError(
            "original plan output filename must be development_selection.json"
        )

    normalized = dict(payload)
    normalized["protocol"] = {"path": str(protocol_path), "sha256": protocol_hash}
    normalized["features"] = features
    normalized["selection"] = rule
    normalized["code_contracts"] = contracts
    return OriginalPlanAudit(
        path=plan_path,
        digest=declared_digest,
        sidecar=sidecar,
        payload=normalized,
        protocol_path=protocol_path,
        groups=groups,
        baseline_group=baseline_group,
    )


def _validate_selector_contracts(
    raw: object, original: OriginalPlanAudit
) -> dict[str, dict[str, str]]:
    if not isinstance(raw, Mapping):
        raise base.AdapterSelectionError("selectors must be an object")
    base._exact_keys(raw, {"original", "amendment"}, label="selectors")
    original_path, original_hash = base._validate_artifact(
        raw["original"], label="original selector"
    )
    amendment_path, amendment_hash = base._validate_artifact(
        raw["amendment"], label="amendment selector"
    )
    expected_original_path = Path(base.__file__).resolve()
    if original_path != expected_original_path:
        raise base.AdapterSelectionError(
            "original selector path does not bind ethnobotany_adapter_selection.py"
        )
    if original_hash != EXPECTED_ORIGINAL_SELECTOR_SHA256:
        raise base.AdapterSelectionError(
            "original selector hash is not the frozen 9d0969 contract"
        )
    expected_amendment_path = Path(__file__).resolve()
    if amendment_path != expected_amendment_path:
        raise base.AdapterSelectionError(
            "amendment selector path does not bind this module"
        )

    development_contracts = [
        contract
        for contract in original.payload["code_contracts"].values()
        if contract.get("kind") == "files"
    ]
    if len(development_contracts) != 1:
        raise base.AdapterSelectionError(
            "original plan must contain one development files code contract"
        )
    planned_selector = development_contracts[0]["selector"]
    planned_path, planned_hash = base._artifact_identity(
        planned_selector, label="original plan selector"
    )
    if planned_path != original_path or planned_hash != original_hash:
        raise base.AdapterSelectionError(
            "amendment original selector differs from the original plan code contract"
        )
    return {
        "original": {"path": str(original_path), "sha256": original_hash},
        "amendment": {"path": str(amendment_path), "sha256": amendment_hash},
    }


_RECEIPT_EQUIVALENCE_FIELDS = (
    "schema",
    "dataset",
    "deep_audit",
    "path_bound",
    "protocol",
    "bundle_manifest",
    "semantic_reduction_manifest",
    "semantic_deep_audit_receipt",
    "canonical_features",
    "missing_masks",
)


def _dependency_identity(record: object, *, label: str) -> tuple[str, str, int]:
    if not isinstance(record, Mapping):
        raise base.AdapterSelectionError(f"{label} must be an object")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("bytes")
    if not isinstance(path, str) or not path:
        raise base.AdapterSelectionError(f"{label}.path must be a non-empty string")
    digest = base._validate_digest(digest, label=label)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise base.AdapterSelectionError(f"{label}.bytes must be a non-negative integer")
    return str(Path(path).resolve()), digest, size


def _prove_receipt_equivalence(
    baseline_path: Path, s5_path: Path
) -> dict[str, Any]:
    baseline = base._load_json(baseline_path, label="baseline deep-audit receipt")
    s5 = base._load_json(s5_path, label="S5 deep-audit receipt")
    if set(baseline) != set(s5):
        raise base.AdapterSelectionError(
            "receipt top-level key sets differ"
        )
    for field in _RECEIPT_EQUIVALENCE_FIELDS:
        if field not in baseline or field not in s5:
            raise base.AdapterSelectionError(
                f"receipt equivalence field is absent: {field}"
            )
        if baseline[field] != s5[field]:
            raise base.AdapterSelectionError(
                f"receipt data identity differs at {field}"
            )
    if baseline["schema"] != RECEIPT_SCHEMA:
        raise base.AdapterSelectionError(
            f"receipt schema must be {RECEIPT_SCHEMA}"
        )
    if baseline["dataset"] != base.DATASET:
        raise base.AdapterSelectionError(
            f"receipt dataset must be {base.DATASET}"
        )
    if baseline["deep_audit"] is not True or baseline["path_bound"] is not True:
        raise base.AdapterSelectionError(
            "both receipts must prove deep_audit=true and path_bound=true"
        )
    allowed_differences = {"created_at", "validator_sources"}
    exact_fields = sorted(set(baseline) - allowed_differences)
    for field in exact_fields:
        if baseline[field] != s5[field]:
            raise base.AdapterSelectionError(
                f"receipt exact field differs at {field}"
            )

    validator_maps: list[dict[str, Mapping[str, Any]]] = []
    for label, receipt in (("baseline", baseline), ("S5", s5)):
        sources = receipt.get("validator_sources")
        if not isinstance(sources, list) or not sources:
            raise base.AdapterSelectionError(
                f"{label} receipt validator_sources must be a non-empty list"
            )
        mapped: dict[str, Mapping[str, Any]] = {}
        for index, value in enumerate(sources):
            identity = _dependency_identity(
                value, label=f"{label} receipt.validator_sources[{index}]"
            )
            if identity[0] in mapped:
                raise base.AdapterSelectionError(
                    f"{label} receipt has duplicate validator source path"
                )
            assert isinstance(value, Mapping)
            mapped[identity[0]] = value
        validator_maps.append(mapped)
    baseline_validators, s5_validators = validator_maps
    if set(baseline_validators) != set(s5_validators):
        raise base.AdapterSelectionError(
            "receipt validator source path sets differ"
        )
    changed_validator_sources: list[str] = []
    for source_path in sorted(baseline_validators):
        if baseline_validators[source_path] == s5_validators[source_path]:
            continue
        if Path(source_path).name != "protocol.py":
            raise base.AdapterSelectionError(
                "only the protocol.py validator source record may differ"
            )
        changed_validator_sources.append(source_path)
    if len(changed_validator_sources) != 1:
        raise base.AdapterSelectionError(
            "receipt validator_sources must differ at exactly one protocol.py record"
        )

    exact_payload = {field: baseline[field] for field in exact_fields}
    encoded = json.dumps(
        exact_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "proven",
        "top_level_key_sets_equal": True,
        "exact_fields": exact_fields,
        "allowed_metadata_differences": ["created_at", "validator_sources"],
        "validator_source_paths_equal": True,
        "changed_validator_sources": changed_validator_sources,
        "validator_source_difference_policy": "protocol.py_record_only",
        "exact_payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_receipt_contracts(
    raw: object, original: OriginalPlanAudit
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise base.AdapterSelectionError("receipt_contracts must be an object")
    base._exact_keys(raw, {"baseline", "s5"}, label="receipt_contracts")
    normalized: dict[str, dict[str, str]] = {}
    for name in ("baseline", "s5"):
        path, digest = base._validate_artifact(
            raw[name], label=f"{name} deep-audit receipt"
        )
        normalized[name] = {"path": str(path), "sha256": digest}
    planned = original.payload["features"]["deep_audit_receipt"]
    planned_path, planned_hash = base._artifact_identity(
        planned, label="original plan S5 receipt"
    )
    if (
        Path(normalized["s5"]["path"]) != planned_path
        or normalized["s5"]["sha256"] != planned_hash
    ):
        raise base.AdapterSelectionError(
            "amendment S5 receipt differs from the original frozen plan"
        )
    if normalized["baseline"] == normalized["s5"]:
        raise base.AdapterSelectionError(
            "baseline and S5 receipts must be distinct for this amendment"
        )
    equivalence = _prove_receipt_equivalence(
        Path(normalized["baseline"]["path"]), Path(normalized["s5"]["path"])
    )
    return normalized["baseline"], normalized["s5"], equivalence


def _expected_s5_manifests(
    original: OriginalPlanAudit,
) -> dict[Path, Mapping[str, Any]]:
    baseline_id = str(original.payload["baseline"]["id"])
    return {
        Path(record["manifest_path"]): record
        for group_id, folds in original.groups.items()
        if group_id != baseline_id
        for record in folds.values()
    }


def _validate_manifest_inventory(
    raw: object, original: OriginalPlanAudit
) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise base.AdapterSelectionError("s5_candidate_manifests must be a list")
    expected = _expected_s5_manifests(original)
    if len(raw) != len(expected):
        raise base.AdapterSelectionError(
            "s5_candidate_manifests must bind every planned S5 manifest exactly once"
        )
    observed: dict[Path, str] = {}
    for index, artifact in enumerate(raw):
        path, digest = base._validate_artifact(
            artifact, label=f"S5 candidate manifest inventory item {index}"
        )
        if path in observed:
            raise base.AdapterSelectionError(
                f"duplicate S5 candidate manifest inventory path: {path}"
            )
        observed[path] = digest
    if set(observed) != set(expected):
        missing = sorted(str(path) for path in set(expected) - set(observed))
        extra = sorted(str(path) for path in set(observed) - set(expected))
        raise base.AdapterSelectionError(
            f"S5 candidate manifest inventory differs: missing={missing}, extra={extra}"
        )
    return [
        {"path": str(path), "sha256": observed[path]}
        for path in sorted(observed, key=str)
    ]


def _validate_outcome_blind_attestation(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise base.AdapterSelectionError(
            "outcome_blind_attestation must be an object"
        )
    base._exact_keys(
        raw,
        {*_ATTESTATION_FLAGS, "failure_stage"},
        label="outcome_blind_attestation",
    )
    for field in _ATTESTATION_FLAGS:
        if raw[field] is not False:
            raise base.AdapterSelectionError(
                f"outcome_blind_attestation.{field} must be false"
            )
    if raw["failure_stage"] != FAILURE_STAGE:
        raise base.AdapterSelectionError(
            "outcome_blind_attestation.failure_stage is invalid"
        )
    return dict(raw)


def _validate_output(raw: object, original: OriginalPlanAudit) -> Path:
    if not isinstance(raw, Mapping):
        raise base.AdapterSelectionError("amendment output must be an object")
    base._exact_keys(raw, {"path"}, label="amendment output")
    output = base._resolve_path(raw["path"], label="amendment output")
    if output.name != AMENDED_OUTPUT_NAME:
        raise base.AdapterSelectionError(
            f"amendment output filename must be {AMENDED_OUTPUT_NAME}"
        )
    if any(base._FORBIDDEN_ARTIFACT.search(part) for part in output.parts[:-1]):
        raise base.AdapterSelectionError(
            "amendment output path contains a forbidden outer/test component"
        )
    all_manifests = {
        Path(record["manifest_path"])
        for folds in original.groups.values()
        for record in folds.values()
    }
    if output in all_manifests:
        raise base.AdapterSelectionError(
            "amendment output cannot alias a candidate manifest"
        )
    if output.exists() or _sidecar_path(output).exists():
        raise base.AdapterSelectionError(
            f"amended development selection already exists: {output}"
        )
    return output


def _validate_amendment(
    amendment_path: Path | str, *, phase: str
) -> AmendmentContext:
    if phase not in {"freeze", "select"}:
        raise ValueError("phase must be freeze or select")
    path = Path(amendment_path).resolve(strict=True)
    digest = base.sha256(path)
    sidecar: Path | None = None
    if phase == "select":
        sidecar, digest = _read_frozen_sidecar(path, label="amendment")
    payload = base._load_json(path, label="selection amendment")
    base._exact_keys(
        payload,
        {
            "schema",
            "amendment_id",
            "reason",
            "original_plan",
            "receipt_contracts",
            "selectors",
            "s5_candidate_manifests",
            "outcome_blind_attestation",
            "output",
        },
        label="selection amendment",
    )
    if payload.get("schema") != AMENDMENT_SCHEMA:
        raise base.AdapterSelectionError(
            f"amendment schema must be {AMENDMENT_SCHEMA}"
        )
    if not isinstance(payload.get("amendment_id"), str) or not payload["amendment_id"]:
        raise base.AdapterSelectionError("amendment_id must be a non-empty string")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        raise base.AdapterSelectionError("amendment reason must be a non-empty string")

    original = _validate_original_plan_outcome_blind(payload["original_plan"])
    receipts = _validate_receipt_contracts(payload["receipt_contracts"], original)
    selectors = _validate_selector_contracts(payload["selectors"], original)
    inventory = _validate_manifest_inventory(
        payload["s5_candidate_manifests"], original
    )
    attestation = _validate_outcome_blind_attestation(
        payload["outcome_blind_attestation"]
    )
    output = _validate_output(payload["output"], original)

    normalized = dict(payload)
    normalized["original_plan"] = {
        "path": str(original.path),
        "sha256": original.digest,
    }
    normalized["receipt_contracts"] = {
        "baseline": dict(receipts[0]),
        "s5": dict(receipts[1]),
    }
    normalized["selectors"] = selectors
    normalized["s5_candidate_manifests"] = inventory
    normalized["outcome_blind_attestation"] = attestation
    normalized["output"] = {"path": str(output)}
    return AmendmentContext(
        path=path,
        digest=digest,
        sidecar=sidecar,
        payload=normalized,
        original=original,
        baseline_receipt=receipts[0],
        s5_receipt=receipts[1],
        receipt_equivalence=receipts[2],
        output_path=output,
    )


def validate_amendment_before_outcome_review(
    amendment_path: Path | str,
) -> AmendmentContext:
    """Validate a post-training amendment without parsing candidate JSON."""

    return _validate_amendment(amendment_path, phase="freeze")


def freeze_amendment(amendment_path: Path | str) -> Path:
    """Atomically freeze the outcome-blind amendment and S5 manifest inventory."""

    context = validate_amendment_before_outcome_review(amendment_path)
    sidecar = _sidecar_path(context.path)
    if sidecar.exists():
        raise base.AdapterSelectionError(
            f"amendment sidecar already exists: {sidecar}"
        )
    base._atomic_text(sidecar, context.digest + "\n")
    return sidecar


def _selection_plan(context: AmendmentContext) -> base.PlanContext:
    """Build the original selection context while using the amended output path."""

    original = context.original
    baseline_id = str(original.payload["baseline"]["id"])
    baseline_folds = original.groups[baseline_id]
    baseline_frozen = base._validate_baseline_frozen_selection(
        original.payload["baseline"], baseline_folds
    )
    return base.PlanContext(
        path=original.path,
        digest=original.digest,
        sidecar=original.sidecar,
        payload=original.payload,
        protocol_path=original.protocol_path,
        output_path=context.output_path,
        groups=original.groups,
        baseline_frozen=baseline_frozen,
    )


def _baseline_receipt_plan(
    plan: base.PlanContext, receipt: Mapping[str, str]
) -> base.PlanContext:
    """Clone only the receipt contract used to audit the historical baseline."""

    payload = dict(plan.payload)
    features = deepcopy(plan.payload["features"])
    features["deep_audit_receipt"] = dict(receipt)
    payload["features"] = features
    return base.PlanContext(
        path=plan.path,
        digest=plan.digest,
        sidecar=plan.sidecar,
        payload=payload,
        protocol_path=plan.protocol_path,
        output_path=plan.output_path,
        groups=plan.groups,
        baseline_frozen=plan.baseline_frozen,
    )


def select_development_model(
    amendment_path: Path | str,
    *,
    protocol_loader=load_protocol_manifest,
) -> dict[str, Any]:
    """Run the original validation-only selection with split receipt contracts."""

    amendment = _validate_amendment(amendment_path, phase="select")
    plan = _selection_plan(amendment)
    baseline_plan = _baseline_receipt_plan(plan, amendment.baseline_receipt)
    protocol = protocol_loader(plan.protocol_path)
    if protocol.dataset != base.DATASET or protocol.n_folds != len(base.EXPECTED_FOLDS):
        raise base.AdapterSelectionError(
            "loaded protocol is not five-fold Ethnobotany"
        )
    validations = {
        fold: base._derive_validation_rows(protocol, fold)
        for fold in base.EXPECTED_FOLDS
    }
    baseline_id = str(plan.payload["baseline"]["id"])
    audited_groups: dict[str, list[dict[str, Any]]] = {}
    for group_id, folds in plan.groups.items():
        audit_plan = baseline_plan if group_id == baseline_id else plan
        audited_groups[group_id] = [
            base._audit_candidate(
                folds[fold],
                fold=fold,
                protocol=protocol,
                validation=validations[fold],
                plan=audit_plan,
            )
            for fold in base.EXPECTED_FOLDS
        ]

    baseline_audits = audited_groups[baseline_id]
    for fold_record in baseline_audits:
        fold = int(fold_record["fold"])
        frozen = plan.baseline_frozen[fold]
        if fold_record["selected_epoch"] != frozen["epoch"] or not math.isclose(
            fold_record["validation_auprc"],
            frozen["auprc"],
            rel_tol=0.0,
            abs_tol=base.METRIC_ATOL,
        ):
            raise base.AdapterSelectionError(
                f"baseline fold {fold} replay differs from old frozen selection"
            )
    baseline_summary = base._group_summary(
        baseline_id, baseline_audits, baseline_values=None
    )
    baseline_values = [
        float(record["validation_auprc"])
        for record in sorted(baseline_audits, key=lambda value: value["fold"])
    ]
    architecture_summaries = {
        group_id: base._group_summary(
            group_id,
            audited_groups[group_id],
            baseline_values=baseline_values,
        )
        for group_id in plan.payload["selection"]["architecture_tie_order"]
    }
    best_id = base._select_best_architecture(
        architecture_summaries,
        plan.payload["selection"]["architecture_tie_order"],
    )
    best = architecture_summaries[best_id]
    promoted = bool(best["gate"]["promoted"])
    selected_id = best_id if promoted else baseline_id
    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "dataset": base.DATASET,
        "experiment_id": plan.payload["experiment_id"],
        "development_only": True,
        "validation_only": True,
        "outer_evaluation_permitted": False,
        "outer_score_command_available": False,
        "policy": (
            "Architecture and epoch selection use inner-validation artifacts only; "
            "the historical baseline and S5 candidates use separately frozen "
            "deep-audit receipts, and this module provides no held-out evaluation."
        ),
        "amendment": {
            "path": str(amendment.path),
            "sha256": amendment.digest,
            "sidecar": str(amendment.sidecar),
            "sidecar_sha256": (
                base.sha256(amendment.sidecar) if amendment.sidecar else None
            ),
            "receipt_contracts": dict(amendment.payload["receipt_contracts"]),
            "receipt_equivalence": dict(amendment.receipt_equivalence),
            "selectors": dict(amendment.payload["selectors"]),
            "s5_candidate_manifests": list(
                amendment.payload["s5_candidate_manifests"]
            ),
            "outcome_blind_attestation": dict(
                amendment.payload["outcome_blind_attestation"]
            ),
        },
        "plan": {
            "path": str(plan.path),
            "sha256": plan.digest,
            "sidecar": str(plan.sidecar),
            "sidecar_sha256": base.sha256(plan.sidecar) if plan.sidecar else None,
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
    base._atomic_json(plan.output_path, payload)
    base._atomic_text(
        _sidecar_path(plan.output_path), base.sha256(plan.output_path) + "\n"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or execute the outcome-blind Ethnobotany S5 receipt amendment"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze-amendment",
        help="freeze split receipts, selectors, and post-training S5 manifest hashes",
    )
    freeze.add_argument("--amendment", type=Path, required=True)
    select = subparsers.add_parser(
        "select", help=f"write validation-only {AMENDED_OUTPUT_NAME}"
    )
    select.add_argument("--amendment", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "freeze-amendment":
            path = freeze_amendment(arguments.amendment)
            print(path)
        elif arguments.command == "select":
            payload = select_development_model(arguments.amendment)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:  # pragma: no cover - argparse enforces the command set.
            raise base.AdapterSelectionError("unsupported command")
    except base.AdapterSelectionError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AMENDMENT_SCHEMA",
    "AMENDED_OUTPUT_NAME",
    "EXPECTED_ORIGINAL_SELECTOR_SHA256",
    "OUTPUT_SCHEMA",
    "freeze_amendment",
    "select_development_model",
    "validate_amendment_before_outcome_review",
]
