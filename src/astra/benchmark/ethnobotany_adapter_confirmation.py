#!/usr/bin/env python3
"""Freeze and score the paired seed-43 semantic-adapter confirmation.

The confirmation has exactly two arms: the existing trainable linear semantic
adapter and the residual-MLP adapter selected during seed-42 development.  A
plan must be frozen before any of the ten validation-only candidate manifests
exists.  Selection uses only inner-validation AUPRC (earlier epoch on an exact
tie) and commits the validation-selected F1 threshold.  Only a verified,
immutable selection may materialise and score the five outer folds, and the
score command always reserves all ten arm/fold evaluations before the first
outer prediction is computed.

This module is intentionally separate from the frozen development selectors
and model runners.  It does not train models and offers no partial-fold outer
scoring API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .ethnobotany_refined_runner import (
    CANDIDATE_SCHEMA,
    CONFIG_SCHEMA,
    CHECKPOINT_EPOCHS,
    derive_training_validation_rows,
    validate_candidate_config,
)
from .ethnobotany_refined_feature_receipt import (
    FeatureDeepAuditReceiptError,
    validate_feature_deep_audit_receipt,
)
from .metrics import (
    BenchmarkAuditError,
    PREDICTION_FIELDS,
    evaluate_test_scores,
    evaluate_validation_scores,
    load_predictions,
    save_predictions,
    select_f1_threshold,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .backend_paths import ETHNOBOTANY_BACKEND_ROOT


PLAN_SCHEMA = "astra-ethnobotany-adapter-confirmation-plan/v1"
SELECTION_SCHEMA = "astra-ethnobotany-adapter-confirmation-selection/v1"
SUMMARY_SCHEMA = "astra-ethnobotany-adapter-confirmation-summary/v1"
PLAN_HASH_SUFFIX = ".sha256"
SELECTION_HASH_SUFFIX = ".sha256"
SUMMARY_HASH_SUFFIX = ".sha256"
DATASET = "ethnobotany"
SEED = 43
ARMS = ("baseline_linear", "residual_mlp")
FOLDS = (1, 2, 3, 4, 5)
RECIPES = {
    "root": {"folds": (1, 2), "learning_rate": 0.0001, "clique": "legacy_prefix"},
    "lr3e4": {"folds": (3, 4), "learning_rate": 0.0003, "clique": "legacy_prefix"},
    "balanced_root": {"folds": (5,), "learning_rate": 0.0001, "clique": "balanced"},
}
EXPECTED_CONFIG_IDS = {
    "baseline_linear": {
        "root": "s4a_linear64_root",
        "lr3e4": "s4a_linear64_lr3e4",
        "balanced_root": "s4a_linear64_balanced_root",
    },
    "residual_mlp": {
        "root": "s5_residual_mlp_root",
        "lr3e4": "s5_residual_mlp_lr3e4",
        "balanced_root": "s5_residual_mlp_balanced_root",
    },
}
EXPECTED_MODEL_CLASS = {
    "baseline_linear": "ASTRARefined",
    "residual_mlp": "ASTRASemanticRefined",
}
EXPECTED_ADAPTER = {
    "baseline_linear": "linear",
    "residual_mlp": "residual_mlp",
}
FEATURE_ENTITIES = ("herb", "compound", "protein", "disease")
METRIC_ATOL = 1e-12
SCORE_ATOL = 1e-12
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_EVALUATION_ARTIFACT = re.compile(
    r"(^|[_.-])(outer(?:[_-]?test)?|test)([_.-]|$)", re.IGNORECASE
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parent
SELECTOR_PATH = Path(__file__).resolve()
METRICS_PATH = SELECTOR_PATH.with_name("metrics.py")
PROTOCOL_PATH = SELECTOR_PATH.with_name("protocol.py")
EXPECTED_ARM_CODE = {
    "baseline_linear": {
        "model": ETHNOBOTANY_BACKEND_ROOT / "ASTRARefined.py",
        "runner": SELECTOR_PATH.with_name("ethnobotany_refined_runner.py"),
    },
    "residual_mlp": {
        "model": ETHNOBOTANY_BACKEND_ROOT / "ASTRASemanticRefined.py",
        "runner": SELECTOR_PATH.with_name("ethnobotany_semantic_adapter_runner.py"),
    },
}
CONFIG_ROOTS = {
    "baseline_linear": PROJECT_ROOT
    / "experiments"
    / "08_model_selection"
    / "ethnobotany_semantic_adapter"
    / "configs"
    / "refined",
    "residual_mlp": PROJECT_ROOT
    / "experiments"
    / "08_model_selection"
    / "ethnobotany_semantic_adapter"
    / "configs",
}


class AdapterConfirmationError(RuntimeError):
    """Raised when confirmation isolation, provenance, or scoring fails."""


def sha256(path: Path | str) -> str:
    path = Path(path)
    if not path.is_file():
        raise AdapterConfirmationError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path | str, *, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterConfirmationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AdapterConfirmationError(f"{label} must contain one JSON object: {path}")
    return payload


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterConfirmationError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        raise AdapterConfirmationError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _resolve(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfirmationError(f"{label} path must be a non-empty string")
    supplied = Path(value)
    return (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()


def _validate_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise AdapterConfirmationError(f"{label} sha256 must be 64 lowercase hex digits")
    return value


def _artifact(record: object, *, label: str) -> dict[str, str]:
    record = _exact_keys(record, {"path", "sha256"}, label=label)
    path = _resolve(record["path"], label=label)
    expected = _validate_hash(record["sha256"], label=label)
    if path.is_symlink() or not path.is_file():
        raise AdapterConfirmationError(f"{label} must be a non-symlink regular file: {path}")
    observed = sha256(path)
    if observed != expected:
        raise AdapterConfirmationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": expected}


def _output_path(value: object, *, label: str) -> Path:
    path = _resolve(value, label=label)
    if path.is_symlink():
        raise AdapterConfirmationError(f"{label} must not be a symlink: {path}")
    return path


def _atomic_text(path: Path, content: str) -> None:
    path = path.resolve(strict=False)
    if path.exists():
        raise AdapterConfirmationError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AdapterConfirmationError(
                f"refusing to overwrite frozen artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _sidecar(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


def _read_sidecar(path: Path, suffix: str, *, label: str) -> str:
    sidecar = _sidecar(path, suffix)
    if sidecar.is_symlink() or not sidecar.is_file():
        raise AdapterConfirmationError(f"{label} sidecar is absent: {sidecar}")
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise AdapterConfirmationError(f"cannot read {label} sidecar: {error}") from error
    _validate_hash(expected, label=f"{label} sidecar")
    observed = sha256(path)
    if observed != expected:
        raise AdapterConfirmationError(
            f"{label} changed after freeze: expected {expected}, observed {observed}"
        )
    if path.stat().st_mtime_ns > sidecar.stat().st_mtime_ns:
        raise AdapterConfirmationError(f"{label} is newer than its sidecar")
    return expected


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _recipe_for_fold(fold: int) -> str:
    for recipe, details in RECIPES.items():
        if fold in details["folds"]:
            return recipe
    raise AdapterConfirmationError(f"fold has no frozen nuisance recipe: {fold}")


def _validate_config(
    record: object, *, arm: str, recipe: str
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact = _artifact(record, label=f"{arm} {recipe} config")
    raw = _load_json(Path(artifact["path"]), label=f"{arm} {recipe} config")
    try:
        config = validate_candidate_config(raw)
    except (RuntimeError, ValueError) as error:
        raise AdapterConfirmationError(
            f"{arm} {recipe} config contract failed: {error}"
        ) from error
    identity = config["config"]
    expected_id = EXPECTED_CONFIG_IDS[arm][recipe]
    if identity.get("id") != expected_id:
        raise AdapterConfirmationError(
            f"{arm} {recipe} config id must be {expected_id!r}"
        )
    training = config["training"]
    expected_training = {
        "epochs": 200,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "batch_size": 1000,
        "learning_rate": RECIPES[recipe]["learning_rate"],
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": 1.0,
    }
    if training != expected_training:
        raise AdapterConfirmationError(
            f"{arm} {recipe} training settings differ from the frozen recipe"
        )
    options = config["model_options"]
    if str(options.get("astra.semantic.adapter", "")).strip().lower() != EXPECTED_ADAPTER[arm]:
        raise AdapterConfirmationError(f"{arm} {recipe} semantic adapter is invalid")
    if int(options.get("astra.semantic.input_dim", -1)) != 64:
        raise AdapterConfirmationError(f"{arm} {recipe} semantic input must be 64")
    if bool(options.get("astra.semantic.freeze_input", True)):
        raise AdapterConfirmationError(f"{arm} {recipe} semantic inputs must be trainable")
    if str(options.get("astra.clique.sampling", "")).strip().lower() != RECIPES[recipe]["clique"]:
        raise AdapterConfirmationError(f"{arm} {recipe} clique recipe is invalid")
    if arm == "residual_mlp" and int(options.get("astra.semantic.mlp_hidden_dim", -1)) != 32:
        raise AdapterConfirmationError("residual_mlp hidden dimension must be 32")
    return artifact, config


def _paired_config_contracts(
    configs: Mapping[str, Mapping[str, dict[str, Any]]]
) -> None:
    for recipe in RECIPES:
        baseline = configs["baseline_linear"][recipe]
        adapter = configs["residual_mlp"][recipe]
        if baseline["training"] != adapter["training"]:
            raise AdapterConfirmationError(f"paired {recipe} training settings differ")
        left = dict(baseline["model_options"])
        right = dict(adapter["model_options"])
        left.pop("astra.semantic.adapter", None)
        right.pop("astra.semantic.adapter", None)
        right.pop("astra.semantic.mlp_hidden_dim", None)
        if left != right:
            raise AdapterConfirmationError(
                f"paired {recipe} model options differ outside adapter architecture"
            )


def _validate_feature_contract(
    raw: object, *, protocol_record: Mapping[str, str]
) -> dict[str, Any]:
    raw = _exact_keys(
        raw,
        {
            "selected_features",
            "semantic_reduction",
            "semantic_deep_audit_receipt",
            "feature_deep_audit_receipt",
        },
        label="features",
    )
    selected_raw = _exact_keys(
        raw["selected_features"], set(FEATURE_ENTITIES), label="selected_features"
    )
    selected = {
        entity: _artifact(selected_raw[entity], label=f"selected {entity} feature")
        for entity in FEATURE_ENTITIES
    }
    semantic = _artifact(raw["semantic_reduction"], label="semantic reduction")
    semantic_receipt = _artifact(
        raw["semantic_deep_audit_receipt"], label="semantic deep-audit receipt"
    )
    feature_receipt = _artifact(
        raw["feature_deep_audit_receipt"], label="feature deep-audit receipt"
    )
    receipt = _load_json(feature_receipt["path"], label="feature deep-audit receipt")
    if receipt.get("schema") != "astra-ethnobotany-refined-feature-deep-audit-receipt/v1":
        raise AdapterConfirmationError("feature deep-audit receipt schema is invalid")
    for key, expected in (
        ("protocol", protocol_record),
        ("semantic_reduction_manifest", semantic),
        ("semantic_deep_audit_receipt", semantic_receipt),
    ):
        observed = receipt.get(key)
        if not isinstance(observed, Mapping) or {
            "path": str(Path(str(observed.get("path", ""))).resolve()),
            "sha256": observed.get("sha256"),
        } != expected:
            raise AdapterConfirmationError(
                f"feature deep-audit receipt differs at {key}"
            )
    canonical = receipt.get("canonical_features")
    if not isinstance(canonical, Mapping):
        raise AdapterConfirmationError("feature deep-audit receipt has no canonical features")
    for entity, expected in selected.items():
        record = canonical.get(entity)
        matrix = record.get("matrix") if isinstance(record, Mapping) else None
        if not isinstance(matrix, Mapping) or {
            "path": str(Path(str(matrix.get("path", ""))).resolve()),
            "sha256": matrix.get("sha256"),
        } != expected:
            raise AdapterConfirmationError(
                f"selected {entity} feature differs from deep-audit receipt"
            )
    return {
        "selected_features": selected,
        "semantic_reduction": semantic,
        "semantic_deep_audit_receipt": semantic_receipt,
        "feature_deep_audit_receipt": feature_receipt,
    }


def _validate_plan_payload(
    plan_path: Path,
    payload: Mapping[str, Any],
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema",
            "experiment_id",
            "dataset",
            "seed",
            "protocol",
            "support_code",
            "arms",
            "recipes",
            "candidates",
            "features",
            "selection",
            "comparison",
            "outputs",
        },
        label="confirmation plan",
    )
    if payload.get("schema") != PLAN_SCHEMA:
        raise AdapterConfirmationError(f"plan schema must be {PLAN_SCHEMA!r}")
    if not isinstance(payload.get("experiment_id"), str) or not payload["experiment_id"]:
        raise AdapterConfirmationError("experiment_id must be a non-empty string")
    if payload.get("dataset") != DATASET or payload.get("seed") != SEED:
        raise AdapterConfirmationError("confirmation plan is fixed to Ethnobotany seed 43")
    protocol_record = _artifact(payload["protocol"], label="protocol manifest")
    protocol = protocol_loader(Path(protocol_record["path"]))
    if (
        protocol.dataset != DATASET
        or int(protocol.seed) != SEED
        or int(protocol.n_folds) != 5
        or not math.isclose(
            float(protocol.validation_fraction), 0.1, rel_tol=0.0, abs_tol=1e-15
        )
        or str(protocol.pair_order) != "seeded"
    ):
        raise AdapterConfirmationError("loaded protocol is not five-fold Ethnobotany seed 43")

    support_raw = _exact_keys(
        payload["support_code"], {"selector", "metrics", "protocol"}, label="support_code"
    )
    support_expected = {
        "selector": SELECTOR_PATH,
        "metrics": METRICS_PATH,
        "protocol": PROTOCOL_PATH,
    }
    support: dict[str, dict[str, str]] = {}
    for role, expected_path in support_expected.items():
        record = _artifact(support_raw[role], label=f"support code {role}")
        if Path(record["path"]) != expected_path:
            raise AdapterConfirmationError(f"support code {role} path is not canonical")
        support[role] = record

    arms_raw = _exact_keys(payload["arms"], set(ARMS), label="arms")
    arms: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        entry = _exact_keys(arms_raw[arm], {"model", "runner"}, label=f"arm {arm}")
        arms[arm] = {}
        for role in ("model", "runner"):
            record = _artifact(entry[role], label=f"arm {arm} {role}")
            if Path(record["path"]) != EXPECTED_ARM_CODE[arm][role].resolve():
                raise AdapterConfirmationError(f"arm {arm} {role} path is not canonical")
            arms[arm][role] = record

    recipes_raw = _exact_keys(payload["recipes"], set(RECIPES), label="recipes")
    configs: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    config_records: dict[str, dict[str, dict[str, str]]] = {arm: {} for arm in ARMS}
    for recipe, expected_recipe in RECIPES.items():
        entry = _exact_keys(
            recipes_raw[recipe], {"folds", "configs"}, label=f"recipe {recipe}"
        )
        if entry["folds"] != list(expected_recipe["folds"]):
            raise AdapterConfirmationError(f"recipe {recipe} folds are not frozen")
        config_raw = _exact_keys(entry["configs"], set(ARMS), label=f"recipe {recipe} configs")
        for arm in ARMS:
            artifact, config = _validate_config(config_raw[arm], arm=arm, recipe=recipe)
            config_records[arm][recipe] = artifact
            configs[arm][recipe] = config
    _paired_config_contracts(configs)

    candidates_raw = _exact_keys(payload["candidates"], set(ARMS), label="candidates")
    candidates: dict[str, dict[int, Path]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        folds_raw = _exact_keys(
            candidates_raw[arm], {str(fold) for fold in FOLDS}, label=f"{arm} candidates"
        )
        for fold in FOLDS:
            record = _exact_keys(
                folds_raw[str(fold)],
                {"path", "must_not_exist_at_freeze"},
                label=f"{arm} fold {fold} candidate",
            )
            if record["must_not_exist_at_freeze"] is not True:
                raise AdapterConfirmationError(
                    f"{arm} fold {fold} candidate must be absent at freeze"
                )
            candidates[arm][fold] = _output_path(
                record["path"], label=f"{arm} fold {fold} candidate"
            )
    candidate_paths = [path for arm in ARMS for path in candidates[arm].values()]
    if len(set(candidate_paths)) != 10:
        raise AdapterConfirmationError("candidate manifest paths must be ten unique paths")

    features = _validate_feature_contract(payload["features"], protocol_record=protocol_record)
    receipt_payload = _load_json(
        features["feature_deep_audit_receipt"]["path"],
        label="feature deep-audit receipt",
    )
    try:
        validate_feature_deep_audit_receipt(
            receipt_path=features["feature_deep_audit_receipt"]["path"],
            protocol_manifest=protocol_record["path"],
            protocol=protocol,
            data_dir=Path(str(receipt_payload.get("data_dir", ""))),
            embedding_dir=Path(str(receipt_payload.get("embedding_dir", ""))),
            semantic_reduction_manifest=features["semantic_reduction"]["path"],
            semantic_deep_audit_receipt=features["semantic_deep_audit_receipt"]["path"],
        )
    except (FeatureDeepAuditReceiptError, OSError, ValueError) as error:
        raise AdapterConfirmationError(
            f"feature deep-audit receipt revalidation failed: {error}"
        ) from error
    selection_raw = _exact_keys(
        payload["selection"],
        {"metric", "epochs", "epoch_tie_break", "threshold_rule"},
        label="selection rule",
    )
    expected_selection = {
        "metric": "Validation_AUPRC",
        "epochs": list(CHECKPOINT_EPOCHS),
        "epoch_tie_break": "lowest_epoch",
        "threshold_rule": "validation_maximum_f1_lowest_threshold",
    }
    if dict(selection_raw) != expected_selection:
        raise AdapterConfirmationError("selection rule differs from the fixed contract")
    comparison_raw = _exact_keys(
        payload["comparison"],
        {
            "primary_metric",
            "minimum_mean_improvement",
            "minimum_improved_folds",
            "improved_fold_rule",
        },
        label="comparison rule",
    )
    expected_comparison = {
        "primary_metric": "AUPRC",
        "minimum_mean_improvement": 0.001,
        "minimum_improved_folds": 3,
        "improved_fold_rule": "delta_gt_1e-12",
    }
    if dict(comparison_raw) != expected_comparison:
        raise AdapterConfirmationError("comparison rule differs from the fixed contract")

    outputs_raw = _exact_keys(
        payload["outputs"], {"selection", "outer_root", "summary"}, label="outputs"
    )
    selection_path = _output_path(outputs_raw["selection"], label="selection output")
    outer_root = _output_path(outputs_raw["outer_root"], label="outer output root")
    summary_path = _output_path(outputs_raw["summary"], label="outer summary")
    if summary_path != outer_root / "summary.json":
        raise AdapterConfirmationError("outer summary must be outer_root/summary.json")
    if selection_path == summary_path:
        raise AdapterConfirmationError("selection and summary outputs must differ")
    try:
        selection_path.relative_to(outer_root)
    except ValueError:
        pass
    else:
        raise AdapterConfirmationError("selection output must be outside outer_root")
    for candidate_path in candidate_paths:
        try:
            selection_path.relative_to(candidate_path.parent)
        except ValueError:
            pass
        else:
            raise AdapterConfirmationError("selection output is inside a candidate directory")
        try:
            outer_root.relative_to(candidate_path.parent)
        except ValueError:
            pass
        else:
            raise AdapterConfirmationError("outer output is inside a candidate directory")

    return {
        "path": plan_path,
        "payload": dict(payload),
        "protocol": protocol,
        "protocol_record": protocol_record,
        "support_code": support,
        "arms": arms,
        "configs": configs,
        "config_records": config_records,
        "candidates": candidates,
        "features": features,
        "selection_path": selection_path,
        "outer_root": outer_root,
        "summary_path": summary_path,
    }


def _load_frozen_plan(
    plan_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    path = Path(plan_path).resolve(strict=True)
    digest = _read_sidecar(path, PLAN_HASH_SUFFIX, label="confirmation plan")
    context = _validate_plan_payload(
        path, _load_json(path, label="confirmation plan"), protocol_loader=protocol_loader
    )
    context["sha256"] = digest
    return context


def _freeze_plan_with_protocol_loader(
    plan_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    path = Path(plan_path).resolve(strict=True)
    sidecar = _sidecar(path, PLAN_HASH_SUFFIX)
    if sidecar.exists():
        raise AdapterConfirmationError(f"plan is already frozen: {sidecar}")
    context = _validate_plan_payload(
        path, _load_json(path, label="confirmation plan"), protocol_loader=protocol_loader
    )
    for arm in ARMS:
        for fold, candidate_path in context["candidates"][arm].items():
            if candidate_path.name != "candidate.json" or FORBIDDEN_EVALUATION_ARTIFACT.search(
                str(candidate_path)
            ):
                raise AdapterConfirmationError(
                    f"candidate path is not a validation-only candidate.json: {candidate_path}"
                )
            if candidate_path.parent.exists():
                raise AdapterConfirmationError(
                    f"candidate directory existed before plan freeze: "
                    f"{arm} fold {fold}: {candidate_path.parent}"
                )
    protected = (
        context["selection_path"],
        _sidecar(context["selection_path"], SELECTION_HASH_SUFFIX),
        context["outer_root"],
    )
    if any(output.exists() for output in protected):
        raise AdapterConfirmationError("confirmation output existed before plan freeze")
    digest = sha256(path)
    _atomic_text(sidecar, digest + "\n")
    return {
        "schema": PLAN_SCHEMA,
        "plan": str(path),
        "plan_sha256": digest,
        "candidate_manifests_absent": 10,
        "selection_absent": True,
        "outer_output_absent": True,
    }


def freeze_plan(plan_path: Path | str) -> dict[str, Any]:
    """Freeze the exact seed-43 plan before candidate training."""

    return _freeze_plan_with_protocol_loader(
        plan_path, protocol_loader=load_protocol_manifest
    )


def _reject_forbidden_candidate_files(candidate_manifest: Path) -> None:
    root = candidate_manifest.parent.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AdapterConfirmationError(
                f"candidate directory contains a symlink: {path}"
            )
        if path.is_file() and FORBIDDEN_EVALUATION_ARTIFACT.search(path.name):
            raise AdapterConfirmationError(
                f"candidate directory contains a forbidden test artifact: {path}"
            )


def _reject_forbidden_candidate_payload(value: object, *, path: str = "candidate") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text != "outer_evaluation_rows_loaded" and FORBIDDEN_EVALUATION_ARTIFACT.search(
                key_text
            ):
                raise AdapterConfirmationError(
                    f"candidate payload contains forbidden evaluation field: {child_path}"
                )
            _reject_forbidden_candidate_payload(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_candidate_payload(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        name = Path(value).name
        if FORBIDDEN_EVALUATION_ARTIFACT.search(name):
            raise AdapterConfirmationError(
                f"candidate payload references a forbidden evaluation artifact: {path}"
            )


def _candidate_artifact(
    record: object, *, candidate_root: Path, label: str
) -> dict[str, str]:
    record = _exact_keys(record, {"path", "sha256"}, label=label)
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise AdapterConfirmationError(f"{label} path must be candidate-relative")
    expected = _validate_hash(record["sha256"], label=label)
    path = (candidate_root / raw_path).resolve()
    try:
        path.relative_to(candidate_root.resolve())
    except ValueError as error:
        raise AdapterConfirmationError(f"{label} escapes candidate directory") from error
    if path.is_symlink() or not path.is_file():
        raise AdapterConfirmationError(f"{label} is absent or a symlink: {path}")
    observed = sha256(path)
    if observed != expected:
        raise AdapterConfirmationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": expected}


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _score_embeddings(
    u_embedding: np.ndarray, i_embedding: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    u = np.asarray(u_embedding, dtype=np.float64)
    i = np.asarray(i_embedding, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if u.ndim != 2 or i.ndim != 2 or u.shape[1] != i.shape[1]:
        raise AdapterConfirmationError("checkpoint u/i embeddings are incompatible")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise AdapterConfirmationError("evaluation pairs must have shape (n, 2)")
    if int(pairs.min()) < 0:
        raise AdapterConfirmationError("evaluation pairs contain negative indices")
    if int(pairs[:, 0].max()) >= len(u) or int(pairs[:, 1].max()) >= len(i):
        raise AdapterConfirmationError("evaluation pair exceeds embedding rows")
    logits = np.einsum("ij,ij->i", u[pairs[:, 0]], i[pairs[:, 1]])
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    if not np.isfinite(scores).all():
        raise AdapterConfirmationError("checkpoint emitted non-finite scores")
    return scores


def _same_artifact_record(observed: object, expected: Mapping[str, str]) -> bool:
    if not isinstance(observed, Mapping):
        return False
    try:
        path = Path(str(observed.get("path", ""))).resolve()
    except (OSError, RuntimeError):
        return False
    return path == Path(expected["path"]) and observed.get("sha256") == expected["sha256"]


def _validate_candidate_feature_binding(
    payload: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    features = payload.get("features")
    if not isinstance(features, Mapping):
        raise AdapterConfirmationError("candidate has no feature diagnostics")
    receipt = features.get("feature_deep_audit_receipt")
    if not _same_artifact_record(receipt, expected["feature_deep_audit_receipt"]):
        raise AdapterConfirmationError("candidate feature receipt differs from plan")
    semantic = features.get("semantic_reduction")
    if not isinstance(semantic, Mapping):
        raise AdapterConfirmationError("candidate semantic-reduction diagnostics are absent")
    if (
        Path(str(semantic.get("path", ""))).resolve()
        != Path(expected["semantic_reduction"]["path"])
        or semantic.get("sha256") != expected["semantic_reduction"]["sha256"]
    ):
        raise AdapterConfirmationError("candidate semantic reduction differs from plan")
    semantic_receipt = semantic.get("deep_audit_receipt")
    if not _same_artifact_record(
        semantic_receipt, expected["semantic_deep_audit_receipt"]
    ):
        raise AdapterConfirmationError("candidate semantic receipt differs from plan")
    selected = features.get("selected_features")
    if not isinstance(selected, Mapping) or set(selected) != set(FEATURE_ENTITIES):
        raise AdapterConfirmationError("candidate selected features are incomplete")
    for entity in FEATURE_ENTITIES:
        record = selected[entity]
        expected_record = expected["selected_features"][entity]
        if (
            not isinstance(record, Mapping)
            or Path(str(record.get("used_path", ""))).resolve()
            != Path(expected_record["path"])
            or record.get("used_sha256") != expected_record["sha256"]
        ):
            raise AdapterConfirmationError(
                f"candidate {entity} feature differs from plan"
            )


def _expected_runtime_subset(
    *, arm: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    training = config["training"]
    options = config["model_options"]
    return {
        "model_class": EXPECTED_MODEL_CLASS[arm],
        "seed": SEED,
        "max_epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "embedding_dimension": training["embedding_dimension"],
        "reg_u": training["reg_u"],
        "text_fusion_alpha": training["text_fusion_alpha"],
        "n_layers": int(options.get("astra.n_layers", 5)),
        "loss_kind": str(options.get("astra.loss.kind", "legacy")).strip().lower(),
        "loss_reduction": str(options.get("astra.loss.reduction", "sum")).strip().lower(),
        "gradient_clip_value": options.get("astra.grad.clip_value", 10.0),
        "gradient_clip_norm": options.get("astra.grad.clip_norm", 0.5),
        "protein_gate": bool(options.get("astra.protein_gate", False)),
        "dead_variables": str(options.get("astra.dead_variables", "preserve")).strip().lower(),
        "clique_sampling": str(options.get("astra.clique.sampling", "legacy_prefix")).strip().lower(),
        "clique_max_pairs_per_hyperedge": int(
            options.get("astra.clique.max_pairs_per_hyperedge", 100)
        ),
        "clique_max_directed_edges": int(
            options.get("astra.clique.max_directed_edges", 200_000)
        ),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "semantic_adapter": EXPECTED_ADAPTER[arm],
        "semantic_input_dim": 64,
        "semantic_freeze_input": False,
        "trace_gradients": True,
    }


def _validate_runtime_receipt(
    payload: Mapping[str, Any], *, arm: str, config: Mapping[str, Any]
) -> None:
    diagnostics = payload.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise AdapterConfirmationError("candidate training diagnostics are absent")
    resolved = diagnostics.get("resolved_hyperparameters")
    if not isinstance(resolved, Mapping):
        raise AdapterConfirmationError("candidate resolved hyperparameters are absent")
    for key, expected in _expected_runtime_subset(arm=arm, config=config).items():
        observed = resolved.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            raise AdapterConfirmationError(
                f"candidate runtime field {key} differs: {observed!r} vs {expected!r}"
            )
    if arm == "residual_mlp":
        if resolved.get("semantic_mlp_hidden_dim") != 32:
            raise AdapterConfirmationError("residual_mlp runtime hidden dimension is not 32")
        implementation = diagnostics.get("implementation")
        expected_implementation = {
            "model_module": "ASTRASemanticRefined",
            "runner_module": "astra.benchmark.ethnobotany_semantic_adapter_runner",
            "frozen_parent_model_module": "ASTRARefined",
        }
        if implementation != expected_implementation:
            raise AdapterConfirmationError(
                "residual_mlp runtime implementation receipt is invalid"
            )


def _load_embedding(record: Mapping[str, str], *, label: str) -> np.ndarray:
    try:
        array = np.load(record["path"], allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AdapterConfirmationError(f"cannot load {label}: {error}") from error
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise AdapterConfirmationError(f"{label} must be a numeric matrix")
    if not np.isfinite(array).all():
        raise AdapterConfirmationError(f"{label} contains NaN/Inf")
    return np.asarray(array)


def _choose_snapshot(snapshots: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not snapshots:
        raise AdapterConfirmationError("cannot select from an empty snapshot set")
    return sorted(
        snapshots,
        key=lambda record: (-float(record["validation_auprc"]), int(record["epoch"])),
    )[0]


def _audit_candidate(
    *,
    context: Mapping[str, Any],
    arm: str,
    fold: int,
) -> dict[str, Any]:
    manifest_path = context["candidates"][arm][fold]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AdapterConfirmationError(
            f"candidate manifest is absent or a symlink: {manifest_path}"
        )
    _reject_forbidden_candidate_files(manifest_path)
    payload = _load_json(manifest_path, label=f"{arm} fold {fold} candidate")
    _reject_forbidden_candidate_payload(payload)
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise AdapterConfirmationError(f"{arm} fold {fold} candidate schema is invalid")
    if payload.get("validation_only") is not True:
        raise AdapterConfirmationError(f"{arm} fold {fold} is not validation-only")
    if payload.get("outer_evaluation_rows_loaded") is not False:
        raise AdapterConfirmationError(f"{arm} fold {fold} loaded outer rows")
    if payload.get("fold") != fold:
        raise AdapterConfirmationError(f"{arm} candidate fold identity differs")
    recipe = _recipe_for_fold(fold)
    config = context["configs"][arm][recipe]
    config_record = context["config_records"][arm][recipe]
    if payload.get("config") != config["config"]:
        raise AdapterConfirmationError(f"{arm} fold {fold} config identity differs")
    config_source = payload.get("config_source")
    if (
        not isinstance(config_source, Mapping)
        or config_source.get("schema") != CONFIG_SCHEMA
        or Path(str(config_source.get("path", ""))).resolve()
        != Path(config_record["path"])
        or config_source.get("sha256") != config_record["sha256"]
    ):
        raise AdapterConfirmationError(f"{arm} fold {fold} config source differs")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or Path(str(protocol.get("path", ""))).resolve()
        != Path(context["protocol_record"]["path"])
        or protocol.get("sha256") != context["protocol_record"]["sha256"]
        or protocol.get("seed") != SEED
    ):
        raise AdapterConfirmationError(f"{arm} fold {fold} protocol differs")
    _validate_candidate_feature_binding(payload, expected=context["features"])
    _validate_runtime_receipt(payload, arm=arm, config=config)

    rows = derive_training_validation_rows(context["protocol"], fold)
    split = payload.get("split")
    expected_split = {
        "train_rows": len(rows.train_row_indices),
        "validation_rows": len(rows.validation_row_indices),
        "train_row_indices_sha256": _array_digest(rows.train_row_indices),
        "validation_row_indices_sha256": _array_digest(rows.validation_row_indices),
    }
    if split != expected_split:
        raise AdapterConfirmationError(f"{arm} fold {fold} split receipt differs")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != len(CHECKPOINT_EPOCHS):
        raise AdapterConfirmationError(f"{arm} fold {fold} snapshot grid is incomplete")
    if [snapshot.get("epoch") for snapshot in snapshots if isinstance(snapshot, Mapping)] != list(
        CHECKPOINT_EPOCHS
    ):
        raise AdapterConfirmationError(f"{arm} fold {fold} snapshot epochs differ")

    audited_snapshots: list[dict[str, Any]] = []
    root = manifest_path.parent
    for snapshot in snapshots:
        assert isinstance(snapshot, Mapping)
        epoch = int(snapshot["epoch"])
        predictions_record = _candidate_artifact(
            snapshot.get("validation_predictions"),
            candidate_root=root,
            label=f"{arm} fold {fold} epoch {epoch} validation predictions",
        )
        u_record = _candidate_artifact(
            snapshot.get("u_embedding"),
            candidate_root=root,
            label=f"{arm} fold {fold} epoch {epoch} u embedding",
        )
        i_record = _candidate_artifact(
            snapshot.get("i_embedding"),
            candidate_root=root,
            label=f"{arm} fold {fold} epoch {epoch} i embedding",
        )
        try:
            with np.load(predictions_record["path"], allow_pickle=False) as archive:
                if set(archive.files) != set(PREDICTION_FIELDS):
                    raise AdapterConfirmationError(
                        f"{arm} fold {fold} epoch {epoch} prediction fields differ"
                    )
            predictions = load_predictions(predictions_record["path"])
            predictions = validate_prediction_coverage(
                predictions_record["path"],
                expected_row_id=rows.validation_row_indices,
                expected_pair=rows.validation_pairs,
                expected_label=rows.validation_labels,
            )
            validation_metrics = evaluate_validation_scores(
                rows.validation_labels, predictions.score
            )
        except BenchmarkAuditError as error:
            raise AdapterConfirmationError(
                f"{arm} fold {fold} epoch {epoch} validation audit failed: {error}"
            ) from error
        u = _load_embedding(u_record, label="u embedding")
        i = _load_embedding(i_record, label="i embedding")
        if len(u) != int(context["protocol"].n_herbs) or len(i) != int(
            context["protocol"].n_diseases
        ):
            raise AdapterConfirmationError(
                f"{arm} fold {fold} epoch {epoch} embedding row count differs"
            )
        recomputed_scores = _score_embeddings(u, i, rows.validation_pairs)
        if not np.allclose(
            recomputed_scores,
            predictions.score,
            rtol=0.0,
            atol=SCORE_ATOL,
        ):
            raise AdapterConfirmationError(
                f"{arm} fold {fold} epoch {epoch} saved scores differ from embeddings"
            )
        observed_auprc = snapshot.get("validation_auprc")
        try:
            valid_metric = math.isclose(
                float(observed_auprc),
                float(validation_metrics["Validation_AUPRC"]),
                rel_tol=0.0,
                abs_tol=METRIC_ATOL,
            )
        except (TypeError, ValueError):
            valid_metric = False
        if not valid_metric:
            raise AdapterConfirmationError(
                f"{arm} fold {fold} epoch {epoch} validation AUPRC is inconsistent"
            )
        audited_snapshots.append(
            {
                "epoch": epoch,
                "validation_auprc": float(validation_metrics["Validation_AUPRC"]),
                "validation_predictions": predictions_record,
                "u_embedding": u_record,
                "i_embedding": i_record,
                "validation_threshold": select_f1_threshold(
                    rows.validation_labels, predictions.score
                ).as_dict(),
            }
        )
    selected = _choose_snapshot(audited_snapshots)
    return {
        "arm": arm,
        "fold": fold,
        "recipe": recipe,
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "config": config_record,
        "snapshots": audited_snapshots,
        "selected": selected,
    }


def _build_selection_payload(
    context: Mapping[str, Any], audited: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key = {(record["arm"], record["fold"]): record for record in audited}
    if set(by_key) != {(arm, fold) for arm in ARMS for fold in FOLDS}:
        raise AdapterConfirmationError("audited candidates do not cover both arms/five folds")
    folds: dict[str, Any] = {}
    for fold in FOLDS:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            record = by_key[(arm, fold)]
            selected = record["selected"]
            arms[arm] = {
                "recipe": record["recipe"],
                "candidate_manifest": record["manifest"],
                "config": record["config"],
                "selected_epoch": selected["epoch"],
                "selected_validation_auprc": selected["validation_auprc"],
                "frozen_f1_threshold": selected["validation_threshold"],
                "validation_predictions": selected["validation_predictions"],
                "u_embedding": selected["u_embedding"],
                "i_embedding": selected["i_embedding"],
            }
        folds[str(fold)] = {"fold": fold, "arms": arms}
    return {
        "schema": SELECTION_SCHEMA,
        "dataset": DATASET,
        "seed": SEED,
        "seed43_validation_used_to_choose_arm": False,
        "outer_rows_or_metrics_used": False,
        "plan": {
            "path": str(context["path"]),
            "sha256": context["sha256"],
            "schema": PLAN_SCHEMA,
        },
        "protocol": dict(context["protocol_record"]),
        "features": context["features"],
        "support_code": context["support_code"],
        "arm_code": context["arms"],
        "selection_rule": dict(context["payload"]["selection"]),
        "comparison_rule": dict(context["payload"]["comparison"]),
        "outer_policy": (
            "all ten arm/fold evaluations reserved before scoring; exactly one "
            "validation-selected embedding pair per arm/fold; no partial-fold API"
        ),
        "folds": folds,
    }


def _select_with_protocol_loader(
    plan_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    context = _load_frozen_plan(plan_path, protocol_loader=protocol_loader)
    selection_path = context["selection_path"]
    if selection_path.exists() or _sidecar(
        selection_path, SELECTION_HASH_SUFFIX
    ).exists():
        raise AdapterConfirmationError("confirmation selection already exists")
    if context["outer_root"].exists():
        raise AdapterConfirmationError("outer output exists before selection freeze")
    audited = [
        _audit_candidate(context=context, arm=arm, fold=fold)
        for arm in ARMS
        for fold in FOLDS
    ]
    payload = _build_selection_payload(context, audited)
    _atomic_json(selection_path, payload)
    digest = sha256(selection_path)
    _atomic_text(_sidecar(selection_path, SELECTION_HASH_SUFFIX), digest + "\n")
    return {**payload, "selection_sha256": digest}


def select_and_freeze(plan_path: Path | str) -> dict[str, Any]:
    """Audit all ten validation candidates and freeze both arm decisions."""

    return _select_with_protocol_loader(
        plan_path, protocol_loader=load_protocol_manifest
    )


def _load_verified_selection(
    plan_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    context = _load_frozen_plan(plan_path, protocol_loader=protocol_loader)
    selection_path = context["selection_path"]
    if selection_path.is_symlink() or not selection_path.is_file():
        raise AdapterConfirmationError(
            f"frozen confirmation selection is absent: {selection_path}"
        )
    selection_hash = _read_sidecar(
        selection_path, SELECTION_HASH_SUFFIX, label="confirmation selection"
    )
    observed = _load_json(selection_path, label="confirmation selection")
    audited = [
        _audit_candidate(context=context, arm=arm, fold=fold)
        for arm in ARMS
        for fold in FOLDS
    ]
    expected = _build_selection_payload(context, audited)
    if _canonical(observed) != _canonical(expected):
        raise AdapterConfirmationError(
            "frozen selection differs from deterministic validation replay"
        )
    return observed, selection_hash, context


def _reserve_outer_outputs(
    *, context: Mapping[str, Any], selection_hash: str
) -> dict[tuple[str, int], Path]:
    root = context["outer_root"]
    if root.exists() or root.is_symlink():
        raise AdapterConfirmationError(
            f"refusing repeated or partial outer scoring: {root}"
        )
    root.mkdir(parents=True, exist_ok=False)
    reservations = {(arm, fold): root / arm / f"fold_{fold}" for arm in ARMS for fold in FOLDS}
    try:
        _atomic_json(
            root / ".all_ten_evaluations_reserved.json",
            {
                "selection_sha256": selection_hash,
                "state": "all-ten-reserved-before-first-outer-score",
                "evaluations": [
                    {"arm": arm, "fold": fold} for arm in ARMS for fold in FOLDS
                ],
            },
        )
        for (arm, fold), output in reservations.items():
            output.mkdir(parents=True, exist_ok=False)
            _atomic_json(
                output / ".reserved_before_outer_score.json",
                {
                    "selection_sha256": selection_hash,
                    "arm": arm,
                    "fold": fold,
                    "state": "reserved-before-any-outer-score",
                },
            )
    except BaseException:
        # Deliberately retain reservations.  A failed reservation attempt must
        # never silently become a fresh opportunity to inspect outer results.
        raise
    if len(reservations) != 10 or not all(path.is_dir() for path in reservations.values()):
        raise AdapterConfirmationError("all ten outer outputs were not reserved")
    return reservations


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.isfinite(array).all():
        raise AdapterConfirmationError("aggregate requires five finite fold values")
    return {
        "fold_values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)),
    }


def _score_with_protocol_loader(
    plan_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    score_callback: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = _score_embeddings,
) -> dict[str, Any]:
    # Full plan, candidate, validation-prediction, embedding, and deterministic
    # selection replay completes before the first explicit fold_rows call.
    selection, selection_hash, context = _load_verified_selection(
        plan_path, protocol_loader=protocol_loader
    )
    reservations = _reserve_outer_outputs(
        context=context, selection_hash=selection_hash
    )

    computed: dict[int, dict[str, dict[str, Any]]] = {}
    score_calls = 0
    for fold in FOLDS:
        fold_rows = context["protocol"].fold_rows(fold)
        computed[fold] = {}
        for arm in ARMS:
            selected = selection["folds"][str(fold)]["arms"][arm]
            u_record = selected["u_embedding"]
            i_record = selected["i_embedding"]
            # Candidate replay already verified both hashes.  Verify again at
            # the sole outer invocation boundary in case bytes changed after
            # reservation.
            if sha256(u_record["path"]) != u_record["sha256"] or sha256(
                i_record["path"]
            ) != i_record["sha256"]:
                raise AdapterConfirmationError(
                    f"selected embeddings changed before outer score: {arm} fold {fold}"
                )
            u = _load_embedding(u_record, label=f"{arm} fold {fold} u embedding")
            i = _load_embedding(i_record, label=f"{arm} fold {fold} i embedding")
            scores = np.asarray(
                score_callback(u, i, fold_rows.test_pairs), dtype=np.float64
            )
            score_calls += 1
            if scores.shape != (len(fold_rows.test_pairs),) or not np.isfinite(scores).all():
                raise AdapterConfirmationError(
                    f"outer scorer returned invalid scores: {arm} fold {fold}"
                )
            threshold_record = selected["frozen_f1_threshold"]
            try:
                threshold = float(threshold_record["Validation_Selected_Threshold"])
                metrics = evaluate_test_scores(
                    fold_rows.test_labels, scores, threshold=threshold
                )
            except (KeyError, TypeError, ValueError, BenchmarkAuditError) as error:
                raise AdapterConfirmationError(
                    f"outer metric evaluation failed: {arm} fold {fold}: {error}"
                ) from error
            computed[fold][arm] = {
                "scores": scores,
                "row_id": np.asarray(fold_rows.test_row_indices),
                "pairs": np.asarray(fold_rows.test_pairs),
                "labels": np.asarray(fold_rows.test_labels),
                "metrics": {
                    **metrics,
                    "Dataset": DATASET,
                    "Seed": SEED,
                    "Fold": fold,
                    "Arm": arm,
                    "Recipe": selected["recipe"],
                    "Selected_Epoch": selected["selected_epoch"],
                    "Selected_Validation_AUPRC": selected[
                        "selected_validation_auprc"
                    ],
                    "Decision_Frozen_Before_Test": True,
                    "Test_Models_Evaluated": 1,
                    "Unselected_Test_Models_Evaluated": 0,
                    "Selection_SHA256": selection_hash,
                },
            }
    if score_calls != 10:
        raise AdapterConfirmationError(
            f"outer scorer must be invoked exactly ten times, got {score_calls}"
        )

    metric_names = ("AUROC", "AUPRC", "F1")
    per_fold: dict[str, Any] = {}
    for fold in FOLDS:
        arm_metrics = {
            arm: computed[fold][arm]["metrics"] for arm in ARMS
        }
        paired = {
            metric: float(
                arm_metrics["residual_mlp"][metric]
                - arm_metrics["baseline_linear"][metric]
            )
            for metric in metric_names
        }
        per_fold[str(fold)] = {
            "fold": fold,
            "arms": arm_metrics,
            "paired_delta_residual_mlp_minus_baseline": paired,
        }
    aggregate_arms = {
        arm: {
            metric: _mean_std(
                [float(per_fold[str(fold)]["arms"][arm][metric]) for fold in FOLDS]
            )
            for metric in metric_names
        }
        for arm in ARMS
    }
    aggregate_paired = {
        metric: {
            **_mean_std(
                [
                    float(
                        per_fold[str(fold)][
                            "paired_delta_residual_mlp_minus_baseline"
                        ][metric]
                    )
                    for fold in FOLDS
                ]
            ),
            "improved_folds": int(
                sum(
                    per_fold[str(fold)][
                        "paired_delta_residual_mlp_minus_baseline"
                    ][metric]
                    > METRIC_ATOL
                    for fold in FOLDS
                )
            ),
        }
        for metric in metric_names
    }
    comparison = context["payload"]["comparison"]
    primary = str(comparison["primary_metric"])
    primary_delta = aggregate_paired[primary]
    gate = {
        "primary_metric": primary,
        "observed_mean_improvement": primary_delta["mean"],
        "required_mean_improvement": float(
            comparison["minimum_mean_improvement"]
        ),
        "observed_improved_folds": primary_delta["improved_folds"],
        "required_improved_folds": int(comparison["minimum_improved_folds"]),
        "mean_gate_passed": bool(
            primary_delta["mean"]
            >= float(comparison["minimum_mean_improvement"])
        ),
        "fold_gate_passed": bool(
            primary_delta["improved_folds"]
            >= int(comparison["minimum_improved_folds"])
        ),
    }
    gate["confirmed"] = bool(
        gate["mean_gate_passed"] and gate["fold_gate_passed"]
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "dataset": DATASET,
        "seed": SEED,
        "selection": {
            "path": str(context["selection_path"]),
            "sha256": selection_hash,
        },
        "protocol": dict(context["protocol_record"]),
        "outer_score_calls": score_calls,
        "all_ten_reserved_before_scoring": True,
        "partial_scoring_api": False,
        "threshold_source": "each arm/fold inner-validation maximum F1",
        "per_fold": per_fold,
        "aggregate": {
            "arms": aggregate_arms,
            "paired_delta_residual_mlp_minus_baseline": aggregate_paired,
            "standard_deviation": (
                "sample standard deviation (ddof=1) is manuscript-facing; "
                "population standard deviation is also recorded"
            ),
        },
        "precommitted_confirmation_gate": gate,
    }

    # No outer artifact is published until all ten score computations and all
    # aggregate calculations have succeeded in memory.
    published_artifacts: dict[str, Any] = {}
    for fold in FOLDS:
        published_artifacts[str(fold)] = {}
        for arm in ARMS:
            record = computed[fold][arm]
            output = reservations[(arm, fold)]
            prediction_path = save_predictions(
                output / "test_predictions.npz",
                row_id=record["row_id"],
                pair=record["pairs"],
                label=record["labels"],
                score=record["scores"],
            )
            metrics_path = output / "metrics.json"
            binding_path = output / "selection_binding.json"
            _atomic_json(metrics_path, record["metrics"])
            _atomic_json(
                binding_path,
                {
                    "selection_path": str(context["selection_path"]),
                    "selection_sha256": selection_hash,
                    "arm": arm,
                    "fold": fold,
                    "selected_epoch": selection["folds"][str(fold)]["arms"][arm][
                        "selected_epoch"
                    ],
                    "u_embedding": selection["folds"][str(fold)]["arms"][arm][
                        "u_embedding"
                    ],
                    "i_embedding": selection["folds"][str(fold)]["arms"][arm][
                        "i_embedding"
                    ],
                    "test_predictions": {
                        "path": str(prediction_path.resolve()),
                        "sha256": sha256(prediction_path),
                    },
                    "outer_score_calls_for_this_arm_fold": 1,
                },
            )
            published_artifacts[str(fold)][arm] = {
                "test_predictions": _record_for(prediction_path),
                "metrics": _record_for(metrics_path),
                "selection_binding": _record_for(binding_path),
            }
    summary["artifacts"] = published_artifacts
    _atomic_json(context["summary_path"], summary)
    summary_digest = sha256(context["summary_path"])
    _atomic_text(
        _sidecar(context["summary_path"], SUMMARY_HASH_SUFFIX),
        summary_digest + "\n",
    )
    return summary


def score_outer_confirmation(plan_path: Path | str) -> dict[str, Any]:
    """Score all ten frozen arm/fold decisions exactly once."""

    return _score_with_protocol_loader(
        plan_path, protocol_loader=load_protocol_manifest
    )


def _record_for(path: Path | str) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved)}


def initialize_plan(
    *,
    plan_path: Path | str,
    protocol_manifest: Path | str,
    feature_deep_audit_receipt: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Write the exact seed-43 plan; ``freeze-plan`` must still be called."""

    plan_path = Path(plan_path).resolve(strict=False)
    output_root = Path(output_root).resolve(strict=False)
    if plan_path.exists() or _sidecar(plan_path, PLAN_HASH_SUFFIX).exists():
        raise AdapterConfirmationError(f"plan output already exists: {plan_path}")
    protocol_record = _record_for(protocol_manifest)
    receipt_record = _record_for(feature_deep_audit_receipt)
    receipt = _load_json(receipt_record["path"], label="feature deep-audit receipt")
    receipt_protocol = receipt.get("protocol")
    if not isinstance(receipt_protocol, Mapping) or {
        "path": str(Path(str(receipt_protocol.get("path", ""))).resolve()),
        "sha256": receipt_protocol.get("sha256"),
    } != protocol_record:
        raise AdapterConfirmationError(
            "feature deep-audit receipt is not bound to the supplied protocol"
        )
    canonical = receipt.get("canonical_features")
    if not isinstance(canonical, Mapping):
        raise AdapterConfirmationError("feature receipt lacks canonical features")
    selected: dict[str, dict[str, str]] = {}
    for entity in FEATURE_ENTITIES:
        entity_record = canonical.get(entity)
        matrix = entity_record.get("matrix") if isinstance(entity_record, Mapping) else None
        if not isinstance(matrix, Mapping):
            raise AdapterConfirmationError(f"feature receipt lacks {entity} matrix")
        selected[entity] = _record_for(Path(str(matrix.get("path", ""))))
        if selected[entity]["sha256"] != matrix.get("sha256"):
            raise AdapterConfirmationError(f"feature receipt {entity} hash is stale")
    semantic_record = receipt.get("semantic_reduction_manifest")
    semantic_receipt_record = receipt.get("semantic_deep_audit_receipt")
    if not isinstance(semantic_record, Mapping) or not isinstance(
        semantic_receipt_record, Mapping
    ):
        raise AdapterConfirmationError("feature receipt lacks semantic provenance")
    semantic = _record_for(Path(str(semantic_record.get("path", ""))))
    semantic_receipt = _record_for(Path(str(semantic_receipt_record.get("path", ""))))
    if semantic["sha256"] != semantic_record.get("sha256") or semantic_receipt[
        "sha256"
    ] != semantic_receipt_record.get("sha256"):
        raise AdapterConfirmationError("feature receipt semantic provenance is stale")

    recipes: dict[str, Any] = {}
    candidates: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for recipe, details in RECIPES.items():
        config_records: dict[str, Any] = {}
        for arm in ARMS:
            config_id = EXPECTED_CONFIG_IDS[arm][recipe]
            config_path = CONFIG_ROOTS[arm] / f"{config_id}.json"
            config_records[arm] = _record_for(config_path)
            for fold in details["folds"]:
                candidates[arm][str(fold)] = {
                    "path": str(
                        output_root
                        / "candidates"
                        / config_id
                        / f"fold_{fold}"
                        / "candidate.json"
                    ),
                    "must_not_exist_at_freeze": True,
                }
        recipes[recipe] = {
            "folds": list(details["folds"]),
            "configs": config_records,
        }
    payload = {
        "schema": PLAN_SCHEMA,
        "experiment_id": "ethnobotany_seed43_cpfix_astra_v3_confirmation_v1",
        "dataset": DATASET,
        "seed": SEED,
        "protocol": protocol_record,
        "support_code": {
            "selector": _record_for(SELECTOR_PATH),
            "metrics": _record_for(METRICS_PATH),
            "protocol": _record_for(PROTOCOL_PATH),
        },
        "arms": {
            arm: {
                role: _record_for(path)
                for role, path in EXPECTED_ARM_CODE[arm].items()
            }
            for arm in ARMS
        },
        "recipes": recipes,
        "candidates": candidates,
        "features": {
            "selected_features": selected,
            "semantic_reduction": semantic,
            "semantic_deep_audit_receipt": semantic_receipt,
            "feature_deep_audit_receipt": receipt_record,
        },
        "selection": {
            "metric": "Validation_AUPRC",
            "epochs": list(CHECKPOINT_EPOCHS),
            "epoch_tie_break": "lowest_epoch",
            "threshold_rule": "validation_maximum_f1_lowest_threshold",
        },
        "comparison": {
            "primary_metric": "AUPRC",
            "minimum_mean_improvement": 0.001,
            "minimum_improved_folds": 3,
            "improved_fold_rule": "delta_gt_1e-12",
        },
        "outputs": {
            "selection": str(output_root / "confirmation_selection.json"),
            "outer_root": str(output_root / "outer_test"),
            "summary": str(output_root / "outer_test" / "summary.json"),
        },
    }
    _atomic_json(plan_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialise = subparsers.add_parser(
        "init-plan", help="write the exact unfrozen seed-43 confirmation plan"
    )
    initialise.add_argument("--plan", type=Path, required=True)
    initialise.add_argument("--protocol-manifest", type=Path, required=True)
    initialise.add_argument("--feature-deep-audit-receipt", type=Path, required=True)
    initialise.add_argument("--output-root", type=Path, required=True)
    freeze = subparsers.add_parser(
        "freeze-plan", help="freeze the plan before any candidate exists"
    )
    freeze.add_argument("--plan", type=Path, required=True)
    select = subparsers.add_parser(
        "select", help="audit validation-only candidates and freeze both arms"
    )
    select.add_argument("--plan", type=Path, required=True)
    score = subparsers.add_parser(
        "score", help="reserve and score all ten outer evaluations exactly once"
    )
    score.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-plan":
            payload = initialize_plan(
                plan_path=args.plan,
                protocol_manifest=args.protocol_manifest,
                feature_deep_audit_receipt=args.feature_deep_audit_receipt,
                output_root=args.output_root,
            )
        elif args.command == "freeze-plan":
            payload = freeze_plan(args.plan)
        elif args.command == "select":
            payload = select_and_freeze(args.plan)
        else:
            payload = score_outer_confirmation(args.plan)
    except AdapterConfirmationError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
