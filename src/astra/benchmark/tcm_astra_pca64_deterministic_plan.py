#!/usr/bin/env python3
"""Freeze the deterministic validation-only AE64 versus PCA64 control.

The control arm is the already selected winner from the deterministic-v3
semantic-adapter stage.  Exactly one PCA64 candidate is executable per fold,
with the selected adapter, epoch budget, learning rate, protocol, runtime
device, and TensorFlow graph-seed contract held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import load_protocol_manifest
from . import tcm_astra_semantic_adapter_runner as adapter_base_runner
from .tcm_astra_semantic_adapter_deterministic_plan import (
    PLAN_SCHEMA as PARENT_PLAN_SCHEMA,
)
from .tcm_astra_semantic_adapter_deterministic_runner import (
    AUDIT_SCHEMA,
    RUNNER_SCHEMA as PARENT_RUNNER_SCHEMA,
)
from .tcm_astra_semantic_adapter_deterministic_selection import (
    SELECTION_SCHEMA as PARENT_SELECTION_SCHEMA,
)
from .tcm_astra_semantic_adapter_pair_guard import (
    build_pair_receipt as build_adapter_pair_receipt,
)
from .tcm_astra_pca64_control_plan import MIN_PCA64_AUPRC_GAIN
from .tcm_feature_bundle import validate_tcm_feature_bundle
from .tcm_pca64_feature_bundle import validate_tcm_pca64_feature_bundle
from .backend_paths import TCM_BACKEND_ROOT


PLAN_SCHEMA = "astra-tcm-pca64-deterministic-plan/v2"
PLAN_STATUS = "frozen-after-v3-adapter-selection-before-pca64-validation"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
PARENT_COMPLETION_NAME = ".astra_semantic_adapter_deterministic_v3_complete"
PARENT_CANDIDATE_COMPLETION_NAME = ".astra_semantic_adapter_complete"

CONTROLLED_FEATURE_INPUT_KEYS = frozenset({"input/herb", "input/disease"})
EXPECTED_SHARED_INITIAL_KEYS = frozenset(
    {
        "input/herb",
        "input/disease",
        "input/compound",
        "input/protein",
        "weight/clique_att_c",
        "weight/clique_att_p",
        "weight/fusion_c",
        "weight/fusion_p",
        "weight/gating1",
        "weight/gating2",
        "weight/gating_bias1",
        "weight/gating_bias2",
        "weight/layer_1",
        "weight/layer_1_1",
        "weight/layer_1_2",
        "weight/layer_1_3",
        "weight/layer_1_4",
        "weight/layer_1_5",
        "weight/layer_2",
        "weight/layer_2_1",
        "weight/layer_2_2",
        "weight/layer_2_3",
        "weight/layer_2_4",
        "weight/layer_2_5",
        "weight/layer_3",
        "weight/layer_4",
        "weight/layer_5",
        "weight/layer_att_1",
        "weight/layer_att_2",
        "weight/layer_att_3",
        "weight/layer_att_4",
        "weight/layer_att_5",
    }
)
EXPECTED_RESIDUAL_ADAPTER_KEYS = frozenset(
    {
        "semantic_adapter_mlp_input_bias_disease",
        "semantic_adapter_mlp_input_bias_herb",
        "semantic_adapter_mlp_input_disease",
        "semantic_adapter_mlp_input_herb",
        "semantic_adapter_mlp_output_bias_disease",
        "semantic_adapter_mlp_output_bias_herb",
        "semantic_adapter_mlp_output_disease",
        "semantic_adapter_mlp_output_herb",
    }
)
EXPECTED_FIXED = {
    "batch_size": 2000,
    "embedding_dimension": 64,
    "reg_u": 0.001,
    "text_fusion_alpha": 1.0,
    "semantic_input_dim": 64,
    "semantic_mlp_hidden_dim": 32,
    "semantic_freeze_input": False,
}
EXPECTED_DETERMINISTIC_INITIALIZATION = {
    "schema": "astra-tcm-semantic-adapter-determinism/v1",
    "tensorflow_graph_seed": 42,
    "graph_must_be_empty_before_seed": True,
    "tensorflow_v1_behavior_enabled_before_seed": True,
    "seed_set_after_runtime_environment": True,
    "seed_set_before_model_import": True,
    "same_fold_shared_initialization_required": True,
}


class TCMPCA64DeterministicPlanError(RuntimeError):
    """Raised when the v2 feature-basis control cannot be frozen."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMPCA64DeterministicPlanError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TCMPCA64DeterministicPlanError(
            f"JSON root must be an object: {path}"
        )
    return payload


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise TCMPCA64DeterministicPlanError(f"artifact is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMPCA64DeterministicPlanError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except OSError as error:
        raise TCMPCA64DeterministicPlanError(f"{label} artifact is absent") from error
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise TCMPCA64DeterministicPlanError(f"{label} artifact changed")
    if "bytes" in record and path.stat().st_size != record.get("bytes"):
        raise TCMPCA64DeterministicPlanError(f"{label} byte count changed")
    return path


def validate_sidecar(path: Path, *, label: str) -> str:
    resolved = path.resolve(strict=True)
    digest = sha256(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMPCA64DeterministicPlanError(f"{label} SHA-256 sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, resolved.name]:
        raise TCMPCA64DeterministicPlanError(f"{label} SHA-256 sidecar differs")
    return digest


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_hash_mapping(
    value: object, *, expected_keys: frozenset[str], label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise TCMPCA64DeterministicPlanError(
            f"{label} key set differs: expected {len(expected_keys)} keys"
        )
    result = {str(key): str(digest) for key, digest in value.items()}
    if not all(_valid_hash(digest) for digest in result.values()):
        raise TCMPCA64DeterministicPlanError(f"{label} contains an invalid hash")
    return result


def expected_adapter_keys(adapter: str) -> frozenset[str]:
    if adapter == "identity":
        return frozenset()
    if adapter == "residual_mlp":
        return EXPECTED_RESIDUAL_ADAPTER_KEYS
    raise TCMPCA64DeterministicPlanError(f"unsupported adapter {adapter!r}")


def protocol_controls(protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "protocol_manifest",
        "protocol_input_manifest",
        "deduplicated_relation_view",
        "split_array_sha256",
        "split_rows",
        "runtime",
        "hyperparameters",
        "optimizer_steps",
    )
    controls = {key: protocol.get(key) for key in required}
    if any(value is None for value in controls.values()):
        raise TCMPCA64DeterministicPlanError(
            "parent AE64 protocol lacks a required runtime/split control"
        )
    for key in required[:-1]:
        if not isinstance(controls[key], Mapping):
            raise TCMPCA64DeterministicPlanError(
                f"parent AE64 protocol control {key} is invalid"
            )
        controls[key] = dict(controls[key])
    try:
        controls["optimizer_steps"] = int(controls["optimizer_steps"])
    except (TypeError, ValueError) as error:
        raise TCMPCA64DeterministicPlanError(
            "parent AE64 optimizer step count is invalid"
        ) from error
    if controls["optimizer_steps"] <= 0:
        raise TCMPCA64DeterministicPlanError(
            "parent AE64 optimizer step count must be positive"
        )
    controls["contract_sha256"] = object_sha256(controls)
    return controls


def _source_freeze() -> dict[str, dict[str, object]]:
    paths = {
        "model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapterDeterministic.py",
        "base_model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapter.py",
        "tensorflow_graph_seed": SOURCE_ROOT / "astra/tf_graph_seed.py",
        "deterministic_adapter_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_deterministic_runner.py",
        "adapter_base_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_runner.py",
        "central_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
        "parent_model": TCM_BACKEND_ROOT / "HGHDA.py",
        "astra_parent": TCM_BACKEND_ROOT / "ASTRA.py",
        "configuration": TCM_BACKEND_ROOT / "ASTRA.conf",
        "herb_recommender": TCM_BACKEND_ROOT / "base/herbRecommender.py",
        "rating": TCM_BACKEND_ROOT / "rating.py",
        "selective_tf_initialization": SOURCE_ROOT / "astra/tf_initialization.py",
        "protocol": SOURCE_ROOT / "astra/benchmark/protocol.py",
        "features": SOURCE_ROOT / "astra/benchmark/features.py",
        "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
        "parent_adapter_plan_generator": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_deterministic_plan.py",
        "parent_adapter_selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_deterministic_selection.py",
        "parent_adapter_pair_guard": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_pair_guard.py",
        "adapter_selection_base": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_selection.py",
        "ae64_bundle_validator": SOURCE_ROOT
        / "astra/benchmark/tcm_feature_bundle.py",
        "pca64_bundle_builder": SOURCE_ROOT
        / "astra/benchmark/tcm_pca64_feature_bundle.py",
        "pca64_runtime_base": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_runner.py",
        "pca64_plan_base": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_plan.py",
        "pca64_selection_base": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_selection.py",
        "plan_generator": Path(__file__).resolve(),
        "candidate_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_deterministic_runner.py",
        "pair_guard": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_deterministic_pair_guard.py",
        "selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_deterministic_selection.py",
        "prepare_launcher": PROJECT_ROOT
        / "tools/prepare_tcm_astra_pca64_deterministic_v2.sh",
        "run_launcher": SOURCE_ROOT / "herbhg/cli.py",
    }
    try:
        return {name: artifact(path) for name, path in paths.items()}
    except FileNotFoundError as error:
        raise TCMPCA64DeterministicPlanError(
            f"deterministic PCA64 v2 implementation is incomplete: {error}"
        ) from error


def _verify_source_freeze(records: object, *, label: str) -> None:
    if not isinstance(records, Mapping) or not records:
        raise TCMPCA64DeterministicPlanError(f"{label} source freeze is absent")
    for name, record in records.items():
        verify_artifact(record, label=f"{label} source {name}")


def _validate_parent_pair_receipts(
    *,
    selection: Mapping[str, Any],
    parent_plan_path: Path,
    candidate_root: Path,
) -> list[dict[str, object]]:
    records = selection.get("pair_receipts")
    if not isinstance(records, list) or len(records) != 5:
        raise TCMPCA64DeterministicPlanError(
            "deterministic adapter selection must bind five pair receipts"
        )
    by_fold = {
        int(record.get("fold", -1)): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(by_fold) != set(range(1, 6)):
        raise TCMPCA64DeterministicPlanError(
            "deterministic adapter pair receipt folds differ"
        )
    result: list[dict[str, object]] = []
    for fold in range(1, 6):
        record = by_fold[fold]
        path = verify_artifact(record, label=f"parent fold {fold} pair receipt")
        validate_sidecar(path, label=f"parent fold {fold} pair receipt")
        expected = build_adapter_pair_receipt(
            plan_path=parent_plan_path,
            candidate_root=candidate_root,
            fold=fold,
        )
        if dict(load_json(path)) != expected:
            raise TCMPCA64DeterministicPlanError(
                f"parent fold {fold} pair receipt no longer replays"
            )
        result.append(
            {
                "fold": fold,
                "receipt": artifact(path),
                "sidecar": artifact(path.with_suffix(path.suffix + ".sha256")),
            }
        )
    return result


def _fold_winner_reference(
    *,
    fold: int,
    fold_selection: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    parent_plan_path: Path,
    parent_plan_sha256: str,
) -> dict[str, Any]:
    winner = fold_selection.get("winner")
    if not isinstance(winner, Mapping) or int(winner.get("fold", -1)) != fold:
        raise TCMPCA64DeterministicPlanError(f"fold {fold} winner is invalid")
    if fold_selection.get("selected_candidate") != winner.get("id"):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected candidate differs from winner"
        )
    if fold_selection.get("selected_adapter") != winner.get("semantic_adapter"):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected adapter differs from winner"
        )
    adapter = str(winner.get("semantic_adapter", ""))
    adapter_keys = expected_adapter_keys(adapter)
    try:
        epochs = int(winner["epochs"])
        learning_rate = float(winner["learning_rate"])
        validation_auprc = float(winner["validation_auprc"])
        validation_auroc = float(winner["validation_auroc"])
        validation_threshold = float(winner["validation_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} winner hyperparameters or metrics are invalid"
        ) from error
    if (
        epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not all(
            math.isfinite(value)
            for value in (validation_auprc, validation_auroc, validation_threshold)
        )
    ):
        raise TCMPCA64DeterministicPlanError(f"fold {fold} winner values are invalid")

    artifact_records: dict[str, dict[str, object]] = {}
    for key in (
        "validation_predictions",
        "herb_embedding",
        "disease_embedding",
        "metrics",
        "protocol",
        "adapter_audit",
    ):
        path = verify_artifact(
            {"path": winner.get(key), "sha256": winner.get(f"{key}_sha256")},
            label=f"fold {fold} AE64 {key}",
        )
        artifact_records[key] = artifact(path)

    audit_path = Path(str(artifact_records["adapter_audit"]["path"]))
    protocol_path = Path(str(artifact_records["protocol"]["path"]))
    metrics_path = Path(str(artifact_records["metrics"]["path"]))
    candidate_root = audit_path.parent
    completion_path = candidate_root / PARENT_CANDIDATE_COMPLETION_NAME
    if not completion_path.is_file():
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected AE64 completion marker is absent"
        )
    try:
        adapter_base_runner._reject_outer_artifacts(candidate_root)
    except Exception as error:
        raise TCMPCA64DeterministicPlanError(str(error)) from error

    audit = load_json(audit_path)
    protocol = load_json(protocol_path)
    metrics = load_json(metrics_path)
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("adapter") != adapter:
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected AE64 audit identity differs"
        )
    initial = audit.get("initial")
    if not isinstance(initial, Mapping):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected AE64 initial audit is absent"
        )
    shared = _validated_hash_mapping(
        initial.get("shared_variable_sha256"),
        expected_keys=EXPECTED_SHARED_INITIAL_KEYS,
        label=f"fold {fold} AE64 shared initialization",
    )
    if dict(winner.get("shared_initial_variable_sha256", {})) != shared:
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} winner shared initialization differs from audit"
        )
    adapter_initial = _validated_hash_mapping(
        initial.get("adapter_variable_sha256"),
        expected_keys=adapter_keys,
        label=f"fold {fold} AE64 adapter initialization",
    )
    seed_audit = audit.get("deterministic_initialization")
    if (
        not isinstance(seed_audit, Mapping)
        or seed_audit.get("tensorflow_graph_seed") != 42
        or seed_audit.get("seed_observed_before_model_construction") is not True
    ):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} selected AE64 lacks deterministic seed evidence"
        )
    try:
        optimizer_steps = int(metrics.get("Optimizer_Steps", -1))
        adapter_base_runner._validate_audit(
            audit, adapter=adapter, expected_steps=optimizer_steps
        )
    except Exception as error:
        raise TCMPCA64DeterministicPlanError(str(error)) from error

    binding = protocol.get("tcm_semantic_adapter_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != PARENT_RUNNER_SCHEMA:
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} deterministic AE64 runner binding is absent"
        )
    if (
        binding.get("plan", {}).get("sha256") != parent_plan_sha256
        or Path(str(binding.get("plan", {}).get("path", ""))).resolve()
        != parent_plan_path.resolve()
        or binding.get("candidate", {}).get("id") != winner.get("id")
        or binding.get("candidate", {}).get("semantic_adapter") != adapter
        or binding.get("deterministic_initialization")
        != parent_plan.get("deterministic_initialization")
        or binding.get("outer_test_materialised") is not False
    ):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} deterministic AE64 binding differs"
        )
    receipt = binding.get("tensorflow_graph_seed_receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("seed") != 42
        or receipt.get("graph_was_empty_before_seed") is not True
        or receipt.get("tensorflow_v1_behavior_enabled_before_seed") is not True
        or receipt.get("seed_set_before_model_import") is not True
    ):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} deterministic AE64 seed receipt differs"
        )
    controls = protocol_controls(protocol)
    if optimizer_steps != controls["optimizer_steps"]:
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} AE64 optimizer step records differ"
        )
    expected_fold = parent_plan.get("fold_hyperparameters", {}).get(str(fold))
    if (
        not isinstance(expected_fold, Mapping)
        or int(expected_fold.get("epochs", -1)) != epochs
        or not math.isclose(
            float(expected_fold.get("learning_rate", float("nan"))),
            learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise TCMPCA64DeterministicPlanError(
            f"fold {fold} AE64 hyperparameters differ from parent plan"
        )
    expected_hyperparameters = {
        **EXPECTED_FIXED,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "semantic_adapter": adapter,
        "model_class": "ASTRATCMSemanticAdapterDeterministic",
        "tensorflow_graph_seed": 42,
        "smoke_override": False,
        "diagnostic_mode": False,
    }
    for key, expected in expected_hyperparameters.items():
        observed = controls["hyperparameters"].get(key)
        if isinstance(expected, float):
            matches = math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            )
        else:
            matches = observed == expected
        if not matches:
            raise TCMPCA64DeterministicPlanError(
                f"fold {fold} AE64 hyperparameter {key} differs"
            )
    hashes: dict[str, str] = {}
    for key in (
        "validation_row_id_sha256",
        "validation_pair_sha256",
        "validation_label_sha256",
    ):
        value = winner.get(key)
        if not _valid_hash(value):
            raise TCMPCA64DeterministicPlanError(f"fold {fold} {key} is invalid")
        hashes[key] = str(value)
    return {
        "adapter_selection_candidate_id": str(winner["id"]),
        "semantic_adapter": adapter,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "ae64_validation_auprc": validation_auprc,
        "ae64_validation_auroc": validation_auroc,
        "ae64_validation_threshold": validation_threshold,
        "ae64_artifacts": artifact_records,
        "ae64_completion_marker": artifact(completion_path),
        "ae64_initial_shared_variable_sha256": shared,
        "ae64_initial_adapter_variable_sha256": adapter_initial,
        "ae64_protocol_controls": controls,
        "ae64_model_source": dict(audit.get("model_source", {})),
        "ae64_base_model_source": dict(audit.get("base_model_source", {})),
        **hashes,
    }


def _bundle_record(validated: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(validated["manifest_path"]))
    return {
        "path": str(manifest_path.parent),
        "manifest": artifact(manifest_path),
        "entities": {
            entity: {
                "matrix": artifact(details["matrix_path"]),
                "row_ids_sidecar": artifact(details["sidecar_path"]),
            }
            for entity, details in validated["entities"].items()
        },
    }


def build_plan(
    *, adapter_selection_path: Path, pca64_bundle_manifest: Path
) -> dict[str, Any]:
    selection_path = adapter_selection_path.resolve(strict=True)
    selection_sha = validate_sidecar(
        selection_path, label="deterministic adapter selection"
    )
    selection = load_json(selection_path)
    if (
        selection.get("schema") != PARENT_SELECTION_SCHEMA
        or selection.get("dataset") != "tcm-suite"
        or int(selection.get("seed", -1)) != 42
        or selection.get("outer_test_accessed") is not False
        or selection.get("outer_test_evaluator_in_this_module") is not False
    ):
        raise TCMPCA64DeterministicPlanError(
            "parent selection is not the deterministic-v3 validation selection"
        )
    parent_plan_path = verify_artifact(
        selection.get("plan"), label="deterministic adapter plan"
    )
    parent_plan_sha = validate_sidecar(
        parent_plan_path, label="deterministic adapter plan"
    )
    if parent_plan_sha != selection.get("plan", {}).get("sha256"):
        raise TCMPCA64DeterministicPlanError(
            "parent selection binds another deterministic plan hash"
        )
    parent_plan = load_json(parent_plan_path)
    if (
        parent_plan.get("schema") != PARENT_PLAN_SCHEMA
        or parent_plan.get("status") != "frozen-before-stage2-candidate-testing"
        or parent_plan.get("dataset") != "tcm-suite"
        or int(parent_plan.get("seed", -1)) != 42
    ):
        raise TCMPCA64DeterministicPlanError(
            "parent deterministic adapter plan is incompatible"
        )
    if (
        selection.get("deterministic_initialization")
        != parent_plan.get("deterministic_initialization")
    ):
        raise TCMPCA64DeterministicPlanError(
            "parent selection deterministic contract differs from its plan"
        )
    if parent_plan.get("deterministic_initialization") != (
        EXPECTED_DETERMINISTIC_INITIALIZATION
    ):
        raise TCMPCA64DeterministicPlanError(
            "parent TensorFlow graph-seed contract differs"
        )
    if dict(parent_plan.get("fixed", {})) != EXPECTED_FIXED:
        raise TCMPCA64DeterministicPlanError(
            "parent deterministic adapter fixed controls differ"
        )
    _verify_source_freeze(
        parent_plan.get("source_freeze"), label="parent deterministic adapter"
    )
    selector_record = selection.get("selector")
    verify_artifact(selector_record, label="parent deterministic selector")

    parent_root = selection_path.parent
    launcher_marker = parent_root / PARENT_COMPLETION_NAME
    if not launcher_marker.is_file():
        raise TCMPCA64DeterministicPlanError(
            "deterministic-v3 launcher completion marker is absent"
        )
    candidate_root = Path(str(selection.get("candidate_root", ""))).resolve(strict=True)
    if not candidate_root.is_dir():
        raise TCMPCA64DeterministicPlanError(
            "deterministic-v3 candidate root is invalid"
        )
    parent_pair_receipts = _validate_parent_pair_receipts(
        selection=selection,
        parent_plan_path=parent_plan_path,
        candidate_root=candidate_root,
    )

    protocol_path = verify_artifact(
        parent_plan.get("protocol_manifest"), label="protocol manifest"
    )
    protocol = load_protocol_manifest(protocol_path)
    if protocol.dataset != "tcm-suite":
        raise TCMPCA64DeterministicPlanError("protocol dataset is not tcm-suite")
    ae_bundle_record = parent_plan.get("feature_bundle")
    if not isinstance(ae_bundle_record, Mapping):
        raise TCMPCA64DeterministicPlanError("parent AE64 bundle binding is absent")
    ae_manifest_path = verify_artifact(
        ae_bundle_record.get("manifest"), label="AE64 feature bundle manifest"
    )
    ae_validated = validate_tcm_feature_bundle(ae_manifest_path, protocol=protocol)

    pca_manifest_path = pca64_bundle_manifest.resolve(strict=True)
    pca_validated = validate_tcm_pca64_feature_bundle(
        pca_manifest_path,
        protocol=protocol,
        verify_transform=True,
    )
    pca_source = pca_validated["manifest"].get("source_ae64_bundle", {})
    if pca_source.get("original_sha256") != sha256(ae_manifest_path):
        raise TCMPCA64DeterministicPlanError(
            "PCA64 bundle was not derived from the parent AE64 bundle"
        )
    for entity in ("compound", "protein"):
        if (
            ae_validated["entities"][entity]["matrix_sha256"]
            != pca_validated["entities"][entity]["matrix_sha256"]
        ):
            raise TCMPCA64DeterministicPlanError(
                f"PCA64 changes the controlled {entity} feature matrix"
            )
    for entity in ("herb", "disease"):
        if (
            ae_validated["entities"][entity]["matrix_sha256"]
            == pca_validated["entities"][entity]["matrix_sha256"]
        ):
            raise TCMPCA64DeterministicPlanError(
                f"PCA64 did not change the intended {entity} feature matrix"
            )

    folds = selection.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {
        str(fold) for fold in range(1, 6)
    }:
        raise TCMPCA64DeterministicPlanError(
            "parent adapter selection must contain five folds"
        )
    fold_hyperparameters = {
        str(fold): _fold_winner_reference(
            fold=fold,
            fold_selection=folds[str(fold)],
            parent_plan=parent_plan,
            parent_plan_path=parent_plan_path,
            parent_plan_sha256=parent_plan_sha,
        )
        for fold in range(1, 6)
    }
    source_freeze = _source_freeze()
    source_freeze.update(
        {
            "parent_adapter_selection": artifact(selection_path),
            "parent_adapter_plan": artifact(parent_plan_path),
            "parent_adapter_selector": artifact(
                Path(str(selector_record["path"]))
            ),
        }
    )
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "dataset": "tcm-suite",
        "seed": 42,
        "interpretation": (
            "seed-42 same-split conditional feature-basis diagnostic; "
            "adapter structure is selected first and all work is validation-only"
        ),
        "parent_adapter_selection": artifact(selection_path),
        "parent_adapter_selection_sha256": selection_sha,
        "parent_adapter_plan": artifact(parent_plan_path),
        "parent_adapter_launcher_completion": artifact(launcher_marker),
        "parent_adapter_pair_receipts": parent_pair_receipts,
        "protocol_manifest": artifact(protocol_path),
        "feature_bundles": {
            "ae64": _bundle_record(ae_validated),
            "pca64": _bundle_record(pca_validated),
        },
        "deterministic_initialization": dict(
            parent_plan["deterministic_initialization"]
        ),
        "initialization_pair_contract": {
            "expected_shared_key_count": len(EXPECTED_SHARED_INITIAL_KEYS),
            "controlled_feature_input_keys": sorted(
                CONTROLLED_FEATURE_INPUT_KEYS
            ),
            "required_equal_shared_key_count": len(
                EXPECTED_SHARED_INITIAL_KEYS - CONTROLLED_FEATURE_INPUT_KEYS
            ),
            "all_noncontrolled_shared_hashes_must_match": True,
            "adapter_initial_hash_mapping_must_match": True,
            "identity_adapter_initial_key_count": 0,
            "residual_adapter_initial_key_count": len(
                EXPECTED_RESIDUAL_ADAPTER_KEYS
            ),
        },
        "fixed": dict(EXPECTED_FIXED),
        "fold_hyperparameters": fold_hyperparameters,
        "candidates": [
            {
                "id": "ae64_frozen_control",
                "feature_basis": "shared_ae64",
                "execution": "frozen_deterministic_v3_winner_reference",
                "feature_bundle_key": "ae64",
            },
            {
                "id": "pca64_candidate",
                "feature_basis": "deterministic_shared_pca64_v1",
                "execution": "validation_only_candidate",
                "feature_bundle_key": "pca64",
            },
        ],
        "selection_metric": "Validation_AUPRC",
        "minimum_pca64_auprc_gain": MIN_PCA64_AUPRC_GAIN,
        "selection_policy": (
            "Within each fold select PCA64 only when PCA64 Validation_AUPRC "
            "minus the frozen deterministic-v3 AE64 winner Validation_AUPRC "
            "is at least 0.0001. Outer-test materialisation is forbidden."
        ),
        "source_freeze": source_freeze,
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "formal_scope": (
            "Development diagnostic only; adapter and feature-basis decisions "
            "reuse seed-42 validation rows and cannot support a formal claim."
        ),
    }


def freeze_plan(
    *, adapter_selection: Path, pca64_bundle_manifest: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMPCA64DeterministicPlanError(
            f"refusing to overwrite plan or sidecar: {output}"
        )
    payload = build_plan(
        adapter_selection_path=adapter_selection,
        pca64_bundle_manifest=pca64_bundle_manifest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {**payload, "plan_path": str(output), "plan_sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-selection", type=Path, required=True)
    parser.add_argument("--pca64-bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_plan(
            adapter_selection=args.adapter_selection,
            pca64_bundle_manifest=args.pca64_bundle_manifest,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM deterministic PCA64 v2 plan freeze failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
