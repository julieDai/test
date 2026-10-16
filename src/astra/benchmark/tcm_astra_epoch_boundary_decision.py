#!/usr/bin/env python3
"""Audit TCM 64-epoch validation candidates and decide whether 128 is needed.

This module never constructs an outer-test split.  It replays the complete
parent ``{16, 32} x learning-rate`` validation selection, validates both
64-epoch arms for every independently triggered fold, and applies the frozen
``epsilon=1e-4`` continuation rule.  It emits a decision plus an immutable
receipt; it does not create a 128/200 training plan and does not authorize an
outer-test evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import (
    AUPRC_DEFINITION,
    AVERAGE_PRECISION_DEFINITION,
    evaluate_validation_scores,
    validate_prediction_coverage,
)
from .tcm_astra_epoch_boundary_plan import (
    EPSILON,
    NUMERIC_TOLERANCE,
    SOURCE_PATHS,
    TIE_BREAK,
    TCMEpochBoundaryPlanError,
    sha256,
    validate_frozen_plan,
)
from .tcm_astra_epoch_boundary_runner import (
    RUNNER_SCHEMA,
    TUNING_STAGE,
    reject_outer_artifacts,
)
from .tcm_astra_hparam_selection import (
    ARRAY_HASH_SCHEMA,
    EXPECTED_ADAPTER_SOURCES,
    EXPECTED_MODEL_SOURCES,
    EXPECTED_SELECTION_METRIC,
    PROVENANCE_AUPRC_DEFINITION,
    PlanContext,
    TrainingValidationRows,
    _canonical_array_sha256,
    _close,
    _derive_training_validation_rows,
    _embedding_scores,
    _expected_protocol_input_manifest,
    _manifest_fold_record,
    _object_sha256,
    _single_embedding,
    _validated_selection_replay,
    _verify_file_record,
    _verify_source_inputs,
    _verify_source_map,
)


DECISION_SCHEMA = "astra-tcm-epoch-boundary-decision/v2"
RECEIPT_SCHEMA = "astra-tcm-epoch-boundary-decision-receipt/v2"


class TCMEpochBoundaryDecisionError(RuntimeError):
    """Raised when a boundary decision cannot be reproduced safely."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMEpochBoundaryDecisionError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TCMEpochBoundaryDecisionError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _receipt_path(decision_path: Path) -> Path:
    return decision_path.with_name(decision_path.name + ".receipt.json")


def _candidate_rank(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        -float(record["validation_auprc"]),
        -float(record["validation_auroc"]),
        float(record["learning_rate"]),
        str(record["id"]),
    )


def decide_fold(
    *,
    fold_plan: Mapping[str, Any],
    epoch64_records: Sequence[Mapping[str, Any]],
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Apply the frozen continuation rule to one outer fold only."""

    fold = int(fold_plan.get("fold", -1))
    if not math.isclose(float(epsilon), EPSILON, rel_tol=0.0, abs_tol=0.0):
        raise TCMEpochBoundaryDecisionError("decision epsilon differs from 1e-4")
    best32 = fold_plan.get("best_epoch32")
    stage1_winner = fold_plan.get("stage1_winner")
    if not isinstance(best32, Mapping) or not isinstance(stage1_winner, Mapping):
        raise TCMEpochBoundaryDecisionError(f"fold {fold} stage-1 records are absent")

    triggered = fold_plan.get("trigger_epoch64") is True
    if not triggered:
        if epoch64_records:
            raise TCMEpochBoundaryDecisionError(
                f"fold {fold} has 64-epoch records despite no trigger"
            )
        return {
            "fold": fold,
            "trigger_epoch64": False,
            "epoch64_candidates": [],
            "best_epoch64": None,
            "delta_epoch64_minus_epoch32": None,
            "epsilon": EPSILON,
            "requires_epoch128": False,
            "decision": "stop-at-stage1",
            "selected_candidate_if_stopped": dict(stage1_winner),
        }

    if len(epoch64_records) != 2:
        raise TCMEpochBoundaryDecisionError(
            f"fold {fold} requires exactly two 64-epoch candidates"
        )
    ids = {str(record.get("id")) for record in epoch64_records}
    expected_ids = {"identity_e64_lr3e3", "identity_e64_lr5e3"}
    if ids != expected_ids:
        raise TCMEpochBoundaryDecisionError(
            f"fold {fold} 64-epoch candidate ids differ: {sorted(ids)}"
        )
    best64 = dict(min(epoch64_records, key=_candidate_rank))
    delta = float(best64["validation_auprc"]) - float(
        best32["validation_auprc"]
    )
    requires128 = delta > EPSILON + NUMERIC_TOLERANCE
    return {
        "fold": fold,
        "trigger_epoch64": True,
        "epoch64_candidates": [dict(record) for record in epoch64_records],
        "best_epoch64": best64,
        "delta_epoch64_minus_epoch32": delta,
        "epsilon": EPSILON,
        "requires_epoch128": requires128,
        "decision": "require-epoch128" if requires128 else "stop-at-epoch32",
        "selected_candidate_if_stopped": None
        if requires128
        else dict(best32),
    }


def _validate_candidate(
    *,
    candidate_root: Path,
    fold: int,
    candidate: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
    plan: Mapping[str, Any],
    context: PlanContext,
    validation: TrainingValidationRows,
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    root = (candidate_root / candidate_id / f"fold_{fold}").resolve()
    metrics_path = root / "results" / "metrics.json"
    protocol_path = root / "protocol.json"
    prediction_path = root / "results" / "validation_predictions.npz"
    for path in (metrics_path, protocol_path, prediction_path):
        if not path.is_file():
            raise TCMEpochBoundaryDecisionError(f"candidate artifact is missing: {path}")
    try:
        reject_outer_artifacts(root)
    except Exception as error:
        raise TCMEpochBoundaryDecisionError(str(error)) from error

    metrics = _load_json(metrics_path)
    protocol = _load_json(protocol_path)
    binding = protocol.get("tcm_epoch_boundary_candidate")
    if not isinstance(binding, Mapping):
        raise TCMEpochBoundaryDecisionError(
            f"candidate binding is absent: {protocol_path}"
        )
    expected_metrics = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": int(context.protocol.seed),
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Unselected_Test_Models_Evaluated": 0,
        "Candidate_ID": candidate_id,
        "Candidate_Plan_SHA256": plan_sha,
        "Tuning_Stage": TUNING_STAGE,
        "Formal_Eligible": False,
        "Diagnostic": False,
        "Smoke": False,
        "Text_Fusion_Alpha": 1.0,
        "Alpha_Selected_On_Validation": False,
        "Decision_Frozen_Before_Test": False,
    }
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        valid = _close(observed, expected) if isinstance(expected, float) else observed == expected
        if not valid:
            raise TCMEpochBoundaryDecisionError(
                f"{candidate_id} fold {fold} metric {key} differs"
            )

    fold_plan = plan["folds"][str(fold)]
    expected_binding = {
        "schema": RUNNER_SCHEMA,
        "plan_path": plan_path,
        "plan_sha": plan_sha,
        "fold": fold,
        "fold_plan_sha": _object_sha256(dict(fold_plan)),
        "candidate": dict(candidate),
    }
    if binding.get("schema") != expected_binding["schema"]:
        raise TCMEpochBoundaryDecisionError("candidate runner schema differs")
    bound_plan = binding.get("plan")
    if (
        not isinstance(bound_plan, Mapping)
        or Path(str(bound_plan.get("path", ""))).resolve() != plan_path
        or bound_plan.get("sha256") != plan_sha
    ):
        raise TCMEpochBoundaryDecisionError("candidate plan binding differs")
    if (
        binding.get("fold") != fold
        or binding.get("fold_plan_sha256") != expected_binding["fold_plan_sha"]
        or binding.get("candidate") != expected_binding["candidate"]
        or binding.get("selection_metric") != EXPECTED_SELECTION_METRIC
        or binding.get("outer_test_materialised") is not False
        or binding.get("source_freeze") != plan.get("source_freeze")
        or binding.get("source_freeze_sha256")
        != plan.get("source_freeze_sha256")
        or binding.get("input_freeze") != plan.get("input_freeze")
        or binding.get("input_freeze_sha256")
        != plan.get("input_freeze_sha256")
        or binding.get("feature_input_freeze")
        != plan.get("feature_input_freeze")
        or binding.get("feature_input_freeze_sha256")
        != plan.get("feature_input_freeze_sha256")
        or binding.get("parent_selection") != plan["parent"]["selection"]
        or binding.get("parent_selection_receipt")
        != plan["parent"]["selection_receipt"]
        or binding.get("parent_plan") != plan["parent"]["plan"]
    ):
        raise TCMEpochBoundaryDecisionError("candidate frozen provenance differs")
    runner_sha = _verify_file_record(
        binding.get("runner"),
        expected_path=SOURCE_PATHS["boundary_runner"],
        label="boundary runner",
    )

    if (
        protocol.get("schema") != "astra-central-fair-run/v1"
        or protocol.get("dataset") != "tcm-suite"
        or protocol.get("seed") != context.protocol.seed
        or protocol.get("fold") != fold
        or protocol.get("validation_only") is not True
        or protocol.get("formal_eligible") is not False
        or protocol.get("smoke") is not False
        or protocol.get("diagnostic") is not False
    ):
        raise TCMEpochBoundaryDecisionError("candidate central-run identity differs")
    _verify_file_record(
        protocol.get("protocol_manifest"),
        expected_path=context.protocol_path,
        label="candidate protocol manifest",
    )
    expected_input_manifest = _expected_protocol_input_manifest(context)
    if protocol.get("protocol_input_manifest") != expected_input_manifest:
        raise TCMEpochBoundaryDecisionError("candidate protocol input manifest differs")
    expected_input_lock = _object_sha256(expected_input_manifest)
    if protocol.get("frozen_input_lock_sha256") != expected_input_lock:
        raise TCMEpochBoundaryDecisionError("candidate input-lock hash differs")

    feature_bundle = protocol.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping):
        raise TCMEpochBoundaryDecisionError("candidate feature bundle is absent")
    _verify_file_record(
        feature_bundle,
        expected_path=context.feature_manifest,
        label="candidate feature-bundle manifest",
    )
    if (
        feature_bundle.get("sha256")
        != plan["feature_bundle"]["manifest"]["sha256"]
        or feature_bundle.get("formal_eligible") is not True
        or feature_bundle.get("recursive_validation") is not True
    ):
        raise TCMEpochBoundaryDecisionError("candidate feature-bundle contract differs")

    adapter_sources_sha = _verify_source_map(
        protocol.get("adapter_sources"),
        expected_paths=EXPECTED_ADAPTER_SOURCES,
        label="adapter_sources",
    )
    model_sources_sha = _verify_source_map(
        protocol.get("model_sources"),
        expected_paths=EXPECTED_MODEL_SOURCES,
        label="model_sources",
    )
    source_inputs_sha = _verify_source_inputs(
        protocol.get("source_inputs_verified"), context=context
    )

    fixed = plan["fixed"]
    expected_hyperparameters = {
        "epochs": 64,
        "formal_epochs": 64,
        "batch_size": int(fixed["batch_size"]),
        "learning_rate": float(candidate["learning_rate"]),
        "embedding_dimension": int(fixed["embedding_dimension"]),
        "reg_u": float(fixed["reg_u"]),
        "text_fusion_alpha": float(fixed["text_fusion_alpha"]),
        "smoke_override": False,
        "diagnostic_mode": False,
    }
    hyperparameters = protocol.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping) or set(hyperparameters) != set(
        expected_hyperparameters
    ):
        raise TCMEpochBoundaryDecisionError("candidate hyperparameter fields differ")
    for key, expected in expected_hyperparameters.items():
        observed = hyperparameters.get(key)
        valid = _close(observed, expected) if isinstance(expected, float) else observed == expected
        if not valid:
            raise TCMEpochBoundaryDecisionError(
                f"candidate hyperparameter {key} differs"
            )

    expected_steps = 64 * int(
        math.ceil(len(validation.train_pairs) / int(fixed["batch_size"]))
    )
    if (
        protocol.get("optimizer_steps") != expected_steps
        or metrics.get("Optimizer_Steps") != expected_steps
        or binding.get("optimizer_steps") != expected_steps
    ):
        raise TCMEpochBoundaryDecisionError("candidate optimizer-step budget differs")

    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if not isinstance(evaluation, Mapping) or not isinstance(split_rows, Mapping):
        raise TCMEpochBoundaryDecisionError("candidate evaluation provenance is absent")
    if (
        evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or evaluation.get("test_models_evaluated") != 0
        or evaluation.get("unselected_test_models_evaluated") != 0
        or evaluation.get("test_threshold_search") is not False
        or evaluation.get("auprc_definition") != PROVENANCE_AUPRC_DEFINITION
        or evaluation.get("average_precision_definition")
        != AVERAGE_PRECISION_DEFINITION
    ):
        raise TCMEpochBoundaryDecisionError("candidate materialised an outer test")

    manifest_fold = _manifest_fold_record(context, fold)
    test_manifest = manifest_fold.get("test")
    if not isinstance(test_manifest, Mapping):
        raise TCMEpochBoundaryDecisionError("protocol test metadata is absent")
    expected_split_rows = {
        "train_full": len(validation.train_pairs),
        "train_used": len(validation.train_pairs),
        "train_positive_graph_edges_full": int(validation.train_labels.sum()),
        "validation_full": len(validation.validation_pairs),
        "validation_used": len(validation.validation_pairs),
        "test_full": int(test_manifest["rows"]),
        "test_used": 0,
    }
    if dict(split_rows) != expected_split_rows:
        raise TCMEpochBoundaryDecisionError("candidate split-row counts differ")
    expected_split_hashes = {
        "train_pairs_used": _canonical_array_sha256(validation.train_pairs),
        "train_labels_used": _canonical_array_sha256(validation.train_labels),
        "train_positive_pairs_full": _canonical_array_sha256(
            validation.train_pairs[validation.train_labels == 1]
        ),
        "validation_pairs_used": _canonical_array_sha256(
            validation.validation_pairs
        ),
        "test_pairs_used": None,
    }
    if protocol.get("split_array_sha256") != expected_split_hashes:
        raise TCMEpochBoundaryDecisionError("candidate split-array hashes differ")

    with np.load(prediction_path, allow_pickle=False) as artifact_file:
        if set(artifact_file.files) != {"row_id", "pair", "label", "score"}:
            raise TCMEpochBoundaryDecisionError("validation artifact fields differ")
    table = validate_prediction_coverage(
        prediction_path,
        expected_row_id=validation.validation_row_indices,
        expected_pair=validation.validation_pairs,
        expected_label=validation.validation_labels,
    )
    recomputed = evaluate_validation_scores(table.label, table.score)
    for key in (
        "Validation_AUPRC",
        "Validation_AUROC",
        "Validation_AveragePrecision",
        "Validation_Selected_Threshold",
        "Validation_F1",
        "Validation_Precision",
        "Validation_Recall",
        "Validation_Accuracy",
    ):
        if not _close(metrics.get(key), recomputed.get(key)):
            raise TCMEpochBoundaryDecisionError(
                f"{candidate_id} fold {fold} {key} cannot be reproduced"
            )
    for key in (
        "Validation_AUPRC_Definition",
        "Validation_AveragePrecision_Definition",
        "Threshold_Candidates",
        "Threshold_Tie_Break",
    ):
        if metrics.get(key) != recomputed.get(key):
            raise TCMEpochBoundaryDecisionError(
                f"{candidate_id} fold {fold} {key} differs"
            )
    if recomputed["Validation_AUPRC_Definition"] != AUPRC_DEFINITION:
        raise TCMEpochBoundaryDecisionError("validation AUPRC definition differs")

    herb_path = _single_embedding(root, "herbedgeDHCN_herb_embedding")
    disease_path = _single_embedding(root, "diseaseedgeDHCN_disease_embedding")
    herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
    disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
    if herb.shape != (context.protocol.n_herbs, 64) or disease.shape != (
        context.protocol.n_diseases,
        64,
    ):
        raise TCMEpochBoundaryDecisionError("candidate embedding shape differs")
    reproduced_scores = _embedding_scores(herb, disease, table.pair)
    if not np.allclose(reproduced_scores, table.score, rtol=0.0, atol=1e-12):
        raise TCMEpochBoundaryDecisionError(
            "candidate embeddings do not reproduce validation scores"
        )

    return {
        "id": candidate_id,
        "fold": fold,
        "epochs": 64,
        "learning_rate": float(candidate["learning_rate"]),
        "semantic_adapter": "identity",
        "validation_auprc": float(recomputed["Validation_AUPRC"]),
        "validation_auroc": float(recomputed["Validation_AUROC"]),
        "validation_f1": float(recomputed["Validation_F1"]),
        "validation_threshold": float(
            recomputed["Validation_Selected_Threshold"]
        ),
        "validation_predictions": str(prediction_path),
        "validation_predictions_sha256": sha256(prediction_path),
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "validation_row_id_sha256": _canonical_array_sha256(table.row_id),
        "validation_pair_sha256": _canonical_array_sha256(table.pair),
        "validation_label_sha256": _canonical_array_sha256(table.label),
        "herb_embedding": str(herb_path),
        "herb_embedding_sha256": sha256(herb_path),
        "disease_embedding": str(disease_path),
        "disease_embedding_sha256": sha256(disease_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "runner_sha256": runner_sha,
        "adapter_sources_sha256": adapter_sources_sha,
        "model_sources_sha256": model_sources_sha,
        "source_inputs_sha256": source_inputs_sha,
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_metric_definition": AUPRC_DEFINITION,
        "optimizer_steps": expected_steps,
    }


def build_decision(*, plan_path: Path, candidate_root: Path) -> dict[str, Any]:
    """Replay all validation inputs and construct the deterministic decision."""

    plan_path = plan_path.resolve(strict=True)
    candidate_root = candidate_root.resolve()
    try:
        plan, plan_sha = validate_frozen_plan(plan_path)
    except TCMEpochBoundaryPlanError as error:
        raise TCMEpochBoundaryDecisionError(str(error)) from error

    selection_path = Path(str(plan["parent"]["selection"]["path"])).resolve()
    try:
        (
            _selection,
            selection_sha,
            receipt_path,
            receipt_sha,
            context,
        ) = _validated_selection_replay(selection_path)
    except Exception as error:
        raise TCMEpochBoundaryDecisionError(
            f"parent selection replay failed: {error}"
        ) from error
    if (
        selection_sha != plan["parent"]["selection"]["sha256"]
        or receipt_sha != plan["parent"]["selection_receipt"]["sha256"]
        or receipt_path
        != Path(str(plan["parent"]["selection_receipt"]["path"])).resolve()
        or context.sha256 != plan["parent"]["plan"]["sha256"]
        or context.path != Path(str(plan["parent"]["plan"]["path"])).resolve()
    ):
        raise TCMEpochBoundaryDecisionError("parent replay differs from boundary plan")

    folds: dict[str, Any] = {}
    all_boundary_records: list[dict[str, Any]] = []
    for fold in range(1, context.protocol.n_folds + 1):
        fold_plan = plan["folds"][str(fold)]
        validation = _derive_training_validation_rows(context.protocol, fold)
        candidates = fold_plan["epoch64_candidates"]
        records: list[dict[str, Any]] = []
        if fold_plan["trigger_epoch64"]:
            records = [
                _validate_candidate(
                    candidate_root=candidate_root,
                    fold=fold,
                    candidate=candidate,
                    plan_path=plan_path,
                    plan_sha=plan_sha,
                    plan=plan,
                    context=context,
                    validation=validation,
                )
                for candidate in candidates
            ]
            identity_hashes = {
                (
                    record["validation_row_id_sha256"],
                    record["validation_pair_sha256"],
                    record["validation_label_sha256"],
                )
                for record in records
            }
            if len(identity_hashes) != 1:
                raise TCMEpochBoundaryDecisionError(
                    f"fold {fold} boundary candidates used different validation rows"
                )
            parent_best32 = fold_plan["best_epoch32"]
            parent_identity = (
                parent_best32["validation_row_id_sha256"],
                parent_best32["validation_pair_sha256"],
                parent_best32["validation_label_sha256"],
            )
            if next(iter(identity_hashes)) != parent_identity:
                raise TCMEpochBoundaryDecisionError(
                    f"fold {fold} boundary validation rows differ from stage 1"
                )
            all_boundary_records.extend(records)
        else:
            for candidate_id in ("identity_e64_lr3e3", "identity_e64_lr5e3"):
                unexpected = candidate_root / candidate_id / f"fold_{fold}"
                if unexpected.exists():
                    raise TCMEpochBoundaryDecisionError(
                        f"untriggered fold {fold} has a boundary output: {unexpected}"
                    )
        folds[str(fold)] = decide_fold(
            fold_plan=fold_plan,
            epoch64_records=records,
            epsilon=EPSILON,
        )

    for field in (
        "runner_sha256",
        "adapter_sources_sha256",
        "model_sources_sha256",
        "source_inputs_sha256",
        "selection_metric",
        "selection_metric_definition",
        "array_hash_schema",
    ):
        observed = {record[field] for record in all_boundary_records}
        if observed and (len(observed) != 1 or not next(iter(observed))):
            raise TCMEpochBoundaryDecisionError(
                f"boundary candidates do not share one non-empty {field}"
            )

    requiring128 = [
        int(fold) for fold, record in folds.items() if record["requires_epoch128"]
    ]
    decision_source = Path(__file__).resolve()
    return {
        "schema": DECISION_SCHEMA,
        "status": "epoch128-required"
        if requiring128
        else "epoch-boundary-selection-complete",
        "dataset": "tcm-suite",
        "seed": int(context.protocol.seed),
        "plan": {
            "path": str(plan_path),
            "sha256": plan_sha,
        },
        "parent_selection": dict(plan["parent"]["selection"]),
        "parent_selection_receipt": dict(plan["parent"]["selection_receipt"]),
        "parent_plan": dict(plan["parent"]["plan"]),
        "candidate_root": str(candidate_root),
        "decision_source": _artifact_record(decision_source),
        "source_freeze": dict(plan["source_freeze"]),
        "source_freeze_sha256": plan["source_freeze_sha256"],
        "input_freeze": dict(plan["input_freeze"]),
        "input_freeze_sha256": plan["input_freeze_sha256"],
        "feature_input_freeze": dict(plan["feature_input_freeze"]),
        "feature_input_freeze_sha256": plan["feature_input_freeze_sha256"],
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_metric_definition": AUPRC_DEFINITION,
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "epsilon": EPSILON,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "tie_break": TIE_BREAK,
        "folds": folds,
        "requires_epoch128": bool(requiring128),
        "folds_requiring_epoch128": requiring128,
        "epoch128_plan_implemented": False,
        "epoch200_plan_implemented": False,
        "outer_test_accessed": False,
        "outer_test_scored": False,
        "outer_test_predictions_materialised": False,
        "outer_test_authorized": False,
        "interpretation": (
            "Validation-only sequential boundary decision. A separate frozen "
            "next-stage or final-selection artifact is required before any "
            "outer-test evaluation."
        ),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_decision(
    *, plan_path: Path, candidate_root: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    receipt_path = _receipt_path(output)
    if output.exists() or receipt_path.exists():
        raise TCMEpochBoundaryDecisionError(
            f"refusing to overwrite decision or receipt: {output}, {receipt_path}"
        )
    payload = build_decision(plan_path=plan_path, candidate_root=candidate_root)
    _write_json(output, payload)
    decision_sha = sha256(output)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "decision": str(output),
        "decision_sha256": decision_sha,
        "decision_source": payload["decision_source"],
        "plan": payload["plan"],
        "parent_selection": payload["parent_selection"],
        "parent_selection_receipt": payload["parent_selection_receipt"],
        "parent_plan": payload["parent_plan"],
        "candidate_root": payload["candidate_root"],
        "source_freeze_sha256": payload["source_freeze_sha256"],
        "input_freeze_sha256": payload["input_freeze_sha256"],
        "feature_input_freeze_sha256": payload["feature_input_freeze_sha256"],
        "outer_test_scored_before_receipt": False,
    }
    _write_json(receipt_path, receipt)
    return {
        **payload,
        "decision_path": str(output),
        "decision_sha256": decision_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = write_decision(
            plan_path=args.plan,
            candidate_root=args.candidate_root,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM epoch-boundary decision failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
