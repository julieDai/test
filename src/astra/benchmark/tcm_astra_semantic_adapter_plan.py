#!/usr/bin/env python3
"""Freeze the TCM semantic-adapter stage after the epoch-boundary decision.

This command never loads protocol rows and never evaluates an outer test.  It
accepts only the immutable final epoch-boundary decision and its receipt, then
copies each fold's ``selected_candidate_if_stopped`` into a two-arm,
validation-only stage-2 plan.  Epoch, learning rate, and artifacts therefore
come from the producer-authorized stopped candidate; none are accepted from
CLI arguments or inferred again here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import tcm_astra_epoch_boundary_decision as boundary_decision_module
from .tcm_astra_epoch_boundary_decision import (
    DECISION_SCHEMA,
    RECEIPT_SCHEMA,
)
from .tcm_astra_epoch_boundary_plan import (
    EPSILON,
    NUMERIC_TOLERANCE,
    TCMEpochBoundaryPlanError,
    validate_frozen_plan,
)
from .backend_paths import TCM_BACKEND_ROOT

PLAN_SCHEMA = "astra-tcm-semantic-adapter-plan/v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"


class TCMAdapterPlanError(RuntimeError):
    """Raised when a stage-2 plan cannot be frozen without ambiguity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMAdapterPlanError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMAdapterPlanError(f"JSON root must be an object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _source_freeze() -> dict[str, dict[str, object]]:
    paths = {
        "model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapter.py",
        "runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_runner.py",
        "selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_selection.py",
        "plan_generator": Path(__file__).resolve(),
        "parent_model": TCM_BACKEND_ROOT / "HGHDA.py",
        "herb_recommender": TCM_BACKEND_ROOT / "base/herbRecommender.py",
        "central_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
        "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
        "epoch_boundary_plan_validator": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_epoch_boundary_plan.py",
        "epoch_boundary_decision": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_epoch_boundary_decision.py",
    }
    try:
        return {name: _artifact(path) for name, path in paths.items()}
    except FileNotFoundError as error:
        raise TCMAdapterPlanError(
            f"stage-2 implementation is incomplete: {error}"
        ) from error


def _positive_int(value: object, *, key: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise TCMAdapterPlanError(f"{key} must be an integer") from error
    if parsed <= 0:
        raise TCMAdapterPlanError(f"{key} must be positive")
    return parsed


def _positive_float(value: object, *, key: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TCMAdapterPlanError(f"{key} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise TCMAdapterPlanError(f"{key} must be finite and positive")
    return parsed


def _object_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_artifact_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise TCMAdapterPlanError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record["path"])).resolve(strict=True)
    except OSError as error:
        raise TCMAdapterPlanError(f"{label} artifact is absent") from error
    if (
        not path.is_file()
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != sha256(path)
    ):
        raise TCMAdapterPlanError(f"{label} artifact changed after freeze")
    return path


def _verify_path_hash(record: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise TCMAdapterPlanError(f"{label} path/hash binding is invalid")
    try:
        path = Path(str(record["path"])).resolve(strict=True)
    except OSError as error:
        raise TCMAdapterPlanError(f"{label} is absent") from error
    digest = sha256(path)
    if record.get("sha256") != digest:
        raise TCMAdapterPlanError(f"{label} changed after decision")
    return path, digest


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _validation_identity(
    record: Mapping[str, Any], *, label: str
) -> tuple[str, str, str]:
    values: list[str] = []
    for key in (
        "validation_row_id_sha256",
        "validation_pair_sha256",
        "validation_label_sha256",
    ):
        value = record.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TCMAdapterPlanError(f"{label} {key} is invalid")
        values.append(value)
    return values[0], values[1], values[2]


def _winner_artifacts(
    winner: Mapping[str, Any], *, fold: int, label: str
) -> dict[str, dict[str, str]]:
    if winner.get("fold") != fold:
        raise TCMAdapterPlanError(f"{label} fold identity differs")
    if winner.get("semantic_adapter") != "identity":
        raise TCMAdapterPlanError(f"{label} is not the identity adapter")
    epochs = _positive_int(winner.get("epochs"), key=f"fold {fold} epochs")
    if epochs not in {16, 32, 64}:
        raise TCMAdapterPlanError(
            f"{label} epoch must be a frozen 16/32/64 boundary candidate"
        )
    learning_rate = _positive_float(
        winner.get("learning_rate"), key=f"fold {fold} learning rate"
    )
    if not any(_close(learning_rate, value, tolerance=0.0) for value in (0.003, 0.005)):
        raise TCMAdapterPlanError(f"{label} learning rate is outside the frozen grid")
    candidate_id = str(winner.get("id", "")).strip()
    if not candidate_id:
        raise TCMAdapterPlanError(f"{label} candidate id is absent")
    for metric in ("validation_auprc", "validation_auroc"):
        try:
            value = float(winner[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise TCMAdapterPlanError(f"{label} {metric} is invalid") from error
        if not math.isfinite(value):
            raise TCMAdapterPlanError(f"{label} {metric} is non-finite")

    required_artifacts = (
        ("validation_predictions", "validation_predictions_sha256"),
        ("herb_embedding", "herb_embedding_sha256"),
        ("disease_embedding", "disease_embedding_sha256"),
        ("metrics", "metrics_sha256"),
        ("protocol", "protocol_sha256"),
    )
    artifacts: dict[str, dict[str, str]] = {}
    for path_key, hash_key in required_artifacts:
        try:
            path = Path(str(winner.get(path_key, ""))).resolve(strict=True)
        except OSError as error:
            raise TCMAdapterPlanError(f"{label} {path_key} is absent") from error
        expected = str(winner.get(hash_key, ""))
        if not path.is_file() or not expected or sha256(path) != expected:
            raise TCMAdapterPlanError(f"{label} {path_key} changed after decision")
        artifacts[path_key] = {"path": str(path), "sha256": expected}
    validation_path = Path(artifacts["validation_predictions"]["path"])
    if (validation_path.parent / "test_predictions.npz").exists():
        raise TCMAdapterPlanError(f"{label} materialised an outer-test prediction")
    return artifacts


def _validate_fold_decision(
    *,
    fold: int,
    boundary_fold: Mapping[str, Any],
    decision_fold: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    if decision_fold.get("fold") != fold:
        raise TCMAdapterPlanError(f"epoch-boundary fold {fold} identity differs")
    if decision_fold.get("requires_epoch128") is not False:
        raise TCMAdapterPlanError(
            f"fold {fold} requires epoch 128; semantic-adapter planning is blocked"
        )
    selected = decision_fold.get("selected_candidate_if_stopped")
    if not isinstance(selected, Mapping):
        raise TCMAdapterPlanError(
            f"fold {fold} has no selected_candidate_if_stopped"
        )
    triggered = boundary_fold.get("trigger_epoch64") is True
    if decision_fold.get("trigger_epoch64") is not triggered:
        raise TCMAdapterPlanError(f"fold {fold} epoch-64 trigger differs")

    stage1_winner = boundary_fold.get("stage1_winner")
    best32 = boundary_fold.get("best_epoch32")
    if not isinstance(stage1_winner, Mapping) or not isinstance(best32, Mapping):
        raise TCMAdapterPlanError(f"fold {fold} boundary parents are absent")
    candidates = decision_fold.get("epoch64_candidates")
    if not triggered:
        if boundary_fold.get("epoch64_candidates") != []:
            raise TCMAdapterPlanError(
                f"fold {fold} untriggered boundary grid is not empty"
            )
        source_stage = "stage1"
    else:
        frozen_specs = boundary_fold.get("epoch64_candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 2
            or not all(isinstance(item, Mapping) for item in candidates)
            or not isinstance(frozen_specs, list)
            or len(frozen_specs) != 2
            or not all(isinstance(item, Mapping) for item in frozen_specs)
        ):
            raise TCMAdapterPlanError(
                f"fold {fold} must contain two frozen epoch-64 candidates"
            )
        spec_by_id = {str(item.get("id")): item for item in frozen_specs}
        if len(spec_by_id) != 2 or {
            str(item.get("id")) for item in candidates
        } != set(spec_by_id):
            raise TCMAdapterPlanError(f"fold {fold} epoch-64 grid differs")
        parent_identity = _validation_identity(
            best32, label=f"fold {fold} best epoch-32 candidate"
        )
        for item in candidates:
            candidate_id = str(item.get("id"))
            expected_spec = spec_by_id[candidate_id]
            observed_spec = {
                key: item.get(key)
                for key in (
                    "id",
                    "semantic_adapter",
                    "epochs",
                    "learning_rate",
                )
            }
            if observed_spec != dict(expected_spec):
                raise TCMAdapterPlanError(
                    f"fold {fold} epoch-64 candidate {candidate_id} "
                    "differs from its frozen spec"
                )
            _winner_artifacts(
                item,
                fold=fold,
                label=f"fold {fold} epoch-64 candidate {candidate_id}",
            )
            if _validation_identity(
                item,
                label=f"fold {fold} epoch-64 candidate {candidate_id}",
            ) != parent_identity:
                raise TCMAdapterPlanError(
                    f"fold {fold} epoch-64 validation identity differs "
                    "from best epoch-32"
                )
        source_stage = "epoch32"

    try:
        replayed = boundary_decision_module.decide_fold(
            fold_plan=boundary_fold,
            epoch64_records=candidates if isinstance(candidates, list) else [],
            epsilon=EPSILON,
        )
    except Exception as error:
        raise TCMAdapterPlanError(
            f"fold {fold} decision producer replay failed: {error}"
        ) from error
    if dict(decision_fold) != replayed:
        raise TCMAdapterPlanError(
            f"fold {fold} decision differs from producer replay"
        )
    return selected, source_stage


def _validate_receipt(
    *,
    decision_path: Path,
    decision: Mapping[str, Any],
    receipt_path: Path,
) -> None:
    receipt = _load_json(receipt_path)
    expected = {
        "schema": RECEIPT_SCHEMA,
        "decision": str(decision_path),
        "decision_sha256": sha256(decision_path),
        "decision_source": decision.get("decision_source"),
        "plan": decision.get("plan"),
        "parent_selection": decision.get("parent_selection"),
        "parent_selection_receipt": decision.get("parent_selection_receipt"),
        "parent_plan": decision.get("parent_plan"),
        "candidate_root": decision.get("candidate_root"),
        "source_freeze_sha256": decision.get("source_freeze_sha256"),
        "input_freeze_sha256": decision.get("input_freeze_sha256"),
        "feature_input_freeze_sha256": decision.get(
            "feature_input_freeze_sha256"
        ),
        "outer_test_scored_before_receipt": False,
    }
    if dict(receipt) != expected:
        raise TCMAdapterPlanError("epoch-boundary decision receipt differs")


def build_plan(epoch_boundary_decision_path: Path) -> dict[str, Any]:
    decision_path = epoch_boundary_decision_path.resolve(strict=True)
    receipt_path = decision_path.with_name(decision_path.name + ".receipt.json")
    if not receipt_path.is_file():
        raise TCMAdapterPlanError("epoch-boundary decision receipt is absent")
    decision = _load_json(decision_path)
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("status") != "epoch-boundary-selection-complete"
        or decision.get("dataset") != "tcm-suite"
        or int(decision.get("seed", -1)) != 42
    ):
        raise TCMAdapterPlanError("unsupported epoch-boundary decision")
    for field in (
        "outer_test_accessed",
        "outer_test_scored",
        "outer_test_predictions_materialised",
        "outer_test_authorized",
    ):
        if decision.get(field) is not False:
            raise TCMAdapterPlanError(
                f"epoch-boundary decision {field} must be false"
            )
    if (
        decision.get("requires_epoch128") is not False
        or decision.get("folds_requiring_epoch128") != []
        or decision.get("epoch128_plan_implemented") is not False
        or decision.get("epoch200_plan_implemented") is not False
    ):
        raise TCMAdapterPlanError(
            "epoch-boundary decision still requires an epoch-128 stage"
        )
    if (
        not _close(decision.get("epsilon"), EPSILON, tolerance=0.0)
        or not _close(
            decision.get("numeric_tolerance"),
            NUMERIC_TOLERANCE,
            tolerance=0.0,
        )
    ):
        raise TCMAdapterPlanError("epoch-boundary numeric contract differs")
    _validate_receipt(
        decision_path=decision_path,
        decision=decision,
        receipt_path=receipt_path,
    )

    decision_source = _verify_artifact_record(
        decision.get("decision_source"), label="epoch-boundary decision source"
    )
    expected_decision_source = Path(
        boundary_decision_module.__file__ or ""
    ).resolve(strict=True)
    if decision_source != expected_decision_source:
        raise TCMAdapterPlanError("epoch-boundary decision source path differs")
    boundary_plan_path, expected_boundary_sha = _verify_path_hash(
        decision.get("plan"), label="epoch-boundary plan"
    )
    try:
        boundary_plan, boundary_sha = validate_frozen_plan(boundary_plan_path)
    except TCMEpochBoundaryPlanError as error:
        raise TCMAdapterPlanError(
            f"epoch-boundary parent plan validation failed: {error}"
        ) from error
    if boundary_sha != expected_boundary_sha:
        raise TCMAdapterPlanError("epoch-boundary plan hash is unstable")

    parent = boundary_plan.get("parent")
    if not isinstance(parent, Mapping):
        raise TCMAdapterPlanError("epoch-boundary parent bindings are absent")
    parent_bindings = {
        "parent_selection": "selection",
        "parent_selection_receipt": "selection_receipt",
        "parent_plan": "plan",
    }
    parent_paths: dict[str, Path] = {}
    for decision_key, parent_key in parent_bindings.items():
        record = parent.get(parent_key)
        if decision.get(decision_key) != record:
            raise TCMAdapterPlanError(
                f"epoch-boundary {decision_key} binding differs"
            )
        parent_paths[parent_key] = _verify_artifact_record(
            record, label=decision_key.replace("_", " ")
        )
    for field in (
        "source_freeze",
        "source_freeze_sha256",
        "input_freeze",
        "input_freeze_sha256",
        "feature_input_freeze",
        "feature_input_freeze_sha256",
        "selection_metric",
        "selection_metric_definition",
        "array_hash_schema",
        "tie_break",
    ):
        if decision.get(field) != boundary_plan.get(field):
            raise TCMAdapterPlanError(
                f"epoch-boundary decision differs from its plan at {field}"
            )

    protocol_path = _verify_artifact_record(
        boundary_plan.get("protocol_manifest"), label="protocol manifest"
    )
    feature_bundle_record = boundary_plan.get("feature_bundle")
    if not isinstance(feature_bundle_record, Mapping) or set(
        feature_bundle_record
    ) != {"path", "manifest"}:
        raise TCMAdapterPlanError("feature-bundle binding is invalid")
    feature_bundle = Path(str(feature_bundle_record["path"])).resolve(strict=True)
    bundle_manifest = _verify_artifact_record(
        feature_bundle_record["manifest"], label="feature-bundle manifest"
    )
    if bundle_manifest.parent != feature_bundle:
        raise TCMAdapterPlanError("feature-bundle root differs")

    boundary_folds = boundary_plan.get("folds")
    decision_folds = decision.get("folds")
    expected_folds = {str(fold) for fold in range(1, 6)}
    if (
        not isinstance(boundary_folds, Mapping)
        or not isinstance(decision_folds, Mapping)
        or set(boundary_folds) != expected_folds
        or set(decision_folds) != expected_folds
    ):
        raise TCMAdapterPlanError("epoch-boundary decision must contain five folds")
    fold_hyperparameters: dict[str, Any] = {}
    for fold in range(1, 6):
        boundary_fold = boundary_folds[str(fold)]
        decision_fold = decision_folds[str(fold)]
        if not isinstance(boundary_fold, Mapping) or not isinstance(
            decision_fold, Mapping
        ):
            raise TCMAdapterPlanError(f"fold {fold} decision is invalid")
        winner, source_stage = _validate_fold_decision(
            fold=fold,
            boundary_fold=boundary_fold,
            decision_fold=decision_fold,
        )
        artifacts = _winner_artifacts(
            winner,
            fold=fold,
            label=f"fold {fold} selected epoch-boundary candidate",
        )
        epochs = _positive_int(winner.get("epochs"), key=f"fold {fold} epochs")
        learning_rate = _positive_float(
            winner.get("learning_rate"), key=f"fold {fold} learning rate"
        )
        fold_hyperparameters[str(fold)] = {
            "stage1_candidate_id": str(winner["id"]),
            "epoch_boundary_candidate_id": str(winner["id"]),
            "epoch_boundary_source_stage": source_stage,
            "epoch_boundary_decision_fold_sha256": _object_sha256(
                dict(decision_fold)
            ),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "stage1_validation_auprc": float(winner["validation_auprc"]),
            "stage1_validation_auroc": float(winner["validation_auroc"]),
            "selected_validation_auprc": float(winner["validation_auprc"]),
            "selected_validation_auroc": float(winner["validation_auroc"]),
            # Kept for the existing runner/selector/evaluator contract.  These
            # artifacts now come from the final boundary decision rather than
            # being selected again by the semantic-adapter stage.
            "stage1_winner_artifacts": artifacts,
            "epoch_boundary_winner_artifacts": artifacts,
        }

    source_freeze = _source_freeze()
    source_freeze.update(
        {
            "provenance_epoch_boundary_decision": _artifact(decision_path),
            "provenance_epoch_boundary_decision_receipt": _artifact(
                receipt_path
            ),
            "provenance_epoch_boundary_plan": _artifact(boundary_plan_path),
            "provenance_stage1_selection_receipt": _artifact(
                parent_paths["selection_receipt"]
            ),
        }
    )
    return {
        "schema": PLAN_SCHEMA,
        "status": "frozen-before-stage2-candidate-testing",
        "dataset": "tcm-suite",
        "seed": 42,
        "interpretation": (
            "Seed-42 same-split post-baseline development diagnostic; "
            "validation-only adapter selection"
        ),
        "epoch_boundary_decision": _artifact(decision_path),
        "epoch_boundary_decision_receipt": _artifact(receipt_path),
        "epoch_boundary_plan": _artifact(boundary_plan_path),
        "epoch_boundary_candidate_root": str(decision.get("candidate_root")),
        "stage1_selection": _artifact(parent_paths["selection"]),
        "stage1_selection_receipt": _artifact(
            parent_paths["selection_receipt"]
        ),
        "stage1_plan": _artifact(parent_paths["plan"]),
        "protocol_manifest": _artifact(protocol_path),
        "feature_bundle": {
            "path": str(feature_bundle),
            "manifest": _artifact(bundle_manifest),
        },
        "fixed": {
            "batch_size": 2000,
            "embedding_dimension": 64,
            "reg_u": 0.001,
            "text_fusion_alpha": 1.0,
            "semantic_input_dim": 64,
            "semantic_mlp_hidden_dim": 32,
            "semantic_freeze_input": False,
        },
        "fold_hyperparameters": fold_hyperparameters,
        "candidates": [
            {
                "id": "identity_control",
                "semantic_adapter": "identity",
            },
            {
                "id": "residual_mlp_h32",
                "semantic_adapter": "residual_mlp",
            },
        ],
        "selection_metric": "Validation_AUPRC",
        "tie_break": [
            "higher Validation_AUROC",
            "identity control",
            "lexicographically smaller candidate id",
        ],
        "selection_policy": (
            "Select independently within each outer fold using only frozen "
            "inner-validation predictions. No stage-2 candidate may create "
            "an outer-test prediction before the stage-2 selection is written."
        ),
        "source_freeze": source_freeze,
        "formal_scope": (
            "Development diagnostic only because seed-42 outer folds were "
            "inspected before this stage; independent confirmation requires "
            "a previously unseen frozen split."
        ),
    }


def freeze_plan(
    *, epoch_boundary_decision: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMAdapterPlanError(
            f"refusing to overwrite plan or sidecar: {output}, {sidecar}"
        )
    payload = build_plan(epoch_boundary_decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    plan_sha = sha256(output)
    sidecar.write_text(f"{plan_sha}  {output.name}\n", encoding="utf-8")
    return {**payload, "plan_path": str(output), "plan_sha256": plan_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epoch-boundary-decision", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_plan(
            epoch_boundary_decision=args.epoch_boundary_decision,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM adapter plan freeze failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
