#!/usr/bin/env python3
"""Run one frozen 64-epoch TCM ASTRA boundary candidate on validation only."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

from . import astra_runner
from .tcm_astra_epoch_boundary_plan import (
    PLAN_SCHEMA,
    TCMEpochBoundaryPlanError,
    artifact,
    sha256,
    validate_frozen_plan,
)


RUNNER_SCHEMA = "astra-tcm-epoch-boundary-candidate/v2"
TUNING_STAGE = "epoch-boundary-64"

_FORBIDDEN_OUTER_FILENAMES = frozenset(
    {
        "test_predictions.npz",
        "test_metrics.json",
        "test_scores.npy",
        "test_scores.npz",
        "outer_test_scores.json",
        "outer_test_predictions.npz",
    }
)


class TCMEpochBoundaryRunnerError(RuntimeError):
    """Raised when a 64-epoch candidate violates its frozen contract."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMEpochBoundaryRunnerError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMEpochBoundaryRunnerError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _float_equal(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def reject_outer_artifacts(root: Path) -> None:
    if not root.exists():
        return
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in _FORBIDDEN_OUTER_FILENAMES
    ]
    if forbidden:
        raise TCMEpochBoundaryRunnerError(
            "epoch-boundary candidate materialised an outer-test artifact: "
            + ", ".join(str(path) for path in forbidden)
        )


def _candidate_for_fold(
    plan: Mapping[str, Any], *, fold: int, candidate_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    folds = plan.get("folds")
    fold_plan = folds.get(str(fold)) if isinstance(folds, Mapping) else None
    if not isinstance(fold_plan, Mapping):
        raise TCMEpochBoundaryRunnerError(f"boundary plan has no fold {fold}")
    if fold_plan.get("trigger_epoch64") is not True:
        raise TCMEpochBoundaryRunnerError(
            f"fold {fold} did not trigger the 64-epoch boundary probe"
        )
    candidates = fold_plan.get("epoch64_candidates")
    if not isinstance(candidates, list):
        raise TCMEpochBoundaryRunnerError(f"fold {fold} candidate list is invalid")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("id") == candidate_id
    ]
    if len(matches) != 1:
        raise TCMEpochBoundaryRunnerError(
            f"candidate {candidate_id!r} is not unique for fold {fold}"
        )
    candidate = matches[0]
    expected_fields = {"id", "semantic_adapter", "epochs", "learning_rate"}
    if set(candidate) != expected_fields:
        raise TCMEpochBoundaryRunnerError("boundary candidate fields differ")
    if candidate.get("semantic_adapter") != "identity" or candidate.get("epochs") != 64:
        raise TCMEpochBoundaryRunnerError("boundary candidate controls differ")
    learning_rate = float(candidate.get("learning_rate", float("nan")))
    if learning_rate not in {0.003, 0.005}:
        raise TCMEpochBoundaryRunnerError("boundary learning rate differs")
    return fold_plan, candidate


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = astra_runner._normalise_dataset(args.dataset)
    if dataset != "tcm-suite":
        raise TCMEpochBoundaryRunnerError("epoch-boundary runner is TCM-suite only")
    if not bool(args.validation_only):
        raise TCMEpochBoundaryRunnerError("boundary candidates require --validation-only")
    if bool(args.smoke) or bool(args.diagnostic):
        raise TCMEpochBoundaryRunnerError(
            "boundary candidates reject smoke and diagnostic modes"
        )
    if not _float_equal(args.text_fusion_alpha, 1.0):
        raise TCMEpochBoundaryRunnerError("boundary candidates fix text_fusion_alpha=1")
    if args.alpha_selection_manifest is not None:
        raise TCMEpochBoundaryRunnerError(
            "boundary candidates reject an alpha-selection override"
        )
    if args.herb_semantic is not None or args.disease_semantic is not None:
        raise TCMEpochBoundaryRunnerError(
            "boundary candidates use only the frozen feature-bundle semantics"
        )
    if args.legacy_ethnobotany_semantic_dir is not None:
        raise TCMEpochBoundaryRunnerError(
            "boundary candidates reject legacy semantic overrides"
        )

    plan_path = args.boundary_plan.resolve(strict=True)
    try:
        plan, plan_sha = validate_frozen_plan(plan_path)
    except TCMEpochBoundaryPlanError as error:
        raise TCMEpochBoundaryRunnerError(str(error)) from error
    if plan.get("schema") != PLAN_SCHEMA:
        raise TCMEpochBoundaryRunnerError("unsupported boundary plan schema")
    if int(plan.get("seed", -1)) != int(args.seed):
        raise TCMEpochBoundaryRunnerError("CLI seed differs from boundary plan")

    protocol = plan.get("protocol_manifest")
    feature = plan.get("feature_bundle")
    if not isinstance(protocol, Mapping) or not isinstance(feature, Mapping):
        raise TCMEpochBoundaryRunnerError("boundary input bindings are absent")
    if Path(str(protocol.get("path", ""))).resolve() != args.protocol_manifest.resolve():
        raise TCMEpochBoundaryRunnerError("CLI protocol differs from boundary plan")
    if Path(str(feature.get("path", ""))).resolve() != args.embedding_dir.resolve():
        raise TCMEpochBoundaryRunnerError("CLI feature bundle differs from boundary plan")

    fold_plan, candidate = _candidate_for_fold(
        plan, fold=int(args.fold), candidate_id=str(args.candidate_id)
    )
    reject_outer_artifacts(args.output_root.resolve())
    return {
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "plan": plan,
        "fold_plan": dict(fold_plan),
        "candidate": dict(candidate),
        "epochs": 64,
        "learning_rate": float(candidate["learning_rate"]),
    }


@contextmanager
def candidate_hyperparameters(*, epochs: int, learning_rate: float) -> Iterator[None]:
    original = astra_runner._HYPERPARAMETERS["tcm-suite"]
    replacement = dict(original)
    replacement["epochs"] = int(epochs)
    replacement["learning_rate"] = float(learning_rate)
    astra_runner._HYPERPARAMETERS["tcm-suite"] = replacement
    try:
        yield
    finally:
        astra_runner._HYPERPARAMETERS["tcm-suite"] = original


def _validate_central_output(
    *,
    metrics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    fold: int,
) -> int:
    candidate = binding["candidate"]
    plan = binding["plan"]
    expected_metrics = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": int(plan["seed"]),
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Unselected_Test_Models_Evaluated": 0,
        "Formal_Eligible": False,
        "Diagnostic": False,
        "Smoke": False,
        "Text_Fusion_Alpha": 1.0,
        "Alpha_Selected_On_Validation": False,
        "Decision_Frozen_Before_Test": False,
    }
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        equal = (
            _float_equal(observed, expected)
            if isinstance(expected, float)
            else observed == expected
        )
        if not equal:
            raise TCMEpochBoundaryRunnerError(
                f"central metric {key} is {observed!r}, expected {expected!r}"
            )
    if (
        protocol.get("schema") != "astra-central-fair-run/v1"
        or protocol.get("dataset") != "tcm-suite"
        or protocol.get("seed") != int(plan["seed"])
        or protocol.get("fold") != fold
        or protocol.get("validation_only") is not True
        or protocol.get("formal_eligible") is not False
        or protocol.get("smoke") is not False
        or protocol.get("diagnostic") is not False
    ):
        raise TCMEpochBoundaryRunnerError("central protocol identity differs")
    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or evaluation.get("test_models_evaluated") != 0
        or evaluation.get("unselected_test_models_evaluated") != 0
        or not isinstance(split_rows, Mapping)
        or split_rows.get("test_used") != 0
    ):
        raise TCMEpochBoundaryRunnerError("central runner touched an outer test")
    hyperparameters = protocol.get("hyperparameters")
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
    if not isinstance(hyperparameters, Mapping) or set(hyperparameters) != set(
        expected_hyperparameters
    ):
        raise TCMEpochBoundaryRunnerError("central hyperparameter fields differ")
    for key, expected in expected_hyperparameters.items():
        observed = hyperparameters.get(key)
        equal = (
            _float_equal(observed, expected)
            if isinstance(expected, float)
            else observed == expected
        )
        if not equal:
            raise TCMEpochBoundaryRunnerError(
                f"central hyperparameter {key} differs"
            )
    train_rows = int(split_rows.get("train_used", -1))
    if train_rows <= 0:
        raise TCMEpochBoundaryRunnerError("central training row count is invalid")
    expected_steps = 64 * int(math.ceil(train_rows / int(fixed["batch_size"])))
    if (
        protocol.get("optimizer_steps") != expected_steps
        or metrics.get("Optimizer_Steps") != expected_steps
    ):
        raise TCMEpochBoundaryRunnerError("64-epoch optimizer-step budget differs")
    return expected_steps


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    with candidate_hyperparameters(
        epochs=int(binding["epochs"]),
        learning_rate=float(binding["learning_rate"]),
    ):
        astra_runner.run(args)

    output_root = args.output_root.resolve()
    metrics_path = output_root / "results" / "metrics.json"
    protocol_path = output_root / "protocol.json"
    stored_metrics = dict(_load_json(metrics_path))
    stored_protocol = dict(_load_json(protocol_path))
    reject_outer_artifacts(output_root)
    if sha256(Path(str(binding["plan_path"]))) != binding["plan_sha256"]:
        raise TCMEpochBoundaryRunnerError("boundary plan changed while training")
    expected_steps = _validate_central_output(
        metrics=stored_metrics,
        protocol=stored_protocol,
        binding=binding,
        fold=int(args.fold),
    )

    source_path = Path(__file__).resolve()
    parent = binding["plan"]["parent"]
    provenance = {
        "schema": RUNNER_SCHEMA,
        "plan": {
            "path": binding["plan_path"],
            "sha256": binding["plan_sha256"],
        },
        "parent_selection": dict(parent["selection"]),
        "parent_selection_receipt": dict(parent["selection_receipt"]),
        "parent_plan": dict(parent["plan"]),
        "fold": int(args.fold),
        "fold_plan_sha256": sha256_object(binding["fold_plan"]),
        "candidate": dict(binding["candidate"]),
        "runner": artifact(source_path),
        "source_freeze": dict(binding["plan"]["source_freeze"]),
        "source_freeze_sha256": binding["plan"]["source_freeze_sha256"],
        "input_freeze": dict(binding["plan"]["input_freeze"]),
        "input_freeze_sha256": binding["plan"]["input_freeze_sha256"],
        "feature_input_freeze": dict(binding["plan"]["feature_input_freeze"]),
        "feature_input_freeze_sha256": binding["plan"][
            "feature_input_freeze_sha256"
        ],
        "selection_metric": "Validation_AUPRC",
        "optimizer_steps": expected_steps,
        "outer_test_materialised": False,
    }
    stored_metrics.update(
        {
            "Method": "ASTRA-TCM-epoch-boundary-candidate",
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Tuning_Stage": TUNING_STAGE,
            "Execution_Mode": "epoch-boundary-validation-candidate",
            "Evaluation_Mode": "validation-only-epoch-boundary-selection",
        }
    )
    stored_protocol["tcm_epoch_boundary_candidate"] = provenance
    _write_json(metrics_path, stored_metrics)
    _write_json(protocol_path, stored_protocol)
    return stored_metrics


def sha256_object(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = astra_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--boundary-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:
        print(f"TCM epoch-boundary candidate failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
