#!/usr/bin/env python3
"""Select AE64 or PCA64 per fold from deterministic validation evidence only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import evaluate_validation_scores
from . import tcm_astra_pca64_control_runner as pca_base_runner
from . import tcm_astra_pca64_control_selection as pca_base_selection
from . import tcm_astra_semantic_adapter_runner as adapter_base_runner
from . import tcm_astra_semantic_adapter_selection as adapter_selection
from .tcm_astra_pca64_deterministic_pair_guard import build_pair_receipt
from .tcm_astra_pca64_deterministic_plan import (
    MIN_PCA64_AUPRC_GAIN,
    PLAN_SCHEMA,
    PLAN_STATUS,
    artifact,
    load_json,
    sha256,
    validate_sidecar,
    verify_artifact,
)
from .tcm_astra_pca64_deterministic_runner import (
    CANDIDATE_COMPLETION_NAME,
    RUNNER_SCHEMA,
)


SELECTION_SCHEMA = "astra-tcm-pca64-deterministic-selection/v2"


class TCMPCA64DeterministicSelectionError(RuntimeError):
    """Raised when deterministic feature-basis evidence is incomplete."""


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _validate_pair_receipts(
    *, plan_path: Path, candidate_root: Path, receipt_root: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold in range(1, 6):
        path = (receipt_root / f"fold_{fold}.json").resolve()
        if not path.is_file():
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} feature-basis pair receipt is absent"
            )
        digest = validate_sidecar(path, label=f"fold {fold} feature-basis receipt")
        expected = build_pair_receipt(
            plan_path=plan_path,
            candidate_root=candidate_root,
            fold=fold,
        )
        if dict(load_json(path)) != expected:
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} feature-basis pair receipt no longer replays"
            )
        records.append(
            {
                "fold": fold,
                "receipt": artifact(path),
                "sidecar": artifact(path.with_suffix(path.suffix + ".sha256")),
                "receipt_sha256": digest,
            }
        )
    return records


def _validate_pca_candidate(
    *,
    plan: Mapping[str, Any],
    plan_sha: str,
    candidate_root: Path,
    fold: int,
) -> dict[str, Any]:
    fold_plan = plan["fold_hyperparameters"][str(fold)]
    root = (candidate_root / "pca64_candidate" / f"fold_{fold}").resolve()
    try:
        pca_base_runner.reject_outer_artifacts(root)
    except Exception as error:
        raise TCMPCA64DeterministicSelectionError(str(error)) from error
    completion = root / CANDIDATE_COMPLETION_NAME
    metrics_path = root / "results" / "metrics.json"
    protocol_path = root / "protocol.json"
    prediction_path = root / "results" / "validation_predictions.npz"
    audit_path = root / "tcm_semantic_adapter_audit.json"
    for path in (completion, metrics_path, protocol_path, prediction_path, audit_path):
        if not path.is_file():
            raise TCMPCA64DeterministicSelectionError(
                f"candidate artifact is absent: {path}"
            )
    metrics = load_json(metrics_path)
    protocol = load_json(protocol_path)
    audit = load_json(audit_path)
    binding = protocol.get("tcm_pca64_deterministic_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != RUNNER_SCHEMA:
        raise TCMPCA64DeterministicSelectionError(
            "PCA64 deterministic runner binding is absent"
        )
    if (
        binding.get("plan", {}).get("sha256") != plan_sha
        or binding.get("candidate", {}).get("id") != "pca64_candidate"
        or dict(binding.get("fold_hyperparameters", {})) != dict(fold_plan)
        or binding.get("deterministic_initialization")
        != plan.get("deterministic_initialization")
        or binding.get("outer_test_materialised") is not False
    ):
        raise TCMPCA64DeterministicSelectionError(
            "PCA64 runner binding differs from the plan"
        )
    expected_metrics = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": 42,
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Unselected_Test_Models_Evaluated": 0,
        "Candidate_ID": "pca64_candidate",
        "Candidate_Plan_SHA256": plan_sha,
        "Semantic_Adapter": fold_plan["semantic_adapter"],
        "Feature_Basis": "deterministic_shared_pca64_v1",
        "TensorFlow_Graph_Seed": 42,
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} PCA64 metric {key} differs"
            )
    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
    ):
        raise TCMPCA64DeterministicSelectionError(
            "PCA64 candidate is not validation-only"
        )
    try:
        adapter_base_runner._validate_audit(
            audit,
            adapter=str(fold_plan["semantic_adapter"]),
            expected_steps=int(metrics.get("Optimizer_Steps", -1)),
        )
    except Exception as error:
        raise TCMPCA64DeterministicSelectionError(str(error)) from error
    audit_seed = audit.get("deterministic_initialization")
    if (
        not isinstance(audit_seed, Mapping)
        or audit_seed.get("tensorflow_graph_seed") != 42
        or audit_seed.get("seed_observed_before_model_construction") is not True
    ):
        raise TCMPCA64DeterministicSelectionError(
            "PCA64 adapter audit lacks graph-seed evidence"
        )

    table = adapter_selection._prediction_table(prediction_path)
    reference_path = verify_artifact(
        fold_plan["ae64_artifacts"]["validation_predictions"],
        label=f"fold {fold} frozen AE64 validation predictions",
    )
    reference = adapter_selection._prediction_table(reference_path)
    for field in ("row_id", "pair", "label"):
        if not np.array_equal(table[field], reference[field]):
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} PCA64 validation {field} differs from AE64"
            )
    for key, field in (
        ("validation_row_id_sha256", "row_id"),
        ("validation_pair_sha256", "pair"),
        ("validation_label_sha256", "label"),
    ):
        if adapter_selection._array_sha256(table[field]) != fold_plan[key]:
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} PCA64 validation {field} hash differs"
            )

    herb_path = adapter_selection._single_embedding(
        root, "herbedgeDHCN_herb_embedding"
    )
    disease_path = adapter_selection._single_embedding(
        root, "diseaseedgeDHCN_disease_embedding"
    )
    herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
    disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
    reproduced_scores = adapter_selection._embedding_scores(
        herb, disease, table["pair"]
    )
    if not np.allclose(reproduced_scores, table["score"], rtol=0.0, atol=1e-12):
        raise TCMPCA64DeterministicSelectionError(
            f"fold {fold} PCA64 validation scores cannot be reproduced"
        )
    recomputed = evaluate_validation_scores(table["label"], table["score"])
    for key in ("Validation_AUPRC", "Validation_AUROC"):
        if not _close(metrics.get(key), recomputed[key]):
            raise TCMPCA64DeterministicSelectionError(
                f"fold {fold} PCA64 {key} cannot be reproduced"
            )
    return {
        "id": "pca64_candidate",
        "fold": fold,
        "feature_basis": "deterministic_shared_pca64_v1",
        "semantic_adapter": fold_plan["semantic_adapter"],
        "epochs": int(fold_plan["epochs"]),
        "learning_rate": float(fold_plan["learning_rate"]),
        "validation_auprc": float(recomputed["Validation_AUPRC"]),
        "validation_auroc": float(recomputed["Validation_AUROC"]),
        "validation_f1": float(recomputed["Validation_F1"]),
        "validation_threshold": float(
            recomputed["Validation_Selected_Threshold"]
        ),
        "validation_predictions": artifact(prediction_path),
        "herb_embedding": artifact(herb_path),
        "disease_embedding": artifact(disease_path),
        "metrics": artifact(metrics_path),
        "protocol": artifact(protocol_path),
        "adapter_audit": artifact(audit_path),
        "completion_marker": artifact(completion),
    }


def select_feature_basis(
    *, ae64_validation_auprc: float, pca64_validation_auprc: float
) -> tuple[str, float]:
    try:
        return pca_base_selection.select_feature_basis(
            ae64_validation_auprc=ae64_validation_auprc,
            pca64_validation_auprc=pca64_validation_auprc,
            minimum_gain=MIN_PCA64_AUPRC_GAIN,
        )
    except Exception as error:
        raise TCMPCA64DeterministicSelectionError(str(error)) from error


def _fold_selection_payload(
    *, fold_plan: Mapping[str, Any], pca: Mapping[str, Any]
) -> dict[str, Any]:
    selected_id, gain = select_feature_basis(
        ae64_validation_auprc=float(fold_plan["ae64_validation_auprc"]),
        pca64_validation_auprc=float(pca["validation_auprc"]),
    )
    return {
        "selected_candidate": selected_id,
        "selected_feature_basis": (
            "deterministic_shared_pca64_v1"
            if selected_id == "pca64_candidate"
            else "shared_ae64"
        ),
        "semantic_adapter": fold_plan["semantic_adapter"],
        "pca64_minus_ae64_validation_auprc": gain,
        "minimum_required_gain": MIN_PCA64_AUPRC_GAIN,
        "ae64_frozen_control": {
            "adapter_selection_candidate_id": fold_plan[
                "adapter_selection_candidate_id"
            ],
            "feature_basis": "shared_ae64",
            "validation_auprc": float(fold_plan["ae64_validation_auprc"]),
            "validation_auroc": float(fold_plan["ae64_validation_auroc"]),
            "validation_threshold": float(
                fold_plan["ae64_validation_threshold"]
            ),
            "semantic_adapter": fold_plan["semantic_adapter"],
            "epochs": int(fold_plan["epochs"]),
            "learning_rate": float(fold_plan["learning_rate"]),
            "artifacts": fold_plan["ae64_artifacts"],
        },
        "pca64_candidate": dict(pca),
    }


def select_candidates(
    *,
    plan_path: Path,
    candidate_root: Path,
    pair_receipt_root: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMPCA64DeterministicSelectionError(
            f"refusing to overwrite selection or sidecar: {output}"
        )
    plan_path = plan_path.resolve(strict=True)
    plan_sha = validate_sidecar(plan_path, label="deterministic PCA64 plan")
    plan = load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != 42
        or plan.get("outer_test_accessed") is not False
    ):
        raise TCMPCA64DeterministicSelectionError(
            "unsupported deterministic PCA64 plan"
        )
    margin = float(plan.get("minimum_pca64_auprc_gain", float("nan")))
    if margin != MIN_PCA64_AUPRC_GAIN:
        raise TCMPCA64DeterministicSelectionError(
            "PCA64 advancement margin differs"
        )
    folds = plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping) or set(folds) != {
        str(fold) for fold in range(1, 6)
    }:
        raise TCMPCA64DeterministicSelectionError(
            "deterministic PCA64 plan must contain five folds"
        )
    for name, record in plan.get("source_freeze", {}).items():
        verify_artifact(record, label=f"source {name}")
    selector = Path(__file__).resolve()
    selector_record = plan.get("source_freeze", {}).get("selector")
    if (
        not isinstance(selector_record, Mapping)
        or Path(str(selector_record.get("path", ""))).resolve() != selector
        or selector_record.get("sha256") != sha256(selector)
    ):
        raise TCMPCA64DeterministicSelectionError(
            "selector source differs from the frozen plan"
        )

    candidate_root = candidate_root.resolve()
    receipts = _validate_pair_receipts(
        plan_path=plan_path,
        candidate_root=candidate_root,
        receipt_root=pair_receipt_root.resolve(),
    )
    selected_folds: dict[str, Any] = {}
    for fold in range(1, 6):
        pca = _validate_pca_candidate(
            plan=plan,
            plan_sha=plan_sha,
            candidate_root=candidate_root,
            fold=fold,
        )
        selected_folds[str(fold)] = _fold_selection_payload(
            fold_plan=folds[str(fold)], pca=pca
        )

    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "selector": artifact(selector),
        "pair_receipts": receipts,
        "deterministic_initialization": plan.get(
            "deterministic_initialization"
        ),
        "selection_metric": "Validation_AUPRC",
        "minimum_pca64_auprc_gain": margin,
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "interpretation": (
            "seed-42 same-split validation-only feature-basis diagnostic; "
            "adapter structure was frozen first"
        ),
        "folds": selected_folds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {**payload, "selection_path": str(output), "selection_sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--pair-receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = select_candidates(
            plan_path=args.plan,
            candidate_root=args.candidate_root,
            pair_receipt_root=args.pair_receipt_root,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM deterministic PCA64 v2 selection failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
