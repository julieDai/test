#!/usr/bin/env python3
"""Select PCA64 per fold using only frozen inner-validation evidence.

This module intentionally has no outer-test evaluator.  PCA64 advances only
when its validation AUPRC exceeds the frozen AE64 reference by at least the
margin written into the plan (v1: 0.0001).
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import evaluate_validation_scores
from .tcm_astra_pca64_control_plan import (
    MIN_PCA64_AUPRC_GAIN,
    PLAN_SCHEMA,
)
from .tcm_astra_pca64_control_runner import RUNNER_SCHEMA, reject_outer_artifacts
from .tcm_astra_semantic_adapter_runner import _validate_audit
from .tcm_astra_semantic_adapter_selection import (
    _array_sha256,
    _embedding_scores,
    _prediction_table,
    _single_embedding,
)


SELECTION_SCHEMA = "astra-tcm-pca64-control-selection/v1"


class TCMPCA64ControlSelectionError(RuntimeError):
    """Raised when PCA64 validation evidence is incomplete or inconsistent."""


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
        raise TCMPCA64ControlSelectionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMPCA64ControlSelectionError(f"JSON root must be an object: {path}")
    return payload


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMPCA64ControlSelectionError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except OSError as error:
        raise TCMPCA64ControlSelectionError(f"{label} artifact is absent") from error
    if sha256(path) != record.get("sha256"):
        raise TCMPCA64ControlSelectionError(f"{label} artifact changed")
    if "bytes" in record and path.stat().st_size != record.get("bytes"):
        raise TCMPCA64ControlSelectionError(f"{label} byte count changed")
    return path


def _validate_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMPCA64ControlSelectionError("PCA64 plan sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise TCMPCA64ControlSelectionError("PCA64 plan sidecar differs")
    return digest


def select_feature_basis(
    *,
    ae64_validation_auprc: float,
    pca64_validation_auprc: float,
    minimum_gain: float = MIN_PCA64_AUPRC_GAIN,
) -> tuple[str, float]:
    """Apply the predeclared one-sided margin without rounding or tolerance."""

    values = (ae64_validation_auprc, pca64_validation_auprc, minimum_gain)
    if not all(math.isfinite(float(value)) for value in values) or minimum_gain < 0.0:
        raise TCMPCA64ControlSelectionError("selection values must be finite")
    try:
        decimal_gain = Decimal(str(pca64_validation_auprc)) - Decimal(
            str(ae64_validation_auprc)
        )
        decimal_margin = Decimal(str(minimum_gain))
    except InvalidOperation as error:
        raise TCMPCA64ControlSelectionError(
            "selection values cannot be represented as finite decimals"
        ) from error
    gain = float(decimal_gain)
    selected = (
        "pca64_candidate"
        if decimal_gain >= decimal_margin
        else "ae64_frozen_control"
    )
    return selected, gain


def _fold_selection_payload(
    *,
    fold_plan: Mapping[str, Any],
    pca: Mapping[str, Any],
    margin: float,
) -> dict[str, Any]:
    """Build the canonical fold decision used by selection and outer replay."""

    ae_auprc = float(fold_plan["ae64_validation_auprc"])
    selected_id, gain = select_feature_basis(
        ae64_validation_auprc=ae_auprc,
        pca64_validation_auprc=float(pca["validation_auprc"]),
        minimum_gain=margin,
    )
    return {
        "selected_candidate": selected_id,
        "selected_feature_basis": (
            pca["feature_basis"]
            if selected_id == "pca64_candidate"
            else "shared_ae64"
        ),
        "pca64_minus_ae64_validation_auprc": gain,
        "minimum_required_gain": margin,
        "ae64_frozen_control": {
            "validation_auprc": ae_auprc,
            "validation_auroc": float(fold_plan["ae64_validation_auroc"]),
            "semantic_adapter": fold_plan["semantic_adapter"],
            "epochs": int(fold_plan["epochs"]),
            "learning_rate": float(fold_plan["learning_rate"]),
            "artifacts": fold_plan["ae64_artifacts"],
        },
        "pca64_candidate": dict(pca),
    }


def _validate_pca_candidate(
    *,
    plan: Mapping[str, Any],
    plan_sha: str,
    candidate_root: Path,
    fold: int,
) -> dict[str, Any]:
    fold_plan = plan["fold_hyperparameters"][str(fold)]
    root = (candidate_root / "pca64_candidate" / f"fold_{fold}").resolve()
    reject_outer_artifacts(root)
    metrics_path = root / "results" / "metrics.json"
    protocol_path = root / "protocol.json"
    prediction_path = root / "results" / "validation_predictions.npz"
    audit_path = root / "tcm_semantic_adapter_audit.json"
    for path in (metrics_path, protocol_path, prediction_path, audit_path):
        if not path.is_file():
            raise TCMPCA64ControlSelectionError(f"candidate artifact is absent: {path}")
    metrics = _load_json(metrics_path)
    protocol = _load_json(protocol_path)
    audit = _load_json(audit_path)
    binding = protocol.get("tcm_pca64_control_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != RUNNER_SCHEMA:
        raise TCMPCA64ControlSelectionError("PCA64 runner binding is absent")
    if (
        binding.get("plan", {}).get("sha256") != plan_sha
        or binding.get("candidate", {}).get("id") != "pca64_candidate"
        or dict(binding.get("fold_hyperparameters", {})) != dict(fold_plan)
        or binding.get("outer_test_materialised") is not False
    ):
        raise TCMPCA64ControlSelectionError("PCA64 runner binding differs from plan")
    expected_metrics = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": 42,
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Candidate_ID": "pca64_candidate",
        "Candidate_Plan_SHA256": plan_sha,
        "Semantic_Adapter": fold_plan["semantic_adapter"],
        "Feature_Basis": "deterministic_shared_pca64_v1",
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise TCMPCA64ControlSelectionError(
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
        raise TCMPCA64ControlSelectionError("PCA64 candidate is not validation-only")
    _validate_audit(
        audit,
        adapter=str(fold_plan["semantic_adapter"]),
        expected_steps=int(metrics.get("Optimizer_Steps", -1)),
    )

    table = _prediction_table(prediction_path)
    reference_path = _verify_artifact(
        fold_plan["ae64_artifacts"]["validation_predictions"],
        label=f"fold {fold} frozen AE64 validation predictions",
    )
    reference = _prediction_table(reference_path)
    for field in ("row_id", "pair", "label"):
        if not np.array_equal(table[field], reference[field]):
            raise TCMPCA64ControlSelectionError(
                f"fold {fold} PCA64 validation {field} differs from AE64"
            )
    for key, field in (
        ("validation_row_id_sha256", "row_id"),
        ("validation_pair_sha256", "pair"),
        ("validation_label_sha256", "label"),
    ):
        if _array_sha256(table[field]) != fold_plan[key]:
            raise TCMPCA64ControlSelectionError(
                f"fold {fold} PCA64 validation {field} hash differs"
            )

    herb_path = _single_embedding(root, "herbedgeDHCN_herb_embedding")
    disease_path = _single_embedding(root, "diseaseedgeDHCN_disease_embedding")
    herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
    disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
    reproduced_scores = _embedding_scores(herb, disease, table["pair"])
    if not np.allclose(reproduced_scores, table["score"], rtol=0.0, atol=1e-12):
        raise TCMPCA64ControlSelectionError(
            f"fold {fold} PCA64 validation scores cannot be reproduced"
        )
    recomputed = evaluate_validation_scores(table["label"], table["score"])
    for key in ("Validation_AUPRC", "Validation_AUROC"):
        if not math.isclose(
            float(metrics.get(key, float("nan"))),
            float(recomputed[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TCMPCA64ControlSelectionError(
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
        "validation_predictions": str(prediction_path),
        "validation_predictions_sha256": sha256(prediction_path),
        "herb_embedding": str(herb_path),
        "herb_embedding_sha256": sha256(herb_path),
        "disease_embedding": str(disease_path),
        "disease_embedding_sha256": sha256(disease_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "adapter_audit": str(audit_path),
        "adapter_audit_sha256": sha256(audit_path),
    }


def select_candidates(
    *, plan_path: Path, candidate_root: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMPCA64ControlSelectionError(
            f"refusing to overwrite selection or sidecar: {output}"
        )
    plan_path = plan_path.resolve(strict=True)
    plan_sha = _validate_sidecar(plan_path)
    plan = _load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen-before-pca64-validation-candidates"
        or plan.get("outer_test_accessed") is not False
    ):
        raise TCMPCA64ControlSelectionError("unsupported PCA64 control plan")
    margin = float(plan.get("minimum_pca64_auprc_gain", float("nan")))
    if margin != MIN_PCA64_AUPRC_GAIN:
        raise TCMPCA64ControlSelectionError("PCA64 advancement margin differs")
    folds = plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping) or set(folds) != {
        "1", "2", "3", "4", "5"
    }:
        raise TCMPCA64ControlSelectionError("PCA64 plan must contain five folds")

    candidate_root = candidate_root.resolve()
    selected_folds: dict[str, Any] = {}
    for fold in range(1, 6):
        fold_plan = folds[str(fold)]
        pca = _validate_pca_candidate(
            plan=plan,
            plan_sha=plan_sha,
            candidate_root=candidate_root,
            fold=fold,
        )
        selected_folds[str(fold)] = _fold_selection_payload(
            fold_plan=fold_plan,
            pca=pca,
            margin=margin,
        )

    selector_path = Path(__file__).resolve()
    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "selector": {"path": str(selector_path), "sha256": sha256(selector_path)},
        "selection_metric": "Validation_AUPRC",
        "minimum_pca64_auprc_gain": margin,
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "interpretation": "seed-42 same-split post-baseline validation diagnostic",
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = select_candidates(
            plan_path=args.plan,
            candidate_root=args.candidate_root,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM PCA64 control selection failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
