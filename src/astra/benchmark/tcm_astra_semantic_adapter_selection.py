#!/usr/bin/env python3
"""Freeze the validation-only TCM semantic-adapter decision.

This module intentionally has no outer-test evaluation command.  It validates
both adapter arms, reproduces every validation score from the saved embeddings,
checks validation-row identity against the stage-1 frozen artifact, and writes
an immutable selection plus SHA-256 sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import evaluate_validation_scores
from .tcm_astra_semantic_adapter_plan import PLAN_SCHEMA
from .tcm_astra_semantic_adapter_runner import AUDIT_SCHEMA, RUNNER_SCHEMA


SELECTION_SCHEMA = "astra-tcm-semantic-adapter-selection/v1"
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


class TCMAdapterSelectionError(RuntimeError):
    """Raised when a stage-2 candidate or selection receipt is invalid."""


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
        raise TCMAdapterSelectionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMAdapterSelectionError(f"JSON root must be an object: {path}")
    return payload


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _validate_sidecar(plan_path: Path) -> str:
    digest = sha256(plan_path)
    sidecar = plan_path.with_suffix(plan_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMAdapterSelectionError("stage-2 plan sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != plan_path.name:
        raise TCMAdapterSelectionError("stage-2 plan sidecar does not match")
    return digest


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMAdapterSelectionError(f"{label} artifact record is invalid")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    if sha256(path) != str(record.get("sha256", "")):
        raise TCMAdapterSelectionError(f"{label} changed after freeze")
    return path


def _prediction_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"row_id", "pair", "label", "score"}
        if set(artifact.files) != required:
            raise TCMAdapterSelectionError(
                f"prediction fields differ at {path}: {sorted(artifact.files)}"
            )
        return {name: np.asarray(artifact[name]) for name in sorted(required)}


def _single_embedding(root: Path, prefix: str) -> Path:
    matches = sorted(root.glob(f"{prefix}*.txt"))
    if len(matches) != 1:
        raise TCMAdapterSelectionError(
            f"expected one {prefix} embedding in {root}, found {len(matches)}"
        )
    return matches[0].resolve()


def _reject_outer_artifacts(root: Path) -> None:
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in _FORBIDDEN_OUTER_FILENAMES
    ]
    if forbidden:
        raise TCMAdapterSelectionError(
            "candidate contains an outer-test artifact: "
            + ", ".join(str(path) for path in forbidden)
        )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _embedding_scores(
    herb: np.ndarray, disease: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise TCMAdapterSelectionError("validation pairs must have shape [n, 2]")
    try:
        logits = np.einsum(
            "ij,ij->i", herb[pairs[:, 0]], disease[pairs[:, 1]]
        )
    except IndexError as error:
        raise TCMAdapterSelectionError(
            "validation pair index is outside the saved embeddings"
        ) from error
    scores = _stable_sigmoid(logits)
    if not np.isfinite(scores).all():
        raise TCMAdapterSelectionError("saved embeddings emitted non-finite scores")
    return scores


def _candidate_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = plan.get("candidates")
    if not isinstance(values, list) or len(values) != 2:
        raise TCMAdapterSelectionError("plan must contain exactly two candidates")
    result: dict[str, dict[str, Any]] = {}
    for candidate in values:
        if not isinstance(candidate, Mapping):
            raise TCMAdapterSelectionError("plan candidate is invalid")
        candidate_id = str(candidate.get("id", ""))
        if (
            not candidate_id
            or candidate_id in result
            or Path(candidate_id).name != candidate_id
        ):
            raise TCMAdapterSelectionError("candidate id is duplicate or unsafe")
        result[candidate_id] = dict(candidate)
    adapters = {value.get("semantic_adapter") for value in result.values()}
    if adapters != {"identity", "residual_mlp"}:
        raise TCMAdapterSelectionError(
            "plan must compare identity with residual_mlp"
        )
    return result


def _validate_audit(
    audit: Mapping[str, Any], *, adapter: str, expected_steps: int
) -> dict[str, str]:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("adapter") != adapter:
        raise TCMAdapterSelectionError("candidate adapter audit is incompatible")
    expected_parameters = 0 if adapter == "identity" else 8384
    if int(audit.get("adapter_parameter_count", -1)) != expected_parameters:
        raise TCMAdapterSelectionError("candidate adapter parameter count differs")
    if int(audit.get("optimizer_steps", -1)) != expected_steps:
        raise TCMAdapterSelectionError("candidate optimizer-step audit differs")
    dependency = audit.get("forward_dependency")
    if not isinstance(dependency, Mapping) or dependency.get(
        "probe_excludes_regularization"
    ) is not True:
        raise TCMAdapterSelectionError("forward-only dependency proof is absent")
    raw_dependency = dependency.get("raw_semantics")
    adapter_dependency = dependency.get("adapter")
    if not isinstance(raw_dependency, Mapping) or not all(raw_dependency.values()):
        raise TCMAdapterSelectionError("raw semantic input is disconnected")
    if not isinstance(adapter_dependency, Mapping) or not all(
        adapter_dependency.values()
    ):
        raise TCMAdapterSelectionError("adapter variable is disconnected")
    raw_trainable = audit.get("raw_semantic_trainable")
    if not isinstance(raw_trainable, Mapping) or not all(raw_trainable.values()):
        raise TCMAdapterSelectionError("raw semantic inputs are not trainable")
    initial = audit.get("initial")
    final = audit.get("final")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise TCMAdapterSelectionError("adapter snapshots are absent")
    initial_delta = initial.get("semantic_delta")
    if not isinstance(initial_delta, Mapping) or any(
        float(value) != 0.0 for value in initial_delta.values()
    ):
        raise TCMAdapterSelectionError("adapter is not identity-initialised")
    shared_hashes = initial.get("shared_variable_sha256")
    if not isinstance(shared_hashes, Mapping) or not shared_hashes:
        raise TCMAdapterSelectionError("shared initialisation hashes are absent")
    if adapter == "residual_mlp":
        if audit.get("adapter_parameters_changed") is not True:
            raise TCMAdapterSelectionError("residual adapter never updated")
        final_delta = final.get("semantic_delta")
        if not isinstance(final_delta, Mapping) or not any(
            float(value) > 0.0 for value in final_delta.values()
        ):
            raise TCMAdapterSelectionError("residual adapter did not alter semantics")
    return {str(key): str(value) for key, value in shared_hashes.items()}


def _validate_candidate(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    candidate_root: Path,
    candidate: Mapping[str, Any],
    fold: int,
    herb_shape: tuple[int, int],
    disease_shape: tuple[int, int],
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    adapter = str(candidate["semantic_adapter"])
    root = (candidate_root / candidate_id / f"fold_{fold}").resolve()
    metrics_path = root / "results" / "metrics.json"
    protocol_path = root / "protocol.json"
    prediction_path = root / "results" / "validation_predictions.npz"
    audit_path = root / "tcm_semantic_adapter_audit.json"
    for path in (metrics_path, protocol_path, prediction_path, audit_path):
        if not path.is_file():
            raise TCMAdapterSelectionError(f"candidate artifact is missing: {path}")
    _reject_outer_artifacts(root)

    metrics = _load_json(metrics_path)
    protocol = _load_json(protocol_path)
    audit = _load_json(audit_path)
    binding = protocol.get("tcm_semantic_adapter_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != RUNNER_SCHEMA:
        raise TCMAdapterSelectionError("candidate runner binding is absent")
    expected_metrics = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": 42,
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Unselected_Test_Models_Evaluated": 0,
        "Candidate_ID": candidate_id,
        "Candidate_Plan_SHA256": plan_sha256,
        "Semantic_Adapter": adapter,
        "Tuning_Stage": "semantic-adapter",
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise TCMAdapterSelectionError(
                f"{candidate_id} fold {fold} metric {key} differs"
            )
    bound_plan = binding.get("plan")
    if not isinstance(bound_plan, Mapping) or bound_plan.get("sha256") != plan_sha256:
        raise TCMAdapterSelectionError("candidate is bound to another plan")
    if dict(binding.get("candidate", {})) != dict(candidate):
        raise TCMAdapterSelectionError("candidate binding differs from the plan")
    folds = plan["fold_hyperparameters"]
    expected_fold = dict(folds[str(fold)])
    if dict(binding.get("fold_hyperparameters", {})) != expected_fold:
        raise TCMAdapterSelectionError("candidate fold hyperparameters differ")
    if binding.get("outer_test_materialised") is not False:
        raise TCMAdapterSelectionError("candidate binding reports outer-test access")

    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
    ):
        raise TCMAdapterSelectionError("candidate protocol is not validation-only")
    hyperparameters = protocol.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise TCMAdapterSelectionError("candidate hyperparameters are absent")
    expected_hyperparameters = {
        "epochs": int(expected_fold["epochs"]),
        "learning_rate": float(expected_fold["learning_rate"]),
        "batch_size": 2000,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": 1.0,
        "semantic_adapter": adapter,
        "semantic_input_dim": 64,
        "semantic_mlp_hidden_dim": 32,
        "semantic_freeze_input": False,
        "smoke_override": False,
        "diagnostic_mode": False,
    }
    for key, expected in expected_hyperparameters.items():
        observed = hyperparameters.get(key)
        if isinstance(expected, float):
            matches = _close(observed, expected, tolerance=1e-15)
        else:
            matches = observed == expected
        if not matches:
            raise TCMAdapterSelectionError(
                f"candidate hyperparameter {key} differs: {observed!r}"
            )

    audit_record = binding.get("adapter_audit")
    if not isinstance(audit_record, Mapping) or Path(
        str(audit_record.get("path", ""))
    ).resolve() != audit_path.resolve() or sha256(audit_path) != audit_record.get(
        "sha256"
    ):
        raise TCMAdapterSelectionError("candidate adapter audit binding differs")
    shared_hashes = _validate_audit(
        audit,
        adapter=adapter,
        expected_steps=int(metrics["Optimizer_Steps"]),
    )

    source_freeze = plan.get("source_freeze")
    if dict(binding.get("source_freeze", {})) != dict(source_freeze):
        raise TCMAdapterSelectionError("candidate source freeze differs from plan")
    for name, record in source_freeze.items():
        _verify_artifact(record, label=f"source {name}")
    runner_record = binding.get("runner")
    model_record = binding.get("model")
    runner_path = _verify_artifact(runner_record, label="candidate runner")
    model_path = _verify_artifact(model_record, label="candidate model")

    table = _prediction_table(prediction_path)
    reference_record = expected_fold["stage1_winner_artifacts"][
        "validation_predictions"
    ]
    reference_path = _verify_artifact(
        reference_record, label=f"fold {fold} reference validation rows"
    )
    reference = _prediction_table(reference_path)
    for field in ("row_id", "pair", "label"):
        if not np.array_equal(table[field], reference[field]):
            raise TCMAdapterSelectionError(
                f"{candidate_id} fold {fold} validation {field} differs "
                "from the frozen stage-1 rows"
            )

    herb_path = _single_embedding(root, "herbedgeDHCN_herb_embedding")
    disease_path = _single_embedding(root, "diseaseedgeDHCN_disease_embedding")
    herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
    disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
    if herb.shape != herb_shape or disease.shape != disease_shape:
        raise TCMAdapterSelectionError(
            f"candidate embedding shapes differ: {herb.shape}, {disease.shape}"
        )
    reproduced_scores = _embedding_scores(herb, disease, table["pair"])
    if not np.allclose(reproduced_scores, table["score"], rtol=0.0, atol=1e-12):
        difference = float(np.max(np.abs(reproduced_scores - table["score"])))
        raise TCMAdapterSelectionError(
            f"candidate validation scores do not match embeddings; max diff={difference}"
        )
    recomputed = evaluate_validation_scores(table["label"], table["score"])
    for key in ("Validation_AUPRC", "Validation_AUROC"):
        if not _close(metrics.get(key), recomputed.get(key)):
            raise TCMAdapterSelectionError(
                f"candidate metric {key} cannot be reproduced"
            )

    return {
        "id": candidate_id,
        "fold": fold,
        "semantic_adapter": adapter,
        "epochs": int(expected_fold["epochs"]),
        "learning_rate": float(expected_fold["learning_rate"]),
        "validation_auprc": float(recomputed["Validation_AUPRC"]),
        "validation_auroc": float(recomputed["Validation_AUROC"]),
        "validation_f1": float(recomputed["Validation_F1"]),
        "validation_threshold": float(
            recomputed["Validation_Selected_Threshold"]
        ),
        "validation_predictions": str(prediction_path.resolve()),
        "validation_predictions_sha256": sha256(prediction_path),
        "validation_row_id_sha256": _array_sha256(table["row_id"]),
        "validation_pair_sha256": _array_sha256(table["pair"]),
        "validation_label_sha256": _array_sha256(table["label"]),
        "herb_embedding": str(herb_path),
        "herb_embedding_sha256": sha256(herb_path),
        "disease_embedding": str(disease_path),
        "disease_embedding_sha256": sha256(disease_path),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "adapter_audit": str(audit_path.resolve()),
        "adapter_audit_sha256": sha256(audit_path),
        "shared_initial_variable_sha256": shared_hashes,
        "runner_path": str(runner_path),
        "runner_sha256": str(runner_record["sha256"]),
        "model_path": str(model_path),
        "model_sha256": str(model_record["sha256"]),
    }


def _winner(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise TCMAdapterSelectionError("cannot select from an empty candidate set")
    return min(
        records,
        key=lambda record: (
            -float(record["validation_auprc"]),
            -float(record["validation_auroc"]),
            0 if record["semantic_adapter"] == "identity" else 1,
            str(record["id"]),
        ),
    )


def select_candidates(
    *, plan_path: Path, candidate_root: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    output_sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or output_sidecar.exists():
        raise TCMAdapterSelectionError(
            f"refusing to overwrite selection or sidecar: {output}"
        )
    plan_path = plan_path.resolve(strict=True)
    plan_sha = _validate_sidecar(plan_path)
    plan = _load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise TCMAdapterSelectionError("unsupported stage-2 plan schema")
    if plan.get("status") != "frozen-before-stage2-candidate-testing":
        raise TCMAdapterSelectionError("stage-2 plan is not frozen")
    candidate_by_id = _candidate_map(plan)
    folds = plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping) or set(folds) != {"1", "2", "3", "4", "5"}:
        raise TCMAdapterSelectionError("stage-2 plan must contain five folds")
    bundle_manifest_path = _verify_artifact(
        plan["feature_bundle"]["manifest"], label="feature bundle manifest"
    )
    bundle_manifest = _load_json(bundle_manifest_path)
    try:
        herb_shape = tuple(
            int(value) for value in bundle_manifest["entities"]["herb"]["shape"]
        )
        disease_shape = tuple(
            int(value)
            for value in bundle_manifest["entities"]["disease"]["shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TCMAdapterSelectionError(
            "feature bundle lacks herb/disease shapes"
        ) from error
    if len(herb_shape) != 2 or len(disease_shape) != 2:
        raise TCMAdapterSelectionError("feature bundle shapes must be matrices")

    candidate_root = candidate_root.resolve()
    fold_payload: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for fold in range(1, 6):
        records = [
            _validate_candidate(
                plan=plan,
                plan_sha256=plan_sha,
                candidate_root=candidate_root,
                candidate=candidate,
                fold=fold,
                herb_shape=herb_shape,
                disease_shape=disease_shape,
            )
            for candidate in candidate_by_id.values()
        ]
        all_records.extend(records)
        shared_hash_sets = {
            json.dumps(
                record["shared_initial_variable_sha256"], sort_keys=True
            )
            for record in records
        }
        if len(shared_hash_sets) != 1:
            raise TCMAdapterSelectionError(
                f"fold {fold} adapter arms have different shared initialization"
            )
        row_identities = {
            (
                record["validation_row_id_sha256"],
                record["validation_pair_sha256"],
                record["validation_label_sha256"],
            )
            for record in records
        }
        if len(row_identities) != 1:
            raise TCMAdapterSelectionError(
                f"fold {fold} adapter arms use different validation rows"
            )
        selected = dict(_winner(records))
        fold_payload[str(fold)] = {
            "selected_candidate": selected["id"],
            "selected_adapter": selected["semantic_adapter"],
            "selected_validation_auprc": selected["validation_auprc"],
            "selected_validation_auroc": selected["validation_auroc"],
            "selected_validation_threshold": selected[
                "validation_threshold"
            ],
            "winner": selected,
            "candidates": records,
        }

    for field in ("runner_sha256", "model_sha256"):
        observed = {record[field] for record in all_records}
        if len(observed) != 1 or not next(iter(observed)):
            raise TCMAdapterSelectionError(
                f"candidates do not share one non-empty {field}"
            )
    selector_path = Path(__file__).resolve()
    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "selector": {
            "path": str(selector_path),
            "sha256": sha256(selector_path),
        },
        "selection_metric": "Validation_AUPRC",
        "tie_break": plan.get("tie_break"),
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "interpretation": (
            "seed-42 same-split post-baseline development diagnostic"
        ),
        "folds": fold_payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256(output)
    output_sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
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
        print(f"TCM semantic-adapter selection failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
