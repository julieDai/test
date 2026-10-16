#!/usr/bin/env python3
"""Run one deterministic PCA64 feature-basis candidate on validation only."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

from . import astra_runner
from . import tcm_astra_pca64_control_runner as pca_base_runner
from . import tcm_astra_semantic_adapter_runner as adapter_base_runner
from . import tcm_astra_semantic_adapter_deterministic_runner as deterministic_runner
from .tcm_astra_pca64_deterministic_plan import (
    EXPECTED_FIXED,
    EXPECTED_SHARED_INITIAL_KEYS,
    PLAN_SCHEMA,
    PLAN_STATUS,
    artifact,
    expected_adapter_keys,
    load_json,
    protocol_controls,
    sha256,
    validate_sidecar,
    verify_artifact,
)


RUNNER_SCHEMA = "astra-tcm-pca64-deterministic-candidate/v2"
CANDIDATE_COMPLETION_NAME = ".astra_pca64_deterministic_complete"


class TCMPCA64DeterministicRunnerError(RuntimeError):
    """Raised when a deterministic PCA64 validation candidate is invalid."""


def _float_equal(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_source_freeze(records: object) -> dict[str, dict[str, Any]]:
    if not isinstance(records, Mapping) or not records:
        raise TCMPCA64DeterministicRunnerError("source freeze is absent")
    verified: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        path = verify_artifact(record, label=f"source {name}")
        verified[str(name)] = {**dict(record), "path": str(path)}
    if set(verified) != set(records):
        raise TCMPCA64DeterministicRunnerError(
            "one or more source records are invalid"
        )
    return verified


def _verify_parent_chain(plan: Mapping[str, Any]) -> None:
    selection_path = verify_artifact(
        plan.get("parent_adapter_selection"), label="parent adapter selection"
    )
    validate_sidecar(selection_path, label="parent adapter selection")
    parent_plan = verify_artifact(
        plan.get("parent_adapter_plan"), label="parent adapter plan"
    )
    validate_sidecar(parent_plan, label="parent adapter plan")
    verify_artifact(
        plan.get("parent_adapter_launcher_completion"),
        label="parent adapter launcher completion",
    )
    receipts = plan.get("parent_adapter_pair_receipts")
    if not isinstance(receipts, list) or len(receipts) != 5:
        raise TCMPCA64DeterministicRunnerError(
            "parent adapter pair-receipt binding is incomplete"
        )
    for fold, record in enumerate(receipts, 1):
        if not isinstance(record, Mapping) or int(record.get("fold", -1)) != fold:
            raise TCMPCA64DeterministicRunnerError(
                "parent adapter pair-receipt order differs"
            )
        receipt = verify_artifact(
            record.get("receipt"), label=f"parent adapter fold {fold} receipt"
        )
        sidecar = verify_artifact(
            record.get("sidecar"), label=f"parent adapter fold {fold} sidecar"
        )
        if sidecar != receipt.with_suffix(receipt.suffix + ".sha256"):
            raise TCMPCA64DeterministicRunnerError(
                f"parent adapter fold {fold} sidecar path differs"
            )
        validate_sidecar(receipt, label=f"parent adapter fold {fold} receipt")


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if astra_runner._normalise_dataset(args.dataset) != "tcm-suite":
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 control is TCM-suite only"
        )
    if not bool(args.validation_only):
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 candidate requires --validation-only"
        )
    if bool(args.smoke) or bool(args.diagnostic):
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 candidate rejects smoke/diagnostic mode"
        )
    if not _float_equal(args.text_fusion_alpha, 1.0):
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 control fixes text_fusion_alpha=1"
        )
    if args.candidate_id != "pca64_candidate":
        raise TCMPCA64DeterministicRunnerError(
            "AE64 is a frozen v3 winner and must not be re-run"
        )

    plan_path = args.pca64_plan.resolve(strict=True)
    plan_sha = validate_sidecar(plan_path, label="deterministic PCA64 plan")
    plan = load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != int(args.seed)
        or plan.get("outer_test_accessed") is not False
        or plan.get("outer_test_evaluator_in_this_module") is not False
    ):
        raise TCMPCA64DeterministicRunnerError(
            "unsupported deterministic PCA64 plan"
        )
    _verify_parent_chain(plan)
    verified_sources = _verify_source_freeze(plan.get("source_freeze"))
    protocol_path = verify_artifact(
        plan.get("protocol_manifest"), label="protocol manifest"
    )
    if protocol_path != args.protocol_manifest.resolve():
        raise TCMPCA64DeterministicRunnerError(
            "CLI protocol differs from the frozen plan"
        )

    bundles = plan.get("feature_bundles")
    pca_record = bundles.get("pca64") if isinstance(bundles, Mapping) else None
    if not isinstance(pca_record, Mapping):
        raise TCMPCA64DeterministicRunnerError("PCA64 bundle binding is absent")
    bundle_root = Path(str(pca_record.get("path", ""))).resolve(strict=True)
    manifest_path = verify_artifact(
        pca_record.get("manifest"), label="PCA64 bundle manifest"
    )
    if manifest_path.parent != bundle_root or bundle_root != args.embedding_dir.resolve():
        raise TCMPCA64DeterministicRunnerError(
            "CLI PCA64 bundle differs from the frozen plan"
        )

    candidates = plan.get("candidates")
    matches = (
        [
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and candidate.get("id") == args.candidate_id
        ]
        if isinstance(candidates, list)
        else []
    )
    if len(matches) != 1 or matches[0] != {
        "id": "pca64_candidate",
        "feature_basis": "deterministic_shared_pca64_v1",
        "execution": "validation_only_candidate",
        "feature_bundle_key": "pca64",
    }:
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 executable candidate differs"
        )

    folds = plan.get("fold_hyperparameters")
    fold = folds.get(str(args.fold)) if isinstance(folds, Mapping) else None
    if not isinstance(fold, Mapping):
        raise TCMPCA64DeterministicRunnerError(f"plan has no fold {args.fold}")
    try:
        epochs = int(fold["epochs"])
        learning_rate = float(fold["learning_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMPCA64DeterministicRunnerError(
            "fold hyperparameters are invalid"
        ) from error
    adapter = str(fold.get("semantic_adapter", ""))
    expected_adapter_keys(adapter)
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise TCMPCA64DeterministicRunnerError(
            "fold adapter/epoch/lr contract is invalid"
        )
    if dict(plan.get("fixed", {})) != EXPECTED_FIXED:
        raise TCMPCA64DeterministicRunnerError("fixed controls differ")
    deterministic = plan.get("deterministic_initialization")
    expected_deterministic = deterministic_runner._expected_determinism(int(args.seed))
    if deterministic != expected_deterministic:
        raise TCMPCA64DeterministicRunnerError(
            "TensorFlow graph-seed contract differs from deterministic v3"
        )
    controls = fold.get("ae64_protocol_controls")
    if not isinstance(controls, Mapping):
        raise TCMPCA64DeterministicRunnerError(
            "frozen AE64 protocol controls are absent"
        )
    runtime = controls.get("runtime")
    expected_gpu = (
        str(runtime.get("cuda_visible_devices", ""))
        if isinstance(runtime, Mapping)
        else ""
    )
    if not expected_gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 candidate must use the selected AE64 winner's CUDA device"
        )
    if str(args.device).strip().lower() != "cuda":
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 control requires the frozen CUDA runtime"
        )
    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "candidate": matches[0],
        "fold_hyperparameters": dict(fold),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "adapter": adapter,
        "deterministic_initialization": dict(deterministic),
        "pca64_manifest": manifest_path,
        "verified_sources": verified_sources,
    }


@contextmanager
def deterministic_pca64_runtime(
    binding: Mapping[str, Any],
) -> Iterator[dict[str, object]]:
    """Compose the audited PCA bundle with the exact v3 seeded model runtime."""

    with pca_base_runner._pca64_bundle_runtime(binding):
        with deterministic_runner._candidate_runtime(binding) as seed_receipt:
            yield seed_receipt


def _validate_initial_key_sets(audit: Mapping[str, Any], *, adapter: str) -> None:
    initial = audit.get("initial")
    if not isinstance(initial, Mapping):
        raise TCMPCA64DeterministicRunnerError("initial adapter audit is absent")
    shared = initial.get("shared_variable_sha256")
    adapter_hashes = initial.get("adapter_variable_sha256")
    if not isinstance(shared, Mapping) or set(shared) != set(
        EXPECTED_SHARED_INITIAL_KEYS
    ):
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 shared initialization key set differs"
        )
    if not isinstance(adapter_hashes, Mapping) or set(adapter_hashes) != set(
        expected_adapter_keys(adapter)
    ):
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 adapter initialization key set differs"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    with deterministic_pca64_runtime(binding) as seed_receipt:
        astra_runner.run(args)

    output_root = args.output_root.resolve()
    metrics_path = output_root / "results" / "metrics.json"
    protocol_path = output_root / "protocol.json"
    audit_path = output_root / "tcm_semantic_adapter_audit.json"
    if sha256(Path(str(binding["plan_path"]))) != binding["plan_sha256"]:
        raise TCMPCA64DeterministicRunnerError(
            "deterministic PCA64 plan changed during the run"
        )
    pca_base_runner.reject_outer_artifacts(output_root)
    stored_metrics = dict(load_json(metrics_path))
    stored_protocol = dict(load_json(protocol_path))
    audit = dict(load_json(audit_path))
    expected_seed = int(
        binding["deterministic_initialization"]["tensorflow_graph_seed"]
    )
    try:
        deterministic_runner._validate_seed_evidence(
            audit=audit,
            seed_receipt=seed_receipt,
            expected_seed=expected_seed,
        )
        adapter_base_runner._validate_audit(
            audit,
            adapter=str(binding["adapter"]),
            expected_steps=int(stored_metrics.get("Optimizer_Steps", -1)),
        )
    except Exception as error:
        raise TCMPCA64DeterministicRunnerError(str(error)) from error
    _validate_initial_key_sets(audit, adapter=str(binding["adapter"]))

    evaluation = stored_protocol.get("evaluation")
    split_rows = stored_protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
        or stored_metrics.get("Validation_Only") is not True
        or int(stored_metrics.get("Test_Models_Evaluated", -1)) != 0
        or int(stored_metrics.get("Unselected_Test_Models_Evaluated", -1)) != 0
    ):
        raise TCMPCA64DeterministicRunnerError(
            "central runner did not remain validation-only"
        )
    observed_controls = protocol_controls(stored_protocol)
    expected_controls = binding["fold_hyperparameters"].get(
        "ae64_protocol_controls"
    )
    if observed_controls != expected_controls:
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 runtime/split/relation/step controls differ from AE64"
        )
    feature_bundle = stored_protocol.get("feature_bundle")
    frozen_bundle = binding["plan"]["feature_bundles"]["pca64"]
    if (
        not isinstance(feature_bundle, Mapping)
        or feature_bundle.get("sha256")
        != frozen_bundle.get("manifest", {}).get("sha256")
        or feature_bundle.get("schema")
        != "astra-tcm-pca64-feature-bundle/v1"
    ):
        raise TCMPCA64DeterministicRunnerError(
            "runtime PCA64 bundle differs from the frozen plan"
        )
    if dict(audit.get("model_source", {})) != dict(
        binding["fold_hyperparameters"].get("ae64_model_source", {})
    ):
        raise TCMPCA64DeterministicRunnerError(
            "PCA64 model source differs from the selected AE64 winner"
        )

    runner_path = Path(__file__).resolve()
    provenance = {
        "schema": RUNNER_SCHEMA,
        "plan": {
            "path": binding["plan_path"],
            "sha256": binding["plan_sha256"],
        },
        "candidate": dict(binding["candidate"]),
        "fold_hyperparameters": dict(binding["fold_hyperparameters"]),
        "runner": artifact(runner_path),
        "model": dict(audit["model_source"]),
        "base_model": dict(audit["base_model_source"]),
        "pca64_bundle_manifest": artifact(Path(binding["pca64_manifest"])),
        "adapter_audit": artifact(audit_path),
        "source_freeze": dict(binding["verified_sources"]),
        "deterministic_initialization": dict(
            binding["deterministic_initialization"]
        ),
        "tensorflow_graph_seed_receipt": dict(seed_receipt),
        "selection_metric": "Validation_AUPRC",
        "outer_test_materialised": False,
    }
    stored_metrics.update(
        {
            "Method": "ASTRA-TCM-PCA64-deterministic-control-candidate",
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Semantic_Adapter": binding["adapter"],
            "Semantic_Adapter_Parameters": audit["adapter_parameter_count"],
            "Feature_Basis": "deterministic_shared_pca64_v1",
            "TensorFlow_Graph_Seed": expected_seed,
            "Tuning_Stage": "semantic-feature-basis-control",
            "Execution_Mode": "pca64-deterministic-validation-candidate",
            "Evaluation_Mode": "validation-only-pca64-deterministic-control",
        }
    )
    stored_protocol["tcm_pca64_deterministic_candidate"] = provenance
    model_sources = dict(stored_protocol.get("model_sources", {}))
    model_sources["tcm_semantic_adapter_deterministic"] = dict(
        audit["model_source"]
    )
    stored_protocol["model_sources"] = model_sources
    _write_json(metrics_path, stored_metrics)
    _write_json(protocol_path, stored_protocol)
    return stored_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = astra_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--pca64-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:
        print(
            f"TCM deterministic PCA64 v2 candidate failed: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
