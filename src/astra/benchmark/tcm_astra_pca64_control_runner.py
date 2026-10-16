#!/usr/bin/env python3
"""Run one frozen PCA64 feature-basis candidate on inner validation only."""

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
from .tcm_astra_pca64_control_plan import PLAN_SCHEMA
from .tcm_astra_semantic_adapter_runner import (
    _candidate_runtime as _adapter_runtime,
    _validate_audit as _validate_adapter_audit,
)
from .tcm_pca64_feature_bundle import (
    PCA64_BUNDLE_SCHEMA,
    TCMPCA64BundleError,
    validate_tcm_pca64_feature_bundle,
)


RUNNER_SCHEMA = "astra-tcm-pca64-control-candidate/v1"
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


class TCMPCA64ControlRunnerError(RuntimeError):
    """Raised when the PCA64 candidate violates its frozen contract."""


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
        raise TCMPCA64ControlRunnerError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMPCA64ControlRunnerError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMPCA64ControlRunnerError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except OSError as error:
        raise TCMPCA64ControlRunnerError(f"{label} artifact is absent") from error
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise TCMPCA64ControlRunnerError(f"{label} artifact changed after freeze")
    if "bytes" in record and path.stat().st_size != record.get("bytes"):
        raise TCMPCA64ControlRunnerError(f"{label} byte count changed after freeze")
    return path


def _validate_sidecar(plan_path: Path) -> str:
    digest = sha256(plan_path)
    sidecar = plan_path.with_suffix(plan_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMPCA64ControlRunnerError("PCA64 control plan sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != plan_path.name:
        raise TCMPCA64ControlRunnerError("PCA64 control plan sidecar differs")
    return digest


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
        raise TCMPCA64ControlRunnerError(
            "PCA64 validation candidate contains an outer-test artifact: "
            + ", ".join(str(path) for path in forbidden)
        )


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if astra_runner._normalise_dataset(args.dataset) != "tcm-suite":
        raise TCMPCA64ControlRunnerError("PCA64 control is TCM-suite only")
    if not bool(args.validation_only):
        raise TCMPCA64ControlRunnerError("PCA64 candidate requires --validation-only")
    if bool(args.smoke) or bool(args.diagnostic):
        raise TCMPCA64ControlRunnerError("PCA64 candidate rejects smoke/diagnostic mode")
    if not _float_equal(args.text_fusion_alpha, 1.0):
        raise TCMPCA64ControlRunnerError("PCA64 control fixes text_fusion_alpha=1")
    if args.candidate_id != "pca64_candidate":
        raise TCMPCA64ControlRunnerError(
            "AE64 is a frozen reference and must not be re-run"
        )

    plan_path = args.control_plan.resolve(strict=True)
    plan_sha = _validate_sidecar(plan_path)
    plan = _load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen-before-pca64-validation-candidates"
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != args.seed
        or plan.get("outer_test_accessed") is not False
    ):
        raise TCMPCA64ControlRunnerError("unsupported PCA64 control plan")
    protocol_path = _verify_artifact(
        plan.get("protocol_manifest"), label="protocol manifest"
    )
    if protocol_path != args.protocol_manifest.resolve():
        raise TCMPCA64ControlRunnerError("CLI protocol differs from frozen plan")

    bundles = plan.get("feature_bundles")
    pca_record = bundles.get("pca64") if isinstance(bundles, Mapping) else None
    if not isinstance(pca_record, Mapping):
        raise TCMPCA64ControlRunnerError("PCA64 bundle binding is absent")
    bundle_root = Path(str(pca_record.get("path", ""))).resolve(strict=True)
    manifest_path = _verify_artifact(
        pca_record.get("manifest"), label="PCA64 bundle manifest"
    )
    if manifest_path.parent != bundle_root or bundle_root != args.embedding_dir.resolve():
        raise TCMPCA64ControlRunnerError("CLI PCA64 bundle differs from frozen plan")

    _verify_artifact(
        plan.get("parent_adapter_selection"), label="parent adapter selection"
    )
    source_freeze = plan.get("source_freeze")
    if not isinstance(source_freeze, Mapping) or not source_freeze:
        raise TCMPCA64ControlRunnerError("source freeze is absent")
    verified_sources = {
        str(name): {
            **dict(record),
            "path": str(_verify_artifact(record, label=f"source {name}")),
        }
        for name, record in source_freeze.items()
        if isinstance(record, Mapping)
    }
    if set(verified_sources) != set(source_freeze):
        raise TCMPCA64ControlRunnerError("one or more source records are invalid")

    candidates = plan.get("candidates")
    matches = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("id") == args.candidate_id
    ] if isinstance(candidates, list) else []
    if len(matches) != 1 or matches[0].get("execution") != "validation_only_candidate":
        raise TCMPCA64ControlRunnerError("PCA64 executable candidate differs")

    folds = plan.get("fold_hyperparameters")
    fold = folds.get(str(args.fold)) if isinstance(folds, Mapping) else None
    if not isinstance(fold, Mapping):
        raise TCMPCA64ControlRunnerError(f"plan has no fold {args.fold}")
    try:
        epochs = int(fold["epochs"])
        learning_rate = float(fold["learning_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMPCA64ControlRunnerError("fold hyperparameters are invalid") from error
    adapter = str(fold.get("semantic_adapter", ""))
    if (
        epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or adapter not in {"identity", "residual_mlp"}
    ):
        raise TCMPCA64ControlRunnerError("fold adapter/epoch/lr contract is invalid")
    expected_fixed = {
        "batch_size": 2000,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": 1.0,
        "semantic_input_dim": 64,
        "semantic_mlp_hidden_dim": 32,
        "semantic_freeze_input": False,
    }
    if dict(plan.get("fixed", {})) != expected_fixed:
        raise TCMPCA64ControlRunnerError("fixed controls differ")
    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "candidate": matches[0],
        "fold_hyperparameters": dict(fold),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "adapter": adapter,
        "pca64_manifest": manifest_path,
        "verified_sources": verified_sources,
    }


@contextmanager
def _pca64_bundle_runtime(binding: Mapping[str, Any]) -> Iterator[None]:
    original = astra_runner._validate_tcm_feature_bundle
    manifest_path = Path(str(binding["pca64_manifest"])).resolve(strict=True)

    def validate_bundle(protocol, embedding_dir, feature_paths, *, smoke):
        if smoke:
            raise TCMPCA64ControlRunnerError("PCA64 control cannot run in smoke mode")
        if embedding_dir.resolve() != manifest_path.parent:
            raise TCMPCA64ControlRunnerError("runtime feature root differs from PCA64 plan")
        try:
            validated = validate_tcm_pca64_feature_bundle(
                manifest_path,
                protocol=protocol,
                expected_feature_paths=feature_paths,
                verify_transform=False,
            )
        except TCMPCA64BundleError as error:
            raise astra_runner.ASTRARunnerError(
                f"TCM-suite PCA64 feature bundle failed validation: {error}"
            ) from error
        return {
            "present": True,
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
            "schema": PCA64_BUNDLE_SCHEMA,
            "profile": validated["manifest"]["profile"],
            "formal_eligible": True,
            "recursive_validation": True,
            "pca64_transform_audited_at_plan_freeze": True,
            "entities": {
                entity: {
                    "matrix_sha256": details["matrix_sha256"],
                    "row_ids_sidecar_sha256": details["sidecar_sha256"],
                }
                for entity, details in validated["entities"].items()
            },
            "evidence_artifact_count": len(validated["evidence_paths"]),
        }

    astra_runner._validate_tcm_feature_bundle = validate_bundle
    try:
        yield
    finally:
        astra_runner._validate_tcm_feature_bundle = original


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    adapter_binding = {
        "epochs": binding["epochs"],
        "learning_rate": binding["learning_rate"],
        "adapter": binding["adapter"],
    }
    with _pca64_bundle_runtime(binding), _adapter_runtime(adapter_binding):
        metrics = astra_runner.run(args)

    output_root = args.output_root.resolve()
    metrics_path = output_root / "results" / "metrics.json"
    protocol_path = output_root / "protocol.json"
    audit_path = output_root / "tcm_semantic_adapter_audit.json"
    if sha256(Path(binding["plan_path"])) != binding["plan_sha256"]:
        raise TCMPCA64ControlRunnerError("PCA64 plan changed during the run")
    reject_outer_artifacts(output_root)
    stored_metrics = dict(_load_json(metrics_path))
    stored_protocol = dict(_load_json(protocol_path))
    audit = dict(_load_json(audit_path))
    _validate_adapter_audit(
        audit,
        adapter=str(binding["adapter"]),
        expected_steps=int(stored_metrics.get("Optimizer_Steps", -1)),
    )
    evaluation = stored_protocol.get("evaluation")
    split_rows = stored_protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
    ):
        raise TCMPCA64ControlRunnerError("central runner did not remain validation-only")

    runner_path = Path(__file__).resolve()
    provenance = {
        "schema": RUNNER_SCHEMA,
        "plan": {"path": binding["plan_path"], "sha256": binding["plan_sha256"]},
        "candidate": dict(binding["candidate"]),
        "fold_hyperparameters": dict(binding["fold_hyperparameters"]),
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        "pca64_bundle_manifest": {
            "path": str(binding["pca64_manifest"]),
            "sha256": sha256(Path(binding["pca64_manifest"])),
        },
        "adapter_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "source_freeze": dict(binding["verified_sources"]),
        "selection_metric": "Validation_AUPRC",
        "outer_test_materialised": False,
    }
    stored_metrics.update(
        {
            "Method": "ASTRA-TCM-PCA64-control-candidate",
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Semantic_Adapter": binding["adapter"],
            "Feature_Basis": "deterministic_shared_pca64_v1",
            "Tuning_Stage": "semantic-feature-basis-control",
            "Execution_Mode": "pca64-control-selection-candidate",
            "Evaluation_Mode": "validation-only-pca64-control",
        }
    )
    stored_protocol["tcm_pca64_control_candidate"] = provenance
    _write_json(metrics_path, stored_metrics)
    _write_json(protocol_path, stored_protocol)
    return stored_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = astra_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:
        print(f"TCM PCA64 candidate failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
