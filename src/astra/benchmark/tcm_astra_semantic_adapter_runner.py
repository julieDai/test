#!/usr/bin/env python3
"""Validation-only runner for the second-stage TCM semantic adapter.

The runner is deliberately a narrow wrapper around the central strict ASTRA
runner.  It replaces only the model entry point and injects the adapter fields
while inheriting the epoch and learning rate selected for the requested fold
by the frozen stage-1 receipt.  It refuses smoke, diagnostic, or test-scoring
runs and verifies every frozen source hash before constructing a model.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

from . import astra_runner
from .tcm_astra_semantic_adapter_plan import PLAN_SCHEMA
from .backend_paths import TCM_BACKEND_ROOT


RUNNER_SCHEMA = "astra-tcm-semantic-adapter-candidate/v1"
AUDIT_SCHEMA = "astra-tcm-semantic-adapter-audit/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"


class TCMAdapterRunnerError(RuntimeError):
    """Raised when a stage-2 candidate violates its frozen contract."""


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
        raise TCMAdapterRunnerError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMAdapterRunnerError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_sidecar(plan_path: Path) -> str:
    digest = sha256(plan_path)
    sidecar = plan_path.with_suffix(plan_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMAdapterRunnerError(f"plan SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != plan_path.name:
        raise TCMAdapterRunnerError("plan SHA-256 sidecar does not match the plan")
    return digest


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMAdapterRunnerError(f"{label} artifact record is invalid")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    expected = str(record.get("sha256", ""))
    if not expected or sha256(path) != expected:
        raise TCMAdapterRunnerError(f"{label} changed after plan freeze")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise TCMAdapterRunnerError(f"{label} byte count changed after plan freeze")
    return path


def _float_equal(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


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


def _reject_outer_artifacts(root: Path) -> None:
    if not root.exists():
        return
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in _FORBIDDEN_OUTER_FILENAMES
    ]
    if forbidden:
        raise TCMAdapterRunnerError(
            "stage-2 candidate materialised an outer-test artifact: "
            + ", ".join(str(path) for path in forbidden)
        )


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = astra_runner._normalise_dataset(args.dataset)
    if dataset != "tcm-suite":
        raise TCMAdapterRunnerError("semantic-adapter runner is TCM-suite only")
    if not bool(args.validation_only):
        raise TCMAdapterRunnerError("stage-2 candidates require --validation-only")
    if bool(args.smoke) or bool(args.diagnostic):
        raise TCMAdapterRunnerError("stage-2 candidates reject smoke/diagnostic mode")
    if not _float_equal(args.text_fusion_alpha, 1.0):
        raise TCMAdapterRunnerError("stage 2 fixes text_fusion_alpha=1")

    plan_path = args.adapter_plan.resolve(strict=True)
    plan_sha = _validate_sidecar(plan_path)
    plan = _load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise TCMAdapterRunnerError("unsupported semantic-adapter plan schema")
    if plan.get("status") != "frozen-before-stage2-candidate-testing":
        raise TCMAdapterRunnerError("semantic-adapter plan is not frozen")
    if plan.get("dataset") != "tcm-suite" or int(plan.get("seed", -1)) != args.seed:
        raise TCMAdapterRunnerError("plan dataset/seed differs from the CLI")

    protocol_path = _verify_artifact(
        plan.get("protocol_manifest"), label="protocol manifest"
    )
    feature_bundle = plan.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping):
        raise TCMAdapterRunnerError("feature-bundle binding is absent")
    feature_path = Path(str(feature_bundle.get("path", ""))).resolve(strict=True)
    _verify_artifact(feature_bundle.get("manifest"), label="feature bundle manifest")
    if protocol_path != args.protocol_manifest.resolve():
        raise TCMAdapterRunnerError("CLI protocol differs from the frozen plan")
    if feature_path != args.embedding_dir.resolve():
        raise TCMAdapterRunnerError("CLI feature bundle differs from the frozen plan")

    _verify_artifact(plan.get("stage1_selection"), label="stage-1 selection")
    _verify_artifact(plan.get("stage1_plan"), label="stage-1 plan")
    source_freeze = plan.get("source_freeze")
    if not isinstance(source_freeze, Mapping) or not source_freeze:
        raise TCMAdapterRunnerError("source freeze is absent")
    verified_sources = {
        str(name): {
            **dict(record),
            "path": str(_verify_artifact(record, label=f"source {name}")),
        }
        for name, record in source_freeze.items()
        if isinstance(record, Mapping)
    }
    if set(verified_sources) != set(source_freeze):
        raise TCMAdapterRunnerError("one or more source-freeze records are invalid")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise TCMAdapterRunnerError("plan candidate list is absent")
    matches = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("id") == args.candidate_id
    ]
    if len(matches) != 1:
        raise TCMAdapterRunnerError(
            f"candidate {args.candidate_id!r} is not unique in the plan"
        )
    candidate = matches[0]
    adapter = str(candidate.get("semantic_adapter", ""))
    if adapter not in {"identity", "residual_mlp"}:
        raise TCMAdapterRunnerError("plan contains an unsupported adapter")

    folds = plan.get("fold_hyperparameters")
    fold_record = folds.get(str(args.fold)) if isinstance(folds, Mapping) else None
    if not isinstance(fold_record, Mapping):
        raise TCMAdapterRunnerError(f"plan has no hyperparameters for fold {args.fold}")
    epochs = int(fold_record.get("epochs", 0))
    learning_rate = float(fold_record.get("learning_rate", 0.0))
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise TCMAdapterRunnerError("fold hyperparameters are invalid")

    fixed = plan.get("fixed")
    expected_fixed = {
        "batch_size": 2000,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": 1.0,
        "semantic_input_dim": 64,
        "semantic_mlp_hidden_dim": 32,
        "semantic_freeze_input": False,
    }
    if not isinstance(fixed, Mapping) or dict(fixed) != expected_fixed:
        raise TCMAdapterRunnerError("stage-2 fixed controls differ from the contract")
    return {
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "candidate": candidate,
        "fold_hyperparameters": dict(fold_record),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "adapter": adapter,
        "verified_sources": verified_sources,
    }


@contextmanager
def _candidate_runtime(binding: Mapping[str, Any]) -> Iterator[None]:
    original_hyperparameters = astra_runner._HYPERPARAMETERS["tcm-suite"]
    original_loader = astra_runner._load_legacy_components
    original_configure = astra_runner._configure_model
    replacement = dict(original_hyperparameters)
    replacement["epochs"] = int(binding["epochs"])
    replacement["learning_rate"] = float(binding["learning_rate"])

    def load_components(dataset: str):
        _, config_class = original_loader(dataset)
        if str(TCM_BACKEND_ROOT) not in sys.path:
            sys.path.insert(0, str(TCM_BACKEND_ROOT))
        try:
            module = importlib.import_module("ASTRATCMSemanticAdapter")
            model_class = getattr(module, "ASTRATCMSemanticAdapter")
        except (ImportError, AttributeError) as error:
            raise TCMAdapterRunnerError(
                f"cannot import TCM semantic-adapter model: {error}"
            ) from error
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        expected = TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapter.py"
        if origin != expected.resolve():
            raise TCMAdapterRunnerError(
                f"semantic-adapter model resolved outside source root: {origin}"
            )
        return model_class, config_class

    def configure_model(*args, **kwargs):
        config, recorded = original_configure(*args, **kwargs)
        config.config.update(
            {
                "model.name": "ASTRATCMSemanticAdapter",
                "astra.semantic.adapter": str(binding["adapter"]),
                "astra.semantic.input_dim": "64",
                "astra.semantic.mlp_hidden_dim": "32",
                "astra.semantic.freeze_input": "false",
            }
        )
        recorded.update(
            {
                "model_class": "ASTRATCMSemanticAdapter",
                "semantic_adapter": str(binding["adapter"]),
                "semantic_input_dim": 64,
                "semantic_mlp_hidden_dim": 32,
                "semantic_freeze_input": False,
            }
        )
        return config, recorded

    astra_runner._HYPERPARAMETERS["tcm-suite"] = replacement
    astra_runner._load_legacy_components = load_components
    astra_runner._configure_model = configure_model
    try:
        yield
    finally:
        astra_runner._configure_model = original_configure
        astra_runner._load_legacy_components = original_loader
        astra_runner._HYPERPARAMETERS["tcm-suite"] = original_hyperparameters


def _validate_audit(
    audit: Mapping[str, Any], *, adapter: str, expected_steps: int
) -> None:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("adapter") != adapter:
        raise TCMAdapterRunnerError("model emitted an incompatible adapter audit")
    expected_parameters = 0 if adapter == "identity" else 8384
    if int(audit.get("adapter_parameter_count", -1)) != expected_parameters:
        raise TCMAdapterRunnerError("adapter parameter count is incorrect")
    raw_trainable = audit.get("raw_semantic_trainable")
    if not isinstance(raw_trainable, Mapping) or not all(
        raw_trainable.get(entity) is True for entity in ("herb", "disease")
    ):
        raise TCMAdapterRunnerError("raw semantic variables are not trainable")
    dependency = audit.get("forward_dependency")
    if not isinstance(dependency, Mapping) or dependency.get(
        "probe_excludes_regularization"
    ) is not True:
        raise TCMAdapterRunnerError("adapter forward-only dependency proof is absent")
    raw_dependency = dependency.get("raw_semantics")
    if not isinstance(raw_dependency, Mapping) or not all(raw_dependency.values()):
        raise TCMAdapterRunnerError("raw semantics are disconnected from prediction")
    adapter_dependency = dependency.get("adapter")
    if not isinstance(adapter_dependency, Mapping) or not all(
        adapter_dependency.values()
    ):
        raise TCMAdapterRunnerError("an adapter variable is disconnected from prediction")
    initial = audit.get("initial")
    final = audit.get("final")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise TCMAdapterRunnerError("adapter pre/post snapshots are absent")
    initial_delta = initial.get("semantic_delta")
    if not isinstance(initial_delta, Mapping) or any(
        float(value) != 0.0 for value in initial_delta.values()
    ):
        raise TCMAdapterRunnerError("adapter is not identity-initialised")
    if int(audit.get("optimizer_steps", -1)) != int(expected_steps):
        raise TCMAdapterRunnerError("adapter optimiser-step count is inconsistent")
    if adapter == "residual_mlp":
        if audit.get("adapter_parameters_changed") is not True:
            raise TCMAdapterRunnerError("residual adapter parameters did not update")
        final_delta = final.get("semantic_delta")
        if not isinstance(final_delta, Mapping) or not any(
            float(value) > 0.0 for value in final_delta.values()
        ):
            raise TCMAdapterRunnerError("residual adapter never changed its semantics")


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    with _candidate_runtime(binding):
        metrics = astra_runner.run(args)

    output_root = args.output_root.resolve()
    result_dir = output_root / "results"
    metrics_path = result_dir / "metrics.json"
    protocol_path = output_root / "protocol.json"
    audit_path = output_root / "tcm_semantic_adapter_audit.json"
    if sha256(Path(str(binding["plan_path"]))) != binding["plan_sha256"]:
        raise TCMAdapterRunnerError("stage-2 plan changed while the candidate ran")
    _reject_outer_artifacts(output_root)
    stored_metrics = dict(_load_json(metrics_path))
    stored_protocol = dict(_load_json(protocol_path))
    audit = dict(_load_json(audit_path))
    expected_steps = int(stored_metrics.get("Optimizer_Steps", -1))
    _validate_audit(
        audit,
        adapter=str(binding["adapter"]),
        expected_steps=expected_steps,
    )
    evaluation = stored_protocol.get("evaluation")
    split_rows = stored_protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("test_predictions") is not None
        or evaluation.get("validation_only") is not True
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
    ):
        raise TCMAdapterRunnerError("central runner did not preserve validation-only mode")

    source_path = Path(__file__).resolve()
    provenance = {
        "schema": RUNNER_SCHEMA,
        "plan": {
            "path": binding["plan_path"],
            "sha256": binding["plan_sha256"],
        },
        "candidate": dict(binding["candidate"]),
        "fold_hyperparameters": dict(binding["fold_hyperparameters"]),
        "runner": {"path": str(source_path), "sha256": sha256(source_path)},
        "model": dict(audit["model_source"]),
        "adapter_audit": {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
        },
        "source_freeze": dict(binding["verified_sources"]),
        "selection_metric": "Validation_AUPRC",
        "outer_test_materialised": False,
    }
    stored_metrics.update(
        {
            "Method": "ASTRA-TCM-semantic-adapter-candidate",
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Semantic_Adapter": binding["adapter"],
            "Semantic_Adapter_Parameters": audit["adapter_parameter_count"],
            "Tuning_Stage": "semantic-adapter",
            "Execution_Mode": "semantic-adapter-selection-candidate",
            "Evaluation_Mode": "validation-only-semantic-adapter-selection",
        }
    )
    stored_protocol["tcm_semantic_adapter_candidate"] = provenance
    model_sources = dict(stored_protocol.get("model_sources", {}))
    model_sources["tcm_semantic_adapter"] = dict(audit["model_source"])
    stored_protocol["model_sources"] = model_sources
    _write_json(metrics_path, stored_metrics)
    _write_json(protocol_path, stored_protocol)
    return stored_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = astra_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--adapter-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:
        print(f"TCM semantic-adapter candidate failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
