#!/usr/bin/env python3
"""Validation-only deterministic rerun of the TCM semantic-adapter stage.

This version preserves the frozen adapter architecture and central strict
runner.  Its only training-semantic change is to seed a fresh TensorFlow-v1
default graph after the runtime environment is configured and before any
model module is imported or graph operation is created.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

from . import astra_runner
from . import tcm_astra_semantic_adapter_runner as base_runner
from .tcm_astra_semantic_adapter_deterministic_plan import PLAN_SCHEMA
from astra.tf_graph_seed import GRAPH_SEED_SCHEMA, seed_fresh_default_graph
from .backend_paths import TCM_BACKEND_ROOT


RUNNER_SCHEMA = "astra-tcm-semantic-adapter-deterministic-candidate/v1"
AUDIT_SCHEMA = base_runner.AUDIT_SCHEMA
DETERMINISM_SCHEMA = "astra-tcm-semantic-adapter-determinism/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"


class TCMDeterministicAdapterRunnerError(RuntimeError):
    """Raised when the deterministic stage-2 contract is violated."""


def _expected_determinism(seed: int) -> dict[str, object]:
    return {
        "schema": DETERMINISM_SCHEMA,
        "tensorflow_graph_seed": seed,
        "graph_must_be_empty_before_seed": True,
        "tensorflow_v1_behavior_enabled_before_seed": True,
        "seed_set_after_runtime_environment": True,
        "seed_set_before_model_import": True,
        "same_fold_shared_initialization_required": True,
    }


@contextmanager
def _base_plan_schema() -> Iterator[None]:
    original = base_runner.PLAN_SCHEMA
    base_runner.PLAN_SCHEMA = PLAN_SCHEMA
    try:
        yield
    finally:
        base_runner.PLAN_SCHEMA = original


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the v3 plan while reusing the frozen v2 field contract."""

    with _base_plan_schema():
        try:
            binding = base_runner.validate_candidate(args)
        except Exception as error:
            raise TCMDeterministicAdapterRunnerError(str(error)) from error
    plan = base_runner._load_json(Path(str(binding["plan_path"])))
    expected = _expected_determinism(int(args.seed))
    if plan.get("deterministic_initialization") != expected:
        raise TCMDeterministicAdapterRunnerError(
            "deterministic initialization contract differs from the v3 runner"
        )
    binding["deterministic_initialization"] = expected
    return binding


@contextmanager
def _candidate_runtime(
    binding: Mapping[str, Any],
) -> Iterator[dict[str, object]]:
    """Scope all overrides and seed TensorFlow before model import/build."""

    original_hyperparameters = astra_runner._HYPERPARAMETERS["tcm-suite"]
    original_loader = astra_runner._load_legacy_components
    original_configure = astra_runner._configure_model
    original_runtime = astra_runner._configure_runtime
    replacement = dict(original_hyperparameters)
    replacement["epochs"] = int(binding["epochs"])
    replacement["learning_rate"] = float(binding["learning_rate"])
    expected_seed = int(
        binding["deterministic_initialization"]["tensorflow_graph_seed"]
    )
    seed_receipt: dict[str, object] = {}

    def configure_runtime(seed: int, device: str, output_root: Path) -> None:
        original_runtime(seed, device, output_root)
        if int(seed) != expected_seed:
            raise TCMDeterministicAdapterRunnerError(
                "runtime seed differs from the frozen TensorFlow graph seed"
            )
        observed = seed_fresh_default_graph(int(seed))
        os.environ["ASTRA_TF_GRAPH_SEED"] = str(seed)
        seed_receipt.clear()
        seed_receipt.update(observed)

    def load_components(dataset: str):
        if seed_receipt.get("seed") != expected_seed:
            raise TCMDeterministicAdapterRunnerError(
                "model import was attempted before TensorFlow graph seeding"
            )
        _, config_class = original_loader(dataset)
        if str(TCM_BACKEND_ROOT) not in sys.path:
            sys.path.insert(0, str(TCM_BACKEND_ROOT))
        try:
            module = importlib.import_module(
                "ASTRATCMSemanticAdapterDeterministic"
            )
            model_class = getattr(
                module, "ASTRATCMSemanticAdapterDeterministic"
            )
        except (ImportError, AttributeError) as error:
            raise TCMDeterministicAdapterRunnerError(
                f"cannot import deterministic TCM adapter model: {error}"
            ) from error
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        expected = TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapterDeterministic.py"
        if origin != expected.resolve():
            raise TCMDeterministicAdapterRunnerError(
                f"deterministic adapter model resolved outside source root: {origin}"
            )
        return model_class, config_class

    def configure_model(*args, **kwargs):
        config, recorded = original_configure(*args, **kwargs)
        config.config.update(
            {
                "model.name": "ASTRATCMSemanticAdapterDeterministic",
                "astra.semantic.adapter": str(binding["adapter"]),
                "astra.semantic.input_dim": "64",
                "astra.semantic.mlp_hidden_dim": "32",
                "astra.semantic.freeze_input": "false",
            }
        )
        recorded.update(
            {
                "model_class": "ASTRATCMSemanticAdapterDeterministic",
                "semantic_adapter": str(binding["adapter"]),
                "semantic_input_dim": 64,
                "semantic_mlp_hidden_dim": 32,
                "semantic_freeze_input": False,
                "tensorflow_graph_seed": expected_seed,
            }
        )
        return config, recorded

    astra_runner._HYPERPARAMETERS["tcm-suite"] = replacement
    astra_runner._load_legacy_components = load_components
    astra_runner._configure_model = configure_model
    astra_runner._configure_runtime = configure_runtime
    try:
        yield seed_receipt
    finally:
        astra_runner._configure_runtime = original_runtime
        astra_runner._configure_model = original_configure
        astra_runner._load_legacy_components = original_loader
        astra_runner._HYPERPARAMETERS["tcm-suite"] = original_hyperparameters
        os.environ.pop("ASTRA_TF_GRAPH_SEED", None)


def _validate_seed_evidence(
    *,
    audit: Mapping[str, Any],
    seed_receipt: Mapping[str, Any],
    expected_seed: int,
) -> None:
    expected_receipt = {
        "schema": GRAPH_SEED_SCHEMA,
        "seed": expected_seed,
        "graph_was_empty_before_seed": True,
        "tensorflow_v1_behavior_enabled_before_seed": True,
        "seed_set_before_model_import": True,
    }
    if dict(seed_receipt) != expected_receipt:
        raise TCMDeterministicAdapterRunnerError(
            "TensorFlow graph seed receipt is absent or inconsistent"
        )
    model_receipt = audit.get("deterministic_initialization")
    if (
        not isinstance(model_receipt, Mapping)
        or model_receipt.get("tensorflow_graph_seed") != expected_seed
        or model_receipt.get("graph_seed_schema") != GRAPH_SEED_SCHEMA
        or model_receipt.get("seed_observed_before_model_construction") is not True
    ):
        raise TCMDeterministicAdapterRunnerError(
            "model did not observe the frozen TensorFlow graph seed"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    seed_receipt: dict[str, object]
    with _candidate_runtime(binding) as seed_receipt:
        metrics = astra_runner.run(args)

    output_root = args.output_root.resolve()
    result_dir = output_root / "results"
    metrics_path = result_dir / "metrics.json"
    protocol_path = output_root / "protocol.json"
    audit_path = output_root / "tcm_semantic_adapter_audit.json"
    if (
        base_runner.sha256(Path(str(binding["plan_path"])))
        != binding["plan_sha256"]
    ):
        raise TCMDeterministicAdapterRunnerError(
            "deterministic stage-2 plan changed while the candidate ran"
        )
    base_runner._reject_outer_artifacts(output_root)
    stored_metrics = dict(base_runner._load_json(metrics_path))
    stored_protocol = dict(base_runner._load_json(protocol_path))
    audit = dict(base_runner._load_json(audit_path))
    expected_seed = int(
        binding["deterministic_initialization"]["tensorflow_graph_seed"]
    )
    _validate_seed_evidence(
        audit=audit,
        seed_receipt=seed_receipt,
        expected_seed=expected_seed,
    )
    expected_steps = int(stored_metrics.get("Optimizer_Steps", -1))
    try:
        base_runner._validate_audit(
            audit,
            adapter=str(binding["adapter"]),
            expected_steps=expected_steps,
        )
    except Exception as error:
        raise TCMDeterministicAdapterRunnerError(str(error)) from error
    evaluation = stored_protocol.get("evaluation")
    split_rows = stored_protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("test_predictions") is not None
        or evaluation.get("validation_only") is not True
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
    ):
        raise TCMDeterministicAdapterRunnerError(
            "central runner did not preserve validation-only mode"
        )

    source_path = Path(__file__).resolve()
    provenance = {
        "schema": RUNNER_SCHEMA,
        "plan": {
            "path": binding["plan_path"],
            "sha256": binding["plan_sha256"],
        },
        "candidate": dict(binding["candidate"]),
        "fold_hyperparameters": dict(binding["fold_hyperparameters"]),
        "runner": {
            "path": str(source_path),
            "sha256": base_runner.sha256(source_path),
        },
        "model": dict(audit["model_source"]),
        "base_model": dict(audit["base_model_source"]),
        "adapter_audit": {
            "path": str(audit_path),
            "sha256": base_runner.sha256(audit_path),
        },
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
            "Method": "ASTRA-TCM-semantic-adapter-deterministic-candidate",
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Semantic_Adapter": binding["adapter"],
            "Semantic_Adapter_Parameters": audit["adapter_parameter_count"],
            "TensorFlow_Graph_Seed": expected_seed,
            "Tuning_Stage": "semantic-adapter",
            "Execution_Mode": "semantic-adapter-deterministic-selection-candidate",
            "Evaluation_Mode": (
                "validation-only-semantic-adapter-deterministic-selection"
            ),
        }
    )
    stored_protocol["tcm_semantic_adapter_candidate"] = provenance
    model_sources = dict(stored_protocol.get("model_sources", {}))
    model_sources["tcm_semantic_adapter"] = dict(audit["base_model_source"])
    model_sources["tcm_semantic_adapter_deterministic"] = dict(
        audit["model_source"]
    )
    stored_protocol["model_sources"] = model_sources
    base_runner._write_json(metrics_path, stored_metrics)
    base_runner._write_json(protocol_path, stored_protocol)
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
        print(
            f"TCM deterministic semantic-adapter candidate failed: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
