#!/usr/bin/env python3
"""Validation-only runner for the post-freeze semantic-adapter search.

The completed ASTRARefined experiment binds the hashes of its model and
runner.  This module deliberately leaves both frozen files untouched and
routes only the three new 64-dimensional adapter candidates through the same
data, split, feature-audit, checkpoint, and manifest machinery.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .astra_runner import (
    ASTRARunnerError,
    _assert_ethnobotany_feature_binding,
    _configure_runtime,
    _initialize_model_variables,
    _rebind_model_mappings,
    _runtime_device_diagnostics,
    compact_pairs_to_training_rows,
)
from .ethnobotany_refined_runner import (
    CHECKPOINT_EPOCHS,
    LEGACY_ROOT,
    PreparedFeatures,
    RefinedCandidateRunnerError,
    TrainingResult,
    TrainingValidationRows,
    _configure_model,
    _load_json,
    run_validation_candidate as _run_refined_validation_candidate,
    validate_candidate_config,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .backend_paths import activate_backend


EXPERIMENTAL_ADAPTERS = frozenset(
    {"residual_linear", "gated_residual", "residual_mlp"}
)
RUNNER_MODULE = "astra.benchmark.ethnobotany_semantic_adapter_runner"
FIXED_INPUT_DIMENSION = 64
FIXED_MLP_HIDDEN_DIMENSION = 32


def _parse_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RefinedCandidateRunnerError(f"{key} must be a boolean")


def validate_semantic_adapter_candidate(
    payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless exactly one of the predeclared S5 adapters is used."""

    candidate = validate_candidate_config(payload)
    options = candidate["model_options"]
    adapter = str(options.get("astra.semantic.adapter", "")).strip().lower()
    if adapter not in EXPERIMENTAL_ADAPTERS:
        raise RefinedCandidateRunnerError(
            "semantic-adapter runner requires one of "
            + ", ".join(sorted(EXPERIMENTAL_ADAPTERS))
        )
    try:
        input_dimension = int(options.get("astra.semantic.input_dim", 64))
        hidden_dimension = int(
            options.get(
                "astra.semantic.mlp_hidden_dim",
                FIXED_MLP_HIDDEN_DIMENSION,
            )
        )
    except (TypeError, ValueError) as error:
        raise RefinedCandidateRunnerError(
            "semantic adapter dimensions must be integers"
        ) from error
    if input_dimension != FIXED_INPUT_DIMENSION:
        raise RefinedCandidateRunnerError(
            "semantic-adapter search requires astra.semantic.input_dim=64"
        )
    if hidden_dimension != FIXED_MLP_HIDDEN_DIMENSION:
        raise RefinedCandidateRunnerError(
            "semantic-adapter search fixes astra.semantic.mlp_hidden_dim=32"
        )
    if _parse_bool(
        options.get("astra.semantic.freeze_input", False),
        key="astra.semantic.freeze_input",
    ):
        raise RefinedCandidateRunnerError(
            "semantic-adapter search requires astra.semantic.freeze_input=false"
        )
    return candidate


def _load_semantic_refined_components() -> tuple[type, type]:
    """Load the new model without allowing stale legacy modules to leak in."""

    activate_backend("ethnobotany")
    candidates = tuple(
        name
        for name in sys.modules
        if name
        in {
            "ASTRASemanticRefined",
            "ASTRARefined",
            "rating",
            "base",
            "util",
            "ethnobotany_feature_paths",
        }
        or name.startswith("base.")
        or name.startswith("util.")
    )
    for name in candidates:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is None or LEGACY_ROOT not in Path(origin).resolve().parents:
            sys.modules.pop(name, None)
    try:
        model_class = getattr(
            importlib.import_module("ASTRASemanticRefined"),
            "ASTRASemanticRefined",
        )
        config_class = getattr(importlib.import_module("util.config"), "ModelConf")
    except (ImportError, AttributeError) as error:
        raise RefinedCandidateRunnerError(
            f"cannot import ASTRASemanticRefined from {LEGACY_ROOT}: {error}"
        ) from error
    return model_class, config_class


def _configure_semantic_model(
    config_class: type,
    *,
    candidate: Mapping[str, Any],
    relation_paths: Mapping[str, Path],
    features: PreparedFeatures,
    embedding_dir: Path,
    output_root: Path,
) -> object:
    config = _configure_model(
        config_class,
        candidate=candidate,
        relation_paths=relation_paths,
        features=features,
        embedding_dir=embedding_dir,
        output_root=output_root,
    )
    config.config["model.name"] = "ASTRASemanticRefined"
    return config


def train_semantic_adapter_candidate(
    *,
    protocol: BenchmarkProtocol,
    rows: TrainingValidationRows,
    candidate: Mapping[str, Any],
    relation_paths: Mapping[str, Path],
    features: PreparedFeatures,
    embedding_dir: Path,
    output_root: Path,
    device: str,
    seed: int,
) -> TrainingResult:
    """Train one new adapter without ever constructing outer-fold rows."""

    _configure_runtime(seed, device, output_root / "runtime")
    model_class, config_class = _load_semantic_refined_components()
    runtime_device = _runtime_device_diagnostics(device)
    config = _configure_semantic_model(
        config_class,
        candidate=candidate,
        relation_paths=relation_paths,
        features=features,
        embedding_dir=embedding_dir,
        output_root=output_root / "runtime",
    )
    training_rows = compact_pairs_to_training_rows(
        protocol, rows.train_pairs, rows.train_labels
    )
    model = model_class(config, training_rows, [], f"[{rows.fold}]")
    _rebind_model_mappings(model, protocol)
    if getattr(model.data, "testData", None):
        raise RefinedCandidateRunnerError("evaluation rows were passed to the model")
    if len(model.data.trainingData) != len(training_rows):
        raise RefinedCandidateRunnerError(
            "model changed the supplied training-row count"
        )
    train_keys = {
        (int(left), int(right)) for left, right in np.asarray(rows.train_pairs)
    }
    validation_keys = {
        (int(left), int(right))
        for left, right in np.asarray(rows.validation_pairs)
    }
    if train_keys.intersection(validation_keys):
        raise RefinedCandidateRunnerError("validation pairs leaked into training")

    started = time.time()
    model.readConfiguration()
    model.initModel()
    _initialize_model_variables(model)
    if model.options.semantic_freeze_input:
        raise RefinedCandidateRunnerError(
            "semantic-adapter candidate unexpectedly froze its input tensors"
        )
    binding = _assert_ethnobotany_feature_binding(
        model,
        protocol,
        features.paths,
        features.mask_paths,
        clip_values=True,
    )
    model.trainModel()
    snapshots: dict[int, Mapping[str, Any]] = {}
    for epoch in CHECKPOINT_EPOCHS:
        try:
            snapshot = model.epoch_snapshots[epoch]
        except (AttributeError, KeyError) as error:
            raise RefinedCandidateRunnerError(
                f"ASTRASemanticRefined did not retain epoch {epoch}"
            ) from error
        snapshots[epoch] = {
            **dict(snapshot),
            "u": np.asarray(snapshot["u"]).copy(),
            "i": np.asarray(snapshot["i"]).copy(),
        }
    session = getattr(model, "sess", None)
    if session is not None:
        session.close()
    return TrainingResult(
        snapshots=snapshots,
        diagnostics={
            "runtime_seconds": time.time() - started,
            "device": runtime_device,
            "feature_binding": binding,
            "resolved_hyperparameters": dict(model.resolved_hyperparameters),
            "implementation": {
                "model_module": "ASTRASemanticRefined",
                "runner_module": RUNNER_MODULE,
                "frozen_parent_model_module": "ASTRARefined",
            },
        },
    )


def run_validation_candidate(
    *,
    protocol_manifest: Path,
    data_dir: Path,
    embedding_dir: Path,
    semantic_reduction_manifest: Path,
    config_json: Path,
    fold: int,
    output_root: Path,
    semantic_deep_audit_receipt: Path | None = None,
    feature_deep_audit_receipt: Path | None = None,
    device: str = "auto",
    protocol_loader: Callable[[Path], BenchmarkProtocol] = load_protocol_manifest,
) -> dict[str, Any]:
    """Validate the S5 contract, then reuse the frozen validation-only shell."""

    config_json = Path(config_json).resolve(strict=True)
    validate_semantic_adapter_candidate(_load_json(config_json))
    return _run_refined_validation_candidate(
        protocol_manifest=protocol_manifest,
        data_dir=data_dir,
        embedding_dir=embedding_dir,
        semantic_reduction_manifest=semantic_reduction_manifest,
        semantic_deep_audit_receipt=semantic_deep_audit_receipt,
        feature_deep_audit_receipt=feature_deep_audit_receipt,
        config_json=config_json,
        fold=fold,
        output_root=output_root,
        device=device,
        protocol_loader=protocol_loader,
        trainer=train_semantic_adapter_candidate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--semantic-reduction-manifest", type=Path, required=True)
    parser.add_argument("--semantic-deep-audit-receipt", type=Path)
    parser.add_argument("--feature-deep-audit-receipt", type=Path)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_validation_candidate(
            protocol_manifest=args.protocol_manifest,
            data_dir=args.data_dir,
            embedding_dir=args.embedding_dir,
            semantic_reduction_manifest=args.semantic_reduction_manifest,
            semantic_deep_audit_receipt=args.semantic_deep_audit_receipt,
            feature_deep_audit_receipt=args.feature_deep_audit_receipt,
            config_json=args.config_json,
            fold=args.fold,
            output_root=args.output_root,
            device=args.device,
        )
    except (RefinedCandidateRunnerError, ASTRARunnerError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
