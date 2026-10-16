#!/usr/bin/env python3
"""Freeze the deterministic v3 rerun of the TCM semantic-adapter stage.

The upstream epoch-boundary decision is reused without alteration.  A failed
v2 adapter plan is bound as provenance, while every executable source used by
the deterministic rerun is hashed into a new, immutable validation-only plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import tcm_astra_semantic_adapter_plan as base_plan
from .backend_paths import TCM_BACKEND_ROOT


PLAN_SCHEMA = "astra-tcm-semantic-adapter-deterministic-plan/v3"
DETERMINISM_SCHEMA = "astra-tcm-semantic-adapter-determinism/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"


class TCMDeterministicAdapterPlanError(RuntimeError):
    """Raised when the deterministic rerun plan cannot be frozen."""


def sha256(path: Path) -> str:
    return base_plan.sha256(path)


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMDeterministicAdapterPlanError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TCMDeterministicAdapterPlanError(
            f"JSON root must be an object: {path}"
        )
    return payload


def _source_freeze() -> dict[str, dict[str, object]]:
    paths = {
        "model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapterDeterministic.py",
        "base_model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapter.py",
        "runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_deterministic_runner.py",
        "base_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_runner.py",
        "selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_deterministic_selection.py",
        "base_selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_selection.py",
        "pair_guard": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_pair_guard.py",
        "plan_generator": Path(__file__).resolve(),
        "base_plan_generator": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_plan.py",
        "tensorflow_graph_seed": SOURCE_ROOT / "astra/tf_graph_seed.py",
        "launcher": SOURCE_ROOT / "herbhg/cli.py",
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
        raise TCMDeterministicAdapterPlanError(
            f"deterministic stage implementation is incomplete: {error}"
        ) from error


def _validate_failed_plan(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = _load_json(resolved)
    if (
        payload.get("schema") != base_plan.PLAN_SCHEMA
        or payload.get("dataset") != "tcm-suite"
        or int(payload.get("seed", -1)) != 42
        or payload.get("status") != "frozen-before-stage2-candidate-testing"
    ):
        raise TCMDeterministicAdapterPlanError(
            "failed v2 plan is not the expected frozen TCM adapter plan"
        )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMDeterministicAdapterPlanError("failed v2 plan sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    digest = sha256(resolved)
    if len(fields) != 2 or fields != [digest, resolved.name]:
        raise TCMDeterministicAdapterPlanError(
            "failed v2 plan sidecar does not match"
        )
    return _artifact(resolved)


def build_plan(
    epoch_boundary_decision: Path, failed_v2_plan: Path
) -> dict[str, Any]:
    try:
        payload = dict(base_plan.build_plan(epoch_boundary_decision))
    except Exception as error:
        raise TCMDeterministicAdapterPlanError(str(error)) from error
    previous = _validate_failed_plan(failed_v2_plan)
    provenance = {
        name: dict(record)
        for name, record in dict(payload["source_freeze"]).items()
        if str(name).startswith("provenance_")
    }
    source_freeze = _source_freeze()
    source_freeze.update(provenance)
    seed = int(payload["seed"])
    payload.update(
        {
            "schema": PLAN_SCHEMA,
            "interpretation": (
                "Seed-42 same-split post-baseline development diagnostic; "
                "validation-only deterministic adapter rerun"
            ),
            "failed_v2_plan": previous,
            "rerun_reason": (
                "v2 adapter arms did not share TensorFlow-random initial weights"
            ),
            "deterministic_initialization": {
                "schema": DETERMINISM_SCHEMA,
                "tensorflow_graph_seed": seed,
                "graph_must_be_empty_before_seed": True,
                "tensorflow_v1_behavior_enabled_before_seed": True,
                "seed_set_after_runtime_environment": True,
                "seed_set_before_model_import": True,
                "same_fold_shared_initialization_required": True,
            },
            "source_freeze": source_freeze,
            "selection_policy": (
                "Select independently within each outer fold using only frozen "
                "inner-validation predictions. Before launcher completion, each "
                "fold must have a signed pair receipt proving identical complete "
                "initial.shared_variable_sha256 mappings across both arms. No "
                "candidate may create an outer-test prediction."
            ),
        }
    )
    return payload


def freeze_plan(
    *, epoch_boundary_decision: Path, failed_v2_plan: Path, output: Path
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMDeterministicAdapterPlanError(
            f"refusing to overwrite plan or sidecar: {output}, {sidecar}"
        )
    payload = build_plan(epoch_boundary_decision, failed_v2_plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {**payload, "plan_path": str(output), "plan_sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-boundary-decision", type=Path, required=True)
    parser.add_argument("--failed-v2-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_plan(
            epoch_boundary_decision=args.epoch_boundary_decision,
            failed_v2_plan=args.failed_v2_plan,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM deterministic adapter plan freeze failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
