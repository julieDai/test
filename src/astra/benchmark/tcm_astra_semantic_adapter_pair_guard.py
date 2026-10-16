#!/usr/bin/env python3
"""Issue a per-fold receipt only when deterministic adapter arms truly match.

The guard is validation-only.  It rejects any outer-test artifact and compares
the complete initial shared-variable hash mapping from both adapter audits.
The launcher requires all five receipts before it can report completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import tcm_astra_semantic_adapter_runner as base_runner
from .tcm_astra_semantic_adapter_deterministic_plan import PLAN_SCHEMA
from .tcm_astra_semantic_adapter_deterministic_runner import (
    AUDIT_SCHEMA,
    RUNNER_SCHEMA,
)


PAIR_RECEIPT_SCHEMA = "astra-tcm-semantic-adapter-shared-init-pair/v1"
CANDIDATES = {
    "identity_control": "identity",
    "residual_mlp_h32": "residual_mlp",
}


class TCMAdapterPairGuardError(RuntimeError):
    """Raised when a same-fold adapter pair is not comparable."""


def sha256(path: Path) -> str:
    return base_runner.sha256(path)


def _object_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMAdapterPairGuardError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMAdapterPairGuardError(f"JSON root must be an object: {path}")
    return payload


def _plan_and_digest(path: Path) -> tuple[Mapping[str, Any], str]:
    resolved = path.resolve(strict=True)
    digest = sha256(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMAdapterPairGuardError("deterministic plan sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, resolved.name]:
        raise TCMAdapterPairGuardError("deterministic plan sidecar differs")
    plan = _load_json(resolved)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen-before-stage2-candidate-testing"
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != 42
    ):
        raise TCMAdapterPairGuardError("deterministic plan contract differs")
    guard_record = plan.get("source_freeze", {}).get("pair_guard")
    source = Path(__file__).resolve()
    if (
        not isinstance(guard_record, Mapping)
        or Path(str(guard_record.get("path", ""))).resolve() != source
        or guard_record.get("sha256") != sha256(source)
    ):
        raise TCMAdapterPairGuardError("pair guard source differs from frozen plan")
    return plan, digest


def _validated_shared_hashes(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
    candidate_root: Path,
    candidate_id: str,
    adapter: str,
    fold: int,
) -> dict[str, Any]:
    root = (candidate_root / candidate_id / f"fold_{fold}").resolve()
    completion = root / ".astra_semantic_adapter_complete"
    protocol_path = root / "protocol.json"
    metrics_path = root / "results" / "metrics.json"
    audit_path = root / "tcm_semantic_adapter_audit.json"
    for path in (completion, protocol_path, metrics_path, audit_path):
        if not path.is_file():
            raise TCMAdapterPairGuardError(f"candidate artifact is absent: {path}")
    try:
        base_runner._reject_outer_artifacts(root)
    except Exception as error:
        raise TCMAdapterPairGuardError(str(error)) from error

    protocol = _load_json(protocol_path)
    metrics = _load_json(metrics_path)
    audit = _load_json(audit_path)
    binding = protocol.get("tcm_semantic_adapter_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != RUNNER_SCHEMA:
        raise TCMAdapterPairGuardError("deterministic candidate binding is absent")
    if (
        protocol.get("fold") != fold
        or metrics.get("Fold") != fold
        or metrics.get("Candidate_ID") != candidate_id
        or metrics.get("Semantic_Adapter") != adapter
        or metrics.get("TensorFlow_Graph_Seed") != 42
    ):
        raise TCMAdapterPairGuardError("candidate fold/adapter/seed identity differs")
    bound_plan = binding.get("plan")
    if (
        not isinstance(bound_plan, Mapping)
        or Path(str(bound_plan.get("path", ""))).resolve() != plan_path.resolve()
        or bound_plan.get("sha256") != plan_sha256
        or metrics.get("Candidate_Plan_SHA256") != plan_sha256
    ):
        raise TCMAdapterPairGuardError("candidate is bound to another plan")
    if binding.get("candidate") != {
        "id": candidate_id,
        "semantic_adapter": adapter,
    }:
        raise TCMAdapterPairGuardError("candidate binding differs")
    if (
        binding.get("deterministic_initialization")
        != plan.get("deterministic_initialization")
        or binding.get("outer_test_materialised") is not False
    ):
        raise TCMAdapterPairGuardError("candidate determinism/outer binding differs")
    seed_receipt = binding.get("tensorflow_graph_seed_receipt")
    if (
        not isinstance(seed_receipt, Mapping)
        or seed_receipt.get("seed") != 42
        or seed_receipt.get("graph_was_empty_before_seed") is not True
        or seed_receipt.get("tensorflow_v1_behavior_enabled_before_seed")
        is not True
        or seed_receipt.get("seed_set_before_model_import") is not True
    ):
        raise TCMAdapterPairGuardError("candidate graph-seed receipt differs")
    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or not isinstance(split_rows, Mapping)
        or int(split_rows.get("test_used", -1)) != 0
        or metrics.get("Validation_Only") is not True
        or int(metrics.get("Test_Models_Evaluated", -1)) != 0
    ):
        raise TCMAdapterPairGuardError("candidate is not validation-only")
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("adapter") != adapter:
        raise TCMAdapterPairGuardError("candidate adapter audit differs")
    audit_seed = audit.get("deterministic_initialization")
    if (
        not isinstance(audit_seed, Mapping)
        or audit_seed.get("tensorflow_graph_seed") != 42
        or audit_seed.get("seed_observed_before_model_construction") is not True
    ):
        raise TCMAdapterPairGuardError("adapter audit lacks graph-seed evidence")
    initial = audit.get("initial")
    shared = (
        initial.get("shared_variable_sha256")
        if isinstance(initial, Mapping)
        else None
    )
    if not isinstance(shared, Mapping) or not shared:
        raise TCMAdapterPairGuardError("shared initialization hashes are absent")
    hashes = {str(key): str(value) for key, value in shared.items()}
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes.values()
    ):
        raise TCMAdapterPairGuardError("shared initialization hash is invalid")
    return {
        "candidate_id": candidate_id,
        "semantic_adapter": adapter,
        "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256(protocol_path),
        },
        "shared_initial_variable_sha256": hashes,
        "shared_initial_mapping_sha256": _object_sha256(hashes),
    }


def require_identical_shared_initialization(
    records: Sequence[Mapping[str, Any]], *, fold: int
) -> dict[str, str]:
    """Return the common mapping or fail if any key or value differs."""

    if len(records) != 2:
        raise TCMAdapterPairGuardError(
            f"fold {fold} must contain exactly two adapter arms"
        )
    mappings = [record.get("shared_initial_variable_sha256") for record in records]
    if not all(isinstance(value, Mapping) and value for value in mappings):
        raise TCMAdapterPairGuardError(
            f"fold {fold} shared initialization mapping is absent"
        )
    serialized = {
        json.dumps(value, sort_keys=True) for value in mappings
    }
    if len(serialized) != 1:
        raise TCMAdapterPairGuardError(
            f"fold {fold} adapter arms have different shared initialization"
        )
    return {str(key): str(value) for key, value in mappings[0].items()}


def build_pair_receipt(
    *, plan_path: Path, candidate_root: Path, fold: int
) -> dict[str, Any]:
    if fold not in range(1, 6):
        raise TCMAdapterPairGuardError("fold must be in 1..5")
    plan, plan_sha = _plan_and_digest(plan_path)
    candidate_root = candidate_root.resolve()
    records = [
        _validated_shared_hashes(
            plan=plan,
            plan_path=plan_path,
            plan_sha256=plan_sha,
            candidate_root=candidate_root,
            candidate_id=candidate_id,
            adapter=adapter,
            fold=fold,
        )
        for candidate_id, adapter in CANDIDATES.items()
    ]
    shared = require_identical_shared_initialization(records, fold=fold)
    return {
        "schema": PAIR_RECEIPT_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "fold": fold,
        "plan": {"path": str(plan_path.resolve()), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "same_fold_shared_initialization": True,
        "outer_test_accessed": False,
        "candidates": records,
        "shared_initial_variable_sha256": shared,
        "shared_initial_mapping_sha256": records[0][
            "shared_initial_mapping_sha256"
        ],
    }


def write_pair_receipt(
    *, plan_path: Path, candidate_root: Path, fold: int, output: Path
) -> dict[str, Any]:
    payload = build_pair_receipt(
        plan_path=plan_path, candidate_root=candidate_root, fold=fold
    )
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        if not output.is_file() or not sidecar.is_file():
            raise TCMAdapterPairGuardError("partial pair receipt already exists")
        existing = _load_json(output)
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if dict(existing) != payload or fields != [sha256(output), output.name]:
            raise TCMAdapterPairGuardError(
                "existing pair receipt does not match current candidate artifacts"
            )
        return {**payload, "receipt_path": str(output), "receipt_sha256": sha256(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {**payload, "receipt_path": str(output), "receipt_sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = write_pair_receipt(
            plan_path=args.plan,
            candidate_root=args.candidate_root,
            fold=args.fold,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM adapter pair guard failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
