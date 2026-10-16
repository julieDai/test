#!/usr/bin/env python3
"""Prove that one AE64/PCA64 fold differs only in semantic feature basis.

The two controlled semantic inputs (``input/herb`` and ``input/disease``)
must differ and must each resolve to their frozen bundle.  Every other shared
initial variable and every adapter initial variable must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import tcm_astra_pca64_control_runner as pca_base_runner
from . import tcm_astra_semantic_adapter_runner as adapter_base_runner
from . import tcm_astra_semantic_adapter_selection as adapter_selection
from .tcm_astra_pca64_deterministic_plan import (
    CONTROLLED_FEATURE_INPUT_KEYS,
    EXPECTED_SHARED_INITIAL_KEYS,
    PLAN_SCHEMA,
    PLAN_STATUS,
    artifact,
    expected_adapter_keys,
    load_json,
    object_sha256,
    protocol_controls,
    sha256,
    validate_sidecar,
    verify_artifact,
)
from .tcm_astra_pca64_deterministic_runner import (
    CANDIDATE_COMPLETION_NAME,
    RUNNER_SCHEMA,
)


PAIR_RECEIPT_SCHEMA = "astra-tcm-pca64-deterministic-feature-basis-pair/v2"


class TCMPCA64DeterministicPairGuardError(RuntimeError):
    """Raised when an AE64/PCA64 validation pair is not controlled."""


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_mapping(
    value: object, *, expected_keys: frozenset[str], label: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise TCMPCA64DeterministicPairGuardError(
            f"{label} must contain exactly {len(expected_keys)} expected keys"
        )
    result = {str(key): str(digest) for key, digest in value.items()}
    if not all(_valid_hash(digest) for digest in result.values()):
        raise TCMPCA64DeterministicPairGuardError(
            f"{label} contains an invalid SHA-256"
        )
    return result


def compare_initialization_maps(
    *,
    ae_shared: object,
    pca_shared: object,
    ae_adapter: object,
    pca_adapter: object,
    adapter: str,
) -> dict[str, Any]:
    """Return a controlled pair mapping or fail on any non-feature difference."""

    ae_shared_hashes = _hash_mapping(
        ae_shared,
        expected_keys=EXPECTED_SHARED_INITIAL_KEYS,
        label="AE64 shared initialization",
    )
    pca_shared_hashes = _hash_mapping(
        pca_shared,
        expected_keys=EXPECTED_SHARED_INITIAL_KEYS,
        label="PCA64 shared initialization",
    )
    allowed = set(CONTROLLED_FEATURE_INPUT_KEYS)
    equal_keys = set(EXPECTED_SHARED_INITIAL_KEYS) - allowed
    differences = {
        key
        for key in EXPECTED_SHARED_INITIAL_KEYS
        if ae_shared_hashes[key] != pca_shared_hashes[key]
    }
    if differences != allowed:
        raise TCMPCA64DeterministicPairGuardError(
            "shared initialization differences must be exactly input/herb and "
            f"input/disease; observed {sorted(differences)}"
        )
    common = {key: ae_shared_hashes[key] for key in sorted(equal_keys)}
    if len(common) != 30:
        raise TCMPCA64DeterministicPairGuardError(
            "common shared initialization must contain exactly 30 keys"
        )
    adapter_keys = expected_adapter_keys(adapter)
    ae_adapter_hashes = _hash_mapping(
        ae_adapter,
        expected_keys=adapter_keys,
        label="AE64 adapter initialization",
    )
    pca_adapter_hashes = _hash_mapping(
        pca_adapter,
        expected_keys=adapter_keys,
        label="PCA64 adapter initialization",
    )
    if ae_adapter_hashes != pca_adapter_hashes:
        raise TCMPCA64DeterministicPairGuardError(
            "adapter initial variable mapping differs across feature bases"
        )
    return {
        "same_noncontrolled_shared_initialization": True,
        "different_controlled_semantic_inputs": True,
        "same_adapter_initialization": True,
        "common_shared_variable_sha256": common,
        "common_shared_mapping_sha256": object_sha256(common),
        "controlled_feature_input_sha256": {
            "ae64": {
                key: ae_shared_hashes[key]
                for key in sorted(CONTROLLED_FEATURE_INPUT_KEYS)
            },
            "pca64": {
                key: pca_shared_hashes[key]
                for key in sorted(CONTROLLED_FEATURE_INPUT_KEYS)
            },
        },
        "adapter_initial_variable_sha256": ae_adapter_hashes,
        "adapter_initial_mapping_sha256": object_sha256(ae_adapter_hashes),
    }


def _audit_array_sha256(values: object) -> str:
    """Use the exact array-hash encoding emitted by the adapter audit."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _semantic_matrix_hash(path: Path) -> str:
    if path.suffix.lower() != ".npy":
        raise TCMPCA64DeterministicPairGuardError(
            f"semantic matrix must be a NumPy artifact: {path}"
        )
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise TCMPCA64DeterministicPairGuardError(
            f"cannot load semantic matrix {path}: {error}"
        ) from error
    return _audit_array_sha256(np.asarray(values, dtype=np.float32))


def _plan_and_digest(path: Path) -> tuple[Mapping[str, Any], str]:
    resolved = path.resolve(strict=True)
    digest = validate_sidecar(resolved, label="deterministic PCA64 plan")
    plan = load_json(resolved)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != 42
        or plan.get("outer_test_accessed") is not False
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "deterministic PCA64 plan contract differs"
        )
    source = Path(__file__).resolve()
    guard_record = plan.get("source_freeze", {}).get("pair_guard")
    if (
        not isinstance(guard_record, Mapping)
        or Path(str(guard_record.get("path", ""))).resolve() != source
        or guard_record.get("sha256") != sha256(source)
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "pair guard source differs from the frozen plan"
        )
    return plan, digest


def _verify_bundle_input_binding(
    *,
    plan: Mapping[str, Any],
    bundle_key: str,
    protocol: Mapping[str, Any],
    shared_hashes: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    bundle = plan.get("feature_bundles", {}).get(bundle_key)
    if not isinstance(bundle, Mapping):
        raise TCMPCA64DeterministicPairGuardError(
            f"{bundle_key} bundle binding is absent"
        )
    manifest = verify_artifact(
        bundle.get("manifest"), label=f"{bundle_key} bundle manifest"
    )
    observed_bundle = protocol.get("feature_bundle")
    if (
        not isinstance(observed_bundle, Mapping)
        or Path(str(observed_bundle.get("path", ""))).resolve() != manifest
        or observed_bundle.get("sha256") != sha256(manifest)
    ):
        raise TCMPCA64DeterministicPairGuardError(
            f"runtime {bundle_key} bundle differs from the plan"
        )
    matrices = bundle.get("entities")
    runtime_matrices = protocol.get("feature_matrices")
    if not isinstance(matrices, Mapping) or not isinstance(runtime_matrices, Mapping):
        raise TCMPCA64DeterministicPairGuardError(
            f"{bundle_key} feature-matrix evidence is absent"
        )
    records: dict[str, dict[str, object]] = {}
    for entity in ("herb", "compound", "protein", "disease"):
        entity_record = matrices.get(entity)
        runtime_record = runtime_matrices.get(entity)
        if not isinstance(entity_record, Mapping) or not isinstance(
            runtime_record, Mapping
        ):
            raise TCMPCA64DeterministicPairGuardError(
                f"{bundle_key} {entity} matrix binding is absent"
            )
        matrix = verify_artifact(
            entity_record.get("matrix"), label=f"{bundle_key} {entity} matrix"
        )
        if (
            Path(str(runtime_record.get("used_path", ""))).resolve() != matrix
            or runtime_record.get("used_sha256") != sha256(matrix)
        ):
            raise TCMPCA64DeterministicPairGuardError(
                f"runtime {bundle_key} {entity} matrix differs"
            )
        records[entity] = {
            "matrix": artifact(matrix),
        }
        if entity in {"herb", "disease"}:
            audit_hash = _semantic_matrix_hash(matrix)
            if shared_hashes[f"input/{entity}"] != audit_hash:
                raise TCMPCA64DeterministicPairGuardError(
                    f"{bundle_key} input/{entity} audit hash is not bundle-bound"
                )
            records[entity]["audit_array_sha256"] = audit_hash
    return records


def _verify_validation_rows(
    *,
    fold: int,
    fold_plan: Mapping[str, Any],
    ae_path: Path,
    pca_path: Path,
) -> dict[str, str]:
    ae_table = adapter_selection._prediction_table(ae_path)
    pca_table = adapter_selection._prediction_table(pca_path)
    for field in ("row_id", "pair", "label"):
        if not np.array_equal(ae_table[field], pca_table[field]):
            raise TCMPCA64DeterministicPairGuardError(
                f"fold {fold} validation {field} differs across feature bases"
            )
    result: dict[str, str] = {}
    for key, field in (
        ("validation_row_id_sha256", "row_id"),
        ("validation_pair_sha256", "pair"),
        ("validation_label_sha256", "label"),
    ):
        digest = adapter_selection._array_sha256(ae_table[field])
        if digest != fold_plan.get(key):
            raise TCMPCA64DeterministicPairGuardError(
                f"fold {fold} frozen validation {field} hash differs"
            )
        result[key] = digest
    return result


def build_pair_receipt(
    *, plan_path: Path, candidate_root: Path, fold: int
) -> dict[str, Any]:
    if fold not in range(1, 6):
        raise TCMPCA64DeterministicPairGuardError("fold must be in 1..5")
    plan, plan_sha = _plan_and_digest(plan_path)
    fold_plan = plan.get("fold_hyperparameters", {}).get(str(fold))
    if not isinstance(fold_plan, Mapping):
        raise TCMPCA64DeterministicPairGuardError(f"plan has no fold {fold}")
    adapter = str(fold_plan.get("semantic_adapter", ""))
    expected_adapter_keys(adapter)

    ae_artifacts = fold_plan.get("ae64_artifacts")
    if not isinstance(ae_artifacts, Mapping):
        raise TCMPCA64DeterministicPairGuardError(
            f"fold {fold} AE64 artifacts are absent"
        )
    ae_protocol_path = verify_artifact(
        ae_artifacts.get("protocol"), label=f"fold {fold} AE64 protocol"
    )
    ae_metrics_path = verify_artifact(
        ae_artifacts.get("metrics"), label=f"fold {fold} AE64 metrics"
    )
    ae_audit_path = verify_artifact(
        ae_artifacts.get("adapter_audit"), label=f"fold {fold} AE64 audit"
    )
    ae_predictions_path = verify_artifact(
        ae_artifacts.get("validation_predictions"),
        label=f"fold {fold} AE64 validation predictions",
    )
    verify_artifact(
        fold_plan.get("ae64_completion_marker"),
        label=f"fold {fold} AE64 completion marker",
    )
    ae_root = ae_audit_path.parent
    try:
        pca_base_runner.reject_outer_artifacts(ae_root)
    except Exception as error:
        raise TCMPCA64DeterministicPairGuardError(str(error)) from error

    pca_root = (candidate_root / "pca64_candidate" / f"fold_{fold}").resolve()
    pca_completion = pca_root / CANDIDATE_COMPLETION_NAME
    pca_protocol_path = pca_root / "protocol.json"
    pca_metrics_path = pca_root / "results" / "metrics.json"
    pca_predictions_path = pca_root / "results" / "validation_predictions.npz"
    pca_audit_path = pca_root / "tcm_semantic_adapter_audit.json"
    for path in (
        pca_completion,
        pca_protocol_path,
        pca_metrics_path,
        pca_predictions_path,
        pca_audit_path,
    ):
        if not path.is_file():
            raise TCMPCA64DeterministicPairGuardError(
                f"PCA64 candidate artifact is absent: {path}"
            )
    try:
        pca_base_runner.reject_outer_artifacts(pca_root)
    except Exception as error:
        raise TCMPCA64DeterministicPairGuardError(str(error)) from error

    ae_protocol = load_json(ae_protocol_path)
    ae_metrics = load_json(ae_metrics_path)
    ae_audit = load_json(ae_audit_path)
    pca_protocol = load_json(pca_protocol_path)
    pca_metrics = load_json(pca_metrics_path)
    pca_audit = load_json(pca_audit_path)
    binding = pca_protocol.get("tcm_pca64_deterministic_candidate")
    if not isinstance(binding, Mapping) or binding.get("schema") != RUNNER_SCHEMA:
        raise TCMPCA64DeterministicPairGuardError(
            "PCA64 deterministic runner binding is absent"
        )
    if (
        binding.get("plan", {}).get("sha256") != plan_sha
        or Path(str(binding.get("plan", {}).get("path", ""))).resolve()
        != plan_path.resolve()
        or binding.get("candidate", {}).get("id") != "pca64_candidate"
        or dict(binding.get("fold_hyperparameters", {})) != dict(fold_plan)
        or binding.get("deterministic_initialization")
        != plan.get("deterministic_initialization")
        or binding.get("outer_test_materialised") is not False
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "PCA64 candidate binding differs from the plan"
        )
    seed_receipt = binding.get("tensorflow_graph_seed_receipt")
    if (
        not isinstance(seed_receipt, Mapping)
        or seed_receipt.get("seed") != 42
        or seed_receipt.get("graph_was_empty_before_seed") is not True
        or seed_receipt.get("tensorflow_v1_behavior_enabled_before_seed") is not True
        or seed_receipt.get("seed_set_before_model_import") is not True
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "PCA64 TensorFlow graph-seed receipt differs"
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
        "Semantic_Adapter": adapter,
        "Feature_Basis": "deterministic_shared_pca64_v1",
        "TensorFlow_Graph_Seed": 42,
    }
    for key, expected in expected_metrics.items():
        if pca_metrics.get(key) != expected:
            raise TCMPCA64DeterministicPairGuardError(
                f"fold {fold} PCA64 metric {key} differs"
            )
    if ae_protocol.get("fold") != fold or ae_metrics.get("Fold") != fold:
        raise TCMPCA64DeterministicPairGuardError("AE64 fold identity differs")
    if (
        ae_metrics.get("Candidate_ID")
        != fold_plan.get("adapter_selection_candidate_id")
        or ae_metrics.get("Semantic_Adapter") != adapter
        or ae_metrics.get("TensorFlow_Graph_Seed") != 42
    ):
        raise TCMPCA64DeterministicPairGuardError("AE64 winner identity differs")

    for name, protocol, metrics, audit in (
        ("AE64", ae_protocol, ae_metrics, ae_audit),
        ("PCA64", pca_protocol, pca_metrics, pca_audit),
    ):
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
            or int(metrics.get("Unselected_Test_Models_Evaluated", -1)) != 0
        ):
            raise TCMPCA64DeterministicPairGuardError(
                f"{name} arm is not validation-only"
            )
        try:
            adapter_base_runner._validate_audit(
                audit,
                adapter=adapter,
                expected_steps=int(metrics.get("Optimizer_Steps", -1)),
            )
        except Exception as error:
            raise TCMPCA64DeterministicPairGuardError(str(error)) from error
        audit_seed = audit.get("deterministic_initialization")
        if (
            not isinstance(audit_seed, Mapping)
            or audit_seed.get("tensorflow_graph_seed") != 42
            or audit_seed.get("seed_observed_before_model_construction") is not True
        ):
            raise TCMPCA64DeterministicPairGuardError(
                f"{name} audit lacks deterministic graph-seed evidence"
            )

    ae_controls = protocol_controls(ae_protocol)
    pca_controls = protocol_controls(pca_protocol)
    frozen_controls = fold_plan.get("ae64_protocol_controls")
    if ae_controls != frozen_controls or pca_controls != frozen_controls:
        raise TCMPCA64DeterministicPairGuardError(
            "protocol split/relation/runtime/hyperparameter/step controls differ"
        )
    if (
        int(ae_metrics.get("Optimizer_Steps", -1))
        != int(pca_metrics.get("Optimizer_Steps", -1))
        or int(ae_audit.get("optimizer_steps", -1))
        != int(pca_audit.get("optimizer_steps", -1))
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "optimizer step counts differ across feature bases"
        )
    if (
        dict(ae_audit.get("model_source", {}))
        != dict(pca_audit.get("model_source", {}))
        or dict(ae_audit.get("model_source", {}))
        != dict(fold_plan.get("ae64_model_source", {}))
    ):
        raise TCMPCA64DeterministicPairGuardError(
            "model source differs across feature bases"
        )

    ae_initial = ae_audit.get("initial")
    pca_initial = pca_audit.get("initial")
    if not isinstance(ae_initial, Mapping) or not isinstance(pca_initial, Mapping):
        raise TCMPCA64DeterministicPairGuardError(
            "initial audit snapshot is absent"
        )
    comparison = compare_initialization_maps(
        ae_shared=ae_initial.get("shared_variable_sha256"),
        pca_shared=pca_initial.get("shared_variable_sha256"),
        ae_adapter=ae_initial.get("adapter_variable_sha256"),
        pca_adapter=pca_initial.get("adapter_variable_sha256"),
        adapter=adapter,
    )
    ae_shared = dict(ae_initial["shared_variable_sha256"])
    pca_shared = dict(pca_initial["shared_variable_sha256"])
    bundle_inputs = {
        "ae64": _verify_bundle_input_binding(
            plan=plan,
            bundle_key="ae64",
            protocol=ae_protocol,
            shared_hashes=ae_shared,
        ),
        "pca64": _verify_bundle_input_binding(
            plan=plan,
            bundle_key="pca64",
            protocol=pca_protocol,
            shared_hashes=pca_shared,
        ),
    }
    row_hashes = _verify_validation_rows(
        fold=fold,
        fold_plan=fold_plan,
        ae_path=ae_predictions_path,
        pca_path=pca_predictions_path,
    )
    return {
        "schema": PAIR_RECEIPT_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "fold": fold,
        "plan": {"path": str(plan_path.resolve()), "sha256": plan_sha},
        "candidate_root": str(candidate_root.resolve()),
        "semantic_adapter": adapter,
        "adapter_selection_candidate_id": fold_plan[
            "adapter_selection_candidate_id"
        ],
        "single_variable_feature_basis_pair": True,
        "outer_test_accessed": False,
        **comparison,
        "bundle_inputs": bundle_inputs,
        "validation_identity": row_hashes,
        "protocol_controls_sha256": ae_controls["contract_sha256"],
        "optimizer_steps": int(ae_metrics["Optimizer_Steps"]),
        "arms": {
            "ae64": {
                "feature_basis": "shared_ae64",
                "protocol": artifact(ae_protocol_path),
                "metrics": artifact(ae_metrics_path),
                "adapter_audit": artifact(ae_audit_path),
                "validation_predictions": artifact(ae_predictions_path),
                "completion_marker": dict(fold_plan["ae64_completion_marker"]),
            },
            "pca64": {
                "feature_basis": "deterministic_shared_pca64_v1",
                "protocol": artifact(pca_protocol_path),
                "metrics": artifact(pca_metrics_path),
                "adapter_audit": artifact(pca_audit_path),
                "validation_predictions": artifact(pca_predictions_path),
                "completion_marker": artifact(pca_completion),
            },
        },
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
            raise TCMPCA64DeterministicPairGuardError(
                "partial feature-basis pair receipt already exists"
            )
        existing = load_json(output)
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if dict(existing) != payload or fields != [sha256(output), output.name]:
            raise TCMPCA64DeterministicPairGuardError(
                "existing pair receipt differs from current candidate artifacts"
            )
        return {
            **payload,
            "receipt_path": str(output),
            "receipt_sha256": sha256(output),
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
        print(f"TCM deterministic PCA64 pair guard failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
