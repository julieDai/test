#!/usr/bin/env python3
"""Freeze the validation-only AE64 versus deterministic shared-PCA64 control.

The parent semantic-adapter selection must already be complete.  For every
fold, this plan freezes its selected adapter, epoch budget, learning rate, and
AE64 validation prediction as an immutable control.  The only executable arm
is then PCA64.  No protocol rows or outer-test predictions are loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import load_protocol_manifest
from .tcm_astra_semantic_adapter_plan import PLAN_SCHEMA as ADAPTER_PLAN_SCHEMA
from .tcm_astra_semantic_adapter_selection import (
    SELECTION_SCHEMA as ADAPTER_SELECTION_SCHEMA,
)
from .tcm_feature_bundle import validate_tcm_feature_bundle
from .tcm_pca64_feature_bundle import validate_tcm_pca64_feature_bundle
from .backend_paths import TCM_BACKEND_ROOT


PLAN_SCHEMA = "astra-tcm-pca64-control-plan/v1"
MIN_PCA64_AUPRC_GAIN = 1e-4
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"


class TCMPCA64ControlPlanError(RuntimeError):
    """Raised when a PCA64 control cannot be frozen unambiguously."""


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
        raise TCMPCA64ControlPlanError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMPCA64ControlPlanError(f"JSON root must be an object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TCMPCA64ControlPlanError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except OSError as error:
        raise TCMPCA64ControlPlanError(f"{label} artifact is absent") from error
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise TCMPCA64ControlPlanError(f"{label} artifact changed")
    if "bytes" in record and path.stat().st_size != record.get("bytes"):
        raise TCMPCA64ControlPlanError(f"{label} byte count changed")
    return path


def _validate_sha_sidecar(path: Path, *, label: str) -> str:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMPCA64ControlPlanError(f"{label} SHA-256 sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise TCMPCA64ControlPlanError(f"{label} SHA-256 sidecar differs")
    return digest


def _source_freeze() -> dict[str, dict[str, object]]:
    paths = {
        "model": TCM_BACKEND_ROOT / "ASTRATCMSemanticAdapter.py",
        "central_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
        "adapter_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_runner.py",
        "plan_generator": Path(__file__).resolve(),
        "candidate_runner": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_runner.py",
        "selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_selection.py",
        "outer_evaluator": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_pca64_control_evaluate.py",
        "outer_safety_primitives": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_evaluate.py",
        "protocol": SOURCE_ROOT / "astra/benchmark/protocol.py",
        "semantic_adapter_selector": SOURCE_ROOT
        / "astra/benchmark/tcm_astra_semantic_adapter_selection.py",
        "pca64_bundle_builder": SOURCE_ROOT
        / "astra/benchmark/tcm_pca64_feature_bundle.py",
        "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
    }
    try:
        return {name: _artifact(path) for name, path in paths.items()}
    except FileNotFoundError as error:
        raise TCMPCA64ControlPlanError(
            f"PCA64 control implementation is incomplete: {error}"
        ) from error


def _winner_reference(
    *, fold: int, fold_selection: Mapping[str, Any]
) -> dict[str, Any]:
    winner = fold_selection.get("winner")
    if not isinstance(winner, Mapping) or winner.get("fold") != fold:
        raise TCMPCA64ControlPlanError(f"fold {fold} adapter winner is invalid")
    if fold_selection.get("selected_candidate") != winner.get("id"):
        raise TCMPCA64ControlPlanError(f"fold {fold} selected candidate differs")
    if fold_selection.get("selected_adapter") != winner.get("semantic_adapter"):
        raise TCMPCA64ControlPlanError(f"fold {fold} selected adapter differs")
    adapter = str(winner.get("semantic_adapter", ""))
    if adapter not in {"identity", "residual_mlp"}:
        raise TCMPCA64ControlPlanError(f"fold {fold} adapter is unsupported")
    try:
        epochs = int(winner["epochs"])
        learning_rate = float(winner["learning_rate"])
        auprc = float(winner["validation_auprc"])
        auroc = float(winner["validation_auroc"])
        threshold = float(winner["validation_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMPCA64ControlPlanError(
            f"fold {fold} winner hyperparameters or metrics are invalid"
        ) from error
    if (
        epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not all(math.isfinite(value) for value in (auprc, auroc, threshold))
    ):
        raise TCMPCA64ControlPlanError(f"fold {fold} winner values are invalid")

    artifacts: dict[str, dict[str, object]] = {}
    for key in (
        "validation_predictions",
        "herb_embedding",
        "disease_embedding",
        "metrics",
        "protocol",
        "adapter_audit",
    ):
        hash_key = f"{key}_sha256"
        record = {"path": winner.get(key), "sha256": winner.get(hash_key)}
        path = _verify_artifact(record, label=f"fold {fold} AE64 {key}")
        if "test" in path.name.lower() or (
            key == "validation_predictions"
            and (path.parent / "test_predictions.npz").exists()
        ):
            raise TCMPCA64ControlPlanError(
                f"fold {fold} AE64 reference exposes an outer-test artifact"
            )
        artifacts[key] = _artifact(path)
    hashes: dict[str, str] = {}
    for key in (
        "validation_row_id_sha256",
        "validation_pair_sha256",
        "validation_label_sha256",
    ):
        value = winner.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TCMPCA64ControlPlanError(f"fold {fold} {key} is invalid")
        hashes[key] = value
    return {
        "adapter_selection_candidate_id": str(winner["id"]),
        "semantic_adapter": adapter,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "ae64_validation_auprc": auprc,
        "ae64_validation_auroc": auroc,
        "ae64_validation_threshold": threshold,
        "ae64_artifacts": artifacts,
        **hashes,
    }


def build_plan(
    *,
    adapter_selection_path: Path,
    pca64_bundle_manifest: Path,
) -> dict[str, Any]:
    selection_path = adapter_selection_path.resolve(strict=True)
    selection_sha = _validate_sha_sidecar(
        selection_path, label="semantic-adapter selection"
    )
    selection = _load_json(selection_path)
    if (
        selection.get("schema") != ADAPTER_SELECTION_SCHEMA
        or selection.get("dataset") != "tcm-suite"
        or int(selection.get("seed", -1)) != 42
        or selection.get("outer_test_accessed") is not False
    ):
        raise TCMPCA64ControlPlanError("unsupported semantic-adapter selection")
    adapter_plan_path = _verify_artifact(
        selection.get("plan"), label="semantic-adapter plan"
    )
    adapter_plan = _load_json(adapter_plan_path)
    if (
        adapter_plan.get("schema") != ADAPTER_PLAN_SCHEMA
        or adapter_plan.get("status") != "frozen-before-stage2-candidate-testing"
    ):
        raise TCMPCA64ControlPlanError("semantic-adapter parent plan is incompatible")
    if sha256(adapter_plan_path) != selection["plan"]["sha256"]:
        raise TCMPCA64ControlPlanError("semantic-adapter plan hash differs")

    protocol_path = _verify_artifact(
        adapter_plan.get("protocol_manifest"), label="protocol manifest"
    )
    protocol = load_protocol_manifest(protocol_path)
    if protocol.dataset != "tcm-suite":
        raise TCMPCA64ControlPlanError("protocol dataset is not tcm-suite")
    ae_bundle_record = adapter_plan.get("feature_bundle")
    if not isinstance(ae_bundle_record, Mapping):
        raise TCMPCA64ControlPlanError("AE64 feature bundle binding is absent")
    ae_manifest_path = _verify_artifact(
        ae_bundle_record.get("manifest"), label="AE64 feature bundle manifest"
    )
    ae_bundle_root = Path(str(ae_bundle_record.get("path", ""))).resolve(strict=True)
    if ae_manifest_path.parent != ae_bundle_root:
        raise TCMPCA64ControlPlanError("AE64 feature bundle root differs")
    ae_validated = validate_tcm_feature_bundle(
        ae_manifest_path,
        protocol=protocol,
    )

    pca_manifest_path = pca64_bundle_manifest.resolve(strict=True)
    pca_validated = validate_tcm_pca64_feature_bundle(
        pca_manifest_path,
        protocol=protocol,
        verify_transform=False,
    )
    for entity in ("compound", "protein"):
        if (
            ae_validated["entities"][entity]["matrix_sha256"]
            != pca_validated["entities"][entity]["matrix_sha256"]
        ):
            raise TCMPCA64ControlPlanError(
                f"PCA64 changes the controlled {entity} feature matrix"
            )

    folds = selection.get("folds")
    expected_folds = {str(fold) for fold in range(1, 6)}
    if not isinstance(folds, Mapping) or set(folds) != expected_folds:
        raise TCMPCA64ControlPlanError("adapter selection must contain five folds")
    fold_hyperparameters = {
        str(fold): _winner_reference(
            fold=fold,
            fold_selection=folds[str(fold)],
        )
        for fold in range(1, 6)
    }
    source_freeze = _source_freeze()
    source_freeze["parent_adapter_selection"] = _artifact(selection_path)
    source_freeze["parent_adapter_plan"] = _artifact(adapter_plan_path)

    return {
        "schema": PLAN_SCHEMA,
        "status": "frozen-before-pca64-validation-candidates",
        "dataset": "tcm-suite",
        "seed": 42,
        "interpretation": (
            "seed-42 same-split post-baseline feature-basis diagnostic; "
            "inner validation only"
        ),
        "parent_adapter_selection": _artifact(selection_path),
        "parent_adapter_selection_sha256": selection_sha,
        "parent_adapter_plan": _artifact(adapter_plan_path),
        "protocol_manifest": _artifact(protocol_path),
        "feature_bundles": {
            "ae64": {
                "path": str(ae_bundle_root),
                "manifest": _artifact(ae_manifest_path),
            },
            "pca64": {
                "path": str(pca_manifest_path.parent),
                "manifest": _artifact(pca_manifest_path),
            },
        },
        "fixed": {
            "batch_size": 2000,
            "embedding_dimension": 64,
            "reg_u": 0.001,
            "text_fusion_alpha": 1.0,
            "semantic_input_dim": 64,
            "semantic_mlp_hidden_dim": 32,
            "semantic_freeze_input": False,
        },
        "fold_hyperparameters": fold_hyperparameters,
        "candidates": [
            {
                "id": "ae64_frozen_control",
                "feature_basis": "shared_ae64",
                "execution": "frozen_parent_validation_reference",
                "feature_bundle_key": "ae64",
            },
            {
                "id": "pca64_candidate",
                "feature_basis": "deterministic_shared_pca64_v1",
                "execution": "validation_only_candidate",
                "feature_bundle_key": "pca64",
            },
        ],
        "selection_metric": "Validation_AUPRC",
        "minimum_pca64_auprc_gain": MIN_PCA64_AUPRC_GAIN,
        "selection_policy": (
            "Within each fold select PCA64 only when PCA64 Validation_AUPRC minus "
            "the frozen AE64 Validation_AUPRC is at least 0.0001; otherwise retain "
            "the frozen AE64 control. Outer-test rows are forbidden."
        ),
        "source_freeze": source_freeze,
        "outer_test_accessed": False,
        "formal_scope": (
            "Development diagnostic only because seed-42 outer folds were already "
            "observed; a fresh frozen split is required for formal confirmation."
        ),
    }


def freeze_plan(
    *,
    adapter_selection: Path,
    pca64_bundle_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMPCA64ControlPlanError(
            f"refusing to overwrite plan or sidecar: {output}, {sidecar}"
        )
    payload = build_plan(
        adapter_selection_path=adapter_selection,
        pca64_bundle_manifest=pca64_bundle_manifest,
    )
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
    parser.add_argument("--adapter-selection", type=Path, required=True)
    parser.add_argument("--pca64-bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_plan(
            adapter_selection=args.adapter_selection,
            pca64_bundle_manifest=args.pca64_bundle_manifest,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM PCA64 control plan freeze failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
