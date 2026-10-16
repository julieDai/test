#!/usr/bin/env python3
"""One-shot final outer evaluation after all TCM validation-only stages.

The evaluator completely replays the frozen AE64-versus-PCA64 decision before
touching protocol arrays.  It then durably burns a content-addressed global
ledger receipt and reserves all five folds.  Only after that irreversible burn
does the official protocol loader expose outer rows.  Exactly one selected
embedding pair per fold is scored with its already-frozen validation threshold.

This is deliberately a seed-42 post-baseline diagnostic, never a formal result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import tcm_astra_pca64_control_selection as selection_module
from . import tcm_astra_semantic_adapter_evaluate as safety
from .metrics import (
    AUPRC_DEFINITION,
    evaluate_test_scores,
    evaluate_validation_scores,
    load_predictions,
    save_predictions,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .tcm_astra_hparam_selection import ARRAY_HASH_SCHEMA
from .tcm_astra_pca64_control_plan import (
    MIN_PCA64_AUPRC_GAIN,
    PLAN_SCHEMA,
)
from .tcm_astra_pca64_control_runner import reject_outer_artifacts
from .tcm_astra_pca64_control_selection import (
    SELECTION_SCHEMA,
    _fold_selection_payload,
    _validate_pca_candidate,
)
from .tcm_astra_semantic_adapter_selection import (
    _array_sha256 as _selection_array_sha256,
    _prediction_table as _selection_prediction_table,
)


EVALUATION_SCHEMA = "astra-tcm-pca64-control-outer-test-summary/v1"
PREFLIGHT_SCHEMA = "astra-tcm-pca64-control-outer-preflight/v1"
ACCESS_SCHEMA = "astra-tcm-pca64-control-outer-access/v1"
COMPLETION_SCHEMA = "astra-tcm-pca64-control-outer-completion/v1"
EXPECTED_FIXED = {
    "batch_size": 2000,
    "embedding_dimension": 64,
    "reg_u": 0.001,
    "text_fusion_alpha": 1.0,
    "semantic_input_dim": 64,
    "semantic_mlp_hidden_dim": 32,
    "semantic_freeze_input": False,
}
EXPECTED_CANDIDATES = [
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
]
SUMMARY_METRICS = safety.SUMMARY_METRICS
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GLOBAL_LEDGER_ROOT = (
    PROJECT_ROOT / "artifacts" / "tcm_suite" / "pca64_control_outer_access_ledger"
)


class TCMPCA64OuterEvaluationError(safety.TCMAdapterEvaluationError):
    """Raised when the one-shot PCA64-control evaluation is not auditable."""


@dataclass(frozen=True)
class WinnerPreflight:
    fold: int
    candidate_id: str
    feature_basis: str
    adapter: str
    epochs: int
    learning_rate: float
    threshold: float
    validation_auprc: float
    validation_auroc: float
    validation_path: Path
    validation_sha256: str
    herb_path: Path
    herb_sha256: str
    disease_path: Path
    disease_sha256: str
    herb: np.ndarray
    disease: np.ndarray
    validation_identity: Mapping[str, str]


@dataclass(frozen=True)
class EvaluationPreflight:
    selection_path: Path
    selection_sha256: str
    plan_path: Path
    plan_sha256: str
    candidate_root: Path
    selection: Mapping[str, Any]
    plan: Mapping[str, Any]
    protocol_package: safety.ProtocolMetadataPackage
    winners: tuple[WinnerPreflight, ...]
    artifact_snapshots: tuple[safety.FileSnapshot, ...]
    validation_roots: tuple[Path, ...]
    candidate_records_audited: int


def sha256(path: Path) -> str:
    return safety.sha256(path)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMPCA64OuterEvaluationError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TCMPCA64OuterEvaluationError(f"JSON root must be an object: {path}")
    return payload


def _validate_sidecar(path: Path, *, label: str) -> str:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMPCA64OuterEvaluationError(f"{label} SHA-256 sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, path.name]:
        raise TCMPCA64OuterEvaluationError(f"{label} SHA-256 sidecar differs")
    return digest


def _artifact_snapshot(record: object, *, label: str) -> safety.FileSnapshot:
    if not isinstance(record, Mapping):
        raise TCMPCA64OuterEvaluationError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record.get("path", ""))).resolve(strict=True)
    except OSError as error:
        raise TCMPCA64OuterEvaluationError(f"{label} artifact is absent") from error
    digest = str(record.get("sha256", ""))
    if not path.is_file() or digest != sha256(path):
        raise TCMPCA64OuterEvaluationError(f"{label} artifact changed")
    expected_bytes = record.get("bytes", path.stat().st_size)
    if isinstance(expected_bytes, bool) or int(expected_bytes) != path.stat().st_size:
        raise TCMPCA64OuterEvaluationError(f"{label} byte count changed")
    return safety.FileSnapshot(
        label=label,
        path=path,
        bytes=path.stat().st_size,
        sha256=digest,
    )


def _path_snapshot(
    path_value: object, digest_value: object, *, label: str
) -> safety.FileSnapshot:
    return _artifact_snapshot(
        {"path": path_value, "sha256": digest_value},
        label=label,
    )


def _unique_snapshots(
    values: Sequence[safety.FileSnapshot],
) -> tuple[safety.FileSnapshot, ...]:
    by_path: dict[Path, safety.FileSnapshot] = {}
    for item in values:
        existing = by_path.get(item.path)
        if existing is not None and (
            existing.bytes != item.bytes or existing.sha256 != item.sha256
        ):
            raise TCMPCA64OuterEvaluationError(
                f"artifact has conflicting frozen identities: {item.path}"
            )
        if existing is None:
            by_path[item.path] = item
    return tuple(by_path[path] for path in sorted(by_path, key=str))


def _snapshot_relative_artifacts(
    value: object,
    *,
    root: Path,
    label: str,
    snapshots: list[safety.FileSnapshot],
) -> None:
    """Recursively re-hash every portable artifact declared by a bundle."""

    if isinstance(value, Mapping):
        if {"relative_path", "bytes", "sha256"}.issubset(value):
            relative = value.get("relative_path")
            if not isinstance(relative, str) or not relative:
                raise TCMPCA64OuterEvaluationError(
                    f"{label} relative artifact path is invalid"
                )
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise TCMPCA64OuterEvaluationError(
                    f"{label} artifact escapes its bundle root"
                ) from error
            snapshots.append(
                _artifact_snapshot(
                    {
                        "path": str(path),
                        "bytes": value.get("bytes"),
                        "sha256": value.get("sha256"),
                    },
                    label=label,
                )
            )
            return
        for key, nested in value.items():
            _snapshot_relative_artifacts(
                nested,
                root=root,
                label=f"{label}/{key}",
                snapshots=snapshots,
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _snapshot_relative_artifacts(
                nested,
                root=root,
                label=f"{label}/{index}",
                snapshots=snapshots,
            )


def _bundle_shape(
    record: object,
    *,
    key: str,
    package: safety.ProtocolMetadataPackage,
    snapshots: list[safety.FileSnapshot],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(record, Mapping):
        raise TCMPCA64OuterEvaluationError(f"{key} feature-bundle record is absent")
    snapshot = _artifact_snapshot(
        record.get("manifest"), label=f"{key} feature-bundle manifest"
    )
    root = Path(str(record.get("path", ""))).resolve(strict=True)
    if snapshot.path.parent != root:
        raise TCMPCA64OuterEvaluationError(f"{key} feature-bundle root differs")
    snapshots.append(snapshot)
    manifest = _load_json(snapshot.path)
    _snapshot_relative_artifacts(
        manifest,
        root=root,
        label=f"{key} bundle",
        snapshots=snapshots,
    )
    try:
        herb_shape = tuple(int(value) for value in manifest["entities"]["herb"]["shape"])
        disease_shape = tuple(
            int(value) for value in manifest["entities"]["disease"]["shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TCMPCA64OuterEvaluationError(
            f"{key} bundle lacks herb/disease shapes"
        ) from error
    counts = package.manifest["counts"]
    if herb_shape != (counts["herbs"], 64) or disease_shape != (
        counts["diseases"],
        64,
    ):
        raise TCMPCA64OuterEvaluationError(
            f"{key} feature dimensions differ from the protocol"
        )
    return herb_shape, disease_shape


def _canonical_validation_identity(table: object) -> dict[str, str]:
    return safety._canonical_prediction_identity(table)


def _standard_ae_record(
    *, fold: int, fold_plan: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = fold_plan.get("ae64_artifacts")
    if not isinstance(artifacts, Mapping):
        raise TCMPCA64OuterEvaluationError(f"fold {fold} AE64 artifacts are absent")
    required = (
        "validation_predictions",
        "herb_embedding",
        "disease_embedding",
        "metrics",
        "protocol",
        "adapter_audit",
    )
    if not all(isinstance(artifacts.get(key), Mapping) for key in required):
        raise TCMPCA64OuterEvaluationError(f"fold {fold} AE64 artifacts differ")
    return {
        "id": "ae64_frozen_control",
        "fold": fold,
        "feature_basis": "shared_ae64",
        "semantic_adapter": str(fold_plan["semantic_adapter"]),
        "epochs": int(fold_plan["epochs"]),
        "learning_rate": float(fold_plan["learning_rate"]),
        "validation_auprc": float(fold_plan["ae64_validation_auprc"]),
        "validation_auroc": float(fold_plan["ae64_validation_auroc"]),
        "validation_threshold": float(fold_plan["ae64_validation_threshold"]),
        **{
            key: str(artifacts[key]["path"])
            for key in required
        },
        **{
            f"{key}_sha256": str(artifacts[key]["sha256"])
            for key in required
        },
    }


def _audit_validation_record(
    *,
    record: Mapping[str, Any],
    herb_shape: tuple[int, int],
    disease_shape: tuple[int, int],
    expected_identity: Mapping[str, str],
    snapshots: list[safety.FileSnapshot],
) -> tuple[WinnerPreflight, Path]:
    fold = int(record["fold"])
    candidate_id = str(record["id"])
    artifact_fields = (
        "validation_predictions",
        "herb_embedding",
        "disease_embedding",
        "metrics",
        "protocol",
        "adapter_audit",
    )
    local: dict[str, safety.FileSnapshot] = {}
    for key in artifact_fields:
        snapshot = _path_snapshot(
            record.get(key),
            record.get(f"{key}_sha256"),
            label=f"fold {fold} {candidate_id} {key}",
        )
        local[key] = snapshot
        snapshots.append(snapshot)
    validation_path = local["validation_predictions"].path
    root = local["metrics"].path.parent.parent.resolve(strict=True)
    for key, snapshot in local.items():
        try:
            snapshot.path.relative_to(root)
        except ValueError as error:
            raise TCMPCA64OuterEvaluationError(
                f"fold {fold} {candidate_id} {key} escapes its validation root"
            ) from error
    reject_outer_artifacts(root)

    raw_table = _selection_prediction_table(validation_path)
    table = load_predictions(validation_path)
    identity = _canonical_validation_identity(table)
    observed_selection_identity = {
        "validation_row_id_sha256": _selection_array_sha256(raw_table["row_id"]),
        "validation_pair_sha256": _selection_array_sha256(raw_table["pair"]),
        "validation_label_sha256": _selection_array_sha256(raw_table["label"]),
    }
    if dict(expected_identity) != observed_selection_identity:
        raise TCMPCA64OuterEvaluationError(
            f"fold {fold} {candidate_id} validation identity differs"
        )
    metrics = _load_json(local["metrics"].path)
    recomputed = evaluate_validation_scores(table.label, table.score)
    for key, value in recomputed.items():
        if key not in metrics or not safety._equal_metric(metrics[key], value):
            raise TCMPCA64OuterEvaluationError(
                f"fold {fold} {candidate_id} validation metric {key} differs"
            )
    bindings = {
        "validation_auprc": recomputed["Validation_AUPRC"],
        "validation_auroc": recomputed["Validation_AUROC"],
        "validation_threshold": recomputed["Validation_Selected_Threshold"],
    }
    for key, value in bindings.items():
        if not safety._equal_metric(record.get(key), value):
            raise TCMPCA64OuterEvaluationError(
                f"fold {fold} {candidate_id} frozen {key} differs"
            )
    herb = np.asarray(np.loadtxt(local["herb_embedding"].path), dtype=np.float64)
    disease = np.asarray(
        np.loadtxt(local["disease_embedding"].path), dtype=np.float64
    )
    if (
        herb.shape != herb_shape
        or disease.shape != disease_shape
        or not np.isfinite(herb).all()
        or not np.isfinite(disease).all()
    ):
        raise TCMPCA64OuterEvaluationError(
            f"fold {fold} {candidate_id} embeddings are invalid"
        )
    reproduced = safety._safe_embedding_scores(herb, disease, table.pair)
    if not np.allclose(reproduced, table.score, rtol=0.0, atol=1e-12):
        raise TCMPCA64OuterEvaluationError(
            f"fold {fold} {candidate_id} validation scores are not reproducible"
        )
    herb.setflags(write=False)
    disease.setflags(write=False)
    winner = WinnerPreflight(
        fold=fold,
        candidate_id=candidate_id,
        feature_basis=str(record["feature_basis"]),
        adapter=str(record["semantic_adapter"]),
        epochs=int(record["epochs"]),
        learning_rate=float(record["learning_rate"]),
        threshold=float(record["validation_threshold"]),
        validation_auprc=float(record["validation_auprc"]),
        validation_auroc=float(record["validation_auroc"]),
        validation_path=validation_path,
        validation_sha256=local["validation_predictions"].sha256,
        herb_path=local["herb_embedding"].path,
        herb_sha256=local["herb_embedding"].sha256,
        disease_path=local["disease_embedding"].path,
        disease_sha256=local["disease_embedding"].sha256,
        herb=herb,
        disease=disease,
        validation_identity=identity,
    )
    return winner, root


def _replay_selection(
    selection_path: Path, *, output_root: Path
) -> EvaluationPreflight:
    selection_path = selection_path.resolve(strict=True)
    selection_sha = _validate_sidecar(selection_path, label="PCA64 selection")
    selection = _load_json(selection_path)
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("dataset") != "tcm-suite"
        or int(selection.get("seed", -1)) != 42
        or selection.get("outer_test_accessed") is not False
        or selection.get("outer_test_evaluator_in_this_module") is not False
    ):
        raise TCMPCA64OuterEvaluationError("PCA64 selection envelope differs")
    selector_record = selection.get("selector")
    selector_path = Path(selection_module.__file__ or "").resolve(strict=True)
    if (
        not isinstance(selector_record, Mapping)
        or Path(str(selector_record.get("path", ""))).resolve(strict=True)
        != selector_path
        or selector_record.get("sha256") != sha256(selector_path)
    ):
        raise TCMPCA64OuterEvaluationError("PCA64 selector changed after selection")

    plan_record = selection.get("plan")
    if not isinstance(plan_record, Mapping):
        raise TCMPCA64OuterEvaluationError("PCA64 plan binding is absent")
    plan_path = Path(str(plan_record.get("path", ""))).resolve(strict=True)
    plan_sha = _validate_sidecar(plan_path, label="PCA64 plan")
    if plan_record.get("sha256") != plan_sha:
        raise TCMPCA64OuterEvaluationError("selection is bound to another plan")
    plan = _load_json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "frozen-before-pca64-validation-candidates"
        or plan.get("dataset") != "tcm-suite"
        or int(plan.get("seed", -1)) != 42
        or plan.get("outer_test_accessed") is not False
        or dict(plan.get("fixed", {})) != EXPECTED_FIXED
        or plan.get("candidates") != EXPECTED_CANDIDATES
        or float(plan.get("minimum_pca64_auprc_gain", math.nan))
        != MIN_PCA64_AUPRC_GAIN
    ):
        raise TCMPCA64OuterEvaluationError("PCA64 plan contract differs")

    snapshots: list[safety.FileSnapshot] = [
        safety.FileSnapshot(
            label="PCA64 selection",
            path=selection_path,
            bytes=selection_path.stat().st_size,
            sha256=selection_sha,
        ),
        safety.FileSnapshot(
            label="PCA64 plan",
            path=plan_path,
            bytes=plan_path.stat().st_size,
            sha256=plan_sha,
        ),
        safety.FileSnapshot(
            label="PCA64 selector",
            path=selector_path,
            bytes=selector_path.stat().st_size,
            sha256=sha256(selector_path),
        ),
    ]
    for key in ("parent_adapter_selection", "parent_adapter_plan"):
        snapshots.append(_artifact_snapshot(plan.get(key), label=key.replace("_", " ")))
    if plan.get("parent_adapter_selection_sha256") not in {
        None,
        plan["parent_adapter_selection"]["sha256"],
    }:
        raise TCMPCA64OuterEvaluationError(
            "parent adapter-selection hash binding differs"
        )
    source_freeze = plan.get("source_freeze")
    if not isinstance(source_freeze, Mapping) or not source_freeze:
        raise TCMPCA64OuterEvaluationError("PCA64 source freeze is absent")
    if not {"outer_evaluator", "outer_safety_primitives"}.issubset(source_freeze):
        raise TCMPCA64OuterEvaluationError(
            "PCA64 plan does not freeze its outer implementation"
        )
    for name, record in source_freeze.items():
        snapshots.append(_artifact_snapshot(record, label=f"source {name}"))
    evaluator_record = source_freeze.get("outer_evaluator")
    evaluator_path = Path(__file__).resolve(strict=True)
    if (
        not isinstance(evaluator_record, Mapping)
        or Path(str(evaluator_record.get("path", ""))).resolve(strict=True)
        != evaluator_path
        or evaluator_record.get("sha256") != sha256(evaluator_path)
    ):
        raise TCMPCA64OuterEvaluationError(
            "plan did not freeze this final outer evaluator"
        )
    safety_record = source_freeze.get("outer_safety_primitives")
    safety_path = Path(safety.__file__ or "").resolve(strict=True)
    if (
        not isinstance(safety_record, Mapping)
        or Path(str(safety_record.get("path", ""))).resolve(strict=True)
        != safety_path
        or safety_record.get("sha256") != sha256(safety_path)
    ):
        raise TCMPCA64OuterEvaluationError(
            "plan did not freeze the outer safety primitives"
        )

    package = safety._read_protocol_metadata(plan)
    snapshots.append(
        safety.FileSnapshot(
            label="protocol manifest",
            path=package.manifest_path,
            bytes=package.manifest_path.stat().st_size,
            sha256=package.manifest_sha256,
        )
    )
    bundles = plan.get("feature_bundles")
    if not isinstance(bundles, Mapping) or set(bundles) != {"ae64", "pca64"}:
        raise TCMPCA64OuterEvaluationError("feature-bundle bindings differ")
    ae_shape = _bundle_shape(
        bundles["ae64"], key="AE64", package=package, snapshots=snapshots
    )
    pca_shape = _bundle_shape(
        bundles["pca64"], key="PCA64", package=package, snapshots=snapshots
    )
    if ae_shape != pca_shape:
        raise TCMPCA64OuterEvaluationError("AE64 and PCA64 entity shapes differ")

    candidate_root = Path(str(selection.get("candidate_root", ""))).resolve(
        strict=True
    )
    output_root = output_root.resolve()
    try:
        output_root.relative_to(candidate_root)
    except ValueError:
        pass
    else:
        raise TCMPCA64OuterEvaluationError(
            "outer output must not be inside the validation candidate tree"
        )
    folds = plan.get("fold_hyperparameters")
    if not isinstance(folds, Mapping) or set(folds) != {
        str(fold) for fold in range(1, 6)
    }:
        raise TCMPCA64OuterEvaluationError("PCA64 plan must contain five folds")

    reconstructed_folds: dict[str, Any] = {}
    winners: list[WinnerPreflight] = []
    validation_roots: set[Path] = set()
    audited_records = 0
    for fold in range(1, 6):
        fold_plan = folds[str(fold)]
        if not isinstance(fold_plan, Mapping):
            raise TCMPCA64OuterEvaluationError(f"fold {fold} plan is invalid")
        pca = _validate_pca_candidate(
            plan=plan,
            plan_sha=plan_sha,
            candidate_root=candidate_root,
            fold=fold,
        )
        reconstructed_folds[str(fold)] = _fold_selection_payload(
            fold_plan=fold_plan,
            pca=pca,
            margin=MIN_PCA64_AUPRC_GAIN,
        )
        ae = _standard_ae_record(fold=fold, fold_plan=fold_plan)
        expected_identity = {
            key: str(fold_plan[key])
            for key in (
                "validation_row_id_sha256",
                "validation_pair_sha256",
                "validation_label_sha256",
            )
        }
        ae_winner, ae_root = _audit_validation_record(
            record=ae,
            herb_shape=ae_shape[0],
            disease_shape=ae_shape[1],
            expected_identity=expected_identity,
            snapshots=snapshots,
        )
        pca_winner, pca_root = _audit_validation_record(
            record=pca,
            herb_shape=pca_shape[0],
            disease_shape=pca_shape[1],
            expected_identity=expected_identity,
            snapshots=snapshots,
        )
        validation_roots.update((ae_root, pca_root))
        audited_records += 2
        selected_id = reconstructed_folds[str(fold)]["selected_candidate"]
        winners.append(
            pca_winner if selected_id == "pca64_candidate" else ae_winner
        )

    reconstructed = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": 42,
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "candidate_root": str(candidate_root),
        "selector": {"path": str(selector_path), "sha256": sha256(selector_path)},
        "selection_metric": "Validation_AUPRC",
        "minimum_pca64_auprc_gain": MIN_PCA64_AUPRC_GAIN,
        "outer_test_accessed": False,
        "outer_test_evaluator_in_this_module": False,
        "interpretation": "seed-42 same-split post-baseline validation diagnostic",
        "folds": reconstructed_folds,
    }
    if dict(selection) != reconstructed:
        raise TCMPCA64OuterEvaluationError(
            "PCA64 selection differs from deterministic validation replay"
        )
    if len(winners) != 5 or audited_records != 10:
        raise TCMPCA64OuterEvaluationError("preflight did not audit five two-arm folds")
    for root in validation_roots:
        try:
            output_root.relative_to(root)
        except ValueError:
            continue
        raise TCMPCA64OuterEvaluationError(
            "outer output must not be inside either validation arm"
        )
    return EvaluationPreflight(
        selection_path=selection_path,
        selection_sha256=selection_sha,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        candidate_root=candidate_root,
        selection=selection,
        plan=plan,
        protocol_package=package,
        winners=tuple(winners),
        artifact_snapshots=_unique_snapshots(snapshots),
        validation_roots=tuple(sorted(validation_roots, key=str)),
        candidate_records_audited=audited_records,
    )


def _verify_preflight_snapshot(preflight: EvaluationPreflight) -> None:
    if _validate_sidecar(
        preflight.selection_path, label="PCA64 selection"
    ) != preflight.selection_sha256:
        raise TCMPCA64OuterEvaluationError("selection changed after replay")
    if _validate_sidecar(
        preflight.plan_path, label="PCA64 plan"
    ) != preflight.plan_sha256:
        raise TCMPCA64OuterEvaluationError("plan changed after replay")
    for snapshot in preflight.artifact_snapshots:
        if (
            not snapshot.path.is_file()
            or snapshot.path.stat().st_size != snapshot.bytes
            or sha256(snapshot.path) != snapshot.sha256
        ):
            raise TCMPCA64OuterEvaluationError(
                f"{snapshot.label} changed after validation replay"
            )
    for root in preflight.validation_roots:
        reject_outer_artifacts(root)


def _verify_winner_snapshot(preflight: EvaluationPreflight) -> None:
    _verify_preflight_snapshot(preflight)
    for winner in preflight.winners:
        for path, digest, label in (
            (winner.validation_path, winner.validation_sha256, "validation predictions"),
            (winner.herb_path, winner.herb_sha256, "herb embedding"),
            (winner.disease_path, winner.disease_sha256, "disease embedding"),
        ):
            if not path.is_file() or sha256(path) != digest:
                raise TCMPCA64OuterEvaluationError(
                    f"fold {winner.fold} {label} changed after replay"
                )


def _staging_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}.incomplete")


def _ledger_key(plan_sha256: str, selection_sha256: str) -> str:
    if not safety._is_sha256(plan_sha256) or not safety._is_sha256(selection_sha256):
        raise TCMPCA64OuterEvaluationError("outer-access ledger identity is invalid")
    return hashlib.sha256(
        f"{plan_sha256}\0{selection_sha256}".encode("ascii")
    ).hexdigest()


def _access_path(plan_sha256: str, selection_sha256: str) -> Path:
    return GLOBAL_LEDGER_ROOT.resolve() / f"{_ledger_key(plan_sha256, selection_sha256)}.json"


def _assert_output_available(
    output_root: Path, *, access_path: Path | None = None
) -> None:
    checks: list[tuple[Path, str]] = [
        (output_root, "outer output"),
        (_staging_path(output_root), "interrupted outer staging directory"),
    ]
    if access_path is not None:
        checks.append((access_path, "global outer-access ledger receipt"))
    for path, label in checks:
        if path.exists() or path.is_symlink():
            raise TCMPCA64OuterEvaluationError(f"refusing existing {label}: {path}")


def _reserve_outer_access(
    preflight: EvaluationPreflight, output_root: Path
) -> tuple[Path, Path]:
    access_path = _access_path(preflight.plan_sha256, preflight.selection_sha256)
    _assert_output_available(output_root, access_path=access_path)
    safety._ensure_directory(output_root.parent)
    staging = _staging_path(output_root)
    evaluator_path = Path(__file__).resolve()
    # The O_EXCL ledger is intentionally the first mutation.  Its directory
    # entry is fsynced by _exclusive_json before any protocol array is opened.
    safety._exclusive_json(
        access_path,
        {
            "schema": ACCESS_SCHEMA,
            "state": "plan-and-selection-content-burned-before-outer-access",
            "ledger_key": _ledger_key(
                preflight.plan_sha256, preflight.selection_sha256
            ),
            "selection": str(preflight.selection_path),
            "selection_sha256": preflight.selection_sha256,
            "plan": str(preflight.plan_path),
            "plan_sha256": preflight.plan_sha256,
            "output_root": str(output_root),
            "staging_root": str(staging),
            "validation_artifacts_audited": preflight.candidate_records_audited,
            "fold_reservations_expected": 5,
            "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
        },
    )
    safety._durable_mkdir(staging)
    markers: list[dict[str, Any]] = []
    for winner in preflight.winners:
        marker_path = staging / f"fold_{winner.fold}.reserved.json"
        safety._exclusive_json(
            marker_path,
            {
                "schema": ACCESS_SCHEMA,
                "state": "reserved-before-outer-access",
                "selection_sha256": preflight.selection_sha256,
                "output_root": str(output_root),
                "fold": winner.fold,
                "winner": winner.candidate_id,
                "feature_basis": winner.feature_basis,
                "threshold": winner.threshold,
            },
        )
        markers.append(
            {
                "path": str(output_root / marker_path.name),
                "sha256": sha256(marker_path),
            }
        )
    safety._atomic_json(
        staging / "preflight.json",
        {
            "schema": PREFLIGHT_SCHEMA,
            "state": "all-five-folds-preflight-complete",
            "selection": str(preflight.selection_path),
            "selection_sha256": preflight.selection_sha256,
            "plan": str(preflight.plan_path),
            "plan_sha256": preflight.plan_sha256,
            "protocol_manifest": str(preflight.protocol_package.manifest_path),
            "protocol_manifest_sha256": preflight.protocol_package.manifest_sha256,
            "outer_rows_accessed": False,
            "output_root": str(output_root),
            "global_access_ledger": str(access_path),
            "ledger_key": _ledger_key(
                preflight.plan_sha256, preflight.selection_sha256
            ),
            "validation_artifacts_audited": preflight.candidate_records_audited,
            "folds_preflighted": 5,
            "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
            "reservations": markers,
            "winners": {
                str(winner.fold): {
                    "candidate": winner.candidate_id,
                    "feature_basis": winner.feature_basis,
                    "adapter": winner.adapter,
                    "validation_threshold": winner.threshold,
                    "validation_predictions_sha256": winner.validation_sha256,
                    "herb_embedding_sha256": winner.herb_sha256,
                    "disease_embedding_sha256": winner.disease_sha256,
                    "validation_identity": dict(winner.validation_identity),
                }
                for winner in preflight.winners
            },
        },
    )
    return staging, access_path


def _evaluate_reserved(
    *,
    preflight: EvaluationPreflight,
    output_root: Path,
    staging: Path,
    access_path: Path,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    test_evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    _verify_winner_snapshot(preflight)
    protocol = protocol_loader(preflight.protocol_package.manifest_path)
    safety._compare_loaded_protocol(protocol, preflight.protocol_package)
    protocol_snapshots = safety._capture_protocol_snapshot(
        preflight.protocol_package
    )
    audited_folds = safety._audit_loaded_folds(
        protocol=protocol,
        preflight=preflight,
    )
    fold_metrics: list[dict[str, Any]] = []
    selected: dict[str, dict[str, str]] = {}
    score_calls = 0
    for audited in audited_folds:
        winner = audited.winner
        scores = safety._safe_embedding_scores(
            winner.herb, winner.disease, audited.pair
        )
        score_calls += 1
        metrics = safety._validated_test_metrics(
            labels=audited.label,
            scores=scores,
            threshold=winner.threshold,
            test_evaluator=test_evaluator,
        )
        if (
            not math.isclose(
                float(metrics.get("Operating_Threshold", math.nan)),
                winner.threshold,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or metrics.get("AUPRC_Definition") != AUPRC_DEFINITION
        ):
            raise TCMPCA64OuterEvaluationError(
                f"fold {winner.fold} did not apply its frozen threshold"
            )
        staged_fold = staging / f"fold_{winner.fold}"
        final_fold = output_root / f"fold_{winner.fold}"
        safety._durable_mkdir(staged_fold)
        prediction_path = save_predictions(
            staged_fold / "test_predictions.npz",
            row_id=audited.row_id,
            pair=audited.pair,
            label=audited.label,
            score=scores,
        )
        safety._fsync_file_and_parent(prediction_path)
        table = validate_prediction_coverage(
            prediction_path,
            expected_row_id=audited.row_id,
            expected_pair=audited.pair,
            expected_label=audited.label,
        )
        test_identity = safety._canonical_prediction_identity(table)
        record = {
            "Dataset": "tcm-suite",
            "Method": "ASTRA-TCM-semantic-basis-inner-selected",
            "Fold": winner.fold,
            "Seed": 42,
            "Diagnostic": True,
            "Formal": False,
            "Formal_Eligible": False,
            "Seed42_Already_Observed": True,
            "Formal_Ineligibility_Reasons": [
                "seed-42 outer folds were observed before this development stage"
            ],
            "Selected_Candidate": winner.candidate_id,
            "Selected_Feature_Basis": winner.feature_basis,
            "Selected_Adapter": winner.adapter,
            "Selected_Epochs": winner.epochs,
            "Selected_Learning_Rate": winner.learning_rate,
            "Selected_Validation_AUPRC": winner.validation_auprc,
            "Selected_Validation_AUROC": winner.validation_auroc,
            "Selected_Validation_Threshold": winner.threshold,
            "Selection_Manifest": str(preflight.selection_path),
            "Selection_Manifest_SHA256": preflight.selection_sha256,
            "Validation_Predictions": str(winner.validation_path),
            "Validation_Predictions_SHA256": winner.validation_sha256,
            "Winner_Herb_Embedding": str(winner.herb_path),
            "Winner_Herb_Embedding_SHA256": winner.herb_sha256,
            "Winner_Disease_Embedding": str(winner.disease_path),
            "Winner_Disease_Embedding_SHA256": winner.disease_sha256,
            "Test_Predictions": str(final_fold / "test_predictions.npz"),
            "Test_Predictions_SHA256": sha256(prediction_path),
            "Test_Row_ID_SHA256": test_identity["row_id_sha256"],
            "Test_Pair_SHA256": test_identity["pair_sha256"],
            "Test_Label_SHA256": test_identity["label_sha256"],
            "Test_Score_SHA256": test_identity["score_sha256"],
            "Test_Array_Hash_Schema": ARRAY_HASH_SCHEMA,
            "Test_Models_Evaluated": 1,
            "Unselected_Test_Models_Evaluated": 0,
            **metrics,
        }
        safety._atomic_json(staged_fold / "metrics.json", record)
        safety._atomic_json(
            staged_fold / "selection_binding.json",
            {
                "schema": "astra-tcm-pca64-control-outer-fold-binding/v1",
                "selection": str(preflight.selection_path),
                "selection_sha256": preflight.selection_sha256,
                "plan": str(preflight.plan_path),
                "plan_sha256": preflight.plan_sha256,
                "fold": winner.fold,
                "winner": winner.candidate_id,
                "feature_basis": winner.feature_basis,
                "adapter": winner.adapter,
                "validation_threshold": winner.threshold,
                "threshold_source": "frozen selected-candidate validation maximum F1",
                "winner_embeddings_only": True,
                "test_models_evaluated": 1,
                "unselected_test_models_evaluated": 0,
                "test_predictions_sha256": sha256(prediction_path),
                "array_hash_schema": ARRAY_HASH_SCHEMA,
                "test_identity": test_identity,
            },
        )
        fold_metrics.append(record)
        selected[str(winner.fold)] = {
            "candidate": winner.candidate_id,
            "feature_basis": winner.feature_basis,
        }
    if score_calls != 5:
        raise TCMPCA64OuterEvaluationError(
            f"outer scoring count is {score_calls}, expected 5"
        )
    _verify_winner_snapshot(preflight)
    evaluator_path = Path(__file__).resolve()
    payload = {
        "schema": EVALUATION_SCHEMA,
        "status": "diagnostic-complete",
        "dataset": "tcm-suite",
        "method": "ASTRA-TCM-semantic-basis-inner-selected",
        "seed": 42,
        "Diagnostic": True,
        "Formal": False,
        "Formal_Eligible": False,
        "Seed42_Already_Observed": True,
        "formal_ineligibility_reasons": [
            "seed-42 outer folds were observed before this development stage"
        ],
        "selection_manifest": str(preflight.selection_path),
        "selection_manifest_sha256": preflight.selection_sha256,
        "plan": str(preflight.plan_path),
        "plan_sha256": preflight.plan_sha256,
        "protocol_manifest": str(preflight.protocol_package.manifest_path),
        "protocol_manifest_sha256": preflight.protocol_package.manifest_sha256,
        "global_access_ledger": str(access_path),
        "ledger_key": _ledger_key(preflight.plan_sha256, preflight.selection_sha256),
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
        "validation_artifacts_preflighted": preflight.candidate_records_audited,
        "folds_preflighted_before_outer_access": 5,
        "outer_score_calls": score_calls,
        "test_models_evaluated": 5,
        "unselected_test_models_evaluated": 0,
        "selected_candidates": selected,
        "fold_metrics": fold_metrics,
        "summary": safety._summary(fold_metrics),
        "interpretation": (
            "Seed-42 same-split post-baseline development diagnostic. Feature "
            "basis, adapter, epoch, learning rate, and threshold were frozen from "
            "inner validation before this single outer access; seed-42 outer folds "
            "had already been observed, so this is not a formal confirmation."
        ),
    }
    summary_path = staging / "summary.json"
    safety._atomic_json(summary_path, payload)
    safety._atomic_json(
        staging / "completion_receipt.json",
        {
            "schema": COMPLETION_SCHEMA,
            "state": "complete-before-atomic-publish",
            "selection_sha256": preflight.selection_sha256,
            "output_root": str(output_root),
            "summary": str(output_root / "summary.json"),
            "summary_sha256": sha256(summary_path),
            "folds": 5,
            "outer_score_calls": score_calls,
            "Diagnostic": True,
            "Formal": False,
            "Seed42_Already_Observed": True,
        },
    )
    _verify_preflight_snapshot(preflight)
    safety._verify_protocol_snapshot(protocol_snapshots)
    if output_root.exists() or output_root.is_symlink():
        raise TCMPCA64OuterEvaluationError(
            f"outer output appeared during evaluation: {output_root}"
        )
    safety._fsync_directory(staging)
    os.rename(staging, output_root)
    safety._fsync_directory(output_root.parent)
    return payload


def _evaluate_selected_with(
    *,
    selection_path: Path,
    output_root: Path,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    test_evaluator: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output_root = output_root.resolve()
    _assert_output_available(output_root)
    preflight = _replay_selection(selection_path, output_root=output_root)
    access_path = _access_path(preflight.plan_sha256, preflight.selection_sha256)
    _assert_output_available(output_root, access_path=access_path)
    _verify_preflight_snapshot(preflight)
    _assert_output_available(output_root, access_path=access_path)
    staging, access_path = _reserve_outer_access(preflight, output_root)
    return _evaluate_reserved(
        preflight=preflight,
        output_root=output_root,
        staging=staging,
        access_path=access_path,
        protocol_loader=protocol_loader,
        test_evaluator=test_evaluator,
    )


def evaluate_selected(*, selection_path: Path, output_root: Path) -> dict[str, Any]:
    """Evaluate the five final validation-selected winners exactly once."""

    return _evaluate_selected_with(
        selection_path=selection_path,
        output_root=output_root,
        protocol_loader=load_protocol_manifest,
        test_evaluator=evaluate_test_scores,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = evaluate_selected(
            selection_path=args.selection,
            output_root=args.output_root,
        )
    except Exception as error:
        print(f"TCM PCA64 final outer evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
