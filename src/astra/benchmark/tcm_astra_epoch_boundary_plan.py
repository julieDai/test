#!/usr/bin/env python3
"""Freeze a per-outer-fold TCM ASTRA epoch-boundary extension plan.

The parent stage searches ``{16, 32} x {0.003, 0.005}`` using inner
validation only.  This module deterministically replays that frozen parent
selection, then schedules both 64-epoch learning-rate arms *only* for an
outer fold whose best 32-epoch arm beat its best 16-epoch arm.  It never
loads or evaluates an outer-test split.

The extension is deliberately a new, immutable v2 artifact.  The parent v1
plan and selection are referenced by path, byte count, and SHA-256 rather
than being edited in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import AUPRC_DEFINITION
from .tcm_astra_hparam_selection import (
    ARRAY_HASH_SCHEMA,
    EXPECTED_SELECTION_METRIC,
    _object_sha256,
    _validated_selection_replay,
)
from .backend_paths import TCM_BACKEND_ROOT


PLAN_SCHEMA = "astra-tcm-epoch-boundary-plan/v2"
PLAN_STATUS = "frozen-before-epoch64-validation"
EPSILON = 1e-4
NUMERIC_TOLERANCE = 1e-12
EXPECTED_EPOCHS = (16, 32)
EXPECTED_LEARNING_RATES = (0.003, 0.005)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"

SOURCE_PATHS = {
    "plan_generator": Path(__file__).resolve(),
    "boundary_runner": SOURCE_ROOT
    / "astra/benchmark/tcm_astra_epoch_boundary_runner.py",
    "boundary_decision": SOURCE_ROOT
    / "astra/benchmark/tcm_astra_epoch_boundary_decision.py",
    "parent_selector": SOURCE_ROOT
    / "astra/benchmark/tcm_astra_hparam_selection.py",
    "parent_runner": SOURCE_ROOT / "astra/benchmark/tcm_astra_hparam_runner.py",
    "central_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
    "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
    "protocol": SOURCE_ROOT / "astra/benchmark/protocol.py",
    "configuration": TCM_BACKEND_ROOT / "ASTRA.conf",
    "model": TCM_BACKEND_ROOT / "HGHDA.py",
    "herb_recommender": TCM_BACKEND_ROOT / "base/herbRecommender.py",
    "rating": TCM_BACKEND_ROOT / "rating.py",
}

TIE_BREAK = [
    "higher Validation_AUPRC",
    "higher Validation_AUROC",
    "lower learning rate",
    "lexicographically smaller candidate id",
]


class TCMEpochBoundaryPlanError(RuntimeError):
    """Raised when the sequential epoch plan cannot be frozen safely."""


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
        raise TCMEpochBoundaryPlanError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMEpochBoundaryPlanError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise TCMEpochBoundaryPlanError(f"artifact is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _verify_artifact(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise TCMEpochBoundaryPlanError(f"{label} artifact record is invalid")
    try:
        path = Path(str(record["path"])).resolve(strict=True)
    except OSError as error:
        raise TCMEpochBoundaryPlanError(f"{label} artifact is absent") from error
    if not path.is_file():
        raise TCMEpochBoundaryPlanError(f"{label} artifact is not a file")
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256(
        path
    ):
        raise TCMEpochBoundaryPlanError(f"{label} artifact changed after freeze")
    return path


def _source_freeze() -> dict[str, dict[str, object]]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise TCMEpochBoundaryPlanError(
            "epoch-boundary implementation is incomplete: " + ", ".join(missing)
        )
    return {name: artifact(path) for name, path in SOURCE_PATHS.items()}


def _candidate_rank(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    try:
        auprc = float(record["validation_auprc"])
        auroc = float(record["validation_auroc"])
        learning_rate = float(record["learning_rate"])
        candidate_id = str(record["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMEpochBoundaryPlanError(
            "parent candidate lacks a valid validation ranking field"
        ) from error
    if not all(math.isfinite(value) for value in (auprc, auroc, learning_rate)):
        raise TCMEpochBoundaryPlanError("parent candidate ranking is non-finite")
    return (-auprc, -auroc, learning_rate, candidate_id)


def _best_for_epoch(
    records: Sequence[Mapping[str, Any]], *, epoch: int
) -> Mapping[str, Any]:
    matches = [record for record in records if record.get("epochs") == epoch]
    if len(matches) != len(EXPECTED_LEARNING_RATES):
        raise TCMEpochBoundaryPlanError(
            f"parent fold has {len(matches)} candidates at epoch {epoch}, expected 2"
        )
    observed_rates = sorted(float(record.get("learning_rate", -1.0)) for record in matches)
    if observed_rates != list(EXPECTED_LEARNING_RATES):
        raise TCMEpochBoundaryPlanError(
            f"parent epoch-{epoch} learning-rate grid differs: {observed_rates}"
        )
    return min(matches, key=_candidate_rank)


def _fold_plan(fold: int, fold_record: Mapping[str, Any]) -> dict[str, Any]:
    records = fold_record.get("candidates")
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise TCMEpochBoundaryPlanError(
            f"parent selection fold {fold} candidate records are invalid"
        )
    if len(records) != len(EXPECTED_EPOCHS) * len(EXPECTED_LEARNING_RATES):
        raise TCMEpochBoundaryPlanError(
            f"parent selection fold {fold} does not contain the complete 2x2 grid"
        )
    if any(record.get("semantic_adapter") != "identity" for record in records):
        raise TCMEpochBoundaryPlanError(
            f"parent selection fold {fold} contains a non-identity adapter"
        )
    best16 = dict(_best_for_epoch(records, epoch=16))
    best32 = dict(_best_for_epoch(records, epoch=32))
    winner = fold_record.get("winner")
    if not isinstance(winner, Mapping):
        raise TCMEpochBoundaryPlanError(f"parent selection fold {fold} has no winner")
    replayed_winner = min(records, key=lambda record: (
        -float(record["validation_auprc"]),
        -float(record["validation_auroc"]),
        int(record["epochs"]),
        float(record["learning_rate"]),
        str(record["id"]),
    ))
    if dict(winner) != dict(replayed_winner):
        raise TCMEpochBoundaryPlanError(
            f"parent selection fold {fold} winner does not replay"
        )

    delta = float(best32["validation_auprc"]) - float(
        best16["validation_auprc"]
    )
    trigger = delta > 0.0
    candidates = (
        [
            {
                "id": "identity_e64_lr3e3",
                "semantic_adapter": "identity",
                "epochs": 64,
                "learning_rate": 0.003,
            },
            {
                "id": "identity_e64_lr5e3",
                "semantic_adapter": "identity",
                "epochs": 64,
                "learning_rate": 0.005,
            },
        ]
        if trigger
        else []
    )
    return {
        "fold": fold,
        "stage1_fold_sha256": _object_sha256(dict(fold_record)),
        "stage1_winner": dict(winner),
        "stage1_winner_sha256": _object_sha256(dict(winner)),
        "best_epoch16": best16,
        "best_epoch16_sha256": _object_sha256(best16),
        "best_epoch32": best32,
        "best_epoch32_sha256": _object_sha256(best32),
        "delta_epoch32_minus_epoch16": delta,
        "trigger_epoch64": trigger,
        "trigger_rule": "best_epoch32_validation_auprc > best_epoch16_validation_auprc",
        "epoch64_candidates": candidates,
    }


def _input_freeze(protocol_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = protocol_manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TCMEpochBoundaryPlanError("protocol manifest source files are absent")
    frozen: dict[str, Any] = {}
    for name, raw in files.items():
        if not isinstance(raw, Mapping):
            raise TCMEpochBoundaryPlanError(f"protocol input {name} is invalid")
        path = Path(str(raw.get("path", ""))).resolve(strict=True)
        current = artifact(path)
        if (
            current["sha256"] != raw.get("sha256")
            or current["bytes"] != raw.get("bytes")
        ):
            raise TCMEpochBoundaryPlanError(
                f"protocol input {name} differs from its manifest"
            )
        frozen[str(name)] = current
    return frozen


def _feature_input_freeze(
    feature_root: Path, feature_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    entities = feature_manifest.get("entities")
    if not isinstance(entities, Mapping) or not entities:
        raise TCMEpochBoundaryPlanError("feature-bundle entity records are absent")
    frozen: dict[str, Any] = {}
    root = feature_root.resolve(strict=True)
    for entity, raw_entity in entities.items():
        if not isinstance(raw_entity, Mapping):
            raise TCMEpochBoundaryPlanError(f"feature entity {entity} is invalid")
        for component in ("matrix", "row_ids_sidecar"):
            raw = raw_entity.get(component)
            if not isinstance(raw, Mapping):
                raise TCMEpochBoundaryPlanError(
                    f"feature entity {entity} {component} is absent"
                )
            relative_path = Path(str(raw.get("relative_path", "")))
            path = (root / relative_path).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise TCMEpochBoundaryPlanError(
                    f"feature entity {entity} {component} escapes the bundle"
                ) from error
            current = artifact(path)
            if (
                current["sha256"] != raw.get("sha256")
                or current["bytes"] != raw.get("bytes")
            ):
                raise TCMEpochBoundaryPlanError(
                    f"feature entity {entity} {component} differs from manifest"
                )
            frozen[f"{entity}.{component}"] = current
    return frozen


def build_plan(stage1_selection_path: Path) -> dict[str, Any]:
    """Replay stage 1 and construct an immutable validation-only v2 plan."""

    selection_path = stage1_selection_path.resolve(strict=True)
    try:
        (
            selection,
            selection_sha,
            receipt_path,
            receipt_sha,
            context,
        ) = _validated_selection_replay(selection_path)
    except Exception as error:
        raise TCMEpochBoundaryPlanError(
            f"parent stage-1 selection replay failed: {error}"
        ) from error
    if selection_sha != sha256(selection_path) or receipt_sha != sha256(receipt_path):
        raise TCMEpochBoundaryPlanError("parent selection or receipt hash is unstable")
    if selection.get("outer_test_accessed") is not False or selection.get(
        "outer_test_scored"
    ) is not False:
        raise TCMEpochBoundaryPlanError("parent selection is not validation-only")

    selected_folds = selection.get("folds")
    expected_folds = {str(fold) for fold in range(1, context.protocol.n_folds + 1)}
    if not isinstance(selected_folds, Mapping) or set(selected_folds) != expected_folds:
        raise TCMEpochBoundaryPlanError("parent selection fold set differs")
    folds = {
        str(fold): _fold_plan(fold, selected_folds[str(fold)])
        for fold in range(1, context.protocol.n_folds + 1)
    }
    protocol_manifest = _load_json(context.protocol_path)
    input_freeze = _input_freeze(protocol_manifest)
    feature_manifest = _load_json(context.feature_manifest)
    feature_input_freeze = _feature_input_freeze(
        context.feature_bundle, feature_manifest
    )
    source_freeze = _source_freeze()
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "dataset": "tcm-suite",
        "seed": int(context.protocol.seed),
        "formal_scope": (
            "Seed-42 same-split post-baseline development diagnostic; a fresh "
            "previously unseen split is required for independent confirmation."
        ),
        "parent": {
            "selection": artifact(selection_path),
            "selection_receipt": artifact(receipt_path),
            "plan": artifact(context.path),
        },
        "protocol_manifest": artifact(context.protocol_path),
        "feature_bundle": {
            "path": str(context.feature_bundle.resolve()),
            "manifest": artifact(context.feature_manifest),
        },
        "input_freeze": input_freeze,
        "input_freeze_sha256": _object_sha256(input_freeze),
        "feature_input_freeze": feature_input_freeze,
        "feature_input_freeze_sha256": _object_sha256(feature_input_freeze),
        "source_freeze": source_freeze,
        "source_freeze_sha256": _object_sha256(source_freeze),
        "fixed": {
            "batch_size": 2000,
            "embedding_dimension": 64,
            "reg_u": 0.001,
            "text_fusion_alpha": 1.0,
            "semantic_adapter": "identity",
        },
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_metric_definition": AUPRC_DEFINITION,
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "epsilon": EPSILON,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "epoch64_trigger": (
            "per fold: best epoch-32 Validation_AUPRC strictly exceeds best "
            "epoch-16 Validation_AUPRC"
        ),
        "epoch128_trigger": (
            "per fold: best epoch-64 Validation_AUPRC exceeds best epoch-32 "
            "Validation_AUPRC by more than epsilon"
        ),
        "tie_break": TIE_BREAK,
        "selection_policy": (
            "Each outer fold is expanded and decided using only that fold's "
            "frozen inner-validation predictions; outer-test rows are forbidden."
        ),
        "folds": folds,
    }


def _validate_sidecar(plan_path: Path) -> str:
    plan_sha = sha256(plan_path)
    sidecar = plan_path.with_suffix(plan_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise TCMEpochBoundaryPlanError(f"plan SHA-256 sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != plan_sha or fields[1] != plan_path.name:
        raise TCMEpochBoundaryPlanError("plan SHA-256 sidecar does not match")
    return plan_sha


def validate_frozen_plan(plan_path: Path) -> tuple[Mapping[str, Any], str]:
    """Validate the immutable plan, source hashes, and all frozen inputs."""

    path = plan_path.resolve(strict=True)
    plan_sha = _validate_sidecar(path)
    plan = _load_json(path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise TCMEpochBoundaryPlanError("unsupported or unfrozen boundary plan")
    if plan.get("dataset") != "tcm-suite" or int(plan.get("seed", -1)) != 42:
        raise TCMEpochBoundaryPlanError("boundary plan dataset/seed differs")
    if (
        plan.get("selection_metric") != EXPECTED_SELECTION_METRIC
        or plan.get("selection_metric_definition") != AUPRC_DEFINITION
        or plan.get("array_hash_schema") != ARRAY_HASH_SCHEMA
        or plan.get("tie_break") != TIE_BREAK
        or not math.isclose(
            float(plan.get("epsilon", float("nan"))),
            EPSILON,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(plan.get("numeric_tolerance", float("nan"))),
            NUMERIC_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise TCMEpochBoundaryPlanError("boundary selection contract differs")
    expected_fixed = {
        "batch_size": 2000,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": 1.0,
        "semantic_adapter": "identity",
    }
    if plan.get("fixed") != expected_fixed:
        raise TCMEpochBoundaryPlanError("boundary fixed controls differ")

    source_freeze = plan.get("source_freeze")
    if not isinstance(source_freeze, Mapping) or set(source_freeze) != set(SOURCE_PATHS):
        raise TCMEpochBoundaryPlanError("boundary source-freeze set differs")
    if plan.get("source_freeze_sha256") != _object_sha256(dict(source_freeze)):
        raise TCMEpochBoundaryPlanError("boundary source-freeze digest differs")
    for name, expected_path in SOURCE_PATHS.items():
        actual = _verify_artifact(source_freeze[name], label=f"source {name}")
        if actual != expected_path.resolve():
            raise TCMEpochBoundaryPlanError(f"source {name} path differs")

    parent = plan.get("parent")
    if not isinstance(parent, Mapping) or set(parent) != {
        "selection",
        "selection_receipt",
        "plan",
    }:
        raise TCMEpochBoundaryPlanError("parent artifact bindings differ")
    selection_path = _verify_artifact(parent["selection"], label="parent selection")
    receipt_path = _verify_artifact(
        parent["selection_receipt"], label="parent selection receipt"
    )
    parent_plan_path = _verify_artifact(parent["plan"], label="parent plan")
    selection = _load_json(selection_path)
    if (
        selection.get("schema") != "astra-tcm-hparam-selection/v1"
        or selection.get("outer_test_accessed") is not False
        or selection.get("outer_test_scored") is not False
        or Path(str(selection.get("plan", ""))).resolve() != parent_plan_path
        or selection.get("plan_sha256") != parent["plan"]["sha256"]
        or Path(str(selection.get("selector_path", ""))).resolve()
        != SOURCE_PATHS["parent_selector"].resolve()
        or selection.get("selector_sha256")
        != source_freeze["parent_selector"]["sha256"]
    ):
        raise TCMEpochBoundaryPlanError("parent selection binding or flags differ")
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema") != "astra-tcm-hparam-selection-receipt/v1"
        or Path(str(receipt.get("selection", ""))).resolve() != selection_path
        or receipt.get("selection_sha256") != parent["selection"]["sha256"]
        or Path(str(receipt.get("plan", ""))).resolve() != parent_plan_path
        or receipt.get("plan_sha256") != parent["plan"]["sha256"]
        or Path(str(receipt.get("selector", ""))).resolve()
        != SOURCE_PATHS["parent_selector"].resolve()
        or receipt.get("selector_sha256")
        != source_freeze["parent_selector"]["sha256"]
        or receipt.get("outer_test_scored_before_receipt") is not False
    ):
        raise TCMEpochBoundaryPlanError("parent selection receipt binding differs")

    protocol_path = _verify_artifact(
        plan.get("protocol_manifest"), label="protocol manifest"
    )
    protocol_manifest = _load_json(protocol_path)
    feature_bundle = plan.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping) or set(feature_bundle) != {
        "path",
        "manifest",
    }:
        raise TCMEpochBoundaryPlanError("feature-bundle binding differs")
    feature_root = Path(str(feature_bundle["path"])).resolve(strict=True)
    feature_manifest = _verify_artifact(
        feature_bundle["manifest"], label="feature-bundle manifest"
    )
    if feature_manifest.parent != feature_root:
        raise TCMEpochBoundaryPlanError("feature-bundle root differs")
    expected_feature_inputs = _feature_input_freeze(
        feature_root, _load_json(feature_manifest)
    )
    if plan.get("feature_input_freeze") != expected_feature_inputs:
        raise TCMEpochBoundaryPlanError("feature input bindings differ")
    if plan.get("feature_input_freeze_sha256") != _object_sha256(
        expected_feature_inputs
    ):
        raise TCMEpochBoundaryPlanError("feature input digest differs")

    inputs = plan.get("input_freeze")
    manifest_files = protocol_manifest.get("files")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(manifest_files, Mapping)
        or set(inputs) != set(manifest_files)
    ):
        raise TCMEpochBoundaryPlanError("frozen input set differs from protocol")
    if plan.get("input_freeze_sha256") != _object_sha256(dict(inputs)):
        raise TCMEpochBoundaryPlanError("frozen input digest differs")
    for name, raw in manifest_files.items():
        path_for_input = _verify_artifact(inputs[name], label=f"input {name}")
        if not isinstance(raw, Mapping):
            raise TCMEpochBoundaryPlanError(f"protocol input {name} is invalid")
        if (
            path_for_input != Path(str(raw.get("path", ""))).resolve()
            or inputs[name].get("sha256") != raw.get("sha256")
            or inputs[name].get("bytes") != raw.get("bytes")
        ):
            raise TCMEpochBoundaryPlanError(
                f"frozen input {name} differs from protocol manifest"
            )

    folds = plan.get("folds")
    selection_folds = selection.get("folds")
    expected_fold_keys = {str(fold) for fold in range(1, 6)}
    if (
        not isinstance(folds, Mapping)
        or not isinstance(selection_folds, Mapping)
        or set(folds) != expected_fold_keys
        or set(selection_folds) != expected_fold_keys
    ):
        raise TCMEpochBoundaryPlanError("boundary or parent fold set differs")
    for fold_key in sorted(expected_fold_keys):
        fold_plan = folds[fold_key]
        parent_fold = selection_folds[fold_key]
        if not isinstance(fold_plan, Mapping) or not isinstance(parent_fold, Mapping):
            raise TCMEpochBoundaryPlanError(f"fold {fold_key} binding is invalid")
        if fold_plan.get("stage1_fold_sha256") != _object_sha256(dict(parent_fold)):
            raise TCMEpochBoundaryPlanError(
                f"fold {fold_key} parent selection binding differs"
            )
        candidates = parent_fold.get("candidates")
        if not isinstance(candidates, list):
            raise TCMEpochBoundaryPlanError(
                f"fold {fold_key} parent candidates are absent"
            )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TCMEpochBoundaryPlanError(
                    f"fold {fold_key} parent candidate is invalid"
                )
            for path_key, hash_key in (
                ("validation_predictions", "validation_predictions_sha256"),
                ("herb_embedding", "herb_embedding_sha256"),
                ("disease_embedding", "disease_embedding_sha256"),
                ("metrics", "metrics_sha256"),
                ("protocol", "protocol_sha256"),
            ):
                candidate_path = Path(str(candidate.get(path_key, ""))).resolve(
                    strict=True
                )
                if sha256(candidate_path) != candidate.get(hash_key):
                    raise TCMEpochBoundaryPlanError(
                        f"fold {fold_key} parent candidate {path_key} changed"
                    )
            prediction_path = Path(
                str(candidate["validation_predictions"])
            ).resolve()
            if (prediction_path.parent / "test_predictions.npz").exists():
                raise TCMEpochBoundaryPlanError(
                    f"fold {fold_key} parent candidate has an outer-test artifact"
                )
        replayed = _fold_plan(int(fold_key), parent_fold)
        if dict(fold_plan) != replayed:
            raise TCMEpochBoundaryPlanError(
                f"fold {fold_key} boundary plan does not replay"
            )
    return plan, plan_sha


def freeze_plan(*, stage1_selection: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMEpochBoundaryPlanError(
            f"refusing to overwrite plan or sidecar: {output}, {sidecar}"
        )
    payload = build_plan(stage1_selection)
    _write_json(output, payload)
    plan_sha = sha256(output)
    sidecar.write_text(f"{plan_sha}  {output.name}\n", encoding="utf-8")
    return {**payload, "plan_path": str(output), "plan_sha256": plan_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_plan(
            stage1_selection=args.stage1_selection,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM epoch-boundary plan freeze failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
