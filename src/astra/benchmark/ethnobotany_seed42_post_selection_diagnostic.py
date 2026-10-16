#!/usr/bin/env python3
"""One-shot seed-42 post-selection diagnostic for the promoted adapter.

This command deliberately does *not* create a manuscript-unbiased result.
The residual-MLP architecture and its per-fold epochs were selected using the
same seed-42 protocol.  ``freeze-selection`` binds that amended development
decision, replays only its inner-validation artifacts, and freezes an F1
threshold for each fold.  ``score`` then reserves all five outer-fold outputs
before the first outer row is requested and permits no retry or fold subset.

The frozen development selectors, candidate runner, and model are imported or
read only.  This module never modifies them and exposes no training operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .ethnobotany_refined_runner import derive_training_validation_rows
from .metrics import (
    BenchmarkAuditError,
    PREDICTION_FIELDS,
    evaluate_test_scores,
    evaluate_validation_scores,
    load_predictions,
    save_predictions,
    select_f1_threshold,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .backend_paths import ETHNOBOTANY_BACKEND_ROOT


SELECTION_SCHEMA = "astra-ethnobotany-seed42-post-selection-diagnostic-selection/v1"
SUMMARY_SCHEMA = "astra-ethnobotany-seed42-post-selection-diagnostic-summary/v1"
DEVELOPMENT_SCHEMA = "astra-ethnobotany-adapter-development-selection-amended/v1"
CANDIDATE_SCHEMA = "astra-ethnobotany-refined-candidate/v1"
DATASET = "ethnobotany"
SEED = 42
MODEL = "residual_mlp"
FOLDS = (1, 2, 3, 4, 5)
METRIC_ATOL = 1e-12
SCORE_ATOL = 1e-12
HASH_SUFFIX = ".sha256"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parent
SELECTOR_PATH = Path(__file__).resolve()
METRICS_PATH = SELECTOR_PATH.with_name("metrics.py")
PROTOCOL_PATH = SELECTOR_PATH.with_name("protocol.py")
REFINED_RUNNER_PATH = SELECTOR_PATH.with_name("ethnobotany_refined_runner.py")
FEATURE_RECEIPT_VALIDATOR_PATH = SELECTOR_PATH.with_name(
    "ethnobotany_refined_feature_receipt.py"
)
EXPECTED_CODE_PATHS = {
    "model": ETHNOBOTANY_BACKEND_ROOT / "ASTRASemanticRefined.py",
    "runner": SELECTOR_PATH.with_name("ethnobotany_semantic_adapter_runner.py"),
    "parent_model": ETHNOBOTANY_BACKEND_ROOT / "ASTRARefined.py",
    "parent_runner": REFINED_RUNNER_PATH,
    "selector": SELECTOR_PATH.with_name("ethnobotany_adapter_selection.py"),
}
AMENDMENT_SELECTOR_PATH = SELECTOR_PATH.with_name(
    "ethnobotany_adapter_selection_amendment.py"
)


class Seed42DiagnosticError(RuntimeError):
    """Raised when the post-selection diagnostic cannot be proven safe."""


def sha256(path: Path | str) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Seed42DiagnosticError(f"artifact is absent or a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path | str, *, label: str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Seed42DiagnosticError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Seed42DiagnosticError(f"{label} must contain one JSON object: {path}")
    return payload


def _validate_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise Seed42DiagnosticError(f"{label} sha256 must be 64 lowercase hex digits")
    return value


def _resolve(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Seed42DiagnosticError(f"{label} path must be a non-empty string")
    supplied = Path(value)
    return (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()


def _record(path: Path | str) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved)}


def _artifact(
    record: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise Seed42DiagnosticError(f"{label} must contain exactly path and sha256")
    path = _resolve(record["path"], label=label)
    if expected_path is not None and path != expected_path.resolve():
        raise Seed42DiagnosticError(f"{label} has an unexpected path: {path}")
    expected = _validate_hash(record["sha256"], label=label)
    observed = sha256(path)
    if observed != expected:
        raise Seed42DiagnosticError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": expected}


def _sidecar(path: Path) -> Path:
    return Path(str(path) + HASH_SUFFIX)


def _read_frozen(path: Path, *, label: str) -> tuple[str, dict[str, str]]:
    digest = sha256(path)
    sidecar = _sidecar(path)
    if sidecar.is_symlink() or not sidecar.is_file():
        raise Seed42DiagnosticError(f"{label} sidecar is absent: {sidecar}")
    try:
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise Seed42DiagnosticError(f"cannot read {label} sidecar: {error}") from error
    _validate_hash(declared, label=f"{label} sidecar")
    if declared != digest:
        raise Seed42DiagnosticError(
            f"{label} changed after freeze: expected {declared}, observed {digest}"
        )
    if path.stat().st_mtime_ns > sidecar.stat().st_mtime_ns:
        raise Seed42DiagnosticError(f"{label} is newer than its sidecar")
    return digest, _record(sidecar)


def _atomic_text(path: Path, content: str) -> None:
    path = path.resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise Seed42DiagnosticError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Seed42DiagnosticError(
                f"refusing to overwrite frozen artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _candidate_artifact(
    record: object, *, root: Path, label: str, expected_hash: str | None = None
) -> dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise Seed42DiagnosticError(f"{label} must contain exactly path and sha256")
    raw = record["path"]
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise Seed42DiagnosticError(f"{label} must be candidate-relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Seed42DiagnosticError(f"{label} escapes candidate directory") from error
    artifact = _artifact(
        {"path": str(path), "sha256": record["sha256"]}, label=label
    )
    if expected_hash is not None and artifact["sha256"] != expected_hash:
        raise Seed42DiagnosticError(f"{label} differs from amended development selection")
    return artifact


def _load_embedding(record: Mapping[str, str], *, label: str) -> np.ndarray:
    try:
        values = np.load(record["path"], allow_pickle=False)
    except (OSError, ValueError) as error:
        raise Seed42DiagnosticError(f"cannot load {label}: {error}") from error
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise Seed42DiagnosticError(f"{label} must be a numeric matrix")
    if not np.isfinite(values).all():
        raise Seed42DiagnosticError(f"{label} contains NaN/Inf")
    return np.asarray(values)


def _score_embeddings(
    u_embedding: np.ndarray, i_embedding: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    u = np.asarray(u_embedding, dtype=np.float64)
    i = np.asarray(i_embedding, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if u.ndim != 2 or i.ndim != 2 or u.shape[1] != i.shape[1]:
        raise Seed42DiagnosticError("checkpoint u/i embeddings are incompatible")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise Seed42DiagnosticError("evaluation pairs must have shape (n, 2)")
    if int(pairs.min()) < 0:
        raise Seed42DiagnosticError("evaluation pairs contain negative indices")
    if int(pairs[:, 0].max()) >= len(u) or int(pairs[:, 1].max()) >= len(i):
        raise Seed42DiagnosticError("evaluation pair exceeds embedding rows")
    logits = np.einsum("ij,ij->i", u[pairs[:, 0]], i[pairs[:, 1]])
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    if not np.isfinite(scores).all():
        raise Seed42DiagnosticError("checkpoint emitted non-finite scores")
    return scores


def _validate_development_selection(
    path: Path,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    digest, sidecar_record = _read_frozen(path, label="amended development selection")
    payload = _load_json(path, label="amended development selection")
    required = {
        "schema": DEVELOPMENT_SCHEMA,
        "dataset": DATASET,
        "development_only": True,
        "validation_only": True,
        "outer_evaluation_permitted": False,
        "outer_score_command_available": False,
        "selected_model": MODEL,
        "best_new_architecture": MODEL,
        "promoted": True,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise Seed42DiagnosticError(
                f"development selection {field} must be {expected!r}"
            )

    protocol_record = _artifact(payload.get("protocol"), label="seed42 protocol")
    protocol = protocol_loader(Path(protocol_record["path"]))
    if (
        protocol.dataset != DATASET
        or int(protocol.seed) != SEED
        or int(protocol.n_folds) != len(FOLDS)
    ):
        raise Seed42DiagnosticError("development protocol is not Ethnobotany seed42/5-fold")

    amendment = payload.get("amendment")
    if not isinstance(amendment, Mapping):
        raise Seed42DiagnosticError("development selection amendment provenance is absent")
    amendment_path = _resolve(amendment.get("path"), label="amendment")
    amendment_record = _artifact(
        {"path": str(amendment_path), "sha256": amendment.get("sha256")},
        label="frozen amendment",
    )
    amendment_sidecar_path = _resolve(
        amendment.get("sidecar"), label="amendment sidecar"
    )
    amendment_sidecar = _artifact(
        {
            "path": str(amendment_sidecar_path),
            "sha256": amendment.get("sidecar_sha256"),
        },
        label="frozen amendment sidecar",
    )
    if amendment_sidecar_path != _sidecar(amendment_path):
        raise Seed42DiagnosticError("amendment sidecar path is inconsistent")
    if amendment_sidecar_path.read_text(encoding="ascii").strip() != amendment_record[
        "sha256"
    ]:
        raise Seed42DiagnosticError("amendment sidecar does not freeze the amendment")

    receipt_contracts = amendment.get("receipt_contracts")
    if not isinstance(receipt_contracts, Mapping) or set(receipt_contracts) != {
        "baseline",
        "s5",
    }:
        raise Seed42DiagnosticError("amendment receipt contracts are incomplete")
    feature_receipt = _artifact(
        receipt_contracts["s5"], label="S5 feature deep-audit receipt"
    )

    selectors = amendment.get("selectors")
    if not isinstance(selectors, Mapping) or set(selectors) != {"original", "amendment"}:
        raise Seed42DiagnosticError("amendment selector bindings are incomplete")
    original_selector = _artifact(
        selectors["original"],
        label="original development selector",
        expected_path=EXPECTED_CODE_PATHS["selector"],
    )
    amendment_selector = _artifact(
        selectors["amendment"],
        label="amendment selector",
        expected_path=AMENDMENT_SELECTOR_PATH,
    )

    code_contracts = payload.get("code_contracts")
    development_code = (
        code_contracts.get("development") if isinstance(code_contracts, Mapping) else None
    )
    if not isinstance(development_code, Mapping) or development_code.get("kind") != "files":
        raise Seed42DiagnosticError("development code contract is absent")
    code: dict[str, dict[str, str]] = {}
    for role, expected_path in EXPECTED_CODE_PATHS.items():
        code[role] = _artifact(
            development_code.get(role),
            label=f"development {role}",
            expected_path=expected_path,
        )
    if code["selector"] != original_selector:
        raise Seed42DiagnosticError("development selector bindings disagree")

    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        raise Seed42DiagnosticError("development plan provenance is absent")
    plan_record = _artifact(
        {"path": plan.get("path"), "sha256": plan.get("sha256")},
        label="development plan",
    )
    plan_sidecar = _artifact(
        {"path": plan.get("sidecar"), "sha256": plan.get("sidecar_sha256")},
        label="development plan sidecar",
    )
    if Path(plan_sidecar["path"]) != _sidecar(Path(plan_record["path"])):
        raise Seed42DiagnosticError("development plan sidecar path is inconsistent")
    if Path(plan_sidecar["path"]).read_text(encoding="ascii").strip() != plan_record[
        "sha256"
    ]:
        raise Seed42DiagnosticError("development plan sidecar does not freeze the plan")

    architecture = payload.get("architectures", {}).get(MODEL)
    folds = architecture.get("folds") if isinstance(architecture, Mapping) else None
    if not isinstance(folds, list) or len(folds) != len(FOLDS):
        raise Seed42DiagnosticError("residual_mlp development folds are incomplete")
    by_fold: dict[int, Mapping[str, Any]] = {}
    for record in folds:
        if not isinstance(record, Mapping) or isinstance(record.get("fold"), bool):
            raise Seed42DiagnosticError("residual_mlp fold record is invalid")
        fold = int(record["fold"])
        if fold in by_fold:
            raise Seed42DiagnosticError(f"duplicate residual_mlp fold: {fold}")
        by_fold[fold] = record
    if set(by_fold) != set(FOLDS):
        raise Seed42DiagnosticError("residual_mlp fold identities differ from 1..5")

    amendment_candidates = amendment.get("s5_candidate_manifests")
    if not isinstance(amendment_candidates, list):
        raise Seed42DiagnosticError("amendment candidate manifest bindings are absent")
    amendment_candidate_set = {
        (str(Path(str(item.get("path"))).resolve()), item.get("sha256"))
        for item in amendment_candidates
        if isinstance(item, Mapping)
    }
    if len(amendment_candidate_set) != 15:
        raise Seed42DiagnosticError("amendment must bind exactly 15 S5 candidate manifests")

    return {
        "path": path,
        "record": {"path": str(path), "sha256": digest},
        "sidecar": sidecar_record,
        "payload": payload,
        "protocol": protocol,
        "protocol_record": protocol_record,
        "feature_receipt": feature_receipt,
        "amendment": amendment_record,
        "amendment_sidecar": amendment_sidecar,
        "development_plan": plan_record,
        "development_plan_sidecar": plan_sidecar,
        "code": code,
        "original_selector": original_selector,
        "amendment_selector": amendment_selector,
        "folds": by_fold,
        "amendment_candidate_set": amendment_candidate_set,
    }


def _audit_selected_fold(context: Mapping[str, Any], fold: int) -> dict[str, Any]:
    development = context["folds"][fold]
    manifest = _artifact(
        {
            "path": development.get("manifest_path"),
            "sha256": development.get("manifest_sha256"),
        },
        label=f"fold {fold} candidate manifest",
    )
    if (manifest["path"], manifest["sha256"]) not in context[
        "amendment_candidate_set"
    ]:
        raise Seed42DiagnosticError(
            f"fold {fold} candidate is not frozen by the outcome-blind amendment"
        )
    config = _artifact(
        {
            "path": development.get("config_path"),
            "sha256": development.get("config_sha256"),
        },
        label=f"fold {fold} config",
    )
    candidate_path = Path(manifest["path"])
    payload = _load_json(candidate_path, label=f"fold {fold} candidate")
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise Seed42DiagnosticError(f"fold {fold} candidate schema is invalid")
    if payload.get("validation_only") is not True:
        raise Seed42DiagnosticError(f"fold {fold} candidate is not validation-only")
    if payload.get("outer_evaluation_rows_loaded") is not False:
        raise Seed42DiagnosticError(f"fold {fold} candidate loaded outer rows")
    if payload.get("fold") != fold:
        raise Seed42DiagnosticError(f"fold {fold} candidate identity differs")
    identity = payload.get("config")
    if (
        not isinstance(identity, Mapping)
        or identity.get("id") != development.get("config_id")
        or identity.get("candidate_value") != MODEL
    ):
        raise Seed42DiagnosticError(f"fold {fold} candidate config identity differs")
    source = payload.get("config_source")
    if (
        not isinstance(source, Mapping)
        or Path(str(source.get("path", ""))).resolve() != Path(config["path"])
        or source.get("sha256") != config["sha256"]
    ):
        raise Seed42DiagnosticError(f"fold {fold} candidate config binding differs")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or Path(str(protocol.get("path", ""))).resolve()
        != Path(context["protocol_record"]["path"])
        or protocol.get("sha256") != context["protocol_record"]["sha256"]
        or protocol.get("seed") != SEED
    ):
        raise Seed42DiagnosticError(f"fold {fold} candidate protocol differs")
    features = payload.get("features")
    receipt = features.get("feature_deep_audit_receipt") if isinstance(features, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or Path(str(receipt.get("path", ""))).resolve()
        != Path(context["feature_receipt"]["path"])
        or receipt.get("sha256") != context["feature_receipt"]["sha256"]
    ):
        raise Seed42DiagnosticError(f"fold {fold} feature receipt differs")

    rows = derive_training_validation_rows(context["protocol"], fold)
    split = payload.get("split")
    expected_split = {
        "train_rows": len(rows.train_row_indices),
        "validation_rows": len(rows.validation_row_indices),
        "train_row_indices_sha256": _array_digest(rows.train_row_indices),
        "validation_row_indices_sha256": _array_digest(rows.validation_row_indices),
    }
    if split != expected_split:
        raise Seed42DiagnosticError(f"fold {fold} inner split receipt differs")

    selected_epoch = development.get("selected_epoch")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int):
        raise Seed42DiagnosticError(f"fold {fold} selected epoch is invalid")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise Seed42DiagnosticError(f"fold {fold} candidate snapshots are absent")
    matches = [
        snapshot
        for snapshot in snapshots
        if isinstance(snapshot, Mapping) and snapshot.get("epoch") == selected_epoch
    ]
    if len(matches) != 1:
        raise Seed42DiagnosticError(
            f"fold {fold} has no unique snapshot for selected epoch {selected_epoch}"
        )
    epoch_audit = development.get("epoch_audit")
    audit_matches = [
        record
        for record in epoch_audit
        if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
    ] if isinstance(epoch_audit, list) else []
    if len(audit_matches) != 1:
        raise Seed42DiagnosticError(f"fold {fold} selected epoch audit is absent")
    snapshot = matches[0]
    epoch_record = audit_matches[0]
    root = candidate_path.parent
    prediction_record = _candidate_artifact(
        snapshot.get("validation_predictions"),
        root=root,
        label=f"fold {fold} validation predictions",
        expected_hash=str(epoch_record.get("validation_predictions_sha256")),
    )
    u_record = _candidate_artifact(
        snapshot.get("u_embedding"),
        root=root,
        label=f"fold {fold} u embedding",
        expected_hash=str(epoch_record.get("u_embedding_sha256")),
    )
    i_record = _candidate_artifact(
        snapshot.get("i_embedding"),
        root=root,
        label=f"fold {fold} i embedding",
        expected_hash=str(epoch_record.get("i_embedding_sha256")),
    )
    try:
        with np.load(prediction_record["path"], allow_pickle=False) as archive:
            if set(archive.files) != set(PREDICTION_FIELDS):
                raise Seed42DiagnosticError(
                    f"fold {fold} validation prediction fields differ"
                )
        predictions = load_predictions(prediction_record["path"])
        predictions = validate_prediction_coverage(
            prediction_record["path"],
            expected_row_id=rows.validation_row_indices,
            expected_pair=rows.validation_pairs,
            expected_label=rows.validation_labels,
        )
        validation_metrics = evaluate_validation_scores(
            rows.validation_labels, predictions.score
        )
        threshold = select_f1_threshold(
            rows.validation_labels, predictions.score
        ).as_dict()
    except BenchmarkAuditError as error:
        raise Seed42DiagnosticError(
            f"fold {fold} validation artifact audit failed: {error}"
        ) from error
    observed_auprc = float(validation_metrics["Validation_AUPRC"])
    for label, value in (
        ("candidate", snapshot.get("validation_auprc")),
        ("development", development.get("validation_auprc")),
        ("epoch audit", epoch_record.get("validation_auprc")),
    ):
        try:
            agrees = math.isclose(
                float(value), observed_auprc, rel_tol=0.0, abs_tol=METRIC_ATOL
            )
        except (TypeError, ValueError):
            agrees = False
        if not agrees:
            raise Seed42DiagnosticError(f"fold {fold} {label} AUPRC differs")

    u = _load_embedding(u_record, label=f"fold {fold} u embedding")
    i = _load_embedding(i_record, label=f"fold {fold} i embedding")
    recomputed = _score_embeddings(u, i, rows.validation_pairs)
    if not np.allclose(recomputed, predictions.score, rtol=0.0, atol=SCORE_ATOL):
        raise Seed42DiagnosticError(
            f"fold {fold} validation scores differ from selected embeddings"
        )
    return {
        "fold": fold,
        "candidate_manifest": manifest,
        "config": config,
        "selected_epoch": selected_epoch,
        "selected_validation_auprc": observed_auprc,
        "frozen_f1_threshold": threshold,
        "validation_predictions": prediction_record,
        "u_embedding": u_record,
        "i_embedding": i_record,
    }


def _support_code() -> dict[str, dict[str, str]]:
    return {
        "diagnostic_selector_scorer": _record(SELECTOR_PATH),
        "metrics": _record(METRICS_PATH),
        "protocol": _record(PROTOCOL_PATH),
        "refined_runner": _record(REFINED_RUNNER_PATH),
        "feature_receipt_validator": _record(FEATURE_RECEIPT_VALIDATOR_PATH),
    }


def _build_selection_payload(
    context: Mapping[str, Any], audited: Sequence[Mapping[str, Any]], output_root: Path
) -> dict[str, Any]:
    by_fold = {int(record["fold"]): dict(record) for record in audited}
    if set(by_fold) != set(FOLDS) or len(audited) != len(FOLDS):
        raise Seed42DiagnosticError("audited selection does not cover exactly five folds")
    return {
        "schema": SELECTION_SCHEMA,
        "dataset": DATASET,
        "seed": SEED,
        "model": MODEL,
        "post_selection_diagnostic": True,
        "manuscript_unbiased": False,
        "diagnostic_warning": (
            "Architecture and per-fold epochs were selected on seed42 inner-validation; "
            "these outer scores are descriptive only and must not replace an unbiased main result."
        ),
        "outer_rows_loaded_during_freeze": False,
        "development_selection": context["record"],
        "development_selection_sidecar": context["sidecar"],
        "development_plan": context["development_plan"],
        "development_plan_sidecar": context["development_plan_sidecar"],
        "amendment": context["amendment"],
        "amendment_sidecar": context["amendment_sidecar"],
        "protocol": context["protocol_record"],
        "feature_deep_audit_receipt": context["feature_receipt"],
        "development_code": context["code"],
        "development_selectors": {
            "original": context["original_selector"],
            "amendment": context["amendment_selector"],
        },
        "support_code": _support_code(),
        "selection_rule": {
            "architecture": "already promoted residual_mlp from amended development selection",
            "epoch": "already selected per fold by maximum inner-validation AUPRC",
            "threshold": "maximum inner-validation F1 over 100 thresholds; lowest-threshold tie break",
            "outer_rows_or_labels_used": False,
        },
        "folds": {str(fold): by_fold[fold] for fold in FOLDS},
        "outputs": {
            "outer_root": str((output_root / "outer_test").resolve()),
            "summary": str((output_root / "outer_test" / "summary.json").resolve()),
        },
    }


def _prepare_selection(
    development_selection: Path | str,
    output_root: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    context = _validate_development_selection(
        Path(development_selection), protocol_loader=protocol_loader
    )
    audited = [_audit_selected_fold(context, fold) for fold in FOLDS]
    return _build_selection_payload(context, audited, Path(output_root).resolve())


def _freeze_selection_with_protocol_loader(
    development_selection: Path | str,
    output_root: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> dict[str, Any]:
    output_root = Path(output_root).resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise Seed42DiagnosticError(
            f"diagnostic output root already exists; refusing retry: {output_root}"
        )
    payload = _prepare_selection(
        development_selection, output_root, protocol_loader=protocol_loader
    )
    selection_path = output_root / "diagnostic_selection.json"
    outer_root = Path(payload["outputs"]["outer_root"])
    if selection_path.exists() or _sidecar(selection_path).exists() or outer_root.exists():
        raise Seed42DiagnosticError("diagnostic outputs existed before selection freeze")
    _atomic_json(selection_path, payload)
    digest = sha256(selection_path)
    _atomic_text(_sidecar(selection_path), digest + "\n")
    return {**payload, "selection": {"path": str(selection_path), "sha256": digest}}


def freeze_selection(
    development_selection: Path | str, output_root: Path | str
) -> dict[str, Any]:
    """Freeze the already-selected seed42 model without requesting outer rows."""

    return _freeze_selection_with_protocol_loader(
        development_selection,
        output_root,
        protocol_loader=load_protocol_manifest,
    )


def _verify_artifact_bindings(selection: Mapping[str, Any]) -> None:
    _artifact(selection.get("development_selection"), label="development selection")
    _artifact(
        selection.get("development_selection_sidecar"),
        label="development selection sidecar",
    )
    _artifact(selection.get("development_plan"), label="development plan")
    _artifact(selection.get("development_plan_sidecar"), label="development plan sidecar")
    _artifact(selection.get("amendment"), label="amendment")
    _artifact(selection.get("amendment_sidecar"), label="amendment sidecar")
    _artifact(selection.get("protocol"), label="protocol")
    _artifact(selection.get("feature_deep_audit_receipt"), label="feature receipt")
    code = selection.get("development_code")
    if not isinstance(code, Mapping) or set(code) != set(EXPECTED_CODE_PATHS):
        raise Seed42DiagnosticError("frozen development code bindings are incomplete")
    for role, expected_path in EXPECTED_CODE_PATHS.items():
        _artifact(code[role], label=f"development {role}", expected_path=expected_path)
    selectors = selection.get("development_selectors")
    if not isinstance(selectors, Mapping) or set(selectors) != {"original", "amendment"}:
        raise Seed42DiagnosticError("frozen selector bindings are incomplete")
    _artifact(
        selectors["original"],
        label="original selector",
        expected_path=EXPECTED_CODE_PATHS["selector"],
    )
    _artifact(
        selectors["amendment"],
        label="amendment selector",
        expected_path=AMENDMENT_SELECTOR_PATH,
    )
    expected_support = {
        "diagnostic_selector_scorer": SELECTOR_PATH,
        "metrics": METRICS_PATH,
        "protocol": PROTOCOL_PATH,
        "refined_runner": REFINED_RUNNER_PATH,
        "feature_receipt_validator": FEATURE_RECEIPT_VALIDATOR_PATH,
    }
    support = selection.get("support_code")
    if not isinstance(support, Mapping) or set(support) != set(expected_support):
        raise Seed42DiagnosticError("frozen support-code bindings are incomplete")
    for role, expected_path in expected_support.items():
        _artifact(support[role], label=f"support {role}", expected_path=expected_path)
    folds = selection.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {str(fold) for fold in FOLDS}:
        raise Seed42DiagnosticError("frozen fold decisions are incomplete")
    for fold in FOLDS:
        record = folds[str(fold)]
        if not isinstance(record, Mapping) or record.get("fold") != fold:
            raise Seed42DiagnosticError(f"frozen fold {fold} decision is invalid")
        for field in (
            "candidate_manifest",
            "config",
            "validation_predictions",
            "u_embedding",
            "i_embedding",
        ):
            _artifact(record.get(field), label=f"fold {fold} {field}")


def _load_verified_selection(
    selection_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
) -> tuple[dict[str, Any], str, BenchmarkProtocol, Path, Path]:
    path = Path(selection_path).resolve(strict=True)
    digest, _ = _read_frozen(path, label="diagnostic selection")
    observed = _load_json(path, label="diagnostic selection")
    if observed.get("schema") != SELECTION_SCHEMA:
        raise Seed42DiagnosticError("diagnostic selection schema is invalid")
    if (
        observed.get("post_selection_diagnostic") is not True
        or observed.get("manuscript_unbiased") is not False
        or observed.get("outer_rows_loaded_during_freeze") is not False
    ):
        raise Seed42DiagnosticError("diagnostic disclosure flags are invalid")
    _verify_artifact_bindings(observed)
    development_path = Path(observed["development_selection"]["path"])
    output_root = path.parent.resolve()
    expected = _prepare_selection(
        development_path, output_root, protocol_loader=protocol_loader
    )
    if _canonical(observed) != _canonical(expected):
        raise Seed42DiagnosticError(
            "frozen diagnostic selection differs from deterministic validation replay"
        )
    protocol = protocol_loader(Path(observed["protocol"]["path"]))
    outer_root = Path(observed["outputs"]["outer_root"]).resolve()
    summary_path = Path(observed["outputs"]["summary"]).resolve()
    if outer_root != output_root / "outer_test" or summary_path != outer_root / "summary.json":
        raise Seed42DiagnosticError("diagnostic output paths escape the frozen output root")
    return observed, digest, protocol, outer_root, summary_path


def _reserve_outer_outputs(
    *, outer_root: Path, selection_hash: str
) -> dict[int, Path]:
    if outer_root.exists() or outer_root.is_symlink():
        raise Seed42DiagnosticError(
            f"refusing repeated or partial outer scoring: {outer_root}"
        )
    outer_root.mkdir(parents=True, exist_ok=False)
    reservations = {fold: outer_root / f"fold_{fold}" for fold in FOLDS}
    _atomic_json(
        outer_root / ".all_five_folds_reserved.json",
        {
            "selection_sha256": selection_hash,
            "state": "all-five-reserved-before-first-outer-score",
            "folds": list(FOLDS),
        },
    )
    for fold, output in reservations.items():
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            output / ".reserved_before_outer_score.json",
            {
                "selection_sha256": selection_hash,
                "fold": fold,
                "state": "reserved-before-any-outer-score",
            },
        )
    if len(reservations) != 5 or not all(path.is_dir() for path in reservations.values()):
        raise Seed42DiagnosticError("all five outer-fold outputs were not reserved")
    return reservations


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.isfinite(array).all():
        raise Seed42DiagnosticError("aggregate requires five finite fold values")
    return {
        "fold_values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)),
    }


def _score_with_protocol_loader(
    selection_path: Path | str,
    *,
    protocol_loader: Callable[[Path], BenchmarkProtocol],
    score_callback: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = _score_embeddings,
) -> dict[str, Any]:
    # Deterministic validation replay completes before the first fold_rows call.
    selection, selection_hash, protocol, outer_root, summary_path = (
        _load_verified_selection(selection_path, protocol_loader=protocol_loader)
    )
    reservations = _reserve_outer_outputs(
        outer_root=outer_root, selection_hash=selection_hash
    )
    computed: dict[int, dict[str, Any]] = {}
    score_calls = 0
    for fold in FOLDS:
        fold_rows = protocol.fold_rows(fold)
        selected = selection["folds"][str(fold)]
        u_record = selected["u_embedding"]
        i_record = selected["i_embedding"]
        if sha256(u_record["path"]) != u_record["sha256"] or sha256(
            i_record["path"]
        ) != i_record["sha256"]:
            raise Seed42DiagnosticError(
                f"selected embeddings changed before outer score: fold {fold}"
            )
        u = _load_embedding(u_record, label=f"fold {fold} u embedding")
        i = _load_embedding(i_record, label=f"fold {fold} i embedding")
        scores = np.asarray(
            score_callback(u, i, fold_rows.test_pairs), dtype=np.float64
        )
        score_calls += 1
        if scores.shape != (len(fold_rows.test_pairs),) or not np.isfinite(scores).all():
            raise Seed42DiagnosticError(f"outer scorer returned invalid fold {fold} scores")
        threshold_record = selected["frozen_f1_threshold"]
        try:
            threshold = float(threshold_record["Validation_Selected_Threshold"])
            metrics = evaluate_test_scores(
                fold_rows.test_labels, scores, threshold=threshold
            )
        except (KeyError, TypeError, ValueError, BenchmarkAuditError) as error:
            raise Seed42DiagnosticError(
                f"outer metric evaluation failed for fold {fold}: {error}"
            ) from error
        computed[fold] = {
            "row_id": np.asarray(fold_rows.test_row_indices),
            "pairs": np.asarray(fold_rows.test_pairs),
            "labels": np.asarray(fold_rows.test_labels),
            "scores": scores,
            "metrics": {
                **metrics,
                "Dataset": DATASET,
                "Seed": SEED,
                "Fold": fold,
                "Model": MODEL,
                "Selected_Epoch": selected["selected_epoch"],
                "Selected_Validation_AUPRC": selected["selected_validation_auprc"],
                "Decision_Frozen_Before_This_Outer_Score": True,
                "Post_Selection_Diagnostic": True,
                "Manuscript_Unbiased": False,
                "Selection_SHA256": selection_hash,
            },
        }
    if score_calls != 5:
        raise Seed42DiagnosticError(
            f"outer scorer must be invoked exactly five times, got {score_calls}"
        )

    metric_names = ("AUROC", "AUPRC", "F1")
    per_fold = {str(fold): computed[fold]["metrics"] for fold in FOLDS}
    aggregate = {
        metric: _mean_std(
            [float(computed[fold]["metrics"][metric]) for fold in FOLDS]
        )
        for metric in metric_names
    }
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "dataset": DATASET,
        "seed": SEED,
        "model": MODEL,
        "post_selection_diagnostic": True,
        "manuscript_unbiased": False,
        "diagnostic_warning": selection["diagnostic_warning"],
        "selection": {
            "path": str(Path(selection_path).resolve()),
            "sha256": selection_hash,
        },
        "protocol": selection["protocol"],
        "outer_score_calls": score_calls,
        "all_five_reserved_before_scoring": True,
        "partial_scoring_api": False,
        "threshold_source": "each fold's selected-epoch inner-validation maximum F1",
        "per_fold": per_fold,
        "aggregate": aggregate,
        "standard_deviation": (
            "sample standard deviation (ddof=1) is manuscript-facing; "
            "population standard deviation is also recorded"
        ),
    }

    # Publish no outer prediction or metric until all five folds have scored.
    artifacts: dict[str, Any] = {}
    for fold in FOLDS:
        record = computed[fold]
        output = reservations[fold]
        prediction_path = save_predictions(
            output / "test_predictions.npz",
            row_id=record["row_id"],
            pair=record["pairs"],
            label=record["labels"],
            score=record["scores"],
        )
        metrics_path = output / "metrics.json"
        binding_path = output / "selection_binding.json"
        _atomic_json(metrics_path, record["metrics"])
        _atomic_json(
            binding_path,
            {
                "selection_path": str(Path(selection_path).resolve()),
                "selection_sha256": selection_hash,
                "fold": fold,
                "selected_epoch": selection["folds"][str(fold)]["selected_epoch"],
                "u_embedding": selection["folds"][str(fold)]["u_embedding"],
                "i_embedding": selection["folds"][str(fold)]["i_embedding"],
                "test_predictions": _record(prediction_path),
                "outer_score_calls_for_this_fold": 1,
                "post_selection_diagnostic": True,
                "manuscript_unbiased": False,
            },
        )
        artifacts[str(fold)] = {
            "test_predictions": _record(prediction_path),
            "metrics": _record(metrics_path),
            "selection_binding": _record(binding_path),
        }
    summary["artifacts"] = artifacts
    _atomic_json(summary_path, summary)
    _atomic_text(_sidecar(summary_path), sha256(summary_path) + "\n")
    return summary


def score(selection_path: Path | str) -> dict[str, Any]:
    """Reserve and score all five seed42 outer folds exactly once."""

    return _score_with_protocol_loader(
        selection_path, protocol_loader=load_protocol_manifest
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze-selection",
        help="freeze selected epochs and validation-only F1 thresholds",
    )
    freeze.add_argument("--development-selection", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    score_parser = subparsers.add_parser(
        "score", help="reserve and score all five outer folds once"
    )
    score_parser.add_argument("--selection", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze-selection":
            payload = freeze_selection(
                args.development_selection, args.output_root
            )
        else:
            payload = score(args.selection)
    except Seed42DiagnosticError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
