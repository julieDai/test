#!/usr/bin/env python3
"""Select validation-only TCM ASTRA candidates and evaluate winners once."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import (
    AUPRC_DEFINITION,
    AVERAGE_PRECISION_DEFINITION,
    evaluate_validation_and_test,
    evaluate_validation_scores,
    save_predictions,
    validate_prediction_coverage,
)
from .protocol import BenchmarkProtocol, load_protocol_manifest
from .tcm_feature_bundle import validate_tcm_feature_bundle
from .backend_paths import TCM_BACKEND_ROOT


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "astra-tcm-hparam-plan/v1"
SELECTION_SCHEMA = "astra-tcm-hparam-selection/v1"
SELECTION_RECEIPT_SCHEMA = "astra-tcm-hparam-selection-receipt/v1"
ARRAY_HASH_SCHEMA = "astra-canonical-array-sha256/v1"
EXPECTED_SELECTION_METRIC = "Validation_AUPRC"
PROVENANCE_AUPRC_DEFINITION = (
    "trapezoidal area: sklearn.metrics.auc(recall, precision) after "
    "precision_recall_curve"
)
EXPECTED_TIE_BREAK = [
    "higher Validation_AUROC",
    "lower epochs",
    "lower learning_rate",
    "lexicographically smaller candidate id",
]
EXPECTED_FIXED = {
    "batch_size": 2000,
    "embedding_dimension": 64,
    "reg_u": 0.001,
    "text_fusion_alpha": 1.0,
    "semantic_adapter": "identity",
}
EXPECTED_ADAPTER_SOURCES = {
    "astra_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
    "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
}
EXPECTED_MODEL_SOURCES = {
    "configuration": TCM_BACKEND_ROOT / "ASTRA.conf",
    "herb_recommender": TCM_BACKEND_ROOT / "base/herbRecommender.py",
    "model": TCM_BACKEND_ROOT / "HGHDA.py",
    "rating": TCM_BACKEND_ROOT / "rating.py",
}
EXPECTED_CANDIDATE_RUNNER = SOURCE_ROOT / "astra/benchmark/tcm_astra_hparam_runner.py"
PROTOCOL_INPUT_FIELDS = (
    "protocol_version",
    "dataset",
    "seed",
    "id_convention_on_disk",
    "id_convention_in_adapters",
    "fold_rule",
    "pair_order",
    "n_folds",
    "validation",
    "counts",
    "negative_sampling",
    "mapping_strategy",
    "embedding_dir",
    "files",
    "arrays",
    "mappings",
    "folds",
)


class TCMHparamSelectionError(RuntimeError):
    """Raised when candidate isolation or selection provenance is invalid."""


@dataclass(frozen=True)
class TrainingValidationRows:
    """Frozen inner rows derived without constructing the outer-test split."""

    fold: int
    train_pairs: np.ndarray
    train_labels: np.ndarray
    train_row_indices: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray
    validation_row_indices: np.ndarray


@dataclass(frozen=True)
class PlanContext:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    protocol_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol: BenchmarkProtocol
    feature_bundle: Path
    feature_manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMHparamSelectionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMHparamSelectionError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_array_sha256(values: np.ndarray) -> str:
    """Use the canonical array digest from ``protocol.py``/``astra_runner.py``."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _object_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _single_embedding(root: Path, prefix: str) -> Path:
    matches = sorted(root.glob(f"{prefix}*.txt"))
    if len(matches) != 1:
        raise TCMHparamSelectionError(
            f"expected one {prefix} embedding in {root}, found {len(matches)}"
        )
    return matches[0].resolve()


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _selection_receipt_path(selection_path: Path) -> Path:
    return selection_path.with_name(selection_path.name + ".receipt.json")


def _freeze(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _validation_size(count: int, fraction: float) -> int:
    if count < 2:
        raise TCMHparamSelectionError(
            "each outer-training class needs at least two rows"
        )
    selected = int(math.floor(count * fraction))
    return min(max(selected, 1), count - 1)


def _derive_training_validation_rows(
    protocol: BenchmarkProtocol, fold: int
) -> TrainingValidationRows:
    """Reproduce only train/validation membership; never construct test rows."""

    fold = int(fold)
    if not 1 <= fold <= int(protocol.n_folds):
        raise TCMHparamSelectionError(
            f"fold must be in 1..{protocol.n_folds}, got {fold}"
        )
    positives = np.asarray(protocol.positive_pairs, dtype=np.int64)
    negatives = np.asarray(protocol.negative_pairs, dtype=np.int64)
    positive_count = len(positives)
    negative_count = len(negatives)
    positive_rows = np.arange(positive_count, dtype=np.int64)
    negative_local = np.arange(negative_count, dtype=np.int64)
    negative_rows = positive_count + negative_local

    positive_outer = positive_rows[positive_rows % protocol.n_folds != fold - 1]
    negative_outer = negative_rows[negative_local % protocol.n_folds != fold - 1]
    positive_rng = np.random.default_rng(
        np.random.SeedSequence((protocol.seed, fold, 1))
    )
    negative_rng = np.random.default_rng(
        np.random.SeedSequence((protocol.seed, fold, 0))
    )
    positive_outer = positive_outer[positive_rng.permutation(len(positive_outer))]
    negative_outer = negative_outer[negative_rng.permutation(len(negative_outer))]
    positive_validation_count = _validation_size(
        len(positive_outer), protocol.validation_fraction
    )
    negative_validation_count = _validation_size(
        len(negative_outer), protocol.validation_fraction
    )
    validation_rows = np.concatenate(
        (
            positive_outer[:positive_validation_count],
            negative_outer[:negative_validation_count],
        )
    )
    train_rows = np.concatenate(
        (
            positive_outer[positive_validation_count:],
            negative_outer[negative_validation_count:],
        )
    )

    def pairs_for(indices: np.ndarray) -> np.ndarray:
        positive_mask = indices < positive_count
        output = np.empty((len(indices), 2), dtype=np.int64)
        output[positive_mask] = positives[indices[positive_mask]]
        output[~positive_mask] = negatives[
            indices[~positive_mask] - positive_count
        ]
        return output

    def labels_for(indices: np.ndarray) -> np.ndarray:
        return np.asarray(indices < positive_count, dtype=np.uint8)

    if np.intersect1d(train_rows, validation_rows).size:
        raise TCMHparamSelectionError("inner train/validation rows overlap")
    return TrainingValidationRows(
        fold=fold,
        train_pairs=_freeze(pairs_for(train_rows), np.dtype(np.int64)),
        train_labels=_freeze(labels_for(train_rows), np.dtype(np.uint8)),
        train_row_indices=_freeze(train_rows, np.dtype(np.int64)),
        validation_pairs=_freeze(
            pairs_for(validation_rows), np.dtype(np.int64)
        ),
        validation_labels=_freeze(
            labels_for(validation_rows), np.dtype(np.uint8)
        ),
        validation_row_indices=_freeze(
            validation_rows, np.dtype(np.int64)
        ),
    )


def _manifest_fold_record(context: PlanContext, fold: int) -> Mapping[str, Any]:
    records = context.protocol_manifest.get("folds")
    if not isinstance(records, list):
        raise TCMHparamSelectionError("protocol manifest has no fold records")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("fold") == fold
    ]
    if len(matches) != 1:
        raise TCMHparamSelectionError(
            f"protocol manifest has {len(matches)} records for fold {fold}"
        )
    return matches[0]


def _load_plan_context(plan_path: Path) -> PlanContext:
    plan_path = plan_path.resolve()
    plan = _load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise TCMHparamSelectionError("unsupported plan schema")
    if plan.get("dataset") != "tcm-suite" or plan.get("seed") != 42:
        raise TCMHparamSelectionError("plan dataset/seed is not frozen to TCM seed 42")
    if plan.get("status") != "frozen-before-new-candidate-testing":
        raise TCMHparamSelectionError("candidate plan is not marked frozen")
    if plan.get("selection_metric") != EXPECTED_SELECTION_METRIC:
        raise TCMHparamSelectionError("plan selection metric is not Validation_AUPRC")
    if plan.get("tie_break") != EXPECTED_TIE_BREAK:
        raise TCMHparamSelectionError("plan tie-break order differs from the selector")
    fixed = plan.get("fixed")
    if not isinstance(fixed, Mapping) or set(fixed) != set(EXPECTED_FIXED):
        raise TCMHparamSelectionError("plan fixed hyperparameter fields differ")
    for key, expected in EXPECTED_FIXED.items():
        observed = fixed.get(key)
        if isinstance(expected, float):
            valid = _close(observed, expected)
        else:
            valid = observed == expected
        if not valid:
            raise TCMHparamSelectionError(
                f"plan fixed {key} is {observed!r}, expected {expected!r}"
            )

    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise TCMHparamSelectionError("plan candidate list is empty")
    candidates: list[Mapping[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "semantic_adapter",
            "epochs",
            "learning_rate",
        }:
            raise TCMHparamSelectionError(
                f"plan candidate {index} has invalid fields"
            )
        candidate_id = raw.get("id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or Path(candidate_id).name != candidate_id
            or candidate_id in {".", ".."}
            or candidate_id in candidate_ids
        ):
            raise TCMHparamSelectionError(
                f"plan candidate id is unsafe or duplicated: {candidate_id!r}"
            )
        if raw.get("semantic_adapter") != fixed["semantic_adapter"]:
            raise TCMHparamSelectionError(
                f"candidate {candidate_id} semantic adapter differs from fixed plan"
            )
        epochs = raw.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise TCMHparamSelectionError(
                f"candidate {candidate_id} epochs is invalid"
            )
        learning_rate = raw.get("learning_rate")
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as error:
            raise TCMHparamSelectionError(
                f"candidate {candidate_id} learning rate is invalid"
            ) from error
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise TCMHparamSelectionError(
                f"candidate {candidate_id} learning rate is invalid"
            )
        candidate_ids.add(candidate_id)
        candidates.append(dict(raw))

    protocol_path = Path(str(plan.get("protocol_manifest", ""))).resolve()
    if not protocol_path.is_file():
        raise TCMHparamSelectionError("frozen protocol manifest is absent")
    if _sha256(protocol_path) != plan.get("protocol_manifest_sha256"):
        raise TCMHparamSelectionError("frozen protocol manifest hash differs")
    protocol_manifest = _load_json(protocol_path)
    protocol = load_protocol_manifest(protocol_path)
    if (
        protocol.dataset != "tcm-suite"
        or protocol.seed != 42
        or protocol.n_folds != 5
    ):
        raise TCMHparamSelectionError("frozen protocol dataset/seed/folds differ")

    feature_bundle = Path(str(plan.get("feature_bundle", ""))).resolve()
    feature_manifest = feature_bundle / "bundle_manifest.json"
    if not feature_manifest.is_file():
        raise TCMHparamSelectionError("frozen feature-bundle manifest is absent")
    if _sha256(feature_manifest) != plan.get("feature_bundle_manifest_sha256"):
        raise TCMHparamSelectionError("frozen feature-bundle manifest hash differs")
    validate_tcm_feature_bundle(feature_manifest, protocol=protocol)

    return PlanContext(
        path=plan_path,
        sha256=_sha256(plan_path),
        payload=plan,
        candidates=tuple(candidates),
        protocol_path=protocol_path,
        protocol_manifest=protocol_manifest,
        protocol=protocol,
        feature_bundle=feature_bundle,
        feature_manifest=feature_manifest,
    )


def _verify_file_record(
    record: object, *, expected_path: Path, label: str
) -> str:
    if not isinstance(record, Mapping):
        raise TCMHparamSelectionError(f"{label} provenance is absent")
    expected_path = expected_path.resolve()
    recorded_path = Path(str(record.get("path", ""))).resolve()
    if recorded_path != expected_path or not expected_path.is_file():
        raise TCMHparamSelectionError(f"{label} path differs from the frozen path")
    actual_sha256 = _sha256(expected_path)
    if record.get("sha256") != actual_sha256:
        raise TCMHparamSelectionError(f"{label} SHA-256 differs from disk")
    if "bytes" in record and record.get("bytes") != expected_path.stat().st_size:
        raise TCMHparamSelectionError(f"{label} byte count differs from disk")
    return actual_sha256


def _expected_protocol_input_manifest(context: PlanContext) -> dict[str, Any]:
    """Rebuild the recorded input manifest without deriving any outer rows."""

    expected: dict[str, Any] = {}
    for field in PROTOCOL_INPUT_FIELDS:
        if field == "pair_order":
            expected[field] = context.protocol_manifest.get(field, "seeded")
        elif field not in context.protocol_manifest:
            raise TCMHparamSelectionError(
                f"protocol manifest is missing input field {field}"
            )
        else:
            expected[field] = context.protocol_manifest[field]
    return expected


def _verify_source_map(
    provenance: object,
    *,
    expected_paths: Mapping[str, Path],
    label: str,
) -> str:
    if not isinstance(provenance, Mapping) or set(provenance) != set(expected_paths):
        raise TCMHparamSelectionError(f"{label} source set differs")
    verified: dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        _verify_file_record(
            provenance.get(name),
            expected_path=expected_path,
            label=f"{label}.{name}",
        )
        verified[name] = dict(provenance[name])
    return _object_sha256(verified)


def _verify_source_inputs(
    provenance: object, *, context: PlanContext
) -> str:
    manifest_files = context.protocol_manifest.get("files")
    if not isinstance(provenance, Mapping) or not isinstance(manifest_files, Mapping):
        raise TCMHparamSelectionError("candidate source-input provenance is absent")
    if set(provenance) != set(manifest_files):
        raise TCMHparamSelectionError("candidate source-input set differs from protocol")
    verified: dict[str, Any] = {}
    for name, raw_expected in manifest_files.items():
        if not isinstance(raw_expected, Mapping):
            raise TCMHparamSelectionError(f"protocol source {name} is invalid")
        expected_path = Path(str(raw_expected.get("path", ""))).resolve()
        record = provenance.get(name)
        _verify_file_record(
            record,
            expected_path=expected_path,
            label=f"source_inputs_verified.{name}",
        )
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != raw_expected.get("sha256")
            or record.get("bytes") != raw_expected.get("bytes")
        ):
            raise TCMHparamSelectionError(
                f"candidate source {name} differs from the protocol manifest"
            )
        verified[name] = dict(record)
    return _object_sha256(verified)


def _validate_candidate(
    *,
    root: Path,
    fold: int,
    candidate: Mapping[str, Any],
    context: PlanContext,
    validation: TrainingValidationRows,
) -> dict[str, Any]:
    candidate_id = str(candidate["id"])
    candidate_root = (root / candidate_id / f"fold_{fold}").resolve()
    metrics_path = candidate_root / "results" / "metrics.json"
    protocol_path = candidate_root / "protocol.json"
    validation_path = candidate_root / "results" / "validation_predictions.npz"
    for path in (metrics_path, protocol_path, validation_path):
        if not path.is_file():
            raise TCMHparamSelectionError(f"candidate artifact is missing: {path}")
    forbidden_tests = [
        path
        for path in candidate_root.rglob("test_predictions.npz")
        if path.is_file()
    ]
    if forbidden_tests:
        raise TCMHparamSelectionError(
            "candidate outer-test prediction exists before selection: "
            + ", ".join(str(path) for path in forbidden_tests)
        )

    metrics = _load_json(metrics_path)
    protocol = _load_json(protocol_path)
    binding = protocol.get("tcm_hparam_candidate")
    if not isinstance(binding, Mapping):
        raise TCMHparamSelectionError(f"candidate binding is absent: {protocol_path}")
    expected = {
        "Dataset": "tcm-suite",
        "Fold": fold,
        "Seed": int(context.protocol.seed),
        "Validation_Only": True,
        "Test_Models_Evaluated": 0,
        "Unselected_Test_Models_Evaluated": 0,
        "Candidate_ID": candidate_id,
        "Candidate_Plan_SHA256": context.sha256,
        "Tuning_Stage": "epoch-learning-rate",
        "Formal_Eligible": False,
        "Diagnostic": False,
        "Smoke": False,
        "Text_Fusion_Alpha": float(
            context.payload["fixed"]["text_fusion_alpha"]
        ),
        "Alpha_Selected_On_Validation": False,
        "Decision_Frozen_Before_Test": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise TCMHparamSelectionError(
                f"{candidate_id} fold {fold} metric {key} is "
                f"{metrics.get(key)!r}, expected {value!r}"
            )
    if binding.get("plan_sha256") != context.sha256:
        raise TCMHparamSelectionError("candidate protocol uses another plan hash")
    if Path(str(binding.get("plan_path", ""))).resolve() != context.path:
        raise TCMHparamSelectionError("candidate protocol uses another plan path")
    if binding.get("selection_metric") != EXPECTED_SELECTION_METRIC:
        raise TCMHparamSelectionError("candidate binding uses another selection metric")
    if binding.get("outer_test_materialised") is not False:
        raise TCMHparamSelectionError("candidate binding claims an outer test exists")
    candidate_runner_sha256 = _verify_file_record(
        {
            "path": binding.get("runner_path"),
            "sha256": binding.get("runner_sha256"),
        },
        expected_path=EXPECTED_CANDIDATE_RUNNER,
        label="candidate runner",
    )
    bound_candidate = binding.get("candidate")
    if not isinstance(bound_candidate, Mapping) or dict(bound_candidate) != dict(candidate):
        raise TCMHparamSelectionError("candidate protocol does not match the plan")

    if (
        protocol.get("schema") != "astra-central-fair-run/v1"
        or protocol.get("dataset") != "tcm-suite"
        or protocol.get("seed") != context.protocol.seed
        or protocol.get("fold") != fold
        or protocol.get("validation_only") is not True
        or protocol.get("formal_eligible") is not False
        or protocol.get("smoke") is not False
        or protocol.get("diagnostic") is not False
    ):
        raise TCMHparamSelectionError("candidate central-run identity differs")

    _verify_file_record(
        protocol.get("protocol_manifest"),
        expected_path=context.protocol_path,
        label="candidate protocol manifest",
    )
    expected_input_manifest = _expected_protocol_input_manifest(context)
    if protocol.get("protocol_input_manifest") != expected_input_manifest:
        raise TCMHparamSelectionError("candidate protocol input manifest differs")
    expected_input_lock = _object_sha256(expected_input_manifest)
    if protocol.get("frozen_input_lock_sha256") != expected_input_lock:
        raise TCMHparamSelectionError("candidate frozen input lock differs")

    feature_bundle = protocol.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping):
        raise TCMHparamSelectionError("candidate feature-bundle provenance is absent")
    _verify_file_record(
        feature_bundle,
        expected_path=context.feature_manifest,
        label="candidate feature-bundle manifest",
    )
    if (
        feature_bundle.get("sha256")
        != context.payload.get("feature_bundle_manifest_sha256")
        or feature_bundle.get("formal_eligible") is not True
        or feature_bundle.get("recursive_validation") is not True
    ):
        raise TCMHparamSelectionError("candidate feature-bundle contract differs")

    adapter_sources_sha256 = _verify_source_map(
        protocol.get("adapter_sources"),
        expected_paths=EXPECTED_ADAPTER_SOURCES,
        label="adapter_sources",
    )
    model_sources_sha256 = _verify_source_map(
        protocol.get("model_sources"),
        expected_paths=EXPECTED_MODEL_SOURCES,
        label="model_sources",
    )
    source_inputs_sha256 = _verify_source_inputs(
        protocol.get("source_inputs_verified"), context=context
    )

    hyperparameters = protocol.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise TCMHparamSelectionError("candidate hyperparameters are absent")
    expected_hyperparameters = {
        "epochs": int(candidate["epochs"]),
        "formal_epochs": int(candidate["epochs"]),
        "batch_size": int(context.payload["fixed"]["batch_size"]),
        "learning_rate": float(candidate["learning_rate"]),
        "embedding_dimension": int(
            context.payload["fixed"]["embedding_dimension"]
        ),
        "reg_u": float(context.payload["fixed"]["reg_u"]),
        "text_fusion_alpha": float(
            context.payload["fixed"]["text_fusion_alpha"]
        ),
        "smoke_override": False,
        "diagnostic_mode": False,
    }
    if set(hyperparameters) != set(expected_hyperparameters):
        raise TCMHparamSelectionError("candidate hyperparameter fields differ")
    for key, expected_value in expected_hyperparameters.items():
        observed_value = hyperparameters.get(key)
        if isinstance(expected_value, float):
            valid = _close(observed_value, expected_value)
        else:
            valid = observed_value == expected_value
        if not valid:
            raise TCMHparamSelectionError(
                f"candidate hyperparameter {key} is {observed_value!r}, "
                f"expected {expected_value!r}"
            )

    expected_steps = int(candidate["epochs"]) * int(
        math.ceil(
            len(validation.train_pairs)
            / int(context.payload["fixed"]["batch_size"])
        )
    )
    if (
        protocol.get("optimizer_steps") != expected_steps
        or metrics.get("Optimizer_Steps") != expected_steps
    ):
        raise TCMHparamSelectionError("candidate optimizer-step budget differs")

    evaluation = protocol.get("evaluation")
    split_rows = protocol.get("split_rows")
    if not isinstance(evaluation, Mapping) or not isinstance(split_rows, Mapping):
        raise TCMHparamSelectionError("candidate evaluation provenance is absent")
    if (
        evaluation.get("validation_only") is not True
        or evaluation.get("test_predictions") is not None
        or evaluation.get("test_models_evaluated") != 0
        or evaluation.get("unselected_test_models_evaluated") != 0
        or evaluation.get("test_threshold_search") is not False
        or evaluation.get("auprc_definition") != PROVENANCE_AUPRC_DEFINITION
        or evaluation.get("average_precision_definition")
        != AVERAGE_PRECISION_DEFINITION
    ):
        raise TCMHparamSelectionError("candidate protocol materialised an outer test")

    fold_manifest = _manifest_fold_record(context, fold)
    train_manifest = fold_manifest.get("train")
    validation_manifest = fold_manifest.get("validation")
    test_manifest = fold_manifest.get("test")
    if not all(
        isinstance(record, Mapping)
        for record in (train_manifest, validation_manifest, test_manifest)
    ):
        raise TCMHparamSelectionError("protocol fold split metadata is absent")
    expected_validation_row_hash = _canonical_array_sha256(
        validation.validation_row_indices
    )
    if (
        validation_manifest.get("row_indices_sha256")
        != expected_validation_row_hash
        or validation_manifest.get("rows") != len(validation.validation_pairs)
        or validation_manifest.get("positive_rows")
        != int(validation.validation_labels.sum())
        or validation_manifest.get("negative_rows")
        != int(len(validation.validation_labels) - validation.validation_labels.sum())
    ):
        raise TCMHparamSelectionError(
            f"fold {fold} derived validation rows differ from the frozen manifest"
        )
    if (
        train_manifest.get("row_indices_sha256")
        != _canonical_array_sha256(validation.train_row_indices)
        or train_manifest.get("rows") != len(validation.train_pairs)
        or train_manifest.get("positive_rows") != int(validation.train_labels.sum())
        or train_manifest.get("negative_rows")
        != int(len(validation.train_labels) - validation.train_labels.sum())
    ):
        raise TCMHparamSelectionError(
            f"fold {fold} derived training rows differ from the frozen manifest"
        )
    expected_split_rows = {
        "train_full": len(validation.train_pairs),
        "train_used": len(validation.train_pairs),
        "train_positive_graph_edges_full": int(validation.train_labels.sum()),
        "validation_full": len(validation.validation_pairs),
        "validation_used": len(validation.validation_pairs),
        "test_full": int(test_manifest["rows"]),
        "test_used": 0,
    }
    if dict(split_rows) != expected_split_rows:
        raise TCMHparamSelectionError("candidate split-row counts differ")
    split_hashes = protocol.get("split_array_sha256")
    expected_split_hashes = {
        "train_pairs_used": _canonical_array_sha256(validation.train_pairs),
        "train_labels_used": _canonical_array_sha256(validation.train_labels),
        "train_positive_pairs_full": _canonical_array_sha256(
            validation.train_pairs[validation.train_labels == 1]
        ),
        "validation_pairs_used": _canonical_array_sha256(
            validation.validation_pairs
        ),
        "test_pairs_used": None,
    }
    if not isinstance(split_hashes, Mapping) or dict(split_hashes) != expected_split_hashes:
        raise TCMHparamSelectionError("candidate split-array hashes differ")

    with np.load(validation_path, allow_pickle=False) as artifact:
        required = {"row_id", "pair", "label", "score"}
        if set(artifact.files) != required:
            raise TCMHparamSelectionError(
                f"validation artifact fields differ: {sorted(artifact.files)}"
            )
    table = validate_prediction_coverage(
        validation_path,
        expected_row_id=validation.validation_row_indices,
        expected_pair=validation.validation_pairs,
        expected_label=validation.validation_labels,
    )
    recomputed = evaluate_validation_scores(table.label, table.score)
    for key in (
        "Validation_AUPRC",
        "Validation_AUROC",
        "Validation_AveragePrecision",
        "Validation_Selected_Threshold",
        "Validation_F1",
        "Validation_Precision",
        "Validation_Recall",
        "Validation_Accuracy",
    ):
        if not _close(metrics.get(key), recomputed.get(key)):
            raise TCMHparamSelectionError(
                f"{candidate_id} fold {fold} {key} cannot be reproduced"
            )
    for key in (
        "Validation_AUPRC_Definition",
        "Validation_AveragePrecision_Definition",
        "Threshold_Candidates",
        "Threshold_Tie_Break",
    ):
        if metrics.get(key) != recomputed.get(key):
            raise TCMHparamSelectionError(
                f"{candidate_id} fold {fold} {key} differs from frozen metrics"
            )
    if (
        recomputed["Validation_AUPRC_Definition"] != AUPRC_DEFINITION
        or recomputed["Validation_AveragePrecision_Definition"]
        != AVERAGE_PRECISION_DEFINITION
    ):
        raise TCMHparamSelectionError("validation metric definition differs")

    herb_path = _single_embedding(candidate_root, "herbedgeDHCN_herb_embedding")
    disease_path = _single_embedding(
        candidate_root, "diseaseedgeDHCN_disease_embedding"
    )
    return {
        "id": candidate_id,
        "fold": fold,
        "epochs": int(candidate["epochs"]),
        "learning_rate": float(candidate["learning_rate"]),
        "semantic_adapter": str(candidate["semantic_adapter"]),
        "validation_auprc": float(recomputed["Validation_AUPRC"]),
        "validation_auroc": float(recomputed["Validation_AUROC"]),
        "validation_f1": float(recomputed["Validation_F1"]),
        "validation_threshold": float(recomputed["Validation_Selected_Threshold"]),
        "validation_predictions": str(validation_path.resolve()),
        "validation_predictions_sha256": _sha256(validation_path),
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "validation_row_id_sha256": _canonical_array_sha256(table.row_id),
        "validation_pair_sha256": _canonical_array_sha256(table.pair),
        "validation_label_sha256": _canonical_array_sha256(table.label),
        "herb_embedding": str(herb_path),
        "herb_embedding_sha256": _sha256(herb_path),
        "disease_embedding": str(disease_path),
        "disease_embedding_sha256": _sha256(disease_path),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "candidate_runner_sha256": candidate_runner_sha256,
        "adapter_sources_sha256": adapter_sources_sha256,
        "metrics_source_sha256": _sha256(EXPECTED_ADAPTER_SOURCES["metrics"]),
        "model_sources_sha256": model_sources_sha256,
        "source_inputs_sha256": source_inputs_sha256,
        "feature_bundle_sha256": str(
            feature_bundle.get("sha256")
        ),
        "frozen_input_lock_sha256": expected_input_lock,
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_metric_definition": AUPRC_DEFINITION,
    }


def _build_selection_payload(
    *, context: PlanContext, candidate_root: Path
) -> dict[str, Any]:
    """Audit all validation-only candidates and deterministically replay selection."""

    candidate_root = candidate_root.resolve()
    selected_folds: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for fold in range(1, context.protocol.n_folds + 1):
        validation = _derive_training_validation_rows(context.protocol, fold)
        records = [
            _validate_candidate(
                root=candidate_root,
                fold=fold,
                candidate=candidate,
                context=context,
                validation=validation,
            )
            for candidate in context.candidates
        ]
        all_records.extend(records)
        identity_hashes = {
            (
                record["validation_row_id_sha256"],
                record["validation_pair_sha256"],
                record["validation_label_sha256"],
            )
            for record in records
        }
        if len(identity_hashes) != 1:
            raise TCMHparamSelectionError(
                f"fold {fold} candidates used different validation rows"
            )
        winner = min(
            records,
            key=lambda record: (
                -record["validation_auprc"],
                -record["validation_auroc"],
                record["epochs"],
                record["learning_rate"],
                record["id"],
            ),
        )
        selected_folds[str(fold)] = {
            "selected_candidate": winner["id"],
            "selected_validation_auprc": winner["validation_auprc"],
            "selected_validation_auroc": winner["validation_auroc"],
            "selected_validation_threshold": winner["validation_threshold"],
            "winner": winner,
            "candidates": records,
        }

    for field in (
        "candidate_runner_sha256",
        "adapter_sources_sha256",
        "metrics_source_sha256",
        "model_sources_sha256",
        "source_inputs_sha256",
        "feature_bundle_sha256",
        "frozen_input_lock_sha256",
        "selection_metric",
        "selection_metric_definition",
        "array_hash_schema",
    ):
        observed = {record[field] for record in all_records}
        if len(observed) != 1 or not next(iter(observed)):
            raise TCMHparamSelectionError(
                f"candidates do not share one non-empty {field}: {sorted(observed)}"
            )

    payload = {
        "schema": SELECTION_SCHEMA,
        "dataset": "tcm-suite",
        "seed": int(context.protocol.seed),
        "plan": str(context.path),
        "plan_sha256": context.sha256,
        "candidate_root": str(candidate_root),
        "selector_path": str(Path(__file__).resolve()),
        "selector_sha256": _sha256(Path(__file__).resolve()),
        "selection_metric": EXPECTED_SELECTION_METRIC,
        "selection_metric_definition": AUPRC_DEFINITION,
        "metrics_source_sha256": _sha256(EXPECTED_ADAPTER_SOURCES["metrics"]),
        "array_hash_schema": ARRAY_HASH_SCHEMA,
        "tie_break": EXPECTED_TIE_BREAK,
        "outer_test_accessed": False,
        "outer_test_scored": False,
        "outer_test_predictions_materialised": False,
        "interpretation": "seed-42 same-split post-baseline development diagnostic",
        "folds": selected_folds,
    }
    return payload


def select_candidates(*, plan_path: Path, candidate_root: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    receipt_path = _selection_receipt_path(output)
    existing = [path for path in (output, receipt_path) if path.exists()]
    if existing:
        raise TCMHparamSelectionError(
            "refusing to overwrite selection artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    context = _load_plan_context(plan_path)
    payload = _build_selection_payload(
        context=context,
        candidate_root=candidate_root,
    )
    _write_json(output, payload)
    selection_sha256 = _sha256(output)
    receipt = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "selection": str(output),
        "selection_sha256": selection_sha256,
        "selector": str(Path(__file__).resolve()),
        "selector_sha256": _sha256(Path(__file__).resolve()),
        "plan": str(context.path),
        "plan_sha256": context.sha256,
        "candidate_root": str(candidate_root.resolve()),
        "outer_test_scored_before_receipt": False,
    }
    _write_json(receipt_path, receipt)
    return payload


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _embedding_scores(herb: np.ndarray, disease: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    logits = np.einsum("ij,ij->i", herb[pairs[:, 0]], disease[pairs[:, 1]])
    scores = _stable_sigmoid(logits)
    if not np.isfinite(scores).all():
        raise TCMHparamSelectionError("selected embedding emitted non-finite scores")
    return scores


def _summary(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("AUROC", "AUPRC", "F1", "Precision", "Recall", "Accuracy"):
        values = np.asarray([float(record[metric]) for record in fold_metrics])
        result[metric] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "folds": int(len(values)),
        }
    return result


def _validated_selection_replay(
    selection_path: Path,
) -> tuple[
    Mapping[str, Any],
    str,
    Path,
    str,
    PlanContext,
]:
    """Verify the receipt and replay validation selection before test access."""

    selection_path = selection_path.resolve()
    selection = _load_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise TCMHparamSelectionError("unsupported selection schema")
    for key in (
        "outer_test_accessed",
        "outer_test_scored",
        "outer_test_predictions_materialised",
    ):
        if selection.get(key) is not False:
            raise TCMHparamSelectionError(
                f"selection {key} was not false before test"
            )
    selector_path = Path(str(selection.get("selector_path", ""))).resolve()
    current_selector = Path(__file__).resolve()
    current_selector_sha256 = _sha256(current_selector)
    if (
        selector_path != current_selector
        or selection.get("selector_sha256") != current_selector_sha256
    ):
        raise TCMHparamSelectionError("selection source changed after freezing")
    if (
        selection.get("selection_metric") != EXPECTED_SELECTION_METRIC
        or selection.get("selection_metric_definition") != AUPRC_DEFINITION
        or selection.get("metrics_source_sha256")
        != _sha256(EXPECTED_ADAPTER_SOURCES["metrics"])
        or selection.get("array_hash_schema") != ARRAY_HASH_SCHEMA
        or selection.get("tie_break") != EXPECTED_TIE_BREAK
    ):
        raise TCMHparamSelectionError("selection metric/tie-break contract differs")

    selection_sha256 = _sha256(selection_path)
    receipt_path = _selection_receipt_path(selection_path)
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != SELECTION_RECEIPT_SCHEMA:
        raise TCMHparamSelectionError("selection receipt schema differs")
    expected_receipt = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "selection": str(selection_path),
        "selection_sha256": selection_sha256,
        "selector": str(current_selector),
        "selector_sha256": current_selector_sha256,
        "plan": str(Path(str(selection.get("plan", ""))).resolve()),
        "plan_sha256": selection.get("plan_sha256"),
        "candidate_root": str(
            Path(str(selection.get("candidate_root", ""))).resolve()
        ),
        "outer_test_scored_before_receipt": False,
    }
    if dict(receipt) != expected_receipt:
        raise TCMHparamSelectionError("selection receipt does not bind this decision")
    receipt_sha256 = _sha256(receipt_path)

    context = _load_plan_context(Path(str(selection.get("plan", ""))))
    if context.sha256 != selection.get("plan_sha256"):
        raise TCMHparamSelectionError("selection plan changed after freezing")
    replay = _build_selection_payload(
        context=context,
        candidate_root=Path(str(selection.get("candidate_root", ""))),
    )
    if dict(selection) != replay:
        raise TCMHparamSelectionError(
            "selection file differs from deterministic validation-only replay"
        )
    return replay, selection_sha256, receipt_path, receipt_sha256, context


def evaluate_selected(*, selection_path: Path, output_root: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise TCMHparamSelectionError(
            f"refusing to overwrite non-empty outer-test directory: {output_root}"
        )
    (
        selection,
        selection_sha256,
        receipt_path,
        receipt_sha256,
        context,
    ) = _validated_selection_replay(selection_path)

    fold_metrics: list[dict[str, Any]] = []
    selected_ids: dict[str, str] = {}
    for fold in range(1, context.protocol.n_folds + 1):
        fold_record = selection.get("folds", {}).get(str(fold))
        if not isinstance(fold_record, Mapping):
            raise TCMHparamSelectionError(f"selection is missing fold {fold}")
        winner = fold_record.get("winner")
        if not isinstance(winner, Mapping):
            raise TCMHparamSelectionError(f"selection winner is missing for fold {fold}")
        herb_path = Path(str(winner["herb_embedding"])).resolve()
        disease_path = Path(str(winner["disease_embedding"])).resolve()
        validation_path = Path(str(winner["validation_predictions"])).resolve()
        if _sha256(herb_path) != winner.get("herb_embedding_sha256"):
            raise TCMHparamSelectionError("selected herb embedding changed")
        if _sha256(disease_path) != winner.get("disease_embedding_sha256"):
            raise TCMHparamSelectionError("selected disease embedding changed")
        if _sha256(validation_path) != winner.get("validation_predictions_sha256"):
            raise TCMHparamSelectionError("selected validation prediction changed")

        herb = np.asarray(np.loadtxt(herb_path), dtype=np.float64)
        disease = np.asarray(np.loadtxt(disease_path), dtype=np.float64)
        if herb.shape != (context.protocol.n_herbs, 64):
            raise TCMHparamSelectionError(f"fold {fold} herb embedding shape is {herb.shape}")
        if disease.shape != (context.protocol.n_diseases, 64):
            raise TCMHparamSelectionError(
                f"fold {fold} disease embedding shape is {disease.shape}"
            )
        validation = _derive_training_validation_rows(context.protocol, fold)
        validation_table = validate_prediction_coverage(
            validation_path,
            expected_row_id=validation.validation_row_indices,
            expected_pair=validation.validation_pairs,
            expected_label=validation.validation_labels,
        )
        if (
            _canonical_array_sha256(validation_table.row_id)
            != winner.get("validation_row_id_sha256")
            or _canonical_array_sha256(validation_table.pair)
            != winner.get("validation_pair_sha256")
            or _canonical_array_sha256(validation_table.label)
            != winner.get("validation_label_sha256")
        ):
            raise TCMHparamSelectionError(
                f"fold {fold} validation identity differs from selection"
            )
        reproduced_validation = _embedding_scores(
            herb, disease, validation_table.pair
        )
        if not np.allclose(
            reproduced_validation,
            validation_table.score,
            rtol=0.0,
            atol=1e-12,
        ):
            difference = float(
                np.max(np.abs(reproduced_validation - validation_table.score))
            )
            raise TCMHparamSelectionError(
                f"fold {fold} selected embeddings do not reproduce validation "
                f"scores; max diff={difference}"
            )
        validation_metrics = evaluate_validation_scores(
            validation_table.label, validation_table.score
        )
        if (
            not _close(
                validation_metrics["Validation_AUPRC"],
                winner.get("validation_auprc"),
            )
            or not _close(
                validation_metrics["Validation_AUROC"],
                winner.get("validation_auroc"),
            )
            or not _close(
                validation_metrics["Validation_Selected_Threshold"],
                winner.get("validation_threshold"),
            )
        ):
            raise TCMHparamSelectionError(
                f"fold {fold} validation metrics differ from selected winner"
            )

        # This is deliberately the first explicit retrieval of outer-test rows
        # after the receipt and validation-only selection replay have passed.
        rows = context.protocol.fold_rows(fold)
        test_pair = np.asarray(rows.test_pairs, dtype=np.int64)
        test_label = np.asarray(rows.test_labels, dtype=np.uint8)
        test_row_id = np.asarray(rows.test_row_indices, dtype=np.int64)
        test_score = _embedding_scores(herb, disease, test_pair)
        metrics = evaluate_validation_and_test(
            validation_table.label,
            validation_table.score,
            test_label,
            test_score,
        )
        if metrics.get("AUPRC_Definition") != AUPRC_DEFINITION:
            raise TCMHparamSelectionError("outer-test AUPRC definition changed")
        fold_dir = output_root / f"fold_{fold}"
        test_path = save_predictions(
            fold_dir / "test_predictions.npz",
            row_id=test_row_id,
            pair=test_pair,
            label=test_label,
            score=test_score,
        )
        validate_prediction_coverage(
            test_path,
            expected_row_id=test_row_id,
            expected_pair=test_pair,
            expected_label=test_label,
        )
        record = {
            "Dataset": "tcm-suite",
            "Method": "ASTRA-inner-selected",
            "Fold": fold,
            "Seed": int(context.protocol.seed),
            "Diagnostic": True,
            "Formal_Eligible": False,
            "Decision_Frozen_Before_Test": True,
            "Test_Models_Evaluated": 1,
            "Unselected_Test_Models_Evaluated": 0,
            "Formal_Ineligibility_Reasons": [
                "seed-42 outer tests were known before this development stage"
            ],
            "Selected_Candidate": str(winner["id"]),
            "Selected_Epochs": int(winner["epochs"]),
            "Selected_Learning_Rate": float(winner["learning_rate"]),
            "Selected_Validation_AUPRC": float(winner["validation_auprc"]),
            "Selection_Manifest": str(selection_path),
            "Selection_Manifest_SHA256": selection_sha256,
            "Selection_Receipt": str(receipt_path),
            "Selection_Receipt_SHA256": receipt_sha256,
            "Validation_Predictions": str(validation_path),
            "Test_Predictions": str(test_path),
            **metrics,
        }
        _write_json(fold_dir / "metrics.json", record)
        _write_json(
            fold_dir / "selection_binding.json",
            {
                "selection_manifest": str(selection_path),
                "selection_manifest_sha256": selection_sha256,
                "selection_receipt": str(receipt_path),
                "selection_receipt_sha256": receipt_sha256,
                "selector_sha256": selection["selector_sha256"],
                "metrics_source_sha256": selection["metrics_source_sha256"],
                "fold": fold,
                "winner": winner,
                "test_predictions_sha256": _sha256(test_path),
                "validation_reproduced_from_selected_embedding": True,
                "test_models_evaluated": 1,
                "unselected_test_models_evaluated": 0,
            },
        )
        fold_metrics.append(record)
        selected_ids[str(fold)] = str(winner["id"])

    payload = {
        "schema": "astra-tcm-hparam-outer-test-summary/v1",
        "status": "diagnostic-complete",
        "formal_result": False,
        "dataset": "tcm-suite",
        "method": "ASTRA-inner-selected",
        "seed": int(context.protocol.seed),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "selection_receipt": str(receipt_path),
        "selection_receipt_sha256": receipt_sha256,
        "selected_candidates": selected_ids,
        "fold_metrics": fold_metrics,
        "summary": _summary(fold_metrics),
        "interpretation": (
            "Same-split post-baseline development diagnostic. Candidate choice "
            "and thresholds use only inner validation, but independent confirmation "
            "requires a previously unseen frozen split."
        ),
    }
    _write_json(output_root / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    selector = subparsers.add_parser("select")
    selector.add_argument("--plan", type=Path, required=True)
    selector.add_argument("--candidate-root", type=Path, required=True)
    selector.add_argument("--output", type=Path, required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--selection", type=Path, required=True)
    evaluator.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "select":
            payload = select_candidates(
                plan_path=args.plan,
                candidate_root=args.candidate_root,
                output=args.output,
            )
        else:
            payload = evaluate_selected(
                selection_path=args.selection,
                output_root=args.output_root,
            )
    except Exception as error:
        print(f"TCM ASTRA selection failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
