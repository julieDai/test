#!/usr/bin/env python3
"""Audit and aggregate protocol-locked predictions from official adapters.

The audit is independent of every model runner. It reloads the frozen split
manifest, checks row/pair/label coverage, recomputes all metrics, rejects
identical cross-method score arrays, and reports only test metrics whose
operating threshold was selected on the validation split.  An explicit
``--legacy-test-oracle`` mode independently reproduces the manuscript's
retained row-based protocol and labels its threshold-dependent values as
descriptive test-oracle diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from astra.benchmark.metrics import (
    AUPRC_DEFINITION,
    AVERAGE_PRECISION_DEFINITION,
    BenchmarkAuditError,
    PredictionTable,
    aggregate_fold_metrics,
    audit_cross_method_predictions,
    audit_milgnet_run,
    evaluate_test_scores,
    evaluate_test_oracle,
    evaluate_validation_and_test,
    load_predictions,
    validate_prediction_coverage,
)
from astra.benchmark.protocol import load_protocol_manifest

from .common import (
    fold_split,
    legacy_protocol_provenance,
    project_git_identity,
    sha256,
    validate_frozen_input_manifest,
    write_json,
)


DEFAULT_METHODS = ("SMGCL", "MilGNet", "LHGCE")
RECOMPUTED_KEYS = (
    "Validation_Selected_Threshold",
    "Validation_F1",
    "Validation_Precision",
    "Validation_Recall",
    "Validation_Accuracy",
    "Threshold_Candidates",
    "AUROC",
    "Operating_Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
)
LEGACY_RECOMPUTED_KEYS = (
    "Threshold_Candidates",
    "AUROC",
    "Operating_Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Test_Oracle_Threshold",
    "Test_Oracle_Accuracy",
    "Test_Oracle_Precision",
    "Test_Oracle_Recall",
    "Test_Oracle_F1",
)


def _metric_mismatches(
    *,
    recorded: Mapping[str, object],
    recomputed: Mapping[str, object],
    keys: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Compare metrics while accepting historical AP-labelled-as-AUPRC files.

    Runs produced before the paper-metric harmonization stored sklearn average
    precision in ``AUPRC`` and did not record a definition.  Their continuous
    predictions remain auditable: verify that legacy value against the newly
    recomputed ``AveragePrecision`` and expose trapezoidal PR-AUC as the
    canonical manuscript-facing ``AUPRC`` in the new audit output.
    """

    mismatches = {
        name: {"recorded": recorded.get(name), "recomputed": recomputed.get(name)}
        for name in keys
        if name not in recorded or not _same_metric(recorded.get(name), recomputed.get(name))
    }
    recorded_definition = str(recorded.get("AUPRC_Definition", "")).lower()
    recorded_is_trapezoidal = "trapezoid" in recorded_definition or "auc(recall" in recorded_definition
    comparison_name = "AUPRC" if recorded_is_trapezoidal else "AveragePrecision"
    if "AUPRC" not in recorded or not _same_metric(
        recorded.get("AUPRC"), recomputed.get(comparison_name)
    ):
        mismatches["AUPRC"] = {
            "recorded": recorded.get("AUPRC"),
            "recorded_definition": (
                recorded.get("AUPRC_Definition")
                if recorded_is_trapezoidal
                else "legacy implicit sklearn.metrics.average_precision_score"
            ),
            "recomputed": recomputed.get(comparison_name),
            "recomputed_metric": comparison_name,
        }
    if "AveragePrecision" in recorded and not _same_metric(
        recorded.get("AveragePrecision"), recomputed.get("AveragePrecision")
    ):
        mismatches["AveragePrecision"] = {
            "recorded": recorded.get("AveragePrecision"),
            "recomputed": recomputed.get("AveragePrecision"),
        }
    return mismatches


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkAuditError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkAuditError(f"JSON artifact is not an object: {path}")
    return value


def _key(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _subset_coverage(
    table: PredictionTable,
    *,
    expected_row_id: np.ndarray,
    expected_pair: np.ndarray,
    expected_label: np.ndarray,
) -> PredictionTable:
    """Validate a smoke subset against the frozen manifest and align it."""

    expected_positions = {_key(value): index for index, value in enumerate(expected_row_id)}
    actual_keys = [_key(value) for value in table.row_id]
    if len(set(actual_keys)) != len(actual_keys):
        raise BenchmarkAuditError("smoke prediction row_id values are not unique")
    outside = [value for value in actual_keys if value not in expected_positions]
    if outside:
        raise BenchmarkAuditError(
            f"smoke predictions contain {len(outside)} rows outside the frozen split"
        )
    order = np.argsort(
        np.asarray([expected_positions[value] for value in actual_keys], dtype=np.int64)
    )
    aligned = PredictionTable(
        row_id=table.row_id[order],
        pair=table.pair[order],
        label=table.label[order],
        score=table.score[order],
    )
    positions = np.asarray(
        [expected_positions[_key(value)] for value in aligned.row_id], dtype=np.int64
    )
    if not np.array_equal(aligned.pair, expected_pair[positions]):
        raise BenchmarkAuditError("smoke prediction pairs differ from the frozen split")
    if not np.array_equal(aligned.label, expected_label[positions]):
        raise BenchmarkAuditError("smoke prediction labels differ from the frozen split")
    if np.unique(aligned.label).size != 2:
        raise BenchmarkAuditError("smoke predictions must contain both label classes")
    return aligned


def _coverage(
    path: Path,
    *,
    expected_row_id: np.ndarray,
    expected_pair: np.ndarray,
    expected_label: np.ndarray,
    allow_subset: bool,
) -> PredictionTable:
    table = load_predictions(path)
    if allow_subset:
        return _subset_coverage(
            table,
            expected_row_id=expected_row_id,
            expected_pair=expected_pair,
            expected_label=expected_label,
        )
    return validate_prediction_coverage(
        table,
        expected_row_id=expected_row_id,
        expected_pair=expected_pair,
        expected_label=expected_label,
    )


def _same_metric(recorded: object, recomputed: object) -> bool:
    if isinstance(recomputed, (float, np.floating)):
        try:
            return bool(
                np.isclose(float(recorded), float(recomputed), rtol=0.0, atol=1e-12)
            )
        except (TypeError, ValueError):
            return False
    return recorded == recomputed


def _strict_json_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise BenchmarkAuditError(f"{label} must be a JSON integer{suffix}")
    return value


def _validate_execution_mode(
    *,
    recorded: Mapping[str, object],
    run_protocol: Mapping[str, object],
    allow_smoke: bool,
    allow_diagnostic: bool,
) -> tuple[bool, bool]:
    """Fail closed on attempts to promote smoke/diagnostic artifacts.

    Diagnostic runs use complete folds and full training budgets, so row
    coverage alone cannot distinguish them from formal candidates.  Their
    explicit mode bit, ineligibility bit, and reasons must therefore agree in
    both independently read artifacts.
    """

    if allow_smoke and allow_diagnostic:
        raise BenchmarkAuditError(
            "--allow-smoke and --allow-diagnostic are mutually exclusive"
        )
    smoke_value = recorded.get("Smoke")
    if type(smoke_value) is not bool:
        raise BenchmarkAuditError("metrics Smoke must be a JSON boolean")
    diagnostic_value = recorded.get("Diagnostic", False)
    if type(diagnostic_value) is not bool:
        raise BenchmarkAuditError("metrics Diagnostic must be a JSON boolean")
    is_smoke = bool(smoke_value)
    is_diagnostic = bool(diagnostic_value)
    if is_smoke and is_diagnostic:
        raise BenchmarkAuditError(
            "a run cannot be both smoke and full-budget diagnostic"
        )

    protocol_smoke = run_protocol.get("smoke", is_smoke)
    protocol_diagnostic = run_protocol.get("diagnostic", False)
    if type(protocol_smoke) is not bool or type(protocol_diagnostic) is not bool:
        raise BenchmarkAuditError(
            "protocol smoke/diagnostic fields must be JSON booleans"
        )
    if protocol_smoke != is_smoke or protocol_diagnostic != is_diagnostic:
        raise BenchmarkAuditError(
            "metrics and protocol execution-mode flags disagree"
        )
    expected_mode = (
        "smoke" if is_smoke else "diagnostic" if is_diagnostic else "formal-candidate"
    )
    metric_mode = recorded.get("Execution_Mode")
    protocol_mode = run_protocol.get("execution_mode")
    if is_diagnostic and (
        metric_mode != expected_mode or protocol_mode != expected_mode
    ):
        raise BenchmarkAuditError(
            "diagnostic artifacts must explicitly record execution_mode=diagnostic"
        )
    if metric_mode is not None and metric_mode != expected_mode:
        raise BenchmarkAuditError("metrics Execution_Mode disagrees with mode flags")
    if protocol_mode is not None and protocol_mode != expected_mode:
        raise BenchmarkAuditError("protocol execution_mode disagrees with mode flags")

    if is_smoke:
        if not allow_smoke:
            raise BenchmarkAuditError(
                "smoke run cannot enter a formal aggregate; use --allow-smoke"
            )
    elif is_diagnostic:
        if not allow_diagnostic:
            raise BenchmarkAuditError(
                "diagnostic run cannot enter a formal aggregate; use "
                "--allow-diagnostic for a diagnostic-only audit"
            )
    elif allow_smoke or allow_diagnostic:
        requested = "smoke" if allow_smoke else "diagnostic"
        raise BenchmarkAuditError(
            f"--allow-{requested} cannot relabel a non-{requested} run"
        )

    formal_value = recorded.get("Formal_Eligible")
    if formal_value is not None and type(formal_value) is not bool:
        raise BenchmarkAuditError("metrics Formal_Eligible must be a JSON boolean")
    protocol_formal = run_protocol.get("formal_eligible")
    if protocol_formal is not None and type(protocol_formal) is not bool:
        raise BenchmarkAuditError("protocol formal_eligible must be a JSON boolean")
    if is_diagnostic:
        if formal_value is not False or protocol_formal is not False:
            raise BenchmarkAuditError(
                "diagnostic artifacts must be explicitly formally ineligible"
            )
        metric_reasons = recorded.get("Formal_Ineligibility_Reasons")
        protocol_reasons = run_protocol.get("formal_ineligibility_reasons")
        if (
            not isinstance(metric_reasons, list)
            or not metric_reasons
            or not all(isinstance(reason, str) and reason for reason in metric_reasons)
            or protocol_reasons != metric_reasons
        ):
            raise BenchmarkAuditError(
                "diagnostic metrics/protocol must contain identical non-empty "
                "formal-ineligibility reasons"
            )
    elif not is_smoke and formal_value is not True:
        raise BenchmarkAuditError(
            "non-smoke, non-diagnostic run is not formally eligible: "
            f"{recorded.get('Feature_Integrity_Warnings', [])}"
        )
    return is_smoke, is_diagnostic


def _assert_diagnostic_full_budget(
    *,
    recorded: Mapping[str, object],
    run_protocol: Mapping[str, object],
    method: str,
    dataset: str,
    fold_rows=None,
) -> None:
    """Require the evidence that distinguishes a diagnostic from a smoke run."""

    method_key = method.strip().lower()
    if method_key not in {"astra", "hghda", "smgcl"}:
        raise BenchmarkAuditError(
            "diagnostic audit supports only ASTRA, HGHDA, and SMGCL artifacts"
        )
    if method_key == "smgcl":
        if fold_rows is None:
            raise BenchmarkAuditError("SMGCL diagnostic audit lacks frozen fold rows")
        expected_train = len(fold_rows.train_pairs)
        expected_positive_graph = int(np.count_nonzero(fold_rows.train_labels == 1))
        training_rows = _strict_json_int(
            recorded.get("Training_Rows"),
            "SMGCL diagnostic Training_Rows",
            minimum=1,
        )
        if training_rows != expected_train:
            raise BenchmarkAuditError(
                "SMGCL diagnostic training rows differ from the complete "
                f"frozen inner-training fold: {training_rows} != {expected_train}"
            )
        positive_graph_rows = _strict_json_int(
            recorded.get("Training_Positive_Graph_Rows"),
            "SMGCL diagnostic Training_Positive_Graph_Rows",
            minimum=1,
        )
        if positive_graph_rows != expected_positive_graph:
            raise BenchmarkAuditError(
                "SMGCL diagnostic training-positive graph differs from the "
                "frozen inner-training positives"
            )
        epochs = _strict_json_int(
            recorded.get("Epochs_Requested"),
            "SMGCL diagnostic Epochs_Requested",
            minimum=1,
        )
        if (
            _strict_json_int(
                recorded.get("Epochs_Executed"),
                "SMGCL diagnostic Epochs_Executed",
                minimum=1,
            )
            != epochs
        ):
            raise BenchmarkAuditError(
                "SMGCL diagnostic metrics disagree with the requested epoch budget"
            )
        batch_size = _strict_json_int(
            recorded.get("Batch_Size"), "SMGCL diagnostic Batch_Size", minimum=1
        )
        expected_steps = epochs * int(math.ceil(expected_train / batch_size))
        if _strict_json_int(
            recorded.get("Optimizer_Steps"),
            "diagnostic Optimizer_Steps",
            minimum=1,
        ) != expected_steps:
            raise BenchmarkAuditError(
                "SMGCL diagnostic optimizer steps differ from the complete "
                "requested budget"
            )
        return
    split_rows = run_protocol.get("split_rows")
    if not isinstance(split_rows, Mapping):
        raise BenchmarkAuditError("diagnostic protocol lacks split_rows provenance")
    prefixes = ("train", "validation", "test")
    if method_key == "hghda":
        prefixes = ("inner_train", "validation", "outer_test")
    used_rows: dict[str, int] = {}
    for prefix in prefixes:
        full = _strict_json_int(
            split_rows.get(f"{prefix}_full"),
            f"diagnostic split_rows {prefix}_full",
            minimum=1,
        )
        used = _strict_json_int(
            split_rows.get(f"{prefix}_used"),
            f"diagnostic split_rows {prefix}_used",
            minimum=1,
        )
        if used != full:
            raise BenchmarkAuditError(
                f"diagnostic {prefix} rows are truncated: {used} != {full}"
            )
        used_rows[prefix] = used

    hyperparameters = run_protocol.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise BenchmarkAuditError("diagnostic protocol lacks hyperparameter provenance")
    if method_key == "astra":
        expected_epochs = {"ethnobotany": 30, "tcm-suite": 16}.get(dataset)
        epochs = _strict_json_int(
            hyperparameters.get("epochs"), "ASTRA diagnostic epochs", minimum=2
        )
        formal_epochs = _strict_json_int(
            hyperparameters.get("formal_epochs"),
            "ASTRA diagnostic formal_epochs",
            minimum=2,
        )
        if (
            expected_epochs is None
            or epochs != expected_epochs
            or formal_epochs != expected_epochs
        ):
            raise BenchmarkAuditError(
                "ASTRA diagnostic epoch budget differs from the normal dataset schedule"
            )
        batch_size = _strict_json_int(
            hyperparameters.get("batch_size"), "ASTRA diagnostic batch_size", minimum=1
        )
        expected_steps = epochs * int(math.ceil(used_rows["train"] / batch_size))
    else:
        epochs = _strict_json_int(
            hyperparameters.get("epochs"), "HGHDA diagnostic epochs", minimum=2
        )
        if epochs not in {4, 16}:
            raise BenchmarkAuditError("HGHDA diagnostic epoch budget is not predeclared")
        if (
            _strict_json_int(
                recorded.get("Epochs_Requested"),
                "HGHDA diagnostic Epochs_Requested",
                minimum=2,
            )
            != epochs
            or _strict_json_int(
                recorded.get("Epochs_Executed"),
                "HGHDA diagnostic Epochs_Executed",
                minimum=2,
            )
            != epochs
        ):
            raise BenchmarkAuditError(
                "HGHDA diagnostic metrics disagree with the full epoch budget"
            )
        batch_size = _strict_json_int(
            hyperparameters.get("batch_size"), "HGHDA diagnostic batch_size", minimum=1
        )
        expected_steps = epochs * int(
            math.ceil(used_rows["inner_train"] / batch_size)
        )
    if _strict_json_int(
        recorded.get("Optimizer_Steps"), "diagnostic Optimizer_Steps", minimum=1
    ) != expected_steps:
        raise BenchmarkAuditError(
            "diagnostic optimizer steps differ from the complete normal budget"
        )


def _assert_formal_tcm_budget(
    *,
    recorded: Mapping[str, object],
    run_protocol: Mapping[str, object],
    method: str,
    fold_rows,
    task: Path | None = None,
) -> None:
    """Lock the predeclared strict TCM budgets in the independent audit.

    General metric recomputation proves score/threshold correctness, but it
    cannot establish how much optimization produced those scores.  These
    checks bind each strict main-comparison artifact to its declared schedule.
    """

    method_key = method.strip().lower()
    supported = {"smgcl", "milgnet", "lhgce", "hghda"}
    if method_key not in supported:
        return

    train_rows = len(fold_rows.train_pairs)
    test_rows = len(fold_rows.test_pairs)

    def require_metric_int(name: str, expected: int) -> None:
        actual = _strict_json_int(
            recorded.get(name), f"{method} strict TCM {name}", minimum=1
        )
        if actual != expected:
            raise BenchmarkAuditError(
                f"{method} strict TCM {name} differs from the locked budget: "
                f"{actual} != {expected}"
            )

    if method_key == "smgcl":
        require_metric_int("Epochs_Requested", 64)
        require_metric_int("Epochs_Executed", 64)
        require_metric_int("Batch_Size", 4096)
        require_metric_int("Training_Rows", train_rows)
        require_metric_int(
            "Optimizer_Steps", 64 * int(math.ceil(train_rows / 4096))
        )
        if recorded.get("Scalable_Mode") is not True:
            raise BenchmarkAuditError(
                "SMGCL strict TCM artifact did not enable scalable mode"
            )
        if recorded.get("Scalable_Mode_Request") != "on":
            raise BenchmarkAuditError(
                "SMGCL strict TCM artifact did not explicitly request scalable mode"
            )
        selection_mode = recorded.get("Epoch_Selection_Mode", "fixed-budget")
        if selection_mode == "inner-validation":
            if task is None:
                raise BenchmarkAuditError(
                    "SMGCL inner-validation selection audit requires its task directory"
                )
            candidates = [16, 30, 48, 64]
            if recorded.get("Epoch_Candidates") != candidates:
                raise BenchmarkAuditError(
                    "SMGCL inner-validation epoch candidates differ from 16/30/48/64"
                )
            records = recorded.get("Checkpoint_Validation_Metrics")
            if not isinstance(records, list) or len(records) != len(candidates):
                raise BenchmarkAuditError(
                    "SMGCL inner-validation checkpoint metrics are incomplete"
                )
            expected_records: dict[int, Mapping[str, object]] = {}
            for item in records:
                if not isinstance(item, Mapping):
                    raise BenchmarkAuditError(
                        "SMGCL checkpoint validation record is not an object"
                    )
                epoch = _strict_json_int(
                    item.get("Epoch"), "SMGCL checkpoint Epoch", minimum=1
                )
                if epoch in expected_records or epoch not in candidates:
                    raise BenchmarkAuditError(
                        "SMGCL checkpoint validation epochs are duplicated or unexpected"
                    )
                expected_records[epoch] = item
            if sorted(expected_records) != candidates:
                raise BenchmarkAuditError(
                    "SMGCL checkpoint validation epochs are incomplete"
                )

            validation_metrics: dict[int, dict[str, object]] = {}
            validation_tables: dict[int, PredictionTable] = {}
            validation_pairs, validation_labels, validation_row_id = fold_split(
                fold_rows, "validation"
            )
            for epoch in candidates:
                path = task / "results" / "validation_checkpoints" / f"epoch_{epoch}.npz"
                if not path.is_file():
                    raise BenchmarkAuditError(
                        f"SMGCL checkpoint validation artifact is missing: {path}"
                    )
                table = _coverage(
                    path,
                    expected_row_id=validation_row_id,
                    expected_pair=validation_pairs,
                    expected_label=validation_labels,
                    allow_subset=False,
                )
                validation_tables[epoch] = table
                recomputed = evaluate_test_scores(
                    table.label, table.score, threshold=0.5
                )
                validation_metrics[epoch] = recomputed
                item = expected_records[epoch]
                comparisons = {
                    "Validation_AUROC": recomputed["AUROC"],
                    "Validation_AUPRC": recomputed["AUPRC"],
                    "Validation_AveragePrecision": recomputed["AveragePrecision"],
                    "Validation_Predictions_SHA256": sha256(path),
                    "Validation_Score_SHA256": hashlib.sha256(
                        np.ascontiguousarray(table.score, dtype=np.float32).view(
                            np.uint8
                        )
                    ).hexdigest(),
                }
                expected_relative_path = (
                    f"results/validation_checkpoints/epoch_{epoch}.npz"
                )
                if item.get("Validation_Predictions") != expected_relative_path:
                    raise BenchmarkAuditError(
                        f"SMGCL checkpoint epoch {epoch} validation artifact path "
                        "differs from the locked layout"
                    )
                for name, expected_value in comparisons.items():
                    actual = item.get(name)
                    equal = (
                        _same_metric(actual, expected_value)
                        if isinstance(expected_value, float)
                        else actual == expected_value
                    )
                    if not equal:
                        raise BenchmarkAuditError(
                            f"SMGCL checkpoint epoch {epoch} {name} differs from "
                            "independent validation recomputation"
                        )

            selected_epoch = min(
                candidates,
                key=lambda epoch: (
                    -float(validation_metrics[epoch]["AUPRC"]),
                    epoch,
                ),
            )
            require_metric_int("Selected_Epoch", selected_epoch)
            require_metric_int(
                "Selected_Checkpoint_Optimizer_Steps",
                selected_epoch * int(math.ceil(train_rows / 4096)),
            )
            if recorded.get("Epoch_Selection_Metric") != (
                "validation trapezoidal AUPRC"
            ):
                raise BenchmarkAuditError(
                    "SMGCL checkpoint selection metric is not validation AUPRC"
                )
            if recorded.get("Epoch_Selection_Tie_Break") != (
                "lowest epoch on an exact AUPRC tie"
            ):
                raise BenchmarkAuditError(
                    "SMGCL checkpoint selection tie-break differs"
                )
            require_metric_int("Test_Checkpoints_Evaluated", 1)
            if recorded.get("Unselected_Test_Checkpoints_Evaluated") != 0:
                raise BenchmarkAuditError(
                    "SMGCL evaluated an unselected checkpoint on the outer test"
                )
            top_validation = _coverage(
                task / "results" / "validation_predictions.npz",
                expected_row_id=validation_row_id,
                expected_pair=validation_pairs,
                expected_label=validation_labels,
                allow_subset=False,
            )
            if not np.array_equal(
                top_validation.score, validation_tables[selected_epoch].score
            ):
                raise BenchmarkAuditError(
                    "SMGCL top-level validation scores are not the selected checkpoint"
                )
            decision = _load_json(task / "results" / "epoch_selection.json")
            protocol_selection = run_protocol.get("epoch_selection")
            if decision != protocol_selection:
                raise BenchmarkAuditError(
                    "SMGCL epoch-selection decision differs between artifacts"
                )
            if decision.get("Candidates") != candidates or decision.get(
                "Selected_Epoch"
            ) != selected_epoch:
                raise BenchmarkAuditError(
                    "SMGCL persisted epoch-selection decision is invalid"
                )
            decision_contract = {
                "Mode": "inner-validation",
                "Selection_Metric": "validation trapezoidal AUPRC",
                "Selection_Tie_Break": "lowest epoch on an exact AUPRC tie",
                "Test_Checkpoints_Evaluated": 1,
                "Unselected_Test_Checkpoints_Evaluated": 0,
            }
            for name, expected_value in decision_contract.items():
                if decision.get(name) != expected_value:
                    raise BenchmarkAuditError(
                        f"SMGCL persisted epoch-selection {name} differs from "
                        "the locked contract"
                    )
            if not _same_metric(
                decision.get("Selected_Validation_AUPRC"),
                validation_metrics[selected_epoch]["AUPRC"],
            ):
                raise BenchmarkAuditError(
                    "SMGCL persisted selected validation AUPRC differs from "
                    "independent recomputation"
                )
            if decision.get("Checkpoint_Validation_Metrics") != records:
                raise BenchmarkAuditError(
                    "SMGCL persisted checkpoint validation metrics differ"
                )
        elif selection_mode not in {"fixed-budget", None}:
            raise BenchmarkAuditError(
                f"unsupported SMGCL epoch selection mode {selection_mode!r}"
            )
        return

    if method_key == "milgnet":
        require_metric_int("Epochs_Requested", 500)
        require_metric_int("Epochs_Executed", 500)
        require_metric_int("Optimizer_Steps", 500)
        require_metric_int("Total_Row_Presentations", 500 * 65_536)
        if recorded.get("Training_Mode") != "scalable-full-batch":
            raise BenchmarkAuditError(
                "MilGNet strict TCM artifact has the wrong training mode"
            )
        expected_schedule = "tcm-scalable-cyclic-balanced-65536-k20-500updates"
        if recorded.get("Training_Schedule") != expected_schedule:
            raise BenchmarkAuditError(
                "MilGNet strict TCM artifact has the wrong training schedule"
            )
        fold_record = run_protocol.get("fold")
        if not isinstance(fold_record, Mapping):
            raise BenchmarkAuditError(
                "MilGNet strict TCM protocol lacks fold row provenance"
            )
        for name, expected in (
            ("train_rows_full", train_rows),
            ("train_rows_executed", train_rows),
        ):
            actual = _strict_json_int(
                fold_record.get(name), f"MilGNet strict TCM fold {name}", minimum=1
            )
            if actual != expected:
                raise BenchmarkAuditError(
                    f"MilGNet strict TCM {name} differs: {actual} != {expected}"
                )
        return

    if method_key == "lhgce":
        require_metric_int("Epochs_Requested", 1000)
        require_metric_int("Epochs_Executed", 999)
        require_metric_int("Optimizer_Steps", 999)
        require_metric_int("Training_Rows", train_rows)
        require_metric_int("Training_Chunk_Size", train_rows)
        require_metric_int("Evaluation_Chunk_Size", test_rows)
        if recorded.get("Chunked_Loss_Exact") is not True:
            raise BenchmarkAuditError(
                "LHGCE strict TCM artifact does not certify exact chunked loss"
            )
        selection_mode = recorded.get("Epoch_Selection_Mode", "fixed-budget")
        if selection_mode == "inner-validation":
            if task is None:
                raise BenchmarkAuditError(
                    "LHGCE inner-validation selection audit requires its task directory"
                )
            candidates = [20, 50, 100, 200, 300, 500, 1000]
            if recorded.get("Epoch_Candidates") != candidates:
                raise BenchmarkAuditError(
                    "LHGCE inner-validation requested-epoch candidates differ "
                    "from 20/50/100/200/300/500/1000"
                )
            records = recorded.get("Checkpoint_Validation_Metrics")
            if not isinstance(records, list) or len(records) != len(candidates):
                raise BenchmarkAuditError(
                    "LHGCE inner-validation checkpoint metrics are incomplete"
                )
            expected_records: dict[int, Mapping[str, object]] = {}
            for item in records:
                if not isinstance(item, Mapping):
                    raise BenchmarkAuditError(
                        "LHGCE checkpoint validation record is not an object"
                    )
                requested_epochs = _strict_json_int(
                    item.get("Requested_Epochs"),
                    "LHGCE checkpoint Requested_Epochs",
                    minimum=2,
                )
                if (
                    requested_epochs in expected_records
                    or requested_epochs not in candidates
                ):
                    raise BenchmarkAuditError(
                        "LHGCE checkpoint requested epochs are duplicated or unexpected"
                    )
                expected_records[requested_epochs] = item
            if sorted(expected_records) != candidates:
                raise BenchmarkAuditError(
                    "LHGCE checkpoint requested epochs are incomplete"
                )

            validation_pairs, validation_labels, validation_row_id = fold_split(
                fold_rows, "validation"
            )
            validation_metrics: dict[int, dict[str, object]] = {}
            validation_tables: dict[int, PredictionTable] = {}
            for requested_epochs in candidates:
                item = expected_records[requested_epochs]
                expected_steps = requested_epochs - 1
                item_steps = _strict_json_int(
                    item.get("Optimizer_Steps"),
                    f"LHGCE checkpoint {requested_epochs} Optimizer_Steps",
                    minimum=1,
                )
                if item_steps != expected_steps:
                    raise BenchmarkAuditError(
                        f"LHGCE requested budget {requested_epochs} maps to "
                        f"{item_steps} updates, expected {expected_steps}"
                    )
                prediction_path = (
                    task
                    / "results"
                    / "validation_checkpoints"
                    / f"requested_epochs_{requested_epochs}.npz"
                )
                checkpoint_path = (
                    task
                    / "results"
                    / "training_checkpoints"
                    / f"requested_epochs_{requested_epochs}.pt"
                )
                if not prediction_path.is_file():
                    raise BenchmarkAuditError(
                        f"LHGCE checkpoint validation artifact is missing: "
                        f"{prediction_path}"
                    )
                if not checkpoint_path.is_file():
                    raise BenchmarkAuditError(
                        f"LHGCE training checkpoint is missing: {checkpoint_path}"
                    )
                try:
                    import torch

                    try:
                        checkpoint_payload = torch.load(
                            checkpoint_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    except TypeError:  # torch < 2.0 compatibility.
                        checkpoint_payload = torch.load(
                            checkpoint_path, map_location="cpu"
                        )
                except Exception as error:
                    raise BenchmarkAuditError(
                        f"LHGCE training checkpoint is unreadable: {checkpoint_path}"
                    ) from error
                if not isinstance(checkpoint_payload, Mapping):
                    raise BenchmarkAuditError(
                        "LHGCE training checkpoint payload is not an object"
                    )
                checkpoint_requested = _strict_json_int(
                    checkpoint_payload.get("Requested_Epochs"),
                    "LHGCE serialized Requested_Epochs",
                    minimum=2,
                )
                checkpoint_steps = _strict_json_int(
                    checkpoint_payload.get("Optimizer_Steps"),
                    "LHGCE serialized Optimizer_Steps",
                    minimum=1,
                )
                model_state = checkpoint_payload.get("Model_State_Dict")
                optimizer_state = checkpoint_payload.get(
                    "Optimizer_State_Dict"
                )
                if (
                    checkpoint_requested != requested_epochs
                    or checkpoint_steps != expected_steps
                    or not isinstance(model_state, Mapping)
                    or not model_state
                    or not isinstance(optimizer_state, Mapping)
                    or not isinstance(optimizer_state.get("state"), Mapping)
                    or not optimizer_state.get("state")
                    or not isinstance(optimizer_state.get("param_groups"), list)
                    or not optimizer_state.get("param_groups")
                ):
                    raise BenchmarkAuditError(
                        f"LHGCE training checkpoint content differs for "
                        f"requested budget {requested_epochs}"
                    )
                del checkpoint_payload
                table = _coverage(
                    prediction_path,
                    expected_row_id=validation_row_id,
                    expected_pair=validation_pairs,
                    expected_label=validation_labels,
                    allow_subset=False,
                )
                validation_tables[requested_epochs] = table
                recomputed = evaluate_test_scores(
                    table.label, table.score, threshold=0.5
                )
                validation_metrics[requested_epochs] = recomputed
                comparisons = {
                    "Validation_AUROC": recomputed["AUROC"],
                    "Validation_AUPRC": recomputed["AUPRC"],
                    "Validation_AveragePrecision": recomputed[
                        "AveragePrecision"
                    ],
                    "Validation_Predictions_SHA256": sha256(prediction_path),
                    "Training_Checkpoint_SHA256": sha256(checkpoint_path),
                    "Validation_Score_SHA256": hashlib.sha256(
                        np.ascontiguousarray(
                            table.score, dtype=np.float32
                        ).view(np.uint8)
                    ).hexdigest(),
                }
                expected_prediction_relative = (
                    "results/validation_checkpoints/"
                    f"requested_epochs_{requested_epochs}.npz"
                )
                expected_checkpoint_relative = (
                    "results/training_checkpoints/"
                    f"requested_epochs_{requested_epochs}.pt"
                )
                if item.get("Validation_Predictions") != (
                    expected_prediction_relative
                ):
                    raise BenchmarkAuditError(
                        f"LHGCE requested budget {requested_epochs} validation "
                        "artifact path differs from the locked layout"
                    )
                if item.get("Training_Checkpoint") != expected_checkpoint_relative:
                    raise BenchmarkAuditError(
                        f"LHGCE requested budget {requested_epochs} training "
                        "checkpoint path differs from the locked layout"
                    )
                if any("Test" in str(name) for name in item):
                    raise BenchmarkAuditError(
                        "LHGCE candidate checkpoint record contains an outer-test field"
                    )
                for name, expected_value in comparisons.items():
                    actual = item.get(name)
                    equal = (
                        _same_metric(actual, expected_value)
                        if isinstance(expected_value, float)
                        else actual == expected_value
                    )
                    if not equal:
                        raise BenchmarkAuditError(
                            f"LHGCE requested budget {requested_epochs} {name} "
                            "differs from independent validation recomputation"
                        )

            selected_requested_epochs = min(
                candidates,
                key=lambda requested_epochs: (
                    -float(validation_metrics[requested_epochs]["AUPRC"]),
                    requested_epochs,
                ),
            )
            require_metric_int(
                "Selected_Requested_Epochs", selected_requested_epochs
            )
            require_metric_int(
                "Selected_Checkpoint_Optimizer_Steps",
                selected_requested_epochs - 1,
            )
            if recorded.get("Epoch_Selection_Metric") != (
                "validation trapezoidal AUPRC"
            ):
                raise BenchmarkAuditError(
                    "LHGCE checkpoint selection metric is not validation AUPRC"
                )
            if recorded.get("Epoch_Selection_Tie_Break") != (
                "lowest requested epoch budget on an exact AUPRC tie"
            ):
                raise BenchmarkAuditError(
                    "LHGCE checkpoint selection tie-break differs"
                )
            if recorded.get(
                "Checkpoint_Trajectory_Equivalent_To_Independent_Fixed_Budget"
            ) is not True:
                raise BenchmarkAuditError(
                    "LHGCE does not certify fixed-budget prefix equivalence"
                )
            if recorded.get("Decision_Frozen_Before_Test") is not True:
                raise BenchmarkAuditError(
                    "LHGCE does not certify that selection was frozen before test"
                )
            require_metric_int("Test_Checkpoints_Evaluated", 1)
            if recorded.get("Unselected_Test_Checkpoints_Evaluated") != 0:
                raise BenchmarkAuditError(
                    "LHGCE evaluated an unselected checkpoint on the outer test"
                )
            top_validation = _coverage(
                task / "results" / "validation_predictions.npz",
                expected_row_id=validation_row_id,
                expected_pair=validation_pairs,
                expected_label=validation_labels,
                allow_subset=False,
            )
            if not np.array_equal(
                top_validation.score,
                validation_tables[selected_requested_epochs].score,
            ):
                raise BenchmarkAuditError(
                    "LHGCE top-level validation scores are not the selected checkpoint"
                )
            decision = _load_json(task / "results" / "epoch_selection.json")
            if decision != run_protocol.get("epoch_selection"):
                raise BenchmarkAuditError(
                    "LHGCE epoch-selection decision differs between artifacts"
                )
            if decision.get("Candidates") != candidates or decision.get(
                "Selected_Requested_Epochs"
            ) != selected_requested_epochs:
                raise BenchmarkAuditError(
                    "LHGCE persisted epoch-selection decision is invalid"
                )
            decision_contract = {
                "Mode": "inner-validation",
                "Selection_Metric": "validation trapezoidal AUPRC",
                "Selection_Tie_Break": (
                    "lowest requested epoch budget on an exact AUPRC tie"
                ),
                "Decision_Frozen_Before_Test": True,
                "Test_Checkpoints_Evaluated": 1,
                "Unselected_Test_Checkpoints_Evaluated": 0,
            }
            for name, expected_value in decision_contract.items():
                if decision.get(name) != expected_value:
                    raise BenchmarkAuditError(
                        f"LHGCE persisted epoch-selection {name} differs from "
                        "the locked contract"
                    )
            if not _same_metric(
                decision.get("Selected_Validation_AUPRC"),
                validation_metrics[selected_requested_epochs]["AUPRC"],
            ):
                raise BenchmarkAuditError(
                    "LHGCE persisted selected validation AUPRC differs from "
                    "independent recomputation"
                )
            if decision.get("Checkpoint_Validation_Metrics") != records:
                raise BenchmarkAuditError(
                    "LHGCE persisted checkpoint validation metrics differ"
                )
            selected_item = expected_records[selected_requested_epochs]
            if decision.get("Selected_Training_Checkpoint") != selected_item.get(
                "Training_Checkpoint"
            ) or decision.get("Selected_Training_Checkpoint_SHA256") != (
                selected_item.get("Training_Checkpoint_SHA256")
            ):
                raise BenchmarkAuditError(
                    "LHGCE persisted selected checkpoint identity differs"
                )
            forbidden_test_checkpoint_dir = task / "results" / "test_checkpoints"
            if forbidden_test_checkpoint_dir.exists() or list(
                (task / "results").glob("**/test_checkpoint*")
            ):
                raise BenchmarkAuditError(
                    "LHGCE materialized candidate-specific outer-test artifacts"
                )
        elif selection_mode not in {"fixed-budget", None}:
            raise BenchmarkAuditError(
                f"unsupported LHGCE epoch selection mode {selection_mode!r}"
            )
        return

    require_metric_int("Epochs_Requested", 16)
    require_metric_int("Epochs_Executed", 16)
    require_metric_int("Batch_Size", 2000)
    require_metric_int("Training_Rows", train_rows)
    require_metric_int(
        "Optimizer_Steps", 16 * int(math.ceil(train_rows / 2000))
    )
    hyperparameters = run_protocol.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise BenchmarkAuditError(
            "HGHDA strict TCM protocol lacks hyperparameter provenance"
        )
    if (
        hyperparameters.get("epochs") != 16
        or hyperparameters.get("batch_size") != 2000
    ):
        raise BenchmarkAuditError(
            "HGHDA strict TCM protocol hyperparameters differ from the locked budget"
        )


def _audit_legacy_provenance(
    *,
    recorded: Mapping[str, object],
    run_protocol: Mapping[str, object],
    protocol,
    fold_rows,
    protocol_manifest: Path,
    project_identity: Mapping[str, object],
    is_smoke: bool,
) -> None:
    """Independently reconstruct and enforce the legacy-run identity."""

    if run_protocol.get("pair_order") != "source":
        raise BenchmarkAuditError("legacy run protocol does not record pair_order='source'")
    expected_input_manifest = protocol.input_manifest()
    run_input_manifest = run_protocol.get("protocol_input_manifest", run_protocol)
    if not isinstance(run_input_manifest, Mapping):
        raise BenchmarkAuditError("legacy run protocol input manifest is invalid")
    input_mismatches = {
        key: {
            "recorded": run_input_manifest.get(key),
            "expected": value,
        }
        for key, value in expected_input_manifest.items()
        if run_input_manifest.get(key) != value
    }
    if input_mismatches:
        raise BenchmarkAuditError(
            "legacy run protocol input manifest differs from the central manifest: "
            + json.dumps(input_mismatches, sort_keys=True, default=str)
        )

    try:
        expected = legacy_protocol_provenance(
            protocol,
            fold_rows,
            protocol_manifest,
            project_identity=dict(project_identity),
            require_clean_project=not is_smoke,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise BenchmarkAuditError(
            f"cannot reconstruct legacy run provenance: {error}"
        ) from error

    run_value = run_protocol.get("legacy_protocol_provenance")
    metric_value = recorded.get("Legacy_Protocol_Provenance")
    if not isinstance(run_value, Mapping):
        raise BenchmarkAuditError(
            "legacy run protocol lacks legacy_protocol_provenance"
        )
    if not isinstance(metric_value, Mapping):
        raise BenchmarkAuditError(
            "legacy metrics lack Legacy_Protocol_Provenance"
        )
    mismatches: dict[str, dict[str, object]] = {}
    for name, expected_value in expected.items():
        run_recorded = run_value.get(name)
        metric_recorded = metric_value.get(name)
        if run_recorded != expected_value or metric_recorded != expected_value:
            mismatches[name] = {
                "protocol_recorded": run_recorded,
                "metrics_recorded": metric_recorded,
                "expected": expected_value,
            }
    if mismatches:
        raise BenchmarkAuditError(
            "legacy run provenance differs from independent reconstruction: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


def _audit_fold(
    *,
    protocol,
    run_root: Path,
    method: str,
    fold: int,
    allow_smoke: bool,
    allow_diagnostic: bool,
    legacy_test_oracle: bool,
    protocol_manifest: Path,
    project_identity: Mapping[str, object] | None,
) -> tuple[dict[str, object], PredictionTable]:
    task = run_root / "tasks" / method / f"fold_{fold}"
    metrics_path = task / "results" / "metrics.json"
    validation_path = task / "results" / "validation_predictions.npz"
    test_path = task / "results" / "test_predictions.npz"
    run_protocol_path = task / "protocol.json"
    missing = [
        str(path)
        for path in (metrics_path, validation_path, test_path, run_protocol_path)
        if not path.is_file()
    ]
    if missing:
        raise BenchmarkAuditError(
            f"{method} fold {fold} is incomplete; missing: {', '.join(missing)}"
        )

    recorded = _load_json(metrics_path)
    run_protocol = _load_json(run_protocol_path)
    expected = protocol.fold_rows(fold)
    is_smoke, is_diagnostic = _validate_execution_mode(
        recorded=recorded,
        run_protocol=run_protocol,
        allow_smoke=allow_smoke,
        allow_diagnostic=allow_diagnostic,
    )
    if is_diagnostic:
        _assert_diagnostic_full_budget(
            recorded=recorded,
            run_protocol=run_protocol,
            method=method,
            dataset=protocol.dataset,
            fold_rows=expected,
        )
    if str(recorded.get("Dataset")) != protocol.dataset:
        raise BenchmarkAuditError(f"{method} fold {fold} dataset identity differs")
    if _strict_json_int(recorded.get("Fold"), "metrics Fold", minimum=1) != fold:
        raise BenchmarkAuditError(f"{method} fold {fold} fold identity differs")
    if _strict_json_int(recorded.get("Seed"), "metrics Seed", minimum=0) != protocol.seed:
        raise BenchmarkAuditError(f"{method} fold {fold} seed differs from protocol")
    if run_protocol.get("dataset") != protocol.dataset:
        raise BenchmarkAuditError(f"{method} fold {fold} protocol dataset differs")
    if method.lower() == "astra":
        hyperparameters = run_protocol.get("hyperparameters")
        metric_has_alpha = "Text_Fusion_Alpha" in recorded
        protocol_has_alpha = isinstance(hyperparameters, Mapping) and (
            "text_fusion_alpha" in hyperparameters
        )
        if metric_has_alpha or protocol_has_alpha:
            try:
                alphas_match = bool(
                    metric_has_alpha
                    and protocol_has_alpha
                    and math.isclose(
                        float(recorded["Text_Fusion_Alpha"]),
                        float(hyperparameters["text_fusion_alpha"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )
            except (TypeError, ValueError):
                alphas_match = False
            if not alphas_match:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} metric/protocol alpha differs"
                )
    expected_input_manifest = protocol.input_manifest()
    nested_input_manifest = run_protocol.get("protocol_input_manifest")
    run_input_manifest = (
        nested_input_manifest
        if nested_input_manifest is not None
        else {
            key: run_protocol.get(key)
            for key in expected_input_manifest
        }
    )
    if not isinstance(run_input_manifest, Mapping):
        raise BenchmarkAuditError(f"{method} fold {fold} protocol payload is invalid")
    if dict(run_input_manifest) != expected_input_manifest:
        raise BenchmarkAuditError(
            f"{method} fold {fold} complete protocol input manifest differs"
        )
    expected_input_lock = validate_frozen_input_manifest(expected_input_manifest)
    if run_protocol.get("frozen_input_lock_sha256") != expected_input_lock:
        raise BenchmarkAuditError(
            f"{method} fold {fold} frozen input lock differs from the central manifest"
        )
    expected_mode = "legacy-test-oracle" if legacy_test_oracle else "validation"
    if recorded.get("Evaluation_Mode", "validation") != expected_mode:
        raise BenchmarkAuditError(
            f"{method} fold {fold} evaluation mode differs: "
            f"expected {expected_mode!r}, got {recorded.get('Evaluation_Mode')!r}"
        )
    if (
        protocol.dataset == "tcm-suite"
        and not legacy_test_oracle
        and not is_smoke
        and not is_diagnostic
    ):
        _assert_formal_tcm_budget(
            recorded=recorded,
            run_protocol=run_protocol,
            method=method,
            fold_rows=expected,
            task=task,
        )

    if legacy_test_oracle:
        if project_identity is None:  # Defensive; audit_run always supplies it.
            raise BenchmarkAuditError("legacy audit lacks a project Git identity")
        _audit_legacy_provenance(
            recorded=recorded,
            run_protocol=run_protocol,
            protocol=protocol,
            fold_rows=expected,
            protocol_manifest=protocol_manifest,
            project_identity=project_identity,
            is_smoke=is_smoke,
        )
    validation = _coverage(
        validation_path,
        expected_row_id=expected.validation_row_indices,
        expected_pair=expected.validation_pairs,
        expected_label=expected.validation_labels,
        allow_subset=is_smoke and allow_smoke,
    )
    test = _coverage(
        test_path,
        expected_row_id=expected.test_row_indices,
        expected_pair=expected.test_pairs,
        expected_label=expected.test_labels,
        allow_subset=is_smoke and allow_smoke,
    )
    if legacy_test_oracle:
        recomputed = evaluate_test_oracle(test.label, test.score)
        recomputed_keys = LEGACY_RECOMPUTED_KEYS
    else:
        recomputed = evaluate_validation_and_test(
            validation.label, validation.score, test.label, test.score
        )
        recomputed_keys = RECOMPUTED_KEYS
    mismatches = _metric_mismatches(
        recorded=recorded,
        recomputed=recomputed,
        keys=recomputed_keys,
    )
    if mismatches:
        raise BenchmarkAuditError(
            f"{method} fold {fold} metrics differ from independent recomputation: "
            + json.dumps(mismatches, sort_keys=True)
        )

    optimizer_steps_int = _strict_json_int(
        recorded.get("Optimizer_Steps"), "metrics Optimizer_Steps", minimum=1
    )
    if method.lower() == "milgnet":
        audit_milgnet_run(test, optimizer_steps=optimizer_steps_int)
    row: dict[str, object] = {
        "Dataset": protocol.dataset,
        "Method": method,
        "Fold": fold,
        "Seed": protocol.seed,
        "Smoke": is_smoke,
        "Diagnostic": is_diagnostic,
        "Optimizer_Steps": optimizer_steps_int,
        "Selected_Epoch": recorded.get("Selected_Epoch"),
        "Selected_Requested_Epochs": recorded.get(
            "Selected_Requested_Epochs"
        ),
        "Selected_Checkpoint_Optimizer_Steps": recorded.get(
            "Selected_Checkpoint_Optimizer_Steps"
        ),
        "Text_Fusion_Alpha": recorded.get("Text_Fusion_Alpha"),
        "Alpha_Selected_On_Validation": recorded.get(
            "Alpha_Selected_On_Validation", False
        ),
        "Validation_Rows": len(validation),
        "Test_Rows": len(test),
        **recomputed,
    }
    return row, test


def audit_run(
    *,
    protocol_manifest: Path | str,
    run_root: Path | str,
    methods: Sequence[str] = DEFAULT_METHODS,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    allow_smoke: bool = False,
    allow_diagnostic: bool = False,
    legacy_test_oracle: bool = False,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, float | int]]]]:
    """Audit every requested task and return fold rows plus summaries."""

    protocol_manifest = Path(protocol_manifest).resolve()
    protocol = load_protocol_manifest(protocol_manifest)
    if legacy_test_oracle:
        if protocol.dataset != "tcm-suite":
            raise BenchmarkAuditError(
                "legacy test-oracle audit is restricted to the TCM-suite dataset"
            )
        if protocol.pair_order != "source":
            raise BenchmarkAuditError(
                "legacy test-oracle audit requires a source-order protocol manifest"
            )
        if len(protocol.positive_pairs) % protocol.n_folds != 0:
            raise BenchmarkAuditError(
                "legacy class-local modulo folds are equivalent to the retained "
                "global-row rule only when the TCM positive-row count is divisible "
                "by n_folds"
            )
    run_root = Path(run_root).resolve()
    methods = tuple(dict.fromkeys(str(method) for method in methods))
    folds = tuple(dict.fromkeys(int(fold) for fold in folds))
    if not methods:
        raise BenchmarkAuditError("at least one method is required")
    if set(folds) - set(range(1, protocol.n_folds + 1)):
        raise BenchmarkAuditError(f"invalid fold selection: {folds}")
    if allow_smoke and allow_diagnostic:
        raise BenchmarkAuditError(
            "allow_smoke and allow_diagnostic are mutually exclusive"
        )
    if (
        not allow_smoke
        and not allow_diagnostic
        and set(folds) != set(range(1, protocol.n_folds + 1))
    ):
        raise BenchmarkAuditError("formal aggregation requires all five folds")

    astra_methods = [method for method in methods if method.lower() == "astra"]
    if astra_methods:
        alpha_manifest_present = (run_root / "alpha_selection.json").is_file()
        selected_flags: list[bool] = []
        for fold in folds:
            metrics_path = (
                run_root
                / "tasks"
                / astra_methods[0]
                / f"fold_{fold}"
                / "results"
                / "metrics.json"
            )
            if metrics_path.is_file():
                recorded_metrics = _load_json(metrics_path)
                run_protocol_path = metrics_path.parents[1] / "protocol.json"
                run_protocol_value = (
                    _load_json(run_protocol_path)
                    if run_protocol_path.is_file()
                    else {}
                )
                try:
                    nondefault_alpha = not math.isclose(
                        float(recorded_metrics.get("Text_Fusion_Alpha", 1.0)),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                except (TypeError, ValueError):
                    nondefault_alpha = True
                selected_flags.append(
                    alpha_manifest_present
                    or recorded_metrics.get("Alpha_Selected_On_Validation") is True
                    or nondefault_alpha
                    or isinstance(run_protocol_value.get("alpha_selection"), Mapping)
                )
        if any(selected_flags):
            if not selected_flags or not all(selected_flags):
                raise BenchmarkAuditError(
                    "ASTRA alpha-selection status differs across outer folds"
                )
            if allow_smoke or allow_diagnostic or legacy_test_oracle:
                raise BenchmarkAuditError(
                    "validation-selected alpha is supported only by the formal protocol"
                )
            try:
                from astra.benchmark.ethnobotany_alpha_selection import (
                    validate_selected_binding,
                    validate_selection_decision,
                )

                selection_path = run_root / "alpha_selection.json"
                _, _, alpha_decisions = validate_selection_decision(
                    selection_path=selection_path,
                    protocol_manifest=protocol_manifest,
                    output_root=run_root,
                )
                for fold in folds:
                    validate_selected_binding(
                        run_dir=(
                            run_root
                            / "tasks"
                            / astra_methods[0]
                            / f"fold_{fold}"
                        ),
                        fold=fold,
                        selected_alpha=float(
                            alpha_decisions[str(fold)]["selected_alpha"]
                        ),
                        selection_path=selection_path,
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise BenchmarkAuditError(
                    f"ASTRA validation-selected alpha audit failed: {error}"
                ) from error

    legacy_project_identity = None
    if legacy_test_oracle:
        # Compute once, then use the same independently observed HEAD/clean
        # identity for every method and fold in this aggregate.
        legacy_project_identity = project_git_identity(require_clean=False)

    rows: list[dict[str, object]] = []
    predictions: dict[str, dict[int, PredictionTable]] = {}
    for method in methods:
        predictions[method] = {}
        for fold in folds:
            row, test = _audit_fold(
                protocol=protocol,
                run_root=run_root,
                method=method,
                fold=fold,
                allow_smoke=allow_smoke,
                allow_diagnostic=allow_diagnostic,
                legacy_test_oracle=legacy_test_oracle,
                protocol_manifest=protocol_manifest,
                project_identity=legacy_project_identity,
            )
            rows.append(row)
            predictions[method][fold] = test
    audit_cross_method_predictions(predictions)

    summaries = {
        method: aggregate_fold_metrics(
            [row for row in rows if row["Method"] == method]
        )
        for method in methods
        if len([row for row in rows if row["Method"] == method]) >= 2
    }
    return rows, summaries


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise BenchmarkAuditError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary_rows(
    summaries: Mapping[str, Mapping[str, Mapping[str, float | int]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, metrics in summaries.items():
        row: dict[str, object] = {"Method": method}
        for metric, values in metrics.items():
            row[f"{metric}_Mean"] = values["mean"]
            row[f"{metric}_Sample_SD"] = values["sample_sd"]
            row[f"{metric}_Folds"] = values["folds"]
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    nonformal_mode = parser.add_mutually_exclusive_group()
    nonformal_mode.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Permit labelled subset smoke artifacts; never formal results.",
    )
    nonformal_mode.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help=(
            "Permit full-row/full-budget diagnostic artifacts. The audit output "
            "is diagnostic-only and can never certify a formal result."
        ),
    )
    parser.add_argument(
        "--legacy-test-oracle",
        action="store_true",
        help=(
            "Audit the retained source-row/full-outer-train comparison; AUROC/AUPRC "
            "remain primary and threshold metrics are descriptive test oracles."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = (args.output_dir or args.run_root / "audit").resolve()
    rows, summaries = audit_run(
        protocol_manifest=args.protocol_manifest,
        run_root=args.run_root,
        methods=args.methods,
        folds=args.folds,
        allow_smoke=args.allow_smoke,
        allow_diagnostic=args.allow_diagnostic,
        legacy_test_oracle=args.legacy_test_oracle,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fold_metrics.csv", rows)
    summary_rows = _summary_rows(summaries)
    if summary_rows:
        _write_csv(output_dir / "summary.csv", summary_rows)
    payload = {
        "status": (
            "smoke-only"
            if args.allow_smoke
            else "diagnostic-only"
            if args.allow_diagnostic
            else "formal-audit-passed"
        ),
        "formal_result": bool(not args.allow_smoke and not args.allow_diagnostic),
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "run_root": str(args.run_root.resolve()),
        "methods": list(args.methods),
        "folds": list(args.folds),
        "fold_metrics": rows,
        "summary": summaries,
        "threshold_policy": (
            "descriptive test-label oracle; AUROC/AUPRC are primary"
            if args.legacy_test_oracle
            else "validation maximum F1; applied unchanged to test"
        ),
        "auprc_definition": AUPRC_DEFINITION,
        "average_precision_definition": AVERAGE_PRECISION_DEFINITION,
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
