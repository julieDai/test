"""Central metrics and fail-closed audits for fair H--D comparisons.

This module deliberately separates threshold selection from test evaluation.
An operating threshold is selected once from validation labels and scores and
is then applied unchanged to the held-out test fold.  Test labels are never
used to choose a threshold or a checkpoint here.

Prediction artifacts use four mandatory arrays: ``row_id``, ``pair``,
``label``, and ``score``.  The row identifier is the stable join key to the
frozen fold manifest; array position alone is not treated as provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 100, dtype=np.float64)
PRIMARY_METRICS = ("AUROC", "AUPRC")
SECONDARY_METRICS = ("AveragePrecision",)
THRESHOLD_METRICS = ("Accuracy", "Precision", "Recall", "F1")
PREDICTION_FIELDS = ("row_id", "pair", "label", "score")
AUPRC_DEFINITION = (
    "trapezoidal area under the precision-recall curve computed as "
    "sklearn.metrics.auc(recall, precision) after precision_recall_curve"
)
AVERAGE_PRECISION_DEFINITION = "sklearn.metrics.average_precision_score"


class BenchmarkAuditError(ValueError):
    """Raised when an artifact cannot be accepted as a formal result."""


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-selected operating point and its validation diagnostics."""

    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    candidates: int
    tie_break: str = "lowest threshold"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "Validation_Selected_Threshold": self.threshold,
            "Validation_F1": self.f1,
            "Validation_Precision": self.precision,
            "Validation_Recall": self.recall,
            "Validation_Accuracy": self.accuracy,
            "Threshold_Candidates": self.candidates,
            "Threshold_Tie_Break": self.tie_break,
        }


@dataclass(frozen=True)
class PredictionTable:
    """One fold's predictions with explicit frozen-manifest identity."""

    row_id: np.ndarray
    pair: np.ndarray
    label: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return len(self.row_id)


def _one_dimensional(name: str, values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise BenchmarkAuditError(f"{name} must be one-dimensional, got {array.shape}")
    if not len(array):
        raise BenchmarkAuditError(f"{name} must not be empty")
    return array


def _labels(values: object, *, require_both_classes: bool) -> np.ndarray:
    labels = _one_dimensional("labels", values)
    if not np.all(np.isin(labels, (0, 1))):
        raise BenchmarkAuditError("labels must contain only binary values 0 and 1")
    labels = labels.astype(np.int64, copy=False)
    if require_both_classes and np.unique(labels).size != 2:
        raise BenchmarkAuditError("both label classes are required")
    return labels


def _scores(
    values: object,
    *,
    require_probability: bool,
    require_nonconstant: bool,
) -> np.ndarray:
    scores = _one_dimensional("scores", values).astype(np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise BenchmarkAuditError("scores contain NaN or infinity")
    if require_probability and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
        raise BenchmarkAuditError("threshold selection requires probability scores in [0, 1]")
    if require_nonconstant and np.unique(scores).size < 2:
        raise BenchmarkAuditError("scores are constant")
    return scores


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | str]:
    """Return the paper's ranking metrics plus average precision.

    The legacy HerbHG evaluation code called ``precision_recall_curve`` and
    integrated the returned points with ``auc``.  ``AUPRC`` therefore retains
    that definition for manuscript-facing comparisons.  Average precision is
    also emitted under an unambiguous secondary name because it is useful for
    interoperability and can differ when scores contain ties.
    """

    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(auc(recall, precision)),
        "AUPRC_Definition": AUPRC_DEFINITION,
        "AveragePrecision": float(average_precision_score(labels, scores)),
        "AveragePrecision_Definition": AVERAGE_PRECISION_DEFINITION,
    }


def select_f1_threshold(
    validation_labels: object,
    validation_scores: object,
    *,
    thresholds: Sequence[float] | np.ndarray = DEFAULT_THRESHOLDS,
) -> ThresholdSelection:
    """Select maximum-F1 threshold on validation data only.

    Candidate thresholds are evaluated in ascending order.  If several yield
    the same F1, the lowest threshold is retained.  The default candidates are
    exactly 100 evenly spaced values on the closed interval ``[0, 1]``.
    """

    labels = _labels(validation_labels, require_both_classes=True)
    scores = _scores(
        validation_scores,
        require_probability=True,
        require_nonconstant=True,
    )
    if len(labels) != len(scores):
        raise BenchmarkAuditError(
            f"validation labels/scores differ in length: {len(labels)} vs {len(scores)}"
        )
    candidates = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if not len(candidates):
        raise BenchmarkAuditError("at least one threshold candidate is required")
    if not np.isfinite(candidates).all():
        raise BenchmarkAuditError("threshold candidates contain NaN or infinity")
    if float(candidates.min()) < 0.0 or float(candidates.max()) > 1.0:
        raise BenchmarkAuditError("threshold candidates must lie in [0, 1]")
    candidates = np.unique(candidates)

    # Counting confusion-matrix entries from one stable score sort is exactly
    # equivalent to materializing ``scores >= threshold`` for every candidate,
    # but avoids 100 full sklearn passes over million-row benchmark folds.
    # ``side='left'`` is required because a score equal to the threshold is a
    # positive prediction.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_positives = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(sorted_labels, dtype=np.int64))
    )
    cut = np.searchsorted(sorted_scores, candidates, side="left")
    total = len(labels)
    total_positives = int(cumulative_positives[-1])
    predicted_positives = total - cut
    true_positives = total_positives - cumulative_positives[cut]
    false_positives = predicted_positives - true_positives
    false_negatives = total_positives - true_positives
    true_negatives = total - true_positives - false_positives - false_negatives

    precision = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=predicted_positives != 0,
    )
    recall = true_positives.astype(np.float64) / total_positives
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros_like(precision),
        where=f1_denominator != 0.0,
    )
    accuracy = (true_positives + true_negatives).astype(np.float64) / total

    # np.argmax returns the first maximum.  Candidates are ascending and
    # unique, so this preserves the documented lowest-threshold tie break.
    best_index = int(np.argmax(f1))
    best_threshold = float(candidates[best_index])
    return ThresholdSelection(
        threshold=best_threshold,
        f1=float(f1[best_index]),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        accuracy=float(accuracy[best_index]),
        candidates=int(len(candidates)),
    )


def evaluate_test_scores(
    test_labels: object,
    test_scores: object,
    *,
    threshold: float,
) -> dict[str, float | str]:
    """Evaluate a held-out fold at a previously selected threshold."""

    labels = _labels(test_labels, require_both_classes=True)
    scores = _scores(test_scores, require_probability=True, require_nonconstant=True)
    if len(labels) != len(scores):
        raise BenchmarkAuditError(
            f"test labels/scores differ in length: {len(labels)} vs {len(scores)}"
        )
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise BenchmarkAuditError("test threshold must be finite and lie in [0, 1]")
    return {
        **_ranking_metrics(labels, scores),
        "Operating_Threshold": threshold,
        "Threshold_Source": "validation maximum F1; test labels not used",
        **_binary_metrics(labels, scores, threshold),
    }


def evaluate_validation_scores(
    validation_labels: object,
    validation_scores: object,
    *,
    thresholds: Sequence[float] | np.ndarray = DEFAULT_THRESHOLDS,
) -> dict[str, float | int | str]:
    """Evaluate one inner-validation prediction table without touching test data.

    This is used for hyperparameter selection.  Ranking metrics are prefixed
    with ``Validation_`` so the artifact cannot be mistaken for an outer-test
    result.  The F1 operating threshold is also selected on this same inner
    split and may later be applied unchanged to the selected model's test fold.
    """

    labels = _labels(validation_labels, require_both_classes=True)
    scores = _scores(
        validation_scores,
        require_probability=True,
        require_nonconstant=True,
    )
    if len(labels) != len(scores):
        raise BenchmarkAuditError(
            "validation labels/scores differ in length: "
            f"{len(labels)} vs {len(scores)}"
        )
    selection = select_f1_threshold(labels, scores, thresholds=thresholds)
    ranking = _ranking_metrics(labels, scores)
    return {
        **selection.as_dict(),
        "Validation_AUROC": ranking["AUROC"],
        "Validation_AUPRC": ranking["AUPRC"],
        "Validation_AUPRC_Definition": ranking["AUPRC_Definition"],
        "Validation_AveragePrecision": ranking["AveragePrecision"],
        "Validation_AveragePrecision_Definition": ranking[
            "AveragePrecision_Definition"
        ],
    }


def evaluate_validation_and_test(
    validation_labels: object,
    validation_scores: object,
    test_labels: object,
    test_scores: object,
    *,
    thresholds: Sequence[float] | np.ndarray = DEFAULT_THRESHOLDS,
) -> dict[str, float | int | str]:
    """Select on validation data and apply the result once to test data."""

    selection = select_f1_threshold(
        validation_labels,
        validation_scores,
        thresholds=thresholds,
    )
    return {
        **selection.as_dict(),
        **evaluate_test_scores(test_labels, test_scores, threshold=selection.threshold),
    }


def evaluate_test_oracle(
    test_labels: object,
    test_scores: object,
    *,
    thresholds: Sequence[float] | np.ndarray = DEFAULT_THRESHOLDS,
) -> dict[str, float | int | str]:
    """Reproduce the retained legacy test-fold threshold diagnostics.

    AUROC and AUPRC remain threshold-free primary metrics.  Precision, recall,
    accuracy, and F1 are explicitly labelled descriptive because the evaluated
    test labels select their threshold.  This function exists only for matched
    reproduction of the manuscript's legacy row-based comparison; new fair
    experiments must use :func:`evaluate_validation_and_test`.
    """

    selection = select_f1_threshold(
        test_labels,
        test_scores,
        thresholds=thresholds,
    )
    evaluated = evaluate_test_scores(
        test_labels,
        test_scores,
        threshold=selection.threshold,
    )
    evaluated.update(
        {
            "Operating_Threshold": selection.threshold,
            "Threshold_Source": (
                "descriptive test-label oracle: maximum F1 on the evaluated fold"
            ),
            "Test_Oracle_Threshold": selection.threshold,
            "Test_Oracle_F1": evaluated["F1"],
            "Test_Oracle_Precision": evaluated["Precision"],
            "Test_Oracle_Recall": evaluated["Recall"],
            "Test_Oracle_Accuracy": evaluated["Accuracy"],
            "Threshold_Candidates": selection.candidates,
            "Threshold_Tie_Break": selection.tie_break,
        }
    )
    return evaluated


def _artifact_array(name: str, values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise BenchmarkAuditError(
            f"{name} uses object dtype; use numeric or fixed-width string identifiers"
        )
    if array.dtype.kind not in "biufSU":
        raise BenchmarkAuditError(
            f"{name} has unsupported dtype {array.dtype}; use numeric or fixed-width strings"
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise BenchmarkAuditError(f"{name} contains NaN or infinity")
    return array


def _prediction_table(
    row_id: object,
    pair: object,
    label: object,
    score: object,
    *,
    require_nonconstant: bool = True,
) -> PredictionTable:
    rows = _artifact_array("row_id", _one_dimensional("row_id", row_id))
    pairs = _artifact_array("pair", pair)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise BenchmarkAuditError(f"pair must have shape (n, 2), got {pairs.shape}")
    labels = _labels(label, require_both_classes=False)
    scores = _scores(
        score,
        require_probability=True,
        require_nonconstant=require_nonconstant,
    )
    lengths = {len(rows), len(pairs), len(labels), len(scores)}
    if len(lengths) != 1:
        raise BenchmarkAuditError(
            "prediction fields have unequal lengths: "
            f"row_id={len(rows)}, pair={len(pairs)}, label={len(labels)}, score={len(scores)}"
        )
    if np.unique(rows).size != len(rows):
        raise BenchmarkAuditError("prediction row_id values are not unique")
    return PredictionTable(rows, pairs, labels, scores)


def save_predictions(
    path: str | Path,
    *,
    row_id: object,
    pair: object,
    label: object,
    score: object,
) -> Path:
    """Validate and atomically save the mandatory prediction artifact."""

    table = _prediction_table(row_id, pair, label, score)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        row_id=table.row_id,
        pair=table.pair,
        label=table.label,
        score=table.score,
    )
    temporary.replace(path)
    return path


def load_predictions(path: str | Path) -> PredictionTable:
    """Load a prediction artifact without permitting pickled object arrays."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in PREDICTION_FIELDS if name not in payload.files]
        if missing:
            raise BenchmarkAuditError(
                f"prediction artifact is missing fields: {', '.join(missing)}"
            )
        values = {name: np.asarray(payload[name]) for name in PREDICTION_FIELDS}
    return _prediction_table(**values)


def _row_key(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def validate_prediction_coverage(
    predictions: PredictionTable | str | Path,
    *,
    expected_row_id: object,
    expected_pair: object,
    expected_label: object,
) -> PredictionTable:
    """Require exact manifest coverage and return predictions in manifest order."""

    table = load_predictions(predictions) if isinstance(predictions, (str, Path)) else predictions
    table = _prediction_table(table.row_id, table.pair, table.label, table.score)
    expected_rows = _artifact_array(
        "expected_row_id", _one_dimensional("expected_row_id", expected_row_id)
    )
    expected_pairs = _artifact_array("expected_pair", expected_pair)
    if expected_pairs.ndim != 2 or expected_pairs.shape[1] != 2:
        raise BenchmarkAuditError(
            f"expected_pair must have shape (n, 2), got {expected_pairs.shape}"
        )
    expected_labels = _labels(expected_label, require_both_classes=False)
    if not len(expected_rows) == len(expected_pairs) == len(expected_labels):
        raise BenchmarkAuditError("expected manifest fields have unequal lengths")
    if np.unique(expected_rows).size != len(expected_rows):
        raise BenchmarkAuditError("expected manifest row_id values are not unique")
    if len(table) != len(expected_rows):
        raise BenchmarkAuditError(
            f"prediction/manifest row counts differ: {len(table)} vs {len(expected_rows)}"
        )

    actual_positions = {_row_key(value): index for index, value in enumerate(table.row_id)}
    expected_keys = [_row_key(value) for value in expected_rows]
    if set(actual_positions) != set(expected_keys):
        missing = set(expected_keys) - set(actual_positions)
        extra = set(actual_positions) - set(expected_keys)
        raise BenchmarkAuditError(
            "prediction row_id coverage differs from manifest: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    order = np.fromiter(
        (actual_positions[key] for key in expected_keys),
        dtype=np.int64,
        count=len(expected_keys),
    )
    aligned = PredictionTable(
        row_id=table.row_id[order],
        pair=table.pair[order],
        label=table.label[order],
        score=table.score[order],
    )
    if not np.array_equal(aligned.row_id, expected_rows):
        raise BenchmarkAuditError("prediction row_id alignment failed")
    if not np.array_equal(aligned.pair, expected_pairs):
        raise BenchmarkAuditError("prediction pairs differ from the frozen manifest")
    if not np.array_equal(aligned.label, expected_labels):
        raise BenchmarkAuditError("prediction labels differ from the frozen manifest")
    return aligned


def audit_cross_method_predictions(
    predictions: Mapping[str, Mapping[int, PredictionTable]],
) -> None:
    """Reject element-wise identical score arrays for any method pair/fold.

    All methods must provide the same non-empty fold set and the same ordered
    test examples.  Call :func:`validate_prediction_coverage` first when raw
    artifacts are not already in frozen-manifest order.
    """

    if len(predictions) < 2:
        return
    validated: dict[str, dict[int, PredictionTable]] = {}
    expected_folds: set[int] | None = None
    for method, method_folds in predictions.items():
        fold_set = set(method_folds)
        if not fold_set:
            raise BenchmarkAuditError(f"{method} has no prediction folds")
        if expected_folds is None:
            expected_folds = fold_set
        elif fold_set != expected_folds:
            raise BenchmarkAuditError(
                f"{method} fold coverage differs from other methods: "
                f"expected={sorted(expected_folds)}, actual={sorted(fold_set)}"
            )
        validated[method] = {
            fold: _prediction_table(
                table.row_id,
                table.pair,
                table.label,
                table.score,
            )
            for fold, table in method_folds.items()
        }

    assert expected_folds is not None
    for left_name, right_name in combinations(sorted(validated), 2):
        for fold in sorted(expected_folds):
            left = validated[left_name][fold]
            right = validated[right_name][fold]
            if not np.array_equal(left.row_id, right.row_id):
                raise BenchmarkAuditError(
                    f"{left_name} and {right_name} fold {fold} use different row order/coverage"
                )
            if not np.array_equal(left.pair, right.pair) or not np.array_equal(
                left.label, right.label
            ):
                raise BenchmarkAuditError(
                    f"{left_name} and {right_name} fold {fold} use different test examples"
                )
            if np.array_equal(left.score, right.score):
                raise BenchmarkAuditError(
                    f"{left_name} and {right_name} fold {fold} predictions are "
                    "element-wise identical"
                )


def audit_milgnet_run(
    predictions: PredictionTable,
    *,
    optimizer_steps: int,
    min_score_sd: float = 1e-8,
    min_unique_scores: int = 3,
) -> dict[str, float | int]:
    """Reject the known MilGNet no-training/degenerate-score failure modes."""

    table = _prediction_table(
        predictions.row_id,
        predictions.pair,
        predictions.label,
        predictions.score,
    )
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, Real)
        or not np.isfinite(float(optimizer_steps))
        or not float(optimizer_steps).is_integer()
    ):
        raise BenchmarkAuditError("optimizer_steps must be a finite integer")
    optimizer_steps = int(optimizer_steps)
    if optimizer_steps <= 0:
        raise BenchmarkAuditError("MilGNet executed no optimizer steps")
    min_score_sd = float(min_score_sd)
    if not np.isfinite(min_score_sd) or min_score_sd < 0:
        raise BenchmarkAuditError("min_score_sd must be a finite non-negative value")
    if (
        isinstance(min_unique_scores, bool)
        or not isinstance(min_unique_scores, Real)
        or not np.isfinite(float(min_unique_scores))
        or not float(min_unique_scores).is_integer()
        or int(min_unique_scores) < 2
    ):
        raise BenchmarkAuditError("min_unique_scores must be an integer of at least 2")
    min_unique_scores = int(min_unique_scores)
    score_sd = float(np.std(table.score, ddof=0))
    unique_scores = int(np.unique(table.score).size)
    if score_sd <= min_score_sd:
        raise BenchmarkAuditError(
            f"MilGNet score SD {score_sd:.3g} is not above {min_score_sd:.3g}"
        )
    if unique_scores < min_unique_scores:
        raise BenchmarkAuditError(
            f"MilGNet produced only {unique_scores} unique scores; "
            f"at least {min_unique_scores} are required"
        )
    return {
        "Optimizer_Steps": optimizer_steps,
        "Score_SD": score_sd,
        "Unique_Scores": unique_scores,
    }


def aggregate_fold_metrics(
    folds: Iterable[Mapping[str, float]],
    *,
    metric_names: Sequence[str] = PRIMARY_METRICS + SECONDARY_METRICS + THRESHOLD_METRICS,
) -> dict[str, dict[str, float | int]]:
    """Compute fold mean and sample SD (``ddof=1``) for common metrics."""

    rows = list(folds)
    if len(rows) < 2:
        raise BenchmarkAuditError("sample SD requires at least two folds")
    result: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        try:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        except KeyError as error:
            raise BenchmarkAuditError(f"fold metrics are missing {metric}") from error
        if not np.isfinite(values).all():
            raise BenchmarkAuditError(f"fold metric {metric} contains NaN or infinity")
        result[metric] = {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "folds": int(len(values)),
        }
    return result
