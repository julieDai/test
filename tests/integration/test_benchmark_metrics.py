from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from astra.benchmark.metrics import (
    BenchmarkAuditError,
    PredictionTable,
    aggregate_fold_metrics,
    audit_cross_method_predictions,
    audit_milgnet_run,
    evaluate_validation_scores,
    evaluate_validation_and_test,
    evaluate_test_oracle,
    load_predictions,
    save_predictions,
    select_f1_threshold,
    validate_prediction_coverage,
)


def _table(scores):
    return PredictionTable(
        row_id=np.array([10, 11, 12, 13]),
        pair=np.array([[1, 5], [2, 6], [3, 7], [4, 8]]),
        label=np.array([0, 1, 0, 1]),
        score=np.asarray(scores, dtype=np.float64),
    )


class BenchmarkMetricsTest(unittest.TestCase):
    def test_validation_only_metrics_are_explicitly_prefixed(self):
        result = evaluate_validation_scores(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
            thresholds=[0.0, 0.5, 1.0],
        )
        self.assertEqual(result["Validation_Selected_Threshold"], 0.5)
        self.assertEqual(result["Validation_AUROC"], 1.0)
        self.assertEqual(result["Validation_AUPRC"], 1.0)
        self.assertNotIn("AUROC", result)
        self.assertNotIn("AUPRC", result)

    def test_validation_threshold_is_applied_unchanged_to_test(self):
        selection = select_f1_threshold(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], thresholds=[0.0, 0.5, 1.0]
        )
        self.assertEqual(selection.threshold, 0.5)
        self.assertEqual(selection.f1, 1.0)
        result = evaluate_validation_and_test(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
            [0, 1, 0, 1], [0.2, 0.8, 0.7, 0.9],
            thresholds=[0.0, 0.5, 1.0],
        )
        self.assertEqual(result["Validation_Selected_Threshold"], 0.5)
        self.assertEqual(result["Operating_Threshold"], 0.5)
        self.assertAlmostEqual(result["AUROC"], 1.0)
        self.assertAlmostEqual(result["AUPRC"], 1.0)
        self.assertAlmostEqual(result["AveragePrecision"], 1.0)
        self.assertAlmostEqual(result["F1"], 0.8)
        self.assertIn("test labels not used", result["Threshold_Source"])

    def test_auprc_is_trapezoidal_and_average_precision_is_separate(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([1.0, 1.0, 0.5, 0.5])
        precision, recall, _ = precision_recall_curve(labels, scores)
        result = evaluate_validation_and_test(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
            labels, scores,
            thresholds=[0.0, 0.5, 1.0],
        )
        self.assertAlmostEqual(result["AUPRC"], auc(recall, precision))
        self.assertAlmostEqual(
            result["AveragePrecision"], average_precision_score(labels, scores)
        )
        self.assertNotEqual(result["AUPRC"], result["AveragePrecision"])
        self.assertIn("trapezoidal", result["AUPRC_Definition"])

    def test_lowest_threshold_wins_an_exact_f1_tie(self):
        selection = select_f1_threshold(
            [0, 1], [0.1, 0.9], thresholds=[0.2, 0.5, 0.8]
        )
        self.assertEqual(selection.threshold, 0.2)
        self.assertEqual(selection.tie_break, "lowest threshold")

    def test_sorted_threshold_scan_matches_brute_force_sklearn(self):
        generator = np.random.default_rng(42)
        labels = np.tile(np.array([0, 1], dtype=np.int64), 257)
        generator.shuffle(labels)
        # Rounding deliberately creates threshold-equality ties, exercising
        # the inclusive ``score >= threshold`` boundary.
        scores = np.round(generator.random(len(labels)), 2)
        thresholds = np.linspace(0.0, 1.0, 100)
        selection = select_f1_threshold(labels, scores, thresholds=thresholds)

        expected = []
        for threshold in thresholds:
            predicted = (scores >= threshold).astype(np.int64)
            expected.append(
                (
                    f1_score(labels, predicted, zero_division=0),
                    precision_score(labels, predicted, zero_division=0),
                    recall_score(labels, predicted, zero_division=0),
                    accuracy_score(labels, predicted),
                )
            )
        best = int(np.argmax([row[0] for row in expected]))
        self.assertEqual(selection.threshold, float(thresholds[best]))
        self.assertEqual(selection.f1, expected[best][0])
        self.assertEqual(selection.precision, expected[best][1])
        self.assertEqual(selection.recall, expected[best][2])
        self.assertEqual(selection.accuracy, expected[best][3])

    def test_legacy_test_oracle_is_explicitly_labelled(self):
        result = evaluate_test_oracle(
            [0, 1, 0, 1],
            [0.2, 0.8, 0.7, 0.9],
            thresholds=[0.0, 0.5, 1.0],
        )
        self.assertEqual(result["Operating_Threshold"], 0.5)
        self.assertAlmostEqual(result["F1"], 0.8)
        self.assertEqual(result["Test_Oracle_F1"], result["F1"])
        self.assertIn("test-label oracle", result["Threshold_Source"])

    def test_prediction_artifact_and_manifest_coverage_are_order_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "predictions.npz"
            save_predictions(
                artifact,
                row_id=np.array([12, 10, 13, 11]),
                pair=np.array([[3, 7], [1, 5], [4, 8], [2, 6]]),
                label=np.array([0, 0, 1, 1]),
                score=np.array([0.3, 0.1, 0.9, 0.8]),
            )
            with np.load(artifact, allow_pickle=False) as payload:
                self.assertEqual(set(payload.files), {"row_id", "pair", "label", "score"})
            aligned = validate_prediction_coverage(
                load_predictions(artifact),
                expected_row_id=np.array([10, 11, 12, 13]),
                expected_pair=np.array([[1, 5], [2, 6], [3, 7], [4, 8]]),
                expected_label=np.array([0, 1, 0, 1]),
            )
            self.assertEqual(aligned.row_id.tolist(), [10, 11, 12, 13])
            np.testing.assert_allclose(aligned.score, [0.1, 0.8, 0.3, 0.9])

    def test_prediction_validation_rejects_invalid_outputs(self):
        with self.assertRaisesRegex(BenchmarkAuditError, "row counts differ"):
            validate_prediction_coverage(
                _table([0.1, 0.8, 0.3, 0.9]),
                expected_row_id=np.array([10, 11, 12]),
                expected_pair=np.array([[1, 5], [2, 6], [3, 7]]),
                expected_label=np.array([0, 1, 0]),
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(BenchmarkAuditError, "NaN or infinity"):
                save_predictions(root / "nan.npz", row_id=[1, 2], pair=[[1, 2], [2, 3]], label=[0, 1], score=[0.2, np.nan])
            with self.assertRaisesRegex(BenchmarkAuditError, "constant"):
                save_predictions(root / "constant.npz", row_id=[1, 2], pair=[[1, 2], [2, 3]], label=[0, 1], score=[0.5, 0.5])

    def test_cross_method_audit_rejects_exact_fold_predictions(self):
        first = _table([0.1, 0.8, 0.3, 0.9])
        identical = _table([0.1, 0.8, 0.3, 0.9])
        with self.assertRaisesRegex(BenchmarkAuditError, "element-wise identical"):
            audit_cross_method_predictions({"SMGCL": {1: first}, "LHGCE": {1: identical}})
        different = _table([0.1, 0.8, 0.31, 0.9])
        audit_cross_method_predictions({"SMGCL": {1: first}, "LHGCE": {1: different}})
        with self.assertRaisesRegex(BenchmarkAuditError, "fold coverage differs"):
            audit_cross_method_predictions({"SMGCL": {1: first}, "LHGCE": {1: different, 2: different}})

    def test_milgnet_audit_requires_training_and_score_dispersion(self):
        good = _table([0.1, 0.8, 0.3, 0.9])
        diagnostics = audit_milgnet_run(good, optimizer_steps=5)
        self.assertEqual(diagnostics["Optimizer_Steps"], 5)
        self.assertEqual(diagnostics["Unique_Scores"], 4)
        for steps, pattern in ((0, "no optimizer steps"), (np.nan, "finite integer")):
            with self.assertRaisesRegex(BenchmarkAuditError, pattern):
                audit_milgnet_run(good, optimizer_steps=steps)
        with self.assertRaisesRegex(BenchmarkAuditError, "only 2 unique"):
            audit_milgnet_run(_table([0.1, 0.1, 0.9, 0.9]), optimizer_steps=1)
        with self.assertRaisesRegex(BenchmarkAuditError, "score SD"):
            audit_milgnet_run(_table([0.5, 0.500000001, 0.500000002, 0.500000003]), optimizer_steps=1)

    def test_fold_aggregation_uses_sample_standard_deviation(self):
        rows = [
            dict(AUROC=1.0, AUPRC=0.8, AveragePrecision=0.75, Accuracy=0.7, Precision=0.6, Recall=0.5, F1=0.4),
            dict(AUROC=3.0, AUPRC=1.0, AveragePrecision=0.95, Accuracy=0.9, Precision=0.8, Recall=0.7, F1=0.6),
        ]
        aggregate = aggregate_fold_metrics(rows)
        self.assertEqual(aggregate["AUROC"]["mean"], 2.0)
        self.assertAlmostEqual(aggregate["AUROC"]["sample_sd"], np.sqrt(2.0))
        self.assertEqual(aggregate["F1"]["folds"], 2)


if __name__ == "__main__":
    unittest.main()
