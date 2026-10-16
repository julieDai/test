"""Fail-closed tests for nested LHGCE epoch-budget selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from astra.benchmark.metrics import (
    BenchmarkAuditError,
    evaluate_validation_and_test,
    load_predictions,
    save_predictions,
)
from astra.official_baselines.common import validate_frozen_input_manifest
from astra.official_baselines.lhgce_budget_selection import (
    CANDIDATE_BUDGETS,
    UPSTREAM_COMMIT,
    _select_budget,
    select_and_evaluate_lhgce_budgets,
)


@dataclass(frozen=True)
class _FoldRows:
    train_pairs: np.ndarray
    train_labels: np.ndarray
    train_row_id: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray
    validation_row_id: np.ndarray
    test_pairs: np.ndarray
    test_labels: np.ndarray
    test_row_id: np.ndarray


class _Protocol:
    dataset = "tcm-suite"
    seed = 42

    def __init__(self) -> None:
        self._folds = {fold: self._rows(fold) for fold in (1, 2)}

    @staticmethod
    def _rows(fold: int) -> _FoldRows:
        offset = fold * 10_000
        pairs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        return _FoldRows(
            train_pairs=pairs.copy(),
            train_labels=labels.copy(),
            train_row_id=offset + np.arange(4, dtype=np.int64),
            validation_pairs=pairs.copy(),
            validation_labels=labels.copy(),
            validation_row_id=offset + 100 + np.arange(4, dtype=np.int64),
            test_pairs=pairs.copy(),
            test_labels=labels.copy(),
            test_row_id=offset + 200 + np.arange(4, dtype=np.int64),
        )

    def fold_rows(self, fold: int) -> _FoldRows:
        return self._folds[fold]

    def input_manifest(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "counts": {"herbs": 2, "diseases": 2},
        }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class LHGCEBudgetSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_lhgce_budget_test_")
        self.root = Path(self.temporary.name)
        self.protocol = _Protocol()
        self.candidate_roots = {
            budget: self.root / f"epoch_{budget}" / "tasks" / "LHGCE"
            for budget in CANDIDATE_BUDGETS
        }
        self._write_grid()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validation_scores(self, fold: int, budget: int) -> np.ndarray:
        winner = 50 if fold == 1 else 20
        if budget == winner:
            return np.asarray([0.05, 0.95, 0.15, 0.85], dtype=np.float64)
        # Nonconstant probabilities with deliberately imperfect ordering.
        shift = CANDIDATE_BUDGETS.index(budget) * 0.001
        return np.asarray([0.10 + shift, 0.60, 0.80, 0.40 - shift], dtype=np.float64)

    def _write_grid(self) -> None:
        input_manifest = self.protocol.input_manifest()
        lock = validate_frozen_input_manifest(input_manifest)
        for budget, method_root in self.candidate_roots.items():
            for fold in (1, 2):
                rows = self.protocol.fold_rows(fold)
                result_dir = method_root / f"fold_{fold}" / "results"
                validation_scores = self._validation_scores(fold, budget)
                test_scores = np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float64)
                save_predictions(
                    result_dir / "validation_predictions.npz",
                    row_id=rows.validation_row_id,
                    pair=rows.validation_pairs,
                    label=rows.validation_labels,
                    score=validation_scores,
                )
                save_predictions(
                    result_dir / "test_predictions.npz",
                    row_id=rows.test_row_id,
                    pair=rows.test_pairs,
                    label=rows.test_labels,
                    score=test_scores,
                )
                evaluated = evaluate_validation_and_test(
                    rows.validation_labels,
                    validation_scores,
                    rows.test_labels,
                    test_scores,
                )
                metrics = {
                    **evaluated,
                    "Dataset": "tcm-suite",
                    "Fold": fold,
                    "Seed": 42,
                    "Epochs_Requested": budget,
                    "Epochs_Executed": budget - 1,
                    "Optimizer_Steps": budget - 1,
                    "Training_Rows": len(rows.train_pairs),
                    "Training_Chunk_Size": len(rows.train_pairs),
                    "Evaluation_Chunk_Size": len(rows.test_pairs),
                    "Evaluation_Mode": "validation",
                    "Smoke": False,
                    "Formal_Eligible": True,
                    "Chunked_Loss_Exact": True,
                    "CP_Included": False,
                    "Upstream_Commit": UPSTREAM_COMMIT,
                    "Device": "cuda:0",
                }
                _write_json(result_dir / "metrics.json", metrics)
                _write_json(
                    method_root / f"fold_{fold}" / "protocol.json",
                    {
                        **input_manifest,
                        "frozen_input_lock_sha256": lock,
                        "dataset": "tcm-suite",
                        "protocol_manifest": str(self.root / "manifest.json"),
                        "optimizer_steps": budget - 1,
                        "evaluation_mode": "validation",
                        "training_split": "inner training fold",
                        "arguments": {
                            "fold": fold,
                            "epochs": budget,
                            "seed": 42,
                            "learning_rate": 1e-3,
                            "layers": 1,
                            "neurons": 168,
                            "agg_linear": "mul",
                            "agg_conv": "var",
                            "agg_hetero": "sum",
                            "train_chunk_size": len(rows.train_pairs),
                            "eval_chunk_size": len(rows.test_pairs),
                            "smoke": False,
                            "legacy_test_oracle": False,
                            "include_mapped_cp": False,
                            "source_dir": "/locked/LHGCE_official",
                        },
                        "model": metrics,
                        "adapter_sources": {
                            "common.py": "a" * 64,
                            "lhgce_identity.py": "b" * 64,
                            "lhgce_runner.py": "c" * 64,
                        },
                        "runtime_versions": {"python": "3.10.20"},
                        "runtime_module_origins": {"torch": "/locked/torch"},
                    },
                )

    def test_validation_only_selection_loads_one_selected_test_per_fold(self) -> None:
        calls: list[Path] = []

        def recording_loader(path: str | Path):
            calls.append(Path(path))
            return load_predictions(path)

        result = select_and_evaluate_lhgce_budgets(
            protocol=self.protocol,
            protocol_manifest=self.root / "manifest.json",
            candidate_roots=self.candidate_roots,
            output_dir=self.root / "selection",
            folds=(1, 2),
            prediction_loader=recording_loader,
        )

        decisions = {
            int(row["fold"]): int(row["selected_budget"])
            for row in result["fold_decisions"]
        }
        self.assertEqual(decisions, {1: 50, 2: 20})
        validation_calls = [path for path in calls if path.name == "validation_predictions.npz"]
        test_calls = [path for path in calls if path.name == "test_predictions.npz"]
        self.assertEqual(len(validation_calls), len(CANDIDATE_BUDGETS) * 2)
        self.assertEqual(len(test_calls), 2)
        self.assertIn(self.candidate_roots[50] / "fold_1" / "results" / "test_predictions.npz", test_calls)
        self.assertIn(self.candidate_roots[20] / "fold_2" / "results" / "test_predictions.npz", test_calls)
        self.assertEqual(result["unselected_test_prediction_files_read"], 0)
        self.assertTrue((self.root / "selection" / "selection_decision.json").is_file())
        self.assertTrue((self.root / "selection" / "summary.json").is_file())
        self.assertTrue((self.root / "selection" / "fold_metrics.csv").is_file())

    def test_exact_validation_tie_selects_lower_budget(self) -> None:
        selected = _select_budget(
            {
                50: {"AUPRC": 0.9},
                20: {"AUPRC": 0.9},
                100: {"AUPRC": 0.8},
            }
        )
        self.assertEqual(selected, 20)

    def test_incomplete_grid_fails_before_any_prediction_is_loaded(self) -> None:
        missing = (
            self.candidate_roots[1000]
            / "fold_2"
            / "results"
            / "test_predictions.npz"
        )
        missing.unlink()
        calls: list[Path] = []

        def recording_loader(path: str | Path):
            calls.append(Path(path))
            return load_predictions(path)

        with self.assertRaisesRegex(BenchmarkAuditError, "complete candidate grid"):
            select_and_evaluate_lhgce_budgets(
                protocol=self.protocol,
                protocol_manifest=self.root / "manifest.json",
                candidate_roots=self.candidate_roots,
                output_dir=self.root / "incomplete",
                folds=(1, 2),
                prediction_loader=recording_loader,
            )
        self.assertEqual(calls, [])

    def test_metadata_tampering_fails_after_selection_but_before_test_load(self) -> None:
        tampered = self.candidate_roots[100] / "fold_1" / "protocol.json"
        payload = json.loads(tampered.read_text(encoding="utf-8"))
        payload["arguments"]["epochs"] = 101
        _write_json(tampered, payload)
        calls: list[Path] = []

        def recording_loader(path: str | Path):
            calls.append(Path(path))
            return load_predictions(path)

        with self.assertRaisesRegex(BenchmarkAuditError, "protocol epoch budget differs"):
            select_and_evaluate_lhgce_budgets(
                protocol=self.protocol,
                protocol_manifest=self.root / "manifest.json",
                candidate_roots=self.candidate_roots,
                output_dir=self.root / "tampered",
                folds=(1, 2),
                prediction_loader=recording_loader,
            )
        self.assertFalse(any(path.name == "test_predictions.npz" for path in calls))
        self.assertTrue((self.root / "tampered" / "selection_decision.json").is_file())
        self.assertFalse((self.root / "tampered" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
