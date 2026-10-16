"""Regression tests for validation-only SMGCL budget selection."""

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
from astra.official_baselines.audit_results import _assert_formal_tcm_budget
from astra.official_baselines.common import sha256, write_json
from astra.official_baselines.smgcl_budget_selection import (
    CANDIDATE_BUDGETS,
    UPSTREAM_COMMIT,
    _select_budget,
    select_and_evaluate_smgcl_budgets,
)
from astra.official_baselines.smgcl_runner import (
    _score_sha256,
    _validation_ranking,
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


class SMGCLBudgetSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_smgcl_budget_test_")
        self.root = Path(self.temporary.name)
        self.protocol = _Protocol()
        self.roots = {
            budget: self.root / f"epoch_{budget}" / "tasks" / "SMGCL"
            for budget in CANDIDATE_BUDGETS
        }
        self._write_grid()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validation_scores(self, fold: int, budget: int) -> np.ndarray:
        winner = 48 if fold == 1 else 30
        if budget == winner:
            return np.asarray([0.05, 0.95, 0.15, 0.85], dtype=np.float64)
        shift = CANDIDATE_BUDGETS.index(budget) * 0.001
        return np.asarray([0.10 + shift, 0.60, 0.80, 0.40 - shift], dtype=np.float64)

    def _write_grid(self) -> None:
        manifest = self.protocol.input_manifest()
        lock = validate_frozen_input_manifest(manifest)
        for budget, method_root in self.roots.items():
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
                    "Epochs_Executed": budget,
                    "Batch_Size": 4096,
                    "Training_Rows": len(rows.train_pairs),
                    "Training_Positive_Graph_Rows": 2,
                    "Optimizer_Steps": budget,
                    "Evaluation_Mode": "validation",
                    "Smoke": False,
                    "Diagnostic": False,
                    "Execution_Mode": "formal-candidate",
                    "Formal_Eligible": True,
                    "Scalable_Mode": True,
                    "Scalable_Mode_Request": "on",
                    "Upstream_Commit": UPSTREAM_COMMIT,
                    "Device": "cuda",
                }
                arguments = {
                    "fold": fold,
                    "seed": 42,
                    "epochs": budget,
                    "batch_size": 4096,
                    "learning_rate": 5e-4,
                    "dropout": 0.4,
                    "embedding_dim": 64,
                    "neighbor_embedding_dim": 32,
                    "hidden_dims": [64, 32],
                    "herb_neighbors": 3,
                    "disease_neighbors": 3,
                    "loss_fn": "focal",
                    "alpha": 0.5,
                    "gamma": 2.0,
                    "restart_probability": 0.1,
                    "rwr_mode": "official-exact",
                    "scalable_mode": "on",
                    "legacy_test_oracle": False,
                    "smoke": False,
                    "diagnostic": False,
                    "source_dir": "/locked/SMGCL_official",
                    "cache_dir": "/locked/cache",
                }
                _write_json(result_dir / "metrics.json", metrics)
                _write_json(
                    method_root / f"fold_{fold}" / "protocol.json",
                    {
                        **manifest,
                        "frozen_input_lock_sha256": lock,
                        "protocol_manifest": str(self.root / "manifest.json"),
                        "dataset": "tcm-suite",
                        "diagnostic": False,
                        "execution_mode": "formal-candidate",
                        "formal_eligible": True,
                        "model": metrics,
                        "arguments": arguments,
                        "adapter_sources": {
                            "common.py": "a" * 64,
                            "smgcl_runner.py": "b" * 64,
                        },
                        "similarity_cache_sha256": "c" * 64,
                        "training_split": "inner training fold",
                        "evaluation_mode": "validation",
                        "runtime_versions": {"python": "3.10.20"},
                        "runtime_environment_isolation_enforced": True,
                        "runtime_module_origins": {"torch": "/locked/torch"},
                    },
                )

    def test_selects_on_validation_and_opens_only_selected_tests(self) -> None:
        calls: list[Path] = []

        def loader(path: str | Path):
            calls.append(Path(path))
            return load_predictions(path)

        result = select_and_evaluate_smgcl_budgets(
            protocol=self.protocol,
            protocol_manifest=self.root / "manifest.json",
            candidate_roots=self.roots,
            output_dir=self.root / "selection",
            folds=(1, 2),
            prediction_loader=loader,
        )
        decisions = {
            int(row["fold"]): int(row["selected_budget"])
            for row in result["fold_decisions"]
        }
        self.assertEqual(decisions, {1: 48, 2: 30})
        validation_calls = [path for path in calls if path.name == "validation_predictions.npz"]
        test_calls = [path for path in calls if path.name == "test_predictions.npz"]
        self.assertEqual(len(validation_calls), 8)
        self.assertEqual(len(test_calls), 2)
        self.assertIn(self.roots[48] / "fold_1" / "results" / "test_predictions.npz", test_calls)
        self.assertIn(self.roots[30] / "fold_2" / "results" / "test_predictions.npz", test_calls)
        self.assertEqual(result["unselected_test_prediction_files_read"], 0)

    def test_incomplete_grid_fails_before_loading_validation(self) -> None:
        (self.roots[64] / "fold_2" / "results" / "test_predictions.npz").unlink()
        calls: list[Path] = []

        def loader(path: str | Path):
            calls.append(Path(path))
            return load_predictions(path)

        with self.assertRaisesRegex(BenchmarkAuditError, "complete candidate grid"):
            select_and_evaluate_smgcl_budgets(
                protocol=self.protocol,
                protocol_manifest=self.root / "manifest.json",
                candidate_roots=self.roots,
                output_dir=self.root / "missing",
                folds=(1, 2),
                prediction_loader=loader,
            )
        self.assertEqual(calls, [])

    def test_exact_validation_tie_selects_lower_budget(self) -> None:
        self.assertEqual(
            _select_budget(
                {
                    16: {"AUPRC": 0.91},
                    30: {"AUPRC": 0.91},
                    48: {"AUPRC": 0.90},
                    64: {"AUPRC": 0.89},
                }
            ),
            16,
        )

    def test_tampered_budget_fails_before_any_test_is_loaded(self) -> None:
        path = self.roots[16] / "fold_1" / "results" / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["Epochs_Executed"] = 15
        _write_json(path, metrics)
        calls: list[Path] = []

        def loader(candidate: str | Path):
            calls.append(Path(candidate))
            return load_predictions(candidate)

        with self.assertRaisesRegex(BenchmarkAuditError, "Epochs_Executed"):
            select_and_evaluate_smgcl_budgets(
                protocol=self.protocol,
                protocol_manifest=self.root / "manifest.json",
                candidate_roots=self.roots,
                output_dir=self.root / "tampered",
                folds=(1, 2),
                prediction_loader=loader,
            )
        self.assertFalse(any(path.name == "test_predictions.npz" for path in calls))
        self.assertTrue((self.root / "tampered" / "selection_decision.json").is_file())
        self.assertFalse((self.root / "tampered" / "summary.json").exists())

    def test_unified_auditor_recomputes_single_trajectory_selection(self) -> None:
        task = self.root / "single_trajectory" / "tasks" / "SMGCL" / "fold_1"
        result_dir = task / "results"
        rows = self.protocol.fold_rows(1)
        checkpoint_records: list[dict[str, object]] = []
        checkpoint_scores: dict[int, np.ndarray] = {}
        for epoch in CANDIDATE_BUDGETS:
            scores = self._validation_scores(1, epoch)
            checkpoint_scores[epoch] = scores
            path = result_dir / "validation_checkpoints" / f"epoch_{epoch}.npz"
            save_predictions(
                path,
                row_id=rows.validation_row_id,
                pair=rows.validation_pairs,
                label=rows.validation_labels,
                score=scores,
            )
            checkpoint_records.append(
                {
                    "Epoch": epoch,
                    **_validation_ranking(rows.validation_labels, scores),
                    "Validation_Score_SHA256": _score_sha256(scores),
                    "Validation_Runtime_Seconds": 1.0,
                    "Validation_Predictions": (
                        f"results/validation_checkpoints/epoch_{epoch}.npz"
                    ),
                    "Validation_Predictions_SHA256": sha256(path),
                }
            )
        selected = 48
        save_predictions(
            result_dir / "validation_predictions.npz",
            row_id=rows.validation_row_id,
            pair=rows.validation_pairs,
            label=rows.validation_labels,
            score=checkpoint_scores[selected],
        )
        selection = {
            "Mode": "inner-validation",
            "Candidates": list(CANDIDATE_BUDGETS),
            "Selection_Metric": "validation trapezoidal AUPRC",
            "Selection_Tie_Break": "lowest epoch on an exact AUPRC tie",
            "Selected_Epoch": selected,
            "Selected_Validation_AUPRC": 1.0,
            "Checkpoint_Validation_Metrics": checkpoint_records,
            "Outer_Test_Access": "selection frozen before test scoring",
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }
        write_json(result_dir / "epoch_selection.json", selection)
        metrics = {
            "Epochs_Requested": 64,
            "Epochs_Executed": 64,
            "Batch_Size": 4096,
            "Training_Rows": len(rows.train_pairs),
            "Optimizer_Steps": 64,
            "Scalable_Mode": True,
            "Scalable_Mode_Request": "on",
            "Epoch_Selection_Mode": "inner-validation",
            "Epoch_Candidates": list(CANDIDATE_BUDGETS),
            "Selected_Epoch": selected,
            "Selected_Checkpoint_Optimizer_Steps": selected,
            "Checkpoint_Validation_Metrics": checkpoint_records,
            "Epoch_Selection_Metric": "validation trapezoidal AUPRC",
            "Epoch_Selection_Tie_Break": "lowest epoch on an exact AUPRC tie",
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }
        _assert_formal_tcm_budget(
            recorded=metrics,
            run_protocol={"epoch_selection": selection},
            method="SMGCL",
            fold_rows=rows,
            task=task,
        )
        tampered = dict(metrics)
        tampered["Selected_Epoch"] = 64
        with self.assertRaises(BenchmarkAuditError):
            _assert_formal_tcm_budget(
                recorded=tampered,
                run_protocol={"epoch_selection": selection},
                method="SMGCL",
                fold_rows=rows,
                task=task,
            )


if __name__ == "__main__":
    unittest.main()
