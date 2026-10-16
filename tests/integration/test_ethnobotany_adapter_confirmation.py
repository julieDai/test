"""Focused isolation tests for the paired seed-43 confirmation workflow."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from astra.benchmark import ethnobotany_adapter_confirmation as confirmation


class _OuterProtocol:
    dataset = "ethnobotany"
    seed = 43
    n_folds = 5
    validation_fraction = 0.1
    pair_order = "seeded"

    def __init__(self) -> None:
        self.fold_rows_calls = 0

    def fold_rows(self, fold: int) -> SimpleNamespace:
        self.fold_rows_calls += 1
        pairs = np.asarray(
            [[0, 0], [1, 0], [2, 1], [3, 1], [0, 1], [1, 1]],
            dtype=np.int64,
        )
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        return SimpleNamespace(
            test_pairs=pairs,
            test_labels=labels,
            test_row_indices=np.arange(fold * 10, fold * 10 + len(pairs)),
        )


class AdapterConfirmationTest(unittest.TestCase):
    def test_snapshot_tie_uses_lower_epoch(self) -> None:
        selected = confirmation._choose_snapshot(
            [
                {"epoch": 200, "validation_auprc": 0.8},
                {"epoch": 30, "validation_auprc": 0.8},
                {"epoch": 60, "validation_auprc": 0.7},
            ]
        )
        self.assertEqual(selected["epoch"], 30)

    def test_forbidden_payload_scan_allows_only_isolation_flag(self) -> None:
        confirmation._reject_forbidden_candidate_payload(
            {"outer_evaluation_rows_loaded": False, "validation": "ok.npz"}
        )
        with self.assertRaisesRegex(
            confirmation.AdapterConfirmationError, "forbidden evaluation field"
        ):
            confirmation._reject_forbidden_candidate_payload({"test_scores": []})
        with self.assertRaisesRegex(
            confirmation.AdapterConfirmationError, "forbidden evaluation artifact"
        ):
            confirmation._reject_forbidden_candidate_payload(
                {"artifact": "snapshots/test_predictions.npz"}
            )

    def test_sample_standard_deviation_is_recorded(self) -> None:
        aggregate = confirmation._mean_std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(aggregate["mean"], 3.0)
        self.assertAlmostEqual(aggregate["std_sample"], np.sqrt(2.5))
        self.assertAlmostEqual(aggregate["std_population"], np.sqrt(2.0))

    def test_freeze_rejects_existing_candidate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}\n", encoding="utf-8")
            candidates = {
                arm: {
                    fold: root
                    / "run"
                    / "candidates"
                    / arm
                    / f"fold_{fold}"
                    / "candidate.json"
                    for fold in confirmation.FOLDS
                }
                for arm in confirmation.ARMS
            }
            candidates["baseline_linear"][1].parent.mkdir(parents=True)
            context = {
                "candidates": candidates,
                "selection_path": root / "run" / "selection.json",
                "outer_root": root / "run" / "outer_test",
            }
            with mock.patch.object(
                confirmation, "_validate_plan_payload", return_value=context
            ):
                with self.assertRaisesRegex(
                    confirmation.AdapterConfirmationError,
                    "candidate directory existed before plan freeze",
                ):
                    confirmation._freeze_plan_with_protocol_loader(
                        plan, protocol_loader=lambda _: object()
                    )
            self.assertFalse(
                confirmation._sidecar(plan, confirmation.PLAN_HASH_SUFFIX).exists()
            )

    def test_outer_score_reserves_all_ten_and_has_no_partial_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            u_path = root / "u.npy"
            i_path = root / "i.npy"
            np.save(
                u_path,
                np.asarray([[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
                allow_pickle=False,
            )
            np.save(i_path, np.asarray([[1.0, 0.0], [0.5, 0.0]]), allow_pickle=False)
            u_record = confirmation._record_for(u_path)
            i_record = confirmation._record_for(i_path)
            selection = {
                "folds": {
                    str(fold): {
                        "arms": {
                            arm: {
                                "recipe": confirmation._recipe_for_fold(fold),
                                "selected_epoch": 30,
                                "selected_validation_auprc": 0.75,
                                "frozen_f1_threshold": {
                                    "Validation_Selected_Threshold": 0.5
                                },
                                "u_embedding": u_record,
                                "i_embedding": i_record,
                            }
                            for arm in confirmation.ARMS
                        }
                    }
                    for fold in confirmation.FOLDS
                }
            }
            protocol = _OuterProtocol()
            outer_root = root / "outer_test"
            context = {
                "outer_root": outer_root,
                "summary_path": outer_root / "summary.json",
                "selection_path": root / "selection.json",
                "protocol": protocol,
                "protocol_record": {
                    "path": str(root / "protocol.json"),
                    "sha256": "a" * 64,
                },
                "payload": {
                    "comparison": {
                        "primary_metric": "AUPRC",
                        "minimum_mean_improvement": 0.001,
                        "minimum_improved_folds": 3,
                        "improved_fold_rule": "delta_gt_1e-12",
                    }
                },
            }
            selection_hash = "b" * 64
            calls: list[int] = []

            def score(u: np.ndarray, i: np.ndarray, pairs: np.ndarray) -> np.ndarray:
                if not calls:
                    reservations = list(
                        outer_root.glob("*/fold_*/.reserved_before_outer_score.json")
                    )
                    self.assertEqual(len(reservations), 10)
                calls.append(len(pairs))
                return confirmation._score_embeddings(u, i, pairs)

            with mock.patch.object(
                confirmation,
                "_load_verified_selection",
                return_value=(selection, selection_hash, context),
            ):
                summary = confirmation._score_with_protocol_loader(
                    root / "plan.json",
                    protocol_loader=lambda _: protocol,
                    score_callback=score,
                )
                self.assertEqual(summary["outer_score_calls"], 10)
                self.assertEqual(len(calls), 10)
                self.assertEqual(protocol.fold_rows_calls, 5)
                self.assertTrue((outer_root / "summary.json").is_file())
                self.assertTrue((outer_root / "summary.json.sha256").is_file())
                self.assertEqual(len(summary["artifacts"]), 5)
                with self.assertRaisesRegex(
                    confirmation.AdapterConfirmationError,
                    "refusing repeated or partial outer scoring",
                ):
                    confirmation._score_with_protocol_loader(
                        root / "plan.json",
                        protocol_loader=lambda _: protocol,
                        score_callback=score,
                    )

    def test_score_cli_exposes_no_fold_subset_argument(self) -> None:
        parser = confirmation._parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["score", "--plan", "plan.json", "--folds", "1"]
                )


if __name__ == "__main__":
    unittest.main()
