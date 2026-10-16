from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark import astra_runner
from astra.benchmark.tcm_astra_epoch_boundary_plan import PLAN_SCHEMA
from astra.benchmark.tcm_astra_epoch_boundary_runner import (
    TCMEpochBoundaryRunnerError,
    _candidate_for_fold,
    candidate_hyperparameters,
    reject_outer_artifacts,
    validate_candidate,
)


class TCMEpochBoundaryRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="astra_epoch_boundary_runner_"
        )
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text("{}", encoding="utf-8")
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text("{}", encoding="utf-8")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.output = self.root / "output"
        self.plan = {
            "schema": PLAN_SCHEMA,
            "seed": 42,
            "protocol_manifest": {"path": str(self.protocol.resolve())},
            "feature_bundle": {"path": str(self.bundle.resolve())},
            "folds": {
                "1": {
                    "fold": 1,
                    "trigger_epoch64": True,
                    "epoch64_candidates": [
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
                    ],
                },
                "2": {
                    "fold": 2,
                    "trigger_epoch64": False,
                    "epoch64_candidates": [],
                },
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, **updates) -> argparse.Namespace:
        values = {
            "dataset": "tcm-suite",
            "validation_only": True,
            "smoke": False,
            "diagnostic": False,
            "text_fusion_alpha": 1.0,
            "alpha_selection_manifest": None,
            "herb_semantic": None,
            "disease_semantic": None,
            "legacy_ethnobotany_semantic_dir": None,
            "boundary_plan": self.plan_path,
            "seed": 42,
            "protocol_manifest": self.protocol,
            "embedding_dir": self.bundle,
            "fold": 1,
            "candidate_id": "identity_e64_lr3e3",
            "output_root": self.output,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_only_triggered_fold_and_planned_arm_are_accepted(self):
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_runner.validate_frozen_plan",
            return_value=(self.plan, "a" * 64),
        ):
            binding = validate_candidate(self._args())
            self.assertEqual(binding["epochs"], 64)
            self.assertEqual(binding["learning_rate"], 0.003)
            with self.assertRaisesRegex(
                TCMEpochBoundaryRunnerError, "did not trigger"
            ):
                validate_candidate(
                    self._args(fold=2, candidate_id="identity_e64_lr3e3")
                )

    def test_non_validation_execution_is_rejected_before_plan_access(self):
        with self.assertRaisesRegex(TCMEpochBoundaryRunnerError, "validation-only"):
            validate_candidate(self._args(validation_only=False))

    def test_semantic_path_override_is_rejected(self):
        with self.assertRaisesRegex(
            TCMEpochBoundaryRunnerError, "frozen feature-bundle semantics"
        ):
            validate_candidate(self._args(herb_semantic=self.root / "other.npy"))

    def test_hyperparameters_are_overridden_and_restored(self):
        original = astra_runner._HYPERPARAMETERS["tcm-suite"]
        with candidate_hyperparameters(epochs=64, learning_rate=0.003):
            active = astra_runner._HYPERPARAMETERS["tcm-suite"]
            self.assertEqual(active["epochs"], 64)
            self.assertEqual(active["learning_rate"], 0.003)
        self.assertIs(astra_runner._HYPERPARAMETERS["tcm-suite"], original)

    def test_outer_prediction_artifact_is_rejected(self):
        forbidden = self.output / "results" / "test_predictions.npz"
        forbidden.parent.mkdir(parents=True)
        forbidden.write_bytes(b"not-a-real-prediction")
        with self.assertRaisesRegex(TCMEpochBoundaryRunnerError, "outer-test artifact"):
            reject_outer_artifacts(self.output)

    def test_candidate_shape_contract_rejects_an_extra_field(self):
        self.plan["folds"]["1"]["epoch64_candidates"][0]["unexpected"] = True
        with self.assertRaisesRegex(TCMEpochBoundaryRunnerError, "fields differ"):
            _candidate_for_fold(
                self.plan, fold=1, candidate_id="identity_e64_lr3e3"
            )


if __name__ == "__main__":
    unittest.main()
