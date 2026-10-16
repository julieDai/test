from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark.tcm_astra_epoch_boundary_decision import (
    DECISION_SCHEMA,
    RECEIPT_SCHEMA,
    TCMEpochBoundaryDecisionError,
    build_decision,
    decide_fold,
    sha256,
    write_decision,
)
from astra.benchmark.tcm_astra_epoch_boundary_plan import EPSILON


class TCMEpochBoundaryDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="astra_epoch_boundary_decision_"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _record(
        *, fold: int, epoch: int, lr: float, auprc: float, candidate_id: str
    ) -> dict[str, object]:
        return {
            "id": candidate_id,
            "fold": fold,
            "epochs": epoch,
            "learning_rate": lr,
            "semantic_adapter": "identity",
            "validation_auprc": auprc,
            "validation_auroc": auprc - 0.001,
            "validation_row_id_sha256": f"row-{fold}",
            "validation_pair_sha256": f"pair-{fold}",
            "validation_label_sha256": f"label-{fold}",
        }

    def _fold_plan(self, fold: int, *, triggered: bool, best32: float) -> dict[str, object]:
        winner = self._record(
            fold=fold,
            epoch=32 if triggered else 16,
            lr=0.003,
            auprc=best32,
            candidate_id=(
                "identity_e32_lr3e3" if triggered else "identity_e16_lr3e3"
            ),
        )
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
            if triggered
            else []
        )
        return {
            "fold": fold,
            "stage1_winner": winner,
            "best_epoch32": self._record(
                fold=fold,
                epoch=32,
                lr=0.003,
                auprc=best32,
                candidate_id="identity_e32_lr3e3",
            ),
            "trigger_epoch64": triggered,
            "epoch64_candidates": candidates,
        }

    def test_exact_epsilon_stops_but_strictly_larger_gain_requires_128(self):
        fold_plan = self._fold_plan(1, triggered=True, best32=0.9)
        exact = [
            self._record(
                fold=1,
                epoch=64,
                lr=0.003,
                auprc=0.9 + EPSILON,
                candidate_id="identity_e64_lr3e3",
            ),
            self._record(
                fold=1,
                epoch=64,
                lr=0.005,
                auprc=0.89,
                candidate_id="identity_e64_lr5e3",
            ),
        ]
        stopped = decide_fold(fold_plan=fold_plan, epoch64_records=exact)
        self.assertFalse(stopped["requires_epoch128"])
        self.assertEqual(stopped["decision"], "stop-at-epoch32")

        exact[0]["validation_auprc"] = 0.9 + EPSILON + 1e-8
        continued = decide_fold(fold_plan=fold_plan, epoch64_records=exact)
        self.assertTrue(continued["requires_epoch128"])
        self.assertIsNone(continued["selected_candidate_if_stopped"])

    def test_untriggered_fold_rejects_epoch64_records(self):
        fold_plan = self._fold_plan(3, triggered=False, best32=0.8)
        with self.assertRaisesRegex(
            TCMEpochBoundaryDecisionError, "despite no trigger"
        ):
            decide_fold(
                fold_plan=fold_plan,
                epoch64_records=[
                    self._record(
                        fold=3,
                        epoch=64,
                        lr=0.003,
                        auprc=0.9,
                        candidate_id="identity_e64_lr3e3",
                    )
                ],
            )

    def test_build_decision_keeps_continuation_per_fold(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text("{}", encoding="utf-8")
        selection_path = self.root / "selection.json"
        selection_path.write_text("{}", encoding="utf-8")
        receipt_path = self.root / "selection.json.receipt.json"
        receipt_path.write_text("{}", encoding="utf-8")
        parent_plan = self.root / "parent.json"
        parent_plan.write_text("{}", encoding="utf-8")
        folds = {
            "1": self._fold_plan(1, triggered=True, best32=0.9000),
            "2": self._fold_plan(2, triggered=True, best32=0.9100),
            "3": self._fold_plan(3, triggered=False, best32=0.9200),
            "4": self._fold_plan(4, triggered=False, best32=0.9300),
            "5": self._fold_plan(5, triggered=False, best32=0.9400),
        }
        plan = {
            "seed": 42,
            "parent": {
                "selection": {
                    "path": str(selection_path.resolve()),
                    "sha256": sha256(selection_path),
                    "bytes": selection_path.stat().st_size,
                },
                "selection_receipt": {
                    "path": str(receipt_path.resolve()),
                    "sha256": sha256(receipt_path),
                    "bytes": receipt_path.stat().st_size,
                },
                "plan": {
                    "path": str(parent_plan.resolve()),
                    "sha256": sha256(parent_plan),
                    "bytes": parent_plan.stat().st_size,
                },
            },
            "source_freeze": {"source": {"sha256": "source"}},
            "source_freeze_sha256": "source-freeze",
            "input_freeze": {"input": {"sha256": "input"}},
            "input_freeze_sha256": "input-freeze",
            "feature_input_freeze": {"feature": {"sha256": "feature"}},
            "feature_input_freeze_sha256": "feature-freeze",
            "folds": folds,
        }
        context = SimpleNamespace(
            protocol=SimpleNamespace(n_folds=5, seed=42),
            sha256=sha256(parent_plan),
            path=parent_plan.resolve(),
        )

        def candidate_record(**kwargs):
            fold = int(kwargs["fold"])
            candidate = kwargs["candidate"]
            best32 = float(folds[str(fold)]["best_epoch32"]["validation_auprc"])
            if fold == 1:
                gain = EPSILON if candidate["learning_rate"] == 0.003 else -0.01
            else:
                gain = EPSILON + 1e-7 if candidate["learning_rate"] == 0.003 else -0.01
            record = self._record(
                fold=fold,
                epoch=64,
                lr=float(candidate["learning_rate"]),
                auprc=best32 + gain,
                candidate_id=str(candidate["id"]),
            )
            record.update(
                {
                    "runner_sha256": "runner",
                    "adapter_sources_sha256": "adapter",
                    "model_sources_sha256": "model",
                    "source_inputs_sha256": "inputs",
                    "selection_metric": "Validation_AUPRC",
                    "selection_metric_definition": "metric-definition",
                    "array_hash_schema": "array-v1",
                }
            )
            return record

        replay = (
            {"folds": {str(fold): {} for fold in range(1, 6)}},
            sha256(selection_path),
            receipt_path.resolve(),
            sha256(receipt_path),
            context,
        )
        with (
            patch(
                "astra.benchmark.tcm_astra_epoch_boundary_decision.validate_frozen_plan",
                return_value=(plan, sha256(plan_path)),
            ),
            patch(
                "astra.benchmark.tcm_astra_epoch_boundary_decision._validated_selection_replay",
                return_value=replay,
            ),
            patch(
                "astra.benchmark.tcm_astra_epoch_boundary_decision."
                "_derive_training_validation_rows",
                return_value=object(),
            ),
            patch(
                "astra.benchmark.tcm_astra_epoch_boundary_decision._validate_candidate",
                side_effect=candidate_record,
            ),
        ):
            decision = build_decision(
                plan_path=plan_path,
                candidate_root=self.root / "candidates",
            )

        self.assertEqual(decision["folds_requiring_epoch128"], [2])
        self.assertFalse(decision["folds"]["1"]["requires_epoch128"])
        self.assertTrue(decision["folds"]["2"]["requires_epoch128"])
        self.assertEqual(decision["folds"]["3"]["decision"], "stop-at-stage1")
        self.assertFalse(decision["outer_test_authorized"])

    def test_receipt_binds_parent_plan_sources_and_inputs(self):
        output = self.root / "decision.json"
        payload = {
            "schema": DECISION_SCHEMA,
            "decision_source": {"path": "decision.py", "sha256": "d", "bytes": 1},
            "plan": {"path": "plan.json", "sha256": "p"},
            "parent_selection": {"path": "selection.json", "sha256": "s"},
            "parent_selection_receipt": {"path": "receipt.json", "sha256": "r"},
            "parent_plan": {"path": "parent.json", "sha256": "parent"},
            "candidate_root": str((self.root / "candidates").resolve()),
            "source_freeze": {"runner": {"sha256": "runner"}},
            "source_freeze_sha256": "source-freeze",
            "input_freeze": {"relations": {"sha256": "input"}},
            "input_freeze_sha256": "input-freeze",
            "feature_input_freeze": {"herb": {"sha256": "feature"}},
            "feature_input_freeze_sha256": "feature-freeze",
        }
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_decision.build_decision",
            return_value=payload,
        ):
            written = write_decision(
                plan_path=self.root / "plan.json",
                candidate_root=self.root / "candidates",
                output=output,
            )
        receipt = json.loads(
            Path(written["receipt_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
        self.assertEqual(receipt["plan"], payload["plan"])
        self.assertEqual(receipt["parent_selection"], payload["parent_selection"])
        self.assertEqual(receipt["decision_sha256"], sha256(output))


if __name__ == "__main__":
    unittest.main()
