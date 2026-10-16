from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark.tcm_astra_epoch_boundary_plan import (
    EPSILON,
    SOURCE_PATHS,
    TCMEpochBoundaryPlanError,
    build_plan,
    freeze_plan,
    sha256,
    validate_frozen_plan,
)


class TCMEpochBoundaryPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_epoch_boundary_plan_")
        self.root = Path(self.temporary.name)
        self.input_file = self._text(self.root / "relations.txt", "0 1\n")
        self.protocol = self._json(
            self.root / "protocol.json",
            {
                "files": {
                    "relations": {
                        "path": str(self.input_file),
                        "bytes": self.input_file.stat().st_size,
                        "sha256": sha256(self.input_file),
                    }
                }
            },
        )
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.feature_matrix = self._text(self.bundle / "herb.txt", "0.1 0.2\n")
        self.row_ids = self._text(self.bundle / "herb.row_ids.json", '["0"]\n')
        self.bundle_manifest = self._json(
            self.bundle / "bundle_manifest.json",
            {
                "formal_eligible": True,
                "entities": {
                    "herb": {
                        "matrix": {
                            "relative_path": self.feature_matrix.name,
                            "bytes": self.feature_matrix.stat().st_size,
                            "sha256": sha256(self.feature_matrix),
                        },
                        "row_ids_sidecar": {
                            "relative_path": self.row_ids.name,
                            "bytes": self.row_ids.stat().st_size,
                            "sha256": sha256(self.row_ids),
                        },
                    }
                },
            },
        )
        self.parent_plan = self._json(self.root / "parent_plan.json", {"seed": 42})
        self.selection_payload = self._selection_payload()
        self.selection_payload.update(
            {
                "schema": "astra-tcm-hparam-selection/v1",
                "plan": str(self.parent_plan.resolve()),
                "plan_sha256": sha256(self.parent_plan),
                "selector_path": str(SOURCE_PATHS["parent_selector"].resolve()),
                "selector_sha256": sha256(SOURCE_PATHS["parent_selector"]),
            }
        )
        self.selection = self._json(
            self.root / "selection.json", self.selection_payload
        )
        self.receipt = self._json(
            self.root / "selection.json.receipt.json",
            {
                "schema": "astra-tcm-hparam-selection-receipt/v1",
                "selection": str(self.selection.resolve()),
                "selection_sha256": sha256(self.selection),
                "selector": str(SOURCE_PATHS["parent_selector"].resolve()),
                "selector_sha256": sha256(SOURCE_PATHS["parent_selector"]),
                "plan": str(self.parent_plan.resolve()),
                "plan_sha256": sha256(self.parent_plan),
                "candidate_root": str((self.root / "parent_candidates").resolve()),
                "outer_test_scored_before_receipt": False,
            },
        )
        self.context = SimpleNamespace(
            protocol=SimpleNamespace(n_folds=5, seed=42),
            path=self.parent_plan.resolve(),
            sha256=sha256(self.parent_plan),
            protocol_path=self.protocol.resolve(),
            feature_bundle=self.bundle.resolve(),
            feature_manifest=self.bundle_manifest.resolve(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _text(path: Path, value: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    @staticmethod
    def _json(path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def _candidate(self, fold: int, epoch: int, lr: float, auprc: float) -> dict[str, object]:
        lr_id = "3e3" if lr == 0.003 else "5e3"
        candidate_id = f"identity_e{epoch}_lr{lr_id}"
        root = self.root / "parent_candidates" / candidate_id / f"fold_{fold}"
        validation = self._text(root / "results" / "validation_predictions.npz", "v")
        herb = self._text(root / "herb.txt", "h")
        disease = self._text(root / "disease.txt", "d")
        metrics = self._text(root / "results" / "metrics.json", "{}")
        protocol = self._text(root / "protocol.json", "{}")
        return {
            "id": candidate_id,
            "fold": fold,
            "semantic_adapter": "identity",
            "epochs": epoch,
            "learning_rate": lr,
            "validation_auprc": auprc,
            "validation_auroc": auprc - 0.001,
            "validation_predictions": str(validation.resolve()),
            "validation_predictions_sha256": sha256(validation),
            "herb_embedding": str(herb.resolve()),
            "herb_embedding_sha256": sha256(herb),
            "disease_embedding": str(disease.resolve()),
            "disease_embedding_sha256": sha256(disease),
            "metrics": str(metrics.resolve()),
            "metrics_sha256": sha256(metrics),
            "protocol": str(protocol.resolve()),
            "protocol_sha256": sha256(protocol),
        }

    def _fold(self, fold: int, *, rising: bool) -> dict[str, object]:
        base = 0.90 + fold / 1000.0
        records = [
            self._candidate(fold, 16, 0.003, base - 0.002),
            self._candidate(fold, 16, 0.005, base),
            self._candidate(fold, 32, 0.003, base + (0.00005 if rising else -0.00005)),
            self._candidate(fold, 32, 0.005, base - 0.003),
        ]
        winner = max(
            records,
            key=lambda record: (
                float(record["validation_auprc"]),
                float(record["validation_auroc"]),
                -int(record["epochs"]),
                -float(record["learning_rate"]),
            ),
        )
        return {"winner": winner, "candidates": records}

    def _selection_payload(self) -> dict[str, object]:
        return {
            "outer_test_accessed": False,
            "outer_test_scored": False,
            "folds": {
                str(fold): self._fold(fold, rising=fold in {1, 2, 5})
                for fold in range(1, 6)
            },
        }

    def _replay(self):
        return (
            self.selection_payload,
            sha256(self.selection),
            self.receipt.resolve(),
            sha256(self.receipt),
            self.context,
        )

    def test_build_plan_triggers_each_fold_independently(self):
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_plan._validated_selection_replay",
            return_value=self._replay(),
        ):
            plan = build_plan(self.selection)

        self.assertEqual(plan["epsilon"], EPSILON)
        self.assertEqual(
            [fold for fold, record in plan["folds"].items() if record["trigger_epoch64"]],
            ["1", "2", "5"],
        )
        self.assertEqual(len(plan["folds"]["1"]["epoch64_candidates"]), 2)
        self.assertEqual(plan["folds"]["3"]["epoch64_candidates"], [])
        self.assertEqual(set(plan["source_freeze"]), set(SOURCE_PATHS))
        self.assertEqual(
            plan["parent"]["selection"]["sha256"], sha256(self.selection)
        )
        self.assertEqual(
            plan["input_freeze"]["relations"]["sha256"],
            sha256(self.input_file),
        )
        self.assertEqual(
            plan["feature_input_freeze"]["herb.matrix"]["sha256"],
            sha256(self.feature_matrix),
        )

    def test_freeze_is_non_overwriting_and_writes_hash_sidecar(self):
        output = self.root / "boundary_v2.json"
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_plan._validated_selection_replay",
            return_value=self._replay(),
        ):
            frozen = freeze_plan(stage1_selection=self.selection, output=output)
        sidecar = output.with_suffix(".json.sha256")
        self.assertTrue(sidecar.is_file())
        self.assertEqual(frozen["plan_sha256"], sha256(output))
        with self.assertRaisesRegex(TCMEpochBoundaryPlanError, "refusing to overwrite"):
            freeze_plan(stage1_selection=self.selection, output=output)

    def test_parent_winner_tamper_is_rejected(self):
        payload = self._selection_payload()
        payload["folds"]["1"]["winner"] = payload["folds"]["1"]["candidates"][0]
        replay = (
            payload,
            sha256(self.selection),
            self.receipt.resolve(),
            sha256(self.receipt),
            self.context,
        )
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_plan._validated_selection_replay",
            return_value=replay,
        ):
            with self.assertRaisesRegex(TCMEpochBoundaryPlanError, "winner does not replay"):
                build_plan(self.selection)

    def test_frozen_plan_rejects_a_changed_feature_input(self):
        output = self.root / "boundary_v2.json"
        with patch(
            "astra.benchmark.tcm_astra_epoch_boundary_plan._validated_selection_replay",
            return_value=self._replay(),
        ):
            freeze_plan(stage1_selection=self.selection, output=output)
        validated, _ = validate_frozen_plan(output)
        self.assertEqual(validated["epsilon"], EPSILON)
        self.feature_matrix.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(
            TCMEpochBoundaryPlanError, "feature entity herb matrix differs"
        ):
            validate_frozen_plan(output)


if __name__ == "__main__":
    unittest.main()
