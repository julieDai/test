"""Isolation and artifact-contract tests for refined ASTRA selection."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import astra.benchmark.ethnobotany_refined_selection as selection_module

from astra.benchmark.ethnobotany_refined_selection import (
    CANDIDATE_SCHEMA,
    EXPECTED_EXECUTION_CODE_PATHS,
    RefinedSelectionError,
    _canonical_digest,
    _audit_and_freeze_selection_with_protocol_loader as audit_and_freeze_selection,
    _evaluate_selected_folds_with_protocol_loader as evaluate_selected_folds,
    _validation_rows_without_outer_materialisation,
    _validate_frozen_selection_with_protocol as validate_frozen_selection,
    sha256,
)
from astra.benchmark.metrics import evaluate_validation_scores, save_predictions


class _Protocol:
    dataset = "ethnobotany"
    seed = 42
    n_herbs = 2
    n_diseases = 2
    n_folds = 2
    validation_fraction = 0.5

    def __init__(self, *, allow_outer_rows: bool = False) -> None:
        self.allow_outer_rows = allow_outer_rows
        self.fold_rows_calls = 0
        self.positive_pairs = np.zeros((8, 2), dtype=np.int64)
        self.positive_pairs[7] = [0, 0]
        self.positive_pairs[1] = [0, 1]
        self.positive_pairs[6] = [0, 0]
        self.positive_pairs[2] = [0, 1]
        self.negative_pairs = np.ones((8, 2), dtype=np.int64)
        self.negative_pairs[1] = [1, 0]
        self.negative_pairs[7] = [1, 1]
        self.negative_pairs[0] = [1, 0]
        self.negative_pairs[6] = [1, 1]
        pairs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        self._folds = {
            fold: SimpleNamespace(
                train_pairs=pairs.copy(),
                train_labels=labels.copy(),
                train_row_indices=np.arange(100, 104, dtype=np.int64),
                validation_pairs=pairs.copy(),
                validation_labels=labels.copy(),
                validation_row_indices=np.arange(4, dtype=np.int64),
                test_pairs=pairs.copy(),
                test_labels=labels.copy(),
                test_row_indices=np.arange(
                    fold * 10, fold * 10 + len(pairs), dtype=np.int64
                ),
            )
            for fold in (1, 2)
        }

    def fold_rows(self, fold: int) -> SimpleNamespace:
        self.fold_rows_calls += 1
        if not self.allow_outer_rows:
            raise AssertionError(
                "freeze/selection replay must not materialize outer-test rows"
            )
        return self._folds[fold]


class EthnobotanyRefinedSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="astra_refined_selection_"
        )
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text("{}", encoding="utf-8")
        self.semantic_path = self.root / "semantic.json"
        self.semantic_path.write_text(
            json.dumps(
                {"schema": "astra-ethnobotany-shared-semantic-pca/v1"}
            ),
            encoding="utf-8",
        )
        self.bundle_path = self.root / "bundle.json"
        self.bundle_path.write_text(
            json.dumps({"schema": "astra-ethnobotany-feature-bundle/v1"}),
            encoding="utf-8",
        )
        self.feature_receipt_path = self.root / "feature_receipt.json"
        self.feature_dependency_path = self.root / "feature_matrix.npy"
        np.save(
            self.feature_dependency_path,
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            allow_pickle=False,
        )

        def artifact(path: Path) -> dict[str, object]:
            return {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

        self.feature_receipt_path.write_text(
            json.dumps(
                {
                    "schema": (
                        "astra-ethnobotany-refined-feature-deep-audit-receipt/v1"
                    ),
                    "dataset": "ethnobotany",
                    "deep_audit": True,
                    "path_bound": True,
                    "all_dependencies_rehashed_on_reuse": True,
                    "bundle_manifest": artifact(self.bundle_path),
                    "semantic_reduction_manifest": artifact(self.semantic_path),
                    "dependencies": [
                        artifact(self.bundle_path),
                        artifact(self.semantic_path),
                        artifact(self.feature_dependency_path),
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.config_root = self.root / "configs"
        self.config_root.mkdir()
        self.plan = {
            "schema": "test-plan/v1",
            "dataset": "ethnobotany",
            "seed": 42,
            "outer_folds": [1, 2],
            "protocol_manifest": str(self.protocol_path),
            "protocol_manifest_sha256": sha256(self.protocol_path),
            "feature_bundle": str(self.bundle_path),
            "feature_bundle_manifest_sha256": sha256(self.bundle_path),
            "candidate_config_order": ["config_a", "config_b"],
            "epoch_selection": {"candidate_epochs": [30, 60]},
            "fixed_controls": {
                "batch_size": 1000,
                "latent_dimension": 64,
                "regularization_u": 0.001,
                "text_fusion_alpha": 1.0,
            },
            "controlled_stages": [
                {
                    "id": "selection_test",
                    "candidate_values": ["config_a", "config_b"],
                }
            ],
            "promotion_and_stop_rules": {
                "per_fold_practical_margin_auprc": 0.001,
                "hard_epoch_limit": 200,
            },
        }
        self.validation_pairs = np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64
        )
        self.validation_labels = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        self.validation_rows = np.asarray([7, 1, 9, 15], dtype=np.int64)
        self.u = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        self.i = np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64)

    def _config_source(
        self,
        config_id: str,
        *,
        chain_index: int,
        incumbent_id: str | None,
        variant: str | None = None,
    ) -> Path:
        suffix = f"__{variant}" if variant else ""
        path = self.config_root / f"{config_id}{suffix}.json"
        payload = {
            "schema": "astra-ethnobotany-refined-candidate-config/v1",
            "config": {
                "id": config_id,
                "stage_id": "selection_test",
                "candidate_value": config_id,
                "chain_index": chain_index,
                "incumbent_id": incumbent_id,
                "base_config_id": "config_a",
            },
            "training": {
                "epochs": 200,
                "checkpoint_epochs": [30, 60],
                "batch_size": 1000,
                "learning_rate": 0.0001,
                "embedding_dimension": 64,
                "reg_u": 0.001,
                "text_fusion_alpha": 1.0,
            },
            "model_options": {},
        }
        if path.exists():
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _resolved_hyperparameters() -> dict[str, object]:
        return {
            "model_class": "ASTRARefined",
            "seed": 42,
            "max_epochs": 200,
            "learning_rate": 0.0001,
            "batch_size": 1000,
            "embedding_dimension": 64,
            "reg_u": 0.001,
            "n_layers": 5,
            "loss_kind": "legacy",
            "loss_reduction": "sum",
            "gradient_clip_value": 10.0,
            "gradient_clip_norm": 0.5,
            "protein_gate": False,
            "dead_variables": "preserve",
            "clique_sampling": "legacy_prefix",
            "clique_max_pairs_per_hyperedge": 100,
            "clique_max_directed_edges": 200000,
            "checkpoint_epochs": [30, 60],
            "semantic_adapter": "identity",
            "semantic_input_dim": 64,
            "semantic_freeze_input": False,
            "trace_gradients": True,
            "text_fusion_alpha": 1.0,
            "c0_compatible": True,
        }

    def _write_plan(
        self,
        *,
        selection: Path,
        manifests: list[Path],
        outer_folds: list[int] | None = None,
        candidate_order: list[str] | None = None,
        output_root: Path | None = None,
    ) -> None:
        folds = outer_folds or [1, 2]
        order = candidate_order or ["config_a", "config_b"]
        grouped: dict[str, dict[str, object]] = {}
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_id = payload["config"]["id"]
            record = grouped.setdefault(
                config_id,
                {
                    "config_id": config_id,
                    "stage_id": payload["config"]["stage_id"],
                    "config_path": payload["config_source"]["path"],
                    "config_sha256": payload["config_source"]["sha256"],
                    "semantic_manifest_path": str(self.semantic_path),
                    "semantic_manifest_sha256": sha256(self.semantic_path),
                    "folds": [],
                },
            )
            if payload["config_source"]["sha256"] != record["config_sha256"]:
                record.setdefault("fold_config_sources", {})[str(payload["fold"])] = {
                    "path": payload["config_source"]["path"],
                    "sha256": payload["config_source"]["sha256"],
                }
            record["folds"].append(int(payload["fold"]))
        source_paths = EXPECTED_EXECUTION_CODE_PATHS
        registry = []
        for record in grouped.values():
            registry.append(
                {
                    **record,
                    "allowed_changes_from_incumbent": [],
                }
            )
        plan = {
            **self.plan,
            "outer_folds": folds,
            "candidate_config_order": order,
            "controlled_stages": [
                {"id": "selection_test", "candidate_values": order}
            ],
            "selection_freeze": {
                "schema": "astra-ethnobotany-refined-freeze-contract/v1",
                "canonical_selection_path": str(selection),
                "outer_test_output_root": str(
                    output_root or (self.root / "test_outputs")
                ),
                "execution_code": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                    }
                    for name, path in source_paths.items()
                },
                "candidate_config_registry": registry,
                "semantic_feature_registry": [
                    {
                        "semantic_manifest_path": str(self.semantic_path),
                        "semantic_manifest_sha256": sha256(self.semantic_path),
                        "deep_audit_receipt_path": str(
                            self.feature_receipt_path
                        ),
                        "deep_audit_receipt_sha256": sha256(
                            self.feature_receipt_path
                        ),
                    }
                ],
                "stage_ledger": [
                    {
                        "stage_id": "selection_test",
                        "status": "completed",
                        "candidates": list(grouped.values()),
                    }
                ],
            },
        }
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def _freeze(
        self,
        *,
        manifests: list[Path],
        selection: Path,
        contract_manifests: list[Path] | None = None,
        outer_folds: list[int] | None = None,
        candidate_order: list[str] | None = None,
        output_root: Path | None = None,
    ):
        self._write_plan(
            selection=selection,
            manifests=contract_manifests or manifests,
            outer_folds=outer_folds,
            candidate_order=candidate_order,
            output_root=output_root,
        )
        return audit_and_freeze_selection(
            plan_path=self.plan_path,
            candidate_manifests=manifests,
            selection_path=selection,
            protocol_loader=lambda _: _Protocol(),
        )

    @staticmethod
    def _rewrite_frozen_selection(selection: Path, payload: dict[str, object]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        selection.write_text(encoded, encoding="utf-8")
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        selection.with_name(selection.name + ".sha256").write_text(
            f"{digest}  {selection.name}\n", encoding="ascii"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _candidate(
        self,
        *,
        fold: int,
        config_id: str,
        snapshots: list[tuple[int, np.ndarray]],
        group: str = "candidates",
        chain_index: int | None = None,
        incumbent_id: str | None = None,
        config_variant: str | None = None,
        complete_epoch_grid: bool = True,
    ) -> Path:
        root = self.root / group / f"fold_{fold}" / config_id
        root.mkdir(parents=True)
        if complete_epoch_grid:
            supplied = {int(epoch): scores for epoch, scores in snapshots}
            fallback = snapshots[0][1]
            snapshots = [
                (epoch, supplied.get(epoch, fallback)) for epoch in (30, 60)
            ]
        expected_rows = _validation_rows_without_outer_materialisation(
            _Protocol(), fold
        )
        validation_pairs = np.asarray(expected_rows.validation_pairs, dtype=np.int64)
        validation_labels = np.asarray(expected_rows.validation_labels, dtype=np.uint8)
        validation_rows = np.asarray(
            expected_rows.validation_row_indices, dtype=np.int64
        )
        train_rows = np.asarray(expected_rows.train_row_indices, dtype=np.int64)
        records = []
        for epoch, scores in snapshots:
            logits = np.log(scores / (1.0 - scores))
            u_path = root / f"u_epoch_{epoch}.npy"
            i_path = root / f"i_epoch_{epoch}.npy"
            np.save(u_path, np.eye(2, dtype=np.float64), allow_pickle=False)
            np.save(
                i_path,
                np.asarray(
                    [[logits[0], logits[2]], [logits[1], logits[3]]],
                    dtype=np.float64,
                ),
                allow_pickle=False,
            )
            predictions = root / f"validation_epoch_{epoch}.npz"
            save_predictions(
                predictions,
                row_id=validation_rows,
                pair=validation_pairs,
                label=validation_labels,
                score=scores,
            )
            auprc = evaluate_validation_scores(
                validation_labels, scores
            )["Validation_AUPRC"]
            records.append(
                {
                    "epoch": epoch,
                    "validation_auprc": auprc,
                    "validation_predictions": {
                        "path": predictions.name,
                        "sha256": sha256(predictions),
                    },
                    "u_embedding": {
                        "path": u_path.name,
                        "sha256": sha256(u_path),
                    },
                    "i_embedding": {
                        "path": i_path.name,
                        "sha256": sha256(i_path),
                    },
                    "training": {
                        "mean_training_loss": 1.0 / epoch,
                        "preclip_gradient_norm": 0.25,
                        "mean_preclip_gradient_norm": 0.25,
                        "mean_pre_global_clip_gradient_norm": 0.25,
                        "mean_postclip_gradient_norm": 0.25,
                        "preclip_exceeds_clip_norm_fraction": 0.0,
                        "global_clip_trigger_fraction": 0.0,
                        "gradient_trace_batches": 1,
                        "optimizer_steps": epoch,
                    },
                }
            )
        if chain_index is None:
            chain_index = 0 if config_id == "config_a" else 1
        if chain_index == 0:
            incumbent_id = None
        elif incumbent_id is None:
            incumbent_id = "config_a"
        config_path = self._config_source(
            config_id,
            chain_index=chain_index,
            incumbent_id=incumbent_id,
            variant=config_variant,
        )
        manifest = root / "candidate.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "fold": fold,
                    "validation_only": True,
                    "outer_evaluation_rows_loaded": False,
                    "config": {
                        "id": config_id,
                        "stage_id": "selection_test",
                        "candidate_value": config_id,
                        "chain_index": chain_index,
                        "incumbent_id": incumbent_id,
                        "base_config_id": "config_a",
                    },
                    "config_source": {
                        "schema": (
                            "astra-ethnobotany-refined-candidate-config/v1"
                        ),
                        "path": str(config_path.resolve()),
                        "sha256": sha256(config_path),
                    },
                    "protocol": {
                        "path": str(self.protocol_path.resolve()),
                        "sha256": sha256(self.protocol_path),
                        "seed": 42,
                    },
                    "split": {
                        "train_rows": len(train_rows),
                        "validation_rows": len(validation_rows),
                        "train_row_indices_sha256": hashlib.sha256(
                            np.ascontiguousarray(train_rows).tobytes()
                        ).hexdigest(),
                        "validation_row_indices_sha256": hashlib.sha256(
                            np.ascontiguousarray(validation_rows).tobytes()
                        ).hexdigest(),
                    },
                    "features": {
                        "base_bundle": {
                            "path": str(self.bundle_path.resolve()),
                            "sha256": sha256(self.bundle_path),
                        },
                        "semantic_reduction": {
                            "path": str(self.semantic_path.resolve()),
                            "sha256": sha256(self.semantic_path),
                        },
                    },
                    "training_diagnostics": {
                        "resolved_hyperparameters": self._resolved_hyperparameters()
                    },
                    "snapshots": records,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def _complete_candidates(self) -> list[Path]:
        # Both configurations intentionally tie. The second, more complex
        # candidate therefore cannot displace the lower-complexity incumbent.
        perfect = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        manifests = []
        for fold in (1, 2):
            manifests.append(
                self._candidate(
                    fold=fold,
                    config_id="config_a",
                    snapshots=[(60, perfect)],
                )
            )
            manifests.append(
                self._candidate(
                    fold=fold,
                    config_id="config_b",
                    snapshots=[(30, perfect)],
                )
            )
        return manifests

    def test_exact_tie_uses_earlier_epoch_within_config_and_retains_incumbent(self) -> None:
        perfect = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        manifests = []
        for fold in (1, 2):
            manifests.append(
                self._candidate(
                    fold=fold,
                    config_id="config_a",
                    snapshots=[(60, perfect), (30, perfect)],
                )
            )
            manifests.append(
                self._candidate(
                    fold=fold,
                    config_id="config_b",
                    snapshots=[(30, perfect)],
                )
            )
        selection = self.root / "first" / "selection.json"
        payload = self._freeze(
            manifests=manifests,
            selection=selection,
        )
        self.assertEqual(payload["folds"]["1"]["selected_config"]["id"], "config_a")
        self.assertEqual(payload["folds"]["1"]["selected_epoch"], 30)
        trace = payload["folds"]["1"]["promotion_trace"]
        self.assertEqual(trace[-1]["decision"], "retain_lower_complexity_incumbent")
        self.assertFalse(trace[-1]["promoted"])

    def test_candidate_above_practical_margin_is_promoted(self) -> None:
        weak = np.asarray([0.4, 0.3, 0.7, 0.6], dtype=np.float64)
        strong = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        incumbent = self._candidate(
            fold=1,
            config_id="config_a",
            snapshots=[(30, weak)],
            group="promotion_candidates",
        )
        candidate = self._candidate(
            fold=1,
            config_id="config_b",
            snapshots=[(30, strong)],
            group="promotion_candidates",
        )
        selected = self._freeze(
            manifests=[candidate, incumbent],
            selection=self.root / "promoted" / "selection.json",
            outer_folds=[1],
        )
        self.assertEqual(selected["folds"]["1"]["selected_config"]["id"], "config_b")
        self.assertTrue(selected["folds"]["1"]["promotion_trace"][-1]["promoted"])

    def test_failed_candidate_cannot_be_used_as_next_incumbent(self) -> None:
        perfect = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        manifests = [
            self._candidate(
                fold=1,
                config_id="config_a",
                snapshots=[(30, perfect)],
                group="stopped_candidates",
            ),
            self._candidate(
                fold=1,
                config_id="config_b",
                snapshots=[(30, perfect)],
                group="stopped_candidates",
            ),
            self._candidate(
                fold=1,
                config_id="config_c",
                snapshots=[(30, perfect)],
                group="stopped_candidates",
                chain_index=2,
                incumbent_id="config_b",
            ),
        ]
        with self.assertRaisesRegex(
            RefinedSelectionError, "retained incumbent is 'config_a'"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "stopped" / "selection.json",
                outer_folds=[1],
                candidate_order=["config_a", "config_b", "config_c"],
            )

    def test_fold_specific_config_sources_support_divergent_incumbents(self) -> None:
        weak = np.asarray([0.4, 0.3, 0.7, 0.6], dtype=np.float64)
        strong = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        manifests = [
            self._candidate(
                fold=1,
                config_id="config_a",
                snapshots=[(30, weak)],
                group="divergent",
            ),
            self._candidate(
                fold=1,
                config_id="config_b",
                snapshots=[(30, strong)],
                group="divergent",
            ),
            self._candidate(
                fold=1,
                config_id="config_c",
                snapshots=[(30, strong)],
                group="divergent",
                chain_index=2,
                incumbent_id="config_b",
                config_variant="fold1",
            ),
            self._candidate(
                fold=2,
                config_id="config_a",
                snapshots=[(30, strong)],
                group="divergent",
            ),
            self._candidate(
                fold=2,
                config_id="config_b",
                snapshots=[(30, weak)],
                group="divergent",
            ),
            self._candidate(
                fold=2,
                config_id="config_c",
                snapshots=[(30, strong)],
                group="divergent",
                chain_index=2,
                incumbent_id="config_a",
                config_variant="fold2",
            ),
        ]
        selected = self._freeze(
            manifests=manifests,
            selection=self.root / "divergent_selection" / "selection.json",
            candidate_order=["config_a", "config_b", "config_c"],
        )
        self.assertEqual(
            selected["folds"]["1"]["selected_config"]["id"], "config_b"
        )
        self.assertEqual(
            selected["folds"]["2"]["selected_config"]["id"], "config_a"
        )

    def test_all_candidates_are_validated_before_selection_is_written(self) -> None:
        manifests = self._complete_candidates()
        forbidden = manifests[-1].parent / "outer_test" / "predictions.npz"
        forbidden.parent.mkdir()
        forbidden.touch()
        selection = self.root / "selection.json"
        with self.assertRaisesRegex(RefinedSelectionError, "forbidden test"):
            self._freeze(
                manifests=manifests,
                selection=selection,
            )
        self.assertFalse(selection.exists())
        self.assertFalse((self.root / "selection.json.sha256").exists())

    def test_selection_hash_detects_tampering(self) -> None:
        selection = self.root / "frozen" / "selection.json"
        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        validate_frozen_selection(
            selection, protocol_loader=lambda _: _Protocol()
        )
        selection.write_text(selection.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(RefinedSelectionError, "selection hash mismatch"):
            validate_frozen_selection(selection)

    def test_replay_rejects_rehashed_selection_or_claim_forgery(self) -> None:
        selection = self.root / "replay" / "selection.json"
        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        payload = json.loads(selection.read_text(encoding="utf-8"))
        payload["folds"]["1"]["selected_config"]["id"] = "forged"
        self._rewrite_frozen_selection(selection, payload)
        with self.assertRaisesRegex(
            RefinedSelectionError, "deterministic validation replay"
        ):
            validate_frozen_selection(
                selection, protocol_loader=lambda _: _Protocol()
            )

    def test_replay_rejects_rehashed_output_root_and_claims(self) -> None:
        selection = self.root / "receipt_replay" / "selection.json"
        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        payload = json.loads(selection.read_text(encoding="utf-8"))
        payload["freeze_contract"]["outer_test_output_root"] = str(
            self.root / "forged_test_root"
        )
        payload["selection_metric"] = "outer-test oracle AUPRC"
        payload["outer_test_pristine"] = True
        self._rewrite_frozen_selection(selection, payload)
        with self.assertRaisesRegex(
            RefinedSelectionError, "freeze-contract receipt differs"
        ):
            validate_frozen_selection(
                selection, protocol_loader=lambda _: _Protocol()
            )

    def test_replay_rejects_rehashed_claim_only_forgery(self) -> None:
        selection = self.root / "claim_replay" / "selection.json"
        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        payload = json.loads(selection.read_text(encoding="utf-8"))
        payload["selection_metric"] = "outer-test oracle AUPRC"
        self._rewrite_frozen_selection(selection, payload)
        with self.assertRaisesRegex(
            RefinedSelectionError, "selection audit claim is inconsistent"
        ):
            validate_frozen_selection(
                selection, protocol_loader=lambda _: _Protocol()
            )

    def test_freeze_rejects_candidate_missing_from_terminal_five_fold_set(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "missing" / "selection.json"
        with self.assertRaisesRegex(
            RefinedSelectionError,
            "candidate manifests differ from terminal stage ledger",
        ):
            self._freeze(
                manifests=manifests[:-1],
                contract_manifests=manifests,
                selection=selection,
            )
        self.assertFalse(selection.exists())

    def test_freeze_requires_training_code_closure_and_each_candidate_all_folds(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "contract" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        del plan["selection_freeze"]["execution_code"]["model"]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "missing required source.*model"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["selection_freeze"]["stage_ledger"][0]["candidates"][0][
            "folds"
        ] = [1]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "must cover every outer fold"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

    def test_freeze_rejects_execution_source_under_wrong_key(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "wrong_source" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        selector = plan["selection_freeze"]["execution_code"]["selector"]
        plan["selection_freeze"]["execution_code"]["model"] = dict(selector)
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(RefinedSelectionError, "points to the wrong source"):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

    def test_precommitted_registry_rejects_multi_factor_config_change(self) -> None:
        manifests = self._complete_candidates()
        target_manifests = [
            path
            for path in manifests
            if json.loads(path.read_text(encoding="utf-8"))["config"]["id"]
            == "config_b"
        ]
        config_path = Path(
            json.loads(target_manifests[0].read_text(encoding="utf-8"))[
                "config_source"
            ]["path"]
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_options"]["astra.loss.kind"] = "bce_logits"
        config["model_options"]["astra.grad.clip_norm"] = 5.0
        config_path.write_text(json.dumps(config), encoding="utf-8")
        for manifest_path in target_manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["config_source"]["sha256"] = sha256(config_path)
            resolved = payload["training_diagnostics"]["resolved_hyperparameters"]
            resolved["loss_kind"] = "bce_logits"
            resolved["gradient_clip_norm"] = 5.0
            resolved["c0_compatible"] = False
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        selection = self.root / "multi_factor" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        registry = next(
            item
            for item in plan["selection_freeze"]["candidate_config_registry"]
            if item["config_id"] == "config_b"
        )
        registry["allowed_changes_from_incumbent"] = [
            "model_options.astra.loss.kind"
        ]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "violates its single-factor config contract"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

    def test_deep_feature_dependency_drift_is_rejected(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "feature_drift" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        np.save(
            self.feature_dependency_path,
            np.asarray([[9.0, 9.0]], dtype=np.float32),
            allow_pickle=False,
        )
        with self.assertRaisesRegex(
            RefinedSelectionError, "dependency.*differs from its deep-audit receipt"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

    def test_plan_gate_rejects_unstructured_hashed_evidence(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "bad_gate" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        gate_stage = {
            "id": "S0_gate",
            "pass_rule": "all folds must pass the equivalence gate",
        }
        plan["controlled_stages"].insert(0, gate_stage)
        evidence_path = self.root / "bad_gate_evidence.json"
        evidence_path.write_text("{}", encoding="utf-8")
        plan["selection_freeze"]["stage_ledger"].insert(
            0,
            {
                "stage_id": "S0_gate",
                "status": "completed",
                "gate_only": True,
                "evidence": [
                    {
                        "path": str(evidence_path),
                        "sha256": sha256(evidence_path),
                    }
                ],
            },
        )
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "does not prove the passed outcome"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

        evidence_path.write_text(
            json.dumps(
                {
                    "schema": "astra-ethnobotany-refined-stage-evidence/v1",
                    "stage_id": "S0_gate",
                    "plan_stage_sha256": _canonical_digest(gate_stage),
                    "outcome": "passed",
                }
            ),
            encoding="utf-8",
        )
        plan["selection_freeze"]["stage_ledger"][0]["evidence"][0][
            "sha256"
        ] = sha256(evidence_path)
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        audit_and_freeze_selection(
            plan_path=self.plan_path,
            candidate_manifests=manifests,
            selection_path=selection,
            protocol_loader=lambda _: _Protocol(),
        )

    def test_runtime_receipt_legacy_compatibility_is_exact_and_fold_scoped(self) -> None:
        manifests = self._complete_candidates()
        legacy_manifest = manifests[0]
        payload = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        resolved = payload["training_diagnostics"]["resolved_hyperparameters"]
        for field in ("learning_rate", "batch_size", "embedding_dimension", "reg_u"):
            del resolved[field]
        legacy_manifest.write_text(json.dumps(payload), encoding="utf-8")
        selection = self.root / "legacy_runtime" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        config_id = payload["config"]["id"]
        contract = next(
            item
            for item in plan["selection_freeze"]["stage_ledger"][0]["candidates"]
            if item["config_id"] == config_id
        )
        contract["legacy_resolved_runtime_folds"] = [int(payload["fold"])]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        audit_and_freeze_selection(
            plan_path=self.plan_path,
            candidate_manifests=manifests,
            selection_path=selection,
            protocol_loader=lambda _: _Protocol(),
        )

    def test_runtime_receipt_legacy_compatibility_rejects_partial_or_false_claim(self) -> None:
        manifests = self._complete_candidates()
        partial_manifest = manifests[0]
        payload = json.loads(partial_manifest.read_text(encoding="utf-8"))
        del payload["training_diagnostics"]["resolved_hyperparameters"][
            "learning_rate"
        ]
        partial_manifest.write_text(json.dumps(payload), encoding="utf-8")
        selection = self.root / "partial_runtime" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        contract = next(
            item
            for item in plan["selection_freeze"]["stage_ledger"][0]["candidates"]
            if item["config_id"] == payload["config"]["id"]
        )
        contract["legacy_resolved_runtime_folds"] = [int(payload["fold"])]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "missing resolved runtime hyperparameters"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

        payload["training_diagnostics"]["resolved_hyperparameters"][
            "learning_rate"
        ] = 0.0001
        partial_manifest.write_text(json.dumps(payload), encoding="utf-8")
        selection = self.root / "false_legacy_runtime" / "selection.json"
        self._write_plan(selection=selection, manifests=manifests)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        contract = plan["selection_freeze"]["stage_ledger"][0]["candidates"][0]
        contract["legacy_resolved_runtime_folds"] = [1]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "compatibility was declared"
        ):
            audit_and_freeze_selection(
                plan_path=self.plan_path,
                candidate_manifests=manifests,
                selection_path=selection,
                protocol_loader=lambda _: _Protocol(),
            )

    def test_formal_entrypoints_do_not_accept_injected_protocol_loaders(self) -> None:
        for function in (
            selection_module.audit_and_freeze_selection,
            selection_module.validate_frozen_selection,
            selection_module.evaluate_selected_folds,
        ):
            self.assertNotIn("protocol_loader", inspect.signature(function).parameters)

    def test_freeze_rejects_incomplete_checkpoint_grid(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        manifests = [
            self._candidate(
                fold=1,
                config_id="config_a",
                snapshots=[(30, scores)],
                group="incomplete_epochs",
                complete_epoch_grid=False,
            )
        ]
        for fold, config_id in ((1, "config_b"), (2, "config_a"), (2, "config_b")):
            manifests.append(
                self._candidate(
                    fold=fold,
                    config_id=config_id,
                    snapshots=[(30, scores)],
                    group="incomplete_epochs",
                )
            )
        with self.assertRaisesRegex(
            RefinedSelectionError, "snapshot epochs are incomplete"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "incomplete" / "selection.json",
            )

    def test_freeze_rejects_validation_rows_not_in_hashed_protocol(self) -> None:
        manifests = self._complete_candidates()
        manifest_path = manifests[0]
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = payload["snapshots"][0]
        predictions_path = manifest_path.parent / snapshot["validation_predictions"]["path"]
        with np.load(predictions_path, allow_pickle=False) as predictions:
            pair = predictions["pair"].copy()
            label = predictions["label"].copy()
            score = predictions["score"].copy()
        save_predictions(
            predictions_path,
            row_id=np.arange(10, 14, dtype=np.int64),
            pair=pair,
            label=label,
            score=score,
        )
        snapshot["validation_predictions"]["sha256"] = sha256(predictions_path)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "validation row IDs differ from protocol"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "wrong_rows" / "selection.json",
            )

    def test_freeze_rejects_candidate_protocol_drift(self) -> None:
        manifests = self._complete_candidates()
        protocol_payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        protocol_payload["protocol"]["sha256"] = "0" * 64
        manifests[0].write_text(json.dumps(protocol_payload), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "protocol differs from locked plan"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "wrong_protocol" / "selection.json",
            )

    def test_freeze_rejects_resolved_hyperparameter_drift(self) -> None:
        manifests = self._complete_candidates()
        resolved_payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        resolved_payload["training_diagnostics"]["resolved_hyperparameters"][
            "gradient_clip_norm"
        ] = 5.0
        manifests[0].write_text(json.dumps(resolved_payload), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError,
            "resolved hyperparameter mismatch at gradient_clip_norm",
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "wrong_resolved" / "selection.json",
            )

    def test_freeze_rejects_nonfinite_or_wrong_step_training_trace(self) -> None:
        manifests = self._complete_candidates()
        manifest_path = manifests[0]
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["snapshots"][0]["training"]["mean_training_loss"] = float("nan")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "mean_training_loss must be finite"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "nonfinite_trace" / "selection.json",
            )

        payload["snapshots"][0]["training"]["mean_training_loss"] = 1.0
        payload["snapshots"][0]["training"]["optimizer_steps"] += 1
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError, "optimizer_steps mismatch"
        ):
            self._freeze(
                manifests=manifests,
                selection=self.root / "wrong_steps" / "selection.json",
            )

    def test_outer_score_rejects_protocol_path_object_root_and_fold_drift(self) -> None:
        manifests = self._complete_candidates()
        selection = self.root / "strict_score" / "selection.json"
        self._freeze(manifests=manifests, selection=selection)

        other_protocol = self.root / "other_protocol.json"
        other_protocol.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(
            RefinedSelectionError,
            "protocol manifest differs from frozen selection",
        ):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=other_protocol,
                output_root=self.root / "test_outputs",
                protocol_loader=lambda _: _Protocol(),
            )

        altered_protocol = _Protocol()
        altered_protocol.dataset = "other"
        with self.assertRaisesRegex(
            RefinedSelectionError,
            "loaded protocol dataset/seed differs from locked plan",
        ):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=self.root / "test_outputs",
                protocol_loader=lambda _: altered_protocol,
            )

        with self.assertRaisesRegex(
            RefinedSelectionError, "atomically reserve every selected fold"
        ):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=self.root / "test_outputs",
                folds=[1],
                protocol_loader=lambda _: _Protocol(),
            )

        with self.assertRaisesRegex(
            RefinedSelectionError, "output root differs from frozen canonical root"
        ):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=self.root / "alternate_test_outputs",
                protocol_loader=lambda _: _Protocol(),
            )
        self.assertFalse((self.root / "test_outputs").exists())

    def test_outer_score_rejects_fold_output_symlink_escape(self) -> None:
        selection = self.root / "symlink_selection" / "selection.json"
        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        escaped = self.root / "escaped"
        escaped.mkdir()
        output_root = self.root / "test_outputs"
        output_root.mkdir()
        (output_root / "fold_1").symlink_to(escaped, target_is_directory=True)
        with self.assertRaisesRegex(RefinedSelectionError, "must not be a symlink"):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=output_root,
                protocol_loader=lambda _: _Protocol(),
            )
        self.assertEqual(list(escaped.iterdir()), [])

    def test_test_scoring_is_zero_before_freeze_and_once_per_fold_after(self) -> None:
        selection = self.root / "formal" / "selection.json"
        calls: list[int] = []

        def counted_score(u: np.ndarray, i: np.ndarray, pairs: np.ndarray) -> np.ndarray:
            calls.append(len(pairs))
            logits = np.einsum("ij,ij->i", u[pairs[:, 0]], i[pairs[:, 1]])
            return 1.0 / (1.0 + np.exp(-logits))

        with self.assertRaisesRegex(RefinedSelectionError, "must both exist"):
            evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=self.root / "test_outputs",
                protocol_loader=lambda _: _Protocol(allow_outer_rows=True),
            )
        self.assertFalse((self.root / "test_outputs").exists())

        self._freeze(
            manifests=self._complete_candidates(),
            selection=selection,
        )
        scorer_path = (
            "astra.benchmark.ethnobotany_refined_selection."
            "_score_outer_test_once"
        )
        with patch(scorer_path, side_effect=counted_score):
            summary = evaluate_selected_folds(
                selection_path=selection,
                protocol_manifest=self.protocol_path,
                output_root=self.root / "test_outputs",
                protocol_loader=lambda _: _Protocol(allow_outer_rows=True),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(summary["test_score_calls"], 2)
        for fold in (1, 2):
            metrics = summary["folds"][str(fold)]
            self.assertEqual(metrics["Test_Models_Evaluated"], 1)
            self.assertTrue(metrics["Decision_Frozen_Before_Test"])
            self.assertTrue(
                (self.root / "test_outputs" / f"fold_{fold}" / "test_predictions.npz").is_file()
            )

        # A repeated command fails before invoking either fold's scorer.
        with patch(scorer_path, side_effect=counted_score):
            with self.assertRaisesRegex(RefinedSelectionError, "refusing to repeat"):
                evaluate_selected_folds(
                    selection_path=selection,
                    protocol_manifest=self.protocol_path,
                    output_root=self.root / "test_outputs",
                    protocol_loader=lambda _: _Protocol(),
                )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
