"""Validation-isolation tests for the independent ASTRARefined runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from astra.benchmark.ethnobotany_refined_runner import (
    CHECKPOINT_EPOCHS,
    CONFIG_SCHEMA,
    PreparedFeatures,
    RefinedCandidateRunnerError,
    TrainingResult,
    derive_training_validation_rows,
    run_validation_candidate,
    validate_semantic_reduction_manifest,
)
from astra.benchmark.ethnobotany_refined_selection import (
    audit_candidate_manifest,
    sha256,
)
from astra.benchmark.features import FEATURE_ROW_IDS_SCHEMA


class _Protocol:
    dataset = "ethnobotany"
    seed = 42
    n_folds = 5
    validation_fraction = 0.25
    positive_pairs = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.int64
    )
    negative_pairs = np.asarray(
        [[0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]], dtype=np.int64
    )
    herb_compound_pairs = np.asarray([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
    compound_protein_pairs = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    protein_disease_pairs = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    entity_ids = {
        "herb": ("h0", "h1", "h2"),
        "compound": ("c0", "c1"),
        "protein": ("p0", "p1"),
        "disease": ("d0", "d1", "d2", "d3"),
    }
    source_files = {
        "mapping/herb": {"sha256": "a" * 64},
        "mapping/disease": {"sha256": "b" * 64},
    }

    def compact_to_raw(self, entity: str, index: int) -> str:
        return self.entity_ids[entity][index]

    def fold_rows(self, fold: int):  # pragma: no cover - failure sentinel
        raise AssertionError("the candidate runner must not call fold_rows")


class EthnobotanyRefinedRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_refined_runner_")
        self.root = Path(self.temporary.name)
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text("{}", encoding="utf-8")
        self.data_dir = self.root / "data"
        self.embedding_dir = self.root / "embeddings"
        self.data_dir.mkdir()
        self.embedding_dir.mkdir()
        self.semantic_manifest = self.root / "semantic.json"
        self.semantic_manifest.write_text("{}", encoding="utf-8")
        self.semantic_receipt = self.root / "semantic.deep-audit-receipt.json"
        self.semantic_receipt.write_text("{}", encoding="utf-8")
        self.config_path = self.root / "config.json"
        self.config = {
            "schema": CONFIG_SCHEMA,
            "config": {
                "id": "base",
                "stage_id": "S1_epoch_budget",
                "candidate_value": 30,
                "chain_index": 0,
                "incumbent_id": None,
                "base_config_id": "base",
            },
            "training": {
                "epochs": 200,
                "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "batch_size": 4,
                "learning_rate": 0.0001,
                "embedding_dimension": 64,
                "reg_u": 0.001,
                "text_fusion_alpha": 1.0,
            },
            "model_options": {
                "astra.semantic.input_dim": 64,
                "astra.semantic.adapter": "identity",
            },
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _features(self, **kwargs) -> PreparedFeatures:
        return PreparedFeatures(
            paths={},
            mask_paths={},
            diagnostics={"semantic_reduction": {"output_dimension": 64}},
        )

    def test_split_matches_locked_nested_algorithm_without_fold_rows(self) -> None:
        protocol = _Protocol()
        observed = derive_training_validation_rows(protocol, 2)

        positive_rows = np.arange(len(protocol.positive_pairs), dtype=np.int64)
        negative_local = np.arange(len(protocol.negative_pairs), dtype=np.int64)
        negative_rows = len(protocol.positive_pairs) + negative_local
        positive_outer = positive_rows[positive_rows % 5 != 1]
        negative_outer = negative_rows[negative_local % 5 != 1]
        positive_outer = positive_outer[
            np.random.default_rng(np.random.SeedSequence((42, 2, 1))).permutation(
                len(positive_outer)
            )
        ]
        negative_outer = negative_outer[
            np.random.default_rng(np.random.SeedSequence((42, 2, 0))).permutation(
                len(negative_outer)
            )
        ]
        # floor(4 * .25) == one validation row per class.
        expected_validation = np.concatenate((positive_outer[:1], negative_outer[:1]))
        expected_training = np.concatenate((positive_outer[1:], negative_outer[1:]))
        np.testing.assert_array_equal(observed.validation_row_indices, expected_validation)
        np.testing.assert_array_equal(observed.train_row_indices, expected_training)
        self.assertFalse(observed.validation_row_indices.flags.writeable)

    def test_one_trajectory_emits_four_auditable_validation_snapshots(self) -> None:
        calls: list[dict[str, object]] = []

        def features_with_receipt(**kwargs) -> PreparedFeatures:
            self.assertEqual(
                kwargs["semantic_deep_audit_receipt"],
                self.semantic_receipt.resolve(),
            )
            return PreparedFeatures(
                paths={},
                mask_paths={},
                diagnostics={
                    "semantic_reduction": {
                        "output_dimension": 64,
                        "deep_audit": True,
                        "deep_audit_receipt": {
                            "path": str(self.semantic_receipt.resolve()),
                            "sha256": sha256(self.semantic_receipt),
                        },
                    }
                },
            )

        def trainer(**kwargs) -> TrainingResult:
            calls.append(kwargs)
            rows = kwargs["rows"]
            self.assertEqual(rows.fold, 1)
            snapshots = {}
            for epoch in CHECKPOINT_EPOCHS:
                # The dot products vary by pair while remaining finite.
                u = np.asarray(
                    [[1.0 + epoch / 1000.0, 0.1], [0.2, 0.9], [0.6, 0.4]],
                    dtype=np.float64,
                )
                i = np.asarray(
                    [[0.9, 0.1], [0.1, 0.8], [-0.5, 0.2], [0.2, -0.4]],
                    dtype=np.float64,
                )
                snapshots[epoch] = {
                    "u": u,
                    "i": i,
                    "mean_training_loss": 1.0 / epoch,
                    "preclip_gradient_norm": 0.5,
                    "optimizer_steps": epoch * 2,
                }
            return TrainingResult(snapshots=snapshots, diagnostics={"calls": 1})

        output = self.root / "candidate"
        payload = run_validation_candidate(
            protocol_manifest=self.protocol_path,
            data_dir=self.data_dir,
            embedding_dir=self.embedding_dir,
            semantic_reduction_manifest=self.semantic_manifest,
            semantic_deep_audit_receipt=self.semantic_receipt,
            config_json=self.config_path,
            fold=1,
            output_root=output,
            protocol_loader=lambda _: _Protocol(),
            feature_preparer=features_with_receipt,
            trainer=trainer,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual([item["epoch"] for item in payload["snapshots"]], [30, 60, 100, 200])
        self.assertTrue(payload["validation_only"])
        self.assertFalse(payload["outer_evaluation_rows_loaded"])
        self.assertEqual(payload["config"]["incumbent_id"], None)
        self.assertEqual(payload["config"]["base_config_id"], "base")
        self.assertEqual(payload["config"]["chain_index"], 0)
        semantic_audit = payload["features"]["semantic_reduction"]
        self.assertTrue(semantic_audit["deep_audit"])
        self.assertEqual(
            semantic_audit["deep_audit_receipt"]["path"],
            str(self.semantic_receipt.resolve()),
        )
        self.assertEqual(
            semantic_audit["deep_audit_receipt"]["sha256"],
            sha256(self.semantic_receipt),
        )
        self.assertFalse(
            any("test" in path.name.casefold() for path in output.rglob("*") if path.is_file())
        )

        plan = {
            "outer_folds": [1],
            "candidate_config_order": ["base"],
            "epoch_selection": {"candidate_epochs": list(CHECKPOINT_EPOCHS)},
        }
        audited = audit_candidate_manifest(output / "candidate.json", plan=plan)
        self.assertEqual(len(audited["snapshots"]), 4)
        self.assertEqual(audited["config"]["base_config_id"], "base")

        hd_lines = (output / "protocol_relations" / "H_D.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        supplied = calls[0]["rows"]
        self.assertEqual(len(hd_lines), int(np.sum(supplied.train_labels == 1)))

    def test_manifest_dimension_must_match_model_input(self) -> None:
        def features_128(**kwargs) -> PreparedFeatures:
            return PreparedFeatures(
                paths={},
                mask_paths={},
                diagnostics={"semantic_reduction": {"output_dimension": 128}},
            )

        with self.assertRaisesRegex(
            RefinedCandidateRunnerError, "semantic input dimension"
        ):
            run_validation_candidate(
                protocol_manifest=self.protocol_path,
                data_dir=self.data_dir,
                embedding_dir=self.embedding_dir,
                semantic_reduction_manifest=self.semantic_manifest,
                config_json=self.config_path,
                fold=1,
                output_root=self.root / "mismatch",
                protocol_loader=lambda _: _Protocol(),
                feature_preparer=features_128,
                trainer=lambda **_: self.fail("trainer must not run"),
            )

    def test_feature_receipt_fast_path_bypasses_recursive_feature_preparer(self) -> None:
        feature_receipt = self.root / "feature.deep-audit-receipt.json"
        feature_receipt.write_text("{}", encoding="utf-8")
        calls: list[dict[str, object]] = []

        def receipt_validator(**kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["receipt_path"], feature_receipt.resolve())
            self.assertEqual(
                kwargs["semantic_deep_audit_receipt"],
                self.semantic_receipt.resolve(),
            )
            return {
                "paths": {},
                "mask_paths": {},
                "diagnostics": {
                    "semantic_reduction": {"output_dimension": 64},
                    "feature_deep_audit_receipt": {
                        "all_dependencies_rehashed": True
                    },
                },
            }

        with self.assertRaisesRegex(RuntimeError, "trainer reached"):
            run_validation_candidate(
                protocol_manifest=self.protocol_path,
                data_dir=self.data_dir,
                embedding_dir=self.embedding_dir,
                semantic_reduction_manifest=self.semantic_manifest,
                semantic_deep_audit_receipt=self.semantic_receipt,
                feature_deep_audit_receipt=feature_receipt,
                config_json=self.config_path,
                fold=1,
                output_root=self.root / "receipt_fast_path",
                protocol_loader=lambda _: _Protocol(),
                feature_preparer=lambda **_: self.fail(
                    "recursive feature preparer must not run"
                ),
                feature_receipt_validator=receipt_validator,
                trainer=lambda **_: (_ for _ in ()).throw(
                    RuntimeError("trainer reached")
                ),
            )
        self.assertEqual(len(calls), 1)

    def test_s4b_zero_pad_requires_frozen_linear_128_frontend(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["config"].update(
            {
                "stage_id": "S4b_semantic_retained_dimension",
                "candidate_value": (
                    "shared_pca64_zeropad128_frozen_linear128to64"
                ),
            }
        )
        config["model_options"] = {
            "astra.semantic.input_dim": 128,
            "astra.semantic.adapter": "linear",
            "astra.semantic.freeze_input": False,
        }
        config_path = self.root / "invalid_s4b.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        def zero_pad_features(**kwargs) -> PreparedFeatures:
            return PreparedFeatures(
                paths={},
                mask_paths={},
                diagnostics={
                    "semantic_reduction": {
                        "schema": (
                            "astra-ethnobotany-semantic-pca64-zero-pad128-control/v1"
                        ),
                        "output_dimension": 128,
                    }
                },
            )

        with self.assertRaisesRegex(
            RefinedCandidateRunnerError, "zero-pad128 control requires frozen"
        ):
            run_validation_candidate(
                protocol_manifest=self.protocol_path,
                data_dir=self.data_dir,
                embedding_dir=self.embedding_dir,
                semantic_reduction_manifest=self.semantic_manifest,
                config_json=config_path,
                fold=1,
                output_root=self.root / "invalid_s4b_output",
                protocol_loader=lambda _: _Protocol(),
                feature_preparer=zero_pad_features,
                trainer=lambda **_: self.fail("trainer must not run"),
            )

    def test_semantic_audit_accepts_both_locked_dimensions(self) -> None:
        protocol = _Protocol()
        cases = (
            (64, "astra-ethnobotany-shared-semantic-pca/v1", "pca64"),
            (128, "astra-ethnobotany-shared-semantic-pca/v1", "pca128"),
            (
                128,
                "astra-ethnobotany-semantic-pca64-zero-pad128-control/v1",
                "pca64_zero_pad128",
            ),
        )
        for dimension, schema, case_name in cases:
            case = self.root / case_name
            case.mkdir()
            entities = {}
            audited_outputs = {}
            for entity, count, digest in (
                ("herb", 3, "a" * 64),
                ("disease", 4, "b" * 64),
            ):
                matrix = np.arange(count * dimension, dtype=np.float32).reshape(
                    count, dimension
                )
                matrix_path = case / f"{entity}.npy"
                np.save(matrix_path, matrix, allow_pickle=False)
                sidecar = matrix_path.with_name(matrix_path.name + ".row_ids.json")
                sidecar.write_text(
                    json.dumps(
                        {
                            "schema": FEATURE_ROW_IDS_SCHEMA,
                            "entity": entity,
                            "id_space": "protocol_raw_id",
                            "mapping_sha256": digest,
                            "row_ids": list(protocol.entity_ids[entity]),
                        }
                    ),
                    encoding="utf-8",
                )
                entities[entity] = {
                    "mapping_sha256": digest,
                    "matrix": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
                    "row_ids_sidecar": {"path": str(sidecar), "sha256": sha256(sidecar)},
                    "shape": [count, dimension],
                }
                audited_outputs[entity] = {
                    "matrix_path": matrix_path,
                    "matrix": matrix,
                }
            manifest = case / "semantic_reduction_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": schema,
                        "output_dimension": dimension,
                        "entities": entities,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "astra.benchmark.ethnobotany_refined_runner.validate_shared_semantic_reduction",
                return_value={"outputs": audited_outputs},
            ):
                paths, diagnostics = validate_semantic_reduction_manifest(
                    manifest, protocol=protocol
                )
            self.assertEqual(diagnostics["output_dimension"], dimension)
            self.assertEqual(set(paths), {"herb", "disease"})


if __name__ == "__main__":
    unittest.main()
