"""Dependency-light guards for the central Scheme-B HGHDA adapter."""

from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from astra.official_baselines import hghda_runner

from astra.official_baselines.hghda_runner import (
    HGHDAFairRunnerError,
    _rebind_model_mappings,
    _standardise_and_clip,
    _validate_tcm_author_feature_bundle,
    assert_author_feature_binding,
    assert_author_training_input,
    assert_split_isolation,
    build_parser,
    prepare_features,
    run,
    score_pairs,
    training_rows,
)
from astra.official_baselines.common import official_baseline_layout
from astra.benchmark.features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path
from astra.benchmark.astra_runner import _validate_ethnobotany_feature_bundle
from astra.benchmark import fair_comparison
from astra.benchmark.protocol import build_protocol
from astra.official_baselines import preflight
from astra.official_baselines.run import RUNNERS
from astra.benchmark.fair_comparison import (
    DEFAULT_METHODS as FAIR_DEFAULT_METHODS,
    METHODS as FAIR_METHODS,
    build_parser as build_fair_parser,
)


FEATURE_FILENAMES_FOR_TEST = {
    "compound": "compound_embedding_kpgt.txt",
    "protein": "protein_embedding_esm2.txt",
}


class _Protocol:
    entity_ids = {
        "herb": ("h10", "h20"),
        "compound": ("c10", "c20"),
        "protein": ("p10", "p20"),
        "disease": ("d10", "d20"),
    }

    def compact_to_raw(self, entity: str, index: int) -> str:
        return self.entity_ids[entity][index]


class HGHDAFairAdapterTest(unittest.TestCase):
    @staticmethod
    def _write_pairs(path: Path, pairs: list[tuple[int, int]]) -> None:
        path.write_text(
            "".join(f"{left}\t{right}\n" for left, right in pairs),
            encoding="utf-8",
        )

    @staticmethod
    def _write_mapping(path: Path, entity: str, count: int) -> None:
        path.write_text(
            f"{entity}_id,{entity}_external\n"
            + "".join(
                f"{index},{entity.upper()}-{index}\n"
                for index in range(1, count + 1)
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _artifact(bundle_root: Path, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "relative_path": str(path.resolve().relative_to(bundle_root.resolve())),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @classmethod
    def _write_bundle_manifest(
        cls,
        embedding_dir: Path,
        data_dir: Path,
        protocol: object,
    ) -> None:
        records: dict[str, dict[str, object]] = {"herb": {}, "disease": {}}
        for entity in ("compound", "protein"):
            matrix = embedding_dir / FEATURE_FILENAMES_FOR_TEST[entity]
            sidecar = feature_row_ids_path(matrix)
            mapping = embedding_dir / "mappings" / f"{entity}_id_mapping.csv"
            mapping.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_dir / mapping.name, mapping)
            mask = embedding_dir / f"{matrix.name}.missing_mask.npy"
            np.save(mask, np.zeros(len(protocol.entity_ids[entity]), dtype=bool))
            cache = embedding_dir / "retrieval" / f"{entity}_retrieval_cache.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {
                        "schema": "astra-ethnobotany-retrieval-cache/v2",
                        "entity": entity,
                        "mapping_sha256": protocol.source_files[
                            f"mapping/{entity}"
                        ]["sha256"],
                        "formal_eligible": True,
                        "missing_sequence_policy": "strict",
                        "status_counts": {
                            "resolved": len(protocol.entity_ids[entity]),
                            "target_not_protein": 0,
                            "failed": 0,
                            "pending": 0,
                        },
                        "target_not_protein_count": 0,
                        "resolved_sequence_count": len(protocol.entity_ids[entity]),
                        "overall_sequence_resolution_rate": 1.0,
                        "protein_target_resolution_rate": 1.0,
                        "rows": [
                            (
                                {
                                    "row_id": row_id,
                                    "status": "resolved",
                                    "value": f"C{row_id}",
                                }
                                if entity == "compound"
                                else {
                                    "row_id": row_id,
                                    "status": "resolved",
                                    "value": "MSEQ",
                                    "accession": f"P-{row_id}",
                                    "sequence_sha256": hashlib.sha256(
                                        b"MSEQ"
                                    ).hexdigest(),
                                }
                            )
                            for row_id in protocol.entity_ids[entity]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            count = len(protocol.entity_ids[entity])
            records[entity] = {
                "matrix": cls._artifact(embedding_dir, matrix),
                "row_ids_sidecar": cls._artifact(embedding_dir, sidecar),
                "mapping": cls._artifact(embedding_dir, mapping),
                "missing_mask": cls._artifact(embedding_dir, mask),
                "retrieval_cache": cls._artifact(embedding_dir, cache),
                "mapping_sha256": protocol.source_files[f"mapping/{entity}"][
                    "sha256"
                ],
                "missing_sequence_policy": "strict",
                "declared_zero_unk_count": 0,
                "declared_zero_unk_row_ids": [],
                "sequence_resolution": (
                    {
                        "target_not_protein_count": 0,
                        "resolved_sequence_count": count,
                        "protein_target_resolution_rate": 1.0,
                    }
                    if entity == "protein"
                    else None
                ),
            }
        (embedding_dir / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "astra-ethnobotany-feature-bundle/v1",
                    "dataset": "ethnobotany",
                    "row_id_space": "protocol_raw_id",
                    "entities": records,
                    "formal_eligible": True,
                    "formal_ineligibility_reasons": [],
                    "random_imputation": False,
                }
            ),
            encoding="utf-8",
        )

    def test_hghda_is_wired_into_launcher_and_formal_preflight(self) -> None:
        self.assertEqual(
            RUNNERS["hghda"],
            ("astra.official_baselines.hghda_runner", "HGHDA_official"),
        )
        self.assertEqual(
            official_baseline_layout()[1]["HGHDA"], "HGHDA_official"
        )
        self.assertIn("hghda", FAIR_METHODS)
        self.assertNotIn("hghda", FAIR_DEFAULT_METHODS)

    def test_fair_launcher_forwards_predeclared_hghda_budget(self) -> None:
        args = build_fair_parser().parse_args(
            [
                "--protocol-manifest", "manifest.json",
                "--dataset", "ethnobotany",
                "--data-dir", "data",
                "--embedding-dir", "embeddings",
                "--output-root", "output",
                "--methods", "hghda",
                "--hghda-epochs", "16",
            ]
        )
        self.assertEqual(args.methods, ["hghda"])
        self.assertEqual(args.hghda_epochs, 16)

    def test_fair_launcher_requires_and_forwards_independent_hghda_features(self) -> None:
        protocol = SimpleNamespace(dataset="tcm-suite", seed=42)
        with tempfile.TemporaryDirectory(prefix="astra_hghda_fair_launcher_") as temporary:
            output = Path(temporary) / "output"
            commands: list[list[str]] = []

            def capture(command, check):
                self.assertTrue(check)
                commands.append(command)
                return SimpleNamespace(returncode=0)

            common = [
                "--protocol-manifest", "manifest.json",
                "--dataset", "tcm-suite",
                "--data-dir", "data",
                "--embedding-dir", "herbhg-features",
                "--output-root", str(output),
                "--methods", "hghda",
                "--folds", "1",
            ]
            with patch.object(
                fair_comparison, "load_protocol_manifest", return_value=protocol
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "--hghda-embedding-dir is required"
                ):
                    fair_comparison.main(common)

            with patch.object(
                fair_comparison, "load_protocol_manifest", return_value=protocol
            ), patch.object(
                fair_comparison.subprocess, "run", side_effect=capture
            ):
                status = fair_comparison.main(
                    common
                    + [
                        "--hghda-embedding-dir", "author-hghda-features",
                        "--tcm-scalable-adaptations",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(commands), 1)
            command = commands[0]
            hghda_dir = command[command.index("--hghda-embedding-dir") + 1]
            self.assertEqual(hghda_dir, str(Path("author-hghda-features").resolve()))
            self.assertNotEqual(hghda_dir, str(Path("herbhg-features").resolve()))
            self.assertIn("--tcm-scalable-adaptations", command)

    def test_hghda_can_consume_biological_subset_of_four_entity_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astra_hghda_four_bundle_") as temporary:
            embedding_dir = Path(temporary)
            (embedding_dir / "bundle_manifest.json").write_text(
                json.dumps({"profile": "herbhg-four-entity"}), encoding="utf-8"
            )
            canonical_paths = {
                "herb": embedding_dir / "herb_semantic_embeddings_64d.npy",
                "compound": embedding_dir / "compound_embedding_kpgt.txt",
                "protein": embedding_dir / "protein_embedding_esm2.txt",
                "disease": embedding_dir / "disease_semantic_embeddings_64d.npy",
            }
            masks = {}
            entities = {}
            for entity in canonical_paths:
                canonical_paths[entity].write_bytes(b"fixture")
                masks[entity] = embedding_dir / f"{entity}.mask.npy"
                np.save(masks[entity], np.zeros(2, dtype=np.bool_))
                entities[entity] = {
                    "mask_path": masks[entity],
                    "mask": np.zeros(2, dtype=np.bool_),
                    "component": {
                        "manifest": {
                            "declared_zero_unk_count": 0,
                            "declared_audited_missing_count": 0,
                        }
                    },
                }
            protocol = SimpleNamespace(
                entity_ids={entity: ("1", "2") for entity in canonical_paths},
                source_files={
                    f"mapping/{entity}": {"sha256": entity * 8}
                    for entity in canonical_paths
                },
            )
            validated = {
                "manifest": {"schema": "astra-ethnobotany-feature-bundle/v1"},
                "entities": entities,
            }
            with patch(
                "astra.benchmark.astra_runner.validate_feature_bundle",
                return_value=validated,
            ) as recursive_validator:
                selected_masks, diagnostics = _validate_ethnobotany_feature_bundle(
                    protocol,
                    embedding_dir,
                    {
                        "compound": canonical_paths["compound"],
                        "protein": canonical_paths["protein"],
                    },
                    smoke=False,
                )
            expected_paths = recursive_validator.call_args.kwargs[
                "expected_feature_paths"
            ]
            self.assertEqual(set(expected_paths), set(canonical_paths))
            self.assertEqual(set(selected_masks), {"compound", "protein"})
            self.assertTrue(diagnostics["recursive_validation"])

    def test_training_rows_use_only_supplied_inner_train_rows(self) -> None:
        protocol = _Protocol()
        rows = training_rows(
            protocol,
            np.asarray([[1, 0], [0, 1]], dtype=np.int64),
            np.asarray([1, 0], dtype=np.uint8),
        )
        self.assertEqual(rows, [["h20", "d10", 1.0], ["h10", "d20", 0.0]])
        assert_author_training_input(rows, rows)
        with self.assertRaisesRegex(HGHDAFairRunnerError, "repartitioned"):
            assert_author_training_input(rows[::-1], rows)

    def test_split_isolation_rejects_test_pair_leakage(self) -> None:
        clean = SimpleNamespace(
            train_pairs=np.asarray([[0, 0], [1, 1]], dtype=np.int64),
            train_labels=np.asarray([1, 0], dtype=np.uint8),
            train_row_indices=np.asarray([10, 11], dtype=np.int64),
            validation_pairs=np.asarray([[0, 1]], dtype=np.int64),
            validation_labels=np.asarray([1], dtype=np.uint8),
            validation_row_indices=np.asarray([12], dtype=np.int64),
            test_pairs=np.asarray([[1, 0]], dtype=np.int64),
            test_labels=np.asarray([0], dtype=np.uint8),
            test_row_indices=np.asarray([13], dtype=np.int64),
        )
        assert_split_isolation(clean)
        leaked = SimpleNamespace(**vars(clean))
        leaked.test_pairs = np.asarray([[0, 0]], dtype=np.int64)
        with self.assertRaisesRegex(HGHDAFairRunnerError, "train/test H-D pairs"):
            assert_split_isolation(leaked)

    def test_feature_transform_is_fixed_and_rejects_degenerate_input(self) -> None:
        values = np.vstack(
            (
                np.zeros((1, 64), dtype=np.float64),
                np.ones((1, 64), dtype=np.float64),
            )
        )
        transformed = _standardise_and_clip(values)
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_allclose(transformed[0], -1.0, atol=1e-6)
        np.testing.assert_allclose(transformed[1], 1.0, atol=1e-6)
        with self.assertRaisesRegex(HGHDAFairRunnerError, "non-degenerate"):
            _standardise_and_clip(np.zeros((2, 64), dtype=np.float64))

        zero_unk_values = np.vstack(
            (
                np.zeros((1, 64), dtype=np.float64),
                np.ones((1, 64), dtype=np.float64),
                np.full((1, 64), 2.0, dtype=np.float64),
            )
        )
        zero_unk = _standardise_and_clip(
            zero_unk_values,
            missing_mask=np.asarray([True, False, False]),
        )
        np.testing.assert_array_equal(zero_unk[0], np.zeros(64))

    def test_tcm_feature_contract_accepts_only_locked_author_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astra_hghda_tcm_features_") as temporary:
            root = Path(temporary)
            embeddings = root / "hghda_features"
            materialised = root / "materialised"
            embeddings.mkdir()
            compound = np.arange(2 * 64, dtype=np.float64).reshape(2, 64)
            protein = np.arange(3 * 64, dtype=np.float64).reshape(3, 64) + 0.25
            feature_paths = {
                "compound": embeddings / "compound_embedding-64.txt",
                "protein": embeddings / "protein_low.txt",
            }
            np.savetxt(feature_paths["compound"], compound)
            np.savetxt(feature_paths["protein"], protein)
            feature_hashes = {
                entity: hashlib.sha256(path.read_bytes()).hexdigest()
                for entity, path in feature_paths.items()
            }
            relation_hashes = {
                "H_C": "1" * 64,
                "C_P": "2" * 64,
                "P_D": "3" * 64,
            }
            protocol = SimpleNamespace(
                dataset="tcm-suite",
                entity_ids={
                    "compound": ("c1", "c2"),
                    "protein": ("p1", "p2", "p3"),
                },
                source_files={
                    relation: {"sha256": digest, "path": f"/{relation}.txt"}
                    for relation, digest in relation_hashes.items()
                },
            )
            with patch.dict(
                hghda_runner.TCM_AUTHOR_FEATURE_SHA256,
                feature_hashes,
                clear=True,
            ), patch.dict(
                hghda_runner.TCM_AUTHOR_RELATION_SHA256,
                relation_hashes,
                clear=True,
            ):
                bundle = _validate_tcm_author_feature_bundle(protocol, embeddings)
                self.assertEqual(
                    bundle["schema"],
                    "astra-tcm-hghda-author-feature-bundle/v1",
                )
                diagnostics = prepare_features(
                    protocol, embeddings, materialised, {}
                )
                np.testing.assert_array_equal(
                    np.loadtxt(materialised / "compound_embedding-64.txt"),
                    compound.astype(np.float32),
                )
                self.assertIn(
                    "no value transform", diagnostics["compound"]["preprocessing"]
                )

                feature_paths["compound"].write_text(
                    feature_paths["compound"].read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    HGHDAFairRunnerError, "hash differs"
                ):
                    _validate_tcm_author_feature_bundle(protocol, embeddings)

    def test_rebind_uses_frozen_mapping_order(self) -> None:
        data = SimpleNamespace(
            hcassociation=[["h10", "c20"]],
            cpassociation=[["c10", "p20"]],
            pdassociation=[["p10", "d20"]],
        )
        model = SimpleNamespace(data=data)
        _rebind_model_mappings(model, _Protocol())
        self.assertEqual(model.data.herb, {"h10": 0, "h20": 1})
        self.assertEqual(model.data.protein, {"p10": 0, "p20": 1})
        self.assertEqual(model.num_compounds, 2)
        self.assertEqual(model.num_diseases, 2)

    def test_author_loaded_matrix_preserves_zero_unk_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astra_hghda_zero_unk_") as temporary:
            dataset_dir = Path(temporary)
            expected = np.vstack(
                (
                    np.zeros((1, 64), dtype=np.float32),
                    np.ones((1, 64), dtype=np.float32),
                    np.full((1, 64), 2.0, dtype=np.float32),
                )
            )
            compound = np.vstack(
                (
                    np.ones((1, 64), dtype=np.float32),
                    np.full((1, 64), 2.0, dtype=np.float32),
                    np.full((1, 64), 3.0, dtype=np.float32),
                )
            )
            np.savetxt(dataset_dir / "compound_embedding-64.txt", compound)
            np.savetxt(dataset_dir / "protein_low.txt", expected)
            mask_path = dataset_dir / "protein_mask.npy"
            np.save(mask_path, np.asarray([True, False, False], dtype=bool))

            compound_token = object()
            protein_token = object()

            class _Session:
                def __init__(self, values: dict[object, np.ndarray]):
                    self.values = values

                def run(self, tensor: object) -> np.ndarray:
                    return self.values[tensor]

            model = SimpleNamespace(
                compound_embeddings=SimpleNamespace(initial_value=compound_token),
                protein_embeddings=SimpleNamespace(initial_value=protein_token),
                sess=_Session(
                    {compound_token: compound.copy(), protein_token: expected.copy()}
                ),
            )
            with patch(
                "astra.official_baselines.hghda_runner._probe_hghda_feature_dependencies",
                return_value={
                    entity: {"connected": True, "gradient_l2_norm": 1.0}
                    for entity in ("compound", "protein")
                },
            ):
                diagnostics = assert_author_feature_binding(
                    model, dataset_dir, mask_path
                )
            self.assertTrue(diagnostics["compound"]["connected"])
            changed = expected.copy()
            changed[0] = 0.5
            model.sess = _Session(
                {compound_token: compound.copy(), protein_token: changed}
            )
            with self.assertRaisesRegex(
                HGHDAFairRunnerError, "differs from the audited matrix"
            ):
                with patch(
                    "astra.official_baselines.hghda_runner._probe_hghda_feature_dependencies",
                    return_value={
                        entity: {"connected": True, "gradient_l2_norm": 1.0}
                        for entity in ("compound", "protein")
                    },
                ):
                    assert_author_feature_binding(model, dataset_dir, mask_path)

    def test_scores_are_continuous_probabilities(self) -> None:
        model = SimpleNamespace(
            u=np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
            i=np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        )
        scores = score_pairs(
            model, np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
        )
        self.assertTrue(np.all((scores > 0.0) & (scores < 1.0)))
        self.assertGreater(np.unique(scores).size, 1)

    def test_epoch_budget_is_limited_to_predeclared_four_or_sixteen(self) -> None:
        parser = build_parser()
        common = [
            "--fold",
            "1",
            "--protocol-manifest",
            "manifest.json",
            "--embedding-dir",
            "embeddings",
            "--output-root",
            "output",
        ]
        self.assertEqual(parser.parse_args(common + ["--epochs", "4"]).epochs, 4)
        self.assertEqual(parser.parse_args(common + ["--epochs", "16"]).epochs, 16)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(common + ["--epochs", "8"])

    def test_dry_run_consumes_manifest_splits_and_identity_sidecars(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astra_hghda_fair_") as temporary:
            root = Path(temporary)
            data = root / "data"
            embeddings = root / "embeddings"
            source = root / "official"
            data.mkdir()
            embeddings.mkdir()
            (source / "src" / "base").mkdir(parents=True)

            herbs = list(range(1, 11))
            diseases = list(range(1, 11))
            self._write_pairs(
                data / "H_D.txt", list(zip(herbs, diseases))
            )
            self._write_pairs(
                data / "negativehdlist.txt",
                [(herb, (herb % 10) + 1) for herb in herbs],
            )
            self._write_pairs(
                data / "H_C.txt",
                [(herb, ((herb - 1) % 6) + 1) for herb in herbs],
            )
            self._write_pairs(
                data / "C_P.txt",
                [
                    (f"COMPOUND-{compound}", ((compound - 1) % 5) + 1)
                    for compound in range(1, 7)
                ],
            )
            self._write_pairs(
                data / "P_D.txt",
                [(index % 5 + 1, disease) for index, disease in enumerate(diseases)],
            )
            for entity, count in (
                ("herb", 10),
                ("compound", 6),
                ("protein", 5),
                ("disease", 10),
            ):
                self._write_mapping(
                    data / f"{entity}_id_mapping.csv", entity, count
                )

            protocol = build_protocol("ethnobotany", data, seed=42)
            manifest = protocol.save(root / "protocol")
            for offset, entity in enumerate(("compound", "protein"), start=1):
                rows = len(protocol.entity_ids[entity])
                values = (
                    np.arange(rows * 64, dtype=np.float64).reshape(rows, 64)
                    + offset
                )
                matrix = embeddings / (
                    "compound_embedding_kpgt.txt"
                    if entity == "compound"
                    else "protein_embedding_esm2.txt"
                )
                np.savetxt(matrix, values)
                sidecar = {
                    "schema": FEATURE_ROW_IDS_SCHEMA,
                    "entity": entity,
                    "id_space": "protocol_raw_id",
                    "mapping_sha256": protocol.source_files[f"mapping/{entity}"]["sha256"],
                    "row_ids": list(protocol.entity_ids[entity]),
                }
                feature_row_ids_path(matrix).write_text(
                    json.dumps(sidecar), encoding="utf-8"
                )
            self._write_bundle_manifest(embeddings, data, protocol)

            for relative in (
                "HGHDA.py",
                "rating.py",
                "base/recommender.py",
                "base/diseaseRecommender.py",
                "base/herbRecommender.py",
                "util/config.py",
                "util/io.py",
                "util/log.py",
                "HGHDA.conf",
            ):
                target = source / "src" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# dry-run fixture\n", encoding="utf-8")

            output = root / "output"
            args = SimpleNamespace(
                fold=1,
                protocol_manifest=manifest,
                embedding_dir=embeddings,
                output_root=output,
                source_dir=source,
                device="cpu",
                seed=42,
                epochs=16,
                smoke=False,
                dry_run=True,
                run_generation_id="unit-test",
            )
            validated_bundle = {
                "manifest": {"schema": "astra-ethnobotany-feature-bundle/v1"},
                "entities": {
                    entity: {
                        "mask_path": embeddings
                        / f"{FEATURE_FILENAMES_FOR_TEST[entity]}.missing_mask.npy",
                        "mask": np.zeros(
                            len(protocol.entity_ids[entity]), dtype=np.bool_
                        ),
                        "component": {
                            "manifest": {
                                "declared_zero_unk_count": 0,
                                "declared_audited_missing_count": 0,
                            }
                        },
                    }
                    for entity in ("compound", "protein")
                },
            }
            with patch(
                "astra.official_baselines.hghda_runner.verify_checkout_commit",
                return_value="c176902f4206f45c2d23bc1c684eeb0c229984cb",
            ), patch(
                "astra.benchmark.astra_runner.validate_feature_bundle",
                return_value=validated_bundle,
            ):
                payload = run(args)

            fold = protocol.fold_rows(1)
            self.assertEqual(payload["status"], "validated-dry-run")
            self.assertEqual(payload["hyperparameters"]["epochs"], 16)
            self.assertEqual(
                payload["split_rows"]["inner_train_used"], len(fold.train_pairs)
            )
            self.assertFalse(payload["data_use_policy"]["validation_used_for_training"])
            self.assertFalse(payload["data_use_policy"]["outer_test_used_for_training"])
            self.assertTrue((output / "protocol.json").is_file())
            self.assertFalse((output / "results" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
