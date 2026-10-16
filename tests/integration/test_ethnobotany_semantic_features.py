from __future__ import annotations

from contextlib import redirect_stderr
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from astra.benchmark import ethnobotany_semantic_features as semantic
from astra.benchmark import ethnobotany_semantic_controls as controls
from astra.benchmark import ethnobotany_feature_bundle as bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _mapping(path: Path, entity: str, source_ids: list[str]) -> None:
    source_column = "herb_curie" if entity == "herb" else "disease_mondo"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{entity}_id", source_column])
        writer.writerows((index, value) for index, value in enumerate(source_ids, 1))


class EthnobotanySemanticFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # A centered 128-D reducer needs at least 129 joint rows.  Keeping that
        # rank available in the shared fixture lets both formal dimensions run
        # through the real SVD and recursive validator.
        self.herb_ids = [f"ncbitaxon:{1000 + index}" for index in range(65)]
        self.disease_ids = [f"MONDO:{index:07d}" for index in range(1, 65)]
        self.herb_mapping = self.root / "herb_id_mapping.csv"
        self.disease_mapping = self.root / "disease_id_mapping.csv"
        _mapping(self.herb_mapping, "herb", self.herb_ids)
        _mapping(self.disease_mapping, "disease", self.disease_ids)

        self.herb_component = self.root / "herb-component.csv"
        with self.herb_component.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["plant_curie", "plant name", "chemical_curie", "chemical name", "database", "evidence"]
            )
            for index, source_id in enumerate(self.herb_ids):
                writer.writerow([source_id, f"Plant {index}", "pubchem:1", "x", "test", "test"])
            # Cross-table agreement and total count make this candidate win.
            writer.writerow([self.herb_ids[0], "Preferred plant", "pubchem:2", "x", "test", "test"])
            writer.writerow([self.herb_ids[0], "Preferred plant", "pubchem:3", "x", "test", "test"])

        nodes = []
        for index, source_id in enumerate(self.disease_ids):
            numeric = source_id.split(":", 1)[1]
            meta: dict[str, object] = {
                "synonyms": [{"val": f"Disease synonym {index}", "pred": "hasExactSynonym"}]
            }
            if index != 1:
                meta["definition"] = {"val": f"Definition {index}"}
            nodes.append(
                {
                    "id": f"http://purl.obolibrary.org/obo/MONDO_{numeric}",
                    "lbl": f"Disease {index}",
                    "meta": meta,
                }
            )
        self.mondo = self.root / "mondo-base.json"
        self.mondo.write_text(json.dumps({"graphs": [{"nodes": nodes}]}), encoding="utf-8")
        self.patches = mock.patch.multiple(
            semantic,
            HERB_COMPONENT_SHA256=_sha(self.herb_component),
            MONDO_BASE_JSON_SHA256=_sha(self.mondo),
        )
        self.patches.start()
        self.environment = mock.patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.patches.stop()
        self.temporary.cleanup()

    def _build_text(self) -> Path:
        return semantic.build_semantic_text_manifest(
            herb_mapping_path=self.herb_mapping,
            disease_mapping_path=self.disease_mapping,
            herb_component_csv=self.herb_component,
            mondo_json=self.mondo,
            output_path=self.root / "semantic_text_manifest.json",
            expected_herb_component_sha256=_sha(self.herb_component),
            expected_mondo_sha256=_sha(self.mondo),
        )

    def _fake_embedding_manifest(self, text_manifest: Path) -> Path:
        text = json.loads(text_manifest.read_text(encoding="utf-8"))
        model_root = self.root / semantic.SCIBERT_REVISION
        model_root.mkdir()
        model_hashes: dict[str, str] = {}
        model_files: dict[str, object] = {}
        for name in semantic.SCIBERT_FILE_SHA256:
            path = model_root / name
            path.write_bytes(f"fixture:{name}".encode())
            model_hashes[name] = _sha(path)
            model_files[name] = _artifact(path)
        self.model_patch = mock.patch.object(semantic, "SCIBERT_FILE_SHA256", model_hashes)
        self.model_patch.start()
        entities: dict[str, object] = {}
        rng = np.random.default_rng(42)
        for entity in ("herb", "disease"):
            rows = text["entities"][entity]["rows"]
            matrix = rng.normal(size=(len(rows), semantic.SCIBERT_DIMENSION)).astype(np.float32)
            path = self.root / f"{entity}_semantic_embeddings_768d.npz"
            np.savez_compressed(
                path,
                row_ids=np.asarray([row["row_id"] for row in rows], dtype=str),
                source_ids=np.asarray([row["source_id"] for row in rows], dtype=str),
                text_sha256=np.asarray([row["text_sha256"] for row in rows], dtype=str),
                embeddings=matrix,
            )
            entities[entity] = {
                "artifact": _artifact(path),
                "row_count": len(rows),
                "dimension": semantic.SCIBERT_DIMENSION,
                "mapping_sha256": text["entities"][entity]["mapping_sha256"],
            }
        payload = {
            "schema": semantic.EMBEDDING_MANIFEST_SCHEMA,
            "semantic_condition": semantic.SEMANTIC_CONDITION,
            "semantic_condition_status": "new_regenerated_condition_not_legacy_reproduction",
            "created_at": "fixture",
            "formal_eligible": True,
            "random_imputation": False,
            "text_manifest": _artifact(text_manifest),
            "model": {
                "model_id": semantic.SCIBERT_MODEL_ID,
                "revision": semantic.SCIBERT_REVISION,
                "files": model_files,
                "pooling": "first_special_classification_token_last_hidden_state",
                "tokenization": {"truncation": True, "max_length": 512, "special_tokens": True},
                "raw_dimension": semantic.SCIBERT_DIMENSION,
            },
            "runtime": {
                "device": "cpu",
                "seed": 42,
                "model_training_flag": False,
                "deterministic_algorithms": True,
                "torch_num_threads": 4,
                "torch_num_interop_threads": 1,
                "thread_environment": {
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                },
            },
            "entities": entities,
        }
        path = self.root / "semantic_embedding_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_text_manifest_rederives_names_and_mondo_fallbacks(self) -> None:
        manifest = self._build_text()
        payload = semantic.validate_semantic_text_manifest(manifest)["manifest"]
        first = payload["entities"]["herb"]["rows"][0]
        self.assertIs(payload["association_label_free"], True)
        self.assertEqual(first["canonical_name"], "Preferred plant")
        self.assertEqual(len(first["name_candidates"]), 2)
        self.assertNotIn("MONDO:", first["text"])
        self.assertNotIn("herb_disease", payload["sources"])
        self.assertEqual(
            payload["sources"]["herb_component"]["columns_read"],
            ["plant_curie", "plant name"],
        )
        self.assertEqual(
            payload["sources"]["herb_component"]["association_columns_read"], []
        )
        self.assertEqual(
            payload["entities"]["disease"]["rows"][1]["text_mode"],
            "label_plus_synonym_fallback",
        )
        payload["entities"]["herb"]["rows"] = list(
            reversed(payload["entities"]["herb"]["rows"])
        )
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(semantic.SemanticFeatureError, "re-derived"):
            semantic.validate_semantic_text_manifest(manifest)

    def test_embedding_rows_and_shared_reducer_are_recursively_bound(self) -> None:
        text_manifest = self._build_text()
        embedding_manifest = self._fake_embedding_manifest(text_manifest)
        try:
            validated = semantic.validate_scibert_embedding_manifest(embedding_manifest)
            disease_path = validated["outputs"]["disease"]["path"]
            with np.load(disease_path, allow_pickle=False) as arrays:
                values = {name: arrays[name] for name in arrays.files}
            values["row_ids"] = values["row_ids"][::-1]
            np.savez_compressed(disease_path, **values)
            payload = json.loads(embedding_manifest.read_text(encoding="utf-8"))
            payload["entities"]["disease"]["artifact"] = _artifact(disease_path)
            embedding_manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(semantic.SemanticFeatureError, "row IDs"):
                semantic.validate_scibert_embedding_manifest(embedding_manifest)

            values["row_ids"] = values["row_ids"][::-1]
            np.savez_compressed(disease_path, **values)
            payload["entities"]["disease"]["artifact"] = _artifact(disease_path)
            embedding_manifest.write_text(json.dumps(payload), encoding="utf-8")
            reduction = semantic.reduce_shared_semantic_embeddings(
                embedding_manifest=embedding_manifest,
                output_dir=self.root / "reduced",
            )
            checked = semantic.validate_shared_semantic_reduction(reduction)
            self.assertEqual(
                checked["checkpoint_path"].name,
                "shared_semantic_centered_svd_pca64.npz",
            )
            self.assertEqual(
                checked["outputs"]["herb"]["matrix"].shape,
                (len(self.herb_ids), 64),
            )
            self.assertEqual(
                checked["outputs"]["disease"]["matrix"].shape,
                (len(self.disease_ids), 64),
            )
            self.assertEqual(
                checked["outputs"]["herb"]["matrix_path"].name,
                "herb_semantic_embeddings_64d.npy",
            )
            self.assertEqual(
                checked["outputs"]["disease"]["matrix_path"].name,
                "disease_semantic_embeddings_64d.npy",
            )
            self.assertEqual(len(checked["dependencies"]), 17)
            self.assertIn(self.herb_mapping.resolve(), checked["dependencies"])
            self.assertIn(self.disease_mapping.resolve(), checked["dependencies"])
            self.assertTrue(all(path.is_file() for path in checked["dependencies"]))
            portable = self.root / "portable_objects"
            portable.mkdir()
            resolver: dict[str, Path] = {}
            for index, dependency in enumerate(checked["dependencies"]):
                destination = portable / f"{index:02d}{dependency.suffix}"
                shutil.copy2(dependency, destination)
                resolver[_sha(dependency)] = destination
            portable_reduction = resolver[_sha(reduction)]
            portable_checked = semantic.validate_shared_semantic_reduction(
                portable_reduction, resolver=resolver
            )
            self.assertEqual(
                portable_checked["outputs"]["disease"]["matrix"].shape,
                (len(self.disease_ids), 64),
            )
            components = bundle.build_verified_semantic_components(
                semantic_reduction_manifest=reduction,
                herb_mapping_path=self.herb_mapping,
                disease_mapping_path=self.disease_mapping,
                output_dir=self.root / "bundle_semantics",
            )
            self.assertEqual(set(components), {"herb", "disease"})
            for entity, component_path in components.items():
                component = bundle.validate_feature_component(component_path)
                self.assertEqual(
                    component["manifest"]["semantic_identity_status"],
                    "independently_id_keyed_verified",
                )
                self.assertEqual(
                    component["manifest"]["semantic_condition"],
                    semantic.SEMANTIC_CONDITION,
                )
                np.testing.assert_array_equal(
                    component["matrix"], checked["outputs"][entity]["matrix"]
                )
                portable_component_root = self.root / f"portable_{entity}_component"
                portable_component_root.mkdir()
                component_resolver: dict[str, Path] = {}
                for index, dependency in enumerate(component["dependencies"]):
                    destination = (
                        portable_component_root
                        / f"{index:02d}_{dependency.name}"
                    )
                    shutil.copy2(dependency, destination)
                    component_resolver[_sha(dependency)] = destination
                portable_component = component_resolver[_sha(component_path)]
                portable_validated = bundle.validate_feature_component(
                    portable_component, resolver=component_resolver
                )
                np.testing.assert_array_equal(
                    portable_validated["matrix"], component["matrix"]
                )
            checkpoint = checked["checkpoint_path"]
            with np.load(checkpoint, allow_pickle=False) as arrays:
                self.assertEqual(arrays["components"].shape, (64, 768))
            # A matrix swap cannot be hidden by editing only the top-level hash;
            # the checkpoint reproduction catches it.
            herb_path = checked["outputs"]["herb"]["matrix_path"]
            matrix = np.load(herb_path, allow_pickle=False)
            np.save(herb_path, matrix[::-1])
            reduction_payload = json.loads(reduction.read_text(encoding="utf-8"))
            reduction_payload["entities"]["herb"]["matrix"] = _artifact(herb_path)
            reduction.write_text(json.dumps(reduction_payload), encoding="utf-8")
            with self.assertRaisesRegex(semantic.SemanticFeatureError, "not reproduced"):
                semantic.validate_shared_semantic_reduction(reduction)
        finally:
            self.model_patch.stop()

    def test_shared_reducer_supports_audited_pca128_and_dynamic_names(self) -> None:
        text_manifest = self._build_text()
        embedding_manifest = self._fake_embedding_manifest(text_manifest)
        try:
            reduction = semantic.reduce_shared_semantic_embeddings(
                embedding_manifest=embedding_manifest,
                output_dir=self.root / "reduced_128d",
                output_dimension=128,
            )
            checked = semantic.validate_shared_semantic_reduction(reduction)
            self.assertEqual(checked["manifest"]["output_dimension"], 128)
            self.assertEqual(
                checked["checkpoint_path"].name,
                "shared_semantic_centered_svd_pca128.npz",
            )
            for entity, expected_rows in (
                ("herb", len(self.herb_ids)),
                ("disease", len(self.disease_ids)),
            ):
                output = checked["outputs"][entity]
                self.assertEqual(output["matrix"].shape, (expected_rows, 128))
                self.assertEqual(
                    output["matrix_path"].name,
                    f"{entity}_semantic_embeddings_128d.npy",
                )
            with np.load(checked["checkpoint_path"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["components"].shape, (128, 768))

            payload = json.loads(reduction.read_text(encoding="utf-8"))
            payload["output_dimension"] = 96
            reduction.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                semantic.SemanticFeatureError,
                "must be one of: 64, 128",
            ):
                semantic.validate_shared_semantic_reduction(reduction)
        finally:
            self.model_patch.stop()

    def test_reduce_cli_forwards_dimension_and_defaults_to_64(self) -> None:
        parser = semantic._parser()
        default_args = parser.parse_args(
            [
                "reduce",
                "--embedding-manifest",
                "embedding.json",
                "--output-dir",
                "reduced",
            ]
        )
        self.assertEqual(default_args.output_dimension, 64)

        expected = self.root / "semantic_reduction_manifest.json"
        with mock.patch.object(
            semantic,
            "reduce_shared_semantic_embeddings",
            return_value=expected,
        ) as reduce_mock, mock.patch("builtins.print"):
            result = semantic.main(
                [
                    "reduce",
                    "--embedding-manifest",
                    "embedding.json",
                    "--output-dir",
                    "reduced",
                    "--output-dimension",
                    "128",
                ]
            )
        self.assertEqual(result, 0)
        reduce_mock.assert_called_once_with(
            embedding_manifest="embedding.json",
            output_dir="reduced",
            output_dimension=128,
        )

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "reduce",
                    "--embedding-manifest",
                    "embedding.json",
                    "--output-dir",
                    "reduced",
                    "--output-dimension",
                    "96",
                ]
            )

    def test_pca64_zero_pad128_control_is_exact_and_label_free(self) -> None:
        text_manifest = self._build_text()
        embedding_manifest = self._fake_embedding_manifest(text_manifest)
        try:
            source_manifest = semantic.reduce_shared_semantic_embeddings(
                embedding_manifest=embedding_manifest,
                output_dir=self.root / "source_pca64",
                output_dimension=64,
            )
            source_dependencies = semantic.validate_shared_semantic_reduction(
                source_manifest
            )["dependencies"]
            source_hashes = {path: _sha(path) for path in source_dependencies}

            control_manifest = controls.build_pca64_zero_pad128_control(
                source_reduction_manifest=source_manifest,
                output_dir=self.root / "pca64_zero_pad128_control",
            )
            checked = controls.validate_pca64_zero_pad128_control(
                control_manifest
            )
            dispatched = semantic.validate_shared_semantic_reduction(
                control_manifest
            )
            self.assertEqual(
                checked["manifest"]["schema"],
                controls.ZERO_PAD_CONTROL_MANIFEST_SCHEMA,
            )
            self.assertEqual(
                checked["manifest"]["consumer_contract"],
                {
                    "semantic_input_dimension": 128,
                    "semantic_adapter": "trainable_linear_128_to_64",
                    "semantic_adapter_output_dimension": 64,
                    "semantic_input_trainability": "frozen",
                    "same_adapter_parameterization_as_genuine_pca128_required": True,
                },
            )
            self.assertEqual(checked["manifest"]["label_inputs"], [])
            self.assertIs(
                checked["manifest"]["uses_association_labels"], False
            )
            self.assertEqual(
                dispatched["manifest_path"], checked["manifest_path"]
            )

            source = semantic.validate_shared_semantic_reduction(source_manifest)
            for entity in ("herb", "disease"):
                observed = checked["outputs"][entity]["matrix"]
                np.testing.assert_array_equal(
                    observed[:, :64], source["outputs"][entity]["matrix"]
                )
                np.testing.assert_array_equal(
                    observed[:, 64:],
                    np.zeros((observed.shape[0], 64), dtype=np.float32),
                )
                self.assertEqual(
                    checked["outputs"][entity]["matrix_path"].name,
                    f"{entity}_semantic_embeddings_pca64_zero_pad_128d.npy",
                )
            with np.load(checked["checkpoint_path"], allow_pickle=False) as arrays:
                components = arrays["components"]
            with np.load(source["checkpoint_path"], allow_pickle=False) as arrays:
                source_components = arrays["components"]
            np.testing.assert_array_equal(components[:64], source_components)
            np.testing.assert_array_equal(
                components[64:], np.zeros_like(components[64:])
            )
            self.assertEqual(
                source_hashes,
                {path: _sha(path) for path in source_dependencies},
            )

            herb_path = checked["outputs"]["herb"]["matrix_path"]
            corrupted = np.load(herb_path, allow_pickle=False)
            corrupted[0, 64] = 1.0
            np.save(herb_path, corrupted)
            payload = json.loads(control_manifest.read_text(encoding="utf-8"))
            payload["entities"]["herb"]["matrix"] = _artifact(herb_path)
            control_manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                semantic.SemanticFeatureError,
                "not exact PCA64 plus zeros",
            ):
                controls.validate_pca64_zero_pad128_control(control_manifest)
        finally:
            self.model_patch.stop()


if __name__ == "__main__":
    unittest.main()
