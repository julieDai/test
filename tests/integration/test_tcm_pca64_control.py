from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from astra.benchmark.features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path
from astra.benchmark.tcm_astra_pca64_control_runner import (
    TCMPCA64ControlRunnerError,
    reject_outer_artifacts,
)
from astra.benchmark.tcm_astra_pca64_control_plan import (
    TCMPCA64ControlPlanError,
    _source_freeze,
    _winner_reference,
)
from astra.benchmark.tcm_astra_pca64_control_selection import (
    TCMPCA64ControlSelectionError,
    select_feature_basis,
)
from astra.benchmark.tcm_feature_bundle import (
    FEATURE_FILENAMES,
    TCM_FEATURE_BUNDLE_PROFILE,
    TCM_FEATURE_BUNDLE_SCHEMA,
    TCM_SEMANTIC_CONDITION,
)
from astra.benchmark.tcm_pca64_feature_bundle import (
    PCA64_ALGORITHM,
    PCA64_BUNDLE_PROFILE,
    PCA64_BUNDLE_SCHEMA,
    PCA64_SEMANTIC_CONDITION,
    TCMPCA64BundleError,
    build_tcm_pca64_feature_bundle,
    fit_centered_shared_pca64,
    validate_tcm_pca64_feature_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _Protocol:
    dataset = "tcm-suite"
    entity_ids = {
        "herb": ("h2", "h1"),
        "compound": ("c2", "c1"),
        "protein": ("p2", "p1"),
        "disease": ("d2", "d1"),
    }

    def input_manifest(self):
        return {
            "dataset": self.dataset,
            "mappings": {
                entity: {
                    "count": len(ids),
                    "sha256": hashlib.sha256(
                        (json.dumps(list(ids), separators=(",", ":")) + "\n").encode()
                    ).hexdigest(),
                }
                for entity, ids in self.entity_ids.items()
            },
        }


class TCMPCA64AlgorithmTest(unittest.TestCase):
    def test_shared_pca64_is_centered_canonical_and_deterministic(self):
        rng = np.random.default_rng(17)
        herb = rng.normal(size=(35, 70))
        disease = rng.normal(loc=0.7, size=(35, 70))
        first = fit_centered_shared_pca64(herb, disease)
        second = fit_centered_shared_pca64(herb, disease)

        for key in (
            "mean",
            "components",
            "singular_values",
            "explained_variance_ratio",
            "herb_scores",
            "disease_scores",
        ):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(
            first["mean"], np.mean(np.concatenate((herb, disease)), axis=0)
        )
        joint_scores = np.concatenate(
            (first["herb_scores"], first["disease_scores"]), axis=0
        )
        np.testing.assert_allclose(np.mean(joint_scores, axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(
            first["components"] @ first["components"].T,
            np.eye(64),
            rtol=0.0,
            atol=1e-10,
        )
        for component in first["components"]:
            pivot = int(np.argmax(np.abs(component)))
            self.assertGreater(component[pivot], 0.0)
        variances = np.var(joint_scores, axis=0, ddof=1)
        self.assertFalse(np.allclose(variances, np.ones(64)))

    def test_margin_is_exactly_one_sided_and_finite(self):
        selected, gain = select_feature_basis(
            ae64_validation_auprc=0.8,
            pca64_validation_auprc=0.8001,
        )
        self.assertEqual(selected, "pca64_candidate")
        self.assertEqual(gain, 0.0001)
        selected, _ = select_feature_basis(
            ae64_validation_auprc=0.8,
            pca64_validation_auprc=0.800099999,
        )
        self.assertEqual(selected, "ae64_frozen_control")
        with self.assertRaises(TCMPCA64ControlSelectionError):
            select_feature_basis(
                ae64_validation_auprc=float("nan"),
                pca64_validation_auprc=0.9,
            )

    def test_outer_artifact_guard(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca64_outer_") as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "test_predictions.npz").write_bytes(b"forbidden")
            with self.assertRaisesRegex(
                TCMPCA64ControlRunnerError, "outer-test artifact"
            ):
                reject_outer_artifacts(root)

    def test_parent_winner_freezes_adapter_epoch_lr_and_validation_identity(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca64_parent_") as directory:
            root = Path(directory)
            artifact_fields = {}
            for key in (
                "validation_predictions",
                "herb_embedding",
                "disease_embedding",
                "metrics",
                "protocol",
                "adapter_audit",
            ):
                path = root / f"{key}.bin"
                path.write_bytes(key.encode("ascii"))
                artifact_fields[key] = str(path)
                artifact_fields[f"{key}_sha256"] = _sha256(path)
            winner = {
                "id": "residual_mlp_h32",
                "fold": 1,
                "semantic_adapter": "residual_mlp",
                "epochs": 32,
                "learning_rate": 0.003,
                "validation_auprc": 0.9,
                "validation_auroc": 0.91,
                "validation_threshold": 0.55,
                "validation_row_id_sha256": "a" * 64,
                "validation_pair_sha256": "b" * 64,
                "validation_label_sha256": "c" * 64,
                **artifact_fields,
            }
            fold_selection = {
                "selected_candidate": winner["id"],
                "selected_adapter": winner["semantic_adapter"],
                "winner": winner,
            }
            frozen = _winner_reference(fold=1, fold_selection=fold_selection)
            self.assertEqual(frozen["semantic_adapter"], "residual_mlp")
            self.assertEqual(frozen["epochs"], 32)
            self.assertEqual(frozen["learning_rate"], 0.003)
            fold_selection["selected_adapter"] = "identity"
            with self.assertRaisesRegex(
                TCMPCA64ControlPlanError, "selected adapter differs"
            ):
                _winner_reference(fold=1, fold_selection=fold_selection)

    def test_plan_source_freeze_binds_final_outer_evaluator(self):
        frozen = _source_freeze()
        for key in ("outer_evaluator", "outer_safety_primitives"):
            path = Path(frozen[key]["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(frozen[key]["sha256"], _sha256(path))


class TCMPCA64BundleValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_pca64_bundle_")
        self.root = Path(self.temporary.name)
        self.protocol = _Protocol()

    def tearDown(self):
        self.temporary.cleanup()

    def _artifact(self, path: Path):
        return {
            "relative_path": path.relative_to(self.root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    def _write_bundle(self) -> Path:
        entities = {}
        for offset, (entity, filename) in enumerate(FEATURE_FILENAMES.items(), 1):
            matrix = np.vstack(
                (
                    np.linspace(float(offset), float(offset + 1), 64),
                    np.linspace(float(offset + 2), float(offset + 4), 64),
                )
            ).astype(np.float32)
            matrix_path = self.root / filename
            if matrix_path.suffix == ".npy":
                np.save(matrix_path, matrix)
            else:
                np.savetxt(matrix_path, matrix)
            mapping_sha = self.protocol.input_manifest()["mappings"][entity]["sha256"]
            sidecar = feature_row_ids_path(matrix_path)
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": FEATURE_ROW_IDS_SCHEMA,
                        "entity": entity,
                        "mapping_sha256": mapping_sha,
                        "row_ids": list(self.protocol.entity_ids[entity]),
                    }
                ),
                encoding="utf-8",
            )
            entities[entity] = {
                "shape": [2, 64],
                "protocol_mapping_sha256": mapping_sha,
                "matrix": self._artifact(matrix_path),
                "row_ids_sidecar": self._artifact(sidecar),
                "semantic_condition": (
                    PCA64_SEMANTIC_CONDITION
                    if entity in {"herb", "disease"}
                    else None
                ),
            }

        provenance = self.root / "provenance"
        provenance.mkdir()
        model = provenance / "shared_pca64_model_v1.npz"
        np.savez(
            model,
            mean=np.zeros(768),
            components=np.eye(64, 768),
            singular_values=np.ones(64),
            explained_variance_ratio=np.full(64, 1.0 / 64.0),
        )
        herb_raw = provenance / "herb_raw.npy"
        disease_raw = provenance / "disease_raw.npy"
        np.save(herb_raw, np.ones((2, 768), dtype=np.float32))
        np.save(disease_raw, np.full((2, 768), 2.0, dtype=np.float32))
        joint = provenance / "joint.json"
        joint.write_text(
            json.dumps(
                {
                    "herb_row_ids": list(self.protocol.entity_ids["herb"]),
                    "disease_row_ids": list(self.protocol.entity_ids["disease"]),
                    "herb_mapping_sha256": self.protocol.input_manifest()[
                        "mappings"
                    ]["herb"]["sha256"],
                    "disease_mapping_sha256": self.protocol.input_manifest()[
                        "mappings"
                    ]["disease"]["sha256"],
                    "joint_row_identity_sha256": _json_sha256(
                        {
                            "herb": list(self.protocol.entity_ids["herb"]),
                            "disease": list(self.protocol.entity_ids["disease"]),
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        metadata = provenance / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema": "astra-tcm-shared-pca64-metadata/v1",
                    "algorithm": PCA64_ALGORITHM,
                    "whitening": False,
                    "model_sha256": _sha256(model),
                    "joint_row_ids_sha256": _sha256(joint),
                    "herb_input_sha256": _sha256(herb_raw),
                    "disease_input_sha256": _sha256(disease_raw),
                }
            ),
            encoding="utf-8",
        )
        audit = provenance / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "schema": "astra-tcm-shared-pca64-audit/v1",
                    "algorithm": PCA64_ALGORITHM,
                    "status": "verified",
                    "exact_top64_svd_recomputed": True,
                    "unwhitened_projection_reproduced": True,
                    "herb_output_sha256": entities["herb"]["matrix"]["sha256"],
                    "disease_output_sha256": entities["disease"]["matrix"][
                        "sha256"
                    ],
                }
            ),
            encoding="utf-8",
        )
        evidence = provenance / "identity.txt"
        evidence.write_text("identity-bound\n", encoding="utf-8")
        source_manifest = provenance / "source_ae64_bundle_manifest.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "schema": TCM_FEATURE_BUNDLE_SCHEMA,
                    "dataset": "tcm-suite",
                    "protocol": {
                        "input_manifest_sha256": _json_sha256(
                            self.protocol.input_manifest()
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema": PCA64_BUNDLE_SCHEMA,
            "dataset": "tcm-suite",
            "profile": PCA64_BUNDLE_PROFILE,
            "formal_eligible": True,
            "protocol": {
                "input_manifest_sha256": _json_sha256(
                    self.protocol.input_manifest()
                ),
                "entity_mapping_sha256": {
                    entity: self.protocol.input_manifest()["mappings"][entity][
                        "sha256"
                    ]
                    for entity in FEATURE_FILENAMES
                },
            },
            "source_ae64_bundle": {
                "original_sha256": _sha256(source_manifest),
                "copied_manifest": self._artifact(source_manifest),
            },
            "entities": entities,
            "transform": {
                "algorithm": PCA64_ALGORITHM,
                "components": 64,
                "input_dimension": 768,
                "joint_fit_order": ["herb", "disease"],
                "centered": True,
                "whitening": False,
                "model": self._artifact(model),
                "metadata": self._artifact(metadata),
                "audit": self._artifact(audit),
                "joint_row_ids": self._artifact(joint),
            },
            "source_evidence": {
                "identity": self._artifact(evidence),
                "herb_scibert_768d": self._artifact(herb_raw),
                "disease_scibert_768d": self._artifact(disease_raw),
            },
        }
        manifest_path = self.root / "bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_recursive_validator_accepts_bundle_and_rejects_tampering(self):
        manifest = self._write_bundle()
        paths = {
            entity: self.root / filename
            for entity, filename in FEATURE_FILENAMES.items()
        }
        validated = validate_tcm_pca64_feature_bundle(
            manifest,
            protocol=self.protocol,
            expected_feature_paths=paths,
        )
        self.assertEqual(set(validated["entities"]), set(FEATURE_FILENAMES))
        (self.root / "provenance" / "identity.txt").write_text(
            "tampered\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(TCMPCA64BundleError, "bytes or SHA-256"):
            validate_tcm_pca64_feature_bundle(manifest, protocol=self.protocol)


class TCMPCA64BundleBuilderIntegrationTest(unittest.TestCase):
    def test_builder_uses_joint_raw_semantics_without_touching_ae_bundle(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca64_build_") as directory:
            root = Path(directory)
            source = root / "ae_source"
            source.mkdir()

            class Protocol:
                dataset = "tcm-suite"
                entity_ids = {
                    "herb": tuple(f"h{index}" for index in range(33)),
                    "compound": ("c0", "c1"),
                    "protein": ("p0", "p1"),
                    "disease": tuple(f"d{index}" for index in range(32)),
                }

                def input_manifest(self):
                    return {
                        "dataset": self.dataset,
                        "mappings": {
                            entity: {
                                "count": len(ids),
                                "sha256": hashlib.sha256(
                                    (
                                        json.dumps(list(ids), separators=(",", ":"))
                                        + "\n"
                                    ).encode()
                                ).hexdigest(),
                            }
                            for entity, ids in self.entity_ids.items()
                        },
                    }

            protocol = Protocol()
            rng = np.random.default_rng(314)

            def artifact(path: Path):
                return {
                    "relative_path": path.relative_to(source).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }

            entities = {}
            for entity, filename in FEATURE_FILENAMES.items():
                rows = len(protocol.entity_ids[entity])
                matrix = rng.normal(size=(rows, 64)).astype(np.float32)
                matrix_path = source / filename
                if matrix_path.suffix == ".npy":
                    np.save(matrix_path, matrix)
                else:
                    np.savetxt(matrix_path, matrix)
                mapping_sha = protocol.input_manifest()["mappings"][entity]["sha256"]
                sidecar = feature_row_ids_path(matrix_path)
                sidecar.write_text(
                    json.dumps(
                        {
                            "schema": FEATURE_ROW_IDS_SCHEMA,
                            "entity": entity,
                            "id_space": "protocol_raw_id",
                            "mapping_sha256": mapping_sha,
                            "row_ids": list(protocol.entity_ids[entity]),
                        }
                    ),
                    encoding="utf-8",
                )
                entities[entity] = {
                    "shape": [rows, 64],
                    "protocol_mapping_sha256": mapping_sha,
                    "matrix": artifact(matrix_path),
                    "row_ids_sidecar": artifact(sidecar),
                    "semantic_condition": (
                        TCM_SEMANTIC_CONDITION
                        if entity in {"herb", "disease"}
                        else None
                    ),
                }

            provenance = source / "provenance"
            provenance.mkdir()
            raw_paths = {}
            for entity in ("herb", "disease"):
                raw_path = provenance / f"{entity}_raw.npy"
                np.save(
                    raw_path,
                    rng.normal(size=(len(protocol.entity_ids[entity]), 768)).astype(
                        np.float32
                    ),
                )
                raw_paths[entity] = raw_path
                entities[entity]["semantic_evidence"] = {
                    "raw_768d_sha256": _sha256(raw_path)
                }
            source_manifest = {
                "schema": TCM_FEATURE_BUNDLE_SCHEMA,
                "dataset": "tcm-suite",
                "profile": TCM_FEATURE_BUNDLE_PROFILE,
                "formal_eligible": True,
                "protocol": {
                    "input_manifest_sha256": _json_sha256(
                        protocol.input_manifest()
                    ),
                    "entity_mapping_sha256": {
                        entity: protocol.input_manifest()["mappings"][entity][
                            "sha256"
                        ]
                        for entity in FEATURE_FILENAMES
                    },
                },
                "entities": entities,
                "evidence": {
                    "herb_scibert_768d": artifact(raw_paths["herb"]),
                    "disease_scibert_768d": artifact(raw_paths["disease"]),
                },
            }
            source_manifest_path = source / "bundle_manifest.json"
            source_manifest_path.write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}\n", encoding="utf-8")
            before = {
                path.relative_to(source).as_posix(): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            output = root / "pca64_bundle_v1"
            with mock.patch(
                "astra.benchmark.tcm_pca64_feature_bundle.load_protocol_manifest",
                return_value=protocol,
            ):
                manifest_path = build_tcm_pca64_feature_bundle(
                    protocol_manifest=protocol_path,
                    source_bundle_manifest=source_manifest_path,
                    output_dir=output,
                )
            self.assertEqual(manifest_path, output / "bundle_manifest.json")
            self.assertEqual(np.load(output / FEATURE_FILENAMES["herb"]).shape, (33, 64))
            self.assertEqual(
                np.load(output / FEATURE_FILENAMES["disease"]).shape, (32, 64)
            )
            after = {
                path.relative_to(source).as_posix(): _sha256(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
