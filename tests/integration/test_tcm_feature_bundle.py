from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from astra.benchmark.astra_runner import _validate_and_align_feature_paths
from astra.benchmark.features import FEATURE_ROW_IDS_SCHEMA, feature_row_ids_path
from astra.benchmark.tcm_feature_bundle import (
    FEATURE_FILENAMES,
    TCM_FEATURE_BUNDLE_PROFILE,
    TCM_FEATURE_BUNDLE_SCHEMA,
    TCM_SEMANTIC_CONDITION,
    TCMFeatureBundleError,
    validate_tcm_feature_bundle,
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
    source_files = {}
    n_herbs = n_compounds = n_proteins = n_diseases = 2

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


class TCMFeatureBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_tcm_bundle_")
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

    def _write_bundle(self):
        entities = {}
        for offset, (entity, filename) in enumerate(FEATURE_FILENAMES.items(), 1):
            matrix = np.vstack(
                (
                    np.linspace(float(offset), float(offset + 1), 64),
                    np.linspace(float(offset + 2), float(offset + 4), 64),
                )
            )
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
                        "id_space": "protocol_raw_id",
                        "mapping_sha256": mapping_sha,
                        "row_ids": list(self.protocol.entity_ids[entity]),
                    }
                )
            )
            entities[entity] = {
                "shape": [2, 64],
                "protocol_mapping_sha256": mapping_sha,
                "matrix": self._artifact(matrix_path),
                "row_ids_sidecar": self._artifact(sidecar),
                "semantic_condition": (
                    TCM_SEMANTIC_CONDITION
                    if entity in {"herb", "disease"}
                    else None
                ),
            }
        evidence_path = self.root / "provenance" / "identity.txt"
        evidence_path.parent.mkdir()
        evidence_path.write_text("verified\n")
        manifest = {
            "schema": TCM_FEATURE_BUNDLE_SCHEMA,
            "dataset": "tcm-suite",
            "profile": TCM_FEATURE_BUNDLE_PROFILE,
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
            "entities": entities,
            "evidence": {"identity": self._artifact(evidence_path)},
        }
        manifest_path = self.root / "bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_validator_accepts_recursive_identity_bound_bundle(self):
        manifest = self._write_bundle()
        paths = {
            entity: self.root / filename
            for entity, filename in FEATURE_FILENAMES.items()
        }
        validated = validate_tcm_feature_bundle(
            manifest,
            protocol=self.protocol,
            expected_feature_paths=paths,
        )
        self.assertEqual(set(validated["entities"]), set(FEATURE_FILENAMES))
        self.assertEqual(len(validated["evidence_paths"]), 1)

    def test_validator_rejects_tampered_evidence(self):
        manifest = self._write_bundle()
        (self.root / "provenance" / "identity.txt").write_text("tampered\n")
        with self.assertRaisesRegex(TCMFeatureBundleError, "bytes or SHA-256"):
            validate_tcm_feature_bundle(manifest, protocol=self.protocol)

    def test_tcm_alignment_uses_protocol_mapping_digest_and_ids(self):
        self._write_bundle()
        compound = self.root / FEATURE_FILENAMES["compound"]
        values = np.loadtxt(compound)
        np.savetxt(compound, values[::-1])
        sidecar = feature_row_ids_path(compound)
        payload = json.loads(sidecar.read_text())
        payload["row_ids"] = list(reversed(payload["row_ids"]))
        sidecar.write_text(json.dumps(payload))
        paths = {
            entity: self.root / filename
            for entity, filename in FEATURE_FILENAMES.items()
        }
        aligned, diagnostics = _validate_and_align_feature_paths(
            "tcm-suite", self.protocol, paths, self.root / "run"
        )
        np.testing.assert_allclose(np.loadtxt(aligned["compound"]), values)
        self.assertFalse(diagnostics["compound"]["source_order_is_canonical"])


if __name__ == "__main__":
    unittest.main()
