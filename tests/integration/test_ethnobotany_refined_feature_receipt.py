"""Tests for the path-bound ASTRARefined feature audit fast path."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from astra.benchmark.ethnobotany_refined_feature_receipt import (
    FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA,
    FeatureDeepAuditReceiptError,
    _sha256,
    create_feature_deep_audit_receipt,
    validate_feature_deep_audit_receipt,
)
from astra.benchmark.ethnobotany_refined_runner import PreparedFeatures


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bundle_record(path: Path, root: Path) -> dict[str, object]:
    return {
        **_record(path),
        "relative_path": str(path.resolve().relative_to(root.resolve())),
    }


class RefinedFeatureReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="astra_feature_receipt_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.bundle = self.root / "bundle"
        self.data.mkdir()
        self.bundle.mkdir()

        self.protocol_manifest = self.root / "protocol.json"
        self.protocol_manifest.write_text('{"dataset":"ethnobotany"}', encoding="utf-8")
        source = self.data / "H_D.txt"
        source.write_text("h d 1\n", encoding="utf-8")
        self.protocol = SimpleNamespace(
            source_files={"H_D": {"path": "/frozen/H_D.txt"}}
        )

        provenance = self.bundle / "provenance.json"
        provenance.write_text('{"audited":true}', encoding="utf-8")
        entities: dict[str, object] = {}
        selected_paths: dict[str, Path] = {}
        mask_paths: dict[str, Path] = {}
        selected_diagnostics: dict[str, object] = {}
        for entity in ("herb", "compound", "protein", "disease"):
            matrix = self.bundle / f"{entity}.npy"
            sidecar = self.bundle / f"{entity}.npy.row_ids.json"
            mask = self.bundle / f"{entity}.mask.npy"
            mapping = self.bundle / f"{entity}.csv"
            reducer = self.bundle / f"{entity}.reducer.npz"
            matrix.write_bytes(f"matrix-{entity}".encode())
            sidecar.write_text(json.dumps({"entity": entity}), encoding="utf-8")
            mask.write_bytes(f"mask-{entity}".encode())
            mapping.write_text(f"id,name\n0,{entity}\n", encoding="utf-8")
            reducer.write_bytes(f"reducer-{entity}".encode())
            entities[entity] = {
                "matrix": _bundle_record(matrix, self.bundle),
                "row_ids_sidecar": _bundle_record(sidecar, self.bundle),
                "missing_mask": _bundle_record(mask, self.bundle),
                "mapping": _bundle_record(mapping, self.bundle),
                "component_manifest": _bundle_record(provenance, self.bundle),
                "assembly_manifest": _bundle_record(provenance, self.bundle),
                "reducer_checkpoint": _bundle_record(reducer, self.bundle),
                "retrieval_cache": None,
            }
            selected_paths[entity] = matrix
            mask_paths[entity] = mask
            selected_diagnostics[entity] = {
                "path": str(matrix.resolve()),
                "used_path": str(matrix.resolve()),
                "used_sha256": _sha256(matrix),
                "sidecar_path": str(sidecar.resolve()),
                "sidecar_sha256": _sha256(sidecar),
                "shape": [1, 64],
                "dtype": "float32",
                "canonical_array_sha256": "a" * 64,
                "source_order_is_canonical": True,
                "degenerate": False,
            }
        (self.bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "astra-ethnobotany-feature-bundle/v1",
                    "entities": entities,
                    "provenance_objects": [
                        _bundle_record(provenance, self.bundle)
                    ],
                }
            ),
            encoding="utf-8",
        )

        self.semantic_manifest = self.root / "semantic.json"
        self.semantic_manifest.write_text('{"output_dimension":64}', encoding="utf-8")
        semantic_dependency = self.root / "semantic_dependency.json"
        semantic_dependency.write_text('{"verified":true}', encoding="utf-8")
        semantic_source = self.root / "semantic_validator.py"
        semantic_source.write_text("# fixed validator\n", encoding="utf-8")
        self.semantic_receipt = self.root / "semantic_receipt.json"
        self.semantic_receipt.write_text(
            json.dumps(
                {
                    "schema": "astra-ethnobotany-semantic-deep-audit-receipt/v1",
                    "deep_audit": True,
                    "dependencies": [
                        _record(self.semantic_manifest),
                        _record(semantic_dependency),
                    ],
                    "validator_sources": [_record(semantic_source)],
                }
            ),
            encoding="utf-8",
        )
        self.semantic_dependency = semantic_dependency
        self.prepared = PreparedFeatures(
            paths=selected_paths,
            mask_paths=mask_paths,
            diagnostics={
                "protocol_sources": {"H_D": _record(source)},
                "base_bundle": {"formal_eligible": True},
                "semantic_reduction": {
                    "output_dimension": 64,
                    "formal_eligible": True,
                },
                "selected_features": selected_diagnostics,
            },
        )
        self.receipt = self.root / "feature_receipt.json"

    def _create(self) -> Path:
        return create_feature_deep_audit_receipt(
            receipt_path=self.receipt,
            protocol_manifest=self.protocol_manifest,
            protocol=self.protocol,
            data_dir=self.data,
            embedding_dir=self.bundle,
            semantic_reduction_manifest=self.semantic_manifest,
            semantic_deep_audit_receipt=self.semantic_receipt,
            prepared=self.prepared,
        )

    def _validate(self):
        return validate_feature_deep_audit_receipt(
            receipt_path=self.receipt,
            protocol_manifest=self.protocol_manifest,
            protocol=self.protocol,
            data_dir=self.data,
            embedding_dir=self.bundle,
            semantic_reduction_manifest=self.semantic_manifest,
            semantic_deep_audit_receipt=self.semantic_receipt,
        )

    def test_receipt_rehashes_unique_closure_and_restores_canonical_paths(self) -> None:
        self._create()
        payload = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], FEATURE_DEEP_AUDIT_RECEIPT_SCHEMA)
        dependency_paths = [item["path"] for item in payload["dependencies"]]
        self.assertEqual(len(dependency_paths), len(set(dependency_paths)))

        restored = self._validate()
        self.assertEqual(restored["paths"], self.prepared.paths)
        self.assertEqual(restored["mask_paths"], self.prepared.mask_paths)
        audit = restored["diagnostics"]["feature_deep_audit_receipt"]
        self.assertTrue(audit["all_dependencies_rehashed"])
        self.assertTrue(audit["canonical_paths_reused"])
        self.assertEqual(audit["dependency_count"], len(dependency_paths))

    def test_any_dependency_change_invalidates_receipt(self) -> None:
        self._create()
        self.semantic_dependency.write_text('{"verified":false}', encoding="utf-8")
        with self.assertRaisesRegex(
            FeatureDeepAuditReceiptError,
            "differs from its feature deep-audit receipt",
        ):
            self._validate()

    def test_noncanonical_materialised_path_is_rejected(self) -> None:
        diagnostics = self.prepared.diagnostics["selected_features"]
        diagnostics["herb"]["source_order_is_canonical"] = False
        with self.assertRaisesRegex(
            FeatureDeepAuditReceiptError, "only reuse original.*canonical"
        ):
            self._create()

    def test_receipt_is_atomic_and_not_overwritten(self) -> None:
        self._create()
        original = self.receipt.read_bytes()
        with self.assertRaisesRegex(
            FeatureDeepAuditReceiptError, "refusing to overwrite"
        ):
            self._create()
        self.assertEqual(self.receipt.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
