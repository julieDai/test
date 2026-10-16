from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from astra.benchmark.astra_runner import (
    ASTRARunnerError,
    _assert_formal_feature_integrity,
)
from astra.benchmark.features import (
    FEATURE_ROW_IDS_SCHEMA,
    FeatureContractError,
    feature_row_ids_path,
    load_aligned_feature_matrix,
)

MAPPING_SHA256 = "a" * 64


def _write_sidecar(path: Path, entity: str, row_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": FEATURE_ROW_IDS_SCHEMA,
                "entity": entity,
                "id_space": "protocol_raw_id",
                "mapping_sha256": MAPPING_SHA256,
                "row_ids": row_ids,
            }
        ),
        encoding="utf-8",
    )


class BenchmarkFeatureContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_feature_contract_")
        self.root = Path(self.temporary.name)
        self.matrix_path = self.root / "compound_embedding_kpgt.txt"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_permuted_rows_are_joined_to_identical_canonical_matrix(self) -> None:
        canonical_ids = ["1", "2", "3"]
        canonical_matrix = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64
        )

        np.savetxt(self.matrix_path, canonical_matrix)
        _write_sidecar(
            feature_row_ids_path(self.matrix_path), "compound", canonical_ids
        )
        canonical = load_aligned_feature_matrix(
            self.matrix_path,
            entity="compound",
            canonical_ids=canonical_ids,
            mapping_sha256=MAPPING_SHA256,
            expected_columns=2,
        )

        permutation = [2, 0, 1]
        permuted_path = self.root / "permuted.txt"
        np.savetxt(permuted_path, canonical_matrix[permutation])
        _write_sidecar(
            feature_row_ids_path(permuted_path),
            "compound",
            [canonical_ids[index] for index in permutation],
        )
        permuted = load_aligned_feature_matrix(
            permuted_path,
            entity="compound",
            canonical_ids=canonical_ids,
            mapping_sha256=MAPPING_SHA256,
            expected_columns=2,
        )

        np.testing.assert_array_equal(permuted.matrix, canonical.matrix)
        self.assertTrue(canonical.diagnostics["source_order_is_canonical"])
        self.assertFalse(permuted.diagnostics["source_order_is_canonical"])
        self.assertEqual(
            canonical.diagnostics["canonical_array_sha256"],
            permuted.diagnostics["canonical_array_sha256"],
        )

    def test_missing_duplicate_and_unexpected_ids_fail_closed(self) -> None:
        sidecar = feature_row_ids_path(self.matrix_path)
        cases = (
            (["1", "2"], "missing=1"),
            (["1", "1", "3"], "duplicate IDs"),
            (["1", "2", "3", "4"], "unexpected=1"),
        )
        for row_ids, message in cases:
            with self.subTest(row_ids=row_ids):
                np.savetxt(
                    self.matrix_path,
                    np.arange(2 * len(row_ids), dtype=float).reshape(len(row_ids), 2),
                )
                _write_sidecar(sidecar, "compound", row_ids)
                with self.assertRaisesRegex(FeatureContractError, message):
                    load_aligned_feature_matrix(
                        self.matrix_path,
                        entity="compound",
                        canonical_ids=["1", "2", "3"],
                        mapping_sha256=MAPPING_SHA256,
                        expected_columns=2,
                    )

    def test_missing_sidecar_fails_closed(self) -> None:
        np.savetxt(self.matrix_path, np.arange(6, dtype=float).reshape(3, 2))
        with self.assertRaisesRegex(FeatureContractError, "missing compound feature"):
            load_aligned_feature_matrix(
                self.matrix_path,
                entity="compound",
                canonical_ids=["1", "2", "3"],
                mapping_sha256=MAPPING_SHA256,
                expected_columns=2,
            )

    def test_sidecar_is_bound_to_the_frozen_mapping_hash(self) -> None:
        np.savetxt(self.matrix_path, np.arange(6, dtype=float).reshape(3, 2))
        _write_sidecar(
            feature_row_ids_path(self.matrix_path),
            "compound",
            ["1", "2", "3"],
        )
        with self.assertRaisesRegex(FeatureContractError, "mapping_sha256"):
            load_aligned_feature_matrix(
                self.matrix_path,
                entity="compound",
                canonical_ids=["1", "2", "3"],
                mapping_sha256="b" * 64,
                expected_columns=2,
            )

    def test_degenerate_matrix_is_rejected_for_formal_but_reported_for_smoke(self) -> None:
        np.savetxt(self.matrix_path, np.ones((3, 2), dtype=float))
        _write_sidecar(
            feature_row_ids_path(self.matrix_path),
            "compound",
            ["1", "2", "3"],
        )
        loaded = load_aligned_feature_matrix(
            self.matrix_path,
            entity="compound",
            canonical_ids=["1", "2", "3"],
            mapping_sha256=MAPPING_SHA256,
            expected_columns=2,
        )
        diagnostics = {"compound": loaded.diagnostics}
        self.assertEqual(
            _assert_formal_feature_integrity(diagnostics, smoke=True),
            ["compound"],
        )
        with self.assertRaisesRegex(ASTRARunnerError, "formal runs reject"):
            _assert_formal_feature_integrity(diagnostics, smoke=False)


if __name__ == "__main__":
    unittest.main()
