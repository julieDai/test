from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from astra.benchmark.astra_runner import (
    _legacy_ethnobotany_semantic_paths,
    _rebind_model_mappings,
    _smoke_evaluation_rows,
    _smoke_training_rows,
    compact_pairs_to_training_rows,
    score_compact_pairs,
)


class _Protocol:
    entity_ids = {
        "herb": ("h20", "h10"),
        "compound": ("c9", "c3"),
        "protein": ("p8", "p2"),
        "disease": ("d7", "d1"),
    }

    @staticmethod
    def compact_to_raw(entity, compact):
        return _Protocol.entity_ids[entity][int(compact)]


def _fake_model():
    data = SimpleNamespace(
        herb={"h10": 0, "h20": 1}, disease={"d1": 0, "d7": 1},
        compound={"c3": 0, "c9": 1}, protein={"p2": 0, "p8": 1},
        id2herb={}, id2disease={}, id2compound={}, id2protein={},
        Hlist=np.array(["h10", "h20"]), Dlist=np.array(["d1", "d7"]),
        Clist=np.array(["c3", "c9"]), Plist=np.array(["p2", "p8"]),
        hcassociation=[["h20", "c9"], ["h10", "c3"]],
        cpassociation=[["c9", "p8"], ["c3", "p2"]],
        pdassociation=[["p8", "d7"], ["p2", "d1"]],
        trainingData=[["h20", "d7", 1.0], ["h10", "d1", 0.0]],
    )
    return SimpleNamespace(data=data, num_herbs=2, num_diseases=2, num_compounds=2, num_proteins=2)


class ASTRABenchmarkRunnerTest(unittest.TestCase):
    def test_legacy_semantic_diagnostic_is_shape_checked_and_unverified(self):
        protocol = SimpleNamespace(n_herbs=3, n_diseases=2)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            np.save(
                directory / "herb_semantic_embeddings_64d.npy",
                np.arange(3 * 64, dtype=np.float32).reshape(3, 64),
            )
            np.save(
                directory / "disease_semantic_embeddings_64d.npy",
                np.arange(2 * 64, dtype=np.float32).reshape(2, 64),
            )
            paths, diagnostics = _legacy_ethnobotany_semantic_paths(
                directory, protocol
            )
        self.assertEqual(set(paths), {"herb", "disease"})
        self.assertFalse(diagnostics["herb"]["formal_eligible"])
        self.assertEqual(
            diagnostics["disease"]["semantic_identity_status"],
            "unverified_assumed_historical_model_index_order",
        )

    def test_compact_pairs_are_converted_back_to_raw_ids(self):
        rows = compact_pairs_to_training_rows(
            _Protocol(), np.array([[0, 1], [1, 0]]), np.array([1, 0])
        )
        self.assertEqual(rows, [["h20", "d1", 1.0], ["h10", "d7", 0.0]])

    def test_model_mappings_are_rebound_before_scoring(self):
        model = _fake_model()
        _rebind_model_mappings(model, _Protocol())
        self.assertEqual(model.data.herb, {"h20": 0, "h10": 1})
        self.assertEqual(model.data.disease, {"d7": 0, "d1": 1})
        model.u = np.array([[2.0, 0.0], [0.0, 2.0]])
        model.i = np.array([[2.0, 0.0], [0.0, 2.0]])
        scores = score_compact_pairs(model, _Protocol(), np.array([[0, 0], [0, 1], [1, 1]]))
        np.testing.assert_allclose(scores, [1 / (1 + np.exp(-4)), 0.5, 1 / (1 + np.exp(-4))])

    def test_smoke_subsets_are_balanced_deterministic_and_row_aligned(self):
        rows = SimpleNamespace(
            train_pairs=np.arange(80, dtype=np.int64).reshape(40, 2),
            train_labels=np.array([1] * 20 + [0] * 20, dtype=np.uint8),
        )
        pairs, labels = _smoke_training_rows(rows, per_class=3)
        self.assertEqual(labels.tolist(), [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(pairs[:3], rows.train_pairs[:3])
        eval_pairs, eval_labels, eval_rows = _smoke_evaluation_rows(
            rows.train_pairs, rows.train_labels, np.arange(40), per_class=2
        )
        self.assertEqual(eval_labels.tolist(), [1, 1, 0, 0])
        self.assertEqual(eval_rows.tolist(), [0, 1, 20, 21])
        np.testing.assert_array_equal(eval_pairs, rows.train_pairs[eval_rows])


if __name__ == "__main__":
    unittest.main()
