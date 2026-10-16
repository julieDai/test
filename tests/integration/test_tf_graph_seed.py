"""Tests for fail-closed TensorFlow-v1 graph seeding."""

from __future__ import annotations

import unittest

import numpy as np

from astra.tf_graph_seed import GRAPH_SEED_SCHEMA, seed_fresh_default_graph


class _Graph:
    def __init__(self, operations=()):
        self._operations = tuple(operations)
        self.seed = None

    def get_operations(self):
        return list(self._operations)


class _TensorFlow:
    def __init__(self, graph: _Graph):
        self.graph = graph
        self.calls: list[int] = []
        self.disable_calls = 0

    def get_default_graph(self):
        return self.graph

    def set_random_seed(self, seed: int):
        self.calls.append(seed)
        self.graph.seed = seed

    def disable_v2_behavior(self):
        self.disable_calls += 1


class TensorFlowGraphSeedTest(unittest.TestCase):
    def test_empty_graph_is_seeded_and_receipted(self):
        tensorflow = _TensorFlow(_Graph())

        receipt = seed_fresh_default_graph(42, tensorflow_module=tensorflow)

        self.assertEqual(tensorflow.calls, [42])
        self.assertEqual(tensorflow.disable_calls, 1)
        self.assertEqual(
            receipt,
            {
                "schema": GRAPH_SEED_SCHEMA,
                "seed": 42,
                "graph_was_empty_before_seed": True,
                "tensorflow_v1_behavior_enabled_before_seed": True,
                "seed_set_before_model_import": True,
            },
        )

    def test_nonempty_graph_is_rejected_before_seed_call(self):
        tensorflow = _TensorFlow(_Graph([object()]))

        with self.assertRaisesRegex(RuntimeError, "after graph operations"):
            seed_fresh_default_graph(42, tensorflow_module=tensorflow)

        self.assertEqual(tensorflow.calls, [])
        self.assertEqual(tensorflow.disable_calls, 0)

    def test_boolean_or_negative_seed_is_rejected(self):
        for seed in (True, -1):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seed_fresh_default_graph(seed, tensorflow_module=_TensorFlow(_Graph()))


class TensorFlowGraphSeedIntegrationTest(unittest.TestCase):
    def tearDown(self):
        import tensorflow.compat.v1 as tf

        tf.reset_default_graph()

    @staticmethod
    def _initial_value():
        import tensorflow.compat.v1 as tf

        tf.reset_default_graph()
        receipt = seed_fresh_default_graph(42, tensorflow_module=tf)
        variable = tf.Variable(
            tf.keras.initializers.glorot_normal()([64, 64]),
            name="shared_weight",
        )
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            value = session.run(variable)
        return receipt, value

    def test_real_tensorflow_replays_identical_initial_values(self):
        first_receipt, first = self._initial_value()
        second_receipt, second = self._initial_value()

        self.assertEqual(first_receipt, second_receipt)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
