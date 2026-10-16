"""Fail-closed TensorFlow-v1 graph seeding for fresh benchmark processes.

TensorFlow is imported lazily so callers can establish ``CUDA_VISIBLE_DEVICES``
before the runtime is loaded.  The helper deliberately refuses a graph that
already contains operations: setting a graph seed after random initializers
have been created would give a misleading determinism receipt.
"""

from __future__ import annotations

from typing import Any


GRAPH_SEED_SCHEMA = "astra-tensorflow-v1-graph-seed/v1"


def seed_fresh_default_graph(
    seed: int, *, tensorflow_module: Any | None = None
) -> dict[str, object]:
    """Seed an empty TensorFlow-v1 default graph and return an audit receipt."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("TensorFlow graph seed must be a non-negative integer")
    if tensorflow_module is None:
        import tensorflow.compat.v1 as tensorflow_module

    graph = tensorflow_module.get_default_graph()
    operations = tuple(graph.get_operations())
    if operations:
        raise RuntimeError(
            "refusing to set the TensorFlow graph seed after graph operations "
            "were created"
        )
    disable_v2 = getattr(tensorflow_module, "disable_v2_behavior", None)
    if not callable(disable_v2):
        raise RuntimeError("TensorFlow-v1 compatibility mode is unavailable")
    disable_v2()
    graph = tensorflow_module.get_default_graph()
    if tuple(graph.get_operations()):
        raise RuntimeError(
            "TensorFlow-v1 compatibility setup created graph operations before "
            "the graph seed was fixed"
        )
    tensorflow_module.set_random_seed(seed)
    observed = getattr(graph, "seed", None)
    if observed != seed:
        raise RuntimeError(
            f"TensorFlow default graph recorded seed {observed!r}, expected {seed}"
        )
    return {
        "schema": GRAPH_SEED_SCHEMA,
        "seed": seed,
        "graph_was_empty_before_seed": True,
        "tensorflow_v1_behavior_enabled_before_seed": True,
        "seed_set_before_model_import": True,
    }
