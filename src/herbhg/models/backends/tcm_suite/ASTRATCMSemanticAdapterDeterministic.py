"""Deterministically seeded TCM semantic-adapter model entry point.

The learned architecture is inherited unchanged from the frozen v2 adapter.
This version only fails closed unless its runner seeded an empty TensorFlow-v1
graph before importing and constructing the model, and records that fact in
the adapter audit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import tensorflow.compat.v1 as tf

from ASTRATCMSemanticAdapter import (
    ASTRATCMSemanticAdapter,
    _atomic_json,
    _file_sha256,
)
from astra.tf_graph_seed import GRAPH_SEED_SCHEMA


DETERMINISTIC_MODEL_SCHEMA = "astra-tcm-semantic-adapter-deterministic-model/v1"


class ASTRATCMSemanticAdapterDeterministic(ASTRATCMSemanticAdapter):
    """The frozen adapter architecture with a mandatory pre-build graph seed."""

    def __init__(self, conf, trainingSet=None, testSet=None, fold="[1]"):
        raw_seed = os.environ.get("ASTRA_TF_GRAPH_SEED")
        if raw_seed is None:
            raise RuntimeError("ASTRA_TF_GRAPH_SEED is absent")
        try:
            expected_seed = int(raw_seed)
        except ValueError as error:
            raise RuntimeError("ASTRA_TF_GRAPH_SEED is not an integer") from error
        benchmark_seed = int(os.environ.get("ASTRA_SEED", expected_seed))
        observed_seed = getattr(tf.get_default_graph(), "seed", None)
        if expected_seed != benchmark_seed or observed_seed != expected_seed:
            raise RuntimeError(
                "TensorFlow graph seed does not match the benchmark seed: "
                f"graph={observed_seed!r}, expected={expected_seed}, "
                f"benchmark={benchmark_seed}"
            )
        self.deterministic_initialization_receipt = {
            "schema": DETERMINISTIC_MODEL_SCHEMA,
            "graph_seed_schema": GRAPH_SEED_SCHEMA,
            "tensorflow_graph_seed": expected_seed,
            "seed_observed_before_model_construction": True,
        }
        super().__init__(conf, trainingSet=trainingSet, testSet=testSet, fold=fold)

    def _write_adapter_audit(
        self,
        *,
        output_dir: Path,
        initial: Mapping[str, object],
        final: Mapping[str, object],
        optimizer_steps: int,
    ) -> Path:
        path = super()._write_adapter_audit(
            output_dir=output_dir,
            initial=initial,
            final=final,
            optimizer_steps=optimizer_steps,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_source = dict(payload["model_source"])
        source = Path(__file__).resolve()
        payload.update(
            {
                "model_source": {
                    "path": str(source),
                    "sha256": _file_sha256(source),
                },
                "base_model_source": base_source,
                "deterministic_initialization": dict(
                    self.deterministic_initialization_receipt
                ),
            }
        )
        _atomic_json(path, payload)
        self.adapter_audit_receipt = payload
        self.adapter_audit_receipt_path = path
        return path


__all__ = ["ASTRATCMSemanticAdapterDeterministic"]
