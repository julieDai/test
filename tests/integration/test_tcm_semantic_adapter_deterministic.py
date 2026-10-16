"""Contract tests for the versioned deterministic TCM adapter rerun."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark import astra_runner
from astra.benchmark import tcm_astra_semantic_adapter_deterministic_runner as runner
from astra.benchmark.tcm_astra_semantic_adapter_deterministic_plan import (
    DETERMINISM_SCHEMA,
    PLAN_SCHEMA,
    build_plan,
)
from astra.benchmark.tcm_astra_semantic_adapter_pair_guard import (
    TCMAdapterPairGuardError,
    require_identical_shared_initialization,
)
from astra.tf_graph_seed import GRAPH_SEED_SCHEMA


class TCMDeterministicRuntimeTest(unittest.TestCase):
    def test_runtime_seed_is_after_environment_and_override_is_restored(self):
        events: list[tuple[str, int]] = []
        original_runtime = astra_runner._configure_runtime
        original_loader = astra_runner._load_legacy_components
        original_configure = astra_runner._configure_model
        original_hyperparameters = astra_runner._HYPERPARAMETERS["tcm-suite"]
        binding = {
            "epochs": 32,
            "learning_rate": 0.003,
            "adapter": "residual_mlp",
            "deterministic_initialization": {
                "tensorflow_graph_seed": 42,
            },
        }

        def fake_runtime(seed, _device, _output_root):
            events.append(("runtime", seed))

        def fake_seed(seed):
            self.assertEqual(events, [("runtime", 42)])
            events.append(("tensorflow", seed))
            return {
                "schema": GRAPH_SEED_SCHEMA,
                "seed": seed,
                "graph_was_empty_before_seed": True,
                "tensorflow_v1_behavior_enabled_before_seed": True,
                "seed_set_before_model_import": True,
            }

        with patch.object(
            astra_runner, "_configure_runtime", side_effect=fake_runtime
        ) as patched_runtime, patch.object(
            runner, "seed_fresh_default_graph", side_effect=fake_seed
        ):
            with runner._candidate_runtime(binding) as receipt:
                astra_runner._configure_runtime(42, "cuda", Path("/tmp/output"))
                self.assertEqual(receipt["seed"], 42)
                self.assertEqual(os.environ.get("ASTRA_TF_GRAPH_SEED"), "42")
            self.assertIs(astra_runner._configure_runtime, patched_runtime)

        self.assertEqual(events, [("runtime", 42), ("tensorflow", 42)])
        self.assertNotIn("ASTRA_TF_GRAPH_SEED", os.environ)
        self.assertIs(astra_runner._configure_runtime, original_runtime)
        self.assertIs(astra_runner._load_legacy_components, original_loader)
        self.assertIs(astra_runner._configure_model, original_configure)
        self.assertIs(
            astra_runner._HYPERPARAMETERS["tcm-suite"],
            original_hyperparameters,
        )
class TCMDeterministicPlanTest(unittest.TestCase):
    def test_new_plan_binds_deterministic_sources_and_failed_v2_plan(self):
        with tempfile.TemporaryDirectory(prefix="astra_deterministic_plan_") as raw:
            root = Path(raw)
            failed = root / "plan.json"
            failed.write_text(
                json.dumps(
                    {
                        "schema": "astra-tcm-semantic-adapter-plan/v2",
                        "status": "frozen-before-stage2-candidate-testing",
                        "dataset": "tcm-suite",
                        "seed": 42,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(failed.read_bytes()).hexdigest()
            failed.with_suffix(".json.sha256").write_text(
                f"{digest}  {failed.name}\n", encoding="utf-8"
            )
            base_payload = {
                "schema": "astra-tcm-semantic-adapter-plan/v2",
                "status": "frozen-before-stage2-candidate-testing",
                "dataset": "tcm-suite",
                "seed": 42,
                "source_freeze": {
                    "provenance_epoch_boundary_decision": {
                        "path": str(failed),
                        "bytes": failed.stat().st_size,
                        "sha256": digest,
                    }
                },
            }
            with patch(
                "astra.benchmark.tcm_astra_semantic_adapter_deterministic_plan."
                "base_plan.build_plan",
                return_value=base_payload,
            ):
                plan = build_plan(root / "decision.json", failed)

        self.assertEqual(plan["schema"], PLAN_SCHEMA)
        self.assertEqual(
            plan["deterministic_initialization"],
            {
                "schema": DETERMINISM_SCHEMA,
                "tensorflow_graph_seed": 42,
                "graph_must_be_empty_before_seed": True,
                "tensorflow_v1_behavior_enabled_before_seed": True,
                "seed_set_after_runtime_environment": True,
                "seed_set_before_model_import": True,
                "same_fold_shared_initialization_required": True,
            },
        )
        for key in (
            "model",
            "base_model",
            "runner",
            "base_runner",
            "selector",
            "pair_guard",
            "tensorflow_graph_seed",
            "launcher",
        ):
            self.assertIn(key, plan["source_freeze"])
        self.assertEqual(plan["failed_v2_plan"]["sha256"], digest)


class TCMAdapterPairGuardTest(unittest.TestCase):
    def test_complete_shared_mapping_must_match_exactly(self):
        records = [
            {"shared_initial_variable_sha256": {"weight/a": "a" * 64}},
            {"shared_initial_variable_sha256": {"weight/a": "a" * 64}},
        ]
        self.assertEqual(
            require_identical_shared_initialization(records, fold=1),
            {"weight/a": "a" * 64},
        )

        records[1]["shared_initial_variable_sha256"]["weight/a"] = "b" * 64
        with self.assertRaisesRegex(
            TCMAdapterPairGuardError, "different shared initialization"
        ):
            require_identical_shared_initialization(records, fold=1)

    def test_missing_arm_is_rejected(self):
        with self.assertRaisesRegex(TCMAdapterPairGuardError, "exactly two"):
            require_identical_shared_initialization([], fold=3)


if __name__ == "__main__":
    unittest.main()
