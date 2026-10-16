from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from astra.benchmark import tcm_astra_pca64_deterministic_runner as runner
from astra.benchmark.tcm_astra_pca64_control_plan import (
    PLAN_SCHEMA as OLD_PLAN_SCHEMA,
)
from astra.benchmark.tcm_astra_pca64_control_selection import (
    SELECTION_SCHEMA as OLD_SELECTION_SCHEMA,
)
from astra.benchmark.tcm_astra_pca64_deterministic_pair_guard import (
    TCMPCA64DeterministicPairGuardError,
    _semantic_matrix_hash,
    compare_initialization_maps,
)
from astra.benchmark.tcm_astra_pca64_deterministic_plan import (
    CONTROLLED_FEATURE_INPUT_KEYS,
    EXPECTED_RESIDUAL_ADAPTER_KEYS,
    EXPECTED_SHARED_INITIAL_KEYS,
    PLAN_SCHEMA,
    TCMPCA64DeterministicPlanError,
    _source_freeze,
    validate_sidecar,
)
from astra.benchmark.tcm_astra_pca64_deterministic_selection import (
    SELECTION_SCHEMA,
    select_feature_basis,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _shared(prefix: str = "same") -> dict[str, str]:
    return {key: _digest(f"{prefix}:{key}") for key in EXPECTED_SHARED_INITIAL_KEYS}


def _adapter() -> dict[str, str]:
    return {key: _digest(key) for key in EXPECTED_RESIDUAL_ADAPTER_KEYS}


class TCMPCA64DeterministicSchemaTest(unittest.TestCase):
    def test_v2_schemas_are_isolated_from_outer_capable_v1(self):
        self.assertEqual(PLAN_SCHEMA, "astra-tcm-pca64-deterministic-plan/v2")
        self.assertEqual(
            SELECTION_SCHEMA, "astra-tcm-pca64-deterministic-selection/v2"
        )
        self.assertNotEqual(PLAN_SCHEMA, OLD_PLAN_SCHEMA)
        self.assertNotEqual(SELECTION_SCHEMA, OLD_SELECTION_SCHEMA)

    def test_source_freeze_has_no_outer_evaluator(self):
        frozen = _source_freeze()
        self.assertNotIn("outer_evaluator", frozen)
        self.assertNotIn("outer_safety_primitives", frozen)
        for key in (
            "model",
            "tensorflow_graph_seed",
            "deterministic_adapter_runner",
            "pca64_bundle_builder",
            "candidate_runner",
            "pair_guard",
            "selector",
            "prepare_launcher",
            "run_launcher",
        ):
            self.assertIn(key, frozen)
            self.assertTrue(Path(frozen[key]["path"]).is_file())

    def test_plan_sidecar_hash_is_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca_v2_sidecar_") as raw:
            path = Path(raw) / "plan.json"
            path.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".json.sha256").write_text(
                f"{digest}  {path.name}\n", encoding="utf-8"
            )
            self.assertEqual(
                validate_sidecar(path, label="test plan"), digest
            )
            path.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                TCMPCA64DeterministicPlanError, "sidecar differs"
            ):
                validate_sidecar(path, label="test plan")

    def test_prepare_requires_v3_selection_receipts_and_launcher_marker(self):
        source = Path(
            "tools/prepare_tcm_astra_pca64_deterministic_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("semantic_adapter_deterministic_v3", source)
        self.assertIn("selection.json", source)
        self.assertIn(".astra_semantic_adapter_deterministic_v3_complete", source)
        self.assertIn("for fold in 1 2 3 4 5", source)
        self.assertIn("pair_receipts/fold_${fold}.json", source)


class TCMPCA64RuntimeCompositionTest(unittest.TestCase):
    def test_pca_bundle_wraps_exact_v3_seeded_runtime(self):
        events: list[str] = []
        receipt = {
            "schema": "astra-tensorflow-v1-graph-seed/v1",
            "seed": 42,
            "graph_was_empty_before_seed": True,
            "tensorflow_v1_behavior_enabled_before_seed": True,
            "seed_set_before_model_import": True,
        }

        @contextmanager
        def fake_bundle(_binding):
            events.append("bundle-enter")
            try:
                yield
            finally:
                events.append("bundle-exit")

        @contextmanager
        def fake_deterministic(_binding):
            self.assertEqual(events, ["bundle-enter"])
            events.append("seeded-runtime-enter")
            try:
                yield receipt
            finally:
                events.append("seeded-runtime-exit")

        with patch.object(
            runner.pca_base_runner,
            "_pca64_bundle_runtime",
            side_effect=fake_bundle,
        ), patch.object(
            runner.deterministic_runner,
            "_candidate_runtime",
            side_effect=fake_deterministic,
        ):
            with runner.deterministic_pca64_runtime({}) as observed:
                self.assertIs(observed, receipt)
                events.append("body")

        self.assertEqual(
            events,
            [
                "bundle-enter",
                "seeded-runtime-enter",
                "body",
                "seeded-runtime-exit",
                "bundle-exit",
            ],
        )

    def test_outer_prediction_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca_v2_outer_") as raw:
            root = Path(raw)
            (root / "results").mkdir()
            (root / "results" / "test_predictions.npz").write_bytes(b"outer")
            with self.assertRaisesRegex(RuntimeError, "outer-test artifact"):
                runner.pca_base_runner.reject_outer_artifacts(root)


class TCMPCA64FeatureBasisPairTest(unittest.TestCase):
    def test_identity_pair_allows_exactly_two_controlled_input_differences(self):
        ae = _shared()
        pca = dict(ae)
        for key in CONTROLLED_FEATURE_INPUT_KEYS:
            pca[key] = _digest(f"pca:{key}")
        receipt = compare_initialization_maps(
            ae_shared=ae,
            pca_shared=pca,
            ae_adapter={},
            pca_adapter={},
            adapter="identity",
        )
        self.assertEqual(len(receipt["common_shared_variable_sha256"]), 30)
        self.assertTrue(receipt["same_noncontrolled_shared_initialization"])
        self.assertTrue(receipt["different_controlled_semantic_inputs"])
        self.assertEqual(receipt["adapter_initial_variable_sha256"], {})

    def test_residual_pair_requires_all_eight_adapter_hashes_equal(self):
        ae = _shared()
        pca = dict(ae)
        for key in CONTROLLED_FEATURE_INPUT_KEYS:
            pca[key] = _digest(f"pca:{key}")
        adapter = _adapter()
        receipt = compare_initialization_maps(
            ae_shared=ae,
            pca_shared=pca,
            ae_adapter=adapter,
            pca_adapter=dict(adapter),
            adapter="residual_mlp",
        )
        self.assertEqual(len(receipt["adapter_initial_variable_sha256"]), 8)
        altered = dict(adapter)
        altered[next(iter(EXPECTED_RESIDUAL_ADAPTER_KEYS))] = "f" * 64
        with self.assertRaisesRegex(
            TCMPCA64DeterministicPairGuardError,
            "adapter initial variable mapping differs",
        ):
            compare_initialization_maps(
                ae_shared=ae,
                pca_shared=pca,
                ae_adapter=adapter,
                pca_adapter=altered,
                adapter="residual_mlp",
            )

    def test_noncontrolled_difference_and_missing_key_fail_closed(self):
        ae = _shared()
        pca = dict(ae)
        for key in CONTROLLED_FEATURE_INPUT_KEYS:
            pca[key] = _digest(f"pca:{key}")
        pca["weight/layer_1"] = _digest("unexpected")
        with self.assertRaisesRegex(
            TCMPCA64DeterministicPairGuardError,
            "differences must be exactly",
        ):
            compare_initialization_maps(
                ae_shared=ae,
                pca_shared=pca,
                ae_adapter={},
                pca_adapter={},
                adapter="identity",
            )
        pca = dict(ae)
        pca.pop("weight/layer_1")
        with self.assertRaisesRegex(
            TCMPCA64DeterministicPairGuardError, "exactly 32 expected keys"
        ):
            compare_initialization_maps(
                ae_shared=ae,
                pca_shared=pca,
                ae_adapter={},
                pca_adapter={},
                adapter="identity",
            )

    def test_both_semantic_inputs_must_actually_change(self):
        ae = _shared()
        pca = dict(ae)
        pca["input/herb"] = _digest("pca:input/herb")
        with self.assertRaisesRegex(
            TCMPCA64DeterministicPairGuardError,
            "differences must be exactly",
        ):
            compare_initialization_maps(
                ae_shared=ae,
                pca_shared=pca,
                ae_adapter={},
                pca_adapter={},
                adapter="identity",
            )

    def test_semantic_bundle_array_hash_uses_float32_audit_encoding(self):
        with tempfile.TemporaryDirectory(prefix="astra_pca_v2_matrix_") as raw:
            path = Path(raw) / "herb.npy"
            matrix = np.arange(12, dtype=np.float64).reshape(3, 4)
            np.save(path, matrix)
            converted = np.ascontiguousarray(matrix, dtype=np.float32)
            digest = hashlib.sha256()
            digest.update(str(converted.dtype).encode("ascii"))
            digest.update(np.asarray(converted.shape, dtype=np.int64).tobytes())
            digest.update(converted.tobytes())
            self.assertEqual(_semantic_matrix_hash(path), digest.hexdigest())


class TCMPCA64SelectionPolicyTest(unittest.TestCase):
    def test_margin_is_exact_and_one_sided(self):
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

    def test_new_chain_declares_no_outer_entrypoint(self):
        module_path = Path(
            "src/astra/benchmark/tcm_astra_pca64_deterministic_selection.py"
        )
        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("evaluate_test_scores", source)
        self.assertNotIn("outer_diagnostic", source)
        self.assertNotIn("tcm_astra_pca64_control_evaluate", source)


if __name__ == "__main__":
    unittest.main()
