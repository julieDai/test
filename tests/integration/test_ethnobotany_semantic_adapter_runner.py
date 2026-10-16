from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark.ethnobotany_refined_runner import (
    CONFIG_SCHEMA,
    RefinedCandidateRunnerError,
)
from astra.benchmark.ethnobotany_semantic_adapter_runner import (
    run_validation_candidate,
    validate_semantic_adapter_candidate,
)


def _candidate_payload(adapter: str = "residual_linear") -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "config": {
            "id": f"s5_{adapter}",
            "stage_id": "S5_semantic_adapter_architecture",
            "candidate_value": adapter,
            "chain_index": 9,
            "incumbent_id": "s4a_linear64_root",
            "base_config_id": "s1_c0_legacy_epoch_grid",
        },
        "training": {
            "epochs": 200,
            "checkpoint_epochs": [30, 60, 100, 200],
            "batch_size": 1000,
            "learning_rate": 0.0001,
            "embedding_dimension": 64,
            "reg_u": 0.001,
            "text_fusion_alpha": 1.0,
        },
        "model_options": {
            "astra.semantic.adapter": adapter,
            "astra.semantic.input_dim": 64,
            "astra.semantic.freeze_input": False,
            "astra.semantic.mlp_hidden_dim": 32,
        },
    }


class SemanticAdapterCandidateContractTest(unittest.TestCase):
    def test_three_predeclared_adapters_are_accepted(self):
        for adapter in (
            "residual_linear",
            "gated_residual",
            "residual_mlp",
        ):
            with self.subTest(adapter=adapter):
                candidate = validate_semantic_adapter_candidate(
                    _candidate_payload(adapter)
                )
                self.assertEqual(
                    candidate["model_options"]["astra.semantic.adapter"],
                    adapter,
                )

    def test_old_or_unplanned_adapter_is_rejected(self):
        for adapter in ("identity", "linear", "unknown"):
            with self.subTest(adapter=adapter), self.assertRaisesRegex(
                RefinedCandidateRunnerError, "requires one of"
            ):
                validate_semantic_adapter_candidate(_candidate_payload(adapter))

    def test_dimension_freeze_and_hidden_controls_are_fail_closed(self):
        cases = (
            ("astra.semantic.input_dim", 128, "input_dim=64"),
            ("astra.semantic.freeze_input", True, "freeze_input=false"),
            ("astra.semantic.mlp_hidden_dim", 64, "hidden_dim=32"),
        )
        for key, value, message in cases:
            payload = _candidate_payload()
            payload["model_options"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                RefinedCandidateRunnerError, message
            ):
                validate_semantic_adapter_candidate(payload)

    def test_wrapper_validates_before_delegating_and_injects_new_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "candidate.json"
            config_path.write_text(
                json.dumps(_candidate_payload()), encoding="utf-8"
            )
            with patch(
                "astra.benchmark.ethnobotany_semantic_adapter_runner."
                "_run_refined_validation_candidate",
                return_value={"validation_only": True},
            ) as delegated:
                observed = run_validation_candidate(
                    protocol_manifest=Path("protocol.json"),
                    data_dir=Path("data"),
                    embedding_dir=Path("embeddings"),
                    semantic_reduction_manifest=Path("semantics.json"),
                    config_json=config_path,
                    fold=1,
                    output_root=Path("output"),
                    device="cpu",
                )
            self.assertEqual(observed, {"validation_only": True})
            kwargs = delegated.call_args.kwargs
            self.assertEqual(kwargs["fold"], 1)
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(
                kwargs["trainer"].__name__,
                "train_semantic_adapter_candidate",
            )


if __name__ == "__main__":
    unittest.main()
