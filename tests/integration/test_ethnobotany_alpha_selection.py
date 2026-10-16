import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.benchmark import ethnobotany_alpha_selection as alpha_selection
from astra.benchmark.ethnobotany_alpha_selection import (
    _alpha_slug,
    _choose_candidate,
    _exclusive_orchestrator_lock,
    _parse_alpha,
    _run_parallel,
    _selected_complete,
    _validate_locked_plan,
    validate_selected_binding,
)


_LOCKED_PLAN = json.loads(alpha_selection.PLAN_PATH.read_text(encoding="utf-8"))
_EXTERNAL_FIXTURE_AVAILABLE = (
    alpha_selection.REPOSITORY_ROOT / _LOCKED_PLAN["protocol_manifest"]
).is_file() and (
    alpha_selection.REPOSITORY_ROOT
    / _LOCKED_PLAN["feature_bundle"]
    / "bundle_manifest.json"
).is_file()


class EthnobotanyAlphaSelectionTest(unittest.TestCase):
    def test_alpha_parser_and_slug_are_stable(self):
        self.assertEqual(_parse_alpha("0.25"), 0.25)
        self.assertEqual(_alpha_slug(0.25), "alpha_0p25")
        self.assertEqual(_alpha_slug(1.0), "alpha_1")
        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.assertRaises(Exception):
                _parse_alpha(value)

    def test_validation_auprc_selects_alpha_and_ties_choose_lower(self):
        selected = _choose_candidate(
            [
                {"alpha": 0.75, "validation_auprc": 0.9},
                {"alpha": 0.25, "validation_auprc": 0.9},
                {"alpha": 1.0, "validation_auprc": 0.8},
            ]
        )
        self.assertEqual(selected["alpha"], 0.25)

    def test_selected_run_requires_both_validation_and_test_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "results").mkdir()
            for path in (
                run_dir / "protocol.json",
                run_dir / "results" / "metrics.json",
                run_dir / "results" / "validation_predictions.npz",
            ):
                path.touch()
            self.assertFalse(_selected_complete(run_dir))
            (run_dir / "results" / "test_predictions.npz").touch()
            self.assertTrue(_selected_complete(run_dir))

    @unittest.skipUnless(
        _EXTERNAL_FIXTURE_AVAILABLE,
        "requires external frozen protocol and feature-bundle fixtures",
    )
    def test_locked_plan_is_valid_and_bound_to_existing_inputs(self):
        locked = _validate_locked_plan()
        self.assertEqual(locked["schema"], "astra-ethnobotany-alpha-plan/v1")
        self.assertEqual(locked["seed"], 42)
        self.assertEqual(locked["device_policy"], "cuda")
        self.assertEqual(locked["max_parallel"], 2)
        self.assertEqual(locked["cuda_devices"], [0, 1])
        self.assertEqual(locked["sha256"], alpha_selection._sha256(alpha_selection.PLAN_PATH))

        protocol = alpha_selection.REPOSITORY_ROOT / locked["protocol_manifest"]
        bundle = alpha_selection.REPOSITORY_ROOT / locked["feature_bundle"]
        self.assertTrue(protocol.is_file())
        self.assertTrue((bundle / "bundle_manifest.json").is_file())
        self.assertEqual(
            alpha_selection._sha256(protocol),
            locked["protocol_manifest_sha256"],
        )
        self.assertEqual(
            alpha_selection._sha256(bundle / "bundle_manifest.json"),
            locked["feature_bundle_manifest_sha256"],
        )

    @unittest.skipUnless(
        _EXTERNAL_FIXTURE_AVAILABLE,
        "requires external frozen protocol and feature-bundle fixtures",
    )
    def test_main_rejects_plan_mismatch_before_starting_candidates(self):
        locked = _validate_locked_plan()
        protocol_path = (
            alpha_selection.REPOSITORY_ROOT / locked["protocol_manifest"]
        )
        protocol = alpha_selection.load_protocol_manifest(protocol_path)
        bundle = alpha_selection.REPOSITORY_ROOT / locked["feature_bundle"]
        expected_output = alpha_selection.REPOSITORY_ROOT / locked["output_root"]

        def argv(*, output: Path, device: str = "cuda", workers: int = 2):
            return [
                "--protocol-manifest",
                str(protocol_path),
                "--data-dir",
                str(protocol.data_dir),
                "--embedding-dir",
                str(bundle),
                "--output-root",
                str(output),
                "--device",
                device,
                "--max-parallel",
                str(workers),
            ]

        cases = (
            (
                argv(output=expected_output.parent / "wrong-alpha-output"),
                "output root differs",
            ),
            (argv(output=expected_output, device="cpu"), "device policy differs"),
            (argv(output=expected_output, workers=1), "parallelism differs"),
        )
        for arguments, message in cases:
            with self.subTest(message=message), patch.object(
                alpha_selection, "_run_candidates"
            ) as run_candidates:
                with self.assertRaisesRegex(RuntimeError, message):
                    alpha_selection.main(arguments)
                run_candidates.assert_not_called()

    def test_orchestrator_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "formal-alpha-run"
            with _exclusive_orchestrator_lock(output_root):
                with self.assertRaisesRegex(RuntimeError, "another alpha-selection"):
                    with _exclusive_orchestrator_lock(output_root):
                        self.fail("a second selector acquired the same run-root lock")

    def test_parallel_scheduler_assigns_one_queue_per_cuda_device(self):
        tasks = [
            (
                f"task-{index}",
                ["noop", str(index)],
                Path(f"log-{index}"),
                index % 2,
            )
            for index in range(6)
        ]
        observed = []

        def record(command, *, log_path, cuda_device=None):
            observed.append((command[-1], log_path.name, cuda_device))

        with patch.object(alpha_selection, "_run_logged", side_effect=record):
            _run_parallel(tasks, 2, [0, 1])
        self.assertEqual(
            {(command_index, device) for command_index, _, device in observed},
            {(str(index), index % 2) for index in range(6)},
        )

    def test_selected_binding_rejects_unfrozen_decision_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            alpha = 0.25
            fold = 1
            candidate_dir = (
                output_root
                / "selection_candidates"
                / "alpha_0p25"
                / f"fold_{fold}"
            )
            selected_dir = output_root / "tasks" / "ASTRA" / f"fold_{fold}"
            (candidate_dir).mkdir(parents=True)
            (selected_dir / "results").mkdir(parents=True)
            (candidate_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "device_policy": "cuda",
                            "cuda_visible_devices": "0",
                            "ld_library_path": str(
                                (Path(alpha_selection.sys.prefix) / "lib").resolve()
                            ),
                            "tensorflow_cuda_available": True,
                            "tensorflow_gpu_device_name": "/device:GPU:0",
                        }
                    }
                ),
                encoding="utf-8",
            )

            selection_path = output_root / "alpha_selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "folds": {
                            str(fold): {
                                "selected_alpha": alpha,
                                "selected_candidate_run_dir": str(candidate_dir.resolve()),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = {
                "Dataset": "ethnobotany",
                "Fold": fold,
                "Seed": 42,
                "Execution_Mode": "formal-candidate",
                "Formal_Eligible": True,
                "Smoke": False,
                "Diagnostic": False,
                "Validation_Only": False,
                "Text_Fusion_Alpha": alpha,
                "Alpha_Selected_On_Validation": True,
                "Decision_Frozen_Before_Test": False,
                "Test_Models_Evaluated": 1,
                "Unselected_Test_Models_Evaluated": 0,
            }
            provenance = {
                "dataset": "ethnobotany",
                "fold": fold,
                "seed": 42,
                "execution_mode": "formal-candidate",
                "formal_eligible": True,
                "validation_only": False,
                "decision_frozen_before_test": True,
                "runtime": {
                    "device_policy": "cuda",
                    "cuda_visible_devices": "0",
                    "ld_library_path": str(
                        (Path(alpha_selection.sys.prefix) / "lib").resolve()
                    ),
                    "tensorflow_cuda_available": True,
                    "tensorflow_gpu_device_name": "/device:GPU:0",
                },
            }
            (selected_dir / "results" / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (selected_dir / "protocol.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            for name in ("validation_predictions.npz", "test_predictions.npz"):
                (selected_dir / "results" / name).touch()

            with self.assertRaisesRegex(RuntimeError, "decision was not frozen"):
                validate_selected_binding(
                    run_dir=selected_dir,
                    fold=fold,
                    selected_alpha=alpha,
                    selection_path=selection_path,
                )


if __name__ == "__main__":
    unittest.main()
