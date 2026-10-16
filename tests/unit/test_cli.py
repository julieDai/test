from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from herbhg import cli


class CommandLineTests(unittest.TestCase):
    def test_help_does_not_launch_a_child(self) -> None:
        output = io.StringIO()
        with patch("herbhg.cli.run_python") as run, redirect_stdout(output):
            status = cli.main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("train", output.getvalue())
        run.assert_not_called()

    def test_version_does_not_launch_a_child(self) -> None:
        output = io.StringIO()
        with patch("herbhg.cli.run_python") as run, redirect_stdout(output):
            status = cli.main(["--version"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), "herbhg 0.1.0")
        run.assert_not_called()

    def test_smoke_does_not_require_a_source_checkout(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                cli.ProjectPaths,
                "discover",
                side_effect=AssertionError("smoke must not discover a checkout"),
            ),
            redirect_stdout(output),
        ):
            status = cli.main(["smoke"])
        self.assertEqual(status, 0)
        self.assertIn('"status": "ok"', output.getvalue())

    def test_tcm_alias_dispatches_to_isolated_backend(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["train", "tcm", "--dry-run"])
        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        command = positional[0]
        self.assertEqual(Path(command[0]).name, "main.py")
        self.assertEqual(command[1:], ["--dry-run"])
        self.assertEqual(keywords["backend"].name, "tcm_suite")
        self.assertEqual(keywords["overrides"]["HERBHG_DATASET"], "tcm-suite")

    def test_ethnobotany_dispatches_to_isolated_backend(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["train", "ethnobotany", "--dry-run"])
        self.assertEqual(status, 0)
        positional, keywords = run.call_args
        self.assertEqual(Path(positional[0][0]).name, "run_training.py")
        self.assertEqual(keywords["backend"].name, "ethnobotany")

    def test_benchmark_command_delegates_to_compatibility_module(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["benchmark", "prepare", "--help"])
        self.assertEqual(status, 0)
        self.assertEqual(
            run.call_args.args[0], ["-m", "astra.benchmark.prepare", "--help"]
        )
        self.assertFalse(run.call_args.kwargs["isolated_python"])

    def test_formal_comparison_uses_an_isolated_python_environment(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["benchmark", "compare", "--help"])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.kwargs["isolated_python"])

    def test_baseline_preflight_uses_an_isolated_python_environment(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["baseline", "preflight", "--help"])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.kwargs["isolated_python"])

    def test_baseline_run_uses_an_isolated_python_environment(self) -> None:
        with patch("herbhg.cli.run_python", return_value=0) as run:
            status = cli.main(["baseline", "run", "--help"])
        self.assertEqual(status, 0)
        self.assertTrue(run.call_args.kwargs["isolated_python"])

    def test_unknown_dataset_returns_usage_error(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error):
            status = cli.main(["train", "unknown"])
        self.assertEqual(status, 2)
        self.assertIn("Unknown dataset", error.getvalue())


if __name__ == "__main__":
    unittest.main()
