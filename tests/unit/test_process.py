from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from herbhg.paths import ProjectPaths
from herbhg.process import (
    FORMAL_PYTHON_ENVIRONMENT,
    run_python,
    runtime_environment,
)


class ProcessEnvironmentTests(unittest.TestCase):
    def test_backend_precedes_public_source(self) -> None:
        paths = ProjectPaths.discover()
        backend = paths.backend("ethnobotany")
        with patch.dict(os.environ, {"PYTHONPATH": "/external/source"}, clear=False):
            environment = runtime_environment(paths, backend=backend)
        entries = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[:3], [str(backend), str(paths.source), "/external/source"])
        self.assertEqual(environment["HERBHG_PROJECT_ROOT"], str(paths.root))
        self.assertEqual(environment["ASTRA_PROJECT_ROOT"], str(paths.root))

    def test_public_runtime_values_are_mirrored_for_compatibility(self) -> None:
        paths = ProjectPaths.discover()
        clean = {
            key: value
            for key, value in os.environ.items()
            if key not in {"ASTRA_SEED", "HERBHG_SEED"}
        }
        clean["HERBHG_SEED"] = "17"
        with patch.dict(os.environ, clean, clear=True):
            environment = runtime_environment(paths)
        self.assertEqual(environment["ASTRA_SEED"], "17")

    def test_child_inherits_caller_working_directory_by_default(self) -> None:
        paths = ProjectPaths.discover()
        with patch("herbhg.process.subprocess.run") as run:
            run.return_value.returncode = 0
            status = run_python(["-c", "pass"], paths=paths)
        self.assertEqual(status, 0)
        self.assertIsNone(run.call_args.kwargs["cwd"])

    def test_explicit_child_working_directory_is_honored(self) -> None:
        paths = ProjectPaths.discover()
        requested = Path("/tmp").resolve()
        with patch("herbhg.process.subprocess.run") as run:
            run.return_value.returncode = 0
            run_python(["-c", "pass"], paths=paths, cwd=requested)
        self.assertEqual(run.call_args.kwargs["cwd"], str(requested))

    def test_formal_child_removes_python_environment_overrides(self) -> None:
        paths = ProjectPaths.discover()
        inherited = {
            "PYTHONHOME": "/tmp/python-home",
            "PYTHONPATH": "/tmp/python-path",
            "PYTHONSTARTUP": "/tmp/startup.py",
            "PYTHONINSPECT": "1",
        }
        with patch.dict(os.environ, inherited, clear=False):
            environment = runtime_environment(
                paths,
                isolated_python=True,
                overrides={"PYTHONPATH": "/tmp/reinjected", "HERBHG_SEED": "23"},
            )
        for name in inherited:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["HERBHG_PROJECT_ROOT"], str(paths.root))
        self.assertEqual(environment["ASTRA_PROJECT_ROOT"], str(paths.root))
        self.assertEqual(environment["ASTRA_SEED"], "23")
        for name, expected in FORMAL_PYTHON_ENVIRONMENT.items():
            self.assertEqual(environment[name], expected)

    def test_formal_policy_reaches_a_real_child_process(self) -> None:
        paths = ProjectPaths.discover()
        names = (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
            *FORMAL_PYTHON_ENVIRONMENT,
        )
        program = (
            "import json, os, pathlib, sys; "
            "pathlib.Path(sys.argv[1]).write_text("
            "json.dumps({name: os.environ.get(name) for name in sys.argv[2:]}), "
            "encoding='utf-8')"
        )
        inherited = {name: "unexpected" for name in names}
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "environment.json"
            with patch.dict(os.environ, inherited, clear=False):
                status = run_python(
                    ["-c", program, receipt, *names],
                    paths=paths,
                    isolated_python=True,
                )
            observed = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT"):
            self.assertIsNone(observed[name])
        for name, expected in FORMAL_PYTHON_ENVIRONMENT.items():
            self.assertEqual(observed[name], expected)


if __name__ == "__main__":
    unittest.main()
