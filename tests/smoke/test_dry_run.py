from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not inherited else os.pathsep.join((source, inherited))
    )
    return environment


class BackendDryRunTests(unittest.TestCase):
    def _run(self, dataset: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "herbhg", "train", dataset, "--dry-run"],
            cwd=ROOT,
            env=_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_tcm_suite_dry_run(self) -> None:
        completed = self._run("tcm-suite")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Configuration:", completed.stdout)
        self.assertNotIn("tensorflow", completed.stderr.lower())

    def test_ethnobotany_dry_run(self) -> None:
        completed = self._run("ethnobotany")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Resolved Ethnobotany run paths:", completed.stdout)
        self.assertNotIn("tensorflow", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
