from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from astra.official_baselines import common
from herbhg.process import FORMAL_PYTHON_ENVIRONMENT


class FormalEnvironmentPolicyTests(unittest.TestCase):
    def test_launcher_policy_matches_baseline_policy(self) -> None:
        expected = {
            name: value
            for name, value in common.FORMAL_PYTHON_ENV_POLICY.items()
            if name != "removed"
        }
        self.assertEqual(FORMAL_PYTHON_ENVIRONMENT, expected)

    def test_incorrect_fixed_values_are_rejected(self) -> None:
        environment = dict(FORMAL_PYTHON_ENVIRONMENT)
        environment["PYTHONHASHSEED"] = "7"
        with (
            patch.object(common, "FORMAL_RUNTIME_MODULES", {}),
            patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(RuntimeError, "incorrect_fixed_values"),
        ):
            common.formal_runtime_module_origins(require_environment=True)

    def test_exact_fixed_values_are_accepted(self) -> None:
        with (
            patch.object(common, "FORMAL_RUNTIME_MODULES", {}),
            patch.dict(os.environ, FORMAL_PYTHON_ENVIRONMENT, clear=True),
        ):
            origins = common.formal_runtime_module_origins(require_environment=True)
        self.assertEqual(origins, {})


if __name__ == "__main__":
    unittest.main()
