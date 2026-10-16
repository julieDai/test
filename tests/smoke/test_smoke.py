from __future__ import annotations

import unittest

from herbhg.smoke import run_smoke


class SmokeTests(unittest.TestCase):
    def test_synthetic_relation_contract(self) -> None:
        result = run_smoke()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["positive_pairs"], 4)
        self.assertEqual(result["negative_pairs"], 4)
        self.assertEqual(result["folds"], 2)
        self.assertEqual(result["backends"], ["tcm-suite", "ethnobotany"])


if __name__ == "__main__":
    unittest.main()
