from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import herbhg
from herbhg.paths import ProjectPaths, project_root


class PublicPackageTests(unittest.TestCase):
    def test_version_is_available_without_tensorflow(self) -> None:
        self.assertEqual(herbhg.__version__, "0.1.0")
        self.assertNotIn("tensorflow", sys.modules)

        importlib.import_module("herbhg.models")
        importlib.import_module("herbhg.cli")
        self.assertNotIn("tensorflow", sys.modules)

    def test_source_checkout_paths(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(project_root(), root)
        paths = ProjectPaths.discover()
        self.assertEqual(paths.root, root)
        self.assertTrue(paths.backend("tcm-suite").is_dir())
        self.assertTrue(paths.backend("ethnobotany").is_dir())

    def test_backend_aliases_are_present(self) -> None:
        paths = ProjectPaths.discover()
        tcm_alias = (paths.backend("tcm-suite") / "HerbHG.py").read_text(
            encoding="utf-8"
        )
        ethno_alias = (paths.backend("ethnobotany") / "HerbHG.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("HGHDA as _TCMHerbHG", tcm_alias)
        self.assertIn("ASTRASemanticRefined as _EthnobotanyHerbHG", ethno_alias)

    def test_default_configs_use_the_public_model_name(self) -> None:
        paths = ProjectPaths.discover()
        tcm_config = (paths.backend("tcm-suite") / "ASTRA.conf").read_text(
            encoding="utf-8"
        )
        ethno_config = (
            paths.backend("ethnobotany") / "ethnobotany.conf"
        ).read_text(encoding="utf-8")
        self.assertIn("model.name=HerbHG", tcm_config)
        self.assertIn("model.name=HerbHG", ethno_config)
        self.assertIn("astra.semantic.adapter=residual_mlp", ethno_config)
        self.assertIn("astra.semantic.mlp_hidden_dim=32", ethno_config)


if __name__ == "__main__":
    unittest.main()
