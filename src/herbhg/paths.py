"""Stable repository paths for installed and source-checkout workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "herbhg").is_dir()


def project_root() -> Path:
    """Return the HerbHG checkout containing the active implementation.

    ``HERBHG_PROJECT_ROOT`` supports an editable installation whose datasets
    and experiment files live in a separate checkout.  Without an override,
    the source module location is used first and the working directory is only
    a final fallback.
    """

    override = os.environ.get("HERBHG_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_project_root(root):
            raise ValueError(
                "HERBHG_PROJECT_ROOT must contain pyproject.toml and src/herbhg: "
                f"{root}"
            )
        return root

    module_root = Path(__file__).resolve().parents[2]
    if _is_project_root(module_root):
        return module_root

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if _is_project_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate a HerbHG checkout. Set HERBHG_PROJECT_ROOT explicitly."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used by public commands."""

    root: Path
    source: Path
    backends: Path
    experiments: Path
    artifacts: Path
    outputs: Path
    results: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = project_root()
        source = root / "src"
        return cls(
            root=root,
            source=source,
            backends=source / "herbhg" / "models" / "backends",
            experiments=root / "experiments",
            artifacts=root / "artifacts",
            outputs=root / "outputs",
            results=root / "results",
        )

    def backend(self, dataset: str) -> Path:
        names = {
            "tcm-suite": "tcm_suite",
            "ethnobotany": "ethnobotany",
        }
        try:
            name = names[dataset]
        except KeyError as error:
            raise ValueError(f"Unsupported dataset: {dataset}") from error
        path = self.backends / name
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
