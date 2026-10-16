"""Repository paths that do not depend on the caller's working directory."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union


PathLike = Union[str, os.PathLike[str]]


def _looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src").is_dir()


def project_root() -> Path:
    """Return the HerbHG checkout root.

    ``HERBHG_PROJECT_ROOT`` is the public override. ``ASTRA_PROJECT_ROOT`` is
    accepted for compatibility with existing experiment receipts.
    """

    override = os.environ.get("HERBHG_PROJECT_ROOT") or os.environ.get(
        "ASTRA_PROJECT_ROOT"
    )
    if override:
        root = Path(override).expanduser().resolve()
        if not _looks_like_project_root(root):
            raise ValueError(
                "project-root override must contain pyproject.toml and src/: "
                f"{root}"
            )
        return root

    module_root = Path(__file__).resolve().parents[2]
    if _looks_like_project_root(module_root):
        return module_root

    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if _looks_like_project_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate the HerbHG checkout. Set HERBHG_PROJECT_ROOT explicitly."
    )


def resolve_project_path(
    value: PathLike,
    *,
    base: PathLike | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve ``value`` against an explicit base or the project root."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        anchor = Path(base).expanduser().resolve() if base else project_root()
        path = anchor / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical code and local-artifact locations for one checkout."""

    root: Path
    source: Path
    artifacts: Path
    outputs: Path
    results: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = project_root()
        return cls(
            root=root,
            source=root / "src",
            artifacts=root / "artifacts",
            outputs=root / "outputs",
            results=root / "results",
        )

    def tcm_config(self) -> Path:
        return (
            self.source
            / "herbhg"
            / "models"
            / "backends"
            / "tcm_suite"
            / "ASTRA.conf"
        )

    def ethnobotany_config(self) -> Path:
        return (
            self.source
            / "herbhg"
            / "models"
            / "backends"
            / "ethnobotany"
            / "ethnobotany.conf"
        )
