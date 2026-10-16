"""Subprocess helpers for isolated model and experiment execution."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from .paths import ProjectPaths


FORMAL_PYTHON_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "42",
}


def runtime_environment(
    paths: ProjectPaths,
    *,
    backend: Path | None = None,
    overrides: Mapping[str, str] | None = None,
    isolated_python: bool = False,
) -> dict[str, str]:
    """Build the environment for a backend or a clean formal-protocol child."""

    environment = os.environ.copy()
    environment["HERBHG_PROJECT_ROOT"] = str(paths.root)
    # The ``astra`` namespace is retained as an implementation compatibility
    # layer.  Point it at this checkout when compatibility modules resolve data.
    environment["ASTRA_PROJECT_ROOT"] = str(paths.root)

    if overrides:
        environment.update({key: str(value) for key, value in overrides.items()})

    if isolated_python:
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT"):
            environment.pop(name, None)
        environment.update(FORMAL_PYTHON_ENVIRONMENT)
    else:
        search_path = []
        if backend is not None:
            search_path.append(str(backend))
        search_path.append(str(paths.source))
        inherited = environment.get("PYTHONPATH")
        if inherited:
            search_path.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(search_path)

    compatibility_pairs = (
        ("HERBHG_SEED", "ASTRA_SEED"),
        ("HERBHG_DEVICE", "ASTRA_DEVICE"),
        ("HERBHG_OUTPUT_DIR", "ASTRA_OUTPUT_DIR"),
        ("HERBHG_LOG_DIR", "ASTRA_LOG_DIR"),
    )
    for public_name, compatibility_name in compatibility_pairs:
        value = environment.get(public_name)
        if value:
            environment[compatibility_name] = value
    return environment


def run_python(
    arguments: Sequence[str | os.PathLike[str]],
    *,
    paths: ProjectPaths,
    backend: Path | None = None,
    cwd: Path | None = None,
    overrides: Mapping[str, str] | None = None,
    isolated_python: bool = False,
) -> int:
    """Run Python in an isolated child and return its process status."""

    command = [sys.executable, *(os.fspath(value) for value in arguments)]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=runtime_environment(
            paths,
            backend=backend,
            overrides=overrides,
            isolated_python=isolated_python,
        ),
        check=False,
    )
    return int(completed.returncode)
