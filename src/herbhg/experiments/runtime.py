"""Resolve repository-local experiment implementations without CWD assumptions."""

from __future__ import annotations

import sys
from pathlib import Path


def source_root() -> Path:
    """Return the repository's ``src`` directory."""

    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the HerbHG repository root."""

    return source_root().parent


def tcm_backend_root(root: Path | None = None) -> Path:
    """Return the compatibility backend used by TCM-suite experiments."""

    resolved_source = (root or source_root()).resolve()
    backend = resolved_source / "herbhg" / "models" / "backends" / "tcm_suite"
    required = ("ASTRA.py", "HGHDA.py", "ASTRA.conf", "base", "util")
    missing = [name for name in required if not (backend / name).exists()]
    if missing:
        raise RuntimeError(
            f"incomplete TCM-suite backend at {backend}: missing {', '.join(missing)}"
        )
    return backend


def activate_tcm_backend(root: Path | None = None) -> Path:
    """Place the isolated TCM backend before repository packages on ``sys.path``.

    The retained implementation uses historical flat imports such as ``base``
    and ``util``.  Keeping those imports inside a dataset-specific backend
    avoids collisions with the Ethnobotany implementation while preserving the
    published model equations.
    """

    resolved_source = (root or source_root()).resolve()
    backend = tcm_backend_root(resolved_source)
    other_backend = (
        resolved_source / "herbhg" / "models" / "backends" / "ethnobotany"
    )
    for entry in (str(backend), str(other_backend), str(resolved_source)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(resolved_source))
    sys.path.insert(0, str(backend))
    return backend
