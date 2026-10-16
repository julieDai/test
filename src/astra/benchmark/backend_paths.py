"""Canonical locations for the isolated numerical backends."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
TCM_BACKEND_ROOT = SOURCE_ROOT / "herbhg" / "models" / "backends" / "tcm_suite"
ETHNOBOTANY_BACKEND_ROOT = (
    SOURCE_ROOT / "herbhg" / "models" / "backends" / "ethnobotany"
)


def backend_root(dataset: str) -> Path:
    """Return the backend root for a normalized benchmark dataset name."""

    if dataset == "tcm-suite":
        return TCM_BACKEND_ROOT
    if dataset == "ethnobotany":
        return ETHNOBOTANY_BACKEND_ROOT
    raise ValueError(f"unsupported dataset: {dataset}")


def activate_backend(dataset: str) -> Path:
    """Put one backend first on ``sys.path`` for its retained flat imports."""

    selected = backend_root(dataset).resolve()
    required = ("ASTRA.py", "HGHDA.py", "base", "util")
    missing = [name for name in required if not (selected / name).exists()]
    if missing:
        raise RuntimeError(
            f"incomplete {dataset} backend at {selected}: missing {', '.join(missing)}"
        )
    for path in (TCM_BACKEND_ROOT, ETHNOBOTANY_BACKEND_ROOT, SOURCE_ROOT):
        entry = str(path.resolve())
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(SOURCE_ROOT.resolve()))
    sys.path.insert(0, str(selected))
    return selected


__all__ = [
    "ETHNOBOTANY_BACKEND_ROOT",
    "SOURCE_ROOT",
    "TCM_BACKEND_ROOT",
    "activate_backend",
    "backend_root",
]
