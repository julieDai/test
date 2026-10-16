"""Path helpers for JSON job manifests.

Paths written inside a manifest are interpreted relative to the manifest
itself.  Command-line paths remain relative to the caller's working directory.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union


ManifestPath = Union[str, PathLike[str]]


def resolve_job_path(value: ManifestPath, jobs_file: ManifestPath) -> Path:
    """Return a manifest field as an absolute, normalized path."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(jobs_file).expanduser().resolve().parent / path
    return path.resolve()
