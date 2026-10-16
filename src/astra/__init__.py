"""Compatibility utilities for existing ASTRA experiment entry points.

New integrations should use :mod:`herbhg`; this namespace remains available
for protocol, evaluation, and baseline workflows created before the rename.
"""

from .paths import ProjectPaths, project_root, resolve_project_path

__all__ = ["ProjectPaths", "project_root", "resolve_project_path"]
__version__ = "0.1.0"
