"""Portable path helpers for the legacy-compatible ASTRA code.

The original implementation interpreted most relative paths against the
process working directory.  These helpers make defaults independent of the
directory from which a command is launched while preserving environment
variable overrides used by the reproduction scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence
from typing import Optional, Union


PathLike = Union[str, os.PathLike]
EnvironmentNames = Union[str, Sequence[str]]


def first_environment_value(names: EnvironmentNames) -> Optional[str]:
    """Return the first non-empty value from one or more environment names."""

    candidates = (names,) if isinstance(names, str) else tuple(names)
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    return None


def source_root() -> Path:
    """Return the repository's ``src`` directory."""

    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Return the repository root."""

    return source_root().parent


def resolve_path(value: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    """Expand and resolve *value*, using *base_dir* for a relative path.

    ``base_dir`` defaults to the repository root.  The path need not already
    exist, which allows the same helper to be used for output directories.
    """

    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    path = Path(expanded)
    if not path.is_absolute():
        base = Path(base_dir) if base_dir is not None else project_root()
        path = base / path
    return path.resolve(strict=False)


def resolve_env_path(
    name: EnvironmentNames,
    default: PathLike,
    base_dir: Optional[PathLike] = None,
) -> Path:
    """Resolve an environment path, falling back to *default*."""

    return resolve_path(first_environment_value(name) or default, base_dir)


def resolve_config_path(
    config,
    key: str,
    default: PathLike,
    env_var: Optional[EnvironmentNames] = None,
) -> Path:
    """Resolve a path from environment, configuration, then a default.

    Relative values are interpreted against the directory containing the
    loaded model configuration.  ``config`` only needs ``contains`` and
    ``__getitem__`` methods, so the helper remains usable by legacy callers.
    """

    environment_value = first_environment_value(env_var) if env_var else None
    if environment_value:
        # Environment overrides are process-level settings. Resolve relative
        # values from the repository root, not from whichever nested config
        # happens to consume them.
        return resolve_path(environment_value, project_root())
    if config is not None and config.contains(key):
        value = config[key]
    else:
        value = default
    base_dir = getattr(config, "config_dir", project_root())
    resolved = resolve_path(value, base_dir)
    expanded_value = Path(
        os.path.expandvars(os.path.expanduser(os.fspath(value)))
    )
    if resolved.exists() or expanded_value.is_absolute():
        return resolved

    # Older experiment configs live below src/ but contain paths that were
    # historically interpreted after ``chdir(src)``.  Prefer the explicit
    # config-relative rule, then accept that source-root location when it is
    # the existing local asset.  This keeps archived runners usable while new
    # configs remain unambiguous.
    legacy_source_path = resolve_path(value, source_root())
    return legacy_source_path if legacy_source_path.exists() else resolved
