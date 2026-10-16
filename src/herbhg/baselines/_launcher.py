"""Shared subprocess launcher used by lightweight baseline entry points."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

from .registry import BaselineSpec


def adapter_command(
    spec: BaselineSpec,
    argv: Sequence[str],
    *,
    fixed_args: Sequence[str] = (),
) -> list[str]:
    """Build the isolated command for one registered adapter."""

    return [sys.executable, "-m", spec.runner_module, *argv, *fixed_args]


def run_adapter(
    spec: BaselineSpec,
    argv: Sequence[str] | None = None,
    *,
    fixed_args: Sequence[str] = (),
) -> int:
    """Execute one adapter without importing its numerical framework."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run(
        adapter_command(spec, arguments, fixed_args=fixed_args),
        check=False,
    )
    return int(completed.returncode)
