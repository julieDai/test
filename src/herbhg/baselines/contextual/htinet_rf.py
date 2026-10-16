"""HTInet random-forest reference entry point."""

from __future__ import annotations

from typing import Sequence

from .._launcher import run_adapter
from ..registry import CONTEXTUAL_BASELINES

SPEC = CONTEXTUAL_BASELINES["htinet-rf"]


def main(argv: Sequence[str] | None = None) -> int:
    return run_adapter(SPEC, argv, fixed_args=("--classifier", "rf"))


if __name__ == "__main__":
    raise SystemExit(main())
