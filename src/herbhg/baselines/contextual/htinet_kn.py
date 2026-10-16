"""HTInet k-nearest-neighbour reference entry point."""

from __future__ import annotations

from typing import Sequence

from .._launcher import run_adapter
from ..registry import CONTEXTUAL_BASELINES

SPEC = CONTEXTUAL_BASELINES["htinet-kn"]


def main(argv: Sequence[str] | None = None) -> int:
    return run_adapter(SPEC, argv, fixed_args=("--classifier", "knn"))


if __name__ == "__main__":
    raise SystemExit(main())
