"""HGHDA adapter entry point."""

from __future__ import annotations

from typing import Sequence

from .._launcher import run_adapter
from ..registry import FORMAL_BASELINES

SPEC = FORMAL_BASELINES["hghda"]


def main(argv: Sequence[str] | None = None) -> int:
    return run_adapter(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
