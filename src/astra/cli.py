"""Compatibility command wrapper for the historical ``astra-train`` CLI."""

from __future__ import annotations

import sys
from typing import Sequence

from herbhg.cli import main as herbhg_main


USAGE = """usage: astra-train {tcm-suite|ethnobotany} [training options]

The remaining options are forwarded to the selected dataset entry point.
Examples:
  astra-train tcm-suite --help
  astra-train ethnobotany --help
"""


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    dataset = arguments[0]
    if dataset not in {"tcm-suite", "tcm", "ethnobotany"}:
        print(f"Unknown dataset: {dataset}\n\n{USAGE}", file=sys.stderr)
        return 2
    return herbhg_main(["train", *arguments])
