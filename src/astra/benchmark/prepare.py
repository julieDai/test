"""Build an immutable fair-comparison protocol package from raw relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .protocol import (
    DEFAULT_FOLDS,
    DEFAULT_PAIR_ORDER,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    PAIR_ORDERS,
    ProtocolError,
    build_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ethnobotany", "tcm-suite"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--negative-file",
        type=Path,
        help=(
            "Stored Ethnobotany negatives or the mandatory explicit frozen "
            "TCM-suite negative file."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--pair-order",
        choices=PAIR_ORDERS,
        default=DEFAULT_PAIR_ORDER,
        help=(
            "Use independently seed-permuted class rows (fair default) or "
            "preserve first-occurrence source order for a matched legacy run."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = build_protocol(
            args.dataset,
            args.data_dir,
            negative_file=args.negative_file,
            seed=args.seed,
            folds=args.folds,
            validation_fraction=args.validation_fraction,
            pair_order=args.pair_order,
        )
        manifest_path = protocol.save(args.output_dir)
    except ProtocolError as error:
        parser.error(str(error))
    summary = {
        "manifest": str(manifest_path),
        "dataset": protocol.dataset,
        "seed": protocol.seed,
        "folds": protocol.n_folds,
        "counts": protocol.input_manifest()["counts"],
        "negative_sampling": protocol.negative_sampling,
        "pair_order": protocol.pair_order,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
