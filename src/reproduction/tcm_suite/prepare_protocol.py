#!/usr/bin/env python3
"""Freeze the retained TCM-suite negative sample and modulo-5 split protocol."""

import argparse
import hashlib
import json
import random
from pathlib import Path


SAMPLE_SIZE = 2_354_225
FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive = args.dataset_dir / "H_D.txt"
    negative_pool = args.dataset_dir / "negativehdlist.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_negative = args.output_dir / f"negative_seed{args.seed}.txt"
    manifest_path = args.output_dir / "protocol.json"

    positive_rows = line_count(positive)
    pool_rows = line_count(negative_pool)
    if positive_rows != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} positives, found {positive_rows}")

    if not frozen_negative.exists() or line_count(frozen_negative) != SAMPLE_SIZE:
        # Match the retained FileIO sampling rule: readlines(), followed by
        # random.sample(lines, positive_count), preserving sampled order.
        with negative_pool.open("r") as stream:
            rows = stream.readlines()
        sampled = random.Random(args.seed).sample(rows, SAMPLE_SIZE)
        with frozen_negative.open("w") as stream:
            stream.writelines(sampled)

    total = positive_rows + SAMPLE_SIZE
    fold_counts = []
    for fold in range(FOLDS):
        indices = range(fold, total, FOLDS)
        test_rows = len(indices)
        fold_counts.append({
            "fold": fold + 1,
            "train_rows": total - test_rows,
            "test_rows": test_rows,
            "test_positive_rows": len(range(fold, positive_rows, FOLDS)),
            "test_negative_rows": len(range(max(fold, positive_rows + ((fold - positive_rows) % FOLDS)), total, FOLDS)),
        })

    manifest = {
        "name": f"tcm_suite_retained_modulo5_seed{args.seed}_v1",
        "seed": args.seed,
        "sampling": "Python random.Random(seed).sample; sampled order retained",
        "split": "positive rows followed by sampled negative rows; zero-based row_index % 5",
        "positive_file": str(positive.resolve()),
        "positive_rows": positive_rows,
        "positive_sha256": sha256(positive),
        "negative_pool_file": str(negative_pool.resolve()),
        "negative_pool_rows": pool_rows,
        "negative_pool_sha256": sha256(negative_pool),
        "frozen_negative_file": str(frozen_negative.resolve()),
        "frozen_negative_rows": SAMPLE_SIZE,
        "frozen_negative_sha256": sha256(frozen_negative),
        "total_rows": total,
        "folds": fold_counts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
