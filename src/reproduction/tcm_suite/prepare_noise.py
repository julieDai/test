#!/usr/bin/env python3
"""Prepare deterministic, edge-count-preserving A_HC/A_PD perturbations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_edges(path):
    rows = [line.rstrip("\n").split("\t") for line in path.open() if line.strip()]
    return rows, sorted({x[0] for x in rows}), sorted({x[1] for x in rows})


def rewire(rows, left, right, ratio, seed):
    rng = np.random.RandomState(seed)
    original = {(x[0], x[1]) for x in rows}
    n_replace = int(round(len(original) * ratio))
    removed_indices = rng.choice(len(rows), n_replace, replace=False)
    removed = {tuple(rows[i][:2]) for i in removed_indices}
    kept = original - removed
    added = set()
    while len(added) < n_replace:
        batch = max(1024, (n_replace - len(added)) * 2)
        li = rng.randint(0, len(left), batch)
        ri = rng.randint(0, len(right), batch)
        for i, j in zip(li, ri):
            edge = (left[i], right[j])
            if edge not in original and edge not in added:
                added.add(edge)
                if len(added) == n_replace:
                    break
    return sorted(kept | added), n_replace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ratios", default="10,20,30,40,50,60,70,80,90")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for percent in (int(x) for x in args.ratios.split(",")):
        ratio = percent / 100.0
        out = args.run_dir / f"noise_{percent:02d}" / "associations"
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"percent": percent, "ratio": ratio, "seed": args.seed,
                    "method": "remove ratio of unique edges and add equal number of sampled non-edges"}
        for offset, name in enumerate(("H_C", "P_D")):
            source = args.dataset_dir / f"{name}.txt"
            rows, left, right = read_edges(source)
            edges, changed = rewire(rows, left, right, ratio, args.seed + offset)
            target = out / f"{name}.txt"
            target.write_text("".join(f"{a}\t{b}\t1\n" for a, b in edges))
            manifest[name] = {"source": str(source.resolve()), "source_sha256": digest(source),
                              "output": str(target.resolve()), "output_sha256": digest(target),
                              "original_edges": len(set(tuple(x[:2]) for x in rows)),
                              "output_edges": len(edges), "replaced_edges": changed}
        (out.parent / "perturbation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(out.parent, flush=True)


if __name__ == "__main__":
    main()
