#!/usr/bin/env python3
"""Build shuffled and distribution-matched random 64-D text controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, values.astype(np.float32))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-herb", type=Path, required=True)
    parser.add_argument("--clean-disease", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    herb = np.load(args.clean_herb).astype(np.float32)
    disease = np.load(args.clean_disease).astype(np.float32)
    rng = np.random.default_rng(args.seed)

    conditions = {
        "clean_shuffled": (
            herb[rng.permutation(len(herb))],
            disease[rng.permutation(len(disease))],
        ),
        "random_matched": (
            rng.normal(herb.mean(0), herb.std(0) + 1e-8, herb.shape),
            rng.normal(disease.mean(0), disease.std(0) + 1e-8, disease.shape),
        ),
    }
    manifest = {"seed": args.seed, "source": {}, "outputs": {}}
    for source in (args.clean_herb, args.clean_disease):
        manifest["source"][str(source.resolve())] = sha256(source)
    for condition, (herb_values, disease_values) in conditions.items():
        herb_path = args.output_root / condition / "herb_semantic_embeddings_64d.npy"
        disease_path = args.output_root / condition / "disease_semantic_embeddings_64d.npy"
        save(herb_path, herb_values)
        save(disease_path, disease_values)
        manifest["outputs"][condition] = {
            "herb": {"path": str(herb_path.resolve()), "shape": list(herb_values.shape), "sha256": sha256(herb_path)},
            "disease": {"path": str(disease_path.resolve()), "shape": list(disease_values.shape), "sha256": sha256(disease_path)},
        }
    manifest_path = args.output_root / "derived_controls_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
