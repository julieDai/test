"""Resolve the four Ethnobotany feature files used by legacy HerbHG.

The historical loader accepted only one directory and reconstructed every
filename inside it.  The fair benchmark runner may materialise just one
permuted matrix into a canonical-order run directory, so each feature path
must remain independently configurable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np


_FEATURE_SPECS = {
    "compound": ("compound.embedding", "compound_embedding_kpgt.txt"),
    "protein": ("protein.embedding", "protein_embedding_esm2.txt"),
    "herb": ("herb.text.embedding", "herb_semantic_embeddings_64d.npy"),
    "disease": (
        "disease.text.embedding",
        "disease_semantic_embeddings_64d.npy",
    ),
}


def _resolved_path(value: object) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    return Path(expanded).resolve(strict=False)


def resolve_feature_paths(config: object) -> Mapping[str, Path]:
    """Return explicit feature paths, with ``embedding.dir`` as fallback.

    The central benchmark runner supplies absolute values for all four
    feature-specific keys.  Directory fallback is retained solely for the
    original Ethnobotany entry points and configuration files.
    """

    if config.contains("embedding.dir"):
        embedding_dir = _resolved_path(config["embedding.dir"])
    else:
        embedding_dir = Path(__file__).resolve().parent / "embeddings"

    resolved: dict[str, Path] = {}
    for entity, (key, filename) in _FEATURE_SPECS.items():
        value = config[key] if config.contains(key) else embedding_dir / filename
        resolved[entity] = _resolved_path(value)
    return resolved


def load_protein_missing_mask(
    config: object, *, expected_rows: int
) -> tuple[np.ndarray | None, Path | None]:
    """Load the explicit zero-UNK mask supplied by the fair runner.

    The legacy standalone entry point remains usable without a mask. Formal
    benchmark runs enforce its presence in the central runner before model
    construction and pass the verified path through a dedicated config key.
    """

    key = "protein.embedding.missing_mask"
    if not config.contains(key):
        return None, None
    path = _resolved_path(config[key])
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load protein missing mask {path}: {error}") from error
    if raw.dtype != np.bool_ or raw.shape != (int(expected_rows),):
        raise ValueError(
            "protein missing mask must be boolean with shape "
            f"({int(expected_rows)},), got dtype={raw.dtype}, shape={raw.shape}"
        )
    return np.asarray(raw, dtype=bool), path


def standardize_and_clip_feature_matrix(
    matrix: np.ndarray,
    *,
    missing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply historical normalization while preserving declared zero-UNK rows."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"feature matrix must be numeric and 2-D, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("feature matrix contains NaN or infinity")
    mask: np.ndarray | None = None
    if missing_mask is not None:
        mask = np.asarray(missing_mask)
        if mask.dtype != np.bool_ or mask.shape != (values.shape[0],):
            raise ValueError(
                "missing mask must be boolean with one entry per feature row"
            )
        if mask.any() and not np.array_equal(
            values[mask],
            np.zeros((int(mask.sum()), values.shape[1]), dtype=values.dtype),
        ):
            raise ValueError("masked zero-UNK feature rows are not exactly zero")
    standardized = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-8)
    standardized = np.clip(standardized, -10, 10)
    if mask is not None and mask.any():
        standardized[mask] = 0
    return standardized
