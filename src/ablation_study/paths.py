"""Path resolution shared by the TCM-suite ablation variants.

All configuration values are interpreted relative to the configuration file,
not the process working directory. ``HERBHG_*`` variables take precedence;
backend compatibility aliases remain accepted.
This module intentionally has no TensorFlow dependency so path handling can be
tested in a lightweight environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from util.paths import resolve_config_path


@dataclass(frozen=True)
class AblationEmbeddingPaths:
    """Resolved biological and semantic feature files for an ablation run."""

    compound: Path
    protein: Path
    compound_legacy: Path
    protein_legacy: Path
    herb_semantic: Path
    disease_semantic: Path


def resolve_embedding_paths(config) -> AblationEmbeddingPaths:
    """Resolve every embedding path used by variants 1--5.

    Explicit configuration keys take precedence over the defaults.  Environment
    overrides take precedence over both, matching the maintained HerbHG runner.
    """

    return AblationEmbeddingPaths(
        compound=resolve_config_path(
            config,
            "compound.embedding",
            "../dataset/compound_embedding_kpgt.txt",
            ("HERBHG_COMPOUND_EMBEDDINGS", "ASTRA_COMPOUND_EMBEDDINGS", "CDRUG_COMPOUND_EMBEDDINGS"),
        ),
        protein=resolve_config_path(
            config,
            "protein.embedding",
            "../dataset/protein_embedding_esm2.txt",
            ("HERBHG_PROTEIN_EMBEDDINGS", "ASTRA_PROTEIN_EMBEDDINGS", "CDRUG_PROTEIN_EMBEDDINGS"),
        ),
        compound_legacy=resolve_config_path(
            config,
            "compound.embedding.legacy",
            "../dataset/compound_embedding-64.txt",
            (
                "HERBHG_COMPOUND_EMBEDDINGS_LEGACY",
                "ASTRA_COMPOUND_EMBEDDINGS_LEGACY",
                "CDRUG_COMPOUND_EMBEDDINGS_LEGACY",
            ),
        ),
        protein_legacy=resolve_config_path(
            config,
            "protein.embedding.legacy",
            "../dataset/protein_low.txt",
            (
                "HERBHG_PROTEIN_EMBEDDINGS_LEGACY",
                "ASTRA_PROTEIN_EMBEDDINGS_LEGACY",
                "CDRUG_PROTEIN_EMBEDDINGS_LEGACY",
            ),
        ),
        herb_semantic=resolve_config_path(
            config,
            "herb.text.embedding",
            "../../artifacts/tcm_suite/herb_semantic_embeddings_64d.npy",
            (
                "HERBHG_HERB_TEXT_EMBEDDINGS",
                "ASTRA_HERB_TEXT_EMBEDDINGS",
                "CDRUG_HERB_TEXT_EMBEDDINGS",
            ),
        ),
        disease_semantic=resolve_config_path(
            config,
            "disease.text.embedding",
            "../../artifacts/tcm_suite/disease_semantic_embeddings_64d.npy",
            (
                "HERBHG_DISEASE_TEXT_EMBEDDINGS",
                "ASTRA_DISEASE_TEXT_EMBEDDINGS",
                "CDRUG_DISEASE_TEXT_EMBEDDINGS",
            ),
        ),
    )
