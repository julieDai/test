"""Runtime configuration shared by public ASTRA entry points."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Mapping

from .paths import PathLike, resolve_project_path


def seed_everything(seed: int, *, include_tensorflow: bool = False) -> int:
    """Seed Python, NumPy, and optionally TensorFlow without requiring TF."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    if include_tensorflow:
        import tensorflow.compat.v1 as tf

        tf.set_random_seed(seed)
    return seed


def configure_legacy_environment(
    *,
    seed: int,
    output_dir: PathLike | None = None,
    data_dir: PathLike | None = None,
) -> dict[str, str]:
    """Backward-compatible alias for :func:`configure_run_environment`."""

    values: dict[str, Any] = {"seed": seed, "output_dir": output_dir}
    if data_dir is not None:
        directory = resolve_project_path(data_dir)
        values.update(
            {
                "data_dir": directory,
                "hc_file": directory / "H_C.txt",
                "pd_file": directory / "P_D.txt",
                "negative_file": directory / "negativehdlist.txt",
            }
        )
    return configure_run_environment(**values)


_RUN_ENVIRONMENT_NAMES = {
    "seed": ("ASTRA_SEED", "CDRUG_SEED"),
    "output_dir": ("ASTRA_OUTPUT_DIR", "CDRUG_OUTPUT_DIR"),
    "log_dir": ("ASTRA_LOG_DIR", "CDRUG_LOG_DIR"),
    "data_dir": ("ASTRA_DATA_DIR", "CDRUG_DATA_DIR"),
    "negative_file": ("ASTRA_NEGATIVE_FILE", "CDRUG_NEGATIVE_FILE"),
    "hc_file": ("ASTRA_HC_FILE", "CDRUG_HC_FILE"),
    "pd_file": ("ASTRA_PD_FILE", "CDRUG_PD_FILE"),
    "cp_file": ("ASTRA_CP_FILE", "CDRUG_CP_FILE"),
    "node_hc_file": ("ASTRA_NODE_HC_FILE", "CDRUG_NODE_HC_FILE"),
    "node_pd_file": ("ASTRA_NODE_PD_FILE", "CDRUG_NODE_PD_FILE"),
    "compound_embeddings": (
        "ASTRA_COMPOUND_EMBEDDINGS",
        "CDRUG_COMPOUND_EMBEDDINGS",
    ),
    "protein_embeddings": (
        "ASTRA_PROTEIN_EMBEDDINGS",
        "CDRUG_PROTEIN_EMBEDDINGS",
    ),
    "herb_text_embeddings": (
        "ASTRA_HERB_TEXT_EMBEDDINGS",
        "CDRUG_HERB_TEXT_EMBEDDINGS",
    ),
    "disease_text_embeddings": (
        "ASTRA_DISEASE_TEXT_EMBEDDINGS",
        "CDRUG_DISEASE_TEXT_EMBEDDINGS",
    ),
    "allow_missing_semantic": ("ASTRA_ALLOW_MISSING_SEMANTIC",),
}


def configure_run_environment(**settings: Any) -> dict[str, str]:
    """Set canonical ``ASTRA_*`` variables and their legacy aliases together.

    Experiment runners call this function after parsing their command-line
    arguments. This guarantees that explicit per-run paths and seeds override
    stale values inherited from a shell or ``.env`` file, regardless of which
    variable name a legacy implementation module reads.
    """

    unknown = sorted(set(settings).difference(_RUN_ENVIRONMENT_NAMES))
    if unknown:
        raise TypeError(f"Unknown runtime settings: {', '.join(unknown)}")

    updated: dict[str, str] = {}
    for setting, value in settings.items():
        if value is None:
            continue
        if setting == "seed":
            text = str(int(value))
        elif isinstance(value, bool):
            text = "1" if value else "0"
        else:
            text = os.fspath(value)
        for name in _RUN_ENVIRONMENT_NAMES[setting]:
            os.environ[name] = text
            updated[name] = text
    return updated


def tcm_dataset_directory(
    protocol: Mapping[str, Any],
    *,
    protocol_dir: PathLike,
    override: PathLike | None = None,
) -> Path:
    """Resolve the TCM-suite dataset used to create a frozen protocol."""

    if override is not None:
        return Path(override).expanduser().resolve()
    try:
        positive_value = protocol["positive_file"]
    except KeyError as exc:
        raise KeyError(
            "protocol.json has no positive_file; pass --dataset-dir explicitly"
        ) from exc
    positive_path = Path(os.path.expandvars(os.path.expanduser(os.fspath(positive_value))))
    if not positive_path.is_absolute():
        positive_path = Path(protocol_dir).expanduser().resolve() / positive_path
    return positive_path.resolve().parent


def configure_tcm_data_paths(
    config: Any,
    data_dir: PathLike,
    *,
    negative_file: PathLike | None = None,
    hc_file: PathLike | None = None,
    pd_file: PathLike | None = None,
    node_hc_file: PathLike | None = None,
    node_pd_file: PathLike | None = None,
) -> dict[str, Path]:
    """Point a legacy-compatible TCM config and environment at one dataset."""

    directory = Path(data_dir).expanduser().resolve()
    paths = {
        "datapath": directory / "H_D.txt",
        "hc.path": Path(hc_file).expanduser().resolve() if hc_file else directory / "H_C.txt",
        "cp.path": directory / "C_P.txt",
        "pd.path": Path(pd_file).expanduser().resolve() if pd_file else directory / "P_D.txt",
        "node.hc.path": (
            Path(node_hc_file).expanduser().resolve()
            if node_hc_file
            else directory / "H_C.txt"
        ),
        "node.pd.path": (
            Path(node_pd_file).expanduser().resolve()
            if node_pd_file
            else directory / "P_D.txt"
        ),
        "negative.path": (
            Path(negative_file).expanduser().resolve()
            if negative_file
            else directory / "negativehdlist.txt"
        ),
        "compound.embedding": directory / "compound_embedding_kpgt.txt",
        "protein.embedding": directory / "protein_embedding_esm2.txt",
    }
    config.set("data.dir", directory)
    for key, path in paths.items():
        config.set(key, path)
    configure_run_environment(
        data_dir=directory,
        negative_file=paths["negative.path"],
        hc_file=paths["hc.path"],
        cp_file=paths["cp.path"],
        pd_file=paths["pd.path"],
        node_hc_file=paths["node.hc.path"],
        node_pd_file=paths["node.pd.path"],
        compound_embeddings=paths["compound.embedding"],
        protein_embeddings=paths["protein.embedding"],
    )
    return paths
