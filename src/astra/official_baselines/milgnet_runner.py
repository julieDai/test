#!/usr/bin/env python3
"""Run the authors' MilGNet model on one central fair-protocol fold.

The neural architecture is imported from the pinned author checkout.  This
adapter deliberately owns only the benchmark boundary: compact identifiers,
the shared five-fold rows, incidence-derived similarity construction,
leakage-safe graph construction, batching, and metric/provenance output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Dict, Tuple

import numpy as np

try:
    from .common import (
        DEFAULT_THIRD_PARTY_DIR,
        balanced_prefix_indices,
        batches,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        legacy_protocol_provenance,
        load_or_build_similarities,
        load_protocol,
        outer_training_split,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )
except ImportError:  # Direct script execution.
    from common import (
        DEFAULT_THIRD_PARTY_DIR,
        balanced_prefix_indices,
        batches,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        legacy_protocol_provenance,
        load_or_build_similarities,
        load_protocol,
        outer_training_split,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )

try:
    from ..benchmark.metrics import (
        PredictionTable,
        audit_milgnet_run,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )
except ImportError:  # Direct script execution with ``src`` on PYTHONPATH.
    from astra.benchmark.metrics import (
        PredictionTable,
        audit_milgnet_run,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )


UPSTREAM_COMMIT = "0149c707e70b039e11ba8c5b88b869aed38e152d"
UPSTREAM_SOURCE = "https://github.com/gu-yaowen/MilGNet"


def _source_revision(source_dir: Path) -> Tuple[str | None, bool | None]:
    """Return the checkout commit and dirty state when Git metadata exists."""
    if not (source_dir / ".git").exists():
        return None, None
    revision = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    # Importing a source checkout may leave Python bytecode beside model.py.
    # Bytecode is not source drift; every other tracked or untracked path is.
    material_changes = []
    for line in status_lines:
        path = line[3:].strip()
        if "__pycache__/" in path or path.endswith(".pyc"):
            continue
        material_changes.append(line)
    dirty = bool(material_changes)
    return revision, dirty


def _validate_source(source_dir: Path, allow_unpinned: bool) -> Dict[str, object]:
    required = (
        source_dir / "model.py",
        source_dir / "layer.py",
        source_dir / "load_data.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "MilGNet checkout is incomplete; run "
            "third_party/fetch_baselines.sh. Missing: "
            + ", ".join(missing)
        )
    if allow_unpinned:
        revision, dirty = _source_revision(source_dir)
    else:
        revision = verify_checkout_commit(source_dir, UPSTREAM_COMMIT)
        _, dirty = _source_revision(source_dir)
    if not allow_unpinned and dirty:
        raise RuntimeError(
            "MilGNet checkout has local changes. A formal run requires the "
            "clean pinned author source; pass --allow-unpinned-source only "
            "for an explicit development run."
        )
    return {
        "path": str(source_dir),
        "git_commit": revision,
        "git_dirty": dirty,
        "expected_commit": UPSTREAM_COMMIT,
        "model_py_sha256": sha256(source_dir / "model.py"),
        "layer_py_sha256": sha256(source_dir / "layer.py"),
        "load_data_py_sha256": sha256(source_dir / "load_data.py"),
    }


def _load_official_model(source_dir: Path):
    """Import Model from the author checkout without copying its architecture."""
    # The released model.py uses ``from layer import *`` rather than a package
    # relative import, so its directory must be temporarily first on sys.path.
    source_text = str(source_dir)
    previous_layer = sys.modules.pop("layer", None)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_prefix = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="herbhg_milgnet_pycache_") as pycache:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = pycache
        sys.path.insert(0, source_text)
        try:
            spec = importlib.util.spec_from_file_location(
                "herbhg_milgnet_official_model", source_dir / "model.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {source_dir / 'model.py'}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(source_text)
            if previous_layer is not None:
                sys.modules["layer"] = previous_layer
            else:
                sys.modules.pop("layer", None)
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    if not hasattr(module, "Model"):
        raise ImportError(f"{source_dir / 'model.py'} does not define Model")
    return module.Model


def _set_seed(seed: int, torch, dgl) -> None:
    """Apply the released runner's deterministic seed policy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(dgl, "seed"):
        dgl.seed(seed)
    if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
        dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _official_topk_filtering(
    similarity: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the released ``topk_filtering`` operation literally.

    In particular, the author code uses ``np.argpartition`` and the slice
    ``[-k-1:-1]``.  It neither sorts ties nor explicitly removes the diagonal.
    The selected entries are set to one, after which *all* exact-one entries
    become side-graph links.  Keeping a private copy prevents mutation of the
    shared, hash-checked Jaccard cache.
    """
    filtered = np.asarray(similarity, dtype=np.float32).copy()
    n = filtered.shape[0]
    if filtered.shape != (n, n):
        raise ValueError("similarity must be square")
    if not 1 <= k < n:
        raise ValueError(f"k must be in 1..{n - 1}")

    selected = np.empty((n, k), dtype=np.int64)
    selected_weights = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        # Do not replace this with a stable/sorted top-k: tie and self
        # behaviour are part of the pinned load_data.py implementation.
        partition = np.argpartition(filtered[row], -k - 1)
        row_selected = partition[-k - 1 : -1]
        selected[row] = row_selected
        selected_weights[row] = filtered[row, row_selected]
        filtered[row, row_selected] = np.float32(1.0)
    edges = np.argwhere(filtered == np.float32(1.0)).astype(
        np.int64, copy=False
    )
    return filtered, edges, selected, selected_weights


def _row_destinations(edges: np.ndarray, n_rows: int) -> Tuple[np.ndarray, ...]:
    """Convert row-major ``np.where`` links into per-source destinations."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (n, 2), got {edges.shape}")
    if len(edges) and (
        edges[:, 0].min() < 0 or edges[:, 0].max() >= n_rows
    ):
        raise ValueError("edge source outside neighbor table")
    counts = np.bincount(edges[:, 0], minlength=n_rows)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return tuple(edges[offsets[row] : offsets[row + 1], 1] for row in range(n_rows))


@dataclass(frozen=True)
class PackedNeighborRows:
    """Bounded dense prefixes plus the exact lengths of variable neighbor rows."""

    values: np.ndarray
    lengths: np.ndarray


def cyclic_balanced_epoch_positions(
    labels: np.ndarray,
    rows_per_epoch: int,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Select a deterministic balanced cyclic block with eventual full coverage."""
    labels = np.asarray(labels).reshape(-1)
    if rows_per_epoch <= 0 or rows_per_epoch >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    if rows_per_epoch < 2 or rows_per_epoch % 2:
        raise ValueError("rows_per_epoch must be zero or a positive even integer")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("cyclic balanced schedule requires both classes")
    per_class = rows_per_epoch // 2
    if per_class > min(len(positive), len(negative)):
        raise ValueError("rows_per_epoch exceeds the available balanced rows")
    def take(values: np.ndarray, class_tag: int) -> np.ndarray:
        """Walk complete seeded permutations, reshuffling only at a cycle edge."""
        absolute_start = epoch * per_class
        remaining = per_class
        selected_parts = []
        while remaining:
            cycle, within_cycle = divmod(absolute_start, len(values))
            permutation = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(class_tag), int(cycle)])
            ).permutation(len(values))
            take_count = min(remaining, len(values) - within_cycle)
            selected_parts.append(
                values[permutation[within_cycle : within_cycle + take_count]]
            )
            remaining -= take_count
            absolute_start += take_count
        return np.concatenate(selected_parts)

    selected = np.concatenate((take(positive, 1), take(negative, 0)))
    shuffle = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), 0x4D494C])
    ).permutation(len(selected))
    return selected[shuffle]


def pack_neighbor_rows(
    rows: Tuple[np.ndarray, ...], max_instances: int
) -> PackedNeighborRows:
    """Pack only neighbor prefixes that can occur before the bag cap.

    The author loader may produce more than ``k`` links when the unmodified
    Jaccard matrix already contains exact-one ties.  Bag construction caps the
    ordered instance list at ``k*(k+2)+1``; consequently no neighbor beyond
    ``max_instances - 1`` can be observed.  Keeping the original row lengths
    preserves all ordering/truncation semantics while bounding storage.
    """
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    rows = tuple(np.asarray(row, dtype=np.int64) for row in rows)
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    width = max(1, min(max_instances - 1, int(lengths.max(initial=0))))
    values = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        take = min(len(row), width)
        if take:
            values[index, :take] = row[:take]
    return PackedNeighborRows(values=values, lengths=lengths)


def make_meta_path_bags_vectorized(
    pairs: np.ndarray,
    herb_neighbors: PackedNeighborRows,
    disease_neighbors: PackedNeighborRows,
    *,
    max_instances: int,
    seed: int = 42,
) -> np.ndarray:
    """Vectorized exact equivalent of :func:`make_meta_path_bags`.

    This eliminates one Python loop per H--D row.  It does not sample or alter
    bags: direct, one-sided, and Cartesian instances retain author-code order,
    prefix truncation, and the reset-seed padding choice.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n, 2), got {pairs.shape}")
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    if len(pairs) == 0:
        return np.empty((0, max_instances, 4), dtype=np.int64)
    herbs = pairs[:, 0]
    diseases = pairs[:, 1]
    if herbs.min() < 0 or herbs.max() >= len(herb_neighbors.lengths):
        raise ValueError("herb index outside neighbor table")
    if diseases.min() < 0 or diseases.max() >= len(disease_neighbors.lengths):
        raise ValueError("disease index outside neighbor table")

    herb_count = herb_neighbors.lengths[herbs][:, None]
    disease_count = disease_neighbors.lengths[diseases][:, None]
    raw_count = (herb_count + 1) * (disease_count + 1)
    kept_count = np.minimum(raw_count, max_instances)
    positions = np.arange(max_instances, dtype=np.int64)[None, :]

    # Upstream calls random.seed(seed) immediately before each padding choice,
    # so the chosen source position is a deterministic function of raw_count.
    padding_choice = np.zeros(max_instances + 1, dtype=np.int64)
    for count in range(1, max_instances):
        padding_choice[count] = random.Random(seed).randrange(count)
    lookup_count = np.minimum(raw_count, max_instances)
    source_position = np.where(
        positions < kept_count,
        positions,
        padding_choice[lookup_count],
    )

    herb_only = (source_position >= 1) & (source_position <= herb_count)
    disease_only = (source_position > herb_count) & (
        source_position <= herb_count + disease_count
    )
    cross = source_position > herb_count + disease_count
    safe_disease_count = np.maximum(disease_count, 1)
    cross_position = source_position - herb_count - disease_count - 1

    herb_neighbor_position = np.where(
        herb_only,
        source_position - 1,
        cross_position // safe_disease_count,
    )
    disease_neighbor_position = np.where(
        disease_only,
        source_position - herb_count - 1,
        cross_position % safe_disease_count,
    )
    herb_neighbor_position = np.clip(
        herb_neighbor_position, 0, herb_neighbors.values.shape[1] - 1
    )
    disease_neighbor_position = np.clip(
        disease_neighbor_position, 0, disease_neighbors.values.shape[1] - 1
    )

    row = np.arange(len(pairs), dtype=np.int64)[:, None]
    herb_lookup = herb_neighbors.values[herbs[row], herb_neighbor_position]
    disease_lookup = disease_neighbors.values[
        diseases[row], disease_neighbor_position
    ]
    bags = np.empty((len(pairs), max_instances, 4), dtype=np.int64)
    bags[:, :, 0] = herbs[:, None]
    bags[:, :, 1] = np.where(herb_only | cross, herb_lookup, herbs[:, None])
    bags[:, :, 2] = np.where(
        disease_only | cross, disease_lookup, diseases[:, None]
    )
    bags[:, :, 3] = diseases[:, None]
    return bags


def make_meta_path_bags(
    pairs: np.ndarray,
    herb_neighbors,
    disease_neighbors,
    *,
    max_instances: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create only the MilGNet bags requested by the current batch.

    The instance ordering matches ``meta_path_instance`` in the released
    ``load_data.py``: direct instance, herb-neighbour instances,
    disease-neighbour instances, then their Cartesian product.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n, 2), got {pairs.shape}")
    # Two-dimensional arrays remain accepted for small unit fixtures.  Formal
    # execution passes variable-length rows derived from the author's exact
    # equality-to-one links (including self and pre-existing exact-one ties).
    if isinstance(herb_neighbors, np.ndarray):
        if herb_neighbors.ndim != 2:
            raise ValueError("herb neighbor array must be two-dimensional")
        herb_neighbors = tuple(row for row in herb_neighbors)
    else:
        herb_neighbors = tuple(herb_neighbors)
    if isinstance(disease_neighbors, np.ndarray):
        if disease_neighbors.ndim != 2:
            raise ValueError("disease neighbor array must be two-dimensional")
        disease_neighbors = tuple(row for row in disease_neighbors)
    else:
        disease_neighbors = tuple(disease_neighbors)
    if max_instances is None:
        herb_widths = {len(row) for row in herb_neighbors}
        disease_widths = {len(row) for row in disease_neighbors}
        if len(herb_widths) != 1 or herb_widths != disease_widths:
            raise ValueError("max_instances is required for variable-length links")
        k = next(iter(herb_widths))
        max_instances = k * (k + 2) + 1
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    if len(pairs) == 0:
        return np.empty((0, max_instances, 4), dtype=np.int64)
    if pairs[:, 0].min() < 0 or pairs[:, 0].max() >= len(herb_neighbors):
        raise ValueError("herb index outside neighbor table")
    if pairs[:, 1].min() < 0 or pairs[:, 1].max() >= len(disease_neighbors):
        raise ValueError("disease index outside neighbor table")

    bags = np.empty((len(pairs), max_instances, 4), dtype=np.int64)
    for position, (herb, disease) in enumerate(pairs):
        herb_side = np.asarray(herb_neighbors[herb], dtype=np.int64)
        disease_side = np.asarray(disease_neighbors[disease], dtype=np.int64)
        raw_count = (len(herb_side) + 1) * (len(disease_side) + 1)
        kept = min(raw_count, max_instances)
        # Generate only the upstream prefix that survives truncation.  This
        # is important for rows with many pre-existing exact-one links: the
        # formal adapter never materializes an unbounded Cartesian bag.
        raw = np.empty((kept, 4), dtype=np.int64)
        raw[:, 0] = herb
        raw[:, 3] = disease
        raw[0, 1:3] = (herb, disease)
        cursor = 1
        take = min(len(herb_side), kept - cursor)
        raw[cursor : cursor + take, 1] = herb_side[:take]
        raw[cursor : cursor + take, 2] = disease
        cursor += take
        take = min(len(disease_side), kept - cursor)
        raw[cursor : cursor + take, 1] = herb
        raw[cursor : cursor + take, 2] = disease_side[:take]
        cursor += take
        take = kept - cursor
        if take:
            cartesian_position = np.arange(take, dtype=np.int64)
            raw[cursor:, 1] = herb_side[
                cartesian_position // len(disease_side)
            ]
            raw[cursor:, 2] = disease_side[
                cartesian_position % len(disease_side)
            ]

        bags[position, :kept] = raw
        if kept < max_instances:
            # The released loop resets random.seed(seed) before every choice,
            # so each padding row is the same seeded choice from the raw bag.
            padding = random.Random(seed).choice(raw.tolist())
            bags[position, kept:] = np.asarray(padding, dtype=np.int64)
    return bags


def _side_edges(similarity: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Match the released loader's side-graph edge rule.

    Upstream first overwrites the selected top-k similarities with one and
    then creates edges at *every* entry equal to one.  The latter includes the
    diagonal and any pre-existing exact-one similarities, not merely the k
    selected neighbours.  Keeping this separate from bag-neighbour ordering
    preserves both behaviours without mutating the cached similarity matrix.
    """
    binarized = np.asarray(similarity, dtype=np.float32).copy()
    rows = np.repeat(np.arange(len(neighbors), dtype=np.int64), neighbors.shape[1])
    binarized[rows, neighbors.reshape(-1)] = np.float32(1.0)
    return np.argwhere(binarized == np.float32(1.0)).astype(np.int64, copy=False)


def _build_graph(
    dgl,
    torch,
    protocol,
    train_positive_pairs,
    herb_edges,
    disease_edges,
):
    associations = np.unique(np.asarray(train_positive_pairs, dtype=np.int64), axis=0)
    reverse = associations[:, [1, 0]]

    def tensors(edges: np.ndarray):
        return (
            torch.from_numpy(edges[:, 0].copy()).long(),
            torch.from_numpy(edges[:, 1].copy()).long(),
        )

    graph_data = {
        ("drug", "drug-drug", "drug"): tensors(herb_edges),
        ("drug", "drug-disease", "disease"): tensors(associations),
        ("disease", "disease-drug", "drug"): tensors(reverse),
        ("disease", "disease-disease", "disease"): tensors(disease_edges),
    }
    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"drug": protocol.n_herbs, "disease": protocol.n_diseases},
    )
    return graph, {
        "drug-drug": int(len(herb_edges)),
        "drug-disease": int(len(associations)),
        "disease-drug": int(len(reverse)),
        "disease-disease": int(len(disease_edges)),
    }


def _build_features(torch, herb_features, disease_features):
    n_herbs = herb_features.shape[0]
    n_diseases = disease_features.shape[0]
    drug = np.concatenate(
        (herb_features, np.zeros((n_herbs, n_diseases), dtype=np.float32)), axis=1
    )
    disease = np.concatenate(
        (np.zeros((n_diseases, n_herbs), dtype=np.float32), disease_features), axis=1
    )
    return {
        "drug": torch.from_numpy(drug),
        "disease": torch.from_numpy(disease),
    }


def _allow_zero_in_degree(model) -> int:
    """Use zero relation messages for isolated nodes instead of fake H-D edges."""
    changed = 0
    for module in model.modules():
        setter = getattr(module, "set_allow_zero_in_degree", None)
        if callable(setter):
            setter(True)
            changed += 1
    return changed


def _pair_overlap(left: np.ndarray, right: np.ndarray) -> int:
    left_set = {tuple(value) for value in np.asarray(left, dtype=np.int64)}
    right_set = {tuple(value) for value in np.asarray(right, dtype=np.int64)}
    return len(left_set.intersection(right_set))


def _predict(
    model,
    graph,
    features,
    pairs,
    herb_neighbors,
    disease_neighbors,
    max_instances,
    seed,
    batch_size,
    device,
    torch,
):
    scores = np.empty(len(pairs), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for positions in batches(np.arange(len(pairs), dtype=np.int64), batch_size):
            bags_np = make_meta_path_bags(
                pairs[positions],
                herb_neighbors,
                disease_neighbors,
                max_instances=max_instances,
                seed=seed,
            )
            bag_tensor = torch.from_numpy(bags_np).long().to(device)
            logits, _ = model(graph, features, bag_tensor)
            scores[positions] = (
                torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
            )
    return scores


def encode_official_model(model, graph, features, torch):
    """Run the pinned model's graph encoder exactly once.

    ``Model.forward`` repeats this graph-wide computation for every pair
    minibatch even though its result is independent of the requested bags.
    Exposing the existing author modules as an encode/decode boundary removes
    that redundancy without replacing any learned component.
    """
    integrated_drug = []
    integrated_disease = []
    hidden = model.lin_transform(features)
    integrated_drug.append(hidden["drug"])
    integrated_disease.append(hidden["disease"])
    for layer_name in model.graph_embedding:
        hidden = model.graph_embedding[layer_name](graph, hidden)
        integrated_drug.append(hidden["drug"])
        integrated_disease.append(hidden["disease"])
    if model.skip:
        hidden = {
            "drug": torch.stack(integrated_drug, dim=1),
            "disease": torch.stack(integrated_disease, dim=1),
        }
        hidden["drug"] = model.layer_attention_drug(hidden["drug"])
        hidden["disease"] = model.layer_attention_dis(hidden["disease"])
    return hidden


def decode_official_model(model, hidden, bags, *, fused_mean: bool = True):
    """Run the pinned model's bag decoder, optionally with an exact fused mean.

    The default author configuration is the parameter-free mean of the four
    node states in each meta-path instance.  Summing those four indexed tensors
    directly avoids materializing a much larger ``[..., 4, hidden]`` tensor.
    All learned MIL attention and prediction modules remain the imported
    author-code objects.
    """
    if fused_mean and getattr(model.aggregator, "agg_type", None) == "mean":
        instance_embedding = (
            hidden["drug"][bags[:, :, 0]]
            + hidden["drug"][bags[:, :, 1]]
            + hidden["disease"][bags[:, :, 2]]
            + hidden["disease"][bags[:, :, 3]]
        ) / 4.0
    else:
        instance_embedding = model.aggregator(hidden, bags)
    if model.mil:
        bag_embedding, attention = model.mil_layer(instance_embedding)
    else:
        bag_embedding = instance_embedding.sum(dim=1)
        attention = None
    bag_prediction = model.bag_predict(bag_embedding)
    if model.ins_predict:
        instance_prediction = model.ins_predict(instance_embedding, attention)
        return (instance_prediction + bag_prediction) / 2, attention
    return bag_prediction, attention


def _predict_encoded(
    model,
    graph,
    features,
    pairs,
    herb_neighbors: PackedNeighborRows,
    disease_neighbors: PackedNeighborRows,
    max_instances,
    seed,
    chunk_size,
    device,
    torch,
):
    """Predict with one graph encoding and exact chunked author bag decoding."""
    scores = np.empty(len(pairs), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        hidden = encode_official_model(model, graph, features, torch)
        for positions in batches(np.arange(len(pairs), dtype=np.int64), chunk_size):
            bags_np = make_meta_path_bags_vectorized(
                pairs[positions],
                herb_neighbors,
                disease_neighbors,
                max_instances=max_instances,
                seed=seed,
            )
            bag_tensor = torch.from_numpy(bags_np).long().to(device)
            logits, _ = decode_official_model(model, hidden, bag_tensor)
            scores[positions] = (
                torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        required=True,
        help="Central fair-protocol manifest file or directory.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_THIRD_PARTY_DIR / "MilGNet_official",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--training-mode",
        choices=("released-minibatch", "scalable-full-batch"),
        default="released-minibatch",
        help=(
            "The released minibatch optimizer trajectory or the explicitly "
            "labelled full-outer-data scalable adaptation."
        ),
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2048,
        help="Pair chunks used by scalable exact bag decoding.",
    )
    parser.add_argument(
        "--scalable-rows-per-epoch",
        type=int,
        default=0,
        help=(
            "For scalable-full-batch only: an even balanced cyclic supervision "
            "block per epoch; zero uses every training row every epoch."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=("sum", "mean", "Linear", "BiTrans"),
    )
    parser.add_argument("--instance-topk", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-rows", type=int, default=128)
    parser.add_argument("--smoke-test-rows", type=int, default=128)
    parser.add_argument("--allow-unpinned-source", action="store_true")
    parser.add_argument(
        "--legacy-test-oracle",
        action="store_true",
        help=(
            "Restore all four outer-training folds and report descriptive "
            "test-fold F1-threshold diagnostics for the retained legacy comparison."
        ),
    )
    parser.add_argument("--run-generation-id")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.decode_chunk_size < 1:
        parser.error("--epochs, --batch-size, and --decode-chunk-size must be positive")
    if args.scalable_rows_per_epoch < 0 or (
        args.scalable_rows_per_epoch and args.scalable_rows_per_epoch % 2
    ):
        parser.error("--scalable-rows-per-epoch must be zero or a positive even integer")
    if (
        args.training_mode != "scalable-full-batch"
        and args.scalable_rows_per_epoch
    ):
        parser.error("--scalable-rows-per-epoch requires --training-mode scalable-full-batch")
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    started = time.time()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.source_dir = args.source_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.cache_dir is None:
        args.cache_dir = args.output_root.parent / "cache"
    args.cache_dir = args.cache_dir.resolve()
    source_provenance = _validate_source(
        args.source_dir, args.allow_unpinned_source
    )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "MilGNet requires PyTorch; use the isolated official-baselines environment"
        ) from error
    # Avoid DGL's first-import attempt to write ~/.dgl/config.json.  The
    # workspace sandbox intentionally keeps the home directory read-only.
    os.environ.setdefault("DGLBACKEND", "pytorch")
    try:
        import dgl
    except ImportError as error:
        raise RuntimeError(
            "MilGNet requires DGL; the author README specifies torch 1.8.0 and "
            "dgl 0.5.2, while the pinned requirements file specifies DGL 0.8.1. "
            "Use the isolated official-baselines environment."
        ) from error

    _set_seed(args.seed, torch, dgl)
    if args.device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)

    protocol = load_protocol(args.protocol_manifest)
    if args.legacy_test_oracle and (
        protocol.dataset != "tcm-suite"
        or getattr(protocol, "pair_order", "seeded") != "source"
    ):
        parser.error(
            "--legacy-test-oracle is defined only for the source-order "
            "TCM-suite manifest"
        )
    initial_input_manifest = protocol.input_manifest()
    frozen_input_lock_sha256 = validate_frozen_input_manifest(initial_input_manifest)
    fold_rows = protocol.fold_rows(args.fold)
    legacy_provenance = (
        legacy_protocol_provenance(
            protocol,
            fold_rows,
            args.protocol_manifest,
            require_clean_project=not args.smoke,
        )
        if args.legacy_test_oracle
        else None
    )
    herb_similarity, disease_similarity, similarity_cache = load_or_build_similarities(
        protocol, args.cache_dir
    )
    (
        herb_features,
        herb_edges,
        herb_selected,
        herb_selected_weights,
    ) = _official_topk_filtering(
        herb_similarity, args.neighbors
    )
    (
        disease_features,
        disease_edges,
        disease_selected,
        disease_selected_weights,
    ) = _official_topk_filtering(
        disease_similarity, args.neighbors
    )
    herb_neighbors = _row_destinations(herb_edges, protocol.n_herbs)
    disease_neighbors = _row_destinations(disease_edges, protocol.n_diseases)
    max_instances = args.neighbors * (args.neighbors + 2) + 1
    packed_herb_neighbors = pack_neighbor_rows(herb_neighbors, max_instances)
    packed_disease_neighbors = pack_neighbor_rows(disease_neighbors, max_instances)

    if args.legacy_test_oracle:
        train_pairs, train_labels, train_row_indices = outer_training_split(fold_rows)
    else:
        train_pairs, train_labels, train_row_indices = fold_split(fold_rows, "train")
    validation_pairs, validation_labels, validation_row_indices = fold_split(
        fold_rows, "validation"
    )
    test_pairs, test_labels, test_row_indices = fold_split(fold_rows, "test")
    full_train_pairs = train_pairs
    full_train_labels = train_labels
    full_train_row_indices = train_row_indices
    graph_training_positive_pairs = np.unique(
        full_train_pairs[full_train_labels == 1], axis=0
    )
    epochs_executed = 1 if args.smoke else args.epochs
    if args.smoke:
        train_positions = balanced_prefix_indices(
            train_labels, max(1, args.smoke_train_rows // 2)
        )
        validation_positions = balanced_prefix_indices(
            validation_labels, max(1, args.smoke_test_rows // 2)
        )
        test_positions = balanced_prefix_indices(
            test_labels, max(1, args.smoke_test_rows // 2)
        )
        train_pairs = train_pairs[train_positions]
        train_labels = train_labels[train_positions]
        train_row_indices = train_row_indices[train_positions]
        validation_pairs = validation_pairs[validation_positions]
        validation_labels = validation_labels[validation_positions]
        validation_row_indices = validation_row_indices[validation_positions]
        test_pairs = test_pairs[test_positions]
        test_labels = test_labels[test_positions]
        test_row_indices = test_row_indices[test_positions]

    test_positive_pairs = np.unique(test_pairs[test_labels == 1], axis=0)
    validation_positive_pairs = np.unique(
        validation_pairs[validation_labels == 1], axis=0
    )
    held_out_positive_pairs = (
        test_positive_pairs
        if args.legacy_test_oracle
        else np.concatenate((validation_positive_pairs, test_positive_pairs), axis=0)
    )
    positive_graph_overlap = _pair_overlap(
        graph_training_positive_pairs,
        held_out_positive_pairs,
    )
    if positive_graph_overlap:
        raise RuntimeError(
            f"{positive_graph_overlap} validation/test positive pair(s) also occur among "
            "training positives; refusing a graph-leaking run"
        )

    graph, graph_edges = _build_graph(
        dgl,
        torch,
        protocol,
        graph_training_positive_pairs,
        herb_edges,
        disease_edges,
    )
    features = _build_features(
        torch,
        herb_features,
        disease_features,
    )
    graph = graph.to(device)
    features = {key: value.to(device) for key, value in features.items()}

    OfficialModel = _load_official_model(args.source_dir)
    model = OfficialModel(
        graph.etypes,
        {key: int(value.shape[1]) for key, value in features.items()},
        hidden_feats=args.hidden_features,
        num_emb_layers=args.layers,
        agg_type=args.aggregate,
        dropout=args.dropout,
        bn=args.batch_norm,
        k=args.instance_topk,
        skip=True,
        mil=True,
    ).to(device)
    zero_degree_modules = _allow_zero_in_degree(model)
    if zero_degree_modules != len(graph.etypes) * args.layers:
        raise RuntimeError(
            "unexpected official Model structure: configured zero-degree handling "
            f"on {zero_degree_modules} graph convolutions, expected "
            f"{len(graph.etypes) * args.layers}"
        )

    positive = float(train_labels.sum())
    negative = float(len(train_labels) - positive)
    if positive == 0 or negative == 0:
        raise RuntimeError("training rows must contain both positive and negative labels")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    epoch_runtime_seconds = []
    shuffle_rng = np.random.default_rng(args.seed)
    # Upstream uses DataLoader(drop_last=True).  With fewer rows than one
    # batch this would silently execute no optimizer step and previously led
    # to division by zero here.  Retain drop-last whenever a full batch exists
    # and use the sole partial batch only for an explicitly undersized run.
    drop_last_active = (
        args.training_mode == "released-minibatch"
        and len(train_pairs) >= args.batch_size
    )
    optimizer_steps = 0
    setup_runtime_seconds = time.time() - started
    training_started = time.time()
    for epoch in range(epochs_executed):
        epoch_started = time.time()
        model.train()
        total_loss = 0.0
        seen_batches = 0
        if args.training_mode == "released-minibatch":
            permutation = shuffle_rng.permutation(len(train_pairs)).astype(np.int64)
            for positions in batches(permutation, args.batch_size):
                # The released training DataLoader uses drop_last=True.
                if drop_last_active and len(positions) < args.batch_size:
                    continue
                bags_np = make_meta_path_bags(
                    train_pairs[positions],
                    herb_neighbors,
                    disease_neighbors,
                    max_instances=max_instances,
                    seed=args.seed,
                )
                bag_tensor = torch.from_numpy(bags_np).long().to(device)
                targets = torch.from_numpy(train_labels[positions]).float().to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(graph, features, bag_tensor)
                loss = criterion(logits.reshape(-1), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite MilGNet loss at epoch {epoch + 1}, "
                        f"batch {seen_batches + 1}"
                    )
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                total_loss += float(loss.detach().cpu())
                seen_batches += 1
        else:
            # The selected BCE objective is accumulated in bounded decoder chunks.  The
            # graph encoder is evaluated/backpropagated exactly once per epoch:
            # a detached leaf accumulates decoder gradients, which are then
            # transferred to the connected author-code encoder output.
            optimizer.zero_grad(set_to_none=True)
            encoded = encode_official_model(model, graph, features, torch)
            decoder_hidden = {
                key: value.detach().requires_grad_(True)
                for key, value in encoded.items()
            }
            epoch_positions = cyclic_balanced_epoch_positions(
                train_labels,
                args.scalable_rows_per_epoch,
                seed=args.seed,
                epoch=epoch,
            )
            for positions in batches(epoch_positions, args.decode_chunk_size):
                bags_np = make_meta_path_bags_vectorized(
                    train_pairs[positions],
                    packed_herb_neighbors,
                    packed_disease_neighbors,
                    max_instances=max_instances,
                    seed=args.seed,
                )
                bag_tensor = torch.from_numpy(bags_np).long().to(device)
                targets = torch.from_numpy(train_labels[positions]).float().to(device)
                logits, _ = decode_official_model(
                    model, decoder_hidden, bag_tensor, fused_mean=True
                )
                loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.reshape(-1),
                    targets,
                    pos_weight=criterion.pos_weight,
                    reduction="sum",
                )
                loss = loss_sum / len(epoch_positions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite MilGNet loss at epoch {epoch + 1}, "
                        f"chunk {seen_batches + 1}"
                    )
                loss.backward()
                total_loss += float(loss_sum.detach().cpu())
                seen_batches += 1
                if seen_batches % 250 == 0:
                    print(
                        f"epoch={epoch + 1}/{epochs_executed} "
                        f"decoder_chunks={seen_batches}",
                        flush=True,
                    )
            hidden_gradients = [decoder_hidden[key].grad for key in encoded]
            if any(gradient is None for gradient in hidden_gradients):
                raise RuntimeError("scalable MilGNet decoder produced no encoder gradient")
            torch.autograd.backward(
                [encoded[key] for key in encoded],
                hidden_gradients,
            )
            optimizer.step()
            optimizer_steps += 1
        if seen_batches == 0:
            raise RuntimeError(
                "MilGNet executed zero training batches; reduce --batch-size "
                "or increase the selected training rows"
            )
        mean_loss = (
            total_loss / seen_batches
            if args.training_mode == "released-minibatch"
            else total_loss / len(epoch_positions)
        )
        history.append(mean_loss)
        epoch_runtime_seconds.append(time.time() - epoch_started)
        if (
            epoch == 0
            or (epoch + 1) % args.print_every == 0
            or epoch + 1 == epochs_executed
        ):
            print(
                f"epoch={epoch + 1}/{epochs_executed} loss={mean_loss:.8f}",
                flush=True,
            )

    training_runtime_seconds = time.time() - training_started
    prediction_started = time.time()
    if args.training_mode == "scalable-full-batch":
        prediction_arguments = (
            packed_herb_neighbors,
            packed_disease_neighbors,
            max_instances,
            args.seed,
            args.decode_chunk_size,
            device,
            torch,
        )
        validation_scores = _predict_encoded(
            model, graph, features, validation_pairs, *prediction_arguments
        )
        test_scores = _predict_encoded(
            model, graph, features, test_pairs, *prediction_arguments
        )
    else:
        prediction_arguments = (
            herb_neighbors,
            disease_neighbors,
            max_instances,
            args.seed,
            args.batch_size,
            device,
            torch,
        )
        validation_scores = _predict(
            model, graph, features, validation_pairs, *prediction_arguments
        )
        test_scores = _predict(
            model, graph, features, test_pairs, *prediction_arguments
        )
    if args.legacy_test_oracle:
        metrics = evaluate_test_oracle(test_labels, test_scores)
        metrics["Evaluation_Mode"] = "legacy-test-oracle"
    else:
        metrics = evaluate_validation_and_test(
            validation_labels,
            validation_scores,
            test_labels,
            test_scores,
        )
        metrics["Evaluation_Mode"] = "validation"
    prediction_runtime_seconds = time.time() - prediction_started
    actual_rows_per_epoch = (
        min(args.scalable_rows_per_epoch, len(train_pairs))
        if args.training_mode == "scalable-full-batch"
        and args.scalable_rows_per_epoch
        else len(train_pairs)
    )
    total_row_presentations = int(actual_rows_per_epoch * epochs_executed)
    if args.scalable_rows_per_epoch and args.scalable_rows_per_epoch < len(train_pairs):
        presentations_per_class = epochs_executed * (
            args.scalable_rows_per_epoch // 2
        )
        unique_supervision_rows_covered = min(
            int(np.count_nonzero(train_labels == 1)), presentations_per_class
        ) + min(int(np.count_nonzero(train_labels == 0)), presentations_per_class)
    else:
        unique_supervision_rows_covered = len(train_pairs)
    if args.training_mode == "released-minibatch":
        schedule_name = f"released-minibatch-k{args.neighbors}-e{args.epochs}"
    elif args.scalable_rows_per_epoch:
        schedule_name = (
            f"tcm-scalable-cyclic-balanced-{args.scalable_rows_per_epoch}"
            f"-k{args.neighbors}-{args.epochs}updates"
        )
    else:
        schedule_name = (
            f"scalable-full-outer-objective-k{args.neighbors}-e{args.epochs}"
        )
    peak_cuda_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    milgnet_audit = audit_milgnet_run(
        PredictionTable(
            row_id=test_row_indices,
            pair=test_pairs,
            label=test_labels,
            score=test_scores,
        ),
        optimizer_steps=optimizer_steps,
    )
    train_positive_degrees_h = np.bincount(
        graph_training_positive_pairs[:, 0], minlength=protocol.n_herbs
    )
    train_positive_degrees_d = np.bincount(
        graph_training_positive_pairs[:, 1], minlength=protocol.n_diseases
    )
    metrics.update(
        {
            "Model": "MilGNet-official-adapted",
            "Fold": args.fold,
            "Dataset": str(protocol.dataset),
            "Seed": args.seed,
            "Device": str(device),
            "Runtime_Seconds": time.time() - started,
            "Setup_Runtime_Seconds": setup_runtime_seconds,
            "Training_Runtime_Seconds": training_runtime_seconds,
            "Prediction_Runtime_Seconds": prediction_runtime_seconds,
            "Epochs_Requested": args.epochs,
            "Epochs_Executed": epochs_executed,
            "Training_Mode": args.training_mode,
            "Training_Schedule": schedule_name,
            "Total_Row_Presentations": total_row_presentations,
            "Effective_Label_Passes": total_row_presentations / len(full_train_pairs),
            "Unique_Supervision_Rows_Covered": unique_supervision_rows_covered,
            "Full_Outer_Row_Coverage": bool(
                not args.smoke
                and unique_supervision_rows_covered == len(full_train_pairs)
            ),
            "Peak_CUDA_Memory_MiB": peak_cuda_memory_mib,
            "Upstream_Commit": source_provenance["git_commit"],
            "Upstream_Source": UPSTREAM_SOURCE,
            "Adapter": (
                "official Model imported from pinned model.py; pinned load_data.py "
                "adjacency/bag semantics reproduced at the frozen protocol boundary"
            ),
            "Smoke": bool(args.smoke),
            "Formal_Eligible": bool(not args.smoke),
            **milgnet_audit,
            "Run_Parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    if legacy_provenance is not None:
        metrics["Legacy_Protocol_Provenance"] = legacy_provenance

    if not args.allow_unpinned_source:
        verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    final_source_hashes = {
        "model_py_sha256": sha256(args.source_dir / "model.py"),
        "layer_py_sha256": sha256(args.source_dir / "layer.py"),
        "load_data_py_sha256": sha256(args.source_dir / "load_data.py"),
    }
    for key, value in final_source_hashes.items():
        if source_provenance.get(key) != value:
            raise RuntimeError(f"MilGNet upstream source changed during execution: {key}")

    final_input_manifest = protocol.input_manifest()
    if final_input_manifest != initial_input_manifest:
        raise RuntimeError("central benchmark inputs changed during the MilGNet run")
    validate_frozen_input_manifest(final_input_manifest)

    result_dir = args.output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        result_dir / "validation_predictions.npz",
        row_id=validation_row_indices,
        pair=validation_pairs,
        label=validation_labels,
        score=validation_scores,
    )
    save_predictions(
        result_dir / "test_predictions.npz",
        row_id=test_row_indices,
        pair=test_pairs,
        label=test_labels,
        score=test_scores,
    )
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        args.output_root / "protocol.json",
        {
            **initial_input_manifest,
            "frozen_input_lock_sha256": frozen_input_lock_sha256,
            "protocol_manifest": str(args.protocol_manifest),
            "dataset": str(protocol.dataset),
            "run_generation_id": args.run_generation_id,
            "legacy_protocol_provenance": legacy_provenance,
            "model": metrics,
            "upstream": source_provenance,
            "adapter_sources": {
                "common.py": sha256(Path(__file__).with_name("common.py")),
                "milgnet_runner.py": sha256(Path(__file__)),
            },
            "runtime_versions": formal_runtime_versions(),
            "runtime_module_origins": formal_runtime_module_origins(
                require_environment=not args.smoke
            ),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "fold": {
                "train_rows_full": int(len(full_train_pairs)),
                "train_rows_executed": int(len(train_pairs)),
                "validation_rows_full": int(len(fold_rows.validation_pairs)),
                "validation_rows_executed": int(len(validation_pairs)),
                "test_rows_full": int(len(fold_rows.test_pairs)),
                "test_rows_executed": int(len(test_pairs)),
                "train_row_indices_sha256": _array_sha256(train_row_indices),
                "full_train_row_indices_sha256": _array_sha256(
                    full_train_row_indices
                ),
                "test_row_indices_sha256": _array_sha256(test_row_indices),
                "train_test_pair_overlap_including_stored_negative_duplicates": _pair_overlap(
                    full_train_pairs, fold_rows.test_pairs
                ),
                "train_positive_validation_or_test_positive_overlap": positive_graph_overlap,
                "legacy_full_outer_training": bool(args.legacy_test_oracle),
                "inner_validation_rows_in_training": bool(args.legacy_test_oracle),
            },
            "similarity": {
                "definition": "Jaccard over unique H-C and P-D incidence sets",
                "cache": str(similarity_cache),
                "cache_sha256": sha256(similarity_cache),
                "cache_validation": (
                    "exact equality to Jaccard matrices recomputed from frozen "
                    "incidence rows"
                ),
                "argpartition_selected_per_entity": args.neighbors,
                "selection": (
                    "literal np.argpartition(row, -k-1)[-k-1:-1] from pinned "
                    "load_data.py; ties are not re-sorted"
                ),
                "self_explicitly_excluded": False,
                "selected_self_rows_herb": int(
                    np.count_nonzero(
                        herb_selected
                        == np.arange(protocol.n_herbs, dtype=np.int64)[:, None]
                    )
                ),
                "selected_self_rows_disease": int(
                    np.count_nonzero(
                        disease_selected
                        == np.arange(protocol.n_diseases, dtype=np.int64)[:, None]
                    )
                ),
                "zero_weight_selected_herb_neighbors": int(
                    np.count_nonzero(herb_selected_weights == 0)
                ),
                "zero_weight_selected_disease_neighbors": int(
                    np.count_nonzero(disease_selected_weights == 0)
                ),
                "topk_entries_set_to_one_in_node_features": True,
            },
            "graph": {
                "edge_counts": graph_edges,
                "association_edges": "unique positive pairs from the training fold only",
                "held_out_positive_edges_present": positive_graph_overlap,
                "side_graph_rule": (
                    "all entries equal to one after top-k-to-one mutation; "
                    "includes diagonal and pre-existing exact-one similarities"
                ),
                "isolated_train_positive_herbs": int(
                    np.count_nonzero(train_positive_degrees_h == 0)
                ),
                "isolated_train_positive_diseases": int(
                    np.count_nonzero(train_positive_degrees_d == 0)
                ),
                "graphconv_zero_in_degree_modules_enabled": zero_degree_modules,
                "zero_degree_policy": (
                    "official GraphConv modules allow zero in-degree; missing H-D "
                    "messages remain zero and no synthetic H-D edge is injected"
                ),
            },
            "bags": {
                "generation": (
                    "exact vectorized on-demand bounded chunks"
                    if args.training_mode == "scalable-full-batch"
                    else "on demand for current train/test batch only"
                ),
                "instances_per_pair": int(max_instances),
                "ordering": "direct, herb-neighbor, disease-neighbor, Cartesian cross",
                "neighbor_source": (
                    "all exact-one side links after official argpartition mutation, "
                    "including diagonal/self and pre-existing exact-one entries"
                ),
                "truncate_or_pad": (
                    "literal upstream cap k*(k+2)+1; prefix truncation and "
                    "random.seed(seed)-reset padding choice"
                ),
                "full_cartesian_herb_disease_dataset_materialized": False,
            },
            "training": {
                "selection": "fixed final epoch; held-out fold never used for model selection",
                "loss": "BCEWithLogitsLoss",
                "positive_weight": negative / positive,
                "epoch_mean_losses": history,
                "epoch_runtime_seconds": epoch_runtime_seconds,
                "optimizer_steps": optimizer_steps,
                "mode": args.training_mode,
                "schedule_name": schedule_name,
                "optimizer_semantics": (
                    (
                        "one deterministic balanced cyclic-block Adam update per "
                        "epoch; decoder loss/gradient accumulated over bounded chunks"
                        if args.scalable_rows_per_epoch
                        else "one exact full-training-objective Adam update per epoch; "
                        "decoder loss/gradient accumulated over bounded pair chunks"
                    )
                    if args.training_mode == "scalable-full-batch"
                    else "released shuffled minibatch Adam trajectory"
                ),
                "graph_encoder_evaluations_per_epoch": (
                    1 if args.training_mode == "scalable-full-batch" else None
                ),
                "decode_chunk_size": (
                    args.decode_chunk_size
                    if args.training_mode == "scalable-full-batch"
                    else None
                ),
                "supervision_rows_per_epoch": actual_rows_per_epoch,
                "positive_rows_per_cyclic_update": (
                    args.scalable_rows_per_epoch // 2
                    if args.scalable_rows_per_epoch
                    else None
                ),
                "negative_rows_per_cyclic_update": (
                    args.scalable_rows_per_epoch // 2
                    if args.scalable_rows_per_epoch
                    else None
                ),
                "total_row_presentations": total_row_presentations,
                "effective_label_passes": total_row_presentations
                / len(full_train_pairs),
                "unique_supervision_rows_covered": unique_supervision_rows_covered,
                "full_outer_row_coverage": bool(
                    not args.smoke
                    and unique_supervision_rows_covered == len(full_train_pairs)
                ),
                "cyclic_balanced_block": bool(args.scalable_rows_per_epoch),
                "cyclic_order": (
                    "separate positive/negative seed-42 permutations; no repeat "
                    "before completing a class cycle; deterministic SeedSequence "
                    "reshuffle at each new cycle"
                    if args.scalable_rows_per_epoch
                    else None
                ),
                "cyclic_full_label_coverage_epochs": (
                    int(
                        np.ceil(
                            max(positive, negative)
                            / (args.scalable_rows_per_epoch // 2)
                        )
                    )
                    if args.scalable_rows_per_epoch
                    else 1
                ),
                "drop_last_requested": args.training_mode == "released-minibatch",
                "drop_last_active": drop_last_active,
                "undersized_single_partial_batch_fallback": (
                    args.training_mode == "released-minibatch" and not drop_last_active
                ),
                "released_minibatch_trajectory_preserved": (
                    args.training_mode == "released-minibatch"
                ),
                "schedule_caveat": (
                    "The scalable cyclic schedule is an audited author-architecture "
                    "adaptation, not the released batch-128 optimizer trajectory: it "
                    "uses fewer repeated label presentations while retaining all "
                    "outer-training rows across deterministic cycles."
                    if args.scalable_rows_per_epoch
                    else None
                ),
            },
            "dataset_specific_adaptations": [
                "The released 10-fold full-Cartesian protocol is replaced by the shared central five-fold positive-plus-frozen-negative rows.",
                "Herb and disease similarities are Jaccard matrices derived from the manifest-locked H-C and P-D incidence relations.",
                "MilGNet's released np.argpartition slice is used literally; ties are not re-sorted, self is not explicitly excluded, and all exact-one links feed bags and side graphs.",
                "Bags retain the released prefix truncation or seeded padding to k*(k+2)+1 instances (441 for k=20).",
                "The H-D graph contains training positives only; no held-out positive edge is used.",
                "Meta-path bags are generated per requested batch instead of materializing every herb-disease candidate.",
                "Official GraphConv zero-in-degree handling is enabled for entities with no training-fold H-D edge, without adding synthetic associations.",
                (
                    "For the explicitly named scalable-full-batch mode, every outer-training row contributes to one exact full-data BCE gradient per epoch; decoder chunks bound memory, the graph encoder is evaluated once, and one Adam update is made per epoch. This preserves the imported architecture and objective but not the released batch-128 optimizer trajectory."
                    if args.training_mode == "scalable-full-batch"
                    and not args.scalable_rows_per_epoch
                    else (
                        "For the explicitly named TCM scalable cyclic schedule, each epoch uses a deterministic balanced block of outer-training supervision; blocks cycle through seeded class permutations, while the structural graph always contains every outer-training positive. This preserves the imported architecture, k=20 bags, 500 Adam updates, frozen folds, and eventual full row coverage, but reduces repeated label exposures relative to the released schedule."
                        if args.training_mode == "scalable-full-batch"
                        else "The released training DataLoader's drop_last=True behaviour is retained whenever at least one full batch exists; an undersized explicit run uses its sole partial batch instead of executing zero updates."
                    )
                ),
                (
                    "Legacy comparison mode restores the inner validation rows to the four outer training folds and uses test labels only for explicitly descriptive threshold diagnostics."
                    if args.legacy_test_oracle
                    else "Validation rows remain excluded from model fitting and select the operating threshold."
                ),
            ],
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "fields": ["row_id", "pair", "label", "score"],
            },
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


if __name__ == "__main__":
    main()
