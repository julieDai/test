#!/usr/bin/env python3
"""Auditable fair-protocol adapter for the commit-pinned official BiGI model.

The released BiGI link-prediction script learns bipartite node embeddings with
the author's GNN and mutual-information objective, then evaluates concatenated
endpoint embeddings with an L2-logistic probe.  This adapter imports those
author modules unchanged and supplies the central ASTRA fold rows.

Formal runs use the released 100-epoch schedule and select a checkpoint with
the frozen inner-validation rows.  Smoke runs execute one tiny update only.
In both modes:

* the message-passing graph contains only inner-training positive H--D edges;
  validation and test positives never enter the graph;
* supervised negative examples are only the frozen inner-training negative
  rows (the released random per-user negative sampler is not used);
* the author BiGI encoder and DGI loss are optimized on frozen training rows;
* the author-style logistic probe is fit only to frozen training rows;
* checkpoint and threshold selection use validation labels only; and
* validation/test artifacts retain row_id, pair, label, and continuous score.

The frozen-negative pointwise BCE batching, validation checkpoint rule, and
modern runtime are recorded as compatibility adaptations in ``protocol.json``.
They do not claim exact reproduction of the Python-3.6/PyTorch-1.1 optimizer
trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import types
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve

try:
    from .common import (
        DEFAULT_THIRD_PARTY_DIR,
        array_sha256,
        balanced_prefix_indices,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        load_protocol,
        project_git_identity,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )
except ImportError:  # Direct script execution with the package on PYTHONPATH.
    from common import (
        DEFAULT_THIRD_PARTY_DIR,
        array_sha256,
        balanced_prefix_indices,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        load_protocol,
        project_git_identity,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )

try:
    from ..benchmark.metrics import evaluate_validation_and_test, save_predictions
except ImportError:  # Direct script execution with ``src`` on PYTHONPATH.
    from astra.benchmark.metrics import evaluate_validation_and_test, save_predictions


UPSTREAM_COMMIT = "ed54c20523a5b3f295b90a9c08f7c54e8258d04a"
UPSTREAM_SOURCE = "https://github.com/caojiangxia/BiGI"
OFFICIAL_PROBE_C = 0.001
OFFICIAL_PROBE_MAX_ITER = 500
# CUDA sparse reductions can differ by a few float32 rounding units between
# otherwise identical encoder evaluations. Checkpoint integrity is verified
# separately and exactly; this tolerance is only for continuous validation
# probabilities produced by the already-frozen logistic probe.
CHECKPOINT_SCORE_ATOL = 1e-6
CHECKPOINT_SCORE_RTOL = 1e-5
FORMAL_SCHEDULE = {
    "epochs": 100,
    "batch_size": 128,
    "feature_dim": 128,
    "hidden_dim": 128,
    "gnn_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "leaky_relu": 0.1,
    "dgi_weight": 0.1,
    "structural_corruption_rate": 1e-4,
    "min_neighbors": 100,
}


@dataclass(frozen=True)
class BipartiteGraphs:
    real_pairs: np.ndarray
    corrupted_pairs: np.ndarray
    real_uv: object
    real_vu: object
    corrupted_uv: object
    corrupted_vu: object
    real_user_neighbors: tuple[tuple[int, ...], ...]
    real_item_neighbors: tuple[tuple[int, ...], ...]
    corrupted_user_neighbors: tuple[tuple[int, ...], ...]
    corrupted_item_neighbors: tuple[tuple[int, ...], ...]


def _load_source_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_official_model(source_dir: Path):
    """Load author ``BiGI`` from source files, never from bundled old pyc files."""

    model_dir = source_dir / "BiGI_src" / "model"
    required = ("GCN", "GAT", "GNN", "GNN2", "AttDGI", "myDGI", "BiGI")
    missing = [name for name in required if not (model_dir / f"{name}.py").is_file()]
    if missing:
        raise FileNotFoundError(
            f"BiGI checkout is missing author model modules: {', '.join(missing)}"
        )

    previous = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "model" or name.startswith("model.")
    }
    for name in previous:
        sys.modules.pop(name, None)

    package = types.ModuleType("model")
    package.__path__ = [str(model_dir)]
    package.__package__ = "model"
    sys.modules["model"] = package
    previous_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        with tempfile.TemporaryDirectory(prefix="herbhg_bigi_pycache_") as pycache:
            sys.pycache_prefix = pycache
            sys.dont_write_bytecode = True
            for name in required:
                _load_source_module(f"model.{name}", model_dir / f"{name}.py")
            official = sys.modules["model.BiGI"].BiGI
    finally:
        for name in tuple(sys.modules):
            if name == "model" or name.startswith("model."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        sys.pycache_prefix = previous_prefix
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return official


def _positive_graph_pairs(pairs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise ValueError("training pairs/labels must have aligned shape (n, 2)/(n,)")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("training labels must be binary")
    positive = pairs[labels == 1]
    if not len(positive):
        raise ValueError("BiGI training graph requires positive training edges")
    return np.unique(positive, axis=0)


def _pair_keys(pairs: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pair)) for pair in np.asarray(pairs, dtype=np.int64)}


def _assert_no_evaluation_edges(
    graph_pairs: np.ndarray,
    validation_pairs: np.ndarray,
    test_pairs: np.ndarray,
) -> None:
    """Reject any graph overlap with held-out rows without reading their labels."""

    graph = _pair_keys(graph_pairs)
    leaked_validation = graph.intersection(_pair_keys(validation_pairs))
    leaked_test = graph.intersection(_pair_keys(test_pairs))
    if leaked_validation or leaked_test:
        raise RuntimeError(
            "BiGI training graph contains held-out evaluation pairs: "
            f"validation={len(leaked_validation)}, test={len(leaked_test)}"
        )


def _corrupt_pairs(
    pairs: np.ndarray,
    n_users: int,
    n_items: int,
    rate: float,
    seed: int,
) -> np.ndarray:
    """Reproduce author edge-toggle corruption with a local deterministic RNG."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("structural corruption rate must lie in [0, 1]")
    edges = _pair_keys(pairs)
    rng = random.Random(int(seed))
    for _ in range(int(n_users * n_items * rate) + 1):
        pair = (rng.randint(0, n_users - 1), rng.randint(0, n_items - 1))
        if pair in edges:
            edges.remove(pair)
        else:
            edges.add(pair)
    if not edges:
        raise RuntimeError("structural corruption removed every training edge")
    return np.asarray(sorted(edges), dtype=np.int64)


def _neighbors(
    pairs: np.ndarray, n_users: int, n_items: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    user: list[set[int]] = [set() for _ in range(n_users)]
    item: list[set[int]] = [set() for _ in range(n_items)]
    for raw_user, raw_item in np.asarray(pairs, dtype=np.int64):
        user_index = int(raw_user)
        item_index = int(raw_item)
        user[user_index].add(item_index)
        item[item_index].add(user_index)
    return (
        tuple(tuple(sorted(values)) for values in user),
        tuple(tuple(sorted(values)) for values in item),
    )


def _torch_row_normalized_adjacencies(
    pairs: np.ndarray, n_users: int, n_items: int, device
):
    import torch

    pairs = np.asarray(pairs, dtype=np.int64)
    values = np.ones(len(pairs), dtype=np.float32)
    uv = sparse.coo_matrix(
        (values, (pairs[:, 0], pairs[:, 1])), shape=(n_users, n_items)
    ).tocsr()
    uv.sum_duplicates()
    uv.data.fill(np.float32(1.0))
    vu = uv.transpose().tocsr()

    def normalize(matrix):
        degree = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
        inverse = np.divide(
            np.float32(1.0),
            degree,
            out=np.zeros_like(degree, dtype=np.float32),
            where=degree != 0,
        )
        return sparse.diags(inverse).dot(matrix).tocoo().astype(np.float32)

    def as_torch(matrix):
        normalized = normalize(matrix)
        indices = np.vstack((normalized.row, normalized.col)).astype(np.int64)
        tensor = torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(normalized.data),
            size=normalized.shape,
            dtype=torch.float32,
        )
        return tensor.coalesce().to(device)

    return as_torch(uv), as_torch(vu)


def _build_graphs(
    pairs: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    structural_corruption_rate: float,
    seed: int,
    device,
) -> BipartiteGraphs:
    real = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
    if np.any(real < 0) or np.any(real[:, 0] >= n_users) or np.any(real[:, 1] >= n_items):
        raise ValueError("BiGI graph pairs are outside the protocol entity ranges")
    corrupted = _corrupt_pairs(
        real, n_users, n_items, structural_corruption_rate, seed
    )
    real_uv, real_vu = _torch_row_normalized_adjacencies(
        real, n_users, n_items, device
    )
    corrupted_uv, corrupted_vu = _torch_row_normalized_adjacencies(
        corrupted, n_users, n_items, device
    )
    real_user, real_item = _neighbors(real, n_users, n_items)
    corrupted_user, corrupted_item = _neighbors(corrupted, n_users, n_items)
    return BipartiteGraphs(
        real_pairs=real,
        corrupted_pairs=corrupted,
        real_uv=real_uv,
        real_vu=real_vu,
        corrupted_uv=corrupted_uv,
        corrupted_vu=corrupted_vu,
        real_user_neighbors=real_user,
        real_item_neighbors=real_item,
        corrupted_user_neighbors=corrupted_user,
        corrupted_item_neighbors=corrupted_item,
    )


def _pair_features(torch, user_embeddings, item_embeddings, pairs: np.ndarray):
    indices = torch.as_tensor(pairs, dtype=torch.long, device=user_embeddings.device)
    return torch.cat(
        (user_embeddings[indices[:, 0]], item_embeddings[indices[:, 1]]), dim=1
    )


def _encode_real(model, graphs: BipartiteGraphs):
    user = model.user_embedding(model.user_index)
    item = model.item_embedding(model.item_index)
    return model(user, item, graphs.real_uv, graphs.real_vu, None)


def _encode_real_and_corrupted(model, graphs: BipartiteGraphs, torch):
    fake_user_index = torch.randperm(
        model.opt["number_user"], device=model.user_index.device
    )
    fake_item_index = torch.randperm(
        model.opt["number_item"], device=model.item_index.device
    )
    fake_user_input = model.user_embedding(fake_user_index).detach()
    fake_item_input = model.item_embedding(fake_item_index).detach()
    fake_user, fake_item = model(
        fake_user_input,
        fake_item_input,
        graphs.corrupted_uv,
        graphs.corrupted_vu,
        None,
    )
    real_user, real_item = _encode_real(model, graphs)
    return real_user, real_item, fake_user, fake_item


def _att_dgi_loss(
    model,
    graphs: BipartiteGraphs,
    real_user,
    real_item,
    fake_user,
    fake_item,
    *,
    min_neighbors: int,
    seed: int,
    torch,
):
    eligible = [
        (int(user), int(item))
        for user, item in graphs.real_pairs
        if len(graphs.real_user_neighbors[int(user)]) > min_neighbors
        and len(graphs.corrupted_user_neighbors[int(user)]) > min_neighbors
        and len(graphs.real_item_neighbors[int(item)]) > min_neighbors
        and len(graphs.corrupted_item_neighbors[int(item)]) > min_neighbors
    ]
    if len(eligible) < 10:
        raise RuntimeError(
            "official AttDGI path requires at least 10 training edges whose four "
            f"endpoint degrees exceed min_neighbors={min_neighbors}; found {len(eligible)}"
        )
    rng = np.random.default_rng(seed)
    chosen = np.asarray(eligible, dtype=np.int64)[rng.permutation(len(eligible))[:10]]
    users = torch.as_tensor(chosen[:, 0], dtype=torch.long, device=real_user.device)
    items = torch.as_tensor(chosen[:, 1], dtype=torch.long, device=real_user.device)

    real_item_ids = []
    real_user_ids = []
    for user, item in chosen:
        item_neighbors = np.asarray(
            graphs.real_user_neighbors[int(user)], dtype=np.int64
        )
        user_neighbors = np.asarray(
            graphs.real_item_neighbors[int(item)], dtype=np.int64
        )
        real_item_ids.append(rng.permutation(item_neighbors)[:min_neighbors])
        real_user_ids.append(rng.permutation(user_neighbors)[:min_neighbors])
    real_item_ids_tensor = torch.as_tensor(
        np.asarray(real_item_ids), dtype=torch.long, device=real_user.device
    )
    real_user_ids_tensor = torch.as_tensor(
        np.asarray(real_user_ids), dtype=torch.long, device=real_user.device
    )

    # The released loader overwrites the sampled corrupted-neighbour IDs with
    # the corresponding real-neighbour IDs.  Retaining that observable source
    # behavior is more faithful than silently fixing it in the adapter.
    probability, labels = model.DGI(
        real_user,
        real_item,
        real_user[users],
        real_item[items],
        fake_user[users],
        fake_item[items],
        real_item[real_item_ids_tensor],
        real_user[real_user_ids_tensor],
        fake_item[real_item_ids_tensor],
        fake_user[real_user_ids_tensor],
    )
    return torch.nn.functional.binary_cross_entropy(probability, labels)


def _one_smoke_update(
    model,
    optimizer,
    graphs: BipartiteGraphs,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    *,
    dgi_weight: float,
    min_neighbors: int,
    seed: int,
    torch,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    real_user, real_item, fake_user, fake_item = _encode_real_and_corrupted(
        model, graphs, torch
    )
    pair_feature = _pair_features(torch, real_user, real_item, train_pairs)
    probability = model.score(pair_feature)
    labels = torch.as_tensor(
        train_labels, dtype=torch.float32, device=probability.device
    )
    reconstruction = torch.nn.functional.binary_cross_entropy(probability, labels)

    if model.opt["number_user"] * model.opt["number_item"] > 10_000_000:
        dgi = _att_dgi_loss(
            model,
            graphs,
            real_user,
            real_item,
            fake_user,
            fake_item,
            min_neighbors=min_neighbors,
            seed=seed,
            torch=torch,
        )
    else:
        positive = np.asarray(train_pairs)[np.asarray(train_labels) == 1]
        users = torch.as_tensor(
            positive[:, 0], dtype=torch.long, device=real_user.device
        )
        items = torch.as_tensor(
            positive[:, 1], dtype=torch.long, device=real_user.device
        )
        probability_dgi, labels_dgi = model.DGI(
            real_user,
            real_item,
            fake_user,
            fake_item,
            graphs.real_uv,
            graphs.real_vu,
            graphs.corrupted_uv,
            graphs.corrupted_vu,
            users,
            items,
        )
        dgi = torch.nn.functional.binary_cross_entropy(
            probability_dgi, labels_dgi
        )
    loss = (1.0 - dgi_weight) * reconstruction + dgi_weight * dgi
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction.detach().cpu()),
        "dgi_loss": float(dgi.detach().cpu()),
    }


def _balanced_epoch_batches(
    labels: np.ndarray, *, batch_size: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Return deterministic class-balanced batches that cover every row once."""

    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if batch_size < 2:
        raise ValueError("BiGI batch_size must be at least two")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("BiGI training labels must be binary")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("BiGI training requires both positive and negative rows")

    rng = np.random.default_rng(int(seed))
    positive = positive[rng.permutation(len(positive))]
    negative = negative[rng.permutation(len(negative))]
    positive_batch = max(1, batch_size // 2)
    negative_batch = max(1, batch_size - positive_batch)
    batches: list[np.ndarray] = []
    positive_offset = 0
    negative_offset = 0
    while positive_offset < len(positive) or negative_offset < len(negative):
        selected = np.concatenate(
            (
                positive[positive_offset : positive_offset + positive_batch],
                negative[negative_offset : negative_offset + negative_batch],
            )
        )
        positive_offset += positive_batch
        negative_offset += negative_batch
        if len(selected):
            batches.append(selected[rng.permutation(len(selected))])
    covered = np.concatenate(batches)
    if len(covered) != len(labels) or len(np.unique(covered)) != len(labels):
        raise RuntimeError("BiGI epoch batching did not cover each frozen row once")
    return tuple(batches)


def _train_epoch(
    model,
    optimizer,
    graphs: BipartiteGraphs,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    *,
    batch_size: int,
    dgi_weight: float,
    min_neighbors: int,
    seed: int,
    torch,
) -> tuple[dict[str, float], int]:
    """Run one deterministic full pass over the frozen inner-training rows."""

    records: list[dict[str, float]] = []
    batches = _balanced_epoch_batches(
        train_labels, batch_size=batch_size, seed=seed
    )
    for step, selected in enumerate(batches):
        records.append(
            _one_smoke_update(
                model,
                optimizer,
                graphs,
                np.asarray(train_pairs)[selected],
                np.asarray(train_labels)[selected],
                dgi_weight=dgi_weight,
                min_neighbors=min_neighbors,
                seed=int(seed) * 1_000_003 + step,
                torch=torch,
            )
        )
    means = {
        key: float(np.mean([record[key] for record in records]))
        for key in ("loss", "reconstruction_loss", "dgi_loss")
    }
    return means, len(batches)


def _validation_probe_scores(
    model,
    graphs: BipartiteGraphs,
    train_pairs: np.ndarray,
    train_labels: np.ndarray,
    validation_pairs: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    torch,
) -> tuple[LogisticRegression, np.ndarray, float]:
    train_features = _numpy_pair_features(model, graphs, train_pairs, torch)
    validation_features = _numpy_pair_features(
        model, graphs, validation_pairs, torch
    )
    probe = _fit_author_probe(train_features, train_labels, seed=seed)
    scores = probe.predict_proba(validation_features)[:, 1]
    return probe, scores, _trapezoidal_pr_auc(validation_labels, scores)


def _numpy_pair_features(
    model, graphs: BipartiteGraphs, pairs: np.ndarray, torch
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user, item = _encode_real(model, graphs)
        feature = _pair_features(torch, user, item, pairs)
    return feature.detach().cpu().numpy().astype(np.float64, copy=False)


def _fit_author_probe(
    train_features: np.ndarray, train_labels: np.ndarray, *, seed: int
) -> LogisticRegression:
    probe = LogisticRegression(
        penalty="l2",
        C=OFFICIAL_PROBE_C,
        max_iter=OFFICIAL_PROBE_MAX_ITER,
        random_state=seed,
    )
    probe.fit(train_features, np.asarray(train_labels, dtype=np.int64))
    return probe


def _trapezoidal_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def _assert_loaded_state_matches(model, expected_state: dict[str, object], torch) -> None:
    """Verify that ``load_state_dict`` restored every serialized tensor exactly."""

    observed_state = model.state_dict()
    if set(observed_state) != set(expected_state):
        missing = sorted(set(expected_state).difference(observed_state))
        extra = sorted(set(observed_state).difference(expected_state))
        raise RuntimeError(
            "reloaded BiGI checkpoint state keys changed: "
            f"missing={missing}, extra={extra}"
        )
    changed = [
        name
        for name, expected in expected_state.items()
        if not torch.equal(observed_state[name].detach().cpu(), expected)
    ]
    if changed:
        raise RuntimeError(
            "reloaded BiGI checkpoint changed serialized tensors: "
            + ", ".join(changed)
        )


def _validation_score_reload_audit(
    selected_scores: np.ndarray,
    reloaded_scores: np.ndarray,
    *,
    atol: float = CHECKPOINT_SCORE_ATOL,
    rtol: float = CHECKPOINT_SCORE_RTOL,
) -> dict[str, float | int]:
    """Audit numerical inference drift without repeating model selection.

    The selected scores are the validation predictions frozen at the winning
    epoch. They remain authoritative for checkpoint and threshold selection.
    Reloaded scores use the same fitted probe and only check that restoring the
    encoder did not cause a material prediction change.
    """

    selected = np.asarray(selected_scores, dtype=np.float64).reshape(-1)
    reloaded = np.asarray(reloaded_scores, dtype=np.float64).reshape(-1)
    if selected.shape != reloaded.shape:
        raise RuntimeError(
            "reloaded BiGI checkpoint changed validation score shape: "
            f"{reloaded.shape} != {selected.shape}"
        )
    if not np.all(np.isfinite(selected)) or not np.all(np.isfinite(reloaded)):
        raise RuntimeError("BiGI checkpoint validation scores contain NaN or Inf")
    absolute = np.abs(reloaded - selected)
    scale = np.maximum(np.abs(selected), np.finfo(np.float64).tiny)
    relative = absolute / scale
    close = np.isclose(reloaded, selected, atol=atol, rtol=rtol)
    audit: dict[str, float | int] = {
        "max_absolute_drift": float(absolute.max(initial=0.0)),
        "max_relative_drift": float(relative.max(initial=0.0)),
        "scores_outside_tolerance": int(np.count_nonzero(~close)),
        "score_count": int(len(selected)),
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
    }
    if not np.all(close):
        raise RuntimeError(
            "reloaded BiGI checkpoint materially changed validation scores: "
            f"max_abs={audit['max_absolute_drift']:.9g}, "
            f"max_rel={audit['max_relative_drift']:.9g}, "
            f"outside={audit['scores_outside_tolerance']}/{audit['score_count']}, "
            f"atol={atol:g}, rtol={rtol:g}"
        )
    return audit


def _save_training_provenance(
    path: Path,
    *,
    row_id: np.ndarray,
    pair: np.ndarray,
    label: np.ndarray,
    graph_pair: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        row_id=np.asarray(row_id),
        pair=np.asarray(pair, dtype=np.int64),
        label=np.asarray(label, dtype=np.uint8),
        graph_pair=np.asarray(graph_pair, dtype=np.int64),
    )
    temporary.replace(path)


def _device(requested: str, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _revalidate_protocol_inputs(
    protocol_manifest: Path,
    *,
    expected_manifest_sha256: str,
    expected_input_manifest: dict[str, object],
    expected_input_lock_sha256: str,
) -> None:
    """Reload the frozen package and fail if on-disk inputs changed mid-run."""

    observed_manifest_sha256 = sha256(protocol_manifest)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "central protocol manifest changed during the BiGI run: "
            f"{observed_manifest_sha256} != {expected_manifest_sha256}"
        )
    reloaded_protocol = load_protocol(protocol_manifest)
    observed_input_manifest = reloaded_protocol.input_manifest()
    observed_input_lock_sha256 = validate_frozen_input_manifest(
        observed_input_manifest
    )
    if observed_input_manifest != expected_input_manifest:
        raise RuntimeError("central benchmark inputs changed during the BiGI run")
    if observed_input_lock_sha256 != expected_input_lock_sha256:
        raise RuntimeError(
            "central benchmark input lock changed during the BiGI run: "
            f"{observed_input_lock_sha256} != {expected_input_lock_sha256}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_THIRD_PARTY_DIR / "BiGI_official",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--leaky-relu", type=float, default=0.1)
    parser.add_argument("--dgi-weight", type=float, default=0.1)
    parser.add_argument("--structural-corruption-rate", type=float, default=1e-4)
    parser.add_argument("--min-neighbors", type=int, default=100)
    parser.add_argument("--smoke-rows-per-class", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-generation-id")
    args = parser.parse_args()

    if args.feature_dim != args.hidden_dim:
        parser.error("released BiGI GNN requires --feature-dim == --hidden-dim")
    if args.smoke_rows_per_class < 1 or args.batch_size < 2:
        parser.error("smoke rows per class and batch size must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0.0 <= args.dgi_weight <= 1.0:
        parser.error("--dgi-weight must lie in [0, 1]")
    if not args.smoke:
        mismatches = [
            f"{name}={getattr(args, name)!r} (required {expected!r})"
            for name, expected in FORMAL_SCHEDULE.items()
            if getattr(args, name) != expected
        ]
        if mismatches:
            parser.error(
                "formal BiGI runs lock the predeclared released schedule: "
                + "; ".join(mismatches)
            )

    started = time.time()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.source_dir = args.source_dir.resolve()
    project_source_identity = project_git_identity(require_clean=not args.smoke)
    runtime_version_identity = formal_runtime_versions(
        require_expected=not args.smoke
    )
    runtime_module_identity = formal_runtime_module_origins(
        require_environment=not args.smoke
    )
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    model_path = args.source_dir / "BiGI_src" / "model" / "BiGI.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"BiGI checkout missing at {args.source_dir}; run third_party/fetch_baselines.sh"
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(args.device, torch)

    initial_protocol_manifest_sha256 = sha256(args.protocol_manifest)
    protocol = load_protocol(args.protocol_manifest)
    if args.seed != int(protocol.seed):
        parser.error(
            f"--seed {args.seed} differs from frozen protocol seed {protocol.seed}"
        )
    initial_input_manifest = protocol.input_manifest()
    frozen_input_lock_sha256 = validate_frozen_input_manifest(initial_input_manifest)
    fold_rows = protocol.fold_rows(args.fold)
    train_pairs_full, train_labels_full, train_row_id_full = fold_split(
        fold_rows, "train"
    )
    validation_pairs_full, validation_labels_full, validation_row_id_full = fold_split(
        fold_rows, "validation"
    )
    # Only pair identities are needed for the pre-training leakage guard.  Do
    # not materialize outer-test labels in adapter code until model/checkpoint
    # selection has finished.
    test_pairs_for_guard = np.asarray(fold_rows.test_pairs, dtype=np.int64)

    graph_pairs = _positive_graph_pairs(train_pairs_full, train_labels_full)
    _assert_no_evaluation_edges(
        graph_pairs,
        validation_pairs_full,
        test_pairs_for_guard,
    )
    if args.smoke:
        train_selection = balanced_prefix_indices(
            train_labels_full, args.smoke_rows_per_class
        )
        validation_selection = balanced_prefix_indices(
            validation_labels_full, args.smoke_rows_per_class
        )
        train_pairs = train_pairs_full[train_selection]
        train_labels = train_labels_full[train_selection]
        train_row_id = train_row_id_full[train_selection]
        validation_pairs = validation_pairs_full[validation_selection]
        validation_labels = validation_labels_full[validation_selection]
        validation_row_id = validation_row_id_full[validation_selection]
    else:
        train_pairs = train_pairs_full
        train_labels = train_labels_full
        train_row_id = train_row_id_full
        validation_pairs = validation_pairs_full
        validation_labels = validation_labels_full
        validation_row_id = validation_row_id_full
    frozen_train_negatives = _pair_keys(train_pairs_full[train_labels_full == 0])
    selected_train_negatives = _pair_keys(train_pairs[train_labels == 0])
    if not selected_train_negatives.issubset(frozen_train_negatives):
        raise RuntimeError("BiGI training selected a negative outside frozen train rows")

    graphs = _build_graphs(
        graph_pairs,
        protocol.n_herbs,
        protocol.n_diseases,
        structural_corruption_rate=args.structural_corruption_rate,
        seed=args.seed,
        device=device,
    )
    OfficialBiGI = _load_official_model(args.source_dir)
    options = {
        "number_user": int(protocol.n_herbs),
        "number_item": int(protocol.n_diseases),
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "GNN": args.gnn_layers,
        "dropout": args.dropout,
        "leakey": args.leaky_relu,
        "cuda": False,
        "margin": 0.3,
        "lambda": args.dgi_weight,
    }
    model = OfficialBiGI(options).to(device)
    # Author code creates these as plain tensors rather than registered buffers.
    # Rebind them to the explicitly requested modern torch device.
    model.user_index = torch.arange(protocol.n_herbs, device=device)
    model.item_index = torch.arange(protocol.n_diseases, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epochs_executed = 1 if args.smoke else args.epochs
    checkpoint_history: list[dict[str, object]] = []
    best_state = None
    best_probe = None
    best_validation_scores = None
    best_epoch = 0
    checkpoint_validation_auprc = -float("inf")
    optimizer_steps = 0
    update_started = time.perf_counter()
    for epoch in range(1, epochs_executed + 1):
        loss_record, epoch_steps = _train_epoch(
            model,
            optimizer,
            graphs,
            train_pairs,
            train_labels,
            batch_size=args.batch_size,
            dgi_weight=args.dgi_weight,
            min_neighbors=args.min_neighbors,
            seed=args.seed + epoch,
            torch=torch,
        )
        optimizer_steps += epoch_steps
        epoch_probe, epoch_validation_scores, epoch_validation_auprc = (
            _validation_probe_scores(
                model,
                graphs,
                train_pairs,
                train_labels,
                validation_pairs,
                validation_labels,
                seed=args.seed,
                torch=torch,
            )
        )
        checkpoint_history.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "validation_auprc": epoch_validation_auprc,
                **loss_record,
            }
        )
        if epoch_validation_auprc > checkpoint_validation_auprc:
            checkpoint_validation_auprc = epoch_validation_auprc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            # The probe and predictions are part of the selected checkpoint.
            # Freezing them here prevents a later probe refit or a numerically
            # different CUDA sparse reduction from silently repeating model
            # selection after training has ended.
            best_probe = epoch_probe
            best_validation_scores = np.asarray(
                epoch_validation_scores, dtype=np.float64
            ).copy()
    optimizer_step_seconds = time.perf_counter() - update_started
    if best_state is None or best_probe is None or best_validation_scores is None:
        raise RuntimeError("BiGI did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    _assert_loaded_state_matches(model, best_state, torch)
    reloaded_validation_features = _numpy_pair_features(
        model, graphs, validation_pairs, torch
    )
    reloaded_validation_scores = best_probe.predict_proba(
        reloaded_validation_features
    )[:, 1]
    checkpoint_reload_audit = _validation_score_reload_audit(
        best_validation_scores, reloaded_validation_scores
    )
    checkpoint_reload_auprc = _trapezoidal_pr_auc(
        validation_labels, reloaded_validation_scores
    )
    # Checkpoint and threshold selection use exactly the probe and validation
    # scores frozen at the selected epoch. The reload pass above is an audit,
    # not a second model-selection event.
    probe = best_probe
    validation_scores = best_validation_scores
    test_pairs_full, test_labels_full, test_row_id_full = fold_split(
        fold_rows, "test"
    )
    # The outer-test labels are first inspected only after the checkpoint and
    # validation-selected probe have been frozen.  Label-aware subsetting is a
    # smoke-only diagnostic and cannot enter a formal aggregate.
    if args.smoke:
        test_selection = balanced_prefix_indices(
            test_labels_full, args.smoke_rows_per_class
        )
        test_pairs = test_pairs_full[test_selection]
        test_labels = test_labels_full[test_selection]
        test_row_id = test_row_id_full[test_selection]
    else:
        test_pairs = test_pairs_full
        test_labels = test_labels_full
        test_row_id = test_row_id_full
    test_features = _numpy_pair_features(model, graphs, test_pairs, torch)
    test_scores = probe.predict_proba(test_features)[:, 1]
    metrics = evaluate_validation_and_test(
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
    )
    metrics.update(
        {
            "Model": "BiGI",
            "Fold": args.fold,
            "Dataset": str(protocol.dataset),
            "Seed": args.seed,
            "Device": str(device),
            "Runtime_Seconds": time.time() - started,
            "Epochs_Requested": args.epochs,
            "Epochs_Executed": epochs_executed,
            "Optimizer_Steps": optimizer_steps,
            "Optimization_And_Validation_Seconds": optimizer_step_seconds,
            "Upstream_Commit": UPSTREAM_COMMIT,
            "Upstream_Source": UPSTREAM_SOURCE,
            "Protocol_Manifest_SHA256": initial_protocol_manifest_sha256,
            "Smoke": bool(args.smoke),
            "Formal_Eligible": bool(not args.smoke),
            "Evaluation_Mode": "validation",
            "Training_Rows": int(len(train_pairs)),
            "Training_Positive_Rows": int(np.count_nonzero(train_labels == 1)),
            "Training_Negative_Rows": int(np.count_nonzero(train_labels == 0)),
            "Training_Graph_Positive_Edges": int(len(graph_pairs)),
            "Checkpoint_Source": "inner-validation trapezoidal AUPRC",
            "Checkpoint_Epoch": best_epoch,
            "Checkpoint_Validation_AUPRC": checkpoint_validation_auprc,
            "Checkpoint_Reload_Validation_AUPRC": checkpoint_reload_auprc,
            "Checkpoint_Reload_Max_Absolute_Score_Drift": (
                checkpoint_reload_audit["max_absolute_drift"]
            ),
            "Checkpoint_Reload_Max_Relative_Score_Drift": (
                checkpoint_reload_audit["max_relative_drift"]
            ),
            "Probe": (
                "author link_prediction_logistic semantics: L2 LogisticRegression, "
                "C=0.001, max_iter=500"
            ),
            "Protocol_Kind": (
                "commit-pinned author BiGI encoder/DGI with frozen-negative "
                "pointwise BCE and inner-validation checkpoint selection"
            ),
            "Run_Parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )

    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    _revalidate_protocol_inputs(
        args.protocol_manifest,
        expected_manifest_sha256=initial_protocol_manifest_sha256,
        expected_input_manifest=initial_input_manifest,
        expected_input_lock_sha256=frozen_input_lock_sha256,
    )

    result_dir = args.output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        result_dir / "validation_predictions.npz",
        row_id=validation_row_id,
        pair=validation_pairs,
        label=validation_labels,
        score=validation_scores,
    )
    save_predictions(
        result_dir / "test_predictions.npz",
        row_id=test_row_id,
        pair=test_pairs,
        label=test_labels,
        score=test_scores,
    )
    _save_training_provenance(
        result_dir / "training_rows_and_graph.npz",
        row_id=train_row_id,
        pair=train_pairs,
        label=train_labels,
        graph_pair=graph_pairs,
    )
    write_json(result_dir / "metrics.json", metrics)
    author_sources = {
        f"BiGI_src/model/{name}.py": sha256(
            args.source_dir / "BiGI_src" / "model" / f"{name}.py"
        )
        for name in ("GCN", "GAT", "GNN", "GNN2", "AttDGI", "myDGI", "BiGI")
    }
    training_negative_row_id = train_row_id[train_labels == 0]
    write_json(
        args.output_root / "protocol.json",
        {
            "protocol_input_manifest": initial_input_manifest,
            "frozen_input_lock_sha256": frozen_input_lock_sha256,
            "protocol_manifest": str(args.protocol_manifest),
            "protocol_manifest_sha256": initial_protocol_manifest_sha256,
            "dataset": str(protocol.dataset),
            "run_generation_id": args.run_generation_id,
            "model": metrics,
            "adapter_sources": {
                "common.py": sha256(Path(__file__).with_name("common.py")),
                "bigi_runner.py": sha256(Path(__file__)),
            },
            "upstream": {
                "repository": UPSTREAM_SOURCE,
                "commit": UPSTREAM_COMMIT,
                "source_sha256": author_sources,
            },
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "training": {
                "schedule": (
                    "one smoke epoch; not formal"
                    if args.smoke
                    else (
                        f"released fixed {args.epochs}-epoch budget; deterministic "
                        "class-balanced passes over every frozen inner-train row"
                    )
                ),
                "selected_checkpoint_loss": checkpoint_history[best_epoch - 1],
                "checkpoint_history": checkpoint_history,
                "optimization_and_validation_seconds": optimizer_step_seconds,
                "graph_source": "central inner-training positive rows only",
                "validation_or_test_edges_in_graph": False,
                "supervised_negative_source": "frozen central inner-training negative rows only",
                "training_row_id_sha256": array_sha256(np.asarray(train_row_id)),
                "training_negative_row_id_sha256": array_sha256(
                    np.asarray(training_negative_row_id)
                ),
                "training_graph_pair_sha256": array_sha256(graph_pairs),
                "corrupted_graph_pair_sha256": array_sha256(graphs.corrupted_pairs),
            },
            "checkpoint_selection": {
                "split": "validation",
                "metric": "trapezoidal AUPRC",
                "outer_test_labels_used": False,
                "candidate_checkpoints": epochs_executed,
                "selected_epoch": best_epoch,
                "selected_update": int(
                    checkpoint_history[best_epoch - 1]["optimizer_steps"]
                ),
                "validation_auprc": checkpoint_validation_auprc,
                "validation_scores_source": (
                    "probe and predictions frozen at selected epoch"
                ),
                "checkpoint_reload_validation_auprc": checkpoint_reload_auprc,
                "checkpoint_reload_score_audit": checkpoint_reload_audit,
            },
            "compatibility_adaptations": [
                (
                    "loaded the released BiGI/GNN/DGI source modules under modern "
                    "Python/PyTorch without executing bundled Python-3.6/3.7 bytecode"
                ),
                (
                    "rebound the author's unregistered user_index/item_index tensors "
                    "to the explicitly requested torch device"
                ),
                (
                    "replaced released random per-user negatives with pointwise BCE "
                    "over frozen manifest negative rows; each formal epoch consumes "
                    "every frozen training row once and performs no complement sampling"
                ),
                (
                    "constructed the message-passing graph only from inner-training "
                    "positive rows; validation and test positives were excluded"
                ),
                (
                    "used a local deterministic edge-toggle RNG and membership test "
                    "instead of the released Python string-identity comparison"
                ),
                (
                    "used device-safe BCE labels; released wiki code hard-codes .cuda()"
                ),
                (
                    "retained the released AttDGI loader behavior that reuses real "
                    "neighbour IDs for the corrupted-neighbour tensors"
                ),
                (
                    "fit the released L2 C=0.001 logistic link probe on frozen train "
                    "rows; selected encoder checkpoint and operating threshold only "
                    "on inner-validation rows"
                ),
                (
                    "froze the fitted logistic probe and validation predictions with "
                    "the selected encoder checkpoint; exact state restoration and a "
                    "bounded continuous-score reload audit prevent CUDA sparse "
                    "rounding from repeating checkpoint selection"
                ),
            ],
            "trajectory_statement": (
                "author architecture/objectives with documented protocol and runtime "
                "adaptations; not an exact Python-3.6/PyTorch-1.1 optimizer trajectory"
            ),
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "training_provenance": "results/training_rows_and_graph.npz",
                "prediction_fields": ["row_id", "pair", "label", "score"],
            },
            "project_source_identity": project_source_identity,
            "runtime_versions": runtime_version_identity,
            "runtime_module_origins": runtime_module_identity,
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": ":16:8",
            },
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
