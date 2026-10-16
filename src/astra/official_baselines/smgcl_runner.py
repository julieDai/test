#!/usr/bin/env python3
"""Run the authors' SMGCL modules in an auditable SMGCL-NS-style adapter.

The released repository cannot be executed unchanged: its advertised demo
imports a class that does not exist, its bipartite graph is constructed from
the current query batch (including negatives), and several temporary tensors
are tied to CPU.  This adapter imports the released ``CSL`` class and its
layers, but supplies the fold-level training-positive graph described by the
paper and records every compatibility correction in ``protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import types

import numpy as np

try:
    from .common import (
        DEFAULT_CACHE_DIR,
        diagnostic_execution_context,
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
        DEFAULT_CACHE_DIR,
        diagnostic_execution_context,
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
        evaluate_test_scores,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )
except ImportError:  # Direct script execution with ``src`` on PYTHONPATH.
    from astra.benchmark.metrics import (
        evaluate_test_scores,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )


UPSTREAM_COMMIT = "09fa09795bc610018f906d26bea5b4259233400a"
OFFICIAL_BATCH_SIZE = 4096
SCALABLE_NODE_THRESHOLD = 4096
INNER_VALIDATION_EPOCH_CANDIDATES = (16, 30, 48, 64)


def _execution_mode_label(*, smoke: bool, diagnostic: bool) -> str:
    if smoke:
        return "smoke"
    if diagnostic:
        return "diagnostic"
    return "formal-candidate"


def _validation_ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the locked continuous metrics used for checkpoint selection."""

    metrics = evaluate_test_scores(labels, scores, threshold=0.5)
    return {
        "Validation_AUROC": float(metrics["AUROC"]),
        "Validation_AUPRC": float(metrics["AUPRC"]),
        "Validation_AveragePrecision": float(metrics["AveragePrecision"]),
    }


def _cpu_model_state(model) -> dict[str, object]:
    """Clone one checkpoint to host memory without changing optimizer state."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _score_sha256(scores: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(scores, dtype=np.float32).view(np.uint8)
    ).hexdigest()


def _install_import_shims() -> list[str]:
    """Provide only the unused legacy imports absent from the run environment."""
    import torch

    shims: list[str] = []
    if importlib.util.find_spec("pytorch_lightning") is None:
        lightning = types.ModuleType("pytorch_lightning")

        class LightningModule(torch.nn.Module):
            def save_hyperparameters(self, *args, **kwargs):
                return None

            def log(self, *args, **kwargs):
                return None

            @property
            def device(self):
                parameter = next(self.parameters(), None)
                return parameter.device if parameter is not None else torch.device("cpu")

        class LightningDataModule:
            pass

        lightning.LightningModule = LightningModule
        lightning.LightningDataModule = LightningDataModule
        sys.modules["pytorch_lightning"] = lightning
        shims.append("minimal pytorch_lightning.LightningModule import shim")

    if importlib.util.find_spec("torchvision") is None:
        torchvision = types.ModuleType("torchvision")
        ops = types.ModuleType("torchvision.ops")

        def unused_focal_loss(*args, **kwargs):
            raise RuntimeError("the unused torchvision focal_loss import was called")

        ops.focal_loss = unused_focal_loss
        torchvision.ops = ops
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.ops"] = ops
        shims.append("unused torchvision.ops.focal_loss import shim")
    return shims


def _load_official_model(source_dir: Path):
    shims = _install_import_shims()
    package_name = "astar_smgcl_official"
    package_dir = source_dir / "src"
    previous_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="herbhg_smgcl_pycache_") as pycache:
        try:
            # Compile the checked source into an isolated empty cache namespace;
            # never execute a pre-existing bytecode file from the author tree.
            sys.pycache_prefix = pycache
            sys.dont_write_bytecode = True
            specification = importlib.util.spec_from_file_location(
                package_name,
                package_dir / "__init__.py",
                submodule_search_locations=[str(package_dir)],
            )
            if specification is None or specification.loader is None:
                raise ImportError(f"cannot load the SMGCL package from {package_dir}")
            package = importlib.util.module_from_spec(specification)
            sys.modules[package_name] = package
            specification.loader.exec_module(package)
            model_module = importlib.import_module(f"{package_name}.model")
        finally:
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    return model_module, model_module.CSL, shims


def _official_similarity_edges(torch, similarity: np.ndarray, k: int, device):
    """Match the released top-k graph construction, including diagonal entries."""
    similarity = np.asarray(similarity, dtype=np.float32)
    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity matrix must be square")
    k = n if k <= 0 or k > n else k
    indices = np.arange(n, dtype=np.int64)
    if k == n:
        # The released ``argpartition(..., kth=n)`` is out of bounds.  This
        # branch is a safety correction for the documented all-neighbour
        # setting; formal runs use k=3 and therefore take the exact author
        # path below.
        neighbours = np.broadcast_to(indices, (n, n)).copy()
    else:
        neighbours = np.argpartition(-similarity, kth=k, axis=1)[:, :k]
    rows = np.repeat(indices, k)
    cols = neighbours.reshape(-1)
    values = similarity[rows, cols]
    return (
        torch.as_tensor(np.vstack((rows, cols)), dtype=torch.long, device=device),
        torch.as_tensor(values, dtype=torch.float32, device=device),
        (n, n),
    )


def _jaccard_topk_from_incidence(incidence, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the released dense ``argpartition`` top-k without dense N x N.

    The complete Jaccard matrix is sparse whenever two incidence rows do not
    share a neighbour.  We materialise one dense row at a time solely to keep
    NumPy's released-code ``argpartition(..., kth=k)[:k]`` tie behaviour.  The
    peak host allocation is therefore O(N + nnz) instead of O(N^2).
    """
    from scipy import sparse

    matrix = sparse.csr_matrix(incidence, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.data.fill(np.float32(1.0))
    intersections = (matrix @ matrix.T).tocsr()
    intersections.sum_duplicates()
    degrees = np.asarray(matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    n = int(matrix.shape[0])
    k = n if k <= 0 or k > n else int(k)
    neighbours = np.empty((n, k), dtype=np.int64)
    weights = np.empty((n, k), dtype=np.float32)
    all_indices = np.arange(n, dtype=np.int64)
    row_values = np.zeros(n, dtype=np.float32)
    for row in range(n):
        row_values.fill(np.float32(0.0))
        start, stop = intersections.indptr[row : row + 2]
        columns = intersections.indices[start:stop]
        overlap = intersections.data[start:stop]
        unions = degrees[row] + degrees[columns] - overlap
        ratios = np.divide(
            overlap,
            unions,
            out=np.zeros_like(overlap, dtype=np.float32),
            where=unions > 0,
        )
        row_values[columns] = ratios
        if degrees[row] > 0:
            row_values[row] = np.float32(1.0)
        if k == n:
            selected = all_indices
        else:
            selected = np.argpartition(-row_values, kth=k)[:k]
        neighbours[row] = selected
        weights[row] = row_values[selected]
    return neighbours, weights


def _topk_cache_identity(protocol, herb_k: int, disease_k: int) -> str:
    payload = {
        "dataset": str(protocol.dataset),
        "shape": [int(protocol.n_herbs), int(protocol.n_diseases)],
        "herb_k": int(herb_k),
        "disease_k": int(disease_k),
        "herb_compound_pairs": hashlib.sha256(
            np.ascontiguousarray(protocol.herb_compound_pairs, dtype=np.int64).view(
                np.uint8
            )
        ).hexdigest(),
        "protein_disease_pairs": hashlib.sha256(
            np.ascontiguousarray(protocol.protein_disease_pairs, dtype=np.int64).view(
                np.uint8
            )
        ).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_or_build_sparse_topk(
    protocol,
    cache_dir: Path,
    herb_k: int,
    disease_k: int,
):
    """Load/build only the Jaccard entries used by released SMGCL."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = _topk_cache_identity(protocol, herb_k, disease_k)
    dataset = str(protocol.dataset).lower().replace("-", "_")
    path = cache_dir / f"{dataset}_smgcl_jaccard_topk_{identity}.npz"
    metadata_path = path.with_suffix(".json")
    expected = {
        "identity": identity,
        "dataset": str(protocol.dataset),
        "herb_k": int(herb_k),
        "disease_k": int(disease_k),
        "definition": (
            "released dense Jaccard argpartition top-k, constructed one row at a time"
        ),
    }

    def validate(arrays):
        values = {
            "herb_neighbors": np.asarray(arrays["herb_neighbors"], dtype=np.int64),
            "herb_weights": np.asarray(arrays["herb_weights"], dtype=np.float32),
            "disease_neighbors": np.asarray(
                arrays["disease_neighbors"], dtype=np.int64
            ),
            "disease_weights": np.asarray(arrays["disease_weights"], dtype=np.float32),
        }
        shapes = {
            "herb_neighbors": (protocol.n_herbs, herb_k),
            "herb_weights": (protocol.n_herbs, herb_k),
            "disease_neighbors": (protocol.n_diseases, disease_k),
            "disease_weights": (protocol.n_diseases, disease_k),
        }
        for name, shape in shapes.items():
            if values[name].shape != shape:
                raise ValueError(
                    f"cached SMGCL {name} has shape {values[name].shape}, expected {shape}"
                )
        for prefix, size in (
            ("herb", protocol.n_herbs),
            ("disease", protocol.n_diseases),
        ):
            indices = values[f"{prefix}_neighbors"]
            weights = values[f"{prefix}_weights"]
            if np.any(indices < 0) or np.any(indices >= size):
                raise ValueError(f"cached SMGCL {prefix} neighbours are out of range")
            if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
                raise ValueError(f"cached SMGCL {prefix} weights are invalid")
            if np.any(np.count_nonzero(weights > 0, axis=1) == 0):
                raise ValueError(
                    f"SMGCL {prefix} top-k contains a zero-neighbour row"
                )
        return values

    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()) and (
            metadata.get("cache_sha256") == sha256(path)
        ):
            try:
                with np.load(path) as cached:
                    return validate(cached), path
            except (KeyError, OSError, ValueError):
                pass

    herb_incidence, disease_incidence = protocol.incidence_matrices()
    herb_neighbors, herb_weights = _jaccard_topk_from_incidence(
        herb_incidence, herb_k
    )
    disease_neighbors, disease_weights = _jaccard_topk_from_incidence(
        disease_incidence, disease_k
    )
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        herb_neighbors=herb_neighbors,
        herb_weights=herb_weights,
        disease_neighbors=disease_neighbors,
        disease_weights=disease_weights,
    )
    temporary.replace(path)
    write_json(metadata_path, {**expected, "cache_sha256": sha256(path)})
    with np.load(path) as cached:
        return validate(cached), path


def _topk_edge_tuple(torch, neighbours: np.ndarray, weights: np.ndarray, device):
    neighbours = np.asarray(neighbours, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    if neighbours.shape != weights.shape or neighbours.ndim != 2:
        raise ValueError("top-k neighbours and weights must have one matching 2-D shape")
    n, k = neighbours.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    return (
        torch.as_tensor(
            np.vstack((rows, neighbours.reshape(-1))),
            dtype=torch.long,
            device=device,
        ),
        torch.as_tensor(weights.reshape(-1), dtype=torch.float32, device=device),
        (n, n),
    )


def _training_positive_graph(
    torch,
    pairs: np.ndarray,
    n_herbs: int,
    n_diseases: int,
    device=None,
):
    """Correct block adjacency: [[0, H-D], [D-H, 0]]."""
    device = torch.device("cpu") if device is None else torch.device(device)
    unique = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
    herbs = unique[:, 0]
    diseases = unique[:, 1] + n_herbs
    row = np.concatenate((herbs, diseases))
    col = np.concatenate((diseases, herbs))
    indices = torch.as_tensor(
        np.vstack((row, col)), dtype=torch.long, device=device
    )
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    size = n_herbs + n_diseases
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce().to_dense()


def _training_positive_sparse_graph(
    torch,
    pairs: np.ndarray,
    n_herbs: int,
    n_diseases: int,
    device=None,
):
    """Sparse equivalent of :func:`_training_positive_graph` for large graphs."""
    device = torch.device("cpu") if device is None else torch.device(device)
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("training-positive pairs must have non-empty shape (n, 2)")
    herbs = pairs[:, 0]
    diseases = pairs[:, 1] + n_herbs
    row = np.concatenate((herbs, diseases))
    col = np.concatenate((diseases, herbs))
    indices = torch.as_tensor(
        np.vstack((row, col)), dtype=torch.long, device=device
    )
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    size = n_herbs + n_diseases
    graph = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
    # The frozen protocol has unique positive pairs.  Coalescing remains a
    # fail-safe deduplication boundary and this assignment keeps adjacency
    # binary if a non-formal input ever repeats a pair.
    return torch.sparse_coo_tensor(
        graph.indices(),
        torch.ones_like(graph.values()),
        graph.shape,
        device=device,
    ).coalesce()


def _prepare_sparse_bipartite_operators(torch, adjacency):
    """Construct exact sparse GCN/BGNN operators used by the author layers."""
    adjacency = adjacency.coalesce()
    size = int(adjacency.shape[0])
    diagonal = torch.arange(size, dtype=torch.long, device=adjacency.device)
    loop_indices = torch.stack((diagonal, diagonal))
    indices = torch.cat((adjacency.indices(), loop_indices), dim=1)
    values = torch.cat(
        (
            adjacency.values(),
            torch.ones(size, dtype=adjacency.dtype, device=adjacency.device),
        )
    )
    with_loops = torch.sparse_coo_tensor(
        indices, values, adjacency.shape, device=adjacency.device
    ).coalesce()
    rows, columns = with_loops.indices()
    weights = with_loops.values()
    degree = torch.zeros(size, dtype=weights.dtype, device=weights.device)
    squared_degree = torch.zeros_like(degree)
    degree.index_add_(0, rows, weights)
    squared_degree.index_add_(0, rows, weights.square())
    degree_inverse_sqrt = degree.pow(-0.5)
    degree_inverse_sqrt.masked_fill_(torch.isinf(degree_inverse_sqrt), 0.0)
    normalized_values = (
        weights * degree_inverse_sqrt[rows] * degree_inverse_sqrt[columns]
    )
    gcn = torch.sparse_coo_tensor(
        with_loops.indices(),
        normalized_values,
        with_loops.shape,
        device=with_loops.device,
    ).coalesce()
    bgnn_normalizer = (degree.square() - squared_degree).pow(-1)
    bgnn_normalizer.masked_fill_(torch.isinf(bgnn_normalizer), 0.0)
    return {
        "adjacency_with_self_loops": with_loops,
        "gcn_normalized_adjacency": gcn,
        "bgnn_normalizer": bgnn_normalizer,
    }


def _sparse_bgcn_state(encoder, node_state, operators):
    """Exact sparse algebra for the released ``BGCNA`` module."""
    import torch
    from torch.nn import functional as F

    module = encoder.bgnn
    gcn_layer = module.gcn
    bgnn_layer = module.bgnn
    gcn_projected = node_state @ gcn_layer.weight
    gcn_state = torch.sparse.mm(
        operators["gcn_normalized_adjacency"], gcn_projected
    )
    if gcn_layer.bias is not None:
        gcn_state = gcn_state + gcn_layer.bias

    bgnn_projected = node_state @ bgnn_layer.weight
    adjacency = operators["adjacency_with_self_loops"]
    neighbour_sum = torch.sparse.mm(adjacency, bgnn_projected)
    squared_sum = torch.sparse.mm(adjacency, bgnn_projected.square())
    bgnn_state = (neighbour_sum.square() - squared_sum) * operators[
        "bgnn_normalizer"
    ].unsqueeze(1)
    if bgnn_layer.bias is not None:
        bgnn_state = bgnn_state + bgnn_layer.bias
    state = module.beta * F.relu(gcn_state) + module.alpha * F.relu(bgnn_state)
    return encoder.dropout(state)


def _sparse_attention_state(layer, edge):
    """Exact top-k masked query-attention output without an N x N tensor.

    The released query-aware score is ``dot(Wh_i, Wh_j) + c_i``.  The second
    term is constant for every retained neighbour in row ``i`` and therefore
    cancels exactly in the row-wise softmax.  Only positive top-k Jaccard
    entries participate in the released mask.
    """
    import torch
    from torch.nn import functional as F

    indices, values, shape = edge
    n = int(shape[0])
    if shape != (n, n) or indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError("SMGCL attention edge tuple is malformed")
    if indices.shape[1] % n:
        raise ValueError("SMGCL attention edges must contain a fixed k per row")
    k = indices.shape[1] // n
    rows = indices[0].reshape(n, k)
    columns = indices[1].reshape(n, k)
    expected_rows = torch.arange(n, device=indices.device).unsqueeze(1).expand(n, k)
    if not bool(torch.equal(rows, expected_rows)):
        raise ValueError("SMGCL attention edges are not grouped by source row")
    mask = values.reshape(n, k) > 0
    if bool((~mask.any(dim=1)).any()):
        raise RuntimeError("SMGCL sparse attention encountered a zero-neighbour row")
    # In the scalable reparameterization ``W`` is the trainable projected
    # entity state itself.  This spans exactly the same state space as QW
    # because the RWR matrix Q is invertible for restart_probability > 0.
    projected = layer.W
    scores = (projected[rows] * projected[columns]).sum(dim=-1)
    scores = scores.masked_fill(~mask, -torch.inf)
    attention = torch.softmax(scores, dim=1)
    attention = F.dropout(attention, layer.dropout, training=layer.training)
    state = (attention.unsqueeze(-1) * projected[columns]).sum(dim=1)
    return F.elu(state) if layer.concat else state


def _direct_interaction_embedding(interaction_encoder, pairs):
    """Exact identity-projection lookup without materialising two identities."""
    from torch.nn import functional as F

    herb = interaction_encoder.drug_project.weight.T[pairs[0]]
    disease = interaction_encoder.disease_project.weight.T[pairs[1]]
    association = F.normalize(herb * disease)
    return interaction_encoder.dropout(association)


def _patch_sparse_graph_loss(model) -> None:
    """Replace dense Laplacian construction by its exact directed-edge form."""
    import torch

    def sparse_graph_loss(self, x, edge, topk=None, cache_name=None, reduction="mean"):
        indices, values, shape = edge
        if tuple(shape) != (x.shape[0], x.shape[0]):
            raise ValueError("SMGCL graph-loss edge shape does not match node state")
        rows, columns = indices
        keep = (rows != columns) & (values > 0)
        differences = x[rows[keep]] - x[columns[keep]]
        # For L=.5*(diag(colsum+rowsum)-(S+S.T)), tr(X^T L X)
        # equals .5*sum_ij S_ij ||x_i-x_j||^2, including asymmetric S.
        result = 0.5 * torch.sum(values[keep] * differences.square().sum(dim=1))
        if reduction == "mean":
            result = result / (x.shape[0] ** 2)
        elif reduction != "sum":
            raise ValueError(f"unsupported graph-loss reduction: {reduction}")
        return result

    model.graph_loss_fn = types.MethodType(sparse_graph_loss, model)


def _patch_scalable_forward(model, static_adjacency, n_herbs: int) -> None:
    """Install a sparse/reparameterized forward for author-code-scale overflow."""
    import torch
    from torch.nn import functional as F

    operators = _prepare_sparse_bipartite_operators(torch, static_adjacency)
    model._astra_sparse_operators = operators
    model._astra_n_herbs = int(n_herbs)

    def node_input(self):
        # Released dense path computes Q @ W.T independently in each channel.
        # Since Q is invertible, these transposed weights directly parameterise
        # the same unconstrained node-state space without constructing Q.
        return torch.cat(
            (
                self.drug_embedding_compress.weight.T,
                self.disease_embedding_compress.weight.T,
            ),
            dim=0,
        )

    def pair_states(self, encoder, pairs):
        state = _sparse_bgcn_state(
            encoder, self._astra_node_input(), self._astra_sparse_operators
        )
        herb = F.normalize(state[pairs[0]])
        disease = F.normalize(state[pairs[1] + self._astra_n_herbs])
        return herb, disease

    def prepare_inference_state(self):
        state = _sparse_bgcn_state(
            self.bgcn_encoder1,
            self._astra_node_input(),
            self._astra_sparse_operators,
        )
        return state[: self._astra_n_herbs], state[self._astra_n_herbs :]

    def score_from_state(self, interaction_pairs, herb_state, disease_state):
        herb = F.normalize(herb_state[interaction_pairs[0]])
        disease = F.normalize(disease_state[interaction_pairs[1]])
        interaction = _direct_interaction_embedding(
            self.interaction_encoder, interaction_pairs
        )
        return self.decoder(torch.cat((herb, interaction, disease), dim=-1)).reshape(-1)

    def scalable_forward(
        self,
        interaction_pairs,
        drug_edge,
        disease_edge,
        ignored_drug_embedding,
        ignored_disease_embedding,
    ):
        herb_index = interaction_pairs[0].unique()
        disease_index = interaction_pairs[1].unique()
        contrast_state = _sparse_bgcn_state(
            self.bgcn_encoder2,
            self._astra_node_input(),
            self._astra_sparse_operators,
        )
        herb_bgcn = F.normalize(contrast_state[herb_index])
        disease_bgcn = F.normalize(
            contrast_state[disease_index + self._astra_n_herbs]
        )
        herb_attention_all = _sparse_attention_state(self.drug_att_encoder, drug_edge)
        disease_attention_all = _sparse_attention_state(
            self.disease_att_encoder, disease_edge
        )
        herb_attention = F.normalize(herb_attention_all[herb_index])
        disease_attention = F.normalize(disease_attention_all[disease_index])
        herb_to_disease = self.SimilarityCompute(herb_attention, disease_attention)
        disease_to_herb = herb_to_disease.T
        positive_herb, negative_herb, positive_disease, negative_disease = (
            self.topk_func_random(
                herb_to_disease,
                disease_to_herb,
                herb_attention,
                disease_attention,
            )
        )
        contrast_herb = self.SSL_topk(
            herb_bgcn, positive_disease, negative_disease
        ) / herb_index.shape[0]
        contrast_disease = self.SSL_topk(
            disease_bgcn, positive_herb, negative_herb
        ) / disease_index.shape[0]
        herb_pair, disease_pair = self._astra_pair_states(
            self.bgcn_encoder1, interaction_pairs
        )
        interaction = _direct_interaction_embedding(
            self.interaction_encoder, interaction_pairs
        )
        score = self.decoder(
            torch.cat((herb_pair, interaction, disease_pair), dim=-1)
        ).reshape(-1)
        return score, contrast_herb, contrast_disease

    model._astra_node_input = types.MethodType(node_input, model)
    model._astra_pair_states = types.MethodType(pair_states, model)
    model._astra_prepare_inference_state = types.MethodType(
        prepare_inference_state, model
    )
    model._astra_score_from_state = types.MethodType(score_from_state, model)
    model.forward = types.MethodType(scalable_forward, model)
    _patch_sparse_graph_loss(model)


def _resolve_device(torch, requested: str):
    """Resolve the explicit CPU/CUDA policy without silently falling back."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SMGCL CUDA was requested, but torch.cuda.is_available() is false"
        )
    return torch.device(requested)


def _device_safe_forward(model, device, *args):
    """Run pinned code with unqualified tensor factories on the target device.

    The released ``CSL.forward`` creates one temporary ``torch.ones`` tensor
    without a device.  A scoped default-device context fixes that compatibility
    issue without editing or shadowing the commit-pinned upstream source.
    """
    import torch

    with torch.device(device):
        return model(*args)


def _patch_vectorized_query_attention(model) -> None:
    """Vectorize the released per-row query-aware attention expression.

    For each row ``i``, the author implementation adds
    ``dot(Wh[i], mean(Wh[nonzero(edge[i])]))`` to every entry of that row's
    pairwise-similarity vector.  The expression below computes the identical
    row shifts in matrix form.  Formal top-k similarity graphs always retain
    at least one nonzero neighbor (in particular, a unit-similarity entry);
    fail explicitly if that invariant is ever violated instead of silently
    changing the released divide-by-neighbor-count behavior.
    """
    import torch

    def vectorized(self, Wh, edge_index):
        nonzero = edge_index.ne(0)
        counts = nonzero.sum(dim=1, keepdim=True)
        if bool(counts.eq(0).any()):
            raise RuntimeError(
                "SMGCL query attention encountered a zero-neighbor row; "
                "the frozen top-k protocol requires at least one nonzero neighbor"
            )
        similarity = torch.mm(Wh, Wh.T)
        neighbor_sum = torch.mm(nonzero.to(dtype=Wh.dtype), Wh)
        neighbor_mean = neighbor_sum / counts.to(dtype=Wh.dtype)
        row_shift = torch.sum(Wh * neighbor_mean, dim=1, keepdim=True)
        return similarity + row_shift

    for layer in (model.drug_att_encoder, model.disease_att_encoder):
        layer._prepare_query_aware_mechanism_input = types.MethodType(
            vectorized, layer
        )


def _patch_safe_contrast_sampler(model) -> None:
    """Allocate samples on the embedding device and tolerate small batches."""
    import torch

    def safe_topk(self, score1, score2, drug_emb, disease_emb):
        n_drug, n_disease = score1.shape
        candidate_count = max(1, min(n_drug // 2, n_drug, n_disease))
        self.num = candidate_count
        self.K = max(1, min(n_drug // 5, candidate_count))
        drug_rank = score1.topk(candidate_count, dim=0, largest=True, sorted=True).indices
        disease_rank = score2.topk(candidate_count, dim=0, largest=True, sorted=True).indices
        # Advanced indexing produces [K, target, D], while SSL_topk consumes
        # [target, K, D].  Keep each target's ranked samples together instead
        # of relying on the released reshape, which interleaves both axes.
        pos_drug = drug_emb[drug_rank[: self.K]].permute(1, 0, 2).contiguous()
        pos_disease = (
            disease_emb[disease_rank[: self.K]].permute(1, 0, 2).contiguous()
        )
        lower = self.K if self.K < candidate_count else 0
        random_slices = torch.randint(
            lower,
            candidate_count,
            (self.K,),
            device=score1.device,
        )
        neg_drug = (
            drug_emb[drug_rank[random_slices]].permute(1, 0, 2).contiguous()
        )
        neg_disease = (
            disease_emb[disease_rank[random_slices]].permute(1, 0, 2).contiguous()
        )
        return pos_drug, neg_drug, pos_disease, neg_disease

    model.topk_func_random = types.MethodType(safe_topk, model)


def _patch_bipartite_embedding_offsets(model, n_herbs: int) -> None:
    """Align disease lookups with the concatenated ``[herb, disease]`` state.

    The released neighbor encoders index the disease half with the raw disease
    identifier.  That can only address the leading herb rows after the adapter
    fixes the bipartite adjacency to use one consistent node ordering.  Keep
    the author encoder computations intact and translate only the disease
    lookup indices at their two call boundaries.
    """
    pair_forward = model.bgcn_encoder1.forward
    unique_forward = model.bgcn_encoder2.forward

    def pair_forward_with_offset(self, pairs, edge_index, embedding):
        offset_pairs = pairs.clone()
        offset_pairs[1] = offset_pairs[1] + n_herbs
        return pair_forward(offset_pairs, edge_index, embedding)

    def unique_forward_with_offset(self, herb_ids, disease_ids, edge_index, embedding):
        return unique_forward(
            herb_ids,
            disease_ids + n_herbs,
            edge_index,
            embedding,
        )

    model.bgcn_encoder1.forward = types.MethodType(
        pair_forward_with_offset, model.bgcn_encoder1
    )
    model.bgcn_encoder2.forward = types.MethodType(
        unique_forward_with_offset, model.bgcn_encoder2
    )


def _official_rwr(similarity: np.ndarray, restart_probability: float) -> np.ndarray:
    """Numerically equivalent dense RWR solve used by the author code."""
    from scipy import linalg

    matrix = np.asarray(similarity, dtype=np.float32)
    matrix = (matrix + matrix.T) * np.float32(0.5)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    column_sums = matrix.sum(axis=0)
    isolated = column_sums == 0
    matrix[isolated, isolated] = 1.0
    column_sums = matrix.sum(axis=0)
    transition = matrix / column_sums[None, :]
    coefficient = np.eye(len(matrix), dtype=np.float32)
    coefficient -= np.float32(1.0 - restart_probability) * transition
    right = np.eye(len(matrix), dtype=np.float32) * np.float32(restart_probability)
    return linalg.solve(
        coefficient,
        right,
        assume_a="gen",
        overwrite_a=True,
        overwrite_b=True,
        check_finite=False,
    ).astype(np.float32, copy=False)


def _validate_rwr_embedding(
    similarity: np.ndarray,
    embedding: np.ndarray,
    restart_probability: float,
    name: str,
) -> float:
    """Verify a cached RWR state against its source-derived linear system.

    A cache hash stored in a neighbouring JSON file is not an independent
    integrity check.  The RWR solution is unique, so checking
    ``(I - (1-r)P)Q = rI`` ties cached ``Q`` directly to the frozen similarity
    matrix without repeating the dense inverse on every fold.
    """
    from scipy import sparse

    matrix = np.asarray(similarity, dtype=np.float64)
    state = np.asarray(embedding, dtype=np.float64)
    if state.shape != matrix.shape:
        raise ValueError(
            f"cached {name} RWR embedding has shape {state.shape}, expected {matrix.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError(f"cached {name} RWR embedding contains NaN or infinity")
    if float(state.min()) < -1e-6:
        raise ValueError(f"cached {name} RWR embedding contains negative probabilities")

    transition_source = (matrix + matrix.T) * 0.5
    transition_source = transition_source.copy()
    np.fill_diagonal(transition_source, 0.0)
    column_sums = transition_source.sum(axis=0)
    isolated = column_sums == 0
    transition_source[isolated, isolated] = 1.0
    column_sums = transition_source.sum(axis=0)
    transition = sparse.csr_matrix(transition_source / column_sums[None, :])

    residual = state - (1.0 - restart_probability) * (transition @ state)
    diagonal = np.diag_indices_from(residual)
    residual[diagonal] -= restart_probability
    maximum_residual = float(np.max(np.abs(residual)))
    maximum_column_error = float(np.max(np.abs(state.sum(axis=0) - 1.0)))
    if maximum_residual > 2e-4 or maximum_column_error > 2e-4:
        raise ValueError(
            f"cached {name} RWR embedding is not the source-derived solution: "
            f"residual={maximum_residual}, column_error={maximum_column_error}"
        )
    return maximum_residual


def _load_or_build_rwr(
    herb_similarity: np.ndarray,
    disease_similarity: np.ndarray,
    similarity_cache: Path,
    cache_dir: Path,
    restart_probability: float,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(restart_probability).replace(".", "p")
    path = cache_dir / f"smgcl_rwr_restart_{suffix}.npz"
    metadata_path = path.with_suffix(".json")
    expected = {
        "similarity_sha256": sha256(similarity_cache),
        "restart_probability": restart_probability,
        "definition": "author-equivalent dense random walk with restart",
        "validation": "source-derived sparse linear-system residual <= 2e-4",
    }
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()) and (
            metadata.get("cache_sha256") == sha256(path)
        ):
            try:
                with np.load(path) as cached:
                    herb = np.asarray(cached["herb"], dtype=np.float32)
                    disease = np.asarray(cached["disease"], dtype=np.float32)
                _validate_rwr_embedding(
                    herb_similarity, herb, restart_probability, "herb"
                )
                _validate_rwr_embedding(
                    disease_similarity, disease, restart_probability, "disease"
                )
                return herb, disease, path
            except (KeyError, OSError, ValueError):
                pass
    herb = _official_rwr(herb_similarity, restart_probability)
    disease = _official_rwr(disease_similarity, restart_probability)
    _validate_rwr_embedding(herb_similarity, herb, restart_probability, "herb")
    _validate_rwr_embedding(
        disease_similarity, disease, restart_probability, "disease"
    )
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, herb=herb, disease=disease)
    temporary.replace(path)
    write_json(metadata_path, {**expected, "cache_sha256": sha256(path)})
    return herb, disease, path


def _loss(model, scores, labels, herb_edge, disease_edge, contrast_h, contrast_d, epoch):
    if model.loss_fn_name == "focal":
        prediction_loss = model.focal_loss_fn(
            scores,
            labels,
            alpha=model.alpha,
            gamma=model.gamma,
        )["loss_focal"]
    elif model.loss_fn_name == "bce":
        prediction_loss = model.bce_loss_fn(
            scores,
            labels,
            model.pos_weight,
        )["loss_bce"]
    else:  # Kept explicit so a new upstream loss cannot silently enter a run.
        raise ValueError(f"unsupported SMGCL loss: {model.loss_fn_name}")
    herb_state = model.interaction_encoder.drug_project.weight.T
    disease_state = model.interaction_encoder.disease_project.weight.T
    herb_graph = model.graph_loss_fn(
        herb_state,
        herb_edge,
        topk=model.config["drug_neighbor_num"],
        cache_name="adapter_ul",
        reduction="sum",
    )
    disease_graph = model.graph_loss_fn(
        disease_state,
        disease_edge,
        topk=model.config["disease_neighbor_num"],
        cache_name="adapter_vl",
        reduction="sum",
    )
    lambda_h = epoch / 18.0 * 0.125
    lambda_d = epoch / 18.0 * 0.0625
    return prediction_loss + lambda_h * herb_graph + lambda_d * disease_graph + 0.1 * (
        contrast_h + contrast_d
    )


def _predict(
    model,
    pairs,
    *,
    herb_edge,
    disease_edge,
    herb_embedding,
    disease_embedding,
    batch_size,
    device,
    torch,
) -> np.ndarray:
    """Score one split without exposing its labels to model selection."""
    parts = []
    model.eval()
    with torch.no_grad():
        scalable_state = None
        if hasattr(model, "_astra_prepare_inference_state"):
            scalable_state = model._astra_prepare_inference_state()
        for positions in batches(np.arange(len(pairs), dtype=np.int64), batch_size):
            pair_tensor = torch.as_tensor(
                pairs[positions].T.copy(), dtype=torch.long, device=device
            )
            if scalable_state is None:
                scores, _, _ = _device_safe_forward(
                    model,
                    device,
                    pair_tensor,
                    herb_edge,
                    disease_edge,
                    herb_embedding,
                    disease_embedding,
                )
            else:
                scores = model._astra_score_from_state(
                    pair_tensor, scalable_state[0], scalable_state[1]
                )
            parts.append(scores.detach().cpu().numpy())
    if not parts:
        raise RuntimeError("SMGCL prediction split is empty")
    return np.concatenate(parts).astype(np.float32, copy=False)


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
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_THIRD_PARTY_DIR / "SMGCL_official")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=OFFICIAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--neighbor-embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(64, 32))
    parser.add_argument("--herb-neighbors", type=int, default=3)
    parser.add_argument("--disease-neighbors", type=int, default=3)
    parser.add_argument("--loss-fn", choices=("focal", "bce"), default="focal")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--restart-probability", type=float, default=0.1)
    parser.add_argument("--rwr-mode", choices=("official-exact", "raw"), default="official-exact")
    parser.add_argument(
        "--scalable-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Use sparse kernels and the invertible-RWR node-state "
            "reparameterization. Auto enables it when either entity side has "
            f"at least {SCALABLE_NODE_THRESHOLD} nodes."
        ),
    )
    parser.add_argument(
        "--legacy-test-oracle",
        action="store_true",
        help=(
            "Restore all four source-order outer-training folds and select "
            "descriptive threshold metrics on the held-out test fold."
        ),
    )
    parser.add_argument(
        "--inner-validation-epoch-selection",
        action="store_true",
        help=(
            "Train one 64-epoch trajectory, score validation checkpoints at "
            "epochs 16/30/48/64, select by trapezoidal validation AUPRC, and "
            "evaluate the outer test only for the frozen winner."
        ),
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--smoke", action="store_true")
    execution_mode.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Run the complete requested budget while recording that the output "
            "is formally ineligible (for example, a CPU sensitivity run)."
        ),
    )
    parser.add_argument("--smoke-train-rows", type=int, default=128)
    parser.add_argument("--smoke-test-rows", type=int, default=128)
    parser.add_argument("--run-generation-id")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.smoke_train_rows < 2 or args.smoke_test_rows < 2:
        parser.error("smoke row counts must be at least two")
    if args.inner_validation_epoch_selection:
        if args.smoke or args.diagnostic or args.legacy_test_oracle:
            parser.error(
                "--inner-validation-epoch-selection is formal-only and cannot "
                "be combined with smoke, diagnostic, or legacy-test-oracle"
            )
        if args.epochs != max(INNER_VALIDATION_EPOCH_CANDIDATES):
            parser.error(
                "--inner-validation-epoch-selection requires --epochs 64"
            )
        if args.batch_size != OFFICIAL_BATCH_SIZE:
            parser.error(
                "--inner-validation-epoch-selection requires --batch-size 4096"
            )

    diagnostic_context = (
        diagnostic_execution_context(args.device) if args.diagnostic else None
    )
    if args.diagnostic:
        formal_eligible = False
        formal_ineligibility_reasons = list(diagnostic_context["reasons"])
    elif args.smoke:
        formal_eligible = False
        formal_ineligibility_reasons = ["smoke run uses truncated rows or budget"]
    else:
        formal_eligible = True
        formal_ineligibility_reasons = []

    started = time.time()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.source_dir = args.source_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if not (args.source_dir / "src" / "model.py").exists():
        raise FileNotFoundError(
            f"SMGCL checkout missing at {args.source_dir}; run third_party/fetch_baselines.sh"
        )
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = _resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    protocol = load_protocol(args.protocol_manifest)
    if args.inner_validation_epoch_selection and protocol.dataset != "tcm-suite":
        parser.error(
            "--inner-validation-epoch-selection is locked to the TCM-suite protocol"
        )
    if args.legacy_test_oracle and (
        protocol.dataset != "tcm-suite" or protocol.pair_order != "source"
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
    if args.legacy_test_oracle:
        train_pairs_full, train_labels_full, train_row_id_full = outer_training_split(
            fold_rows
        )
    else:
        train_pairs_full, train_labels_full, train_row_id_full = fold_split(
            fold_rows, "train"
        )
    training_positive_pairs = train_pairs_full[train_labels_full == 1]
    scalable = args.scalable_mode == "on" or (
        args.scalable_mode == "auto"
        and max(protocol.n_herbs, protocol.n_diseases) >= SCALABLE_NODE_THRESHOLD
    )
    if args.inner_validation_epoch_selection and (
        args.scalable_mode != "on" or not scalable
    ):
        parser.error(
            "--inner-validation-epoch-selection requires --scalable-mode on"
        )
    if scalable:
        topk, similarity_cache = _load_or_build_sparse_topk(
            protocol,
            args.cache_dir,
            args.herb_neighbors,
            args.disease_neighbors,
        )
        herb_similarity = disease_similarity = None
        herb_embedding = disease_embedding = None
        embedding_cache = None
        rwr_mode = "invertible-rwr-node-state-reparameterization"
    else:
        herb_similarity, disease_similarity, similarity_cache = load_or_build_similarities(
            protocol, args.cache_dir
        )
        rwr_mode = "raw" if args.smoke and args.rwr_mode == "official-exact" else args.rwr_mode
        if rwr_mode == "official-exact":
            herb_embedding, disease_embedding, embedding_cache = _load_or_build_rwr(
                herb_similarity,
                disease_similarity,
                similarity_cache,
                args.cache_dir,
                args.restart_probability,
            )
        else:
            herb_embedding = herb_similarity
            disease_embedding = disease_similarity
            embedding_cache = None

    model_module, OfficialCSL, import_shims = _load_official_model(args.source_dir)
    if scalable:
        static_adjacency = _training_positive_sparse_graph(
            torch,
            training_positive_pairs,
            protocol.n_herbs,
            protocol.n_diseases,
            device,
        )
    else:
        static_adjacency = _training_positive_graph(
            torch,
            training_positive_pairs,
            protocol.n_herbs,
            protocol.n_diseases,
            device,
        )
        # The author forward method calls constructNet for every query batch.
        # This callback supplies the fixed fold-training graph and corrected
        # block ordering on datasets where the dense official path is viable.
        model_module.constructNet = lambda ignored_query_matrix: static_adjacency

    model = OfficialCSL(
        n_drug=protocol.n_herbs,
        n_disease=protocol.n_diseases,
        embedding_dim=args.embedding_dim,
        neighbor_embedding_dim=args.neighbor_embedding_dim,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.learning_rate,
        dropout=args.dropout,
        pos_weight=1.0,
        alpha=args.alpha,
        gamma=args.gamma,
        loss_fn=args.loss_fn,
        drug_neighbor_num=args.herb_neighbors,
        disease_neighbor_num=args.disease_neighbors,
    ).to(device)
    _patch_bipartite_embedding_offsets(model, protocol.n_herbs)
    _patch_safe_contrast_sampler(model)
    if scalable:
        herb_edge = _topk_edge_tuple(
            torch, topk["herb_neighbors"], topk["herb_weights"], device
        )
        disease_edge = _topk_edge_tuple(
            torch,
            topk["disease_neighbors"],
            topk["disease_weights"],
            device,
        )
        _patch_scalable_forward(model, static_adjacency, protocol.n_herbs)
        # The scalable forward deliberately ignores these placeholders.
        herb_embedding_tensor = torch.empty(0, dtype=torch.float32, device=device)
        disease_embedding_tensor = torch.empty(0, dtype=torch.float32, device=device)
    else:
        _patch_vectorized_query_attention(model)
        herb_edge = _official_similarity_edges(
            torch, herb_similarity, args.herb_neighbors, device
        )
        disease_edge = _official_similarity_edges(
            torch, disease_similarity, args.disease_neighbors, device
        )
        herb_embedding_tensor = torch.as_tensor(
            herb_embedding, dtype=torch.float32, device=device
        )
        disease_embedding_tensor = torch.as_tensor(
            disease_embedding, dtype=torch.float32, device=device
        )

    validation_pairs_full, validation_labels_full, validation_row_id_full = fold_split(
        fold_rows, "validation"
    )
    test_pairs_full, test_labels_full, test_row_id_full = fold_split(fold_rows, "test")
    train_selection = np.arange(len(train_pairs_full), dtype=np.int64)
    validation_selection = np.arange(len(validation_pairs_full), dtype=np.int64)
    test_selection = np.arange(len(test_pairs_full), dtype=np.int64)
    epochs = args.epochs
    if args.smoke:
        train_selection = balanced_prefix_indices(
            train_labels_full, max(1, args.smoke_train_rows // 2)
        )
        validation_selection = balanced_prefix_indices(
            validation_labels_full, max(1, args.smoke_test_rows // 2)
        )
        test_selection = balanced_prefix_indices(
            test_labels_full, max(1, args.smoke_test_rows // 2)
        )
        epochs = 1
    train_pairs = train_pairs_full[train_selection]
    train_labels = train_labels_full[train_selection]
    validation_pairs = validation_pairs_full[validation_selection]
    validation_labels = validation_labels_full[validation_selection]
    validation_row_id = validation_row_id_full[validation_selection]
    test_pairs = test_pairs_full[test_selection]
    test_labels = test_labels_full[test_selection]
    test_row_id = test_row_id_full[test_selection]
    evaluation_batch_size = max(args.batch_size, 4096)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.05 * args.learning_rate,
        max_lr=args.learning_rate,
        gamma=0.95,
        mode="exp_range",
        step_size_up=4,
        cycle_momentum=False,
    )
    initial_learning_rate = float(optimizer.param_groups[0]["lr"])
    generator = np.random.default_rng(args.seed)
    history: list[float] = []
    epoch_runtime_seconds: list[float] = []
    optimizer_steps = 0
    checkpoint_validation_scores: dict[int, np.ndarray] = {}
    checkpoint_validation_records: list[dict[str, object]] = []
    checkpoint_validation_runtime_seconds: list[float] = []
    selected_epoch: int | None = None
    selected_validation_auprc = -np.inf
    selected_state: dict[str, object] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        order = generator.permutation(len(train_pairs))
        running = []
        for batch_indices in batches(order, args.batch_size):
            pair_tensor = torch.as_tensor(
                train_pairs[batch_indices].T.copy(), dtype=torch.long, device=device
            )
            label_tensor = torch.as_tensor(
                train_labels[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            scores, contrast_h, contrast_d = _device_safe_forward(
                model,
                device,
                pair_tensor,
                herb_edge,
                disease_edge,
                herb_embedding_tensor,
                disease_embedding_tensor,
            )
            loss = _loss(
                model,
                scores,
                label_tensor,
                herb_edge,
                disease_edge,
                contrast_h,
                contrast_d,
                epoch,
            )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            running.append(float(loss.detach()))
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_runtime_seconds.append(time.perf_counter() - epoch_started)
        history.append(float(np.mean(running)))
        print(
            f"epoch={epoch + 1}/{epochs} loss={history[-1]:.8f} "
            f"runtime_seconds={epoch_runtime_seconds[-1]:.6f}",
            flush=True,
        )
        completed_epoch = epoch + 1
        if (
            args.inner_validation_epoch_selection
            and completed_epoch in INNER_VALIDATION_EPOCH_CANDIDATES
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            checkpoint_started = time.perf_counter()
            checkpoint_scores = _predict(
                model,
                validation_pairs,
                herb_edge=herb_edge,
                disease_edge=disease_edge,
                herb_embedding=herb_embedding_tensor,
                disease_embedding=disease_embedding_tensor,
                batch_size=evaluation_batch_size,
                device=device,
                torch=torch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            checkpoint_runtime = time.perf_counter() - checkpoint_started
            ranking = _validation_ranking(validation_labels, checkpoint_scores)
            checkpoint_validation_scores[completed_epoch] = checkpoint_scores
            checkpoint_validation_runtime_seconds.append(checkpoint_runtime)
            checkpoint_validation_records.append(
                {
                    "Epoch": completed_epoch,
                    **ranking,
                    "Validation_Score_SHA256": _score_sha256(checkpoint_scores),
                    "Validation_Runtime_Seconds": checkpoint_runtime,
                }
            )
            # Candidates are encountered in ascending order.  Strictly greater
            # replacement makes the deterministic exact-tie rule the lower
            # requested epoch budget.
            if ranking["Validation_AUPRC"] > selected_validation_auprc:
                selected_validation_auprc = ranking["Validation_AUPRC"]
                selected_epoch = completed_epoch
                selected_state = _cpu_model_state(model)

    result_dir = args.output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    epoch_selection: dict[str, object]
    if args.inner_validation_epoch_selection:
        if tuple(checkpoint_validation_scores) != INNER_VALIDATION_EPOCH_CANDIDATES:
            raise RuntimeError(
                "SMGCL inner-validation selection did not evaluate every locked checkpoint"
            )
        if selected_epoch is None or selected_state is None:
            raise RuntimeError("SMGCL inner-validation selection produced no winner")
        checkpoint_dir = result_dir / "validation_checkpoints"
        for candidate_epoch in INNER_VALIDATION_EPOCH_CANDIDATES:
            checkpoint_path = checkpoint_dir / f"epoch_{candidate_epoch}.npz"
            save_predictions(
                checkpoint_path,
                row_id=validation_row_id,
                pair=validation_pairs,
                label=validation_labels,
                score=checkpoint_validation_scores[candidate_epoch],
            )
            record = next(
                item
                for item in checkpoint_validation_records
                if item["Epoch"] == candidate_epoch
            )
            record["Validation_Predictions"] = str(
                checkpoint_path.relative_to(args.output_root)
            )
            record["Validation_Predictions_SHA256"] = sha256(checkpoint_path)
        epoch_selection = {
            "Mode": "inner-validation",
            "Candidates": list(INNER_VALIDATION_EPOCH_CANDIDATES),
            "Selection_Metric": "validation trapezoidal AUPRC",
            "Selection_Tie_Break": "lowest epoch on an exact AUPRC tie",
            "Selected_Epoch": selected_epoch,
            "Selected_Validation_AUPRC": selected_validation_auprc,
            "Checkpoint_Validation_Metrics": checkpoint_validation_records,
            "Outer_Test_Access": (
                "selection frozen before test scoring; exactly one selected "
                "checkpoint is evaluated on the outer test"
            ),
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }
        # Persist the decision before restoring/scoring the selected test
        # checkpoint.  A failure after this point leaves no complete metrics.
        write_json(result_dir / "epoch_selection.json", epoch_selection)
        model.load_state_dict(selected_state)
        validation_scores = checkpoint_validation_scores[selected_epoch]
        validation_runtime_seconds = float(
            sum(checkpoint_validation_runtime_seconds)
        )
    else:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_started = time.perf_counter()
        validation_scores = _predict(
            model,
            validation_pairs,
            herb_edge=herb_edge,
            disease_edge=disease_edge,
            herb_embedding=herb_embedding_tensor,
            disease_embedding=disease_embedding_tensor,
            batch_size=evaluation_batch_size,
            device=device,
            torch=torch,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_runtime_seconds = time.perf_counter() - validation_started
        selected_epoch = epochs
        epoch_selection = {
            "Mode": "fixed-budget",
            "Candidates": [epochs],
            "Selected_Epoch": epochs,
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }
    test_started = time.perf_counter()
    test_scores = _predict(
        model,
        test_pairs,
        herb_edge=herb_edge,
        disease_edge=disease_edge,
        herb_embedding=herb_embedding_tensor,
        disease_embedding=disease_embedding_tensor,
        batch_size=evaluation_batch_size,
        device=device,
        torch=torch,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_runtime_seconds = time.perf_counter() - test_started
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    if args.legacy_test_oracle:
        metrics = evaluate_test_oracle(test_labels, test_scores)
    else:
        metrics = evaluate_validation_and_test(
            validation_labels,
            validation_scores,
            test_labels,
            test_scores,
        )
    corrections = [
        "fixed fold-level bipartite graph built from training positives only",
        "corrected bipartite block ordering",
        "offset disease lookups into the disease block of concatenated graph states",
        "device-safe and small-batch-safe contrast sampling",
        "corrected contrast samples from rank-major to target-major [target,K,D] layout",
        (
            "target-device fixed training-positive adjacency and scoped "
            "default-device context for released temporary tensors"
        ),
        (
            "mathematically equivalent vectorization of the released "
            "per-row query-aware attention expression"
        ),
        "official GraphDataIterator default batch size 4096 (paper reports 64)",
        "released argpartition similarity-neighbour selection",
        (
            "central frozen source-order five-fold rows with the complete outer "
            "training fold restored"
            if args.legacy_test_oracle
            else "central frozen five-fold rows with inner training-only validation"
        ),
        (
            "descriptive F1 threshold selected on the held-out test fold"
            if args.legacy_test_oracle
            else "validation-selected F1 threshold applied unchanged to the test fold"
        ),
    ]
    if scalable:
        corrections.extend(
            (
                (
                    "exact sparse top-k masked attention; the released query-aware "
                    "row shift cancels in row-wise softmax"
                ),
                "exact sparse GCN/BGNN adjacency normalization and message algebra",
                "exact direct lookup for released identity-matrix linear projections",
                "exact sparse directed-edge form of the released Laplacian trace loss",
                (
                    "evaluation global graph state computed once per split and reused "
                    "by the unchanged pair decoder"
                ),
                (
                    "invertible-RWR node-state reparameterization preserves the model "
                    "function class but not the released initialization or Adam trajectory"
                ),
            )
        )
    metrics.update(
        {
            "Model": "SMGCL-NS-style-official-adapted",
            "Fold": args.fold,
            "Dataset": str(protocol.dataset),
            "Seed": args.seed,
            "Device": str(device),
            "Requested_Device": args.device,
            "Query_Attention_Implementation": (
                (
                    "exact sparse retained-edge form; query row shift cancels in softmax"
                    if scalable
                    else "runtime-monkeypatched exact matrix-form vectorization of released row loop"
                )
            ),
            "Scalable_Mode": bool(scalable),
            "Scalable_Mode_Request": args.scalable_mode,
            "Scalable_Mathematical_Equivalence": (
                (
                    "attention, BGNN/GCN, identity projection, graph loss, and eval "
                    "reuse are forward/gradient equivalent; invertible-RWR direct-state "
                    "parameterization preserves the hypothesis class but changes "
                    "initialization and Adam optimization geometry"
                )
                if scalable
                else "not applicable; dense released-module path"
            ),
            "Runtime_Seconds": time.time() - started,
            "Training_Runtime_Seconds": float(sum(epoch_runtime_seconds)),
            "Validation_Runtime_Seconds": validation_runtime_seconds,
            "Test_Runtime_Seconds": test_runtime_seconds,
            "Evaluation_Runtime_Seconds": (
                validation_runtime_seconds + test_runtime_seconds
            ),
            "Peak_CUDA_Memory_Bytes": peak_cuda_memory_bytes,
            "Epoch_Runtime_Seconds": epoch_runtime_seconds,
            "Epochs_Requested": args.epochs,
            "Epochs_Executed": epochs,
            "Epoch_Selection_Mode": epoch_selection["Mode"],
            "Epoch_Candidates": epoch_selection["Candidates"],
            "Selected_Epoch": selected_epoch,
            "Selected_Checkpoint_Optimizer_Steps": (
                int(selected_epoch)
                * int(np.ceil(len(train_pairs) / args.batch_size))
            ),
            "Checkpoint_Validation_Metrics": checkpoint_validation_records,
            "Epoch_Selection_Metric": epoch_selection.get("Selection_Metric"),
            "Epoch_Selection_Tie_Break": epoch_selection.get(
                "Selection_Tie_Break"
            ),
            "Test_Checkpoints_Evaluated": epoch_selection[
                "Test_Checkpoints_Evaluated"
            ],
            "Unselected_Test_Checkpoints_Evaluated": epoch_selection[
                "Unselected_Test_Checkpoints_Evaluated"
            ],
            "Upstream_Commit": UPSTREAM_COMMIT,
            "Upstream_Source": "https://github.com/Jcmorz/SMGCL",
            "Protocol_Kind": (
                "official modules with matched stored negatives (SMGCL-NS-style, "
                "not full-unknown weighted SMGCL) and documented corrections"
            ),
            "Training_Objective": (
                f"released-code {args.loss_fn} loss on frozen 1:1 "
                "positive/stored-negative rows (SMGCL-NS-style) plus "
                "released-code graph and contrastive terms"
            ),
            "Negative_Protocol": "frozen matched stored negatives; no all-unknown expansion",
            "Batch_Size": args.batch_size,
            "Effective_Max_Training_Batch_Size": min(args.batch_size, len(train_pairs)),
            "Batch_Size_Provenance": (
                "released GraphDataIterator default is 4096; paper reports 64"
            ),
            "Optimizer": "Adam(weight_decay=1e-4)",
            "Learning_Rate_Provenance": "released model/demo default 5e-4",
            "Scheduler": "CyclicLR(mode=exp_range,gamma=0.95,step_size_up=4)",
            "Scheduler_Interval": "epoch (released Lightning default)",
            "Initial_Learning_Rate": initial_learning_rate,
            "Maximum_Learning_Rate": args.learning_rate,
            "RWR_Mode": rwr_mode,
            "Smoke": bool(args.smoke),
            "Diagnostic": bool(args.diagnostic),
            "Execution_Mode": _execution_mode_label(
                smoke=bool(args.smoke), diagnostic=bool(args.diagnostic)
            ),
            "Formal_Eligible": formal_eligible,
            "Formal_Ineligibility_Reasons": formal_ineligibility_reasons,
            "Optimizer_Steps": optimizer_steps,
            "Evaluation_Mode": (
                "legacy-test-oracle" if args.legacy_test_oracle else "validation"
            ),
            "Training_Rows": int(len(train_pairs)),
            "Training_Positive_Graph_Rows": int(len(training_positive_pairs)),
            "Run_Parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    if legacy_provenance is not None:
        metrics["Legacy_Protocol_Provenance"] = legacy_provenance
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    final_input_manifest = protocol.input_manifest()
    if final_input_manifest != initial_input_manifest:
        raise RuntimeError("central benchmark inputs changed during the SMGCL run")
    validate_frozen_input_manifest(final_input_manifest)
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
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        args.output_root / "protocol.json",
        {
            **initial_input_manifest,
            "frozen_input_lock_sha256": frozen_input_lock_sha256,
            "protocol_manifest": str(args.protocol_manifest),
            "dataset": str(protocol.dataset),
            "run_generation_id": args.run_generation_id,
            "diagnostic": bool(args.diagnostic),
            "execution_mode": _execution_mode_label(
                smoke=bool(args.smoke), diagnostic=bool(args.diagnostic)
            ),
            "formal_eligible": formal_eligible,
            "formal_ineligibility_reasons": formal_ineligibility_reasons,
            "diagnostic_context": diagnostic_context,
            "legacy_protocol_provenance": legacy_provenance,
            "model": metrics,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "compatibility_corrections": corrections,
            "import_shims": import_shims,
            "adapter_sources": {
                "common.py": sha256(Path(__file__).with_name("common.py")),
                "smgcl_runner.py": sha256(Path(__file__)),
            },
            "similarity_cache": str(similarity_cache),
            "similarity_cache_sha256": sha256(similarity_cache),
            "similarity_cache_validation": (
                (
                    "released argpartition top-k Jaccard entries rebuilt from frozen incidence rows"
                    if scalable
                    else "exact equality to Jaccard matrices recomputed from frozen incidence rows"
                )
            ),
            "embedding_cache": str(embedding_cache) if embedding_cache else None,
            "embedding_cache_sha256": sha256(embedding_cache) if embedding_cache else None,
            "embedding_cache_validation": (
                "source-derived RWR linear-system residual <= 2e-4"
                if embedding_cache
                else None
            ),
            "training_loss": history,
            "epoch_runtime_seconds": epoch_runtime_seconds,
            "optimizer_steps": optimizer_steps,
            "epoch_selection": epoch_selection,
            "evaluation_mode": (
                "legacy-test-oracle" if args.legacy_test_oracle else "validation"
            ),
            "training_split": (
                "complete source-order outer training fold"
                if args.legacy_test_oracle
                else "inner training fold"
            ),
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "fields": ["row_id", "pair", "label", "score"],
            },
            "runtime_versions": formal_runtime_versions(),
            "runtime_environment_isolation_enforced": bool(
                not args.smoke and not args.diagnostic
            ),
            "runtime_module_origins": formal_runtime_module_origins(
                require_environment=not (args.smoke or args.diagnostic)
            ),
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
