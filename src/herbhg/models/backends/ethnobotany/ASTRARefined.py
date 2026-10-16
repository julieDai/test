# coding: utf-8
"""Independent, configurable Ethnobotany ASTRA implementation.

This module intentionally does not inherit from :mod:`HGHDA`.  The historical
class remains a frozen compatibility implementation, while ``ASTRARefined``
provides explicit switches for controlled experiments.  Its default options
reproduce the legacy graph/training choices (the C0 condition); every proposed
change must be enabled through an ``astra.*`` configuration key.

The pure configuration and clique-planning helpers live in this file so they
can be audited without constructing a TensorFlow session.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import re
from time import localtime, strftime, time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from astra.data_validation import group_indexed_relation
from astra.tf_initialization import initialize_uninitialized_variables
from base.herbRecommender import herbRecommender


_SEED = int(os.environ.get("ASTRA_SEED", os.environ.get("CDRUG_SEED", "42")))
random.seed(_SEED)
np.random.seed(_SEED)
tf.set_random_seed(_SEED)


def _reproduction_output_dir() -> str:
    default = Path(__file__).resolve().parent / "results"
    return os.environ.get(
        "ASTRA_OUTPUT_DIR",
        os.environ.get("CDRUG_OUTPUT_DIR", str(default)),
    )


def _parse_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean, got {value!r}")


def _parse_positive_int(value: object, *, key: str, maximum: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer, got {value!r}") from error
    if parsed <= 0 or (maximum is not None and parsed > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{key} must be > 0{suffix}, got {parsed}")
    return parsed


def _parse_optional_positive_float(value: object, *, key: str) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a positive number or none") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{key} must be a finite positive number or none")
    return parsed


def _parse_checkpoint_epochs(value: object, *, max_epochs: int) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return (max_epochs,)
    tokens = [token for token in re.split(r"[\s,]+", str(value).strip()) if token]
    epochs = sorted(
        {
            _parse_positive_int(token, key="astra.checkpoint.epochs")
            for token in tokens
        }
    )
    if any(epoch > max_epochs for epoch in epochs):
        raise ValueError(
            "astra.checkpoint.epochs cannot exceed num.max.epoch "
            f"({max_epochs}), got {epochs}"
        )
    # The final state is always retained, even when the requested selection
    # grid stops earlier. This keeps trainModel's legacy final-state contract.
    if max_epochs not in epochs:
        epochs.append(max_epochs)
    return tuple(epochs)


@dataclass(frozen=True)
class ASTRARefinedOptions:
    """Validated experimental controls for :class:`ASTRARefined`."""

    max_epochs: int
    n_layers: int = 5
    loss_kind: str = "legacy"
    loss_reduction: str = "sum"
    gradient_clip_value: float | None = 10.0
    gradient_clip_norm: float | None = 0.5
    protein_gate: bool = False
    dead_variables: str = "preserve"
    clique_sampling: str = "legacy_prefix"
    clique_max_pairs_per_hyperedge: int = 100
    clique_max_directed_edges: int = 200_000
    checkpoint_epochs: tuple[int, ...] = ()
    semantic_adapter: str = "identity"
    semantic_input_dim: int = 64
    semantic_freeze_input: bool = False
    trace_gradients: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "ASTRARefinedOptions":
        max_epochs = _parse_positive_int(
            config.get("num.max.epoch", 30), key="num.max.epoch"
        )
        n_layers = _parse_positive_int(
            config.get("astra.n_layers", 5), key="astra.n_layers", maximum=32
        )
        loss_kind = str(config.get("astra.loss.kind", "legacy")).strip().lower()
        if loss_kind not in {"legacy", "bce_logits"}:
            raise ValueError("astra.loss.kind must be legacy or bce_logits")
        loss_reduction = str(
            config.get("astra.loss.reduction", "sum")
        ).strip().lower()
        if loss_reduction not in {"sum", "mean"}:
            raise ValueError("astra.loss.reduction must be sum or mean")
        dead_variables = str(
            config.get("astra.dead_variables", "preserve")
        ).strip().lower()
        if dead_variables not in {"preserve", "remove"}:
            raise ValueError("astra.dead_variables must be preserve or remove")
        clique_sampling = str(
            config.get("astra.clique.sampling", "legacy_prefix")
        ).strip().lower()
        if clique_sampling not in {"legacy_prefix", "balanced"}:
            raise ValueError(
                "astra.clique.sampling must be legacy_prefix or balanced"
            )
        max_directed_edges = _parse_positive_int(
            config.get("astra.clique.max_directed_edges", 200_000),
            key="astra.clique.max_directed_edges",
        )
        if max_directed_edges % 2:
            raise ValueError(
                "astra.clique.max_directed_edges must be even because every "
                "sampled pair is materialised in both directions"
            )
        semantic_adapter = str(
            config.get("astra.semantic.adapter", "identity")
        ).strip().lower()
        if semantic_adapter not in {"identity", "linear"}:
            raise ValueError("astra.semantic.adapter must be identity or linear")
        semantic_input_dim = _parse_positive_int(
            config.get("astra.semantic.input_dim", 64),
            key="astra.semantic.input_dim",
        )
        if semantic_input_dim not in {64, 128}:
            raise ValueError("astra.semantic.input_dim currently supports 64 or 128")
        if semantic_input_dim == 128 and semantic_adapter != "linear":
            raise ValueError(
                "128-dimensional semantics require astra.semantic.adapter=linear"
            )
        return cls(
            max_epochs=max_epochs,
            n_layers=n_layers,
            loss_kind=loss_kind,
            loss_reduction=loss_reduction,
            gradient_clip_value=_parse_optional_positive_float(
                config.get("astra.grad.clip_value", 10.0),
                key="astra.grad.clip_value",
            ),
            gradient_clip_norm=_parse_optional_positive_float(
                config.get("astra.grad.clip_norm", 0.5),
                key="astra.grad.clip_norm",
            ),
            protein_gate=_parse_bool(
                config.get("astra.protein_gate", False), key="astra.protein_gate"
            ),
            dead_variables=dead_variables,
            clique_sampling=clique_sampling,
            clique_max_pairs_per_hyperedge=_parse_positive_int(
                config.get("astra.clique.max_pairs_per_hyperedge", 100),
                key="astra.clique.max_pairs_per_hyperedge",
            ),
            clique_max_directed_edges=max_directed_edges,
            checkpoint_epochs=_parse_checkpoint_epochs(
                config.get("astra.checkpoint.epochs"), max_epochs=max_epochs
            ),
            semantic_adapter=semantic_adapter,
            semantic_input_dim=semantic_input_dim,
            semantic_freeze_input=_parse_bool(
                config.get("astra.semantic.freeze_input", False),
                key="astra.semantic.freeze_input",
            ),
            trace_gradients=_parse_bool(
                config.get("astra.training.trace_gradients", False),
                key="astra.training.trace_gradients",
            ),
        )

    def as_jsonable(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "n_layers": self.n_layers,
            "loss_kind": self.loss_kind,
            "loss_reduction": self.loss_reduction,
            "gradient_clip_value": self.gradient_clip_value,
            "gradient_clip_norm": self.gradient_clip_norm,
            "protein_gate": self.protein_gate,
            "dead_variables": self.dead_variables,
            "clique_sampling": self.clique_sampling,
            "clique_max_pairs_per_hyperedge": self.clique_max_pairs_per_hyperedge,
            "clique_max_directed_edges": self.clique_max_directed_edges,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "semantic_adapter": self.semantic_adapter,
            "semantic_input_dim": self.semantic_input_dim,
            "semantic_freeze_input": self.semantic_freeze_input,
            "trace_gradients": self.trace_gradients,
        }

    @property
    def c0_compatible(self) -> bool:
        return (
            self.n_layers == 5
            and self.loss_kind == "legacy"
            and self.loss_reduction == "sum"
            and self.gradient_clip_value == 10.0
            and self.gradient_clip_norm == 0.5
            and not self.protein_gate
            and self.dead_variables == "preserve"
            and self.clique_sampling == "legacy_prefix"
            and self.clique_max_pairs_per_hyperedge == 100
            and self.clique_max_directed_edges == 200_000
            and self.semantic_adapter == "identity"
            and self.semantic_input_dim == 64
            and not self.semantic_freeze_input
        )


@dataclass(frozen=True)
class PairwiseEdgePlan:
    indices: np.ndarray
    edge_to_hyperedge: np.ndarray
    diagnostics: dict[str, object]


def _pair_rank_to_indices(rank: int, node_count: int) -> tuple[int, int]:
    """Map a lexicographic combination rank to ``(i, j)`` without expansion."""

    pair_count = node_count * (node_count - 1) // 2
    if node_count < 2 or rank < 0 or rank >= pair_count:
        raise ValueError("pair rank is outside the requested complete graph")
    i = 0
    remaining = int(rank)
    while True:
        row_width = node_count - i - 1
        if remaining < row_width:
            return i, i + 1 + remaining
        remaining -= row_width
        i += 1


def _stable_hyperedge_seed(seed: int, channel: str, hyperedge_id: object) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}\0{channel}\0{hyperedge_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _validate_hyperedge_nodes(
    nodes: Sequence[int], *, num_nodes: int, hyperedge: object
) -> tuple[int, ...]:
    parsed = tuple(int(node) for node in nodes)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"hyperedge {hyperedge!r} contains duplicate node indices")
    if any(node < 0 or node >= num_nodes for node in parsed):
        raise ValueError(f"hyperedge {hyperedge!r} contains an out-of-range node")
    return parsed


def _balanced_pair_allocations(capacities: Sequence[int], budget: int) -> list[int]:
    """Water-fill a pair budget so every eligible hyperedge is represented."""

    allocations = [0] * len(capacities)
    remaining = min(int(budget), int(sum(capacities)))
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    while remaining and active:
        next_active: list[int] = []
        for index in active:
            if remaining == 0:
                next_active.extend(
                    candidate
                    for candidate in active
                    if candidate > index and allocations[candidate] < capacities[candidate]
                )
                break
            allocations[index] += 1
            remaining -= 1
            if allocations[index] < capacities[index]:
                next_active.append(index)
        else:
            active = next_active
            continue
        active = next_active
    return allocations


def build_pairwise_edge_plan(
    hyperedge_to_nodes: Mapping[object, Sequence[int]],
    *,
    num_nodes: int,
    strategy: str = "legacy_prefix",
    max_pairs_per_hyperedge: int = 100,
    max_directed_edges: int = 200_000,
    seed: int = 42,
    channel: str = "graph",
    hyperedge_to_index: Mapping[object, int] | None = None,
) -> PairwiseEdgePlan:
    """Build deterministic bidirectional clique edges under a fixed budget."""

    if strategy not in {"legacy_prefix", "balanced"}:
        raise ValueError("strategy must be legacy_prefix or balanced")
    if num_nodes <= 0 or max_pairs_per_hyperedge <= 0:
        raise ValueError("node and per-hyperedge limits must be positive")
    if max_directed_edges <= 0 or max_directed_edges % 2:
        raise ValueError("max_directed_edges must be a positive even integer")

    original_records: list[tuple[object, int, tuple[int, ...]]] = []
    for position, (hyperedge, raw_nodes) in enumerate(hyperedge_to_nodes.items()):
        nodes = _validate_hyperedge_nodes(
            raw_nodes, num_nodes=num_nodes, hyperedge=hyperedge
        )
        if hyperedge_to_index is None:
            canonical_index = position
        else:
            if hyperedge not in hyperedge_to_index:
                raise ValueError(f"missing canonical index for hyperedge {hyperedge!r}")
            canonical_index = int(hyperedge_to_index[hyperedge])
        original_records.append((hyperedge, canonical_index, nodes))

    rows: list[int] = []
    columns: list[int] = []
    owners: list[int] = []
    represented: set[int] = set()
    eligible = sum(len(nodes) >= 2 for _, _, nodes in original_records)

    if strategy == "legacy_prefix":
        visited = 0
        for _hyperedge, owner, nodes in original_records:
            if len(rows) >= max_directed_edges:
                break
            visited += 1
            pair_limit = min(
                len(nodes) * (len(nodes) - 1) // 2,
                max_pairs_per_hyperedge,
            )
            pair_count = 0
            for left in range(len(nodes)):
                for right in range(left + 1, len(nodes)):
                    if (
                        pair_count >= pair_limit
                        or len(rows) >= max_directed_edges
                    ):
                        break
                    node_i, node_j = nodes[left], nodes[right]
                    rows.extend((node_i, node_j))
                    columns.extend((node_j, node_i))
                    owners.extend((owner, owner))
                    represented.add(owner)
                    pair_count += 1
                if pair_count >= pair_limit or len(rows) >= max_directed_edges:
                    break
    else:
        records = sorted(original_records, key=lambda record: record[1])
        if len({record[1] for record in records}) != len(records):
            raise ValueError("canonical hyperedge indices must be unique")
        capacities = [
            min(len(nodes) * (len(nodes) - 1) // 2, max_pairs_per_hyperedge)
            for _, _, nodes in records
        ]
        allocations = _balanced_pair_allocations(
            capacities, max_directed_edges // 2
        )
        visited = len(records)
        for (hyperedge, owner, nodes), allocation in zip(records, allocations):
            if allocation == 0:
                continue
            total_pairs = len(nodes) * (len(nodes) - 1) // 2
            rng = random.Random(_stable_hyperedge_seed(seed, channel, owner))
            ranks = sorted(rng.sample(range(total_pairs), allocation))
            for rank in ranks:
                left, right = _pair_rank_to_indices(rank, len(nodes))
                node_i, node_j = nodes[left], nodes[right]
                rows.extend((node_i, node_j))
                columns.extend((node_j, node_i))
                owners.extend((owner, owner))
            represented.add(owner)

    indices = np.asarray([rows, columns], dtype=np.int64).T
    if indices.size == 0:
        indices = np.empty((0, 2), dtype=np.int64)
    owner_array = np.asarray(owners, dtype=np.int32)
    diagnostics: dict[str, object] = {
        "strategy": strategy,
        "channel": channel,
        "total_hyperedges": len(original_records),
        "eligible_hyperedges": eligible,
        "visited_hyperedges": visited,
        "represented_hyperedges": len(represented),
        "eligible_coverage": (
            float(len(represented) / eligible) if eligible else 1.0
        ),
        "directed_edges": int(len(indices)),
        "undirected_pairs": int(len(indices) // 2),
        "max_pairs_per_hyperedge": int(max_pairs_per_hyperedge),
        "max_directed_edges": int(max_directed_edges),
        "seed": int(seed),
    }
    return PairwiseEdgePlan(indices, owner_array, diagnostics)


def stable_binary_cross_entropy_numpy(
    logits: np.ndarray, labels: np.ndarray, *, reduction: str = "sum"
) -> float:
    """Reference implementation of TensorFlow's stable sigmoid BCE formula."""

    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if scores.shape != targets.shape:
        raise ValueError("logits and labels must have identical shapes")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("logits and labels must be finite")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("labels must be in [0, 1]")
    losses = np.maximum(scores, 0.0) - scores * targets + np.log1p(
        np.exp(-np.abs(scores))
    )
    if reduction == "sum":
        return float(losses.sum())
    if reduction == "mean":
        return float(losses.mean()) if losses.size else 0.0
    raise ValueError("reduction must be sum or mean")


class ASTRARefined(herbRecommender):
    """ASTRA model with auditable, independently configurable refinements."""

    def __init__(self, conf, trainingSet=None, testSet=None, fold="[1]"):
        super(ASTRARefined, self).__init__(conf, trainingSet, testSet, fold)
        self.options = ASTRARefinedOptions.from_mapping(conf.config)
        self.use_clique_expansion = True
        self.clique_fusion_init = 0.5
        self.text_fusion_alpha = float(conf.config.get("text.fusion.alpha", "1.0"))
        if (
            not np.isfinite(self.text_fusion_alpha)
            or not 0.0 <= self.text_fusion_alpha <= 1.0
        ):
            raise ValueError("text.fusion.alpha must be finite and in [0, 1]")
        self.epoch_snapshots: dict[int, dict[str, object]] = {}
        self.clique_diagnostics: dict[str, dict[str, object]] = {}
        self.resolved_hyperparameters = {
            "model_class": "ASTRARefined",
            "text_fusion_alpha": self.text_fusion_alpha,
            "seed": _SEED,
            "c0_compatible": self.options.c0_compatible,
            **self.options.as_jsonable(),
        }
        if self.options.dead_variables == "remove":
            # The dead trainable parameters are excluded from the objective and
            # optimizer.  Non-trainable shadows retain their initializer graph
            # positions so later active weights start from exactly the same
            # values as the preserve control.
            self.resolved_hyperparameters[
                "dead_variable_initialization_alignment"
            ] = "nontrainable_shadow_variables"

    def readConfiguration(self):
        """Resolve base trainer controls before emitting the execution receipt."""

        super(ASTRARefined, self).readConfiguration()
        self.resolved_hyperparameters.update(
            {
                "learning_rate": float(self.lRate),
                "batch_size": int(self.batch_size),
                "embedding_dimension": int(self.emb_size),
                "reg_u": float(self.regU),
            }
        )

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            if int(pair[2]) != 0:
                row.append(self.data.herb[pair[0]])
                col.append(self.data.disease[pair[1]])
                entries.append(1)
        return coo_matrix(
            (entries, (row, col)),
            shape=(self.num_herbs, self.num_diseases),
            dtype=np.float32,
        )

    def buildhcAdjacencyMatrix(self):
        row = [self.data.herb[pair[0]] for pair in self.data.hcassociation]
        col = [self.data.compound[pair[1]] for pair in self.data.hcassociation]
        return coo_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(self.num_herbs, self.num_compounds),
            dtype=np.float32,
        )

    def buildpdAdjacencyMatrix(self):
        row = [self.data.protein[pair[0]] for pair in self.data.pdassociation]
        col = [self.data.disease[pair[1]] for pair in self.data.pdassociation]
        return coo_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(self.num_proteins, self.num_diseases),
            dtype=np.float32,
        )

    def build_hyperedge_node_mapping(self):
        herb_to_compounds = group_indexed_relation(
            self.data.hcassociation,
            group_column=0,
            member_column=1,
            member_to_index=self.data.compound,
            relation_name="H--C",
        )
        disease_to_proteins = group_indexed_relation(
            self.data.pdassociation,
            group_column=1,
            member_column=0,
            member_to_index=self.data.protein,
            relation_name="P--D",
        )
        return herb_to_compounds, disease_to_proteins

    @staticmethod
    def _incidence_operators(incidence):
        node_degree = incidence.sum(axis=1).reshape(1, -1)
        edge_degree = incidence.sum(axis=0).reshape(1, -1)
        edge_matrix = (incidence.multiply(1.0 / edge_degree)).transpose().tocoo()
        node_matrix = (
            incidence.transpose().multiply(1.0 / node_degree)
        ).transpose().tocoo()
        edge_indices = np.mat([edge_matrix.row, edge_matrix.col]).transpose()
        node_indices = np.mat([node_matrix.row, node_matrix.col]).transpose()
        edge_operator = tf.SparseTensor(
            edge_indices,
            edge_matrix.data.astype(np.float32),
            edge_matrix.shape,
        )
        node_operator = tf.SparseTensor(
            node_indices,
            node_matrix.data.astype(np.float32),
            node_matrix.shape,
        )
        return edge_operator, node_operator

    def _weight(self, key: str, shape: Sequence[int], variable_name: str, initializer):
        variable = tf.Variable(initializer(shape), name=variable_name)
        self.weights[key] = variable
        return variable

    def _initialization_shadow(
        self,
        key: str,
        shape: Sequence[int],
        variable_name: str,
        initializer,
    ):
        """Preserve seeded initializer order without retaining a dead parameter.

        A non-trainable variable deliberately mirrors the historical variable
        creation op.  It is absent from ``self.weights`` and from TensorFlow's
        trainable collection, so it contributes neither regularization nor a
        gradient.  Keeping the creation op prevents later active variables from
        receiving different seeded initial values in the remove control.
        """

        variable = tf.Variable(
            initializer(shape), name=variable_name, trainable=False
        )
        self.initialization_alignment_variables[key] = variable
        return variable

    def _prepare_semantic_embeddings(self, initializer):
        if not self.use_text_embeddings:
            return None, None
        herb_dim = int(self.herb_text_embeddings.shape.as_list()[1])
        disease_dim = int(self.disease_text_embeddings.shape.as_list()[1])
        expected = self.options.semantic_input_dim
        if herb_dim != expected or disease_dim != expected:
            raise ValueError(
                "semantic matrix width differs from astra.semantic.input_dim: "
                f"herb={herb_dim}, disease={disease_dim}, configured={expected}"
            )
        herb_input = self.herb_text_embeddings
        disease_input = self.disease_text_embeddings
        if self.options.semantic_freeze_input:
            herb_input = tf.stop_gradient(herb_input)
            disease_input = tf.stop_gradient(disease_input)
        if self.options.semantic_adapter == "identity":
            if expected != self.emb_size:
                raise ValueError(
                    "identity semantic adapter requires semantic width to equal "
                    f"num.factors ({self.emb_size})"
                )
            herb = herb_input
            disease = disease_input
        else:
            herb_projection = self._weight(
                "semantic_projection_h",
                (expected, self.emb_size),
                "semantic_projection_herb",
                initializer,
            )
            disease_projection = self._weight(
                "semantic_projection_d",
                (expected, self.emb_size),
                "semantic_projection_disease",
                initializer,
            )
            herb_bias = self._weight(
                "semantic_projection_bias_h",
                (1, self.emb_size),
                "semantic_projection_bias_herb",
                tf.zeros_initializer(),
            )
            disease_bias = self._weight(
                "semantic_projection_bias_d",
                (1, self.emb_size),
                "semantic_projection_bias_disease",
                tf.zeros_initializer(),
            )
            herb = tf.matmul(herb_input, herb_projection) + herb_bias
            disease = (
                tf.matmul(disease_input, disease_projection) + disease_bias
            )
        return (
            tf.nn.l2_normalize(herb, axis=1),
            tf.nn.l2_normalize(disease, axis=1),
        )

    @staticmethod
    def _self_gate(embeddings, matrix, bias):
        return tf.multiply(
            embeddings,
            tf.nn.sigmoid(tf.matmul(embeddings, matrix) + bias),
        )

    @staticmethod
    def _pairwise_aggregate(
        embeddings,
        indices,
        attention_matrix,
        *,
        num_nodes: int,
    ):
        embedding_i = tf.gather(embeddings, indices[:, 0])
        embedding_j = tf.gather(embeddings, indices[:, 1])
        attention_i = tf.matmul(embedding_i, attention_matrix)
        attention_j = tf.matmul(embedding_j, attention_matrix)
        edge_score = tf.nn.leaky_relu(
            tf.reduce_sum(attention_i * attention_j, axis=1)
        )
        sources = indices[:, 0]
        maximum = tf.math.unsorted_segment_max(
            edge_score, sources, num_segments=num_nodes
        )
        maximum = tf.where(tf.is_inf(maximum), tf.zeros_like(maximum), maximum)
        exponent = tf.exp(edge_score - tf.gather(maximum, sources))
        denominator = tf.math.unsorted_segment_sum(
            exponent, sources, num_segments=num_nodes
        )
        weights = exponent / (tf.gather(denominator, sources) + 1e-8)
        adjacency = tf.SparseTensor(
            indices=indices,
            values=weights,
            dense_shape=[num_nodes, num_nodes],
        )
        return tf.sparse_tensor_dense_matmul(adjacency, embeddings)

    def initModel(self):
        super(ASTRARefined, self).initModel()
        # Retained in C0 even though H--D adjacency is not consumed downstream.
        self.buildAdjacencyMatrix()
        hc = self.buildhcAdjacencyMatrix()
        pd = self.buildpdAdjacencyMatrix()
        h_edge, h_node = self._incidence_operators(hc.transpose())
        p_edge, p_node = self._incidence_operators(pd)

        herb_to_compounds, disease_to_proteins = self.build_hyperedge_node_mapping()
        compound_plan = build_pairwise_edge_plan(
            herb_to_compounds,
            num_nodes=self.num_compounds,
            strategy=self.options.clique_sampling,
            max_pairs_per_hyperedge=self.options.clique_max_pairs_per_hyperedge,
            max_directed_edges=self.options.clique_max_directed_edges,
            seed=_SEED,
            channel="compound",
            hyperedge_to_index=(
                self.data.herb
                if self.options.clique_sampling == "balanced"
                else None
            ),
        )
        protein_plan = build_pairwise_edge_plan(
            disease_to_proteins,
            num_nodes=self.num_proteins,
            strategy=self.options.clique_sampling,
            max_pairs_per_hyperedge=self.options.clique_max_pairs_per_hyperedge,
            max_directed_edges=self.options.clique_max_directed_edges,
            seed=_SEED,
            channel="protein",
            hyperedge_to_index=(
                self.data.disease
                if self.options.clique_sampling == "balanced"
                else None
            ),
        )
        self.compound_pairwise_indices = compound_plan.indices
        self.protein_pairwise_indices = protein_plan.indices
        self.compound_edge_to_hyperedge = compound_plan.edge_to_hyperedge
        self.protein_edge_to_hyperedge = protein_plan.edge_to_hyperedge
        self.clique_diagnostics = {
            "compound": compound_plan.diagnostics,
            "protein": protein_plan.diagnostics,
        }
        self.resolved_hyperparameters["clique_diagnostics"] = self.clique_diagnostics
        self.compound_pairwise_indices_tf = tf.constant(
            self.compound_pairwise_indices, dtype=tf.int64
        )
        self.protein_pairwise_indices_tf = tf.constant(
            self.protein_pairwise_indices, dtype=tf.int64
        )
        # These tensors are retained in C0 even though the historical forward
        # pass does not consume them. Keeping graph construction order intact
        # preserves TensorFlow's seeded initializer stream.
        self.compound_edge_to_hyperedge_tf = tf.constant(
            self.compound_edge_to_hyperedge, dtype=tf.int32
        )
        self.protein_edge_to_hyperedge_tf = tf.constant(
            self.protein_edge_to_hyperedge, dtype=tf.int32
        )

        self.isTraining = tf.cast(tf.placeholder(tf.int32), tf.bool)
        initializer = tf.keras.initializers.glorot_normal()
        self.n_layer = self.options.n_layers
        self.weights: dict[str, Any] = {}
        self.initialization_alignment_variables: dict[str, Any] = {}
        preserve_dead = self.options.dead_variables == "preserve"
        for layer in range(self.n_layer):
            number = layer + 1
            self._weight(
                f"layer_{number}",
                (self.emb_size, self.emb_size),
                f"JU_{number}",
                initializer,
            )
            self._weight(
                f"layer_1_{number}",
                (self.emb_size, self.emb_size),
                f"JU_1_{number}",
                initializer,
            )
            self._weight(
                f"layer_2_{number}",
                (self.emb_size, self.emb_size),
                f"JU_2_{number}",
                initializer,
            )
            if preserve_dead:
                self._weight(
                    f"layer_att_{number}",
                    (self.emb_size, self.emb_size),
                    f"layer_bias_{number}",
                    initializer,
                )
            else:
                self._initialization_shadow(
                    f"layer_att_{number}",
                    (self.emb_size, self.emb_size),
                    f"layer_bias_{number}",
                    initializer,
                )

        for channel in range(1, 3):
            gate_is_active = channel == 1 or preserve_dead or self.options.protein_gate
            if not gate_is_active:
                self._initialization_shadow(
                    "gating2",
                    (self.emb_size, self.emb_size),
                    "g_W_2_1",
                    initializer,
                )
                self._initialization_shadow(
                    "gating_bias2",
                    (1, self.emb_size),
                    "g_W_b_2_1",
                    initializer,
                )
                continue
            self._weight(
                f"gating{channel}",
                (self.emb_size, self.emb_size),
                f"g_W_{channel}_1",
                initializer,
            )
            self._weight(
                f"gating_bias{channel}",
                (1, self.emb_size),
                f"g_W_b_{channel}_1",
                initializer,
            )

        self._weight(
            "clique_att_c",
            (self.emb_size, self.emb_size),
            "clique_att_compound",
            initializer,
        )
        self._weight(
            "clique_att_p",
            (self.emb_size, self.emb_size),
            "clique_att_protein",
            initializer,
        )
        self.weights["fusion_c"] = tf.Variable(
            tf.constant(self.clique_fusion_init), name="fusion_weight_compound"
        )
        self.weights["fusion_p"] = tf.Variable(
            tf.constant(self.clique_fusion_init), name="fusion_weight_protein"
        )

        herb_embeddings, disease_embeddings = self._prepare_semantic_embeddings(
            initializer
        )
        compound_embeddings = self._self_gate(
            self.compound_embeddings,
            self.weights["gating1"],
            self.weights["gating_bias1"],
        )
        if self.options.protein_gate:
            protein_embeddings = self._self_gate(
                self.protein_embeddings,
                self.weights["gating2"],
                self.weights["gating_bias2"],
            )
        else:
            protein_embeddings = self.protein_embeddings

        all_compound_embeddings = [compound_embeddings]
        all_protein_embeddings = [protein_embeddings]
        all_herb_embeddings = [herb_embeddings] if herb_embeddings is not None else []
        all_disease_embeddings = (
            [disease_embeddings] if disease_embeddings is not None else []
        )
        all_hc_embeddings = []
        all_pd_embeddings = []

        for layer in range(self.n_layer):
            number = layer + 1
            new_hc_edge = tf.sparse_tensor_dense_matmul(h_edge, compound_embeddings)
            new_compound_embeddings = tf.sparse_tensor_dense_matmul(
                h_node, new_hc_edge
            )
            new_pd_edge = tf.sparse_tensor_dense_matmul(p_edge, protein_embeddings)
            new_protein_embeddings = tf.sparse_tensor_dense_matmul(
                p_node, new_pd_edge
            )

            if len(self.compound_pairwise_indices):
                pairwise_compound = self._pairwise_aggregate(
                    compound_embeddings,
                    self.compound_pairwise_indices_tf,
                    self.weights["clique_att_c"],
                    num_nodes=self.num_compounds,
                )
                fusion = tf.nn.sigmoid(self.weights["fusion_c"])
                new_compound_embeddings = (
                    fusion * new_compound_embeddings
                    + (1.0 - fusion) * pairwise_compound
                )
            if len(self.protein_pairwise_indices):
                pairwise_protein = self._pairwise_aggregate(
                    protein_embeddings,
                    self.protein_pairwise_indices_tf,
                    self.weights["clique_att_p"],
                    num_nodes=self.num_proteins,
                )
                fusion = tf.nn.sigmoid(self.weights["fusion_p"])
                new_protein_embeddings = (
                    fusion * new_protein_embeddings
                    + (1.0 - fusion) * pairwise_protein
                )

            compound_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_compound_embeddings, self.weights[f"layer_{number}"])
                + compound_embeddings
            )
            protein_embeddings = tf.nn.leaky_relu(
                tf.matmul(
                    new_protein_embeddings, self.weights[f"layer_1_{number}"]
                )
                + protein_embeddings
            )
            compound_embeddings = tf.math.l2_normalize(compound_embeddings, axis=1)
            protein_embeddings = tf.math.l2_normalize(protein_embeddings, axis=1)
            new_hc_edge = tf.math.l2_normalize(new_hc_edge, axis=1)
            new_pd_edge = tf.math.l2_normalize(new_pd_edge, axis=1)

            if herb_embeddings is not None and disease_embeddings is not None:
                shared_transform = self.weights[f"layer_2_{number}"]
                herb_embeddings = tf.nn.leaky_relu(
                    tf.matmul(new_hc_edge, shared_transform) + herb_embeddings
                )
                disease_embeddings = tf.nn.leaky_relu(
                    tf.matmul(new_pd_edge, shared_transform) + disease_embeddings
                )
                herb_embeddings = tf.math.l2_normalize(herb_embeddings, axis=1)
                disease_embeddings = tf.math.l2_normalize(
                    disease_embeddings, axis=1
                )
                all_herb_embeddings.append(herb_embeddings)
                all_disease_embeddings.append(disease_embeddings)

            all_compound_embeddings.append(compound_embeddings)
            all_protein_embeddings.append(protein_embeddings)
            all_hc_embeddings.append(new_hc_edge)
            all_pd_embeddings.append(new_pd_edge)

        compound_embeddings = tf.reduce_sum(all_compound_embeddings, axis=0)
        protein_embeddings = tf.reduce_sum(all_protein_embeddings, axis=0)
        hc_edge = tf.reduce_sum(all_hc_embeddings, axis=0)
        pd_edge = tf.reduce_sum(all_pd_embeddings, axis=0)
        if all_herb_embeddings and all_disease_embeddings:
            herb_final = tf.reduce_sum(all_herb_embeddings, axis=0)
            disease_final = tf.reduce_sum(all_disease_embeddings, axis=0)
            alpha = tf.constant(self.text_fusion_alpha, dtype=tf.float32)
            self.final_hcedge = (1.0 - alpha) * hc_edge + alpha * herb_final
            self.final_pdedge = (1.0 - alpha) * pd_edge + alpha * disease_final
        else:
            self.final_hcedge = hc_edge
            self.final_pdedge = pd_edge

        self.neg_idx = tf.placeholder(tf.float32, name="neg_holder")
        self.neg_disease_embedding = tf.convert_to_tensor(
            self.neg_idx, dtype=tf.float32
        )
        self.final_iembedding = protein_embeddings
        self.final_uembedding = compound_embeddings
        self.u_embedding = tf.nn.embedding_lookup(self.final_hcedge, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_pdedge, self.v_idx)
        self.build_training_graph()

    @staticmethod
    def _clip_gradient_value(gradient, limit: float):
        if isinstance(gradient, tf.IndexedSlices):
            return tf.IndexedSlices(
                tf.clip_by_value(gradient.values, -limit, limit),
                gradient.indices,
                gradient.dense_shape,
            )
        return tf.clip_by_value(gradient, -limit, limit)

    def build_training_graph(self):
        if getattr(self, "_training_graph_built", False):
            return
        labels = tf.clip_by_value(self.neg_disease_embedding, 0.0, 1.0)
        if self.options.loss_kind == "legacy":
            u_emb_clipped = tf.clip_by_value(self.u_embedding, -100.0, 100.0)
            v_emb_clipped = tf.clip_by_value(self.v_embedding, -100.0, 100.0)
            sigmoid_input = tf.reduce_sum(
                tf.multiply(u_emb_clipped, v_emb_clipped), 1
            )
            sigmoid_input = tf.clip_by_value(sigmoid_input, -50.0, 50.0)
            sigmoid_output = tf.sigmoid(sigmoid_input)
            sigmoid_output = tf.clip_by_value(
                sigmoid_output, 1e-7, 1.0 - 1e-7
            )
            y = tf.reduce_sum(
                tf.multiply(labels, tf.log(sigmoid_output))
                + tf.multiply((1.0 - labels), tf.log(1.0 - sigmoid_output)),
                0,
            )
            y = tf.clip_by_value(y, -1e10, 1e10)
            data_loss_sum = -tf.reduce_sum(y)
        else:
            logits = tf.reduce_sum(self.u_embedding * self.v_embedding, axis=1)
            per_example = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            data_loss_sum = tf.reduce_sum(per_example)

        regularization_loss = 0
        for key in self.weights:
            regularization_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        regularization_loss += self.regU * (
            tf.nn.l2_loss(self.final_hcedge)
            + tf.nn.l2_loss(self.final_pdedge)
        )
        if self.options.loss_kind == "legacy":
            regularization_loss = tf.clip_by_value(
                regularization_loss, 0.0, 1e10
            )
        if self.options.loss_reduction == "mean":
            example_count = tf.maximum(
                tf.cast(tf.size(labels), tf.float32), 1.0
            )
            data_loss = data_loss_sum / example_count
        else:
            data_loss = data_loss_sum
        # ``mean`` refers only to the empirical BCE term. Regularisation keeps
        # its declared coefficient, rather than changing with batch size.
        loss = data_loss + regularization_loss
        if self.options.loss_kind == "legacy":
            loss = tf.clip_by_value(loss, -1e10, 1e10)

        optimizer = tf.train.AdamOptimizer(self.lRate)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in optimizer.compute_gradients(loss)
            if gradient is not None
        ]
        if not gradient_pairs:
            raise RuntimeError("ASTRARefined training graph has no gradients")
        raw_gradients = [pair[0] for pair in gradient_pairs]
        variables = [pair[1] for pair in gradient_pairs]
        self.preclip_gradient_norm = tf.global_norm(raw_gradients)
        gradients = raw_gradients
        if self.options.gradient_clip_value is not None:
            gradients = [
                self._clip_gradient_value(
                    gradient, self.options.gradient_clip_value
                )
                for gradient in gradients
            ]
        # This is the norm that actually enters clip_by_global_norm.  It can
        # differ from the raw/preclip norm when elementwise clipping is active.
        self.pre_global_clip_gradient_norm = tf.global_norm(gradients)
        if self.options.gradient_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.options.gradient_clip_norm
            )
        self.postclip_gradient_norm = tf.global_norm(gradients)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))
        self.training_loss = loss
        self.data_loss_sum = data_loss_sum
        self.data_loss = data_loss
        self.regularization_loss = regularization_loss
        self._training_graph_built = True

    def initialize_uninitialized_variables(self):
        self.build_training_graph()
        return initialize_uninitialized_variables(self.sess, tf)

    def _capture_epoch_snapshot(
        self,
        epoch: int,
        *,
        mean_training_loss: float,
        mean_preclip_gradient_norm: float | None,
        mean_pre_global_clip_gradient_norm: float | None,
        mean_postclip_gradient_norm: float | None,
        preclip_exceeds_clip_norm_fraction: float | None,
        global_clip_trigger_fraction: float | None,
        gradient_trace_batches: int,
        optimizer_steps: int,
    ) -> dict[str, object]:
        herb, disease = self.sess.run(
            [self.final_hcedge, self.final_pdedge],
            feed_dict={self.isTraining: 0},
        )
        snapshot: dict[str, object] = {
            "epoch": int(epoch),
            "u": np.asarray(herb).copy(),
            "i": np.asarray(disease).copy(),
            "mean_training_loss": float(mean_training_loss),
            "preclip_gradient_norm": (
                None
                if mean_preclip_gradient_norm is None
                else float(mean_preclip_gradient_norm)
            ),
            "mean_preclip_gradient_norm": (
                None
                if mean_preclip_gradient_norm is None
                else float(mean_preclip_gradient_norm)
            ),
            "mean_pre_global_clip_gradient_norm": (
                None
                if mean_pre_global_clip_gradient_norm is None
                else float(mean_pre_global_clip_gradient_norm)
            ),
            "mean_postclip_gradient_norm": (
                None
                if mean_postclip_gradient_norm is None
                else float(mean_postclip_gradient_norm)
            ),
            # The first ratio follows the requested raw-norm diagnostic; the
            # second is the exact activation rate of global clipping after any
            # elementwise clipping has been applied.
            "preclip_exceeds_clip_norm_fraction": (
                None
                if preclip_exceeds_clip_norm_fraction is None
                else float(preclip_exceeds_clip_norm_fraction)
            ),
            "global_clip_trigger_fraction": (
                None
                if global_clip_trigger_fraction is None
                else float(global_clip_trigger_fraction)
            ),
            "gradient_trace_batches": int(gradient_trace_batches),
            "optimizer_steps": int(optimizer_steps),
        }
        self.epoch_snapshots[int(epoch)] = snapshot
        return snapshot

    def select_epoch_snapshot(self, epoch: int) -> dict[str, object]:
        try:
            snapshot = self.epoch_snapshots[int(epoch)]
        except KeyError as error:
            raise ValueError(f"no retained checkpoint for epoch {epoch}") from error
        self.u = np.asarray(snapshot["u"]).copy()
        self.i = np.asarray(snapshot["i"]).copy()
        self.selected_epoch = int(epoch)
        return snapshot

    def trainModel(self):
        self.build_training_graph()
        self.initialize_uninitialized_variables()
        if self.maxEpoch != self.options.max_epochs:
            raise RuntimeError(
                "num.max.epoch changed after ASTRARefined configuration parsing"
            )
        checkpoints = set(self.options.checkpoint_epochs)
        optimizer_steps = 0
        for epoch_index in range(self.maxEpoch):
            losses: list[float] = []
            preclip_gradient_norms: list[float] = []
            pre_global_clip_gradient_norms: list[float] = []
            postclip_gradient_norms: list[float] = []
            for batch_index, batch in enumerate(self.next_batch_pairwise()):
                herb_idx, disease_idx, labels = batch
                feed_dict = {
                    self.u_idx: herb_idx,
                    self.neg_idx: labels,
                    self.v_idx: disease_idx,
                    self.isTraining: 1,
                }
                trace_gradients = (
                    self.options.trace_gradients
                    or not self.options.c0_compatible
                )
                if not trace_gradients:
                    _, loss_value = self.sess.run(
                        [self.train_op, self.training_loss], feed_dict=feed_dict
                    )
                else:
                    (
                        _,
                        loss_value,
                        preclip_norm_value,
                        pre_global_norm_value,
                        postclip_norm_value,
                    ) = self.sess.run(
                        [
                            self.train_op,
                            self.training_loss,
                            self.preclip_gradient_norm,
                            self.pre_global_clip_gradient_norm,
                            self.postclip_gradient_norm,
                        ],
                        feed_dict=feed_dict,
                    )
                optimizer_steps += 1
                losses.append(float(loss_value))
                if trace_gradients:
                    preclip_gradient_norms.append(float(preclip_norm_value))
                    pre_global_clip_gradient_norms.append(
                        float(pre_global_norm_value)
                    )
                    postclip_gradient_norms.append(float(postclip_norm_value))
                print(
                    "training:",
                    epoch_index + 1,
                    "batch",
                    batch_index,
                    "loss:",
                    loss_value,
                )
            if not losses:
                raise RuntimeError("ASTRARefined completed an epoch with no batches")
            epoch = epoch_index + 1
            if epoch in checkpoints:
                self._capture_epoch_snapshot(
                    epoch,
                    mean_training_loss=float(np.mean(losses)),
                    mean_preclip_gradient_norm=(
                        float(np.mean(preclip_gradient_norms))
                        if preclip_gradient_norms
                        else None
                    ),
                    mean_pre_global_clip_gradient_norm=(
                        float(np.mean(pre_global_clip_gradient_norms))
                        if pre_global_clip_gradient_norms
                        else None
                    ),
                    mean_postclip_gradient_norm=(
                        float(np.mean(postclip_gradient_norms))
                        if postclip_gradient_norms
                        else None
                    ),
                    preclip_exceeds_clip_norm_fraction=(
                        float(
                            np.mean(
                                np.asarray(preclip_gradient_norms)
                                > float(self.options.gradient_clip_norm)
                            )
                        )
                        if preclip_gradient_norms
                        and self.options.gradient_clip_norm is not None
                        else None
                    ),
                    global_clip_trigger_fraction=(
                        float(
                            np.mean(
                                np.asarray(pre_global_clip_gradient_norms)
                                > float(self.options.gradient_clip_norm)
                            )
                        )
                        if pre_global_clip_gradient_norms
                        and self.options.gradient_clip_norm is not None
                        else None
                    ),
                    gradient_trace_batches=len(preclip_gradient_norms),
                    optimizer_steps=optimizer_steps,
                )

        final_snapshot = self.select_epoch_snapshot(self.maxEpoch)
        self.weight = self.sess.run(self.weights, feed_dict={self.isTraining: 0})
        output_dir = _reproduction_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        np.savetxt(
            os.path.join(
                output_dir,
                "ASTRARefined_herb_embedding" + current_time + ".txt",
            ),
            final_snapshot["u"],
        )
        np.savetxt(
            os.path.join(
                output_dir,
                "ASTRARefined_disease_embedding" + current_time + ".txt",
            ),
            final_snapshot["i"],
        )

    def predictForRanking(self):
        return self.u.dot(self.i.transpose())


__all__ = [
    "ASTRARefined",
    "ASTRARefinedOptions",
    "PairwiseEdgePlan",
    "build_pairwise_edge_plan",
    "stable_binary_cross_entropy_numpy",
]
