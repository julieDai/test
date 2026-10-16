#!/usr/bin/env python3
"""Auditable HTInet-style HDA adaptation on the central fair protocol.

Wang et al. did not release an herb--disease implementation of HTInet.  The
HGHDA repository contains an HDA adaptation, but not the ``n2vnew_1`` graph,
walk, and embedding assets required by its scripts.  This runner therefore
reconstructs the documented pipeline while recording the boundary explicitly:

* a typed, undirected graph is built from H--C, C--P, and P--D relations only;
* H--D supervision is never inserted into the representation graph;
* p=q=1 random walks use 50 walks per node and length 50;
* deterministic 64-dimensional skip-gram uses window 10 and five epochs;
* the otherwise ambiguous released Word2Vec objective is fixed to
  ``sg=1, hs=1, negative=0`` with one worker;
* herb--disease pair features are endpoint Hadamard products; and
* either 5-NN or a 999-tree sklearn random forest is trained on the frozen
  inner-training rows, with the operating threshold selected on validation.

The result is a controlled Hu-et-al. HDA adaptation, not an author-code
reproduction of Wang et al.'s original heterogeneous herb--target method.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import time
from typing import Iterator, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from .common import (
        DEFAULT_CACHE_DIR,
        DEFAULT_THIRD_PARTY_DIR,
        array_sha256,
        balanced_prefix_indices,
        fold_split,
        load_protocol,
        project_git_identity,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )
except ImportError:  # Direct script execution with the package on PYTHONPATH.
    from common import (
        DEFAULT_CACHE_DIR,
        DEFAULT_THIRD_PARTY_DIR,
        array_sha256,
        balanced_prefix_indices,
        fold_split,
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


UPSTREAM_COMMIT = "c176902f4206f45c2d23bc1c684eeb0c229984cb"
UPSTREAM_SOURCE = "https://github.com/bioxjz/HGHDA"
METHOD_SOURCE = "HGHDA/HTInet adaptation; original HTInet has no released HDA code"
GENSIM_VERSION = "4.3.2"
FORMAL_SCHEDULE = {
    "num_walks": 50,
    "walk_length": 50,
    "embedding_epochs": 5,
}
WORD2VEC_PARAMETERS = {
    "vector_size": 64,
    "window": 10,
    "min_count": 0,
    "sg": 1,
    "hs": 1,
    "negative": 0,
    "workers": 1,
    "seed": 42,
}


@dataclass(frozen=True)
class TypedStaticGraph:
    """Canonical typed graph used by the HTInet representation stage."""

    offsets: dict[str, int]
    counts: dict[str, int]
    undirected_edges: np.ndarray
    adjacency: tuple[np.ndarray, ...]
    relation_edge_counts: dict[str, int]

    @property
    def n_nodes(self) -> int:
        return len(self.adjacency)


class UnbiasedWalkCorpus:
    """Re-iterable deterministic walk corpus, equivalent to node2vec p=q=1.

    Walk transitions are generated one complete node permutation at a time.
    Vectorising that operation is important for the released 50-by-50 walk
    budget: the Ethnobotany graph requires more than one hundred million
    transitions per corpus pass.  Integer tokens are accepted directly by
    Gensim and avoid a second, unnecessary string-conversion pass.
    """

    def __init__(
        self,
        adjacency: tuple[np.ndarray, ...],
        *,
        num_walks: int,
        walk_length: int,
        seed: int,
    ) -> None:
        if num_walks < 1 or walk_length < 1:
            raise ValueError("num_walks and walk_length must be positive")
        self.adjacency = adjacency
        self.num_walks = int(num_walks)
        self.walk_length = int(walk_length)
        self.seed = int(seed)

        degrees = np.fromiter(
            (len(neighbours) for neighbours in adjacency),
            dtype=np.int64,
            count=len(adjacency),
        )
        indptr = np.empty(len(adjacency) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(degrees, out=indptr[1:])
        indices = (
            np.concatenate(adjacency).astype(np.int64, copy=False)
            if int(indptr[-1])
            else np.empty(0, dtype=np.int64)
        )
        self.degrees = degrees
        self.indptr = indptr
        self.indices = indices

    def __iter__(self) -> Iterator[list[int]]:
        # Reinitialising the RNG makes every corpus pass identical.  This
        # matches the released two-stage workflow: generate a fixed walk file,
        # then reuse that file for every Word2Vec epoch.
        rng = np.random.default_rng(self.seed)
        nodes = np.arange(len(self.adjacency), dtype=np.int64)
        for _ in range(self.num_walks):
            starts = rng.permutation(nodes)
            traces = np.empty(
                (len(nodes), self.walk_length), dtype=np.int32
            )
            traces[:, 0] = starts
            current = starts
            active = self.degrees[current] > 0
            for step in range(1, self.walk_length):
                degree = self.degrees[current]
                # Drawing one uniform variate for every row keeps the RNG
                # stream independent of adjacency ordering and vectorises the
                # source implementation's uniform-neighbour transition.
                offset = np.floor(rng.random(len(nodes)) * degree).astype(
                    np.int64, copy=False
                )
                following = current.copy()
                following[active] = self.indices[
                    self.indptr[current[active]] + offset[active]
                ]
                traces[:, step] = following
                current = following
            for row, start in zip(traces, starts):
                if self.degrees[start] == 0:
                    yield [int(start)]
                else:
                    yield row.tolist()

    def __len__(self) -> int:
        return len(self.adjacency) * self.num_walks


def _canonical_undirected_edges(edges: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(edge, dtype=np.int64) for edge in edges if len(edge)]
    if not nonempty:
        return np.empty((0, 2), dtype=np.int64)
    joined = np.concatenate(nonempty, axis=0)
    if joined.ndim != 2 or joined.shape[1] != 2:
        raise ValueError("typed relation edges must have shape (n, 2)")
    joined = np.sort(joined, axis=1)
    if np.any(joined[:, 0] == joined[:, 1]):
        raise ValueError("typed relation graph unexpectedly contains a self-loop")
    return np.unique(joined, axis=0)


def build_static_heterogeneous_graph(protocol: object) -> TypedStaticGraph:
    """Build the label-free H--C--P--D graph with disjoint type offsets."""

    counts = {
        "herb": int(protocol.n_herbs),
        "compound": int(protocol.n_compounds),
        "protein": int(protocol.n_proteins),
        "disease": int(protocol.n_diseases),
    }
    offsets = {
        "herb": 0,
        "compound": counts["herb"],
        "protein": counts["herb"] + counts["compound"],
        "disease": counts["herb"] + counts["compound"] + counts["protein"],
    }

    herb_compound = np.asarray(protocol.herb_compound_pairs, dtype=np.int64)
    compound_protein = np.asarray(protocol.compound_protein_pairs, dtype=np.int64)
    protein_disease = np.asarray(protocol.protein_disease_pairs, dtype=np.int64)
    relations = {
        "herb_compound": np.column_stack(
            (herb_compound[:, 0] + offsets["herb"],
             herb_compound[:, 1] + offsets["compound"])
        ),
        "compound_protein": np.column_stack(
            (compound_protein[:, 0] + offsets["compound"],
             compound_protein[:, 1] + offsets["protein"])
        ),
        "protein_disease": np.column_stack(
            (protein_disease[:, 0] + offsets["protein"],
             protein_disease[:, 1] + offsets["disease"])
        ),
    }
    # No function argument accepts H--D pairs or labels.  This makes accidental
    # supervision leakage into the embedding graph structurally impossible.
    canonical_by_relation = {
        name: _canonical_undirected_edges((edge,))
        for name, edge in relations.items()
    }
    edges = _canonical_undirected_edges(tuple(canonical_by_relation.values()))
    n_nodes = sum(counts.values())
    if len(edges) and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError("typed graph contains an out-of-range compact identifier")
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for left, right in edges:
        neighbours[int(left)].append(int(right))
        neighbours[int(right)].append(int(left))
    adjacency = tuple(
        np.asarray(sorted(values), dtype=np.int64) for values in neighbours
    )
    return TypedStaticGraph(
        offsets=offsets,
        counts=counts,
        undirected_edges=edges,
        adjacency=adjacency,
        relation_edge_counts={
            name: int(len(edge)) for name, edge in canonical_by_relation.items()
        },
    )


def _validate_schedule(
    *, num_walks: int, walk_length: int, embedding_epochs: int, smoke: bool
) -> None:
    observed = {
        "num_walks": int(num_walks),
        "walk_length": int(walk_length),
        "embedding_epochs": int(embedding_epochs),
    }
    if any(value < 1 for value in observed.values()):
        raise ValueError("walk and embedding epoch parameters must be positive")
    if not smoke:
        mismatches = [
            f"{name}={observed[name]!r} (required {expected!r})"
            for name, expected in FORMAL_SCHEDULE.items()
            if observed[name] != expected
        ]
        if mismatches:
            raise ValueError(
                "formal HTInet adaptation runs lock the documented schedule: "
                + "; ".join(mismatches)
            )


def _validate_certified_compound_protein_support(protocol, *, smoke: bool) -> None:
    """Fail closed when Ethnobotany cannot certify its C--P relation.

    HTInet needs compound--protein edges to connect its herb--compound and
    protein--disease subnetworks.  The corrected Ethnobotany manifest records
    how many raw C--P identifiers can be joined through the authoritative
    PubChem mapping.  A non-smoke result is not eligible when any deduplicated
    C--P row remains unresolved; otherwise a severely truncated bridge could
    be emitted as an apparently formal baseline merely because the runner
    completed.  Smoke mode remains available for wiring checks, and is already
    marked formally ineligible by the runner.
    """

    if getattr(protocol, "dataset", None) != "ethnobotany":
        return
    strategy = getattr(protocol, "mapping_strategy", None)
    alignment = (
        strategy.get("c_p_alignment") if isinstance(strategy, Mapping) else None
    )
    if not isinstance(alignment, Mapping):
        raise ValueError(
            "Ethnobotany HTInet requires audited c_p_alignment metadata in "
            "the central protocol manifest"
        )
    required = (
        "deduplicated_input_rows",
        "retained_rows",
        "excluded_unmapped_rows",
        "collapsed_after_alias_rows",
    )
    missing = [name for name in required if name not in alignment]
    if missing:
        raise ValueError(
            "Ethnobotany c_p_alignment metadata is incomplete: "
            + ", ".join(missing)
        )
    try:
        deduplicated = int(alignment["deduplicated_input_rows"])
        retained = int(alignment["retained_rows"])
        excluded = int(alignment["excluded_unmapped_rows"])
        collapsed = int(alignment["collapsed_after_alias_rows"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Ethnobotany c_p_alignment counts must be integers"
        ) from error
    if min(deduplicated, retained, excluded, collapsed) < 0:
        raise ValueError("Ethnobotany c_p_alignment counts must be non-negative")
    actual_retained = int(len(protocol.compound_protein_pairs))
    if retained != actual_retained:
        raise ValueError(
            "Ethnobotany c_p_alignment retained_rows disagrees with the frozen "
            f"C--P array: {retained} != {actual_retained}"
        )
    if retained + excluded + collapsed != deduplicated:
        raise ValueError(
            "Ethnobotany c_p_alignment counts do not reconcile: retained + "
            "excluded + collapsed must equal deduplicated_input_rows"
        )
    if retained == 0:
        raise ValueError("Ethnobotany HTInet has no certified C--P bridge")
    if excluded and not smoke:
        raise ValueError(
            "formal Ethnobotany HTInet is disabled because the identity-safe "
            f"join certifies only {retained}/{deduplicated} C--P rows and "
            f"excludes {excluded} unresolved rows; use --smoke only for a "
            "formally ineligible wiring check"
        )


def _embedding_cache_identity(
    graph: TypedStaticGraph,
    *,
    num_walks: int,
    walk_length: int,
    embedding_epochs: int,
    gensim_version: str,
) -> dict[str, object]:
    return {
        "format": "htinet-static-embedding-v2-fixed-walk-file",
        "graph_edges_sha256": array_sha256(graph.undirected_edges),
        "node_count": graph.n_nodes,
        "type_offsets": graph.offsets,
        "entity_counts": graph.counts,
        "relation_edge_counts": graph.relation_edge_counts,
        "walk": {
            "p": 1,
            "q": 1,
            "num_walks": int(num_walks),
            "walk_length": int(walk_length),
            "directed": False,
            "weighted": False,
        },
        "word2vec": {
            **WORD2VEC_PARAMETERS,
            "epochs": int(embedding_epochs),
            "gensim_version": gensim_version,
        },
    }


def _import_word2vec():
    try:
        from gensim.models import Word2Vec
    except ImportError as error:
        raise RuntimeError(
            "HTInet embedding requires gensim==4.3.2; install ASTRA with "
            "the htinet optional dependency"
        ) from error
    observed = metadata.version("gensim")
    if observed != GENSIM_VERSION:
        raise RuntimeError(
            f"HTInet formal runtime requires gensim=={GENSIM_VERSION}, got {observed}"
        )
    return Word2Vec, observed


def _write_walk_corpus(
    path: Path,
    graph: TypedStaticGraph,
    *,
    num_walks: int,
    walk_length: int,
) -> dict[str, object]:
    """Materialise the fixed walk corpus once, as in the released pipeline."""

    corpus = UnbiasedWalkCorpus(
        graph.adjacency,
        num_walks=num_walks,
        walk_length=walk_length,
        seed=WORD2VEC_PARAMETERS["seed"],
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="ascii", buffering=1024 * 1024) as handle:
        for walk in corpus:
            handle.write(" ".join(map(str, walk)))
            handle.write("\n")
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "sentences": len(corpus),
    }


def _train_embeddings(
    graph: TypedStaticGraph,
    *,
    walk_corpus_path: Path,
    embedding_epochs: int,
) -> tuple[np.ndarray, str]:
    Word2Vec, gensim_version = _import_word2vec()
    model = Word2Vec(
        corpus_file=str(walk_corpus_path),
        vector_size=WORD2VEC_PARAMETERS["vector_size"],
        window=WORD2VEC_PARAMETERS["window"],
        min_count=WORD2VEC_PARAMETERS["min_count"],
        sg=WORD2VEC_PARAMETERS["sg"],
        hs=WORD2VEC_PARAMETERS["hs"],
        negative=WORD2VEC_PARAMETERS["negative"],
        workers=WORD2VEC_PARAMETERS["workers"],
        seed=WORD2VEC_PARAMETERS["seed"],
        epochs=embedding_epochs,
        sorted_vocab=1,
    )
    embeddings = np.vstack(
        [model.wv[str(node)] for node in range(graph.n_nodes)]
    ).astype(np.float32, copy=False)
    if embeddings.shape != (graph.n_nodes, WORD2VEC_PARAMETERS["vector_size"]):
        raise RuntimeError("Word2Vec returned an unexpected embedding shape")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Word2Vec embeddings contain NaN or infinity")
    return embeddings, gensim_version


def _load_or_train_embeddings(
    graph: TypedStaticGraph,
    *,
    cache_dir: Path,
    num_walks: int,
    walk_length: int,
    embedding_epochs: int,
) -> tuple[np.ndarray, dict[str, object], bool]:
    try:
        gensim_version = metadata.version("gensim")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "HTInet embedding requires gensim==4.3.2; install ASTRA with "
            "the htinet optional dependency"
        ) from error
    if gensim_version != GENSIM_VERSION:
        raise RuntimeError(
            f"HTInet formal runtime requires gensim=={GENSIM_VERSION}, got {gensim_version}"
        )
    identity = _embedding_cache_identity(
        graph,
        num_walks=num_walks,
        walk_length=walk_length,
        embedding_epochs=embedding_epochs,
        gensim_version=gensim_version,
    )
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    key = hashlib.sha256(encoded).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / f"{key}.npz"
    metadata_path = cache_dir / f"{key}.json"
    walk_path = cache_dir / f"{key}.walks.txt"
    if array_path.is_file() and metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise RuntimeError("HTInet embedding cache identity differs")
        if record.get("array_file_sha256") != sha256(array_path):
            raise RuntimeError("HTInet embedding cache file hash differs")
        with np.load(array_path, allow_pickle=False) as artifact:
            embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
        if record.get("embeddings_sha256") != array_sha256(embeddings):
            raise RuntimeError("HTInet cached embedding array hash differs")
        if embeddings.shape != (
            graph.n_nodes,
            WORD2VEC_PARAMETERS["vector_size"],
        ):
            raise RuntimeError("HTInet cached embedding shape differs")
        return embeddings, {**record, "cache_key": key}, True

    walk_record = _write_walk_corpus(
        walk_path,
        graph,
        num_walks=num_walks,
        walk_length=walk_length,
    )
    embeddings, trained_version = _train_embeddings(
        graph,
        walk_corpus_path=walk_path,
        embedding_epochs=embedding_epochs,
    )
    if trained_version != gensim_version:
        raise RuntimeError("gensim version changed during embedding training")
    temporary = array_path.with_name(array_path.name + ".tmp.npz")
    np.savez_compressed(temporary, embeddings=embeddings)
    temporary.replace(array_path)
    record = {
        "identity": identity,
        "array_file": str(array_path),
        "array_file_sha256": sha256(array_path),
        "embeddings_sha256": array_sha256(embeddings),
        "walk_corpus": walk_record,
    }
    write_json(metadata_path, record)
    return embeddings, {**record, "cache_key": key}, False


def pair_hadamard_features(
    pairs: np.ndarray,
    embeddings: np.ndarray,
    *,
    disease_offset: int,
) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("herb--disease pairs must have shape (n, 2)")
    herb = embeddings[pairs[:, 0]]
    disease = embeddings[pairs[:, 1] + int(disease_offset)]
    return np.multiply(herb, disease, dtype=np.float32)


def _build_classifier(classifier: str, *, fold: int):
    if classifier == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    if classifier == "rf":
        return RandomForestClassifier(
            n_estimators=999,
            random_state=42 + int(fold),
            n_jobs=-1,
        )
    raise ValueError(f"unsupported HTInet classifier: {classifier}")


def _positive_probability(classifier, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    classes = np.asarray(classifier.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise RuntimeError("HTInet classifier did not learn the positive class")
    return probabilities[:, int(positions[0])]


def _save_training_rows(
    path: Path,
    *,
    row_id: np.ndarray,
    pair: np.ndarray,
    label: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, row_id=row_id, pair=pair, label=label)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_THIRD_PARTY_DIR / "HGHDA_official",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "htinet")
    parser.add_argument("--classifier", choices=("knn", "rf"), required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Accepted for orchestrator compatibility; sklearn/gensim execute on CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-walks", type=int, default=50)
    parser.add_argument("--walk-length", type=int, default=50)
    parser.add_argument("--embedding-epochs", type=int, default=5)
    parser.add_argument("--smoke-rows-per-class", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-generation-id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("the central HTInet adaptation is frozen to seed 42")
    if args.smoke_rows_per_class < 1:
        parser.error("--smoke-rows-per-class must be positive")
    try:
        _validate_schedule(
            num_walks=args.num_walks,
            walk_length=args.walk_length,
            embedding_epochs=args.embedding_epochs,
            smoke=args.smoke,
        )
    except ValueError as error:
        parser.error(str(error))

    started = time.time()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.source_dir = args.source_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    source_files = tuple(
        args.source_dir / "HTInet" / name
        for name in ("e2v_walks.py", "learn_vecs.py", "KNN.py", "HTInet.py")
    )
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("HGHDA HTInet source files are missing: " + ", ".join(missing))

    project_source_identity = project_git_identity(require_clean=not args.smoke)
    protocol = load_protocol(args.protocol_manifest)
    if protocol.dataset != "ethnobotany":
        parser.error(
            "the controlled HTInet HDA adaptation currently supports Ethnobotany only"
        )
    if int(protocol.seed) != args.seed:
        parser.error(
            f"protocol seed {protocol.seed} differs from requested seed {args.seed}"
        )
    try:
        _validate_certified_compound_protein_support(
            protocol, smoke=bool(args.smoke)
        )
    except ValueError as error:
        parser.error(str(error))
    input_manifest = protocol.input_manifest()
    input_lock = validate_frozen_input_manifest(input_manifest)
    manifest_sha256 = sha256(args.protocol_manifest)
    fold_rows = protocol.fold_rows(args.fold)
    train_pairs, train_labels, train_row_id = fold_split(fold_rows, "train")
    validation_pairs, validation_labels, validation_row_id = fold_split(
        fold_rows, "validation"
    )
    test_pairs, test_labels, test_row_id = fold_split(fold_rows, "test")

    if args.smoke:
        train_selected = balanced_prefix_indices(
            train_labels, args.smoke_rows_per_class
        )
        validation_selected = balanced_prefix_indices(
            validation_labels, args.smoke_rows_per_class
        )
        test_selected = balanced_prefix_indices(test_labels, args.smoke_rows_per_class)
        train_pairs, train_labels, train_row_id = (
            train_pairs[train_selected],
            train_labels[train_selected],
            train_row_id[train_selected],
        )
        validation_pairs, validation_labels, validation_row_id = (
            validation_pairs[validation_selected],
            validation_labels[validation_selected],
            validation_row_id[validation_selected],
        )
        test_pairs, test_labels, test_row_id = (
            test_pairs[test_selected],
            test_labels[test_selected],
            test_row_id[test_selected],
        )

    graph = build_static_heterogeneous_graph(protocol)
    embeddings, cache_record, cache_hit = _load_or_train_embeddings(
        graph,
        cache_dir=args.cache_dir,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        embedding_epochs=args.embedding_epochs,
    )
    train_features = pair_hadamard_features(
        train_pairs, embeddings, disease_offset=graph.offsets["disease"]
    )
    validation_features = pair_hadamard_features(
        validation_pairs, embeddings, disease_offset=graph.offsets["disease"]
    )
    test_features = pair_hadamard_features(
        test_pairs, embeddings, disease_offset=graph.offsets["disease"]
    )
    classifier = _build_classifier(args.classifier, fold=args.fold)
    classifier.fit(train_features, np.asarray(train_labels, dtype=np.int64))
    validation_scores = _positive_probability(classifier, validation_features)
    test_scores = _positive_probability(classifier, test_features)
    metrics = evaluate_validation_and_test(
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
    )
    method = "HTInet-KN" if args.classifier == "knn" else "HTInet-RF"
    optimizer_steps = int(
        graph.n_nodes * args.num_walks * args.embedding_epochs
    )
    metrics.update(
        {
            "Model": method,
            "Dataset": str(protocol.dataset),
            "Fold": int(args.fold),
            "Seed": int(args.seed),
            "Device": "cpu",
            "Runtime_Seconds": time.time() - started,
            "Optimizer_Steps": optimizer_steps,
            "Optimizer_Steps_Definition": (
                "Word2Vec walk sequences presented across embedding epochs"
            ),
            "Training_Rows": int(len(train_pairs)),
            "Validation_Rows": int(len(validation_pairs)),
            "Test_Rows": int(len(test_pairs)),
            "Static_Graph_Nodes": graph.n_nodes,
            "Static_Graph_Undirected_Edges": int(len(graph.undirected_edges)),
            "Static_Graph_Herb_Disease_Edges": 0,
            "Embedding_Cache_Hit": bool(cache_hit),
            "Embedding_Cache_Key": cache_record["cache_key"],
            "Smoke": bool(args.smoke),
            "Formal_Eligible": bool(not args.smoke),
            "Execution_Mode": "smoke" if args.smoke else "formal-candidate",
            "Evaluation_Mode": "validation",
            "Protocol_Kind": (
                "controlled Hu-et-al HDA adaptation with label-free typed graph"
            ),
            "Upstream_Commit": UPSTREAM_COMMIT,
            "Upstream_Source": UPSTREAM_SOURCE,
            "Method_Source": METHOD_SOURCE,
            "Protocol_Manifest_SHA256": manifest_sha256,
        }
    )

    # Detect input mutation before admitting any artifacts.
    reloaded = load_protocol(args.protocol_manifest)
    if sha256(args.protocol_manifest) != manifest_sha256:
        raise RuntimeError("central protocol manifest changed during HTInet run")
    if validate_frozen_input_manifest(reloaded.input_manifest()) != input_lock:
        raise RuntimeError("central protocol inputs changed during HTInet run")

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
    _save_training_rows(
        result_dir / "training_rows.npz",
        row_id=train_row_id,
        pair=train_pairs,
        label=train_labels,
    )
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        args.output_root / "protocol.json",
        {
            "protocol_input_manifest": input_manifest,
            "frozen_input_lock_sha256": input_lock,
            "protocol_manifest": str(args.protocol_manifest),
            "protocol_manifest_sha256": manifest_sha256,
            "dataset": str(protocol.dataset),
            "run_generation_id": args.run_generation_id,
            "smoke": bool(args.smoke),
            "formal_eligible": bool(not args.smoke),
            "execution_mode": "smoke" if args.smoke else "formal-candidate",
            "model": metrics,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "graph": {
                "relations": ["herb_compound", "compound_protein", "protein_disease"],
                "herb_disease_edges": 0,
                "type_offsets": graph.offsets,
                "entity_counts": graph.counts,
                "relation_edge_counts": graph.relation_edge_counts,
                "undirected_edge_count": int(len(graph.undirected_edges)),
                "undirected_edges_sha256": array_sha256(graph.undirected_edges),
                "compound_protein_source": (
                    "protocol.compound_protein_pairs; compound identifiers are "
                    "joined only through unique PubChem IDs in the compound "
                    "mapping second column"
                ),
            },
            "embedding": {
                "walk": {
                    "p": 1,
                    "q": 1,
                    "num_walks": args.num_walks,
                    "walk_length": args.walk_length,
                    "directed": False,
                    "weighted": False,
                },
                "word2vec": {
                    **WORD2VEC_PARAMETERS,
                    "epochs": args.embedding_epochs,
                    "gensim_version": metadata.version("gensim"),
                },
                "source_ambiguity_resolution": (
                    "released learn_vecs.py sets negative=0 but omits hs; this "
                    "adapter explicitly uses hs=1 so skip-gram has a training objective"
                ),
                "cache": cache_record,
            },
            "classifier": {
                "kind": args.classifier,
                "knn_neighbors": 5 if args.classifier == "knn" else None,
                "rf_estimators": 999 if args.classifier == "rf" else None,
                "rf_random_state": 42 + args.fold if args.classifier == "rf" else None,
                "pair_features": "64-dimensional endpoint Hadamard product",
                "fit_split": "inner training only",
            },
            "threshold_selection": {
                "split": "validation",
                "rule": "maximum F1 over the common 100-threshold grid",
                "outer_test_labels_used": False,
            },
            "adapter_sources": {
                "common.py": sha256(Path(__file__).with_name("common.py")),
                "htinet_runner.py": sha256(Path(__file__)),
            },
            "upstream": {
                "repository": UPSTREAM_SOURCE,
                "commit": UPSTREAM_COMMIT,
                "role": METHOD_SOURCE,
                "source_sha256": {
                    str(path.relative_to(args.source_dir)): sha256(path)
                    for path in source_files
                },
            },
            "compatibility_adaptations": [
                "reconstructed missing n2vnew_1 graph/walk/embedding assets from the central compact relations",
                "used disjoint type offsets to prevent cross-type identifier collisions",
                "excluded all H--D edges from representation learning to prevent label leakage",
                "fixed random seeds and one Word2Vec worker for deterministic training",
                "used sklearn KNN/RF instead of the released cuML runtime",
                "selected the operating threshold on validation rather than fixing 0.5 or inspecting test labels",
            ],
            "trajectory_statement": (
                "controlled reconstruction of the HGHDA-distributed HTInet HDA "
                "pipeline; not Wang-author code and not an exact released trajectory"
            ),
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "training": "results/training_rows.npz",
                "prediction_fields": ["row_id", "pair", "label", "score"],
            },
            "project_source_identity": project_source_identity,
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
