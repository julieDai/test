"""Frozen, auditable data protocol for fair herb--disease benchmarks.

The protocol is deliberately model agnostic.  It is the single place where
raw identifiers are compacted, relation rows are de-duplicated, negatives are
frozen, and the nested train/validation/test splits are defined.  Baseline
adapters should consume :class:`BenchmarkProtocol` rather than independently
reimplementing any of these operations.

Two datasets are supported:

``ethnobotany``
    The first occurrence of every stored negative pair is retained.  If this
    leaves fewer negatives than recorded positives, seed-42 sampling without
    replacement from the *full* remaining herb--disease complement fills the
    deficit.

``tcm-suite``
    An explicit frozen negative file is mandatory.  After first-occurrence
    de-duplication it must contain exactly as many pairs as the positive set.

For both datasets positives and negatives are independently permuted with the
configured seed and assigned to five outer folds cyclically.  Ten percent of
each class in every outer-training partition is selected for validation with
a deterministic class-stratified permutation.  Validation positives are thus
never exposed through ``train_positive_pairs``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_FOLDS = 5
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_PAIR_ORDER = "seeded"
PAIR_ORDERS = ("seeded", "source")
ARRAY_FILENAME = "protocol_arrays.npz"
MAPPING_FILENAME = "entity_mappings.json"
MANIFEST_FILENAME = "manifest.json"

_RELATION_FILENAMES = {
    "H_D": ("H_D.txt", "H_D.csv", "H-D.txt", "H-D.csv"),
    "H_C": ("H_C.txt", "H_C.csv", "H-C.txt", "H-C.csv"),
    "C_P": ("C_P.txt", "C_P.csv", "C-P.txt", "C-P.csv"),
    "P_D": ("P_D.txt", "P_D.csv", "P-D.txt", "P-D.csv"),
}
_NEGATIVE_FILENAMES = (
    "negativehdlist.txt",
    "negativehdlist.csv",
    "negative_h_d.txt",
    "negative_h_d.csv",
)
_HEADER_PAIRS = {
    "H_D": {
        ("h", "d"),
        ("herb", "disease"),
        ("herb_id", "disease_id"),
        ("plant", "disease"),
        ("plant_id", "disease_id"),
    },
    "H_C": {
        ("h", "c"),
        ("herb", "compound"),
        ("herb_id", "compound_id"),
        ("plant", "compound"),
        ("plant_id", "compound_id"),
    },
    "C_P": {
        ("c", "p"),
        ("compound", "protein"),
        ("compound_id", "protein_id"),
    },
    "P_D": {
        ("p", "d"),
        ("protein", "disease"),
        ("protein_id", "disease_id"),
    },
}
_ARRAY_NAMES = (
    "positive_pairs",
    "negative_pairs",
    "herb_compound_pairs",
    "compound_protein_pairs",
    "protein_disease_pairs",
)
_ENTITY_NAMES = ("herb", "compound", "protein", "disease")
_MAPPING_FILENAMES = {
    "herb": "herb_id_mapping.csv",
    "compound": "compound_id_mapping.csv",
    "protein": "protein_id_mapping.csv",
    "disease": "disease_id_mapping.csv",
}


class ProtocolError(ValueError):
    """Raised when a benchmark cannot satisfy the frozen fair protocol."""


def _normalise_dataset(dataset: str) -> str:
    value = str(dataset).strip().lower().replace("_", "-")
    aliases = {
        "ethno": "ethnobotany",
        "ethnobotany": "ethnobotany",
        "tcm": "tcm-suite",
        "tcmsuite": "tcm-suite",
        "tcm-suite": "tcm-suite",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ProtocolError(
            f"unsupported dataset {dataset!r}; expected 'ethnobotany' or 'tcm-suite'"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _locate(data_dir: Path, relation: str) -> Path:
    for name in _RELATION_FILENAMES[relation]:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    expected = ", ".join(_RELATION_FILENAMES[relation])
    raise ProtocolError(f"missing {relation} relation in {data_dir}; tried {expected}")


def _locate_negative(data_dir: Path) -> Path:
    for name in _NEGATIVE_FILENAMES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    expected = ", ".join(_NEGATIVE_FILENAMES)
    raise ProtocolError(
        f"missing Ethnobotany stored-negative relation in {data_dir}; tried {expected}"
    )


def _clean_token(token: str) -> str:
    return token.strip().strip("\ufeff").strip().strip('"').strip("'")


def _read_raw_pairs(path: Path, relation: str) -> list[tuple[str, str]]:
    """Read the first two columns of a whitespace-, tab-, or comma-delimited file."""

    rows: list[tuple[str, str]] = []
    header_checked = False
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [_clean_token(value) for value in re.split(r"[\s,]+", stripped)]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                raise ProtocolError(
                    f"{path}:{line_number}: expected at least two non-empty columns"
                )
            pair = (parts[0], parts[1])
            if not header_checked:
                header_checked = True
                normalised = (pair[0].lower(), pair[1].lower())
                if normalised in _HEADER_PAIRS[relation]:
                    continue
            rows.append(pair)
    if not rows:
        raise ProtocolError(f"{path} contains no {relation} pairs")
    return rows


def _deduplicate(rows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in rows:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


_INTEGER_ID = re.compile(r"^[+-]?\d+$")


def _id_sort_key(value: str) -> tuple[int, object, str]:
    if _INTEGER_ID.fullmatch(value):
        return (0, int(value), value)
    return (1, value, value)


def _ordered_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=_id_sort_key))


def _first_occurrence_ids(*sequences: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sequence in sequences:
        for value in sequence:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
    return tuple(ordered)


def _read_entity_mapping(
    path: Path, entity: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read an Ethnobotany mapping and enforce its one-based row identity."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, csv.Error) as error:
        raise ProtocolError(f"cannot read {entity} mapping {path}: {error}") from error
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ProtocolError(f"{entity} mapping {path} must have a header and two columns")
    compact_ids: list[str] = []
    external_ids: list[str] = []
    for row_number, row in enumerate(rows[1:], start=1):
        if len(row) < 2:
            raise ProtocolError(
                f"{entity} mapping {path} row {row_number + 1} has fewer than two columns"
            )
        compact = _clean_token(row[0])
        external = _clean_token(row[1])
        if compact != str(row_number):
            raise ProtocolError(
                f"{entity} mapping {path} first column must be the consecutive "
                f"one-based row range; expected {row_number}, got {compact!r}"
            )
        if not external:
            raise ProtocolError(
                f"{entity} mapping {path} row {row_number + 1} has an empty external ID"
            )
        compact_ids.append(compact)
        external_ids.append(external)
    return tuple(compact_ids), tuple(external_ids)


def _compact(
    rows: Sequence[tuple[str, str]],
    left: Mapping[str, int],
    right: Mapping[str, int],
    relation: str,
) -> np.ndarray:
    compact: list[tuple[int, int]] = []
    for row_number, (left_id, right_id) in enumerate(rows, start=1):
        try:
            compact.append((left[left_id], right[right_id]))
        except KeyError as error:
            raise ProtocolError(
                f"{relation} row {row_number} contains ID {error.args[0]!r} "
                "outside the frozen entity mapping"
            ) from error
    return np.asarray(compact, dtype=np.int64).reshape(-1, 2)


def _pair_keys(pairs: np.ndarray, right_size: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * np.int64(right_size) + pairs[:, 1]


def _freeze(array: np.ndarray, *, dtype: np.dtype[Any] = np.dtype(np.int64)) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _validation_size(count: int, fraction: float) -> int:
    selected = int(math.floor(count * fraction))
    if fraction > 0.0 and count > 1:
        selected = max(1, selected)
    if selected >= count:
        selected = count - 1
    return selected


@dataclass(frozen=True)
class FoldRows:
    """One nested split, with stable row identifiers into ``all_pairs``."""

    fold: int
    train_pairs: np.ndarray
    train_labels: np.ndarray
    validation_pairs: np.ndarray
    validation_labels: np.ndarray
    test_pairs: np.ndarray
    test_labels: np.ndarray
    train_row_indices: np.ndarray
    validation_row_indices: np.ndarray
    test_row_indices: np.ndarray

    @property
    def train_positive_pairs(self) -> np.ndarray:
        """Positive H--D edges permitted in the model's training graph."""

        return self.train_pairs[self.train_labels == 1]

    @property
    def val_pairs(self) -> np.ndarray:
        return self.validation_pairs

    @property
    def val_labels(self) -> np.ndarray:
        return self.validation_labels

    @property
    def val_row_indices(self) -> np.ndarray:
        return self.validation_row_indices


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Compact relations and deterministic nested splits for one dataset."""

    dataset: str
    data_dir: Path
    embedding_dir: Path | None
    seed: int
    n_folds: int
    validation_fraction: float
    entity_ids: Mapping[str, tuple[str, ...]]
    positive_pairs: np.ndarray
    negative_pairs: np.ndarray
    herb_compound_pairs: np.ndarray
    compound_protein_pairs: np.ndarray
    protein_disease_pairs: np.ndarray
    source_files: Mapping[str, Mapping[str, object]]
    negative_sampling: Mapping[str, object]
    mapping_strategy: Mapping[str, object]
    pair_order: str = DEFAULT_PAIR_ORDER

    @property
    def n_herbs(self) -> int:
        return len(self.entity_ids["herb"])

    @property
    def n_compounds(self) -> int:
        return len(self.entity_ids["compound"])

    @property
    def n_proteins(self) -> int:
        return len(self.entity_ids["protein"])

    @property
    def n_diseases(self) -> int:
        return len(self.entity_ids["disease"])

    @property
    def all_pairs(self) -> np.ndarray:
        return np.concatenate((self.positive_pairs, self.negative_pairs), axis=0)

    @property
    def all_labels(self) -> np.ndarray:
        return np.concatenate(
            (
                np.ones(len(self.positive_pairs), dtype=np.uint8),
                np.zeros(len(self.negative_pairs), dtype=np.uint8),
            )
        )

    def raw_to_compact(self, entity: str, raw_id: object) -> int:
        entity = str(entity).strip().lower()
        if entity not in _ENTITY_NAMES:
            raise ProtocolError(f"unknown entity type {entity!r}")
        lookup = {value: index for index, value in enumerate(self.entity_ids[entity])}
        try:
            return lookup[str(raw_id)]
        except KeyError as error:
            raise ProtocolError(f"unknown {entity} raw identifier {raw_id!r}") from error

    def compact_to_raw(self, entity: str, compact_id: int) -> str:
        entity = str(entity).strip().lower()
        if entity not in _ENTITY_NAMES:
            raise ProtocolError(f"unknown entity type {entity!r}")
        try:
            return self.entity_ids[entity][int(compact_id)]
        except (IndexError, ValueError) as error:
            raise ProtocolError(
                f"{entity} compact identifier {compact_id!r} is outside the mapping"
            ) from error

    def fold_rows(self, fold: int, n_folds: int | None = None) -> FoldRows:
        """Return one outer fold and its training-only stratified validation split."""

        if n_folds is not None and int(n_folds) != self.n_folds:
            raise ProtocolError(
                f"protocol is frozen to {self.n_folds} folds, not {n_folds}"
            )
        if not 1 <= int(fold) <= self.n_folds:
            raise ProtocolError(f"fold must be in 1..{self.n_folds}, got {fold}")
        fold = int(fold)
        positive_count = len(self.positive_pairs)
        negative_count = len(self.negative_pairs)

        positive_rows = np.arange(positive_count, dtype=np.int64)
        negative_rows = positive_count + np.arange(negative_count, dtype=np.int64)
        positive_test = positive_rows[positive_rows % self.n_folds == fold - 1]
        negative_local = np.arange(negative_count, dtype=np.int64)
        negative_test = negative_rows[negative_local % self.n_folds == fold - 1]
        positive_outer_train = positive_rows[positive_rows % self.n_folds != fold - 1]
        negative_outer_train = negative_rows[negative_local % self.n_folds != fold - 1]

        # Each class is independently seeded.  Fold and class are included in
        # SeedSequence so validation membership is fixed yet distinct.
        positive_rng = np.random.default_rng(
            np.random.SeedSequence((self.seed, fold, 1))
        )
        negative_rng = np.random.default_rng(
            np.random.SeedSequence((self.seed, fold, 0))
        )
        positive_outer_train = positive_outer_train[
            positive_rng.permutation(len(positive_outer_train))
        ]
        negative_outer_train = negative_outer_train[
            negative_rng.permutation(len(negative_outer_train))
        ]
        positive_val_count = _validation_size(
            len(positive_outer_train), self.validation_fraction
        )
        negative_val_count = _validation_size(
            len(negative_outer_train), self.validation_fraction
        )
        validation_rows = np.concatenate(
            (
                positive_outer_train[:positive_val_count],
                negative_outer_train[:negative_val_count],
            )
        )
        train_rows = np.concatenate(
            (
                positive_outer_train[positive_val_count:],
                negative_outer_train[negative_val_count:],
            )
        )
        test_rows = np.concatenate((positive_test, negative_test))
        pairs = self.all_pairs
        labels = self.all_labels
        return FoldRows(
            fold=fold,
            train_pairs=_freeze(pairs[train_rows]),
            train_labels=_freeze(labels[train_rows], dtype=np.dtype(np.uint8)),
            validation_pairs=_freeze(pairs[validation_rows]),
            validation_labels=_freeze(
                labels[validation_rows], dtype=np.dtype(np.uint8)
            ),
            test_pairs=_freeze(pairs[test_rows]),
            test_labels=_freeze(labels[test_rows], dtype=np.dtype(np.uint8)),
            train_row_indices=_freeze(train_rows),
            validation_row_indices=_freeze(validation_rows),
            test_row_indices=_freeze(test_rows),
        )

    def fold(self, fold: int) -> FoldRows:
        return self.fold_rows(fold)

    def incidence_matrices(self) -> tuple[Any, Any]:
        """Return herb--compound and disease--protein CSR incidence matrices."""

        try:
            from scipy import sparse
        except ImportError as error:  # pragma: no cover - formal env has scipy
            raise RuntimeError("scipy is required to construct incidence matrices") from error
        hc_values = np.ones(len(self.herb_compound_pairs), dtype=np.float32)
        herb_compound = sparse.csr_matrix(
            (
                hc_values,
                (
                    self.herb_compound_pairs[:, 0],
                    self.herb_compound_pairs[:, 1],
                ),
            ),
            shape=(self.n_herbs, self.n_compounds),
            dtype=np.float32,
        )
        pd_values = np.ones(len(self.protein_disease_pairs), dtype=np.float32)
        disease_protein = sparse.csr_matrix(
            (
                pd_values,
                (
                    self.protein_disease_pairs[:, 1],
                    self.protein_disease_pairs[:, 0],
                ),
            ),
            shape=(self.n_diseases, self.n_proteins),
            dtype=np.float32,
        )
        return herb_compound, disease_protein

    def _fold_manifest(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for fold in range(1, self.n_folds + 1):
            rows = self.fold_rows(fold)
            record: dict[str, object] = {"fold": fold}
            for name, indices, labels in (
                ("train", rows.train_row_indices, rows.train_labels),
                (
                    "validation",
                    rows.validation_row_indices,
                    rows.validation_labels,
                ),
                ("test", rows.test_row_indices, rows.test_labels),
            ):
                record[name] = {
                    "rows": int(len(indices)),
                    "positive_rows": int(labels.sum()),
                    "negative_rows": int(len(labels) - labels.sum()),
                    "row_indices_sha256": _array_digest(indices),
                }
            records.append(record)
        return records

    def input_manifest(self) -> dict[str, object]:
        """Return the explicit, JSON-serialisable protocol manifest."""

        arrays = {
            name: {
                "shape": list(getattr(self, name).shape),
                "dtype": str(getattr(self, name).dtype),
                "sha256": _array_digest(getattr(self, name)),
            }
            for name in _ARRAY_NAMES
        }
        mappings = {
            entity: {
                "count": len(self.entity_ids[entity]),
                "sha256": hashlib.sha256(
                    _json_bytes(list(self.entity_ids[entity]))
                ).hexdigest(),
            }
            for entity in _ENTITY_NAMES
        }
        return {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": self.dataset,
            "seed": self.seed,
            "id_convention_on_disk": "raw identifiers",
            "id_convention_in_adapters": "zero-based compact identifiers",
            "fold_rule": (
                (
                    f"independent seed-{self.seed} positive/negative permutations; "
                    "class-local index modulo 5; outer-train-only stratified validation"
                )
                if self.pair_order == "seeded"
                else (
                    "source-order de-duplicated positive/negative rows; class-local "
                    "index modulo 5; outer-train-only stratified validation"
                )
            ),
            "pair_order": self.pair_order,
            "n_folds": self.n_folds,
            "validation": {
                "fraction_per_class": self.validation_fraction,
                "rounding": "floor; minimum one when a class has at least two rows",
                "selection": "deterministic SeedSequence(seed, fold, class)",
                "positive_edges_in_training_graph": False,
            },
            "counts": {
                "herbs": self.n_herbs,
                "compounds": self.n_compounds,
                "proteins": self.n_proteins,
                "diseases": self.n_diseases,
                "positive_rows": len(self.positive_pairs),
                "negative_rows": len(self.negative_pairs),
                "herb_compound_rows": len(self.herb_compound_pairs),
                "compound_protein_rows": len(self.compound_protein_pairs),
                "protein_disease_rows": len(self.protein_disease_pairs),
            },
            "negative_sampling": dict(self.negative_sampling),
            "mapping_strategy": dict(self.mapping_strategy),
            "embedding_dir": (
                str(self.embedding_dir.resolve()) if self.embedding_dir is not None else None
            ),
            "files": {key: dict(value) for key, value in self.source_files.items()},
            "arrays": arrays,
            "mappings": mappings,
            "folds": self._fold_manifest(),
        }

    def validate(self) -> None:
        """Fail closed on duplicate, overlap, bounds, or split leakage."""

        if self.dataset not in {"ethnobotany", "tcm-suite"}:
            raise ProtocolError(f"invalid normalised dataset {self.dataset!r}")
        if self.n_folds != DEFAULT_FOLDS:
            raise ProtocolError("the formal protocol requires exactly five folds")
        if self.pair_order not in PAIR_ORDERS:
            raise ProtocolError(
                f"pair_order must be one of {PAIR_ORDERS}, got {self.pair_order!r}"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ProtocolError("validation_fraction must lie strictly between 0 and 1")
        if tuple(self.entity_ids) != _ENTITY_NAMES:
            raise ProtocolError(
                f"entity mappings must be ordered as {_ENTITY_NAMES}, got {tuple(self.entity_ids)}"
            )
        if not self.mapping_strategy:
            raise ProtocolError("mapping_strategy must explicitly document entity row order")
        for entity in _ENTITY_NAMES:
            ids = self.entity_ids[entity]
            if len(ids) != len(set(ids)):
                raise ProtocolError(f"{entity} raw-to-compact mapping contains duplicates")

        bounds = {
            "positive_pairs": (self.n_herbs, self.n_diseases),
            "negative_pairs": (self.n_herbs, self.n_diseases),
            "herb_compound_pairs": (self.n_herbs, self.n_compounds),
            "compound_protein_pairs": (self.n_compounds, self.n_proteins),
            "protein_disease_pairs": (self.n_proteins, self.n_diseases),
        }
        for name, shape in bounds.items():
            pairs = getattr(self, name)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ProtocolError(f"{name} must have shape (n, 2), got {pairs.shape}")
            if len(pairs) == 0:
                raise ProtocolError(f"{name} must not be empty")
            if not np.issubdtype(pairs.dtype, np.integer):
                raise ProtocolError(f"{name} must use an integer dtype")
            if int(pairs.min()) < 0 or np.any(pairs.max(axis=0) >= np.asarray(shape)):
                raise ProtocolError(f"{name} contains compact identifiers outside {shape}")
            keys = _pair_keys(pairs, shape[1])
            if np.unique(keys).size != len(keys):
                raise ProtocolError(f"{name} contains duplicate compact pairs")

        if len(self.positive_pairs) != len(self.negative_pairs):
            raise ProtocolError("positive and negative pair counts must be balanced")
        positive_keys = set(_pair_keys(self.positive_pairs, self.n_diseases).tolist())
        negative_keys = set(_pair_keys(self.negative_pairs, self.n_diseases).tolist())
        overlap = positive_keys.intersection(negative_keys)
        if overlap:
            raise ProtocolError(
                f"negative samples overlap {len(overlap)} recorded positive pairs"
            )

        all_rows = set(range(len(self.all_pairs)))
        test_occurrences: list[int] = []
        for fold in range(1, self.n_folds + 1):
            rows = self.fold_rows(fold)
            row_sets = {
                "train": set(rows.train_row_indices.tolist()),
                "validation": set(rows.validation_row_indices.tolist()),
                "test": set(rows.test_row_indices.tolist()),
            }
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                if row_sets[first].intersection(row_sets[second]):
                    raise ProtocolError(
                        f"fold {fold} has row overlap between {first} and {second}"
                    )
            if set().union(*row_sets.values()) != all_rows:
                raise ProtocolError(f"fold {fold} does not partition all association rows")

            pair_sets = {
                "train": set(
                    _pair_keys(rows.train_pairs, self.n_diseases).tolist()
                ),
                "validation": set(
                    _pair_keys(rows.validation_pairs, self.n_diseases).tolist()
                ),
                "test": set(_pair_keys(rows.test_pairs, self.n_diseases).tolist()),
            }
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                if pair_sets[first].intersection(pair_sets[second]):
                    raise ProtocolError(
                        f"fold {fold} has H-D pair overlap between {first} and {second}"
                    )
            validation_positive = set(
                _pair_keys(
                    rows.validation_pairs[rows.validation_labels == 1],
                    self.n_diseases,
                ).tolist()
            )
            train_graph = set(
                _pair_keys(rows.train_positive_pairs, self.n_diseases).tolist()
            )
            if validation_positive.intersection(train_graph):
                raise ProtocolError(
                    f"fold {fold} validation positives leaked into the training graph"
                )
            test_occurrences.extend(rows.test_row_indices.tolist())
        if sorted(test_occurrences) != sorted(all_rows):
            raise ProtocolError("outer test folds do not cover every association exactly once")

    def save(self, output_dir: Path | str) -> Path:
        """Save arrays, raw-ID mappings, and an integrity-checked manifest."""

        self.validate()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        array_path = output_dir / ARRAY_FILENAME
        mapping_path = output_dir / MAPPING_FILENAME
        manifest_path = output_dir / MANIFEST_FILENAME
        _atomic_npz(
            array_path,
            {name: getattr(self, name) for name in _ARRAY_NAMES},
        )
        mapping_payload = {
            "id_convention": "list index is the zero-based compact identifier",
            "entities": {
                entity: list(self.entity_ids[entity]) for entity in _ENTITY_NAMES
            },
        }
        _atomic_bytes(mapping_path, _json_bytes(mapping_payload))
        manifest = self.input_manifest()
        manifest["artifacts"] = {
            ARRAY_FILENAME: {
                "bytes": array_path.stat().st_size,
                "sha256": _sha256(array_path),
            },
            MAPPING_FILENAME: {
                "bytes": mapping_path.stat().st_size,
                "sha256": _sha256(mapping_path),
            },
        }
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        return manifest_path

    @classmethod
    def load(cls, manifest_or_directory: Path | str) -> "BenchmarkProtocol":
        return load_protocol_manifest(manifest_or_directory)


def _source_record(
    path: Path, *, raw_rows: int, unique_rows: int
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "raw_rows": int(raw_rows),
        "unique_rows": int(unique_rows),
    }


def _nonempty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return sum(bool(line.strip()) for line in handle)


def build_protocol(
    dataset: str,
    data_dir: Path | str,
    *,
    negative_file: Path | str | None = None,
    embedding_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    pair_order: str = DEFAULT_PAIR_ORDER,
) -> BenchmarkProtocol:
    """Build and validate a fair protocol from raw relation files."""

    dataset = _normalise_dataset(dataset)
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ProtocolError(f"data directory does not exist: {data_dir}")
    if int(folds) != DEFAULT_FOLDS:
        raise ProtocolError("the formal protocol requires exactly five folds")
    if int(seed) < 0:
        raise ProtocolError("seed must be a non-negative integer")
    pair_order = str(pair_order).strip().lower()
    if pair_order not in PAIR_ORDERS:
        raise ProtocolError(
            f"pair_order must be one of {PAIR_ORDERS}, got {pair_order!r}"
        )
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ProtocolError("validation_fraction must lie strictly between 0 and 1")
    resolved_embedding_dir = (
        Path(embedding_dir).resolve() if embedding_dir is not None else None
    )
    if resolved_embedding_dir is not None and not resolved_embedding_dir.is_dir():
        raise ProtocolError(
            f"embedding directory does not exist: {resolved_embedding_dir}"
        )

    paths = {relation: _locate(data_dir, relation) for relation in _RELATION_FILENAMES}
    if dataset == "tcm-suite":
        if negative_file is None:
            raise ProtocolError(
                "tcm-suite requires an explicit frozen negative_file; no implicit file is accepted"
            )
        negative_path = Path(negative_file).resolve()
        if not negative_path.is_file():
            raise ProtocolError(f"frozen TCM negative file does not exist: {negative_path}")
    else:
        negative_path = (
            Path(negative_file).resolve()
            if negative_file is not None
            else _locate_negative(data_dir)
        )
        if not negative_path.is_file():
            raise ProtocolError(f"Ethnobotany negative file does not exist: {negative_path}")

    raw: dict[str, list[tuple[str, str]]] = {}
    unique: dict[str, list[tuple[str, str]]] = {}
    source_files: dict[str, dict[str, object]] = {}
    for relation, path in paths.items():
        raw[relation] = _read_raw_pairs(path, relation)
        unique[relation] = _deduplicate(raw[relation])
        source_files[relation] = _source_record(
            path, raw_rows=len(raw[relation]), unique_rows=len(unique[relation])
        )
    negative_raw = _read_raw_pairs(negative_path, "H_D")
    negative_unique = _deduplicate(negative_raw)
    source_files["negative_H_D"] = _source_record(
        negative_path,
        raw_rows=len(negative_raw),
        unique_rows=len(negative_unique),
    )

    compound_external_lookup: dict[str, int] | None = None
    if dataset == "ethnobotany":
        entity_ids = {}
        external_ids: dict[str, tuple[str, ...]] = {}
        for entity in _ENTITY_NAMES:
            mapping_path = data_dir / _MAPPING_FILENAMES[entity]
            if not mapping_path.is_file():
                raise ProtocolError(
                    "Ethnobotany requires explicit entity mapping CSVs; missing "
                    f"{mapping_path}"
                )
            compact_ids, mapped_external_ids = _read_entity_mapping(
                mapping_path, entity
            )
            entity_ids[entity] = compact_ids
            external_ids[entity] = mapped_external_ids
            source_files[f"mapping/{entity}"] = _source_record(
                mapping_path,
                raw_rows=len(compact_ids),
                unique_rows=len(set(compact_ids)),
            )
        external_counts: dict[str, int] = {}
        for external_id in external_ids["compound"]:
            external_counts[external_id] = external_counts.get(external_id, 0) + 1
        compound_external_lookup = {
            external_id: index
            for index, external_id in enumerate(external_ids["compound"])
            if external_counts[external_id] == 1
        }
        mapping_strategy: dict[str, object] = {
            "dataset_policy": "authoritative Ethnobotany mapping CSVs",
            "entity_order": {
                entity: (
                    f"{_MAPPING_FILENAMES[entity]} first-column consecutive row order"
                )
                for entity in _ENTITY_NAMES
            },
            "relation_id_spaces": {
                "H_D": "herb and disease mapping first-column IDs",
                "H_C": "herb and compound mapping first-column IDs",
                "P_D": "protein and disease mapping first-column IDs",
                "C_P": (
                    "compound uses only a unique PubChem ID from the compound "
                    "mapping second column; protein uses the mapping first-column "
                    "ID; compact compound IDs and unresolved PubChem IDs are excluded"
                ),
            },
        }
    else:
        # These are the exact row orders used to create the TCM precomputed
        # biological features.  IDs seen only outside the structural source
        # are appended deterministically and recorded, never numerically sorted.
        herb_base = _first_occurrence_ids(a for a, _ in unique["H_C"])
        compound_base = _first_occurrence_ids(b for _, b in unique["H_C"])
        protein_base = _first_occurrence_ids(b for _, b in unique["C_P"])
        disease_base = _first_occurrence_ids(b for _, b in unique["P_D"])
        entity_ids = {
            "herb": _first_occurrence_ids(
                herb_base,
                (a for a, _ in unique["H_D"]),
                (a for a, _ in negative_unique),
            ),
            "compound": _first_occurrence_ids(
                compound_base, (a for a, _ in unique["C_P"])
            ),
            "protein": _first_occurrence_ids(
                protein_base, (a for a, _ in unique["P_D"])
            ),
            "disease": _first_occurrence_ids(
                disease_base,
                (b for _, b in unique["H_D"]),
                (b for _, b in negative_unique),
            ),
        }
        bases = {
            "herb": herb_base,
            "compound": compound_base,
            "protein": protein_base,
            "disease": disease_base,
        }
        mapping_strategy = {
            "dataset_policy": "TCM feature-generation first-occurrence order",
            "entity_order": {
                "herb": "H_C first column, then unseen H_D/negative IDs",
                "compound": "H_C second column, then unseen C_P IDs",
                "protein": "C_P second column, then unseen P_D IDs",
                "disease": "P_D second column, then unseen H_D/negative IDs",
            },
            "base_feature_rows": {
                entity: len(bases[entity]) for entity in _ENTITY_NAMES
            },
            "appended_ids": {
                entity: len(entity_ids[entity]) - len(bases[entity])
                for entity in _ENTITY_NAMES
            },
        }
    lookups = {
        entity: {raw_id: index for index, raw_id in enumerate(entity_ids[entity])}
        for entity in _ENTITY_NAMES
    }
    positive = _compact(
        unique["H_D"], lookups["herb"], lookups["disease"], "H_D"
    )
    stored_negative = _compact(
        negative_unique,
        lookups["herb"],
        lookups["disease"],
        "negative_H_D",
    )
    herb_compound = _compact(
        unique["H_C"], lookups["herb"], lookups["compound"], "H_C"
    )
    if compound_external_lookup is not None:
        mapped_compact_rows: list[tuple[int, int]] = []
        unmapped_examples: list[str] = []
        unmapped_cp_rows = 0
        compact_id_collision_rows = 0
        for compound_id, protein_id in unique["C_P"]:
            try:
                protein_index = lookups["protein"][protein_id]
            except KeyError as error:
                raise ProtocolError(
                    f"C_P contains protein ID {protein_id!r} outside the frozen "
                    "protein mapping"
                ) from error
            if compound_id in compound_external_lookup:
                compound_index = compound_external_lookup[compound_id]
            else:
                unmapped_cp_rows += 1
                if compound_id in lookups["compound"]:
                    compact_id_collision_rows += 1
                if len(unmapped_examples) < 10:
                    unmapped_examples.append(compound_id)
                continue
            mapped_compact_rows.append((compound_index, protein_index))
        retained_compact_rows = list(dict.fromkeys(mapped_compact_rows))
        if not retained_compact_rows:
            raise ProtocolError(
                "no Ethnobotany C-P compound identifier maps through the "
                "unique PubChem IDs in the compound mapping second column"
            )
        compound_protein = np.asarray(retained_compact_rows, dtype=np.int64)
        mapping_strategy["c_p_alignment"] = {
            "raw_input_rows": len(raw["C_P"]),
            "deduplicated_input_rows": len(unique["C_P"]),
            "retained_rows": len(retained_compact_rows),
            "excluded_unmapped_rows": unmapped_cp_rows,
            "excluded_compact_id_collision_rows": compact_id_collision_rows,
            "collapsed_after_alias_rows": (
                len(mapped_compact_rows) - len(retained_compact_rows)
            ),
            "drop_reason": (
                "compound ID did not match a unique PubChem ID in the compound "
                "mapping second column; compact first-column numeric collisions "
                "are not accepted"
            ),
            "dropped_compound_id_examples": unmapped_examples,
        }
    else:
        compound_protein = _compact(
            unique["C_P"],
            lookups["compound"],
            lookups["protein"],
            "C_P",
        )
    protein_disease = _compact(
        unique["P_D"], lookups["protein"], lookups["disease"], "P_D"
    )

    if resolved_embedding_dir is not None:
        for entity, filename in (
            ("compound", "compound_embedding_kpgt.txt"),
            ("protein", "protein_embedding_esm2.txt"),
        ):
            feature_path = resolved_embedding_dir / filename
            if not feature_path.is_file():
                raise ProtocolError(f"missing {entity} feature matrix: {feature_path}")
            rows = _nonempty_line_count(feature_path)
            expected = len(entity_ids[entity])
            if rows != expected:
                raise ProtocolError(
                    f"{entity} feature matrix has {rows} rows but the frozen "
                    f"mapping has {expected}; row alignment is not provable"
                )
            source_files[f"embedding/{entity}"] = _source_record(
                feature_path, raw_rows=rows, unique_rows=rows
            )

    positive_keys = set(_pair_keys(positive, len(entity_ids["disease"])).tolist())
    stored_negative_keys = set(
        _pair_keys(stored_negative, len(entity_ids["disease"])).tolist()
    )
    overlap = positive_keys.intersection(stored_negative_keys)
    if overlap:
        raise ProtocolError(
            f"stored negatives overlap {len(overlap)} de-duplicated H-D positives"
        )

    supplemented = 0
    if dataset == "ethnobotany":
        if len(stored_negative) > len(positive):
            raise ProtocolError(
                "Ethnobotany has more unique stored negatives than positives; "
                "the first-occurrence retention rule cannot form a balanced set"
            )
        supplemented = len(positive) - len(stored_negative)
        if supplemented:
            n_diseases = len(entity_ids["disease"])
            candidate_count = len(entity_ids["herb"]) * n_diseases
            available = np.ones(candidate_count, dtype=bool)
            available[np.fromiter(positive_keys, dtype=np.int64)] = False
            available[np.fromiter(stored_negative_keys, dtype=np.int64)] = False
            complement = np.flatnonzero(available)
            if supplemented > len(complement):
                raise ProtocolError(
                    f"full H-D complement has only {len(complement)} available pairs; "
                    f"cannot supplement {supplemented} negatives"
                )
            sampled_keys = np.random.default_rng(int(seed)).choice(
                complement, size=supplemented, replace=False
            )
            sampled = np.column_stack(
                (sampled_keys // n_diseases, sampled_keys % n_diseases)
            ).astype(np.int64, copy=False)
            negative = np.concatenate((stored_negative, sampled), axis=0)
        else:
            negative = stored_negative
        negative_sampling: dict[str, object] = {
            "policy": (
                "retain first occurrence of stored negatives, then sample without "
                "replacement from the full remaining H-D complement"
            ),
            "seed": int(seed),
            "stored_raw_rows": len(negative_raw),
            "stored_unique_rows": len(stored_negative),
            "supplemented_rows": supplemented,
            "final_rows": len(negative),
        }
    else:
        if len(stored_negative) != len(positive):
            raise ProtocolError(
                "the de-duplicated explicit TCM frozen negative file must contain "
                f"exactly {len(positive)} pairs, got {len(stored_negative)}"
            )
        negative = stored_negative
        negative_sampling = {
            "policy": "explicit frozen negative file; first occurrence retained",
            "seed": None,
            "stored_raw_rows": len(negative_raw),
            "stored_unique_rows": len(stored_negative),
            "supplemented_rows": 0,
            "final_rows": len(negative),
        }

    if pair_order == "seeded":
        # The fair protocol independently randomises both classes before
        # cyclic assignment.  ``source`` is retained only for matched legacy
        # comparisons whose published split was defined on source row order.
        positive_permutation = np.random.default_rng(int(seed)).permutation(
            len(positive)
        )
        negative_permutation = np.random.default_rng(int(seed)).permutation(
            len(negative)
        )
        positive = positive[positive_permutation]
        negative = negative[negative_permutation]

    protocol = BenchmarkProtocol(
        dataset=dataset,
        data_dir=data_dir,
        embedding_dir=resolved_embedding_dir,
        seed=int(seed),
        n_folds=int(folds),
        validation_fraction=float(validation_fraction),
        entity_ids={entity: entity_ids[entity] for entity in _ENTITY_NAMES},
        positive_pairs=_freeze(positive),
        negative_pairs=_freeze(negative),
        herb_compound_pairs=_freeze(herb_compound),
        compound_protein_pairs=_freeze(compound_protein),
        protein_disease_pairs=_freeze(protein_disease),
        source_files=source_files,
        negative_sampling=negative_sampling,
        mapping_strategy=mapping_strategy,
        pair_order=pair_order,
    )
    protocol.validate()
    return protocol


def _compare_manifest_section(
    name: str, expected: object, actual: object
) -> None:
    if expected != actual:
        raise ProtocolError(f"saved manifest integrity check failed at {name}")


def load_protocol_manifest(
    manifest_or_directory: Path | str,
) -> BenchmarkProtocol:
    """Reload a protocol package and verify every saved digest and split."""

    supplied = Path(manifest_or_directory).resolve()
    manifest_path = supplied / MANIFEST_FILENAME if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise ProtocolError(f"protocol manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read protocol manifest {manifest_path}: {error}") from error
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {manifest.get('protocol_version')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProtocolError("protocol manifest has no artifact integrity section")
    root = manifest_path.parent
    for name in (ARRAY_FILENAME, MAPPING_FILENAME):
        path = root / name
        record = artifacts.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ProtocolError(f"protocol artifact is missing: {path}")
        if path.stat().st_size != record.get("bytes") or _sha256(path) != record.get(
            "sha256"
        ):
            raise ProtocolError(f"protocol artifact hash/size mismatch: {path}")

    try:
        mapping_document = json.loads(
            (root / MAPPING_FILENAME).read_text(encoding="utf-8")
        )
        mapping_values = mapping_document["entities"]
        entity_ids = {
            entity: tuple(str(value) for value in mapping_values[entity])
            for entity in _ENTITY_NAMES
        }
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise ProtocolError(f"invalid entity mapping artifact: {error}") from error
    try:
        with np.load(root / ARRAY_FILENAME, allow_pickle=False) as archive:
            if set(archive.files) != set(_ARRAY_NAMES):
                raise ProtocolError(
                    "array artifact members differ from the frozen protocol schema"
                )
            arrays = {name: _freeze(archive[name]) for name in _ARRAY_NAMES}
    except (OSError, ValueError) as error:
        raise ProtocolError(f"invalid protocol array artifact: {error}") from error

    protocol = BenchmarkProtocol(
        dataset=_normalise_dataset(manifest.get("dataset", "")),
        data_dir=Path(
            next(
                (
                    Path(record["path"]).parent
                    for record in manifest.get("files", {}).values()
                    if isinstance(record, dict) and "path" in record
                ),
                root,
            )
        ),
        embedding_dir=(
            Path(manifest["embedding_dir"])
            if manifest.get("embedding_dir") is not None
            else None
        ),
        seed=int(manifest.get("seed")),
        n_folds=int(manifest.get("n_folds")),
        validation_fraction=float(manifest.get("validation", {}).get("fraction_per_class")),
        entity_ids=entity_ids,
        positive_pairs=arrays["positive_pairs"],
        negative_pairs=arrays["negative_pairs"],
        herb_compound_pairs=arrays["herb_compound_pairs"],
        compound_protein_pairs=arrays["compound_protein_pairs"],
        protein_disease_pairs=arrays["protein_disease_pairs"],
        source_files={
            str(key): dict(value)
            for key, value in manifest.get("files", {}).items()
            if isinstance(value, dict)
        },
        negative_sampling=dict(manifest.get("negative_sampling", {})),
        mapping_strategy=dict(manifest.get("mapping_strategy", {})),
        pair_order=str(manifest.get("pair_order", DEFAULT_PAIR_ORDER)),
    )
    protocol.validate()
    regenerated = protocol.input_manifest()
    for section in (
        "dataset",
        "seed",
        "n_folds",
        "validation",
        "counts",
        "negative_sampling",
        "mapping_strategy",
        "embedding_dir",
        "files",
        "arrays",
        "mappings",
        "folds",
    ):
        _compare_manifest_section(section, manifest.get(section), regenerated.get(section))
    _compare_manifest_section(
        "pair_order",
        manifest.get("pair_order", DEFAULT_PAIR_ORDER),
        regenerated.get("pair_order"),
    )
    return protocol


def load_protocol(manifest_or_directory: Path | str) -> BenchmarkProtocol:
    """Compatibility alias for :func:`load_protocol_manifest`."""

    return load_protocol_manifest(manifest_or_directory)


__all__ = [
    "ARRAY_FILENAME",
    "BenchmarkProtocol",
    "DEFAULT_FOLDS",
    "DEFAULT_PAIR_ORDER",
    "DEFAULT_SEED",
    "DEFAULT_VALIDATION_FRACTION",
    "FoldRows",
    "MANIFEST_FILENAME",
    "MAPPING_FILENAME",
    "PAIR_ORDERS",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "build_protocol",
    "load_protocol",
    "load_protocol_manifest",
]
