"""Lightweight relation readers and split checks used before model startup."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping
from pathlib import Path
import re
from typing import Iterable, Sequence, TypeVar

from .paths import PathLike, resolve_project_path


Pair = tuple[str, str]
Group = TypeVar("Group", bound=Hashable)


def read_pairs(path: PathLike) -> list[Pair]:
    """Read the first two columns of a whitespace/comma separated relation."""

    relation_path = resolve_project_path(path, must_exist=True)
    pairs: list[Pair] = []
    for line_number, raw_line in enumerate(
        relation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [item for item in re.split(r"[\s,]+", line) if item]
        if len(fields) < 2:
            raise ValueError(
                f"Expected at least two columns at {relation_path}:{line_number}"
            )
        pairs.append((fields[0], fields[1]))
    return pairs


def duplicate_pairs(pairs: Iterable[Pair]) -> set[Pair]:
    return {pair for pair, count in Counter(pairs).items() if count > 1}


def overlapping_pairs(left: Iterable[Pair], right: Iterable[Pair]) -> set[Pair]:
    return set(left).intersection(right)


def modulo_folds(records: Sequence[object], folds: int = 5) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    return [
        [index for index in range(len(records)) if index % folds == fold]
        for fold in range(folds)
    ]


def validate_binary_pairs(positives: Sequence[Pair], negatives: Sequence[Pair]) -> None:
    positive_duplicates = duplicate_pairs(positives)
    negative_duplicates = duplicate_pairs(negatives)
    overlap = overlapping_pairs(positives, negatives)
    problems = []
    if positive_duplicates:
        problems.append(f"{len(positive_duplicates)} duplicated positive pairs")
    if negative_duplicates:
        problems.append(f"{len(negative_duplicates)} duplicated negative pairs")
    if overlap:
        problems.append(f"{len(overlap)} pairs occur in both classes")
    if problems:
        raise ValueError("; ".join(problems))


def group_indexed_relation(
    rows: Iterable[Sequence[Group]],
    *,
    group_column: int,
    member_column: int,
    member_to_index: Mapping[Group, int],
    relation_name: str,
) -> dict[Group, list[int]]:
    """Group a two-or-more-column relation and map member identifiers.

    Historical relation files may contain either two endpoint columns or an
    additional weight column. Only the requested endpoint columns are
    structural; trailing metadata is ignored.
    """

    required_width = max(group_column, member_column) + 1
    grouped: dict[Group, list[int]] = {}
    for line_number, row in enumerate(rows, start=1):
        if len(row) < required_width:
            raise ValueError(
                f"{relation_name} row {line_number} has {len(row)} columns; "
                f"at least {required_width} are required"
            )
        group = row[group_column]
        member = row[member_column]
        try:
            member_index = member_to_index[member]
        except KeyError as exc:
            raise ValueError(
                f"{relation_name} row {line_number} references unknown member "
                f"{member!r}"
            ) from exc
        grouped.setdefault(group, []).append(member_index)
    return grouped
