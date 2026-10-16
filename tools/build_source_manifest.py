#!/usr/bin/env python3
"""Build or verify a deterministic SHA-256 manifest of repository sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence


DEFAULT_EXCLUDED_ROOT_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "artifacts",
        "build",
        "data",
        "dist",
        "local_archive",
        "outputs",
    }
)
DEFAULT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "checkouts",
    }
)
DEFAULT_EXCLUDED_FILES = frozenset(
    {
        ".coverage",
        ".DS_Store",
        ".env",
        "provenance/source_manifest.json",
    }
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root: Path, extra_exclusions: Iterable[str] = ()) -> list[Path]:
    """Return deterministic repository-relative paths included in the manifest."""

    exclusions = DEFAULT_EXCLUDED_FILES | frozenset(extra_exclusions)
    selected: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_directory = current_path.relative_to(root)
        at_root = relative_directory == Path(".")
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in DEFAULT_EXCLUDED_DIRECTORY_NAMES
            and not name.endswith((".egg-info", ".dist-info"))
            and (not at_root or name not in DEFAULT_EXCLUDED_ROOT_DIRECTORIES)
        )
        if relative_directory == Path("results"):
            directory_names[:] = [
                name for name in directory_names if name == "reference"
            ]
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root)
            relative_name = relative.as_posix()
            if (
                relative_name in exclusions
                or name in DEFAULT_EXCLUDED_FILES
                or name.startswith(".env.")
            ):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            selected.append(relative)
    return sorted(selected, key=lambda item: item.as_posix())


def build_manifest(root: Path, extra_exclusions: Iterable[str] = ()) -> dict[str, object]:
    """Create the canonical manifest payload for ``root``."""

    records = []
    for relative in source_files(root, extra_exclusions):
        path = root / relative
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "root": ".",
        "file_count": len(records),
        "files": records,
    }


def encode_manifest(payload: dict[str, object]) -> str:
    """Serialize a manifest in its canonical representation."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _by_path(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    records = payload.get("files")
    if not isinstance(records, list):
        raise ValueError("manifest field 'files' must be a list")
    mapped: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("each manifest record must contain a string path")
        if record["path"] in mapped:
            raise ValueError(f"duplicate manifest path: {record['path']}")
        mapped[record["path"]] = record
    return mapped


def check_manifest(expected: dict[str, object], observed: dict[str, object]) -> list[str]:
    """Describe differences between two manifest payloads."""

    differences = []
    for field in ("schema_version", "algorithm", "root", "file_count"):
        if expected.get(field) != observed.get(field):
            differences.append(
                f"metadata {field}: {expected.get(field)!r} != {observed.get(field)!r}"
            )
    before = _by_path(expected)
    after = _by_path(observed)
    differences.extend(f"added: {name}" for name in sorted(after.keys() - before.keys()))
    differences.extend(f"removed: {name}" for name in sorted(before.keys() - after.keys()))
    for name in sorted(before.keys() & after.keys()):
        if before[name] != after[name]:
            differences.append(f"changed: {name}")
    return differences


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("provenance/source_manifest.json"),
        help="Manifest path, interpreted relative to --root when not absolute.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing manifest instead of replacing it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"repository root does not exist: {root}")

    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    try:
        output_relative = output.relative_to(root).as_posix()
    except ValueError:
        output_relative = ""
    extra_exclusions = (output_relative,) if output_relative else ()
    observed = build_manifest(root, extra_exclusions)

    if args.check:
        if not output.is_file():
            print(f"manifest does not exist: {output}", file=sys.stderr)
            return 1
        expected = json.loads(output.read_text(encoding="utf-8"))
        differences = check_manifest(expected, observed)
        if differences:
            print("source manifest differs:", file=sys.stderr)
            for difference in differences:
                print(f"  {difference}", file=sys.stderr)
            return 1
        print(f"verified {observed['file_count']} source files")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(encode_manifest(observed), encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output} with {observed['file_count']} source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
