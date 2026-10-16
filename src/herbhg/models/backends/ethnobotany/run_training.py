#!/usr/bin/env python3
"""Train HerbHG on the Ethnobotany dataset.

All default paths are derived from this file rather than the caller's current
working directory. Command-line paths remain relative to the directory from
which the command is invoked, following normal CLI conventions.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[4]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

# Direct source execution must find both the Ethnobotany compatibility
# modules and the lightweight public ``astra`` package without an install.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem locations for one Ethnobotany run."""

    config: Path
    data_dir: Path
    embedding_dir: Path
    output_dir: Path
    negative_file: Path


def _resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    return (path if path.is_absolute() else base / path).resolve()


def _first_environment(*names: str) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def _read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed config line {line_number} in {path}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _option_value(content: str | None, flag: str) -> str | None:
    if not content:
        return None
    tokens = shlex.split(content)
    try:
        index = tokens.index(flag)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    """Resolve CLI, environment, config, and default paths in that order."""

    config_value = args.config or _first_environment(
        "HERBHG_ETHNOBOTANY_CONFIG", "ASTRA_ETHNOBOTANY_CONFIG", "ASTRA_CONFIG"
    )
    config = _resolve_path(
        config_value or SCRIPT_DIR / "ethnobotany.conf",
        Path.cwd() if args.config else REPOSITORY_ROOT,
    )
    values = _read_config(config)
    config_dir = config.parent

    configured_data = values.get("data.dir")
    if not configured_data and values.get("datapath"):
        configured_data = str(Path(values["datapath"]).parent)
    data_environment = _first_environment(
        "HERBHG_ETHNOBOTANY_DATA_DIR", "ASTRA_ETHNOBOTANY_DATA_DIR", "ASTRA_DATA_DIR"
    )
    if args.data_dir:
        data_dir = _resolve_path(args.data_dir, Path.cwd())
    elif data_environment:
        data_dir = _resolve_path(data_environment, REPOSITORY_ROOT)
    else:
        data_dir = _resolve_path(
            configured_data or SCRIPT_DIR / "dataset_filtered_final", config_dir
        )

    embedding_environment = _first_environment(
        "HERBHG_ETHNOBOTANY_EMBEDDING_DIR", "ASTRA_ETHNOBOTANY_EMBEDDING_DIR", "ASTRA_EMBEDDING_DIR"
    )
    if args.embedding_dir:
        embedding_dir = _resolve_path(args.embedding_dir, Path.cwd())
    elif embedding_environment:
        embedding_dir = _resolve_path(embedding_environment, REPOSITORY_ROOT)
    else:
        embedding_dir = _resolve_path(
            values.get("embedding.dir", SCRIPT_DIR / "embeddings"), config_dir
        )

    output_environment = _first_environment(
        "HERBHG_OUTPUT_DIR", "ASTRA_OUTPUT_DIR", "CDRUG_OUTPUT_DIR"
    )
    configured_output = _option_value(values.get("output.setup"), "-dir")
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir, Path.cwd())
    elif output_environment:
        output_dir = _resolve_path(output_environment, REPOSITORY_ROOT)
    else:
        output_dir = _resolve_path(
            configured_output
            or REPOSITORY_ROOT / "outputs" / "ethnobotany" / "herbhg",
            config_dir,
        )

    negative_environment = _first_environment(
        "HERBHG_ETHNOBOTANY_NEGATIVE_FILE",
        "ASTRA_ETHNOBOTANY_NEGATIVE_FILE",
        "ASTRA_NEGATIVE_FILE",
        "CDRUG_NEGATIVE_FILE",
    )
    if args.negative_file:
        negative_file = _resolve_path(args.negative_file, Path.cwd())
    elif negative_environment:
        negative_file = _resolve_path(negative_environment, REPOSITORY_ROOT)
    elif args.data_dir or data_environment:
        negative_file = data_dir / "negativehdlist.txt"
    else:
        configured_negative = values.get("negative.file") or values.get(
            "negative.path"
        )
        negative_file = _resolve_path(
            configured_negative or data_dir / "negativehdlist.txt", config_dir
        )

    return RunPaths(
        config=config,
        data_dir=data_dir,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        negative_file=negative_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train HerbHG on the Ethnobotany herb--disease benchmark."
    )
    parser.add_argument(
        "--config",
        help="Model configuration file (default: the Ethnobotany backend config).",
    )
    parser.add_argument(
        "--data-dir",
        help="Directory containing H_D.txt, H_C.txt, C_P.txt, and P_D.txt.",
    )
    parser.add_argument(
        "--embedding-dir",
        help="Directory containing biological and semantic feature files.",
    )
    parser.add_argument(
        "--negative-file",
        help="Negative H--D pair file (default: <data-dir>/negativehdlist.txt).",
    )
    parser.add_argument(
        "--output-dir",
        help="Run output directory (default: outputs/ethnobotany/herbhg).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: HERBHG_SEED, compatibility alias, or 42).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="TensorFlow device policy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and exit without importing TensorFlow or reading data.",
    )
    parser.add_argument(
        "--allow-missing-semantic",
        action="store_true",
        help=(
            "Allow structure-only fallback when herb/disease semantic arrays "
            "are absent. By default all four feature files are required."
        ),
    )
    return parser


def _print_paths(paths: RunPaths) -> None:
    print("Resolved Ethnobotany run paths:")
    print(f"  config:       {paths.config}")
    print(f"  data:         {paths.data_dir}")
    print(f"  embeddings:   {paths.embedding_dir}")
    print(f"  negatives:    {paths.negative_file}")
    print(f"  output:       {paths.output_dir}")


def _require_inputs(paths: RunPaths, *, allow_missing_semantic: bool = False) -> None:
    required = [
        paths.config,
        paths.data_dir / "H_D.txt",
        paths.data_dir / "H_C.txt",
        paths.data_dir / "C_P.txt",
        paths.data_dir / "P_D.txt",
        paths.negative_file,
        paths.embedding_dir / "compound_embedding_kpgt.txt",
        paths.embedding_dir / "protein_embedding_esm2.txt",
    ]
    if not allow_missing_semantic:
        required.extend(
            [
                paths.embedding_dir / "herb_semantic_embeddings_64d.npy",
                paths.embedding_dir / "disease_semantic_embeddings_64d.npy",
            ]
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required Ethnobotany inputs are missing:\n"
            f"{formatted}\n"
            "See docs/data.md for local data placement instructions."
        )


def _configure_runtime(seed: int, device: str, output_dir: Path) -> None:
    os.environ["HERBHG_SEED"] = str(seed)
    os.environ["ASTRA_SEED"] = str(seed)
    # Retained for compatibility with the historical experiment scripts.
    os.environ["CDRUG_SEED"] = str(seed)
    os.environ["HERBHG_OUTPUT_DIR"] = str(output_dir)
    os.environ["ASTRA_OUTPUT_DIR"] = str(output_dir)
    os.environ["CDRUG_OUTPUT_DIR"] = str(output_dir)
    os.environ["ASTRA_LOG_DIR"] = str(output_dir / "logs")
    os.environ["CDRUG_LOG_DIR"] = str(output_dir / "logs")
    os.environ["HERBHG_LOG_DIR"] = str(output_dir / "logs")
    os.environ["HERBHG_DEVICE"] = device
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ.pop("CUDA_VISIBLE_DEVICES")

    random.seed(seed)
    # NumPy is part of the training environment but is deliberately imported
    # here so that --help and --dry-run stay dependency-light.
    import numpy as np

    np.random.seed(seed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = int(
            _first_environment("HERBHG_SEED", "ASTRA_SEED", "CDRUG_SEED") or "42"
        )
    if args.device is None:
        args.device = _first_environment("HERBHG_DEVICE", "ASTRA_DEVICE") or "auto"
    if args.device not in {"auto", "cpu", "cuda"}:
        parser.error("ASTRA_DEVICE must be one of: auto, cpu, cuda")
    paths = resolve_run_paths(args)

    if args.dry_run:
        _print_paths(paths)
        return 0

    try:
        _require_inputs(
            paths,
            allow_missing_semantic=args.allow_missing_semantic,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "cv").mkdir(parents=True, exist_ok=True)
    if args.allow_missing_semantic:
        os.environ["ASTRA_ALLOW_MISSING_SEMANTIC"] = "1"
    else:
        os.environ.pop("ASTRA_ALLOW_MISSING_SEMANTIC", None)
    _configure_runtime(args.seed, args.device, paths.output_dir)

    # Keep legacy modules importable while avoiding any TensorFlow import when
    # this module is inspected, used by a CLI dispatcher, or run with --help.
    from HDR import HDR
    from util.config import ModelConf

    conf = ModelConf(str(paths.config))
    conf.config.update(
        {
            "model.name": "HerbHG",
            "datapath": str(paths.data_dir / "H_D.txt"),
            "data.dir": str(paths.data_dir),
            "embedding.dir": str(paths.embedding_dir),
            "negative.file": str(paths.negative_file),
            "output.setup": f'on -dir "{paths.output_dir}"',
        }
    )

    print("=" * 80)
    print("HerbHG training on Ethnobotany")
    print("=" * 80)
    _print_paths(paths)
    print(f"  seed:         {args.seed}")
    print(f"  device:       {args.device}")

    try:
        print("\nInitializing HerbHG...")
        model = HDR(conf)
        print("\nStarting training...")
        model.execute()
    except Exception as exc:
        print(f"\nHerbHG training failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print("\nHerbHG training completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
