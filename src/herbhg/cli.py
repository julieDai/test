"""Unified command-line interface for training and reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Sequence

from . import __version__
from .paths import ProjectPaths
from .process import run_python


DATASET_ALIASES = {
    "tcm": "tcm-suite",
    "tcm-suite": "tcm-suite",
    "tcmsuite": "tcm-suite",
    "ethno": "ethnobotany",
    "ethnobotany": "ethnobotany",
}

BENCHMARK_COMMANDS = {
    "prepare": "astra.benchmark.prepare",
    "run": "astra.benchmark.astra_runner",
    "compare": "astra.benchmark.fair_comparison",
    "harmonize": "astra.benchmark.harmonize_metrics",
}

BASELINE_COMMANDS = {
    "run": "astra.official_baselines.run",
    "audit": "astra.official_baselines.audit_results",
    "preflight": "astra.official_baselines.preflight",
}

EXPERIMENT_COMMANDS = {
    "ethnobotany-alpha": "astra.benchmark.ethnobotany_alpha_selection",
    "ethnobotany-refined": "astra.benchmark.ethnobotany_refined_runner",
    "ethnobotany-semantic-adapter": (
        "astra.benchmark.ethnobotany_semantic_adapter_runner"
    ),
    "tcm-epoch-boundary": "astra.benchmark.tcm_astra_epoch_boundary_runner",
    "tcm-hyperparameters": "astra.benchmark.tcm_astra_hparam_runner",
    "tcm-pca64-control": "astra.benchmark.tcm_astra_pca64_control_runner",
    "tcm-semantic-adapter": (
        "astra.benchmark.tcm_astra_semantic_adapter_deterministic_runner"
    ),
}

ISOLATED_COMMANDS = {
    "benchmark": frozenset({"compare"}),
    "baseline": frozenset({"run", "preflight"}),
}


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="herbhg",
        description="Train HerbHG and run protocol-controlled HDA experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("train", "smoke", "benchmark", "baseline", "experiment"),
        help="Command group to execute.",
    )
    return parser


def _print_group_help(group: str, commands: dict[str, str]) -> None:
    print(f"usage: herbhg {group} <command> [options]\n")
    print("commands:")
    for name in commands:
        print(f"  {name}")
    print(f"\nRun 'herbhg {group} <command> --help' for command options.")


def _dispatch_module(
    group: str,
    arguments: list[str],
    modules: dict[str, str],
    paths: ProjectPaths,
) -> int:
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_group_help(group, modules)
        return 0
    name, *forwarded = arguments
    module = modules.get(name)
    if module is None:
        print(f"Unknown {group} command: {name}", file=sys.stderr)
        _print_group_help(group, modules)
        return 2
    return run_python(
        ["-m", module, *forwarded],
        paths=paths,
        isolated_python=name in ISOLATED_COMMANDS.get(group, ()),
    )


def _train(arguments: list[str], paths: ProjectPaths) -> int:
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: herbhg train {tcm-suite|ethnobotany} [training options]")
        print("\nDataset-specific options are forwarded to the selected backend.")
        return 0

    requested, *forwarded = arguments
    dataset = DATASET_ALIASES.get(requested.lower().replace("_", "-"))
    if dataset is None:
        print(f"Unknown dataset: {requested}", file=sys.stderr)
        return 2
    backend = paths.backend(dataset)
    entry = backend / ("main.py" if dataset == "tcm-suite" else "run_training.py")
    if not entry.is_file():
        print(f"Training entry point is missing: {entry}", file=sys.stderr)
        return 2
    return run_python(
        [entry, *forwarded],
        paths=paths,
        backend=backend,
        overrides={"HERBHG_DATASET": dataset},
    )


def _smoke(arguments: list[str], _paths: ProjectPaths | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="herbhg smoke",
        description="Validate package paths and a synthetic two-fold relation set.",
    )
    parser.parse_args(arguments)
    from .smoke import main as smoke_main

    return smoke_main()


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a HerbHG command without importing a numerical backend."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _root_parser()
    if not arguments or arguments[0] in {"-h", "--help", "--version"}:
        if arguments and arguments[0] == "--version":
            print(f"herbhg {__version__}")
            return 0
        parser.print_help()
        return 0

    command, *forwarded = arguments
    if command not in {"train", "smoke", "benchmark", "baseline", "experiment"}:
        parser.error(f"argument command: invalid choice: {command!r}")
    if command == "smoke":
        return _smoke(forwarded)
    paths = ProjectPaths.discover()
    handlers: dict[str, Callable[[list[str], ProjectPaths], int]] = {
        "train": _train,
        "benchmark": lambda values, found: _dispatch_module(
            "benchmark", values, BENCHMARK_COMMANDS, found
        ),
        "baseline": lambda values, found: _dispatch_module(
            "baseline", values, BASELINE_COMMANDS, found
        ),
        "experiment": lambda values, found: _dispatch_module(
            "experiment", values, EXPERIMENT_COMMANDS, found
        ),
    }
    return handlers[command](forwarded, paths)


if __name__ == "__main__":
    raise SystemExit(main())
