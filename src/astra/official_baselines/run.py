#!/usr/bin/env python3
"""Launch commit-pinned official adapters on one central fair protocol.

This is deliberately a thin process orchestrator.  Every child receives the
same explicit protocol manifest and writes its own validation/test prediction
artifacts.  Model-specific defaults remain in the individual author-code
adapters, where they are recorded in ``protocol.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from astra.benchmark.protocol import load_protocol_manifest

from .common import (
    DEFAULT_CACHE_DIR,
    OFFICIAL_BASELINES_LOCK_PATH,
    diagnostic_execution_context,
    official_baseline_layout,
    resolve_official_source_root,
    sha256,
    validate_frozen_input_manifest,
    write_json,
)


_SOURCE_DIRECTORIES = official_baseline_layout(OFFICIAL_BASELINES_LOCK_PATH)[1]
RUNNERS = {
    "smgcl": (
        "astra.official_baselines.smgcl_runner",
        _SOURCE_DIRECTORIES["SMGCL"],
    ),
    "milgnet": (
        "astra.official_baselines.milgnet_runner",
        _SOURCE_DIRECTORIES["MilGNet"],
    ),
    "lhgce": (
        "astra.official_baselines.lhgce_runner",
        _SOURCE_DIRECTORIES["LHGCE"],
    ),
    "bigi": (
        "astra.official_baselines.bigi_runner",
        _SOURCE_DIRECTORIES["BiGI"],
    ),
    "hghda": (
        "astra.official_baselines.hghda_runner",
        _SOURCE_DIRECTORIES["HGHDA"],
    ),
    "htinet-kn": (
        "astra.official_baselines.htinet_runner",
        _SOURCE_DIRECTORIES["HGHDA"],
    ),
    "htinet-rf": (
        "astra.official_baselines.htinet_runner",
        _SOURCE_DIRECTORIES["HGHDA"],
    ),
}
DEFAULT_METHODS = ("smgcl", "milgnet", "lhgce")
DISPLAY_NAMES = {
    "smgcl": "SMGCL",
    "milgnet": "MilGNet",
    "lhgce": "LHGCE",
    "bigi": "BiGI",
    "hghda": "HGHDA",
    "htinet-kn": "HTInet-KN",
    "htinet-rf": "HTInet-RF",
}
RUNNER_SOURCE_NAMES = {
    "htinet-kn": "htinet_runner.py",
    "htinet-rf": "htinet_runner.py",
}


def _completed(path: Path) -> bool:
    required = (
        path / "results" / "metrics.json",
        path / "results" / "validation_predictions.npz",
        path / "results" / "test_predictions.npz",
        path / "protocol.json",
    )
    return all(item.is_file() for item in required)


def _resume_mismatches(
    path: Path,
    *,
    protocol,
    manifest: Path,
    method: str,
    fold: int,
    legacy_test_oracle: bool,
    smoke: bool,
    diagnostic: bool,
    tcm_scalable_adaptations: bool,
    smgcl_inner_validation_epoch_selection: bool,
    lhgce_inner_validation_budget_selection: bool,
    hghda_epochs: int,
) -> list[str]:
    """Fail closed when an existing task does not match this invocation."""

    mismatches: list[str] = []
    try:
        run_protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        metrics = json.loads(
            (path / "results" / "metrics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable JSON: {error}"]
    expected_mode = "legacy-test-oracle" if legacy_test_oracle else "validation"
    expected = {
        "protocol manifest": (
            str(run_protocol.get("protocol_manifest")), str(manifest)
        ),
        "input lock": (
            run_protocol.get("frozen_input_lock_sha256"),
            validate_frozen_input_manifest(protocol.input_manifest()),
        ),
        "dataset": (metrics.get("Dataset"), protocol.dataset),
        "fold": (metrics.get("Fold"), fold),
        "evaluation mode": (metrics.get("Evaluation_Mode", "validation"), expected_mode),
        "smoke flag": (bool(metrics.get("Smoke")), bool(smoke)),
        "diagnostic flag": (
            bool(metrics.get("Diagnostic", False)),
            bool(diagnostic),
        ),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            mismatches.append(f"{name}: {actual!r} != {wanted!r}")
    if run_protocol.get("diagnostic", False) is not diagnostic:
        mismatches.append(
            "protocol diagnostic flag: "
            f"{run_protocol.get('diagnostic', False)!r} != {diagnostic!r}"
        )
    if diagnostic:
        metric_reasons = metrics.get("Formal_Ineligibility_Reasons")
        protocol_reasons = run_protocol.get("formal_ineligibility_reasons")
        if metrics.get("Formal_Eligible") is not False:
            mismatches.append("diagnostic metrics are not formally ineligible")
        if run_protocol.get("formal_eligible") is not False:
            mismatches.append("diagnostic protocol is not formally ineligible")
        if not isinstance(metric_reasons, list) or not metric_reasons:
            mismatches.append("diagnostic metrics lack ineligibility reasons")
        elif protocol_reasons != metric_reasons:
            mismatches.append("diagnostic metrics/protocol reasons differ")

    adapter_sources = run_protocol.get("adapter_sources", {})
    runner_name = RUNNER_SOURCE_NAMES.get(method, f"{method}_runner.py")
    current_sources = {
        "common.py": sha256(Path(__file__).with_name("common.py")),
        runner_name: sha256(Path(__file__).with_name(runner_name)),
    }
    if method == "lhgce":
        current_sources["lhgce_identity.py"] = sha256(
            Path(__file__).with_name("lhgce_identity.py")
        )
    for name, wanted in current_sources.items():
        if adapter_sources.get(name) != wanted:
            mismatches.append(f"adapter source {name} differs")

    if tcm_scalable_adaptations:
        if method == "smgcl":
            if not bool(metrics.get("Scalable_Mode")):
                mismatches.append("SMGCL scalable mode was not enabled")
            if int(metrics.get("Epochs_Requested", -1)) != 64:
                mismatches.append("SMGCL requested epoch budget differs")
            if int(metrics.get("Batch_Size", -1)) != 4096:
                mismatches.append("SMGCL batch size differs")
            wanted_selection_mode = (
                "inner-validation"
                if smgcl_inner_validation_epoch_selection
                else "fixed-budget"
            )
            actual_selection_mode = metrics.get(
                "Epoch_Selection_Mode", "fixed-budget"
            )
            if actual_selection_mode != wanted_selection_mode:
                mismatches.append(
                    "SMGCL epoch-selection mode: "
                    f"{actual_selection_mode!r} "
                    f"!= {wanted_selection_mode!r}"
                )
            if smgcl_inner_validation_epoch_selection:
                wanted_candidates = [16, 30, 48, 64]
                if metrics.get("Epoch_Candidates") != wanted_candidates:
                    mismatches.append(
                        "SMGCL epoch candidates: "
                        f"{metrics.get('Epoch_Candidates')!r} "
                        f"!= {wanted_candidates!r}"
                    )
                if metrics.get("Selected_Epoch") not in wanted_candidates:
                    mismatches.append("SMGCL selected epoch is outside the locked grid")
        elif method == "milgnet":
            wanted = "tcm-scalable-cyclic-balanced-65536-k20-500updates"
            if metrics.get("Training_Schedule") != wanted:
                mismatches.append(
                    f"MilGNet schedule: {metrics.get('Training_Schedule')!r} != {wanted!r}"
                )
        elif method == "lhgce":
            fold_rows = protocol.fold_rows(fold)
            wanted_train = len(fold_rows.train_pairs)
            if legacy_test_oracle:
                wanted_train += len(fold_rows.validation_pairs)
            wanted_test = len(fold_rows.test_pairs)
            if int(metrics.get("Epochs_Requested", -1)) != 1000:
                mismatches.append("LHGCE requested epoch budget differs")
            if int(metrics.get("Training_Chunk_Size", -1)) != wanted_train:
                mismatches.append("LHGCE training chunk differs")
            if int(metrics.get("Evaluation_Chunk_Size", -1)) != wanted_test:
                mismatches.append("LHGCE evaluation chunk differs")
            wanted_selection_mode = (
                "inner-validation"
                if lhgce_inner_validation_budget_selection
                else "fixed-budget"
            )
            actual_selection_mode = metrics.get(
                "Epoch_Selection_Mode", "fixed-budget"
            )
            if actual_selection_mode != wanted_selection_mode:
                mismatches.append(
                    "LHGCE epoch-selection mode: "
                    f"{actual_selection_mode!r} != {wanted_selection_mode!r}"
                )
            if lhgce_inner_validation_budget_selection:
                wanted_candidates = [20, 50, 100, 200, 300, 500, 1000]
                if metrics.get("Epoch_Candidates") != wanted_candidates:
                    mismatches.append(
                        "LHGCE epoch candidates: "
                        f"{metrics.get('Epoch_Candidates')!r} "
                        f"!= {wanted_candidates!r}"
                    )
                if metrics.get("Selected_Requested_Epochs") not in wanted_candidates:
                    mismatches.append(
                        "LHGCE selected requested-epoch budget is outside the locked grid"
                    )
    if method == "hghda":
        if int(metrics.get("Epochs_Requested", -1)) != int(hghda_epochs):
            mismatches.append(
                f"HGHDA epoch budget: {metrics.get('Epochs_Requested')!r} "
                f"!= {hghda_epochs!r}"
            )
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(RUNNERS),
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(1, 6), default=range(1, 6)
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Parent of the lock-pinned author checkouts. By default this is "
            "read from third_party/baselines.lock.json. Passing the HerbHG "
            "project root is accepted as a convenience alias."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--hghda-embedding-dir",
        type=Path,
        help=(
            "Independent HGHDA compound/protein feature directory: the corrected "
            "row-ID-sidecar bundle for Ethnobotany or the exact locked author "
            "feature directory for TCM-suite. Required when --methods includes "
            "hghda."
        ),
    )
    parser.add_argument(
        "--hghda-epochs",
        type=int,
        choices=(4, 16),
        default=4,
        help="Predeclared HGHDA budget: upstream-default 4 or matched-budget 16.",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--smoke", action="store_true")
    execution_mode.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Run complete HGHDA splits/budgets while forcing diagnostic-only "
            "provenance. Other official adapters do not yet expose this mode."
        ),
    )
    parser.add_argument(
        "--legacy-test-oracle",
        action="store_true",
        help=(
            "Use source-row folds, restore the complete outer-training fold, "
            "and report threshold-dependent test-oracle diagnostics."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip only complete runs whose manifest, mode, fold, source hashes, "
            "and requested scalable schedule match this invocation."
        ),
    )
    parser.add_argument(
        "--tcm-scalable-adaptations",
        action="store_true",
        help=(
            "Lock the audited TCM scalable schedules: SMGCL sparse mode with "
            "released 64x4096 optimization; MilGNet cyclic balanced 65536-row "
            "blocks for 500 updates at k=20; and exact full-chunk LHGCE for "
            "999 updates. These are documented adaptations, not identical "
            "author-code optimizer trajectories."
        ),
    )
    parser.add_argument(
        "--smgcl-inner-validation-epoch-selection",
        action="store_true",
        help=(
            "For strict TCM-suite SMGCL, train one 64-epoch trajectory, score "
            "validation checkpoints at epochs 16/30/48/64, freeze the best "
            "trapezoidal-AUPRC checkpoint (lower epoch wins an exact tie), "
            "and evaluate the outer test exactly once."
        ),
    )
    parser.add_argument(
        "--lhgce-inner-validation-budget-selection",
        action="store_true",
        help=(
            "For strict TCM-suite LHGCE, train one requested-1000-epoch "
            "trajectory, checkpoint requested budgets 20/50/100/200/300/500/1000, "
            "select on validation trapezoidal AUPRC, freeze the decision, and "
            "evaluate only the selected checkpoint on the outer test."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if "bigi" in args.methods and args.legacy_test_oracle:
        parser.error("BiGI uses validation selection, not legacy test-oracle mode")
    htinet_methods = {"htinet-kn", "htinet-rf"}.intersection(args.methods)
    if htinet_methods and args.legacy_test_oracle:
        parser.error(
            "the controlled HTInet adaptation uses validation selection, not "
            "legacy test-oracle mode"
        )
    if "hghda" in args.methods and args.hghda_embedding_dir is None:
        parser.error("--hghda-embedding-dir is required when running HGHDA")
    if "hghda" in args.methods and args.legacy_test_oracle:
        parser.error("central Scheme-B HGHDA requires validation-based evaluation")
    if args.diagnostic and set(args.methods) != {"hghda"}:
        parser.error(
            "--diagnostic currently supports only --methods hghda; this guard "
            "prevents mixed formal/diagnostic output roots"
        )

    manifest = args.protocol_manifest.resolve()
    protocol = load_protocol_manifest(manifest)
    if "hghda" in args.methods and protocol.dataset not in {
        "ethnobotany",
        "tcm-suite",
    }:
        parser.error(
            "the central Scheme-B HGHDA adapter supports Ethnobotany and "
            "TCM-suite only"
        )
    if htinet_methods and protocol.dataset != "ethnobotany":
        parser.error(
            "the controlled HTInet HDA adaptation currently supports "
            "Ethnobotany only"
        )
    if args.legacy_test_oracle and (
        protocol.dataset != "tcm-suite" or protocol.pair_order != "source"
    ):
        parser.error(
            "--legacy-test-oracle is defined only for the source-order "
            "TCM-suite manifest"
        )
    if args.tcm_scalable_adaptations and protocol.dataset != "tcm-suite":
        parser.error(
            "--tcm-scalable-adaptations is defined only for TCM-suite; it may "
            "be used with either validation-based strict evaluation or the "
            "separately labelled legacy test-oracle mode"
        )
    if args.smgcl_inner_validation_epoch_selection:
        if protocol.dataset != "tcm-suite":
            parser.error(
                "--smgcl-inner-validation-epoch-selection is defined only for "
                "TCM-suite"
            )
        if "smgcl" not in args.methods:
            parser.error(
                "--smgcl-inner-validation-epoch-selection requires --methods "
                "to include smgcl"
            )
        if not args.tcm_scalable_adaptations:
            parser.error(
                "--smgcl-inner-validation-epoch-selection requires "
                "--tcm-scalable-adaptations"
            )
        if args.smoke or args.diagnostic or args.legacy_test_oracle:
            parser.error(
                "--smgcl-inner-validation-epoch-selection is formal validation "
                "only and cannot be combined with smoke, diagnostic, or "
                "legacy-test-oracle"
            )
    if args.lhgce_inner_validation_budget_selection:
        if protocol.dataset != "tcm-suite":
            parser.error(
                "--lhgce-inner-validation-budget-selection is defined only "
                "for TCM-suite"
            )
        if "lhgce" not in args.methods:
            parser.error(
                "--lhgce-inner-validation-budget-selection requires --methods "
                "to include lhgce"
            )
        if not args.tcm_scalable_adaptations:
            parser.error(
                "--lhgce-inner-validation-budget-selection requires "
                "--tcm-scalable-adaptations"
            )
        if args.smoke or args.diagnostic or args.legacy_test_oracle:
            parser.error(
                "--lhgce-inner-validation-budget-selection is formal validation "
                "only and cannot be combined with smoke, diagnostic, or "
                "legacy-test-oracle"
            )
    output_root = args.output_root.resolve()
    source_root = resolve_official_source_root(args.source_root)
    cache_dir = args.cache_dir.resolve()
    generation = f"{protocol.dataset}-{int(time.time())}"
    diagnostic_context = (
        diagnostic_execution_context(args.device) if args.diagnostic else None
    )
    records: list[dict[str, object]] = []

    for method in args.methods:
        module, source_name = RUNNERS[method]
        for fold in args.folds:
            run_dir = output_root / "tasks" / DISPLAY_NAMES[method] / f"fold_{fold}"
            if args.resume and _completed(run_dir):
                mismatches = _resume_mismatches(
                    run_dir,
                    protocol=protocol,
                    manifest=manifest,
                    method=method,
                    fold=fold,
                    legacy_test_oracle=args.legacy_test_oracle,
                    smoke=args.smoke,
                    diagnostic=args.diagnostic,
                    tcm_scalable_adaptations=args.tcm_scalable_adaptations,
                    smgcl_inner_validation_epoch_selection=(
                        args.smgcl_inner_validation_epoch_selection
                    ),
                    lhgce_inner_validation_budget_selection=(
                        args.lhgce_inner_validation_budget_selection
                    ),
                    hghda_epochs=args.hghda_epochs,
                )
                if mismatches:
                    raise RuntimeError(
                        f"refusing mismatched resume artifact {run_dir}: "
                        + "; ".join(mismatches)
                    )
                records.append(
                    {"method": method, "fold": fold, "status": "already-complete"}
                )
                continue
            if run_dir.exists():
                raise RuntimeError(
                    f"refusing to reuse incomplete/existing run directory: {run_dir}"
                )
            command = [
                sys.executable,
                "-m",
                module,
                "--protocol-manifest",
                str(manifest),
                "--fold",
                str(fold),
                "--output-root",
                str(run_dir),
                "--source-dir",
                str(source_root / source_name),
                "--run-generation-id",
                generation,
            ]
            if method in {"smgcl", "milgnet", "htinet-kn", "htinet-rf"}:
                command.extend(("--cache-dir", str(cache_dir)))
            if method in {"htinet-kn", "htinet-rf"}:
                command.extend(
                    (
                        "--classifier",
                        "knn" if method == "htinet-kn" else "rf",
                    )
                )
            if method == "hghda":
                command.extend(
                    (
                        "--embedding-dir",
                        str(args.hghda_embedding_dir.resolve()),
                        "--epochs",
                        str(args.hghda_epochs),
                    )
                )
            command.extend(("--device", args.device))
            if args.legacy_test_oracle:
                command.append("--legacy-test-oracle")
            if args.tcm_scalable_adaptations:
                fold_rows = protocol.fold_rows(fold)
                if method == "smgcl":
                    command.extend(
                        (
                            "--scalable-mode", "on",
                            "--epochs", "64",
                            "--batch-size", "4096",
                        )
                    )
                    if args.smgcl_inner_validation_epoch_selection:
                        command.append("--inner-validation-epoch-selection")
                elif method == "milgnet":
                    command.extend(
                        (
                            "--training-mode", "scalable-full-batch",
                            "--epochs", "500",
                            "--neighbors", "20",
                            "--decode-chunk-size", "8192",
                            "--scalable-rows-per-epoch", "65536",
                        )
                    )
                elif method == "lhgce":
                    training_rows = len(fold_rows.train_pairs)
                    if args.legacy_test_oracle:
                        training_rows += len(fold_rows.validation_pairs)
                    command.extend(
                        (
                            "--epochs", "1000",
                            "--train-chunk-size", str(training_rows),
                            "--eval-chunk-size", str(len(fold_rows.test_pairs)),
                        )
                    )
                    if args.lhgce_inner_validation_budget_selection:
                        command.append("--inner-validation-budget-selection")
            if args.smoke:
                command.append("--smoke")
                # Smoke settings exercise the real full graph and author-code
                # paths with one tiny, common pair subset.  They are labelled
                # non-formal and never enter the five-fold aggregate.
                if method == "smgcl":
                    command.extend(
                        (
                            "--batch-size", "128",
                            "--embedding-dim", "4",
                            "--neighbor-embedding-dim", "4",
                            "--hidden-dims", "8", "4",
                            "--herb-neighbors", "1",
                            "--disease-neighbors", "1",
                            "--rwr-mode", "raw",
                        )
                    )
                elif method == "milgnet":
                    command.extend(
                        (
                            "--batch-size", "128",
                            "--neighbors", "1",
                            "--hidden-features", "4",
                            "--layers", "1",
                            "--instance-topk", "1",
                            "--smoke-train-rows", "128",
                            "--smoke-test-rows", "128",
                        )
                    )
                elif method == "lhgce":
                    command.extend(("--neurons", "8"))
                elif method == "bigi":
                    command.extend(
                        (
                            "--feature-dim", "8",
                            "--hidden-dim", "8",
                            "--gnn-layers", "1",
                            "--smoke-rows-per-class", "64",
                        )
                    )
                elif method in {"htinet-kn", "htinet-rf"}:
                    command.extend(
                        (
                            "--num-walks", "1",
                            "--walk-length", "4",
                            "--embedding-epochs", "1",
                            "--smoke-rows-per-class", "64",
                        )
                    )
            elif args.diagnostic:
                command.append("--diagnostic")
            started = time.time()
            print("launching " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
            records.append(
                {
                    "method": method,
                    "fold": fold,
                    "status": "complete",
                    "runtime_seconds": time.time() - started,
                    "output": str(run_dir),
                }
            )

    summary = {
        "dataset": protocol.dataset,
        "protocol_manifest": str(manifest),
        "run_generation_id": generation,
        "smoke": bool(args.smoke),
        "diagnostic": bool(args.diagnostic),
        "execution_mode": (
            "smoke"
            if args.smoke
            else "diagnostic"
            if args.diagnostic
            else "formal-candidate"
        ),
        "diagnostic_context": diagnostic_context,
        "diagnostic_reasons": (
            list(diagnostic_context["reasons"]) if diagnostic_context else []
        ),
        "Formal_Eligible": False if args.diagnostic else None,
        "Formal_Ineligibility_Reasons": (
            list(diagnostic_context["reasons"]) if diagnostic_context else []
        ),
        "formal_result": False if args.smoke or args.diagnostic else "requires audit",
        "legacy_test_oracle": bool(args.legacy_test_oracle),
        "tcm_scalable_adaptations": bool(args.tcm_scalable_adaptations),
        "smgcl_inner_validation_epoch_selection": bool(
            args.smgcl_inner_validation_epoch_selection
        ),
        "lhgce_inner_validation_budget_selection": bool(
            args.lhgce_inner_validation_budget_selection
        ),
        "runs": records,
    }
    write_json(output_root / "official_run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
