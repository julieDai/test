#!/usr/bin/env python3
"""Launch ASTRA and commit-pinned baselines under one frozen protocol.

Every child writes to ``tasks/<Method>/fold_<n>`` so the independent audit can
compare the exact same row IDs, pairs, labels, and scores.  Smoke mode is a
development check only and is labelled as such by every child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from astra.benchmark.protocol import load_protocol_manifest
from astra.official_baselines.common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_THIRD_PARTY_DIR,
    diagnostic_execution_context,
    write_json,
)


METHODS = ("astra", "smgcl", "milgnet", "lhgce", "bigi", "hghda")
DEFAULT_METHODS = ("astra", "smgcl", "milgnet", "lhgce")


def _parse_unit_float(value: str | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("alpha must be a number in [0, 1]") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be finite and in [0, 1]")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_alphas(path: Path, folds: Sequence[int]) -> dict[int, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read alpha-selection manifest: {error}") from error
    if payload.get("schema") != "astra-ethnobotany-alpha-selection/v1":
        raise RuntimeError("unsupported alpha-selection manifest schema")
    records = payload.get("folds")
    if not isinstance(records, dict):
        raise RuntimeError("alpha-selection manifest has no fold decisions")
    selected: dict[int, float] = {}
    for fold in folds:
        record = records.get(str(fold))
        if not isinstance(record, dict):
            raise RuntimeError(f"alpha-selection manifest has no fold {fold}")
        selected[fold] = _parse_unit_float(record.get("selected_alpha"))
    return selected


def _complete(run_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "protocol.json",
            run_dir / "results" / "metrics.json",
            run_dir / "results" / "validation_predictions.npz",
            run_dir / "results" / "test_predictions.npz",
        )
    )


def _astra_resume_mode_mismatches(
    run_dir: Path,
    *,
    smoke: bool,
    diagnostic: bool,
    text_fusion_alpha: float = 1.0,
    alpha_selection_sha256: str | None = None,
) -> list[str]:
    """Prevent resume from crossing formal/smoke/diagnostic boundaries."""

    try:
        metrics = json.loads(
            (run_dir / "results" / "metrics.json").read_text(encoding="utf-8")
        )
        protocol = json.loads(
            (run_dir / "protocol.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable mode provenance: {error}"]
    mismatches = []
    expected = {
        "metrics Smoke": (metrics.get("Smoke"), smoke),
        "metrics Diagnostic": (metrics.get("Diagnostic", False), diagnostic),
        "protocol smoke": (protocol.get("smoke"), smoke),
        "protocol diagnostic": (protocol.get("diagnostic", False), diagnostic),
        "metrics Text_Fusion_Alpha": (
            metrics.get("Text_Fusion_Alpha", 1.0),
            text_fusion_alpha,
        ),
        "protocol text_fusion_alpha": (
            protocol.get("hyperparameters", {}).get("text_fusion_alpha", 1.0),
            text_fusion_alpha,
        ),
    }
    for name, (actual, wanted) in expected.items():
        if isinstance(wanted, bool):
            differs = type(actual) is not bool or actual is not wanted
        else:
            try:
                differs = not math.isclose(
                    float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                differs = True
        if differs:
            mismatches.append(f"{name}: {actual!r} != {wanted!r}")
    if diagnostic:
        metric_reasons = metrics.get("Formal_Ineligibility_Reasons")
        protocol_reasons = protocol.get("formal_ineligibility_reasons")
        if metrics.get("Formal_Eligible") is not False:
            mismatches.append("diagnostic metrics are not formally ineligible")
        if protocol.get("formal_eligible") is not False:
            mismatches.append("diagnostic protocol is not formally ineligible")
        if not isinstance(metric_reasons, list) or not metric_reasons:
            mismatches.append("diagnostic metrics lack ineligibility reasons")
        elif protocol_reasons != metric_reasons:
            mismatches.append("diagnostic metrics/protocol reasons differ")
    selection = protocol.get("alpha_selection")
    actual_selection_sha256 = (
        selection.get("sha256") if isinstance(selection, dict) else None
    )
    if actual_selection_sha256 != alpha_selection_sha256:
        mismatches.append(
            "protocol alpha-selection SHA256: "
            f"{actual_selection_sha256!r} != {alpha_selection_sha256!r}"
        )
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ethnobotany", "tcm-suite"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument(
        "--hghda-embedding-dir",
        type=Path,
        help=(
            "Independent HGHDA compound/protein feature directory. This is "
            "required when --methods includes hghda and is intentionally not "
            "inferred from HerbHG's --embedding-dir."
        ),
    )
    parser.add_argument("--herb-semantic", type=Path)
    parser.add_argument("--disease-semantic", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_THIRD_PARTY_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-fusion-alpha",
        type=_parse_unit_float,
        default=1.0,
        help="Fixed ASTRA structure/text-enhanced branch interpolation.",
    )
    parser.add_argument(
        "--alpha-selection-manifest",
        type=Path,
        help="Per-fold validation-only alpha decisions for selected ASTRA runs.",
    )
    parser.add_argument(
        "--hghda-epochs",
        type=int,
        choices=(4, 16),
        default=4,
        help="Predeclared HGHDA budget: upstream-default 4 or matched-budget 16.",
    )
    parser.add_argument(
        "--tcm-scalable-adaptations",
        action="store_true",
        help=(
            "Forward the audited scalable schedules for the TCM SMGCL, "
            "MilGNet, and LHGCE adapters. The flag is compatible with the "
            "validation-based strict protocol; unrelated methods are unchanged."
        ),
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--smoke", action="store_true")
    execution_mode.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "Run full frozen splits and normal epoch budgets while marking all "
            "ASTRA/HGHDA artifacts diagnostic-only."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.protocol_manifest.resolve()
    protocol = load_protocol_manifest(manifest)
    if protocol.dataset != args.dataset:
        raise RuntimeError(
            f"manifest dataset {protocol.dataset!r} differs from --dataset {args.dataset!r}"
        )
    if protocol.seed != args.seed:
        raise RuntimeError(
            f"manifest seed {protocol.seed} differs from --seed {args.seed}"
        )
    if args.alpha_selection_manifest is not None:
        if "astra" not in args.methods:
            raise RuntimeError(
                "--alpha-selection-manifest requires --methods to include astra"
            )
        if protocol.dataset != "ethnobotany":
            raise RuntimeError(
                "--alpha-selection-manifest is currently restricted to Ethnobotany"
            )
    if (args.herb_semantic is None) != (args.disease_semantic is None):
        raise RuntimeError("--herb-semantic and --disease-semantic must be supplied together")
    if "hghda" in args.methods and args.hghda_embedding_dir is None:
        raise RuntimeError(
            "--hghda-embedding-dir is required when --methods includes hghda; "
            "HerbHG's --embedding-dir is never reused implicitly"
        )
    if args.tcm_scalable_adaptations and protocol.dataset != "tcm-suite":
        raise RuntimeError(
            "--tcm-scalable-adaptations is defined only for TCM-suite"
        )
    unsupported_diagnostics = set(args.methods) - {"astra", "hghda"}
    if args.diagnostic and unsupported_diagnostics:
        raise RuntimeError(
            "--diagnostic currently supports only ASTRA and HGHDA; unsupported: "
            + ", ".join(sorted(unsupported_diagnostics))
        )
    diagnostic_context = (
        diagnostic_execution_context(args.device) if args.diagnostic else None
    )
    alpha_selection_path = (
        args.alpha_selection_manifest.resolve()
        if args.alpha_selection_manifest is not None
        else None
    )
    selected_alpha_by_fold = (
        _selected_alphas(alpha_selection_path, args.folds)
        if alpha_selection_path is not None
        else {fold: float(args.text_fusion_alpha) for fold in args.folds}
    )
    alpha_selection_sha256 = (
        _sha256(alpha_selection_path) if alpha_selection_path is not None else None
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    if "astra" in args.methods:
        for fold in args.folds:
            fold_alpha = selected_alpha_by_fold[fold]
            run_dir = output_root / "tasks" / "ASTRA" / f"fold_{fold}"
            if _complete(run_dir):
                if args.resume:
                    mismatches = _astra_resume_mode_mismatches(
                        run_dir,
                        smoke=bool(args.smoke),
                        diagnostic=bool(args.diagnostic),
                        text_fusion_alpha=fold_alpha,
                        alpha_selection_sha256=alpha_selection_sha256,
                    )
                    if mismatches:
                        raise RuntimeError(
                            f"refusing mismatched ASTRA resume artifact {run_dir}: "
                            + "; ".join(mismatches)
                        )
                    records.append({"method": "ASTRA", "fold": fold, "status": "already-complete"})
                    continue
                raise RuntimeError(f"refusing to overwrite completed ASTRA task: {run_dir}")
            if run_dir.exists() and not args.resume:
                raise RuntimeError(f"refusing to reuse existing ASTRA task: {run_dir}")
            command = [
                sys.executable, "-m", "astra.benchmark.astra_runner",
                "--protocol-manifest", str(manifest),
                "--dataset", args.dataset,
                "--data-dir", str(args.data_dir.resolve()),
                "--embedding-dir", str(args.embedding_dir.resolve()),
                "--output-root", str(run_dir),
                "--fold", str(fold),
                "--seed", str(args.seed),
                "--device", args.device,
                "--text-fusion-alpha", format(fold_alpha, ".17g"),
            ]
            if alpha_selection_path is not None:
                command.extend(
                    (
                        "--alpha-selection-manifest",
                        str(alpha_selection_path),
                    )
                )
            if args.herb_semantic is not None:
                command.extend(("--herb-semantic", str(args.herb_semantic.resolve())))
                command.extend(("--disease-semantic", str(args.disease_semantic.resolve())))
            if args.smoke:
                command.append("--smoke")
            elif args.diagnostic:
                command.append("--diagnostic")
            started = time.time()
            subprocess.run(command, check=True)
            records.append(
                {
                    "method": "ASTRA",
                    "fold": fold,
                    "status": "complete",
                    "runtime_seconds": time.time() - started,
                    "output": str(run_dir),
                }
            )

    baselines = [method for method in args.methods if method != "astra"]
    if baselines:
        command = [
            sys.executable, "-m", "astra.official_baselines.run",
            "--protocol-manifest", str(manifest),
            "--output-root", str(output_root),
            "--methods", *baselines,
            "--folds", *(str(fold) for fold in args.folds),
            "--source-root", str(args.source_root.resolve()),
            "--cache-dir", str(args.cache_dir.resolve()),
            "--device", args.device,
        ]
        if args.smoke:
            command.append("--smoke")
        elif args.diagnostic:
            command.append("--diagnostic")
        if args.resume:
            command.append("--resume")
        if args.tcm_scalable_adaptations:
            command.append("--tcm-scalable-adaptations")
        if "hghda" in baselines:
            command.extend(
                (
                    "--hghda-embedding-dir",
                    str(args.hghda_embedding_dir.resolve()),
                    "--hghda-epochs",
                    str(args.hghda_epochs),
                )
            )
        subprocess.run(command, check=True)
        official_summary = output_root / "official_run_summary.json"
        if official_summary.is_file():
            records.extend(json.loads(official_summary.read_text(encoding="utf-8"))["runs"])

    payload = {
        "schema": "astra-fair-comparison-run/v1",
        "dataset": protocol.dataset,
        "seed": protocol.seed,
        "protocol_manifest": str(manifest),
        "methods": list(args.methods),
        "folds": list(args.folds),
        "text_fusion_alpha": (
            None if alpha_selection_path is not None else float(args.text_fusion_alpha)
        ),
        "selected_alpha_by_fold": {
            str(fold): alpha for fold, alpha in selected_alpha_by_fold.items()
        },
        "alpha_selection_manifest": (
            str(alpha_selection_path) if alpha_selection_path is not None else None
        ),
        "alpha_selection_manifest_sha256": alpha_selection_sha256,
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
        "hghda_epochs": int(args.hghda_epochs) if "hghda" in args.methods else None,
        "hghda_embedding_dir": (
            str(args.hghda_embedding_dir.resolve())
            if "hghda" in args.methods
            else None
        ),
        "tcm_scalable_adaptations": bool(args.tcm_scalable_adaptations),
        "formal_result": (
            False
            if args.smoke or args.diagnostic
            else "requires independent audit"
        ),
        "runs": records,
    }
    write_json(output_root / "fair_run_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
