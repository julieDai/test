#!/usr/bin/env python3
"""Legacy read-only selector for already materialized LHGCE diagnostics.

This module is retained only to audit the abandoned multi-run grid.  It is not
eligible for the strict primary comparison because those candidate runs had
already materialized every outer-test prediction.  New formal runs must use
``lhgce_runner --inner-validation-budget-selection``, which trains one
trajectory and does not construct the test index before freezing the winner.

The historical selector implements selection for the fixed candidate set
``{20, 50, 100, 200, 300, 500, 1000}``.  Selection is deliberately split into
ordered phases:

1. require every candidate artifact for every requested fold to exist;
2. load and validate *validation predictions only*;
3. freeze one budget per outer fold by validation trapezoidal AUPRC;
4. audit the run metadata for every candidate; and
5. load exactly one test-prediction artifact per fold, from the frozen winner.

The ordering prevents outer-test scores from influencing the epoch budget.
The official LHGCE runner is not modified and candidate artifacts remain
read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from astra.benchmark.metrics import (
    AUPRC_DEFINITION,
    AVERAGE_PRECISION_DEFINITION,
    BenchmarkAuditError,
    PredictionTable,
    aggregate_fold_metrics,
    evaluate_test_scores,
    evaluate_validation_and_test,
    load_predictions,
    validate_prediction_coverage,
)
from astra.benchmark.protocol import BenchmarkProtocol, load_protocol_manifest

from .common import fold_split, sha256, validate_frozen_input_manifest, write_json


CANDIDATE_BUDGETS = (20, 50, 100, 200, 300, 500, 1000)
UPSTREAM_COMMIT = "eeedb4cd6cf64166bd0f8a0ebc2cca96c60874dc"


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BenchmarkAuditError(f"{label} must be a JSON integer >= {minimum}")
    return value


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkAuditError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkAuditError(f"JSON artifact is not an object: {path}")
    return payload


def _candidate_fold_dir(method_root: Path, fold: int) -> Path:
    return method_root / f"fold_{fold}"


def _require_complete_candidates(
    candidate_roots: Mapping[int, Path], folds: Sequence[int]
) -> dict[tuple[int, int], dict[str, Path]]:
    """Check every file before loading any prediction payload."""

    artifacts: dict[tuple[int, int], dict[str, Path]] = {}
    missing: list[str] = []
    for budget in sorted(candidate_roots):
        method_root = candidate_roots[budget].resolve()
        for fold in folds:
            fold_dir = _candidate_fold_dir(method_root, fold)
            paths = {
                "protocol": fold_dir / "protocol.json",
                "metrics": fold_dir / "results" / "metrics.json",
                "validation": fold_dir / "results" / "validation_predictions.npz",
                "test": fold_dir / "results" / "test_predictions.npz",
            }
            artifacts[(fold, budget)] = paths
            missing.extend(str(path) for path in paths.values() if not path.is_file())
    if missing:
        raise BenchmarkAuditError(
            "LHGCE budget selection requires a complete candidate grid; missing:\n"
            + "\n".join(missing)
        )
    return artifacts


def _expected_split(protocol: BenchmarkProtocol, fold: int, split: str):
    pairs, labels, row_id = fold_split(protocol.fold_rows(fold), split)
    return {
        "expected_row_id": row_id,
        "expected_pair": pairs,
        "expected_label": labels,
    }


def _validation_ranking(table: PredictionTable) -> dict[str, float]:
    # The threshold argument is irrelevant to the returned continuous ranking
    # metrics.  Calling the central evaluator keeps the manuscript definitions
    # (notably trapezoidal PR-AUC) in one place.
    metrics = evaluate_test_scores(table.label, table.score, threshold=0.5)
    return {
        "AUROC": float(metrics["AUROC"]),
        "AUPRC": float(metrics["AUPRC"]),
        "AveragePrecision": float(metrics["AveragePrecision"]),
    }


def _select_budget(records: Mapping[int, Mapping[str, float]]) -> int:
    """Maximize validation trapezoidal AUPRC; exact ties use fewer epochs."""

    if not records:
        raise BenchmarkAuditError("no validation candidates were supplied")
    return min(records, key=lambda budget: (-float(records[budget]["AUPRC"]), budget))


def _assert_candidate_metadata(
    *,
    protocol: BenchmarkProtocol,
    protocol_manifest: Path,
    fold: int,
    budget: int,
    paths: Mapping[str, Path],
) -> dict[str, object]:
    """Audit candidate identity after, and never during, budget selection."""

    metrics = _load_json(paths["metrics"])
    run_protocol = _load_json(paths["protocol"])
    fold_rows = protocol.fold_rows(fold)
    train_pairs, _, _ = fold_split(fold_rows, "train")
    test_pairs, _, _ = fold_split(fold_rows, "test")
    executed = budget - 1

    if metrics.get("Dataset") != "tcm-suite" or run_protocol.get("dataset") != "tcm-suite":
        raise BenchmarkAuditError(f"LHGCE epoch {budget} fold {fold} dataset differs")
    if _strict_int(metrics.get("Fold"), "metrics Fold", minimum=1) != fold:
        raise BenchmarkAuditError(f"LHGCE epoch {budget} fold identity differs")
    if _strict_int(metrics.get("Seed"), "metrics Seed") != protocol.seed:
        raise BenchmarkAuditError(f"LHGCE epoch {budget} seed differs")
    expected_fields = {
        "Epochs_Requested": budget,
        "Epochs_Executed": executed,
        "Optimizer_Steps": executed,
        "Training_Rows": len(train_pairs),
        "Training_Chunk_Size": len(train_pairs),
        "Evaluation_Chunk_Size": len(test_pairs),
    }
    for name, expected in expected_fields.items():
        actual = _strict_int(metrics.get(name), f"metrics {name}", minimum=1)
        if actual != expected:
            raise BenchmarkAuditError(
                f"LHGCE epoch {budget} fold {fold} {name}={actual}, expected {expected}"
            )
    if metrics.get("Optimizer_Steps") != run_protocol.get("optimizer_steps"):
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} optimizer provenance differs"
        )
    if metrics.get("Evaluation_Mode") != "validation" or run_protocol.get(
        "evaluation_mode"
    ) != "validation":
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} is not validation based"
        )
    if run_protocol.get("training_split") != "inner training fold":
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} did not use inner training only"
        )
    if metrics.get("Smoke") is not False or metrics.get("Formal_Eligible") is not True:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} is not a formal candidate"
        )
    if metrics.get("Chunked_Loss_Exact") is not True:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} lacks exact chunked loss"
        )
    if metrics.get("CP_Included") is not False:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} changes the locked CP configuration"
        )
    if metrics.get("Upstream_Commit") != UPSTREAM_COMMIT:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} upstream commit differs"
        )
    if run_protocol.get("model") != metrics:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} protocol/metrics payloads differ"
        )
    recorded_manifest = run_protocol.get("protocol_manifest")
    if not isinstance(recorded_manifest, str) or Path(recorded_manifest).resolve() != (
        protocol_manifest.resolve()
    ):
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} protocol manifest path differs"
        )

    arguments = run_protocol.get("arguments")
    if not isinstance(arguments, dict):
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} protocol arguments are missing"
        )
    if _strict_int(arguments.get("epochs"), "protocol arguments epochs", minimum=2) != budget:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} protocol epoch budget differs"
        )
    if arguments.get("legacy_test_oracle") is not False or arguments.get(
        "include_mapped_cp"
    ) is not False:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} protocol configuration differs"
        )
    locked_arguments = {
        "fold": fold,
        "seed": protocol.seed,
        "learning_rate": 1e-3,
        "layers": 1,
        "neurons": 168,
        "agg_linear": "mul",
        "agg_conv": "var",
        "agg_hetero": "sum",
        "train_chunk_size": len(train_pairs),
        "eval_chunk_size": len(test_pairs),
        "smoke": False,
    }
    for name, expected in locked_arguments.items():
        if arguments.get(name) != expected:
            raise BenchmarkAuditError(
                f"LHGCE epoch {budget} fold {fold} argument {name}="
                f"{arguments.get(name)!r}, expected {expected!r}"
            )

    expected_manifest = protocol.input_manifest()
    observed_manifest = {key: run_protocol.get(key) for key in expected_manifest}
    if observed_manifest != expected_manifest:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} frozen input manifest differs"
        )
    expected_lock = validate_frozen_input_manifest(expected_manifest)
    if run_protocol.get("frozen_input_lock_sha256") != expected_lock:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} frozen input lock differs"
        )
    adapter_sources = run_protocol.get("adapter_sources")
    runtime_versions = run_protocol.get("runtime_versions")
    runtime_module_origins = run_protocol.get("runtime_module_origins")
    if not isinstance(adapter_sources, dict) or set(adapter_sources) != {
        "common.py",
        "lhgce_identity.py",
        "lhgce_runner.py",
    }:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} adapter-source identity is incomplete"
        )
    if not all(isinstance(value, str) and value for value in adapter_sources.values()):
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} adapter hashes are invalid"
        )
    if not isinstance(runtime_versions, dict) or not runtime_versions:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} runtime versions are missing"
        )
    if not isinstance(runtime_module_origins, dict) or not runtime_module_origins:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} runtime origins are missing"
        )
    device = metrics.get("Device")
    if not isinstance(device, str) or not device:
        raise BenchmarkAuditError(
            f"LHGCE epoch {budget} fold {fold} execution device is missing"
        )
    return {
        "adapter_sources": adapter_sources,
        "runtime_versions": runtime_versions,
        "runtime_module_origins": runtime_module_origins,
        "device": device,
        "source_dir": arguments.get("source_dir"),
    }


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BenchmarkAuditError(
            f"refusing to overwrite non-empty budget-selection output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_and_evaluate_lhgce_budgets(
    *,
    protocol: BenchmarkProtocol,
    protocol_manifest: Path,
    candidate_roots: Mapping[int, Path],
    output_dir: Path,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    prediction_loader: Callable[[str | Path], PredictionTable] = load_predictions,
) -> dict[str, object]:
    """Run fail-closed nested selection without modifying candidate artifacts."""

    budgets = tuple(sorted(candidate_roots))
    if budgets != CANDIDATE_BUDGETS:
        raise BenchmarkAuditError(
            f"candidate budgets must be exactly {CANDIDATE_BUDGETS}, got {budgets}"
        )
    folds = tuple(int(fold) for fold in folds)
    if not folds or len(set(folds)) != len(folds):
        raise BenchmarkAuditError("folds must be a non-empty unique sequence")
    if protocol.dataset != "tcm-suite":
        raise BenchmarkAuditError("LHGCE budget selection is defined only for TCM-suite")

    # Phase 0: completeness is checked globally before any score is loaded.
    artifacts = _require_complete_candidates(candidate_roots, folds)
    output_dir = _prepare_output_dir(output_dir)

    # Phase 1: validation-only candidate scoring and immutable selection.
    validation_records: dict[int, dict[int, dict[str, object]]] = {}
    selected: dict[int, int] = {}
    validation_tables: dict[tuple[int, int], PredictionTable] = {}
    for fold in folds:
        expected_validation = _expected_split(protocol, fold, "validation")
        fold_records: dict[int, dict[str, object]] = {}
        ranking_for_selection: dict[int, dict[str, float]] = {}
        for budget in budgets:
            path = artifacts[(fold, budget)]["validation"]
            loaded = prediction_loader(path)
            table = validate_prediction_coverage(loaded, **expected_validation)
            validation_tables[(fold, budget)] = table
            ranking = _validation_ranking(table)
            ranking_for_selection[budget] = ranking
            fold_records[budget] = {
                "budget": budget,
                "validation_AUROC": ranking["AUROC"],
                "validation_AUPRC": ranking["AUPRC"],
                "validation_AveragePrecision": ranking["AveragePrecision"],
                "validation_predictions": str(path.resolve()),
                "validation_predictions_sha256": sha256(path),
            }
        selected[fold] = _select_budget(ranking_for_selection)
        validation_records[fold] = fold_records

    decision = {
        "schema_version": 1,
        "dataset": protocol.dataset,
        "seed": protocol.seed,
        "protocol_manifest": str(protocol_manifest.resolve()),
        "candidate_budgets": list(budgets),
        "folds": list(folds),
        "selection_metric": "validation trapezoidal AUPRC",
        "selection_metric_definition": AUPRC_DEFINITION,
        "tie_break": "lowest requested epoch budget on an exact AUPRC tie",
        "outer_test_access": (
            "no test prediction content was loaded before this decision; after "
            "metadata audit, exactly one selected test artifact is loaded per fold"
        ),
        "fold_decisions": [
            {
                "fold": fold,
                "selected_budget": selected[fold],
                "candidates": [validation_records[fold][budget] for budget in budgets],
            }
            for fold in folds
        ],
    }
    # This on-disk decision is written before metadata or selected test content
    # is read.  A subsequent failure leaves no publishable summary.json.
    write_json(output_dir / "selection_decision.json", decision)

    # Phase 2: the decision is fixed.  Metadata may contain recorded test
    # metrics, so it is intentionally not parsed until after the decision file.
    execution_identity: dict[str, object] | None = None
    for fold in folds:
        for budget in budgets:
            candidate_identity = _assert_candidate_metadata(
                protocol=protocol,
                protocol_manifest=protocol_manifest,
                fold=fold,
                budget=budget,
                paths=artifacts[(fold, budget)],
            )
            if execution_identity is None:
                execution_identity = candidate_identity
            elif candidate_identity != execution_identity:
                raise BenchmarkAuditError(
                    "LHGCE budget candidates differ in adapter, runtime, device, "
                    "or upstream source identity"
                )

    # Phase 3: load exactly one test-prediction file per fold.  Unselected test
    # arrays are never opened by this tool.
    fold_results: list[dict[str, object]] = []
    selected_test_reads = 0
    for fold in folds:
        budget = selected[fold]
        test_path = artifacts[(fold, budget)]["test"]
        test_table = validate_prediction_coverage(
            prediction_loader(test_path), **_expected_split(protocol, fold, "test")
        )
        selected_test_reads += 1
        validation_table = validation_tables[(fold, budget)]
        metrics = evaluate_validation_and_test(
            validation_table.label,
            validation_table.score,
            test_table.label,
            test_table.score,
        )
        fold_results.append(
            {
                "fold": fold,
                "selected_budget": budget,
                "selected_validation_AUPRC": validation_records[fold][budget][
                    "validation_AUPRC"
                ],
                "test_predictions": str(test_path.resolve()),
                "test_predictions_sha256": sha256(test_path),
                **metrics,
            }
        )
    if selected_test_reads != len(folds):
        raise BenchmarkAuditError("selected test predictions were not read exactly once per fold")

    aggregate = aggregate_fold_metrics(fold_results)
    final = {
        **decision,
        "formal_eligible": True,
        "test_prediction_files_read": selected_test_reads,
        "unselected_test_prediction_files_read": 0,
        "AUPRC_Definition": AUPRC_DEFINITION,
        "AveragePrecision_Definition": AVERAGE_PRECISION_DEFINITION,
        "fold_results": fold_results,
        "aggregate": aggregate,
    }
    write_json(output_dir / "summary.json", final)
    with (output_dir / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "fold",
            "selected_budget",
            "selected_validation_AUPRC",
            "AUROC",
            "AUPRC",
            "AveragePrecision",
            "F1",
            "Precision",
            "Recall",
            "Accuracy",
            "Operating_Threshold",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_results:
            writer.writerow({name: row[name] for name in fieldnames})
    return final


def _candidate_spec(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", 1)
        budget = int(budget_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "candidate must have the form REQUESTED_EPOCHS=/path/to/tasks/LHGCE"
        ) from error
    if not path_text:
        raise argparse.ArgumentTypeError("candidate path must not be empty")
    return budget, Path(path_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate_spec,
        required=True,
        metavar="EPOCHS=METHOD_ROOT",
        help=(
            "Repeat once for each locked budget; METHOD_ROOT contains fold_1, "
            "..., fold_5 directories."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6))
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if tuple(args.folds) != (1, 2, 3, 4, 5):
        raise BenchmarkAuditError("formal LHGCE budget selection requires folds 1 2 3 4 5")
    candidate_roots: dict[int, Path] = {}
    for budget, path in args.candidate:
        if budget in candidate_roots:
            raise BenchmarkAuditError(f"duplicate candidate budget {budget}")
        candidate_roots[budget] = path
    protocol = load_protocol_manifest(args.protocol_manifest)
    select_and_evaluate_lhgce_budgets(
        protocol=protocol,
        protocol_manifest=args.protocol_manifest,
        candidate_roots=candidate_roots,
        output_dir=args.output_dir,
        folds=args.folds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
