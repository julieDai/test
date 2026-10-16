#!/usr/bin/env python3
"""Select a strict TCM SMGCL epoch budget using inner validation only.

The four candidate runs remain read-only.  The selector first requires a
complete 4-by-5 artifact grid, loads only validation predictions, freezes one
budget per outer fold, audits all candidate provenance, and finally loads one
selected test-prediction artifact per fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


CANDIDATE_BUDGETS = (16, 30, 48, 64)
UPSTREAM_COMMIT = "09fa09795bc610018f906d26bea5b4259233400a"
BATCH_SIZE = 4096


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


def _require_complete_grid(
    candidate_roots: Mapping[int, Path], folds: Sequence[int]
) -> dict[tuple[int, int], dict[str, Path]]:
    artifacts: dict[tuple[int, int], dict[str, Path]] = {}
    missing: list[str] = []
    for budget in sorted(candidate_roots):
        for fold in folds:
            fold_dir = candidate_roots[budget].resolve() / f"fold_{fold}"
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
            "SMGCL budget selection requires a complete candidate grid; missing:\n"
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
    metrics = evaluate_test_scores(table.label, table.score, threshold=0.5)
    return {
        "AUROC": float(metrics["AUROC"]),
        "AUPRC": float(metrics["AUPRC"]),
        "AveragePrecision": float(metrics["AveragePrecision"]),
    }


def _select_budget(records: Mapping[int, Mapping[str, float]]) -> int:
    if not records:
        raise BenchmarkAuditError("no validation candidates were supplied")
    return min(records, key=lambda budget: (-float(records[budget]["AUPRC"]), budget))


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BenchmarkAuditError(
            f"refusing to overwrite non-empty budget-selection output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _assert_candidate_metadata(
    *,
    protocol: BenchmarkProtocol,
    protocol_manifest: Path,
    fold: int,
    budget: int,
    paths: Mapping[str, Path],
) -> dict[str, object]:
    metrics = _load_json(paths["metrics"])
    run_protocol = _load_json(paths["protocol"])
    fold_rows = protocol.fold_rows(fold)
    train_pairs, train_labels, _ = fold_split(fold_rows, "train")
    expected_steps = budget * int(math.ceil(len(train_pairs) / BATCH_SIZE))
    expected_positive_graph_rows = int((train_labels == 1).sum())

    if metrics.get("Dataset") != "tcm-suite" or run_protocol.get("dataset") != "tcm-suite":
        raise BenchmarkAuditError(f"SMGCL epoch {budget} fold {fold} dataset differs")
    if _strict_int(metrics.get("Fold"), "metrics Fold", minimum=1) != fold:
        raise BenchmarkAuditError(f"SMGCL epoch {budget} fold identity differs")
    if _strict_int(metrics.get("Seed"), "metrics Seed") != protocol.seed:
        raise BenchmarkAuditError(f"SMGCL epoch {budget} fold seed differs")
    expected_fields = {
        "Epochs_Requested": budget,
        "Epochs_Executed": budget,
        "Batch_Size": BATCH_SIZE,
        "Training_Rows": len(train_pairs),
        "Training_Positive_Graph_Rows": expected_positive_graph_rows,
        "Optimizer_Steps": expected_steps,
    }
    for name, expected in expected_fields.items():
        actual = _strict_int(metrics.get(name), f"metrics {name}", minimum=1)
        if actual != expected:
            raise BenchmarkAuditError(
                f"SMGCL epoch {budget} fold {fold} {name}={actual}, expected {expected}"
            )
    if metrics.get("Evaluation_Mode") != "validation" or run_protocol.get(
        "evaluation_mode"
    ) != "validation":
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} is not validation based"
        )
    if run_protocol.get("training_split") != "inner training fold":
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} did not use inner training only"
        )
    if (
        metrics.get("Smoke") is not False
        or metrics.get("Diagnostic") is not False
        or metrics.get("Execution_Mode") != "formal-candidate"
        or metrics.get("Formal_Eligible") is not True
    ):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} is not a formal candidate"
        )
    if (
        run_protocol.get("diagnostic") is not False
        or run_protocol.get("execution_mode") != "formal-candidate"
        or run_protocol.get("formal_eligible") is not True
    ):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} protocol mode is not formal"
        )
    if metrics.get("Scalable_Mode") is not True or metrics.get(
        "Scalable_Mode_Request"
    ) != "on":
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} scalable mode differs"
        )
    if metrics.get("Upstream_Commit") != UPSTREAM_COMMIT:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} upstream commit differs"
        )
    if run_protocol.get("model") != metrics:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} protocol/metrics payloads differ"
        )
    recorded_manifest = run_protocol.get("protocol_manifest")
    if not isinstance(recorded_manifest, str) or Path(recorded_manifest).resolve() != (
        protocol_manifest.resolve()
    ):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} protocol manifest path differs"
        )

    arguments = run_protocol.get("arguments")
    if not isinstance(arguments, dict):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} protocol arguments are missing"
        )
    locked_arguments = {
        "fold": fold,
        "seed": protocol.seed,
        "epochs": budget,
        "batch_size": BATCH_SIZE,
        "learning_rate": 5e-4,
        "dropout": 0.4,
        "embedding_dim": 64,
        "neighbor_embedding_dim": 32,
        "hidden_dims": [64, 32],
        "herb_neighbors": 3,
        "disease_neighbors": 3,
        "loss_fn": "focal",
        "alpha": 0.5,
        "gamma": 2.0,
        "restart_probability": 0.1,
        "rwr_mode": "official-exact",
        "scalable_mode": "on",
        "legacy_test_oracle": False,
        "smoke": False,
        "diagnostic": False,
    }
    for name, expected in locked_arguments.items():
        if arguments.get(name) != expected:
            raise BenchmarkAuditError(
                f"SMGCL epoch {budget} fold {fold} argument {name}="
                f"{arguments.get(name)!r}, expected {expected!r}"
            )

    expected_manifest = protocol.input_manifest()
    observed_manifest = {key: run_protocol.get(key) for key in expected_manifest}
    if observed_manifest != expected_manifest:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} frozen input manifest differs"
        )
    if run_protocol.get("frozen_input_lock_sha256") != validate_frozen_input_manifest(
        expected_manifest
    ):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} frozen input lock differs"
        )

    adapter_sources = run_protocol.get("adapter_sources")
    runtime_versions = run_protocol.get("runtime_versions")
    runtime_module_origins = run_protocol.get("runtime_module_origins")
    if not isinstance(adapter_sources, dict) or set(adapter_sources) != {
        "common.py",
        "smgcl_runner.py",
    }:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} adapter-source identity is incomplete"
        )
    if not all(isinstance(value, str) and value for value in adapter_sources.values()):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} adapter hashes are invalid"
        )
    if not isinstance(runtime_versions, dict) or not runtime_versions:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} runtime versions are missing"
        )
    if not isinstance(runtime_module_origins, dict) or not runtime_module_origins:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} runtime origins are missing"
        )
    if run_protocol.get("runtime_environment_isolation_enforced") is not True:
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} runtime isolation was not enforced"
        )
    device = metrics.get("Device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise BenchmarkAuditError(
            f"SMGCL epoch {budget} fold {fold} was not executed on CUDA"
        )
    return {
        "adapter_sources": adapter_sources,
        "runtime_versions": runtime_versions,
        "runtime_module_origins": runtime_module_origins,
        "device": device,
        "source_dir": arguments.get("source_dir"),
        "cache_dir": arguments.get("cache_dir"),
        "similarity_cache_sha256": run_protocol.get("similarity_cache_sha256"),
    }


def select_and_evaluate_smgcl_budgets(
    *,
    protocol: BenchmarkProtocol,
    protocol_manifest: Path,
    candidate_roots: Mapping[int, Path],
    output_dir: Path,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    prediction_loader: Callable[[str | Path], PredictionTable] = load_predictions,
) -> dict[str, object]:
    budgets = tuple(sorted(candidate_roots))
    if budgets != CANDIDATE_BUDGETS:
        raise BenchmarkAuditError(
            f"candidate budgets must be exactly {CANDIDATE_BUDGETS}, got {budgets}"
        )
    folds = tuple(int(fold) for fold in folds)
    if not folds or len(set(folds)) != len(folds):
        raise BenchmarkAuditError("folds must be a non-empty unique sequence")
    if protocol.dataset != "tcm-suite":
        raise BenchmarkAuditError("SMGCL budget selection is defined only for TCM-suite")

    artifacts = _require_complete_grid(candidate_roots, folds)
    output_dir = _prepare_output_dir(output_dir)
    validation_records: dict[int, dict[int, dict[str, object]]] = {}
    validation_tables: dict[tuple[int, int], PredictionTable] = {}
    selected: dict[int, int] = {}
    for fold in folds:
        expected = _expected_split(protocol, fold, "validation")
        rankings: dict[int, dict[str, float]] = {}
        records: dict[int, dict[str, object]] = {}
        for budget in budgets:
            path = artifacts[(fold, budget)]["validation"]
            table = validate_prediction_coverage(prediction_loader(path), **expected)
            validation_tables[(fold, budget)] = table
            ranking = _validation_ranking(table)
            rankings[budget] = ranking
            records[budget] = {
                "budget": budget,
                "validation_AUROC": ranking["AUROC"],
                "validation_AUPRC": ranking["AUPRC"],
                "validation_AveragePrecision": ranking["AveragePrecision"],
                "validation_predictions": str(path.resolve()),
                "validation_predictions_sha256": sha256(path),
            }
        selected[fold] = _select_budget(rankings)
        validation_records[fold] = records

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
    write_json(output_dir / "selection_decision.json", decision)

    execution_identity: dict[str, object] | None = None
    for fold in folds:
        for budget in budgets:
            identity = _assert_candidate_metadata(
                protocol=protocol,
                protocol_manifest=protocol_manifest,
                fold=fold,
                budget=budget,
                paths=artifacts[(fold, budget)],
            )
            if execution_identity is None:
                execution_identity = identity
            elif identity != execution_identity:
                raise BenchmarkAuditError(
                    "SMGCL budget candidates differ in adapter, runtime, device, "
                    "source, cache, or similarity identity"
                )

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
            "candidate must have the form EPOCHS=/path/to/tasks/SMGCL"
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
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6))
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if tuple(args.folds) != (1, 2, 3, 4, 5):
        raise BenchmarkAuditError("formal SMGCL budget selection requires folds 1 2 3 4 5")
    roots: dict[int, Path] = {}
    for budget, path in args.candidate:
        if budget in roots:
            raise BenchmarkAuditError(f"duplicate candidate budget {budget}")
        roots[budget] = path
    protocol = load_protocol_manifest(args.protocol_manifest)
    select_and_evaluate_smgcl_budgets(
        protocol=protocol,
        protocol_manifest=args.protocol_manifest,
        candidate_roots=roots,
        output_dir=args.output_dir,
        folds=args.folds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
