#!/usr/bin/env python3
"""Fail-closed re-audit of pre-schema official baseline predictions.

The October 2026 Ethnobotany SMGCL, MilGNet, and LHGCE runs predate two
bookkeeping fields required by the current auditor: ``pair_order`` in the
frozen input lock and ``Formal_Eligible`` in each metrics file.  This module
does not edit those historical artifacts and does not invent either field.
Instead, it accepts them only when a committed curation record, committed
runner source, the old frozen-input digest, every saved prediction, and every
published fold metric all agree.  It then writes a new hash-backed audit
package that records the narrowly recovered schema facts.

This is intentionally separate from :mod:`audit_results`: ordinary formal
audits remain strict and continue to reject artifacts with missing fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from astra.benchmark.metrics import (
    AUPRC_DEFINITION,
    AVERAGE_PRECISION_DEFINITION,
    BenchmarkAuditError,
    PredictionTable,
    aggregate_fold_metrics,
    audit_cross_method_predictions,
    audit_milgnet_run,
    evaluate_validation_and_test,
    load_predictions,
    validate_prediction_coverage,
)
from astra.benchmark.protocol import DEFAULT_PAIR_ORDER, load_protocol_manifest

from .audit_results import _metric_mismatches
from .common import validate_frozen_input_manifest, write_json


SCHEMA = "astra-historical-official-baseline-reaudit/v1"
CURATION_SCHEMA = "astra-curated-official-baseline-results/v1"
DEFAULT_METHODS = ("SMGCL", "MilGNet", "LHGCE")
_HEX_SHA1 = re.compile(r"[0-9a-f]{40}\Z")

_RUNNER_SOURCE_PATHS = {
    "common.py": "src/astra/official_baselines/common.py",
    "smgcl_runner.py": "src/astra/official_baselines/smgcl_runner.py",
    "milgnet_runner.py": "src/astra/official_baselines/milgnet_runner.py",
    "lhgce_identity.py": "src/astra/official_baselines/lhgce_identity.py",
    "lhgce_runner.py": "src/astra/official_baselines/lhgce_runner.py",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BenchmarkAuditError(f"required artifact is missing: {resolved}")
    if relative_to is None:
        display = str(resolved)
    else:
        try:
            display = str(resolved.relative_to(relative_to.resolve()))
        except ValueError as error:
            raise BenchmarkAuditError(
                f"artifact escapes its declared root: {resolved}"
            ) from error
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkAuditError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkAuditError(f"JSON artifact is not an object: {path}")
    return payload


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    if not _HEX_SHA1.fullmatch(commit):
        raise BenchmarkAuditError(f"Git commit is not a full lowercase SHA-1: {commit!r}")
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise BenchmarkAuditError(f"unsafe repository-relative path: {relative_path!r}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace").strip()
        raise BenchmarkAuditError(
            f"cannot read committed Git blob {commit}:{relative_path}: {detail}"
        ) from error
    return completed.stdout


def _verify_committed_file(
    *, path: Path, repo_root: Path, commit: str
) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise BenchmarkAuditError(
            f"curation evidence is outside the ASTRA repository: {resolved}"
        ) from error
    disk = resolved.read_bytes()
    committed = _git_blob(repo_root, commit, relative)
    if disk != committed:
        raise BenchmarkAuditError(
            f"curation evidence differs from committed blob {commit}:{relative}"
        )
    return {
        "path": relative,
        "bytes": len(disk),
        "sha256": _sha256_bytes(disk),
        "git_commit": commit,
        "git_blob_sha256": _sha256_bytes(committed),
    }


def _strict_historical_execution_mode(
    *, recorded: Mapping[str, object], run_protocol: Mapping[str, object]
) -> None:
    """Accept only the exact pre-formal-field, non-smoke schema."""

    for field in ("Formal_Eligible", "Diagnostic", "Formal_Ineligibility_Reasons"):
        if field in recorded:
            raise BenchmarkAuditError(
                f"historical migration requires metrics field {field!r} to be absent"
            )
    for field in (
        "formal_eligible",
        "diagnostic",
        "formal_ineligibility_reasons",
        "pair_order",
        "protocol_input_manifest",
    ):
        if field in run_protocol:
            raise BenchmarkAuditError(
                f"historical migration requires protocol field {field!r} to be absent"
            )
    if recorded.get("Smoke") is not False:
        raise BenchmarkAuditError("historical metrics must explicitly record Smoke=false")
    arguments = run_protocol.get("arguments")
    if not isinstance(arguments, Mapping) or arguments.get("smoke") is not False:
        raise BenchmarkAuditError(
            "historical protocol arguments must explicitly record smoke=false"
        )
    embedded_metrics = run_protocol.get("model")
    if not isinstance(embedded_metrics, Mapping) or embedded_metrics.get("Smoke") is not False:
        raise BenchmarkAuditError(
            "historical protocol must embed model metrics with Smoke=false"
        )


def _historical_input_manifest(
    *, protocol, raw_manifest: Mapping[str, object]
) -> tuple[dict[str, object], str, str]:
    """Recover the v1 lock only when ``pair_order`` is the sole omission."""

    if "pair_order" in raw_manifest:
        raise BenchmarkAuditError(
            "historical migration is restricted to manifests that predate pair_order"
        )
    current = protocol.input_manifest()
    if current.get("pair_order") != DEFAULT_PAIR_ORDER or DEFAULT_PAIR_ORDER != "seeded":
        raise BenchmarkAuditError(
            "the omitted historical pair_order cannot be recovered as seeded"
        )
    historical = {key: value for key, value in current.items() if key != "pair_order"}
    current_lock = validate_frozen_input_manifest(current)
    historical_lock = validate_frozen_input_manifest(historical)
    return historical, historical_lock, current_lock


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(expected, bool) and isinstance(actual, str):
        if actual == "True":
            return expected is True
        if actual == "False":
            return expected is False
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return actual == expected


def _published_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise BenchmarkAuditError(f"cannot read published fold table {path}: {error}") from error
    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        try:
            key = (str(row["Method"]), int(row["Fold"]))
        except (KeyError, TypeError, ValueError) as error:
            raise BenchmarkAuditError("published fold table has an invalid key") from error
        if key in indexed:
            raise BenchmarkAuditError(f"published fold table duplicates {key}")
        indexed[key] = row
    return indexed


def _compare_published_fold(
    *, published: Mapping[str, str], historical_row: Mapping[str, object]
) -> None:
    missing = [key for key in historical_row if key not in published]
    mismatches = {
        key: {"published": published.get(key), "reaudited": value}
        for key, value in historical_row.items()
        if key in published and not _same_value(published[key], value)
    }
    if missing or mismatches:
        raise BenchmarkAuditError(
            "historical predictions differ from the committed published fold table: "
            + json.dumps(
                {"missing_columns": missing, "mismatches": mismatches},
                sort_keys=True,
                default=str,
            )
        )


def _verify_runner_sources(
    *,
    run_protocol: Mapping[str, object],
    repo_root: Path,
    source_commit: str,
) -> dict[str, str]:
    recorded = run_protocol.get("adapter_sources")
    if not isinstance(recorded, Mapping) or not recorded:
        raise BenchmarkAuditError("historical protocol lacks adapter source hashes")
    verified: dict[str, str] = {}
    for name, expected_hash in recorded.items():
        path = _RUNNER_SOURCE_PATHS.get(str(name))
        if path is None or not isinstance(expected_hash, str):
            raise BenchmarkAuditError(f"unsupported historical adapter source: {name!r}")
        actual_hash = _sha256_bytes(_git_blob(repo_root, source_commit, path))
        if actual_hash != expected_hash:
            raise BenchmarkAuditError(
                f"historical adapter hash for {name} does not match source commit"
            )
        verified[str(name)] = actual_hash
    return verified


def _verify_method_configuration(
    *,
    method: str,
    arguments: Mapping[str, object],
    recorded: Mapping[str, object],
    configuration: Mapping[str, object],
) -> None:
    """Bind recorded runner arguments to the committed curation settings."""

    if method == "SMGCL":
        actual = {
            "epochs": arguments.get("epochs"),
            "batch_size": arguments.get("batch_size"),
            "initial_learning_rate": recorded.get("Initial_Learning_Rate"),
            "maximum_learning_rate": arguments.get("learning_rate"),
            "dropout": arguments.get("dropout"),
            "embedding_dimension": arguments.get("embedding_dim"),
            "projection_dimension": arguments.get("neighbor_embedding_dim"),
            "neighbors": [
                arguments.get("herb_neighbors"),
                arguments.get("disease_neighbors"),
            ],
            "focal_alpha": arguments.get("alpha"),
            "focal_gamma": arguments.get("gamma"),
            "rwr_restart": arguments.get("restart_probability"),
            "rwr_mode": arguments.get("rwr_mode"),
        }
        attention = str(recorded.get("Query_Attention_Implementation", "")).lower()
        if "exact" not in attention or "vector" not in attention:
            raise BenchmarkAuditError(
                "SMGCL historical run lacks exact vectorized attention provenance"
            )
        actual["query_attention"] = "mathematically exact vectorized adaptation"
    elif method == "MilGNet":
        actual = {
            "epochs": arguments.get("epochs"),
            "batch_size": arguments.get("batch_size"),
            "learning_rate": arguments.get("learning_rate"),
            "weight_decay": arguments.get("weight_decay"),
            "neighbors": arguments.get("neighbors"),
            "hidden_features": arguments.get("hidden_features"),
            "layers": arguments.get("layers"),
            "aggregation": arguments.get("aggregate"),
            "instance_topk": arguments.get("instance_topk"),
            "dropout": arguments.get("dropout"),
            "batch_normalization": arguments.get("batch_norm"),
        }
        if arguments.get("allow_unpinned_source") is not False:
            raise BenchmarkAuditError("MilGNet historical source was not pin-enforced")
    elif method == "LHGCE":
        actual = {
            "epochs_requested": arguments.get("epochs"),
            "optimizer_updates_executed": recorded.get("Optimizer_Steps"),
            "learning_rate": arguments.get("learning_rate"),
            "layers": arguments.get("layers"),
            "neurons": arguments.get("neurons"),
            "aggregate_linear": arguments.get("agg_linear"),
            "aggregate_convolution": arguments.get("agg_conv"),
            "aggregate_heterogeneous": arguments.get("agg_hetero"),
            "mapped_compound_protein_relation": (
                "included" if arguments.get("include_mapped_cp") else "excluded"
            ),
        }
    else:  # pragma: no cover - guarded by the exact method tuple.
        raise BenchmarkAuditError(f"unsupported historical method: {method}")
    mismatches = {
        key: {"recorded": actual.get(key), "curated": expected}
        for key, expected in configuration.items()
        if key not in actual or not _same_value(actual.get(key), expected)
    }
    if mismatches:
        raise BenchmarkAuditError(
            f"{method} runner configuration differs from committed curation: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


def _verify_runtime(
    *,
    method: str,
    recorded: Mapping[str, object],
    run_protocol: Mapping[str, object],
    provenance_runtime: Mapping[str, object],
) -> None:
    if recorded.get("Device") != "cuda":
        raise BenchmarkAuditError(f"{method} historical run was not recorded on CUDA")
    versions = run_protocol.get("runtime_versions")
    environments = provenance_runtime.get("method_environments")
    if not isinstance(versions, Mapping) or not isinstance(environments, Mapping):
        raise BenchmarkAuditError("runtime provenance is incomplete")
    method_environment = environments.get(method)
    if not isinstance(method_environment, Mapping):
        raise BenchmarkAuditError(f"committed runtime lacks {method} environment")
    expected_versions = {
        "python": provenance_runtime.get("python"),
        "numpy": provenance_runtime.get("numpy"),
        "pandas": provenance_runtime.get("pandas"),
        "scipy": provenance_runtime.get("scipy"),
        "scikit-learn": provenance_runtime.get("scikit_learn"),
        "torch": provenance_runtime.get("torch"),
        "torch-geometric": provenance_runtime.get("torch_geometric"),
        "dgl": method_environment.get("dgl"),
    }
    mismatches = {
        name: {"recorded": versions.get(name), "curated": expected}
        for name, expected in expected_versions.items()
        if not isinstance(expected, str) or versions.get(name) != expected
    }
    if method_environment.get("device") != "cuda":
        mismatches["device"] = {
            "recorded": recorded.get("Device"),
            "curated": method_environment.get("device"),
        }
    if mismatches:
        raise BenchmarkAuditError(
            f"{method} runtime differs from committed curation: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise BenchmarkAuditError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary_rows(
    summaries: Mapping[str, Mapping[str, Mapping[str, float | int]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, metrics in summaries.items():
        row: dict[str, object] = {"Method": method}
        for metric, values in metrics.items():
            row[f"{metric}_Mean"] = values["mean"]
            row[f"{metric}_Sample_SD"] = values["sample_sd"]
            row[f"{metric}_Folds"] = values["folds"]
        rows.append(row)
    return rows


def reaudit_historical_run(
    *,
    protocol_manifest: Path | str,
    run_root: Path | str,
    published_provenance: Path | str,
    published_fold_metrics: Path | str,
    published_summary: Path | str,
    curation_commit: str,
    repo_root: Path | str,
    methods: Sequence[str] = DEFAULT_METHODS,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Re-audit a narrowly recognized pre-schema run without changing it."""

    repo_root = Path(repo_root).resolve()
    protocol_manifest = Path(protocol_manifest).resolve()
    run_root = Path(run_root).resolve()
    published_provenance = Path(published_provenance).resolve()
    published_fold_metrics = Path(published_fold_metrics).resolve()
    published_summary = Path(published_summary).resolve()
    methods = tuple(dict.fromkeys(str(value) for value in methods))
    folds = tuple(dict.fromkeys(int(value) for value in folds))
    if methods != DEFAULT_METHODS:
        raise BenchmarkAuditError(
            f"historical migration requires exactly {DEFAULT_METHODS}, got {methods}"
        )
    if folds != (1, 2, 3, 4, 5):
        raise BenchmarkAuditError("historical migration requires folds 1,2,3,4,5 in order")

    committed_evidence = {
        "provenance": _verify_committed_file(
            path=published_provenance,
            repo_root=repo_root,
            commit=curation_commit,
        ),
        "fold_metrics": _verify_committed_file(
            path=published_fold_metrics,
            repo_root=repo_root,
            commit=curation_commit,
        ),
        "summary": _verify_committed_file(
            path=published_summary,
            repo_root=repo_root,
            commit=curation_commit,
        ),
    }
    provenance = _load_json(published_provenance)
    if provenance.get("schema") != CURATION_SCHEMA:
        raise BenchmarkAuditError("unsupported committed curation schema")
    if provenance.get("status") != "formal-audit-passed":
        raise BenchmarkAuditError("committed curation did not declare a formal audit")

    protocol = load_protocol_manifest(protocol_manifest)
    if protocol.embedding_dir is not None:
        raise BenchmarkAuditError(
            "historical reuse is restricted to the feature-independent protocol"
        )
    raw_manifest = _load_json(protocol_manifest)
    historical_input, historical_lock, current_lock = _historical_input_manifest(
        protocol=protocol, raw_manifest=raw_manifest
    )
    protocol_record = provenance.get("protocol")
    if not isinstance(protocol_record, Mapping):
        raise BenchmarkAuditError("committed provenance lacks protocol evidence")
    if protocol_record.get("manifest_sha256") != _sha256(protocol_manifest):
        raise BenchmarkAuditError("protocol manifest hash differs from committed provenance")
    if protocol_record.get("input_manifest_sha256") != historical_lock:
        raise BenchmarkAuditError("historical input lock differs from committed provenance")
    artifacts = raw_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise BenchmarkAuditError("protocol manifest lacks artifact hashes")
    protocol_arrays = artifacts.get("protocol_arrays.npz")
    entity_mappings = artifacts.get("entity_mappings.json")
    if not isinstance(protocol_arrays, Mapping) or not isinstance(entity_mappings, Mapping):
        raise BenchmarkAuditError("protocol manifest artifact records are incomplete")
    if protocol_record.get("protocol_arrays_sha256") != protocol_arrays.get("sha256"):
        raise BenchmarkAuditError("protocol array hash differs from committed provenance")
    if protocol_record.get("entity_mappings_sha256") != entity_mappings.get("sha256"):
        raise BenchmarkAuditError("entity mapping hash differs from committed provenance")
    if provenance.get("dataset") != protocol.dataset or provenance.get("seed") != protocol.seed:
        raise BenchmarkAuditError("committed provenance dataset/seed differs")
    if provenance.get("folds") != list(folds):
        raise BenchmarkAuditError("committed provenance fold set differs")

    code = provenance.get("code")
    if not isinstance(code, Mapping) or not isinstance(code.get("astra_run_commit"), str):
        raise BenchmarkAuditError("committed provenance lacks the ASTRA run commit")
    source_commit = str(code["astra_run_commit"])
    if not _HEX_SHA1.fullmatch(source_commit):
        raise BenchmarkAuditError("committed ASTRA run commit is not a full SHA-1")
    published = _published_rows(published_fold_metrics)
    if set(published) != {(method, fold) for method in methods for fold in folds}:
        raise BenchmarkAuditError("committed published fold table coverage differs")

    upstream = provenance.get("upstream")
    if not isinstance(upstream, Mapping):
        raise BenchmarkAuditError("committed provenance lacks upstream identities")
    method_configurations = provenance.get("method_configurations")
    provenance_runtime = provenance.get("runtime")
    if not isinstance(method_configurations, Mapping):
        raise BenchmarkAuditError("committed provenance lacks method configurations")
    if not isinstance(provenance_runtime, Mapping):
        raise BenchmarkAuditError("committed provenance lacks runtime evidence")
    rows: list[dict[str, object]] = []
    predictions: dict[str, dict[int, PredictionTable]] = {}
    original_artifacts: list[dict[str, object]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    generation_ids: dict[str, set[str]] = {method: set() for method in methods}

    for method in methods:
        predictions[method] = {}
        upstream_method = upstream.get(method)
        if not isinstance(upstream_method, Mapping) or not isinstance(
            upstream_method.get("commit"), str
        ):
            raise BenchmarkAuditError(f"committed provenance lacks {method} upstream commit")
        method_configuration = method_configurations.get(method)
        if not isinstance(method_configuration, Mapping):
            raise BenchmarkAuditError(
                f"committed provenance lacks {method} configuration"
            )
        for fold in folds:
            task = run_root / "tasks" / method / f"fold_{fold}"
            paths = {
                "metrics": task / "results" / "metrics.json",
                "protocol": task / "protocol.json",
                "validation_predictions": task / "results" / "validation_predictions.npz",
                "test_predictions": task / "results" / "test_predictions.npz",
            }
            for kind, path in paths.items():
                record = _artifact(path, relative_to=run_root)
                record.update({"method": method, "fold": fold, "kind": kind})
                original_artifacts.append(record)
            recorded = _load_json(paths["metrics"])
            run_protocol = _load_json(paths["protocol"])
            _strict_historical_execution_mode(
                recorded=recorded, run_protocol=run_protocol
            )
            if run_protocol.get("model") != recorded:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} protocol-embedded metrics differ from "
                    "the metrics artifact"
                )
            if recorded.get("Dataset") != protocol.dataset:
                raise BenchmarkAuditError(f"{method} fold {fold} dataset differs")
            if recorded.get("Fold") != fold or recorded.get("Seed") != protocol.seed:
                raise BenchmarkAuditError(f"{method} fold {fold} fold/seed differs")
            if run_protocol.get("dataset") != protocol.dataset:
                raise BenchmarkAuditError(f"{method} fold {fold} protocol dataset differs")
            run_input = {
                key: run_protocol.get(key) for key in historical_input
            }
            if run_input != historical_input:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} historical protocol input differs"
                )
            if run_protocol.get("frozen_input_lock_sha256") != historical_lock:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} historical frozen input lock differs"
                )
            arguments = run_protocol["arguments"]
            assert isinstance(arguments, Mapping)
            generation_id = run_protocol.get("run_generation_id")
            if not isinstance(generation_id, str) or not generation_id:
                raise BenchmarkAuditError(f"{method} fold {fold} lacks run generation ID")
            if arguments.get("run_generation_id") != generation_id:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} generation IDs disagree"
                )
            generation_ids[method].add(generation_id)
            _verify_method_configuration(
                method=method,
                arguments=arguments,
                recorded=recorded,
                configuration=method_configuration,
            )
            _verify_runtime(
                method=method,
                recorded=recorded,
                run_protocol=run_protocol,
                provenance_runtime=provenance_runtime,
            )
            verified_sources = _verify_runner_sources(
                run_protocol=run_protocol,
                repo_root=repo_root,
                source_commit=source_commit,
            )
            previous_sources = source_hashes.setdefault(method, verified_sources)
            if previous_sources != verified_sources:
                raise BenchmarkAuditError(f"{method} adapter source hashes vary by fold")
            if recorded.get("Upstream_Commit") != upstream_method.get("commit"):
                raise BenchmarkAuditError(f"{method} fold {fold} upstream commit differs")

            expected = protocol.fold_rows(fold)
            validation = validate_prediction_coverage(
                load_predictions(paths["validation_predictions"]),
                expected_row_id=expected.validation_row_indices,
                expected_pair=expected.validation_pairs,
                expected_label=expected.validation_labels,
            )
            test = validate_prediction_coverage(
                load_predictions(paths["test_predictions"]),
                expected_row_id=expected.test_row_indices,
                expected_pair=expected.test_pairs,
                expected_label=expected.test_labels,
            )
            recomputed = evaluate_validation_and_test(
                validation.label, validation.score, test.label, test.score
            )
            mismatches = _metric_mismatches(
                recorded=recorded,
                recomputed=recomputed,
                keys=(
                    "Validation_Selected_Threshold",
                    "Validation_F1",
                    "Validation_Precision",
                    "Validation_Recall",
                    "Validation_Accuracy",
                    "Threshold_Candidates",
                    "AUROC",
                    "Operating_Threshold",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                ),
            )
            if mismatches:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} metrics differ from predictions: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            optimizer_steps = recorded.get("Optimizer_Steps")
            if type(optimizer_steps) is not int or optimizer_steps < 1:
                raise BenchmarkAuditError(
                    f"{method} fold {fold} optimizer steps are invalid"
                )
            if method == "MilGNet":
                audit_milgnet_run(test, optimizer_steps=optimizer_steps)
            row: dict[str, object] = {
                "Dataset": protocol.dataset,
                "Method": method,
                "Fold": fold,
                "Seed": protocol.seed,
                "Smoke": False,
                "Diagnostic": False,
                "Historical_Schema_Migration": True,
                "Optimizer_Steps": optimizer_steps,
                "Validation_Rows": len(validation),
                "Test_Rows": len(test),
                **recomputed,
            }
            historical_published_row = {
                "Dataset": protocol.dataset,
                "Method": method,
                "Fold": fold,
                "Seed": protocol.seed,
                "Smoke": False,
                "Optimizer_Steps": optimizer_steps,
                "Validation_Rows": len(validation),
                "Test_Rows": len(test),
                **{
                    key: value
                    for key, value in recomputed.items()
                    if key
                    not in {
                        "AUPRC_Definition",
                        "AveragePrecision",
                        "AveragePrecision_Definition",
                    }
                },
                # The committed table predates metric harmonization and calls
                # average precision AUPRC.  Both names are explicit above in
                # the new row; only this comparison view preserves that label.
                "AUPRC": recomputed["AveragePrecision"],
            }
            _compare_published_fold(
                published=published[(method, fold)],
                historical_row=historical_published_row,
            )
            rows.append(row)
            predictions[method][fold] = test

    audit_cross_method_predictions(predictions)
    summaries = {
        method: aggregate_fold_metrics(
            [row for row in rows if row["Method"] == method]
        )
        for method in methods
    }
    audit_manifest: dict[str, object] = {
        "schema": SCHEMA,
        "status": "formal-historical-evidence-audit-passed",
        "formal_result": True,
        "training_rerun_required": False,
        "scope": {
            "dataset": protocol.dataset,
            "seed": protocol.seed,
            "methods": list(methods),
            "folds": list(folds),
        },
        "schema_migration": {
            "original_protocol_version": raw_manifest.get("protocol_version"),
            "recovered_pair_order": DEFAULT_PAIR_ORDER,
            "recovery_basis": (
                "pair_order is absent from the immutable v1 manifest; the v1 "
                "loader default and recorded fold rule are seeded"
            ),
            "original_formal_eligible_field": "absent",
            "formal_evidence_basis": (
                "committed formal-audit-passed curation, non-smoke artifacts, "
                "committed exact runner blobs, full prediction re-audit"
            ),
            "ordinary_auditor_relaxed": False,
            "original_artifacts_modified": False,
        },
        "protocol": {
            "manifest": _artifact(protocol_manifest),
            "historical_frozen_input_lock_sha256": historical_lock,
            "current_explicit_pair_order_lock_sha256": current_lock,
            "pair_order": DEFAULT_PAIR_ORDER,
        },
        "git_evidence": {
            "repository_root": str(repo_root),
            "curation_commit": curation_commit,
            "runner_source_commit": source_commit,
            "committed_curation_files": committed_evidence,
            "verified_runner_source_sha256": source_hashes,
        },
        "run_generation_ids": {
            method: sorted(values) for method, values in generation_ids.items()
        },
        "original_artifacts": original_artifacts,
        "checks": [
            "committed curation artifacts are byte-identical to Git blobs",
            "historical adapter SHA-256 values match committed runner blobs",
            "protocol manifest and old frozen-input lock match committed provenance",
            "the frozen protocol has no molecular or semantic embedding dependency",
            "pair_order is the sole protocol-input schema omission",
            "all fifteen runs are explicitly non-smoke and predate formal flags",
            "validation/test row IDs, pairs, and labels exactly cover frozen folds",
            "scores are finite and nonconstant",
            "metrics are independently recomputed from saved predictions",
            "old average-precision-labelled AUPRC matches committed fold metrics",
            "MilGNet optimizer/dispersion audit passes",
            "cross-method prediction arrays are not identical",
        ],
        "auprc_definition": AUPRC_DEFINITION,
        "average_precision_definition": AVERAGE_PRECISION_DEFINITION,
    }
    payload: dict[str, object] = {
        **audit_manifest,
        "fold_metrics": rows,
        "summary": summaries,
    }
    return payload, rows, summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--published-provenance", type=Path, required=True)
    parser.add_argument("--published-fold-metrics", type=Path, required=True)
    parser.add_argument("--published-summary", type=Path, required=True)
    parser.add_argument("--curation-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, rows, summaries = reaudit_historical_run(
        protocol_manifest=args.protocol_manifest,
        run_root=args.run_root,
        published_provenance=args.published_provenance,
        published_fold_metrics=args.published_fold_metrics,
        published_summary=args.published_summary,
        curation_commit=args.curation_commit,
        repo_root=args.repo_root,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fold_metrics.csv", rows)
    _write_csv(output_dir / "summary.csv", _summary_rows(summaries))
    write_json(output_dir / "audit_manifest.json", {
        key: value for key, value in payload.items() if key not in {"fold_metrics", "summary"}
    })
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
