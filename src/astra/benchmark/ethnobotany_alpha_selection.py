#!/usr/bin/env python3
"""Select Ethnobotany ASTRA text-fusion alpha without outer-test leakage.

For each outer fold, every candidate is trained and evaluated on that fold's
inner-validation rows only.  The selected alpha is frozen in a decision
manifest before exactly one model for that fold is allowed to score outer-test
rows.  Candidate directories are audited to reject any test prediction file.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .metrics import (
    evaluate_validation_scores,
    load_predictions,
    validate_prediction_coverage,
)
from .protocol import load_protocol_manifest
from astra.official_baselines.common import write_json
from .backend_paths import ETHNOBOTANY_BACKEND_ROOT


SCHEMA = "astra-ethnobotany-alpha-selection/v1"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CUDA_DEVICES = (0, 1)
SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SOURCE_ROOT.parent
PLAN_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "08_model_selection"
    / "ethnobotany_semantic_adapter"
    / "alpha_nested_seed42.json"
)


def _parse_alpha(value: str | float) -> float:
    try:
        alpha = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("alpha must be a number in [0, 1]") from error
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be finite and in [0, 1]")
    return alpha


def _alpha_slug(alpha: float) -> str:
    text = format(_parse_alpha(alpha), ".12g").replace("-", "m").replace(".", "p")
    return f"alpha_{text}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def _validate_locked_plan() -> dict[str, Any]:
    plan = _load_json(PLAN_PATH)
    expected = {
        "schema": "astra-ethnobotany-alpha-plan/v1",
        "dataset": "ethnobotany",
        "seed": 42,
        "semantic_condition": "ethnobotany_qwen_llm_clean_v1/final_bundle",
        "protocol_manifest": (
            "outputs/fair_protocols/"
            "ethnobotany_seed42_cpfix_20261007/manifest.json"
        ),
        "protocol_manifest_sha256": (
            "1c57270f9cbaa5f2ec0e7bb22b7e9471a0bf214b68a0ca9fa12da941ec6a080f"
        ),
        "feature_bundle": (
            "outputs/feature_rebuild/"
            "ethnobotany_qwen_llm_clean_v1/final_bundle"
        ),
        "feature_bundle_manifest_sha256": (
            "be6c35a443c2983b096bae6b3284092e76e81f4311fcc87f817b01010cefe3da"
        ),
        "output_root": (
            "outputs/fair_runs/"
            "ethnobotany_seed42_cpfix_qwen_llm_clean_alpha_nested_v1"
        ),
        "device_policy": "cuda",
        "max_parallel": 2,
        "cuda_devices": list(DEFAULT_CUDA_DEVICES),
        "device_assignment": (
            "outer fold n -> cuda_devices[(n-1) mod 2]; all candidates and "
            "the selected rerun stay on that device"
        ),
        "tensorflow_library_policy": (
            "prepend sys.prefix/lib to LD_LIBRARY_PATH before every "
            "TensorFlow subprocess"
        ),
        "candidate_alphas": list(DEFAULT_ALPHAS),
        "outer_folds": list(range(1, 6)),
        "selection_scope": "independent within each outer fold",
        "selection_metric": "inner-validation trapezoidal AUPRC",
        "selection_tie_break": "lowest alpha",
        "f1_threshold_rule": (
            "maximum inner-validation F1 over 100 thresholds; "
            "lowest-threshold tie break"
        ),
        "training_budget": {
            "epochs": 30,
            "batch_size": 1000,
            "learning_rate": 0.0001,
            "embedding_dimension": 64,
            "reg_u": 0.001,
        },
        "candidate_test_policy": (
            "no outer-test prediction or metric may be materialised"
        ),
        "selected_test_policy": (
            "freeze all five fold decisions first; then evaluate exactly one "
            "selected model per outer fold"
        ),
        "historical_outer_test_context": (
            "The same seed-42 outer folds already have a pre-existing alpha=1 "
            "baseline result. This selector never reads those scores, but these "
            "folds are not a pristine previously unseen holdout."
        ),
    }
    mismatches = {
        key: {"recorded": plan.get(key), "expected": expected.get(key)}
        for key in set(plan) | set(expected)
        if plan.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "locked alpha experiment plan differs: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "path": str(PLAN_PATH.resolve()),
        "bytes": PLAN_PATH.stat().st_size,
        "sha256": _sha256(PLAN_PATH),
        "schema": plan["schema"],
        "seed": plan["seed"],
        "protocol_manifest": plan["protocol_manifest"],
        "protocol_manifest_sha256": plan["protocol_manifest_sha256"],
        "feature_bundle": plan["feature_bundle"],
        "feature_bundle_manifest_sha256": plan[
            "feature_bundle_manifest_sha256"
        ],
        "output_root": plan["output_root"],
        "device_policy": plan["device_policy"],
        "max_parallel": plan["max_parallel"],
        "cuda_devices": plan["cuda_devices"],
        "device_assignment": plan["device_assignment"],
        "tensorflow_library_policy": plan["tensorflow_library_policy"],
    }


@contextmanager
def _exclusive_orchestrator_lock(output_root: Path):
    """Prevent two selectors from training/scoring the same formal run root."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".alpha_selection.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another alpha-selection process holds {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _same_number(left: object, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _choose_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise RuntimeError("cannot select alpha from an empty candidate set")
    return min(
        candidates,
        key=lambda item: (-float(item["validation_auprc"]), float(item["alpha"])),
    )


def _candidate_dir(output_root: Path, alpha: float, fold: int) -> Path:
    return output_root / "selection_candidates" / _alpha_slug(alpha) / f"fold_{fold}"


def _selected_dir(output_root: Path, fold: int) -> Path:
    return output_root / "tasks" / "ASTRA" / f"fold_{fold}"


def _cuda_subprocess_environment(cuda_device: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    library_dir = (Path(sys.prefix) / "lib").resolve()
    if not (library_dir / "libcudart.so.11.0").is_file():
        raise RuntimeError(
            f"TensorFlow CUDA runtime is missing from interpreter prefix: {library_dir}"
        )
    existing = [
        item
        for item in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item and Path(item).resolve() != library_dir
    ]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(library_dir), *existing]
    )
    return environment


def _run_logged(
    command: Sequence[str], *, log_path: Path, cuda_device: int | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    environment = os.environ.copy()
    if cuda_device is not None:
        environment = _cuda_subprocess_environment(cuda_device)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"command failed with exit {result.returncode}; log={log_path}\n"
            + "\n".join(tail)
        )
    print(
        f"completed {log_path.parent.name}/{log_path.name} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


def _runner_command(
    args: argparse.Namespace,
    *,
    fold: int,
    alpha: float,
    output_dir: Path,
    validation_only: bool,
    selection_manifest: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "astra.benchmark.astra_runner",
        "--protocol-manifest",
        str(args.protocol_manifest.resolve()),
        "--dataset",
        "ethnobotany",
        "--data-dir",
        str(args.data_dir.resolve()),
        "--embedding-dir",
        str(args.embedding_dir.resolve()),
        "--output-root",
        str(output_dir.resolve()),
        "--fold",
        str(fold),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--text-fusion-alpha",
        format(alpha, ".17g"),
    ]
    if validation_only:
        command.append("--validation-only")
    elif selection_manifest is not None:
        command.extend(("--alpha-selection-manifest", str(selection_manifest.resolve())))
    return command


def _candidate_complete(run_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "protocol.json",
            run_dir / "results" / "metrics.json",
            run_dir / "results" / "validation_predictions.npz",
        )
    ) and not (run_dir / "results" / "test_predictions.npz").exists()


def _verified_source_hashes(
    provenance: Mapping[str, Any],
    key: str,
    *,
    expected_paths: Mapping[str, Path],
) -> dict[str, str]:
    records = provenance.get(key)
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError(f"candidate provenance has no {key}")
    if set(records) != set(expected_paths):
        raise RuntimeError(f"candidate {key} source set differs")
    hashes: dict[str, str] = {}
    for name, raw_record in records.items():
        if not isinstance(raw_record, Mapping):
            raise RuntimeError(f"candidate {key}/{name} record is invalid")
        path = Path(str(raw_record.get("path", ""))).resolve()
        recorded_sha = raw_record.get("sha256")
        if (
            path != expected_paths[str(name)].resolve()
            or not path.is_file()
            or recorded_sha != _sha256(path)
        ):
            raise RuntimeError(f"candidate {key}/{name} source identity differs")
        hashes[str(name)] = str(recorded_sha)
    return hashes


def _selected_complete(run_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "protocol.json",
            run_dir / "results" / "metrics.json",
            run_dir / "results" / "validation_predictions.npz",
            run_dir / "results" / "test_predictions.npz",
        )
    )


def _audit_candidate(
    *,
    protocol: object,
    run_dir: Path,
    fold: int,
    alpha: float,
) -> dict[str, Any]:
    if not _candidate_complete(run_dir):
        raise RuntimeError(f"incomplete validation-only candidate: {run_dir}")
    forbidden = list(run_dir.rglob("test_predictions.npz"))
    if forbidden:
        raise RuntimeError(
            "alpha candidate materialised forbidden outer-test predictions: "
            + ", ".join(str(path) for path in forbidden)
        )
    metrics_path = run_dir / "results" / "metrics.json"
    provenance_path = run_dir / "protocol.json"
    predictions_path = run_dir / "results" / "validation_predictions.npz"
    metrics = _load_json(metrics_path)
    provenance = _load_json(provenance_path)
    if (
        metrics.get("Dataset") != "ethnobotany"
        or metrics.get("Fold") != fold
        or metrics.get("Seed") != protocol.seed
    ):
        raise RuntimeError(f"candidate dataset/fold/seed identity differs: {run_dir}")
    if metrics.get("Execution_Mode") != "alpha-selection-candidate":
        raise RuntimeError(f"candidate is not labelled validation-only: {run_dir}")
    if metrics.get("Formal_Eligible") is not False:
        raise RuntimeError(f"candidate is incorrectly formal-eligible: {run_dir}")
    if metrics.get("Test_Models_Evaluated") != 0:
        raise RuntimeError(f"candidate reports an outer-test evaluation: {run_dir}")
    if metrics.get("Unselected_Test_Models_Evaluated") != 0:
        raise RuntimeError(f"candidate reports an unselected test model: {run_dir}")
    if (
        metrics.get("Alpha_Selected_On_Validation") is not False
        or metrics.get("Decision_Frozen_Before_Test") is not False
    ):
        raise RuntimeError(f"candidate incorrectly claims a frozen decision: {run_dir}")
    forbidden_metric_keys = {
        "AUROC",
        "AUPRC",
        "AveragePrecision",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Selected_Threshold",
    }
    forbidden_metric_keys.update(
        key
        for key in metrics
        if str(key).startswith("Test_") and key != "Test_Models_Evaluated"
    )
    leaked_metric_keys = sorted(key for key in forbidden_metric_keys if key in metrics)
    if leaked_metric_keys:
        raise RuntimeError(
            f"candidate materialised outer-test metrics {leaked_metric_keys}: {run_dir}"
        )
    if not _same_number(metrics.get("Text_Fusion_Alpha"), alpha):
        raise RuntimeError(f"candidate alpha differs from directory identity: {run_dir}")
    if (
        provenance.get("dataset") != "ethnobotany"
        or provenance.get("fold") != fold
        or provenance.get("seed") != protocol.seed
    ):
        raise RuntimeError(f"candidate provenance identity differs: {run_dir}")
    if (
        provenance.get("validation_only") is not True
        or provenance.get("smoke") is not False
        or provenance.get("diagnostic") is not False
        or provenance.get("execution_mode") != "alpha-selection-candidate"
        or provenance.get("formal_eligible") is not False
        or provenance.get("decision_frozen_before_test") is not False
    ):
        raise RuntimeError(f"candidate provenance is not validation-only: {run_dir}")
    protocol_record = provenance.get("protocol_manifest")
    expected_protocol_path = (
        REPOSITORY_ROOT
        / "outputs/fair_protocols/"
        "ethnobotany_seed42_cpfix_20261007/manifest.json"
    ).resolve()
    recorded_protocol_path = (
        Path(str(protocol_record.get("path", ""))).resolve()
        if isinstance(protocol_record, Mapping)
        else Path()
    )
    if (
        not isinstance(protocol_record, Mapping)
        or recorded_protocol_path != expected_protocol_path
        or protocol_record.get("bytes") != expected_protocol_path.stat().st_size
        or protocol_record.get("sha256") != _sha256(expected_protocol_path)
    ):
        raise RuntimeError(f"candidate protocol-manifest identity differs: {run_dir}")
    if provenance.get("protocol_input_manifest") != protocol.input_manifest():
        raise RuntimeError(f"candidate frozen input identity differs: {run_dir}")
    expected_input_lock = hashlib.sha256(
        json.dumps(
            protocol.input_manifest(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if provenance.get("frozen_input_lock_sha256") != expected_input_lock:
        raise RuntimeError(f"candidate frozen input lock differs: {run_dir}")
    hyperparameters = provenance.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise RuntimeError(f"candidate has no hyperparameter provenance: {run_dir}")
    expected_hyperparameters = {
        "epochs": 30,
        "formal_epochs": 30,
        "batch_size": 1000,
        "learning_rate": 0.0001,
        "embedding_dimension": 64,
        "reg_u": 0.001,
        "text_fusion_alpha": float(alpha),
        "smoke_override": False,
        "diagnostic_mode": False,
    }
    if dict(hyperparameters) != expected_hyperparameters:
        raise RuntimeError(f"candidate hyperparameters differ: {run_dir}")
    fold_rows = protocol.fold_rows(fold)
    expected_steps = 30 * int(math.ceil(len(fold_rows.train_pairs) / 1000))
    if metrics.get("Optimizer_Steps") != expected_steps:
        raise RuntimeError(f"candidate optimizer-step budget differs: {run_dir}")
    split_rows = provenance.get("split_rows")
    split_hashes = provenance.get("split_array_sha256")
    if not isinstance(split_rows, Mapping) or split_rows.get("test_used") != 0:
        raise RuntimeError(f"candidate provenance claims test rows were used: {run_dir}")
    if not isinstance(split_hashes, Mapping) or split_hashes.get("test_pairs_used") is not None:
        raise RuntimeError(f"candidate provenance contains a test-pair hash: {run_dir}")
    adapter_source_sha256 = _verified_source_hashes(
        provenance,
        "adapter_sources",
        expected_paths={
            "astra_runner": SOURCE_ROOT / "astra/benchmark/astra_runner.py",
            "metrics": SOURCE_ROOT / "astra/benchmark/metrics.py",
        },
    )
    model_source_sha256 = _verified_source_hashes(
        provenance,
        "model_sources",
        expected_paths={
            "entrypoint": ETHNOBOTANY_BACKEND_ROOT / "ASTRA.py",
            "model": ETHNOBOTANY_BACKEND_ROOT / "HGHDA.py",
            "herb_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/herbRecommender.py",
            "disease_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/diseaseRecommender.py",
            "base_recommender": ETHNOBOTANY_BACKEND_ROOT
            / "base/recommender.py",
            "rating": ETHNOBOTANY_BACKEND_ROOT / "rating.py",
            "feature_paths": ETHNOBOTANY_BACKEND_ROOT
            / "ethnobotany_feature_paths.py",
            "configuration_loader": ETHNOBOTANY_BACKEND_ROOT / "util/config.py",
            "configuration": ETHNOBOTANY_BACKEND_ROOT / "ethnobotany.conf",
        },
    )
    feature_bundle = provenance.get("feature_bundle")
    if not isinstance(feature_bundle, Mapping):
        raise RuntimeError(f"candidate feature-bundle provenance is missing: {run_dir}")
    feature_bundle_path = Path(str(feature_bundle.get("path", ""))).resolve()
    if (
        not feature_bundle_path.is_file()
        or feature_bundle.get("sha256") != _sha256(feature_bundle_path)
        or feature_bundle.get("formal_eligible") is not True
    ):
        raise RuntimeError(f"candidate feature-bundle identity differs: {run_dir}")
    runtime = provenance.get("runtime")
    expected_cuda_device = str(DEFAULT_CUDA_DEVICES[(fold - 1) % len(DEFAULT_CUDA_DEVICES)])
    expected_library_dir = str((Path(sys.prefix) / "lib").resolve())
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device_policy") != "cuda"
        or str(runtime.get("cuda_visible_devices")) != expected_cuda_device
        or str(runtime.get("ld_library_path", "")).split(os.pathsep)[0]
        != expected_library_dir
        or runtime.get("tensorflow_cuda_available") is not True
        or not runtime.get("tensorflow_gpu_device_name")
    ):
        raise RuntimeError(f"candidate CUDA runtime differs: {run_dir}")
    evaluation = provenance.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("test_predictions") is not None
        or evaluation.get("test_models_evaluated") != 0
        or evaluation.get("unselected_test_models_evaluated") != 0
        or evaluation.get("decision_frozen_before_test") is not False
    ):
        raise RuntimeError(f"candidate provenance references test predictions: {run_dir}")
    table = load_predictions(predictions_path)
    validate_prediction_coverage(
        predictions_path,
        expected_row_id=fold_rows.validation_row_indices,
        expected_pair=fold_rows.validation_pairs,
        expected_label=fold_rows.validation_labels,
    )
    recomputed = evaluate_validation_scores(table.label, table.score)
    for key in (
        "Validation_AUROC",
        "Validation_AUPRC",
        "Validation_AveragePrecision",
        "Validation_Selected_Threshold",
        "Validation_F1",
    ):
        if not math.isclose(
            float(metrics.get(key, math.nan)),
            float(recomputed[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"candidate {run_dir} has mismatched {key}")
    return {
        "alpha": float(alpha),
        "validation_auroc": float(recomputed["Validation_AUROC"]),
        "validation_auprc": float(recomputed["Validation_AUPRC"]),
        "validation_average_precision": float(
            recomputed["Validation_AveragePrecision"]
        ),
        "validation_selected_threshold": float(
            recomputed["Validation_Selected_Threshold"]
        ),
        "validation_f1": float(recomputed["Validation_F1"]),
        "run_dir": str(run_dir.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "protocol_sha256": _sha256(provenance_path),
        "validation_predictions_sha256": _sha256(predictions_path),
        "feature_bundle_sha256": str(feature_bundle["sha256"]),
        "model_source_sha256": model_source_sha256,
        "adapter_source_sha256": adapter_source_sha256,
        "cuda_device": str(runtime["cuda_visible_devices"]),
    }


def _run_parallel(
    tasks: Sequence[tuple[str, Sequence[str], Path, int]],
    workers: int,
    cuda_devices: Sequence[int],
) -> None:
    if not tasks:
        return
    devices = tuple(int(device) for device in cuda_devices)
    if len(devices) != workers or len(set(devices)) != len(devices):
        raise RuntimeError("parallel workers require one distinct CUDA device each")
    task_groups: dict[int, list[tuple[str, Sequence[str], Path, int]]] = {
        device: [] for device in devices
    }
    for task in tasks:
        assigned_device = int(task[3])
        if assigned_device not in task_groups:
            raise RuntimeError(
                f"task {task[0]} targets unlocked CUDA device {assigned_device}"
            )
        task_groups[assigned_device].append(task)
    stop = threading.Event()

    def run_group(
        device: int, group: Sequence[tuple[str, Sequence[str], Path, int]]
    ) -> None:
        for label, command, log_path, assigned_device in group:
            if stop.is_set():
                return
            if assigned_device != device:
                raise RuntimeError(f"task {label} lost its CUDA device affinity")
            try:
                _run_logged(
                    command,
                    log_path=log_path,
                    cuda_device=device,
                )
            except Exception as error:
                stop.set()
                raise RuntimeError(
                    f"task {label} failed on CUDA device {device}: {error}"
                ) from error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_group, device, group): device
            for device, group in task_groups.items()
            if group
        }
        for future in as_completed(futures):
            device = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(
                    f"CUDA device {device} task group failed: {error}"
                ) from error


def _run_candidates(
    args: argparse.Namespace,
    *,
    protocol: object,
    alphas: Sequence[float],
    folds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    tasks: list[tuple[str, Sequence[str], Path, int]] = []
    for fold in folds:
        for alpha in alphas:
            run_dir = _candidate_dir(args.output_root, alpha, fold)
            if _candidate_complete(run_dir):
                if not args.resume:
                    raise RuntimeError(f"candidate already exists: {run_dir}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    f"partial candidate requires manual inspection before resume: {run_dir}"
                )
            command = _runner_command(
                args,
                fold=fold,
                alpha=alpha,
                output_dir=run_dir,
                validation_only=True,
            )
            tasks.append(
                (
                    f"fold={fold},alpha={alpha}",
                    command,
                    run_dir / "run.log",
                    int(args.cuda_devices[(fold - 1) % len(args.cuda_devices)]),
                )
            )
    _run_parallel(tasks, args.max_parallel, args.cuda_devices)

    decisions: dict[str, dict[str, Any]] = {}
    for fold in folds:
        candidates = [
            _audit_candidate(
                protocol=protocol,
                run_dir=_candidate_dir(args.output_root, alpha, fold),
                fold=fold,
                alpha=alpha,
            )
            for alpha in alphas
        ]
        selected = _choose_candidate(candidates)
        decisions[str(fold)] = {
            "candidate_results": candidates,
            "selected_alpha": float(selected["alpha"]),
            "selected_validation_auprc": float(selected["validation_auprc"]),
            "selected_candidate_run_dir": selected["run_dir"],
            "test_models_evaluated_in_workflow_before_decision_freeze": 0,
        }
    return decisions


def _selection_payload(
    args: argparse.Namespace,
    *,
    locked_plan: Mapping[str, Any],
    alphas: Sequence[float],
    folds: Sequence[int],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = args.protocol_manifest.resolve()
    return {
        "schema": SCHEMA,
        "locked_plan": dict(locked_plan),
        "dataset": "ethnobotany",
        "seed": int(args.seed),
        "protocol_manifest": {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": _sha256(manifest),
        },
        "feature_bundle": {
            "path": str(args.embedding_dir.resolve()),
            "manifest_sha256": _sha256(
                args.embedding_dir.resolve() / "bundle_manifest.json"
            ),
        },
        "selector_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "candidate_alphas": [float(alpha) for alpha in alphas],
        "folds_evaluated": [int(fold) for fold in folds],
        "selection_metric": "Validation_AUPRC",
        "selection_metric_definition": (
            "trapezoidal area under the inner-validation precision-recall curve"
        ),
        "tie_break": "highest Validation_AUPRC, then lowest alpha",
        "test_policy": (
            "candidate runs materialise validation predictions only; after this "
            "manifest is frozen, exactly one selected alpha may score each outer fold"
        ),
        "historical_outer_test_context": (
            "A seed-42 alpha=1 baseline predates this workflow. The automated "
            "selector does not read it, so selection is computationally isolated "
            "from outer-test scores, but the folds are not pristine unseen holdouts."
        ),
        "outer_test_pristine": False,
        "folds": dict(decisions),
    }


def validate_selection_decision(
    *,
    selection_path: Path | str,
    protocol_manifest: Path | str,
    output_root: Path | str,
    embedding_dir: Path | str | None = None,
) -> tuple[object, Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Independently validate a complete decision before any outer-test run."""

    selection_path = Path(selection_path).resolve()
    protocol_manifest = Path(protocol_manifest).resolve()
    output_root = Path(output_root).resolve()
    if selection_path != output_root / "alpha_selection.json":
        raise RuntimeError("alpha-selection manifest is outside its canonical run root")
    locked_plan = _validate_locked_plan()
    expected_output_root = (REPOSITORY_ROOT / locked_plan["output_root"]).resolve()
    expected_protocol_path = (
        REPOSITORY_ROOT / locked_plan["protocol_manifest"]
    ).resolve()
    expected_bundle = (REPOSITORY_ROOT / locked_plan["feature_bundle"]).resolve()
    if output_root != expected_output_root:
        raise RuntimeError("alpha-selection output root differs from the locked plan")
    if protocol_manifest != expected_protocol_path:
        raise RuntimeError("protocol path differs from the locked alpha plan")
    protocol = load_protocol_manifest(protocol_manifest)
    if protocol.seed != locked_plan["seed"] or protocol.dataset != "ethnobotany":
        raise RuntimeError("loaded protocol differs from the locked alpha plan")
    payload = _load_json(selection_path)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("unsupported selection manifest schema")
    if payload.get("locked_plan") != locked_plan:
        raise RuntimeError("selection manifest locked-plan identity differs")
    if payload.get("dataset") != "ethnobotany" or payload.get("seed") != protocol.seed:
        raise RuntimeError("selection manifest dataset/seed identity differs")
    if payload.get("outer_test_pristine") is not False:
        raise RuntimeError("selection manifest omits the historical holdout status")
    protocol_record = payload.get("protocol_manifest")
    if not isinstance(protocol_record, Mapping) or (
        Path(str(protocol_record.get("path", ""))).resolve() != protocol_manifest
        or protocol_record.get("bytes") != protocol_manifest.stat().st_size
        or protocol_record.get("sha256") != _sha256(protocol_manifest)
        or protocol_record.get("sha256")
        != locked_plan["protocol_manifest_sha256"]
    ):
        raise RuntimeError("selection manifest protocol identity differs")
    expected_alphas = [float(alpha) for alpha in DEFAULT_ALPHAS]
    expected_folds = list(range(1, 6))
    if payload.get("candidate_alphas") != expected_alphas:
        raise RuntimeError("selection manifest does not contain the locked alpha grid")
    if payload.get("folds_evaluated") != expected_folds:
        raise RuntimeError("selection manifest does not contain all five outer folds")
    if payload.get("selection_metric") != "Validation_AUPRC" or payload.get(
        "tie_break"
    ) != "highest Validation_AUPRC, then lowest alpha":
        raise RuntimeError("selection rule differs from the locked decision rule")
    selector_source = payload.get("selector_source")
    selector_path = Path(__file__).resolve()
    if (
        not isinstance(selector_source, Mapping)
        or Path(str(selector_source.get("path", ""))).resolve() != selector_path
        or selector_source.get("sha256") != _sha256(selector_path)
    ):
        raise RuntimeError("selection manifest selector-source identity differs")
    bundle_record = payload.get("feature_bundle")
    if not isinstance(bundle_record, Mapping):
        raise RuntimeError("selection manifest has no feature-bundle identity")
    recorded_bundle = Path(str(bundle_record.get("path", ""))).resolve()
    requested_bundle = (
        Path(embedding_dir).resolve() if embedding_dir is not None else recorded_bundle
    )
    bundle_manifest = expected_bundle / "bundle_manifest.json"
    if (
        recorded_bundle != expected_bundle
        or requested_bundle != expected_bundle
        or not bundle_manifest.is_file()
        or bundle_record.get("manifest_sha256") != _sha256(bundle_manifest)
        or bundle_record.get("manifest_sha256")
        != locked_plan["feature_bundle_manifest_sha256"]
    ):
        raise RuntimeError("selection manifest feature-bundle identity differs")
    forbidden = list((output_root / "selection_candidates").rglob("test_predictions.npz"))
    if forbidden:
        raise RuntimeError(
            "unselected candidate test artifacts exist: "
            + ", ".join(str(path) for path in forbidden)
        )
    allowed_selected_tests = {
        (_selected_dir(output_root, fold) / "results" / "test_predictions.npz").resolve()
        for fold in range(1, 6)
    }
    observed_tests = {
        path.resolve() for path in output_root.rglob("test_predictions.npz")
    }
    unexpected_tests = sorted(observed_tests - allowed_selected_tests)
    if unexpected_tests:
        raise RuntimeError(
            "non-canonical outer-test artifacts exist: "
            + ", ".join(str(path) for path in unexpected_tests)
        )
    decisions = payload.get("folds")
    expected_fold_keys = {str(fold) for fold in expected_folds}
    if not isinstance(decisions, Mapping) or set(decisions) != expected_fold_keys:
        raise RuntimeError("selection manifest fold decisions are incomplete")
    for fold in expected_folds:
        record = decisions[str(fold)]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"selection manifest fold {fold} is invalid")
        candidate_rows = record.get("candidate_results")
        if not isinstance(candidate_rows, list) or len(candidate_rows) != len(
            expected_alphas
        ):
            raise RuntimeError(f"selection manifest fold {fold} grid is incomplete")
        recorded_alphas = [float(item.get("alpha", math.nan)) for item in candidate_rows]
        if recorded_alphas != expected_alphas or len(set(recorded_alphas)) != len(
            expected_alphas
        ):
            raise RuntimeError(f"selection manifest fold {fold} alpha grid differs")
        recomputed: list[dict[str, Any]] = []
        for alpha, stored in zip(expected_alphas, candidate_rows):
            if not isinstance(stored, Mapping):
                raise RuntimeError(f"selection manifest fold {fold} candidate is invalid")
            expected_dir = _candidate_dir(output_root, alpha, fold).resolve()
            if Path(str(stored.get("run_dir", ""))).resolve() != expected_dir:
                raise RuntimeError(
                    f"selection manifest fold {fold} candidate path differs"
                )
            audited = _audit_candidate(
                protocol=protocol,
                run_dir=expected_dir,
                fold=fold,
                alpha=alpha,
            )
            if audited["feature_bundle_sha256"] != bundle_record.get(
                "manifest_sha256"
            ):
                raise RuntimeError(
                    f"selection manifest fold {fold} candidate bundle differs"
                )
            if dict(stored) != audited:
                raise RuntimeError(
                    f"selection manifest fold {fold} candidate record/hash differs"
                )
            recomputed.append(audited)
        selected = _choose_candidate(recomputed)
        expected_selected = {
            "selected_alpha": float(selected["alpha"]),
            "selected_validation_auprc": float(selected["validation_auprc"]),
            "selected_candidate_run_dir": selected["run_dir"],
            "test_models_evaluated_in_workflow_before_decision_freeze": 0,
        }
        for key, expected_value in expected_selected.items():
            if record.get(key) != expected_value:
                raise RuntimeError(
                    f"selection manifest fold {fold} has a wrong {key}"
                )
    return protocol, payload, decisions  # type: ignore[return-value]


def _write_or_validate_selection(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        existing = _load_json(path)
        if not resume:
            raise RuntimeError(f"selection manifest already exists: {path}")
        if dict(existing) != dict(payload):
            raise RuntimeError(
                "existing selection manifest differs from recomputed validation decisions"
            )
        return
    write_json(path, payload)


def _run_selected(
    args: argparse.Namespace,
    *,
    selection_path: Path,
    decisions: Mapping[str, Mapping[str, Any]],
    folds: Sequence[int],
) -> None:
    tasks: list[tuple[str, Sequence[str], Path, int]] = []
    for fold in folds:
        alpha = float(decisions[str(fold)]["selected_alpha"])
        run_dir = _selected_dir(args.output_root, fold)
        if _selected_complete(run_dir):
            if not args.resume:
                raise RuntimeError(f"selected run already exists: {run_dir}")
            validate_selected_binding(
                run_dir=run_dir,
                fold=fold,
                selected_alpha=alpha,
                selection_path=selection_path,
            )
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"partial selected run requires manual inspection before resume: {run_dir}"
            )
        command = _runner_command(
            args,
            fold=fold,
            alpha=alpha,
            output_dir=run_dir,
            validation_only=False,
            selection_manifest=selection_path,
        )
        tasks.append(
            (
                f"selected-fold={fold},alpha={alpha}",
                command,
                run_dir / "run.log",
                int(args.cuda_devices[(fold - 1) % len(args.cuda_devices)]),
            )
        )
    _run_parallel(tasks, args.max_parallel, args.cuda_devices)


def validate_selected_binding(
    *,
    run_dir: Path,
    fold: int,
    selected_alpha: float,
    selection_path: Path,
) -> None:
    selection_path = selection_path.resolve()
    run_dir = run_dir.resolve()
    expected_run_dir = _selected_dir(selection_path.parent, fold).resolve()
    if run_dir != expected_run_dir:
        raise RuntimeError(f"selected run path is non-canonical for fold {fold}")
    if not _selected_complete(run_dir):
        raise RuntimeError(f"selected outer-test run is incomplete: {run_dir}")
    metrics = _load_json(run_dir / "results" / "metrics.json")
    provenance = _load_json(run_dir / "protocol.json")
    selection = _load_json(selection_path)
    decisions = selection.get("folds")
    decision = decisions.get(str(fold)) if isinstance(decisions, Mapping) else None
    if not isinstance(decision, Mapping):
        raise RuntimeError(f"selection manifest lacks fold {fold} decision")
    if not _same_number(decision.get("selected_alpha"), selected_alpha):
        raise RuntimeError(f"decision alpha differs for fold {fold}")
    candidate_run_dir = Path(
        str(decision.get("selected_candidate_run_dir", ""))
    ).resolve()
    expected_candidate_dir = _candidate_dir(
        selection_path.parent, selected_alpha, fold
    ).resolve()
    if candidate_run_dir != expected_candidate_dir:
        raise RuntimeError(f"winning candidate path differs for fold {fold}")
    candidate_provenance_path = candidate_run_dir / "protocol.json"
    if not candidate_provenance_path.is_file():
        raise RuntimeError(f"winning candidate provenance is missing for fold {fold}")
    candidate_provenance = _load_json(candidate_provenance_path)
    if (
        metrics.get("Dataset") != "ethnobotany"
        or metrics.get("Fold") != fold
        or metrics.get("Seed") != selection.get("seed")
        or metrics.get("Execution_Mode") != "formal-candidate"
        or metrics.get("Formal_Eligible") is not True
        or metrics.get("Smoke") is not False
        or metrics.get("Diagnostic") is not False
        or metrics.get("Validation_Only") is not False
    ):
        raise RuntimeError(f"selected run identity/mode differs for fold {fold}")
    if (
        provenance.get("dataset") != "ethnobotany"
        or provenance.get("fold") != fold
        or provenance.get("seed") != selection.get("seed")
        or provenance.get("execution_mode") != "formal-candidate"
        or provenance.get("formal_eligible") is not True
        or provenance.get("validation_only") is not False
        or provenance.get("decision_frozen_before_test") is not True
    ):
        raise RuntimeError(f"selected provenance identity/mode differs for fold {fold}")
    runtime = provenance.get("runtime")
    expected_library_dir = str((Path(sys.prefix) / "lib").resolve())
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device_policy") != "cuda"
        or str(runtime.get("cuda_visible_devices")) not in {"0", "1"}
        or str(runtime.get("ld_library_path", "")).split(os.pathsep)[0]
        != expected_library_dir
        or runtime.get("tensorflow_cuda_available") is not True
        or not runtime.get("tensorflow_gpu_device_name")
    ):
        raise RuntimeError(f"selected CUDA runtime differs for fold {fold}")
    candidate_runtime = candidate_provenance.get("runtime")
    if (
        not isinstance(candidate_runtime, Mapping)
        or runtime.get("cuda_visible_devices")
        != candidate_runtime.get("cuda_visible_devices")
    ):
        raise RuntimeError(
            f"selected run changed CUDA device from its candidate for fold {fold}"
        )
    if not _same_number(metrics.get("Text_Fusion_Alpha"), selected_alpha):
        raise RuntimeError(f"selected run alpha differs for fold {fold}")
    if metrics.get("Alpha_Selected_On_Validation") is not True:
        raise RuntimeError(f"selected run lacks validation-selection status for fold {fold}")
    if metrics.get("Decision_Frozen_Before_Test") is not True:
        raise RuntimeError(f"selected run decision was not frozen for fold {fold}")
    if metrics.get("Test_Models_Evaluated") != 1 or metrics.get(
        "Unselected_Test_Models_Evaluated"
    ) != 0:
        raise RuntimeError(f"selected run test-model count differs for fold {fold}")
    selection_ref = provenance.get("alpha_selection")
    if (
        not isinstance(selection_ref, Mapping)
        or selection_ref.get("path") != str(selection_path.resolve())
        or selection_ref.get("sha256") != _sha256(selection_path)
        or selection_ref.get("bytes") != selection_path.stat().st_size
        or not _same_number(selection_ref.get("selected_alpha"), selected_alpha)
    ):
        raise RuntimeError(f"selected run is not bound to the decision for fold {fold}")
    evaluation = provenance.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("decision_frozen_before_test") is not True
        or evaluation.get("test_models_evaluated") != 1
        or evaluation.get("unselected_test_models_evaluated") != 0
        or evaluation.get("test_predictions")
        != str((run_dir / "results" / "test_predictions.npz").resolve())
    ):
        raise RuntimeError(f"selected evaluation provenance differs for fold {fold}")
    for key in (
        "protocol_manifest",
        "protocol_input_manifest",
        "frozen_input_lock_sha256",
        "source_inputs_verified",
        "feature_bundle",
        "model_sources",
        "adapter_sources",
        "hyperparameters",
    ):
        if provenance.get(key) != candidate_provenance.get(key):
            raise RuntimeError(
                f"selected run {key} differs from winning candidate for fold {fold}"
            )
    selected_rows = provenance.get("split_rows")
    candidate_rows = candidate_provenance.get("split_rows")
    if not isinstance(selected_rows, Mapping) or not isinstance(
        candidate_rows, Mapping
    ):
        raise RuntimeError(f"selected/candidate split rows are missing for fold {fold}")
    for key in (
        "train_full",
        "train_used",
        "train_positive_graph_edges_full",
        "validation_full",
        "validation_used",
        "test_full",
    ):
        if selected_rows.get(key) != candidate_rows.get(key):
            raise RuntimeError(
                f"selected run split {key} differs from candidate for fold {fold}"
            )
    if (
        candidate_rows.get("test_used") != 0
        or selected_rows.get("test_used") != selected_rows.get("test_full")
    ):
        raise RuntimeError(f"selected/candidate test-row use differs for fold {fold}")
    selected_hashes = provenance.get("split_array_sha256")
    candidate_hashes = candidate_provenance.get("split_array_sha256")
    if not isinstance(selected_hashes, Mapping) or not isinstance(
        candidate_hashes, Mapping
    ):
        raise RuntimeError(f"selected/candidate split hashes are missing for fold {fold}")
    for key in (
        "train_pairs_used",
        "train_labels_used",
        "train_positive_pairs_full",
        "validation_pairs_used",
    ):
        if selected_hashes.get(key) != candidate_hashes.get(key):
            raise RuntimeError(
                f"selected run split hash {key} differs for fold {fold}"
            )
    if candidate_hashes.get("test_pairs_used") is not None or not isinstance(
        selected_hashes.get("test_pairs_used"), str
    ):
        raise RuntimeError(f"selected/candidate test hash differs for fold {fold}")
    if provenance.get("optimizer_steps") != candidate_provenance.get(
        "optimizer_steps"
    ) or metrics.get("Optimizer_Steps") != provenance.get("optimizer_steps"):
        raise RuntimeError(f"selected optimizer budget differs for fold {fold}")


def _audit_selection_and_selected(
    args: argparse.Namespace,
    *,
    selection_path: Path,
    folds: Sequence[int],
) -> Mapping[str, Any]:
    _, _, decisions = validate_selection_decision(
        selection_path=selection_path,
        protocol_manifest=args.protocol_manifest,
        output_root=args.output_root,
        embedding_dir=args.embedding_dir,
    )
    selected_alpha_by_fold: dict[str, float] = {}
    for fold in folds:
        record = decisions[str(fold)]
        selected_alpha = float(record["selected_alpha"])
        selected_run = _selected_dir(args.output_root, fold)
        validate_selected_binding(
            run_dir=selected_run,
            fold=fold,
            selected_alpha=selected_alpha,
            selection_path=selection_path,
        )
        selected_alpha_by_fold[str(fold)] = selected_alpha

    expected_test_artifacts = {
        (_selected_dir(args.output_root, fold) / "results" / "test_predictions.npz").resolve()
        for fold in folds
    }
    observed_test_artifacts = {
        path.resolve() for path in args.output_root.rglob("test_predictions.npz")
    }
    if observed_test_artifacts != expected_test_artifacts:
        raise RuntimeError(
            "outer-test artifact set is not exactly the five selected fold files: "
            + json.dumps(
                {
                    "missing": sorted(
                        str(path)
                        for path in expected_test_artifacts - observed_test_artifacts
                    ),
                    "unexpected": sorted(
                        str(path)
                        for path in observed_test_artifacts - expected_test_artifacts
                    ),
                },
                sort_keys=True,
            )
        )

    audit_dir = args.output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    alpha_audit = {
        "status": "alpha-selection-audit-passed",
        "formal_candidate": True,
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": _sha256(selection_path),
        "selected_alpha_by_fold": selected_alpha_by_fold,
        "candidate_test_models_evaluated": 0,
        "selected_test_models_evaluated": len(folds),
        "decision_frozen_before_selected_test_runs": True,
        "outer_test_pristine": False,
        "historical_outer_test_context": (
            "A separate alpha=1 baseline existed before this workflow; the "
            "selector did not read it, so this is not a pristine unseen-holdout claim."
        ),
    }
    write_json(audit_dir / "alpha_selection_audit.json", alpha_audit)
    return alpha_audit


def _run_metric_audit(args: argparse.Namespace, *, folds: Sequence[int]) -> None:
    command = [
        sys.executable,
        "-m",
        "astra.official_baselines.audit_results",
        "--protocol-manifest",
        str(args.protocol_manifest.resolve()),
        "--run-root",
        str(args.output_root.resolve()),
        "--output-dir",
        str((args.output_root / "audit").resolve()),
        "--methods",
        "ASTRA",
        "--folds",
        *(str(fold) for fold in folds),
    ]
    _run_logged(command, log_path=args.output_root / "audit" / "metric_audit.log")


def _assert_cuda_devices(cuda_devices: Sequence[int]) -> None:
    """Fail before training if the two locked physical GPUs are unavailable."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot execute nvidia-smi: {error}") from error
    if query.returncode:
        raise RuntimeError(f"nvidia-smi preflight failed: {query.stderr.strip()}")
    available = {
        int(line.strip())
        for line in query.stdout.splitlines()
        if line.strip().lstrip("-").isdigit()
    }
    requested = {int(device) for device in cuda_devices}
    if not requested or not requested.issubset(available):
        raise RuntimeError(
            f"locked CUDA devices {sorted(requested)} are not all available; "
            f"observed {sorted(available)}"
        )
    probe = (
        "import sys; import tensorflow.compat.v1 as tf; "
        "sys.exit(0 if tf.test.is_gpu_available(cuda_only=True) else 3)"
    )
    for device in sorted(requested):
        environment = _cuda_subprocess_environment(device)
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        checked = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if checked.returncode:
            details = (checked.stderr or checked.stdout).strip().splitlines()[-5:]
            raise RuntimeError(
                f"TensorFlow cannot use locked CUDA device {device}: "
                + " | ".join(details)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=_parse_alpha,
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--cuda-devices", nargs="+", type=int, default=[0, 1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--selection-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _execute_workflow(
    args: argparse.Namespace,
    *,
    protocol: object,
    alphas: Sequence[float],
    folds: Sequence[int],
    locked_plan: Mapping[str, Any],
) -> int:
    if _validate_locked_plan() != locked_plan:
        raise RuntimeError("locked alpha plan changed before workflow execution")
    selection_path = args.output_root / "alpha_selection.json"

    if args.audit_only:
        _audit_selection_and_selected(
            args,
            selection_path=selection_path,
            folds=folds,
        )
        _run_metric_audit(args, folds=folds)
        return 0

    if args.evaluate_only:
        _, _, decisions = validate_selection_decision(
            selection_path=selection_path,
            protocol_manifest=args.protocol_manifest,
            output_root=args.output_root,
            embedding_dir=args.embedding_dir,
        )
    else:
        decisions = _run_candidates(
            args,
            protocol=protocol,
            alphas=alphas,
            folds=folds,
        )
        if _validate_locked_plan() != locked_plan:
            raise RuntimeError("locked alpha plan changed while candidates ran")
        selection = _selection_payload(
            args,
            locked_plan=locked_plan,
            alphas=alphas,
            folds=folds,
            decisions=decisions,
        )
        _write_or_validate_selection(selection_path, selection, resume=args.resume)

    # This independent recomputation is intentionally before any selected
    # outer-test command.  A stale, incomplete, or edited decision cannot be
    # used to probe the held-out folds.
    _, _, decisions = validate_selection_decision(
        selection_path=selection_path,
        protocol_manifest=args.protocol_manifest,
        output_root=args.output_root,
        embedding_dir=args.embedding_dir,
    )

    if args.selection_only:
        print(json.dumps(_load_json(selection_path), indent=2, ensure_ascii=False))
        return 0

    _run_selected(
        args,
        selection_path=selection_path,
        decisions=decisions,
        folds=folds,
    )
    alpha_audit = _audit_selection_and_selected(
        args,
        selection_path=selection_path,
        folds=folds,
    )
    _run_metric_audit(args, folds=folds)
    print(json.dumps(alpha_audit, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise RuntimeError("seed must be non-negative")
    if args.max_parallel < 1:
        raise RuntimeError("--max-parallel must be positive")
    if any(device < 0 for device in args.cuda_devices):
        raise RuntimeError("--cuda-devices must contain non-negative GPU indices")
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.data_dir = args.data_dir.resolve()
    args.embedding_dir = args.embedding_dir.resolve()
    args.output_root = args.output_root.resolve()
    alphas = tuple(sorted(set(float(alpha) for alpha in args.alphas)))
    folds = tuple(dict.fromkeys(int(fold) for fold in args.folds))
    locked_plan = _validate_locked_plan()
    expected_protocol = (
        REPOSITORY_ROOT / locked_plan["protocol_manifest"]
    ).resolve()
    expected_bundle = (REPOSITORY_ROOT / locked_plan["feature_bundle"]).resolve()
    expected_output = (REPOSITORY_ROOT / locked_plan["output_root"]).resolve()
    if args.seed != locked_plan["seed"]:
        raise RuntimeError("seed differs from the locked alpha plan")
    if args.protocol_manifest != expected_protocol:
        raise RuntimeError("protocol path differs from the locked alpha plan")
    if (
        not args.protocol_manifest.is_file()
        or _sha256(args.protocol_manifest)
        != locked_plan["protocol_manifest_sha256"]
    ):
        raise RuntimeError("protocol hash differs from the locked alpha plan")
    if args.embedding_dir != expected_bundle:
        raise RuntimeError("feature-bundle path differs from the locked alpha plan")
    bundle_manifest = args.embedding_dir / "bundle_manifest.json"
    if (
        not bundle_manifest.is_file()
        or _sha256(bundle_manifest)
        != locked_plan["feature_bundle_manifest_sha256"]
    ):
        raise RuntimeError("feature-bundle hash differs from the locked alpha plan")
    if args.output_root != expected_output:
        raise RuntimeError("output root differs from the locked alpha plan")
    if args.device != locked_plan["device_policy"]:
        raise RuntimeError("device policy differs from the locked alpha plan")
    if args.max_parallel != locked_plan["max_parallel"]:
        raise RuntimeError("parallelism differs from the locked alpha plan")
    if list(args.cuda_devices) != locked_plan["cuda_devices"]:
        raise RuntimeError("CUDA device assignment differs from the locked alpha plan")
    if alphas != DEFAULT_ALPHAS:
        raise RuntimeError(
            f"formal alpha selection requires the locked grid {DEFAULT_ALPHAS}"
        )
    if folds != tuple(range(1, 6)):
        raise RuntimeError("formal alpha selection requires folds 1 2 3 4 5")
    protocol = load_protocol_manifest(args.protocol_manifest)
    if protocol.dataset != "ethnobotany" or protocol.seed != args.seed:
        raise RuntimeError("protocol must be the requested seed's Ethnobotany manifest")
    if args.data_dir != protocol.data_dir.resolve():
        raise RuntimeError("data directory differs from the locked protocol")
    if not args.audit_only:
        _assert_cuda_devices(args.cuda_devices)
    with _exclusive_orchestrator_lock(args.output_root):
        return _execute_workflow(
            args,
            protocol=protocol,
            alphas=alphas,
            folds=folds,
            locked_plan=locked_plan,
        )


if __name__ == "__main__":
    raise SystemExit(main())
