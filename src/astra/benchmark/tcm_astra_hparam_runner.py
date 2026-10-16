#!/usr/bin/env python3
"""Validation-only TCM ASTRA hyperparameter candidate runner.

The central ASTRA runner intentionally fixes the historical defaults.  This
small wrapper changes only the pre-registered epoch and learning-rate values
for TCM-suite candidates, while delegating all split, feature, training, and
validation-only safeguards to :mod:`astra.benchmark.astra_runner`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

from . import astra_runner


class TCMHparamRunnerError(RuntimeError):
    """Raised when a candidate differs from its frozen plan."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMHparamRunnerError(f"cannot read candidate plan {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TCMHparamRunnerError("candidate plan must be a JSON object")
    return payload


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epochs must be an integer") from error
    if parsed <= 0 or parsed > 512:
        raise argparse.ArgumentTypeError("epochs must be in 1..512")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("learning rate must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("learning rate must be in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = astra_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--epochs", type=_positive_int, required=True)
    parser.add_argument("--learning-rate", type=_positive_float, required=True)
    return parser


def validate_candidate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = astra_runner._normalise_dataset(args.dataset)
    if dataset != "tcm-suite":
        raise TCMHparamRunnerError("this wrapper is restricted to TCM-suite")
    if not bool(args.validation_only):
        raise TCMHparamRunnerError("hyperparameter candidates require --validation-only")
    if bool(args.smoke) or bool(args.diagnostic):
        raise TCMHparamRunnerError("candidate runs cannot use smoke or diagnostic mode")
    if not math.isclose(float(args.text_fusion_alpha), 1.0, abs_tol=1e-15):
        raise TCMHparamRunnerError("stage-1 tuning fixes text_fusion_alpha=1")

    plan_path = args.candidate_plan.resolve()
    plan = _load_json(plan_path)
    if plan.get("schema") != "astra-tcm-hparam-plan/v1":
        raise TCMHparamRunnerError("unsupported candidate-plan schema")
    if plan.get("dataset") != "tcm-suite" or int(plan.get("seed", -1)) != args.seed:
        raise TCMHparamRunnerError("candidate plan dataset/seed mismatch")

    protocol_path = Path(str(plan.get("protocol_manifest", ""))).resolve()
    feature_path = Path(str(plan.get("feature_bundle", ""))).resolve()
    if protocol_path != args.protocol_manifest.resolve():
        raise TCMHparamRunnerError("protocol manifest differs from the frozen plan")
    if feature_path != args.embedding_dir.resolve():
        raise TCMHparamRunnerError("feature bundle differs from the frozen plan")
    expected_protocol_sha = str(plan.get("protocol_manifest_sha256", ""))
    expected_bundle_sha = str(plan.get("feature_bundle_manifest_sha256", ""))
    if _sha256(protocol_path) != expected_protocol_sha:
        raise TCMHparamRunnerError("protocol manifest hash differs from the frozen plan")
    bundle_manifest = feature_path / "bundle_manifest.json"
    if _sha256(bundle_manifest) != expected_bundle_sha:
        raise TCMHparamRunnerError("feature-bundle manifest hash differs from the frozen plan")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise TCMHparamRunnerError("candidate plan has no candidate list")
    matching = [item for item in candidates if isinstance(item, Mapping) and item.get("id") == args.candidate_id]
    if len(matching) != 1:
        raise TCMHparamRunnerError(f"candidate id {args.candidate_id!r} is not unique in the plan")
    candidate = dict(matching[0])
    if int(candidate.get("epochs", -1)) != args.epochs:
        raise TCMHparamRunnerError("requested epochs differ from the frozen candidate")
    try:
        planned_lr = float(candidate["learning_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise TCMHparamRunnerError("candidate learning rate is invalid") from error
    if not math.isclose(planned_lr, args.learning_rate, rel_tol=0.0, abs_tol=1e-15):
        raise TCMHparamRunnerError("requested learning rate differs from the frozen candidate")
    if candidate.get("semantic_adapter") != "identity":
        raise TCMHparamRunnerError("stage-1 candidates must use the identity adapter")
    return {
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "candidate": candidate,
    }


@contextmanager
def _candidate_hyperparameters(*, epochs: int, learning_rate: float) -> Iterator[None]:
    original = astra_runner._HYPERPARAMETERS["tcm-suite"]
    replacement = dict(original)
    replacement["epochs"] = int(epochs)
    replacement["learning_rate"] = float(learning_rate)
    astra_runner._HYPERPARAMETERS["tcm-suite"] = replacement
    try:
        yield
    finally:
        astra_runner._HYPERPARAMETERS["tcm-suite"] = original


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    binding = validate_candidate(args)
    with _candidate_hyperparameters(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    ):
        metrics = astra_runner.run(args)

    output_root = args.output_root.resolve()
    metrics_path = output_root / "results" / "metrics.json"
    protocol_path = output_root / "protocol.json"
    stored_metrics = dict(_load_json(metrics_path))
    stored_protocol = dict(_load_json(protocol_path))
    source_path = Path(__file__).resolve()
    provenance = {
        **binding,
        "runner_path": str(source_path),
        "runner_sha256": _sha256(source_path),
        "selection_metric": "Validation_AUPRC",
        "outer_test_materialised": False,
    }
    stored_metrics.update(
        {
            "Candidate_ID": args.candidate_id,
            "Candidate_Plan_SHA256": binding["plan_sha256"],
            "Tuning_Stage": "epoch-learning-rate",
        }
    )
    stored_protocol["tcm_hparam_candidate"] = provenance
    _write_json(metrics_path, stored_metrics)
    _write_json(protocol_path, stored_protocol)
    return stored_metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run(args)
    except Exception as error:
        print(f"TCM ASTRA candidate failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
