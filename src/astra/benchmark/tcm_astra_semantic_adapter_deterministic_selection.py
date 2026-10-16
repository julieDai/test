#!/usr/bin/env python3
"""Freeze the deterministic TCM adapter decision after all pair receipts exist.

The frozen v2 selector remains the metric and tie-break implementation.  This
entry point adds the v3 plan/runner contracts and independently revalidates all
five same-initialization receipts before writing a versioned selection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from . import tcm_astra_semantic_adapter_selection as base_selection
from .tcm_astra_semantic_adapter_deterministic_plan import PLAN_SCHEMA
from .tcm_astra_semantic_adapter_deterministic_runner import RUNNER_SCHEMA
from .tcm_astra_semantic_adapter_pair_guard import (
    build_pair_receipt,
    sha256,
)


SELECTION_SCHEMA = "astra-tcm-semantic-adapter-deterministic-selection/v2"


class TCMDeterministicAdapterSelectionError(RuntimeError):
    """Raised when the deterministic selection chain is incomplete."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TCMDeterministicAdapterSelectionError(
            f"cannot read JSON {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TCMDeterministicAdapterSelectionError(
            f"JSON root must be an object: {path}"
        )
    return payload


@contextmanager
def _deterministic_contracts() -> Iterator[None]:
    old_plan = base_selection.PLAN_SCHEMA
    old_runner = base_selection.RUNNER_SCHEMA
    old_selection = base_selection.SELECTION_SCHEMA
    base_selection.PLAN_SCHEMA = PLAN_SCHEMA
    base_selection.RUNNER_SCHEMA = RUNNER_SCHEMA
    base_selection.SELECTION_SCHEMA = SELECTION_SCHEMA
    try:
        yield
    finally:
        base_selection.SELECTION_SCHEMA = old_selection
        base_selection.RUNNER_SCHEMA = old_runner
        base_selection.PLAN_SCHEMA = old_plan


def _validate_pair_receipts(
    *, plan_path: Path, candidate_root: Path, receipt_root: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold in range(1, 6):
        path = (receipt_root / f"fold_{fold}.json").resolve()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise TCMDeterministicAdapterSelectionError(
                f"fold {fold} pair receipt is absent"
            )
        expected = build_pair_receipt(
            plan_path=plan_path,
            candidate_root=candidate_root,
            fold=fold,
        )
        observed = _load_json(path)
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        digest = sha256(path)
        if dict(observed) != expected or fields != [digest, path.name]:
            raise TCMDeterministicAdapterSelectionError(
                f"fold {fold} pair receipt differs from candidate artifacts"
            )
        records.append(
            {"fold": fold, "path": str(path), "sha256": digest}
        )
    return records


def select_candidates(
    *,
    plan_path: Path,
    candidate_root: Path,
    pair_receipt_root: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise TCMDeterministicAdapterSelectionError(
            f"refusing to overwrite selection or sidecar: {output}"
        )
    receipts = _validate_pair_receipts(
        plan_path=plan_path,
        candidate_root=candidate_root,
        receipt_root=pair_receipt_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="astra_tcm_deterministic_selection_", dir=output.parent
    ) as temporary:
        temporary_output = Path(temporary) / "selection.json"
        with _deterministic_contracts():
            try:
                base_selection.select_candidates(
                    plan_path=plan_path,
                    candidate_root=candidate_root,
                    output=temporary_output,
                )
            except Exception as error:
                raise TCMDeterministicAdapterSelectionError(str(error)) from error
        payload = dict(_load_json(temporary_output))
    selector = Path(__file__).resolve()
    plan = _load_json(plan_path.resolve(strict=True))
    payload.update(
        {
            "schema": SELECTION_SCHEMA,
            "selector": {"path": str(selector), "sha256": sha256(selector)},
            "pair_receipts": receipts,
            "deterministic_initialization": plan.get(
                "deterministic_initialization"
            ),
            "outer_test_accessed": False,
            "outer_test_evaluator_in_this_module": False,
        }
    )
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {**payload, "selection_path": str(output), "selection_sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--pair-receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = select_candidates(
            plan_path=args.plan,
            candidate_root=args.candidate_root,
            pair_receipt_root=args.pair_receipt_root,
            output=args.output,
        )
    except Exception as error:
        print(f"TCM deterministic adapter selection failed: {error}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
