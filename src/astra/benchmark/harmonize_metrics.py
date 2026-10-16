#!/usr/bin/env python3
"""Recompute manuscript ranking metrics from saved fold predictions.

This utility is deliberately model-independent.  It accepts one path template
per method, auto-detects the two prediction schemas retained in this project,
and writes a hash-backed fold table plus aggregate table.  It never selects a
checkpoint or changes model predictions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from .metrics import AUPRC_DEFINITION, AVERAGE_PRECISION_DEFINITION


LABEL_FIELDS = ("label", "labels")
SCORE_FIELDS = ("score", "scores", "predictions")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(payload: np.lib.npyio.NpzFile, candidates: Sequence[str], kind: str) -> str:
    matches = [name for name in candidates if name in payload.files]
    if len(matches) != 1:
        raise ValueError(
            f"prediction file must contain exactly one supported {kind} field; "
            f"found {matches}, available={payload.files}"
        )
    return matches[0]


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    with np.load(path, allow_pickle=False) as payload:
        label_field = _field(payload, LABEL_FIELDS, "label")
        score_field = _field(payload, SCORE_FIELDS, "score")
        labels = np.asarray(payload[label_field]).reshape(-1)
        scores = np.asarray(payload[score_field], dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError(f"invalid label/score lengths in {path}")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError(f"non-binary labels in {path}")
    if np.unique(labels).size != 2:
        raise ValueError(f"both classes are required in {path}")
    if not np.isfinite(scores).all() or np.unique(scores).size < 2:
        raise ValueError(f"scores are non-finite or constant in {path}")
    return labels.astype(np.int64, copy=False), scores, label_field, score_field


def _resolve(template: str, fold: int) -> Path:
    pattern = template.format(fold=fold)
    matches = [Path(value).resolve() for value in glob.glob(pattern)]
    if len(matches) != 1:
        raise ValueError(
            f"fold {fold} template must resolve to exactly one file: "
            f"pattern={pattern!r}, matches={len(matches)}"
        )
    return matches[0]


def _method_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must use NAME=PATH_TEMPLATE")
    name, template = value.split("=", 1)
    if not name.strip() or "{fold}" not in template:
        raise argparse.ArgumentTypeError(
            "method must have a non-empty name and PATH_TEMPLATE containing {fold}"
        )
    return name.strip(), template


def recompute(
    *, dataset: str, methods: Sequence[tuple[str, str]], folds: Sequence[int]
) -> dict[str, object]:
    fold_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for method, template in methods:
        method_rows: list[dict[str, object]] = []
        for fold in folds:
            path = _resolve(template, fold)
            labels, scores, label_field, score_field = _load_prediction(path)
            precision, recall, _ = precision_recall_curve(labels, scores)
            row = {
                "Dataset": dataset,
                "Method": method,
                "Fold": int(fold),
                "Rows": int(len(labels)),
                "Positive_Rows": int(labels.sum()),
                "AUROC": float(roc_auc_score(labels, scores)),
                "AUPRC": float(auc(recall, precision)),
                "AveragePrecision": float(average_precision_score(labels, scores)),
                "Prediction_Path": str(path),
                "Prediction_SHA256": _sha256(path),
                "Label_Field": label_field,
                "Score_Field": score_field,
            }
            method_rows.append(row)
            fold_rows.append(row)
        method_summary: dict[str, object] = {}
        for metric in ("AUROC", "AUPRC", "AveragePrecision"):
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            method_summary[metric] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "fold_values": values.tolist(),
            }
        summary[method] = method_summary
    return {
        "dataset": dataset,
        "folds": [int(fold) for fold in folds],
        "auprc_definition": AUPRC_DEFINITION,
        "average_precision_definition": AVERAGE_PRECISION_DEFINITION,
        "fold_metrics": fold_rows,
        "summary": summary,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    summary = payload["summary"]
    assert isinstance(summary, dict)
    for method, metrics in summary.items():
        assert isinstance(metrics, dict)
        row: dict[str, object] = {"Dataset": payload["dataset"], "Method": method}
        for metric, values in metrics.items():
            assert isinstance(values, dict)
            row[f"{metric}_Mean"] = values["mean"]
            row[f"{metric}_Sample_SD"] = values["sample_sd"]
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--method",
        action="append",
        type=_method_spec,
        required=True,
        help="Repeat NAME=PATH_TEMPLATE; template must contain {fold}.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(set(name for name, _ in args.method)) != len(args.method):
        raise SystemExit("method names must be unique")
    payload = recompute(dataset=args.dataset, methods=args.method, folds=args.folds)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "fold_metrics.csv", payload["fold_metrics"])
    _write_csv(output_dir / "summary.csv", _summary_rows(payload))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
