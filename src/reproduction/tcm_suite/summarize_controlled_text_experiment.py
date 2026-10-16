#!/usr/bin/env python3
"""Summarize controlled text conditions and run pre-specified paired tests."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["Recall", "Precision", "F1-Score", "AUC", "AUPR"]
PRIMARY = ["AUC", "AUPR"]
COMPARISONS = [
    ("llm_original_prompt", "llm_clean", "cross-association wording in LLM prompts"),
    ("structured_with_indication", "structured_clean", "effect/indication fields in herb template"),
    ("llm_clean", "no_text_alpha_0", "clean LLM text versus no text"),
    ("structured_clean", "no_text_alpha_0", "clean structured text versus no text"),
    ("name_only", "no_text_alpha_0", "names versus no text"),
    ("llm_clean", "clean_shuffled", "aligned clean LLM text versus shuffled text"),
    ("llm_clean", "random_matched", "clean LLM text versus matched random vectors"),
    ("structured_clean", "llm_clean", "clean structured text versus clean LLM text"),
    ("legacy_embeddings_alpha_0.25", "no_text_alpha_0", "legacy text embeddings versus no text"),
]


def load_statuses(root, condition=None):
    rows = []
    patterns = [
        str(root / "alpha_*" / "exp_*" / "status.json"),
        str(root / "exp_*" / "status.json"),
    ]
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    for path_text in paths:
        path = Path(path_text)
        value = json.loads(path.read_text())
        if value.get("status") != "success":
            raise RuntimeError(f"non-success status: {path}")
        row = {
            "condition": condition or value["condition"],
            "experiment": int(value["experiment"]),
            "target_herb": str(value["target_herb"]),
            "elapsed_seconds": float(value["elapsed_seconds"]),
            "status_path": str(path.resolve()),
        }
        row.update({metric: float(value["metrics"][metric]) for metric in METRICS})
        rows.append(row)
    return rows


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--legacy-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition_dir in sorted(path for path in args.new_results.iterdir() if path.is_dir()):
        rows.extend(load_statuses(condition_dir))
    rows.extend(load_statuses(args.legacy_results / "alpha_0p25", "legacy_embeddings_alpha_0.25"))
    rows.extend(load_statuses(args.legacy_results / "alpha_0p00", "no_text_alpha_0"))
    frame = pd.DataFrame(rows).sort_values(["condition", "experiment"])
    counts = frame.groupby("condition").size()
    if not (counts == 5).all():
        raise RuntimeError(f"expected five repeats per condition:\n{counts}")
    targets = frame.groupby("condition")["target_herb"].apply(list)
    reference = targets.iloc[0]
    if any(values != reference for values in targets):
        raise RuntimeError(f"paired target mismatch:\n{targets}")
    frame.to_csv(args.output_dir / "per_target_metrics.csv", index=False)

    summary_rows = []
    for condition, part in frame.groupby("condition", sort=True):
        row = {"condition": condition, "n": len(part), "mean_elapsed_seconds": part["elapsed_seconds"].mean()}
        for metric in METRICS:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "five_target_summary.csv", index=False)

    tests = []
    indexed = frame.set_index(["condition", "experiment"])
    for left, right, purpose in COMPARISONS:
        for metric in PRIMARY + ["F1-Score"]:
            left_values = indexed.loc[left].sort_index()[metric].to_numpy()
            right_values = indexed.loc[right].sort_index()[metric].to_numpy()
            differences = left_values - right_values
            t_result = stats.ttest_rel(left_values, right_values)
            try:
                w_result = stats.wilcoxon(left_values, right_values, alternative="two-sided", method="exact")
                wilcoxon_p = float(w_result.pvalue)
                wilcoxon_stat = float(w_result.statistic)
            except ValueError:
                wilcoxon_p, wilcoxon_stat = 1.0, 0.0
            tests.append({
                "left": left, "right": right, "purpose": purpose, "metric": metric,
                "mean_difference_left_minus_right": differences.mean(),
                "difference_std": differences.std(ddof=1),
                "paired_t_statistic": float(t_result.statistic),
                "paired_t_p_raw": float(t_result.pvalue),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_exact": wilcoxon_p,
            })
    tests = pd.DataFrame(tests)
    for metric in tests["metric"].unique():
        mask = tests["metric"] == metric
        tests.loc[mask, "paired_t_p_holm_within_metric"] = holm(tests.loc[mask, "paired_t_p_raw"])
        tests.loc[mask, "wilcoxon_p_holm_within_metric"] = holm(tests.loc[mask, "wilcoxon_p_exact"])
    tests.to_csv(args.output_dir / "planned_paired_tests.csv", index=False)

    payload = {
        "status": "success", "new_tasks": 35, "new_failures": 0,
        "conditions": summary.to_dict(orient="records"),
        "paired_targets": reference,
        "test_note": "AUC/AUPR are primary. F1 is supplementary because the legacy threshold maximizes F1 on test labels.",
        "wilcoxon_note": "With n=5 nonzero pairs, the minimum exact two-sided Wilcoxon p-value is 0.0625.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(summary[["condition"] + [f"{metric}_mean" for metric in METRICS]].to_string(index=False))
    print("\nPlanned primary comparisons:")
    print(tests[tests["metric"].isin(PRIMARY)][[
        "left", "right", "metric", "mean_difference_left_minus_right",
        "paired_t_p_raw", "paired_t_p_holm_within_metric", "wilcoxon_p_exact",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
