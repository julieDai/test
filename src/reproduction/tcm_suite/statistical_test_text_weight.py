#!/usr/bin/env python3
"""Paired significance tests for the TCM-suite text-fusion experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["AUC", "AUPR", "F1-Score"]
REFERENCE_ALPHA = 1.0
COMPARISON_ALPHAS = [0.0, 0.25, 0.5, 0.75]


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_max = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * values[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(1.0, running_max)
    adjusted = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted


def paired_test(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    difference = left - right
    n = len(difference)
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    se = sd / np.sqrt(n)
    margin = float(stats.t.ppf(0.975, n - 1) * se)
    t_result = stats.ttest_rel(left, right, alternative="two-sided")
    wilcoxon_result = stats.wilcoxon(
        left, right, zero_method="wilcox", alternative="two-sided", method="exact"
    )
    shapiro = stats.shapiro(difference)
    return {
        "n_pairs": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "paired_cohen_dz": mean / sd if sd else np.inf,
        "t_statistic": float(t_result.statistic),
        "t_p_raw": float(t_result.pvalue),
        "wilcoxon_statistic": float(wilcoxon_result.statistic),
        "wilcoxon_p_raw": float(wilcoxon_result.pvalue),
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "positive_differences": int((difference > 0).sum()),
        "negative_differences": int((difference < 0).sum()),
        "zero_differences": int((difference == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.task_metrics)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = ["experiment", "target_herb"]
    reference = data[data["alpha"] == REFERENCE_ALPHA].sort_values(keys)
    if len(reference) != 5:
        raise RuntimeError(f"expected five reference pairs, found {len(reference)}")

    rows = []
    for metric in METRICS:
        for alpha in COMPARISON_ALPHAS:
            comparison = data[data["alpha"] == alpha].sort_values(keys)
            if comparison[keys].reset_index(drop=True).equals(reference[keys].reset_index(drop=True)) is False:
                raise RuntimeError(f"pairing mismatch for alpha={alpha}, metric={metric}")
            result = {
                "metric": metric,
                "comparison_alpha": alpha,
                "reference_alpha": REFERENCE_ALPHA,
                "comparison_mean": float(comparison[metric].mean()),
                "reference_mean": float(reference[metric].mean()),
            }
            result.update(paired_test(comparison[metric], reference[metric]))
            rows.append(result)

    result_df = pd.DataFrame(rows)
    # Four alpha-vs-reference hypotheses are corrected separately for each metric.
    for metric in METRICS:
        mask = result_df["metric"] == metric
        result_df.loc[mask, "t_p_holm"] = holm_adjust(result_df.loc[mask, "t_p_raw"])
        result_df.loc[mask, "wilcoxon_p_holm"] = holm_adjust(
            result_df.loc[mask, "wilcoxon_p_raw"]
        )
    result_df["t_significant_holm_0.05"] = result_df["t_p_holm"] < 0.05
    result_df["wilcoxon_significant_holm_0.05"] = result_df["wilcoxon_p_holm"] < 0.05
    result_df.to_csv(output_dir / "paired_tests_vs_alpha_1.csv", index=False)

    # A secondary comparison asks whether the selected alpha improves over no text.
    selected = data[data["alpha"] == 0.25].sort_values(keys)
    no_text = data[data["alpha"] == 0.0].sort_values(keys)
    secondary_rows = []
    for metric in METRICS:
        row = {
            "metric": metric, "comparison_alpha": 0.25, "reference_alpha": 0.0,
            "comparison_mean": float(selected[metric].mean()),
            "reference_mean": float(no_text[metric].mean()),
        }
        row.update(paired_test(selected[metric], no_text[metric]))
        secondary_rows.append(row)
    secondary_df = pd.DataFrame(secondary_rows)
    secondary_df["t_p_holm_across_metrics"] = holm_adjust(secondary_df["t_p_raw"])
    secondary_df["wilcoxon_p_holm_across_metrics"] = holm_adjust(
        secondary_df["wilcoxon_p_raw"]
    )
    secondary_df.to_csv(output_dir / "paired_tests_alpha_0p25_vs_0.csv", index=False)

    payload = {
        "pairing_unit": "five identical held-out target herbs",
        "alternative": "two-sided",
        "reference_alpha": REFERENCE_ALPHA,
        "holm_family": "four alpha-vs-1 comparisons, separately for each metric",
        "minimum_two_sided_exact_wilcoxon_p_with_n5_no_ties": 0.0625,
        "results_vs_alpha_1": result_df.to_dict(orient="records"),
        "results_alpha_0p25_vs_0": secondary_df.to_dict(orient="records"),
    }
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(result_df.to_string(index=False))
    print("\nalpha=0.25 vs alpha=0\n")
    print(secondary_df.to_string(index=False))


if __name__ == "__main__":
    main()
