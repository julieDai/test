#!/usr/bin/env python3
"""Aggregate cold-start text-fusion sensitivity runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "AUC", "AUPR", "F1-Score", "Precision", "Recall",
    "Hits@10", "Precision@10", "Recall@10", "MRR", "NDCG@10",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=25)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    rows = []
    failures = []
    for path in sorted(run_dir.glob("alpha_*/exp_*/status.json")):
        status = json.loads(path.read_text())
        if status.get("status") != "success":
            failures.append({"path": str(path), "status": status.get("status"),
                             "error": status.get("error")})
            continue
        row = {
            "alpha": status["text_fusion_alpha"],
            "experiment": status["experiment"],
            "target_herb": status["target_herb"],
            "elapsed_seconds": status["elapsed_seconds"],
        }
        row.update(status["metrics"])
        rows.append(row)

    if not rows:
        raise SystemExit(f"no successful tasks found under {run_dir}")
    task_df = pd.DataFrame(rows).sort_values(["alpha", "experiment"])
    task_df.to_csv(run_dir / "task_metrics.csv", index=False)

    summary_rows = []
    for alpha, group in task_df.groupby("alpha", sort=True):
        row = {"alpha": alpha, "successful_tasks": len(group),
               "total_elapsed_seconds": group["elapsed_seconds"].sum()}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "alpha_summary.csv", index=False)

    anchor = task_df[task_df["alpha"] == 1.0].set_index("experiment")
    delta_rows = []
    if not anchor.empty:
        for _, row in task_df.iterrows():
            if row["experiment"] not in anchor.index:
                continue
            delta = {"alpha": row["alpha"], "experiment": row["experiment"],
                     "target_herb": row["target_herb"]}
            for metric in METRICS:
                delta[f"{metric}_delta_vs_alpha_1"] = row[metric] - anchor.loc[row["experiment"], metric]
            delta_rows.append(delta)
    pd.DataFrame(delta_rows).to_csv(run_dir / "paired_vs_alpha_1.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, metric, title in zip(axes, ["AUC", "AUPR"], ["AUROC", "AUPRC"]):
        axis.errorbar(summary_df["alpha"], summary_df[f"{metric}_mean"],
                      yerr=summary_df[f"{metric}_std"], marker="o", capsize=4)
        axis.set_xlabel("Text fusion coefficient α")
        axis.set_ylabel(title)
        axis.set_xticks(summary_df["alpha"])
        axis.grid(alpha=0.25)
    fig.suptitle("TCM-suite leave-one-herb-out text sensitivity")
    fig.savefig(run_dir / "text_weight_sensitivity.png", dpi=200)
    plt.close(fig)

    complete = len(rows) == args.expected_tasks and not failures
    payload = {
        "successful_tasks": len(rows), "expected_tasks": args.expected_tasks,
        "complete": complete, "failed_or_incomplete": failures,
        "population_std_ddof": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# TCM-suite 冷启动文本融合系数实验",
        "",
        f"完成度：{len(rows)}/{args.expected_tasks}；状态：{'完整' if complete else '未完整'}。",
        "",
        "α=1 是当前 HerbHG 文本增强表示，α=0 是同一计算图中的无文本结构表示；"
        "中间值在两者间线性融合。每个 α 对相同五个留出草药重新训练。",
        "",
        "| α | AUROC（均值±标准差） | AUPRC（均值±标准差） | F1（均值±标准差） |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['alpha']:.2f} | {row['AUC_mean']:.4f} ± {row['AUC_std']:.4f} | "
            f"{row['AUPR_mean']:.4f} ± {row['AUPR_std']:.4f} | "
            f"{row['F1-Score_mean']:.4f} ± {row['F1-Score_std']:.4f} |"
        )
    lines += [
        "",
        "Protocol note: the text condition and threshold rule are recorded in the run metadata. "
        "AUROC and AUPRC are the primary metrics for comparing α.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
