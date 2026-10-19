# Current manuscript result snapshot

This directory is the compact, machine-readable result snapshot for
`ASTAR-review/HerbHGreview/BDMA_LaTeX_template.tex` as audited on 2026-10-19.
The manuscript SHA-256 and the checksums of the principal source artifacts are
recorded in `provenance.json`.

## Contents

| File | Manuscript content |
| --- | --- |
| `dataset_statistics.csv` | Processed-dataset statistics and reported H-D density |
| `main_comparison.csv` | Fivefold method-comparison table for TCM-suite and Ethnobotany |
| `reported_comparisons.csv` | Relative improvements stated in the comparison text |
| `component_ablation.csv` | TCM-suite component ablation |
| `propagation_depth.csv` | TCM-suite propagation-depth sensitivity |
| `relation_perturbation.csv` | Coordinated perturbation table and the figure's 0% reference |
| `cold_start.csv` | Five held-out-herb runs and their across-herb summary |
| `semantic_fusion.csv` | Fixed branch-fusion coefficient sweep |
| `semantic_controls.csv` | Controlled semantic-input analysis |
| `semantic_control_tests.csv` | Planned paired AUROC/AUPRC tests supporting the Holm-adjusted inference |
| `case_study.csv` | Top-10 recovery for Chuanxiong and Huangqi |
| `ethnobotany_adapter_confirmation_seed43.csv` | Separate seed-43 adapter confirmation cited in the text |
| `provenance.json` | Snapshot identity, protocols, source checksums, and interpretation boundaries |

## Evidence fields

Every result file carries an `evidence_type`, a source path, and a concise
limitation. Evidence types distinguish strict or corrected-protocol audits,
task-adapted reproductions, externally supplied benchmark values, historical
sensitivity analyses, and descriptive post-selection diagnostics. The main
comparison CSV intentionally preserves the rows displayed in the manuscript;
its rows therefore do not all share one evidence type.

Paths beginning with `ASTRA/`, `Graduation_Project/`, or `ASTAR-review/` are
relative to the workspace root used to build this snapshot. Those archives are
provenance inputs and are not duplicated here.

## Statistical conventions

- Main-comparison entries are fivefold means and sample standard deviations
  unless a row records a source-specific convention.
- The five historical ablation variants retain population SD across folds.
  The full-model ablation row reproduces the manuscript-facing values and
  records its supporting fold archive separately.
- Blank standard-deviation cells mean that the current manuscript reports only
  the mean; they do not denote zero variance.
- `cold_start.csv` summarizes variation across five distinct target herbs. Its
  displayed SD is the population SD retained by the manuscript source, not a
  cross-validation uncertainty estimate.
- `semantic_fusion.csv` and `semantic_controls.csv` report sample SD across the
  same five target herbs.
- AUROC and AUPRC use continuous scores. Threshold-dependent metrics inherit the
  threshold rule recorded for each experiment in `provenance.json`.
- The planned semantic-control tests use two-sided paired t-tests and exact
  Wilcoxon signed-rank tests across matched herbs, with Holm correction within
  each metric. No planned AUROC or AUPRC comparison was significant after
  correction.

## Scope

The snapshot contains the values used by the current English manuscript, not a
newly unified rerun of every method. It excludes raw predictions, model
checkpoints, embeddings, logs, and manuscript figures. Figure assets remain
manuscript presentation files and are listed only in `provenance.json`.
