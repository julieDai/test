# Experiments

The numbered directories follow the order in which experiments are presented
in the manuscript. Generated protocols belong in `artifacts/protocols/`; run
artifacts belong in `outputs/runs/`. Neither data nor generated results are
stored with source code.

| Directory | Purpose | Dataset |
|---|---|---|
| `01_main_comparison` | Frozen five-fold benchmark against reference methods | TCM-suite, Ethnobotany |
| `02_component_ablation` | Remove or replace one HerbHG component at a time | TCM-suite |
| `03_propagation_depth` | Sweep hypergraph propagation depth | TCM-suite |
| `04_relation_perturbation` | Measure robustness to controlled relation perturbations | TCM-suite |
| `05_cold_start` | Leave-one-herb-out evaluation and semantic-fusion sweep | TCM-suite |
| `06_semantic_controls` | Compare controlled text and embedding conditions | TCM-suite |
| `07_case_study` | Withheld-association recovery for selected herbs | TCM-suite |
| `08_model_selection` | Validation-only hyperparameter and adapter selection | Both |

Run commands from the repository root after installing the project. Every
long-running entry point accepts `--help`; task-oriented runners also accept
`--resume` so completed folds are not repeated.
