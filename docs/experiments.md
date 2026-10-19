# Experiment organization

Experiment directories follow the order of the empirical study. Each recipe
contains only experiment-specific configuration and launch logic; reusable
model, protocol, feature, and metric code remains under `src/`.

| Directory | Study | Primary controlled variable |
|---|---|---|
| `01_main_comparison/` | Main comparison on TCM-suite and Ethnobotany | Method |
| `02_component_ablation/` | Component ablation | Enabled model component |
| `03_propagation_depth/` | Propagation-depth sensitivity | Number of graph layers |
| `04_relation_perturbation/` | Relation robustness | Perturbation type and rate |
| `05_cold_start/held_out_herbs/` | Held-out-herb evaluation | Held-out entity set |
| `05_cold_start/semantic_fusion/` | Cold-start semantic fusion | Fusion weight |
| `06_semantic_controls/` | Semantic-feature controls | Text condition or embedding control |
| `07_case_study/` | Case-study ranking | Trained final model |
| `08_model_selection/ethnobotany_semantic_adapter/` | Semantic-adapter selection | Adapter architecture and training recipe |
| `08_model_selection/tcm_hyperparameters/` | TCM-suite model selection | Epoch, learning rate, and feature configuration |

## Evaluation protocols

Main comparisons and model-selection workflows share the same frozen protocol
manifest, split rows, feature bundle, seed policy, and metric implementation.
Candidate and checkpoint selection use a predeclared inner-validation
criterion; the selected configuration is then evaluated on the outer test
partition.

The ablation, propagation-depth, perturbation, and cold-start directories
preserve the retained secondary-study designs used for their corresponding
within-study comparisons. Results should be compared only among runs produced
by the same numbered recipe and protocol family.

Outputs should use a run identifier that does not encode performance. Store
the resolved configuration, protocol digest, source manifest, logs,
checkpoints, prediction arrays, and metrics below the same run root. The
repository versions only the compact snapshot selected by the current
manuscript under `results/reference/manuscript_current/`; complete and
superseded runs remain outside version control.
