# Paper-to-code map

| Paper element | Implementation |
|---|---|
| HerbHG public model entry points | `src/herbhg/models/backends/*/HerbHG.py` |
| TCM-suite numerical backend | `src/herbhg/models/backends/tcm_suite/` |
| Ethnobotany numerical backend | `src/herbhg/models/backends/ethnobotany/` |
| Hypergraph encoder and decoder implementation | Backend `HGHDA.py` and `HDR.py` modules |
| Ethnobotany residual semantic adapter | `src/herbhg/models/backends/ethnobotany/ASTRASemanticRefined.py` |
| TCM-suite semantic-adapter candidates | `src/herbhg/models/backends/tcm_suite/ASTRATCMSemanticAdapter*.py` |
| Semantic prompt conditions | `configs/prompts/` |
| Frozen five-fold protocol | `src/astra/benchmark/protocol.py` |
| Shared metrics | `src/astra/benchmark/metrics.py` |
| Main comparison orchestration | `src/astra/benchmark/fair_comparison.py` and `experiments/01_main_comparison/` |
| Commit-pinned baseline adapters | `src/astra/official_baselines/` |
| Public baseline registry | `src/herbhg/baselines/` |
| Component ablation | `experiments/02_component_ablation/` |
| Propagation-depth study | `experiments/03_propagation_depth/` |
| Relation-perturbation study | `experiments/04_relation_perturbation/` |
| Cold-start experiments | `experiments/05_cold_start/` |
| Semantic controls | `experiments/06_semantic_controls/` |
| Case study | `experiments/07_case_study/` |
| Model-selection experiments | `experiments/08_model_selection/` |
| Source identity and migration provenance | `provenance/` |
