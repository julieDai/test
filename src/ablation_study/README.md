# Component-variant implementations

This package contains the five TCM-suite component variants used by
`src/reproduction/tcm_suite/run_ablation_folds.py`. TensorFlow-backed classes
are imported lazily, so path and configuration utilities remain lightweight.

The runner resolves all data and embedding paths explicitly and writes each
variant/fold task to an isolated output directory. See
`experiments/02_component_ablation/` for the task map and command recipe.
