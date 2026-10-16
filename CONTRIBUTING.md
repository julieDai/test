# Contributing to HerbHG

Contributions should preserve reproducibility across both supported datasets
and keep the public package importable without initializing a numerical
backend.

## Development setup

```bash
conda env create -f environment.yml
conda activate herbhg
herbhg smoke
pytest
```

## Change guidelines

1. Keep reusable public code under `src/herbhg/`.
2. Keep compatibility implementations under their existing namespace when a
   result depends on their import path or behavior.
3. Execute the TCM-suite and Ethnobotany backends in separate processes.
4. Add or update a numbered recipe under `experiments/` for experimental
   changes.
5. Select hyperparameters and operating thresholds on validation data only.
6. Record seeds, resolved paths, dependency versions, and immutable input
   identifiers in generated run metadata.
7. Do not commit datasets, embeddings, checkpoints, logs, or full run trees.

## Tests

Run the dependency-light checks for command, path, or packaging changes:

```bash
herbhg smoke
pytest tests/unit tests/smoke
```

Changes to model execution or evaluation also require the relevant integration
tests and a smoke-size protocol run. A pull request should state the exact
commands used and summarize their results.

## External methods

Do not vendor third-party source into the repository. Update the pinned
revision metadata and fetch tooling under `third_party/`, retain upstream
license and citation information, and keep method-specific dependencies in an
isolated environment when required.
