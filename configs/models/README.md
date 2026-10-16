# Model configurations

`tcm_suite.conf` and `ethnobotany.conf` mirror the canonical HerbHG model and
training defaults. Their paths target the repository's standard `data/`,
`artifacts/`, and `outputs/` directories and are resolved relative to each
configuration file. Explicit CLI paths take precedence.

Validation-search candidates are stored with their experiment under
`experiments/08_model_selection/`.
