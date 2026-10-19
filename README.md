# HerbHG

HerbHG is a structure--semantic hypergraph learning framework for
herb--disease association prediction. It combines heterogeneous biological
relations with semantic representations and supports experiments on the
TCM-suite and Ethnobotany datasets.

The repository provides a unified command-line interface, dataset-specific
model backends, protocol-controlled comparison runners, baseline adapters,
and experiment recipes aligned with the accompanying paper.

## Repository layout

```text
configs/       Model, protocol, and prompt configuration
data/          Local datasets (not versioned)
experiments/   Numbered experiment recipes and launch instructions
src/herbhg/    Public package and dataset-specific model backends
src/astra/     Compatibility layer for reproducible experiment runners
third_party/   Pinned metadata and fetch tools for external methods
results/       Small curated reference summaries
artifacts/     Generated features and checkpoints (not versioned)
outputs/       Run outputs (not versioned)
tests/         Unit, integration, and smoke tests
```

The two model backends execute in separate Python processes. This preserves
their dataset-specific module layouts while exposing one stable `herbhg`
interface.

## Installation

The reference environment uses Python 3.10 and TensorFlow 2.10:

```bash
conda env create -f environment.yml
conda activate herbhg
```

For an existing compatible environment:

```bash
python -m pip install -e .
```

Optional semantic-generation and plotting tools can be installed with:

```bash
python -m pip install -e '.[semantic,plots]'
```

Author-code baselines and KPGT extraction use their pinned standalone
environments; the preflight and worker commands validate those runtimes.

Raw relations, pretrained embeddings, checkpoints, and full run outputs are
kept outside version control. Place local assets according to the selected
experiment recipe under `experiments/` or pass their locations explicitly on
the command line.

## Quick verification

The smoke check uses a temporary synthetic relation set. It does not train a
model or initialize TensorFlow:

```bash
herbhg smoke
pytest
```

## Backend development runs

The direct training commands exercise each dataset backend with its native
cross-validation entry point. They are intended for development and local
checks. Paper comparisons use the protocol-controlled commands in the next
section.

Train the TCM-suite backend:

```bash
herbhg train tcm-suite \
  --data-dir /path/to/tcm-suite \
  --output-dir outputs/tcm_suite/run_01 \
  --seed 42 --device 0
```

Train the Ethnobotany backend:

```bash
herbhg train ethnobotany \
  --data-dir /path/to/ethnobotany/relations \
  --embedding-dir /path/to/ethnobotany/embeddings \
  --output-dir outputs/ethnobotany/run_01 \
  --seed 42 --device cuda
```

Use `--dry-run` to inspect resolved paths without loading TensorFlow. Run
`herbhg train <dataset> --help` for all backend options.

## Protocol-controlled paper evaluation

Create a frozen five-fold benchmark protocol:

```bash
herbhg benchmark prepare ethnobotany \
  --data-dir /path/to/relations \
  --negative-file /path/to/negativehdlist.txt \
  --output-dir artifacts/protocols/ethnobotany_seed42 \
  --seed 42 --folds 5
```

Run HerbHG on one frozen fold:

```bash
herbhg benchmark run \
  --protocol-manifest artifacts/protocols/ethnobotany_seed42/manifest.json \
  --dataset ethnobotany \
  --data-dir /path/to/relations \
  --embedding-dir /path/to/embeddings \
  --fold 1 \
  --seed 42 \
  --output-root outputs/main_comparison/herbhg/fold_1
```

Available command groups can be listed with:

```bash
herbhg benchmark --help
herbhg baseline --help
herbhg experiment --help
```

The numbered directories under `experiments/` contain the complete commands
for main comparisons, ablations, propagation depth, relation perturbation,
cold-start evaluation, semantic controls, case studies, and model selection.

## Documentation

- [Architecture](docs/architecture.md)
- [Data and evaluation protocol](docs/data.md)
- [Experiment organization](docs/experiments.md)
- [Comparison methods](docs/baselines.md)
- [Reproducibility](docs/reproducibility.md)
- [Paper-to-code map](docs/paper_to_code.md)
- [Current manuscript results](results/reference/manuscript_current/README.md)

## Reproducibility conventions

- Dataset splits are materialized before model execution and shared by all
  protocol-matched methods.
- Model selection uses validation data; the test fold is evaluated with the
  selected configuration and operating threshold.
- External baseline revisions are pinned in `third_party/baselines.lock.json`.
- Generated runs record resolved configuration, inputs, seeds, and metrics in
  their output directories.

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and
[CITATION.cff](CITATION.cff) for citation metadata. Security reports should
follow [SECURITY.md](SECURITY.md).
