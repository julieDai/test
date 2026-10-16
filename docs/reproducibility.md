# Reproducibility workflow

## 1. Create the core environment

```bash
conda env create -f environment.yml
conda activate herbhg
python -m pip install -e .
herbhg smoke
```

The core environment contains HerbHG and its TensorFlow backend. Baseline
adapters additionally check the framework versions required by the pinned
author implementations.

For semantic controls or plot generation, install the corresponding optional
dependencies:

```bash
python -m pip install -e '.[semantic,plots]'
```

KPGT feature extraction and author-code baselines run in their pinned
standalone environments rather than the core TensorFlow environment.

## 2. Prepare a frozen protocol

```bash
herbhg benchmark prepare ethnobotany \
  --data-dir data/ethnobotany/relations \
  --negative-file data/ethnobotany/relations/negativehdlist.txt \
  --output-dir artifacts/protocols/ethnobotany/seed42 \
  --seed 42 \
  --folds 5 \
  --validation-fraction 0.10
```

For TCM-suite, use `tcm-suite` as the dataset and pass its explicit frozen
negative file. The generated manifest is the common input to every method in
the comparison.

## 3. Run a backend development check

Dataset-specific training options are forwarded to the isolated backend. The
direct commands retain each backend's native cross-validation workflow and
are intended for implementation checks:

```bash
herbhg train tcm-suite --help
herbhg train ethnobotany --help
```

Use the benchmark entry points for paper results and comparisons:

```bash
herbhg benchmark run --help
herbhg benchmark compare --help
```

## 4. Prepare comparison methods

```bash
bash third_party/fetch_baselines.sh
herbhg baseline preflight \
  --protocol-manifest artifacts/protocols/ethnobotany/seed42/manifest.json \
  --require-ready
herbhg baseline run --help
```

The lock file fixes upstream commits, while every adapter records the
effective runtime, source identity, protocol digest, predictions, and
hyperparameters with its output.

## 5. Run controlled studies

The public experiment dispatcher exposes validation-controlled selection
workflows:

```bash
herbhg experiment ethnobotany-alpha --help
herbhg experiment ethnobotany-refined --help
herbhg experiment ethnobotany-semantic-adapter --help
herbhg experiment tcm-epoch-boundary --help
herbhg experiment tcm-hyperparameters --help
herbhg experiment tcm-pca64-control --help
herbhg experiment tcm-semantic-adapter --help
```

Numbered recipes under `experiments/` provide the paper-oriented organization
for main comparisons, ablations, sensitivity studies, robustness tests,
cold-start evaluation, semantic controls, and case studies.

## 6. Verify source identity

```bash
python tools/build_source_manifest.py --check
```

After an intentional versioned source change, regenerate
`provenance/source_manifest.json` and commit it together with the changed
files. Runtime data, model outputs, external checkouts, and cached artifacts
are excluded from this inventory.
