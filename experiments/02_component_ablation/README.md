# Component ablation

The runner keeps the data split and training budget fixed while replacing one
model component. Variant keys are:

| Key | Variant |
|---|---|
| `a` | remove the disease-side hypergraph |
| `b` | replace learned aggregation with mean pooling |
| `d` | remove semantic text features |
| `e` | replace pretrained biological features with random initialization |
| `f` | remove clique expansion |

First freeze the retained TCM-suite sampling and fold assignment:

```bash
python src/reproduction/tcm_suite/prepare_protocol.py \
  --dataset-dir data/tcm_suite \
  --output-dir artifacts/protocols/tcm_suite_retained_seed42 \
  --seed 42
```

Run any comma-separated set of `variant:fold` tasks:

```bash
python src/reproduction/tcm_suite/run_ablation_folds.py \
  --tasks a:1,b:1,d:1,e:1,f:1 \
  --protocol-dir artifacts/protocols/tcm_suite_retained_seed42 \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/component_ablation \
  --seed 42 --resume
```

Use folds `1` through `5` for each reported variant.
