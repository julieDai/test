# Ethnobotany main comparison

Prepare a five-fold protocol from the processed relation directory:

```bash
herbhg benchmark prepare ethnobotany \
  --data-dir data/ethnobotany \
  --negative-file data/ethnobotany/negativehdlist.txt \
  --output-dir artifacts/protocols/ethnobotany_seed42 \
  --seed 42
```

Run the shared benchmark:

```bash
herbhg benchmark compare \
  --protocol-manifest artifacts/protocols/ethnobotany_seed42/manifest.json \
  --dataset ethnobotany \
  --data-dir data/ethnobotany \
  --embedding-dir artifacts/ethnobotany/features \
  --hghda-embedding-dir artifacts/ethnobotany/hghda_features \
  --methods astra smgcl milgnet lhgce bigi hghda \
  --output-root outputs/runs/main_comparison/ethnobotany_seed42 \
  --seed 42 --resume
```

The comparison launcher evaluates the centralized base model and external
baselines. The residual-MLP HerbHG variant is selected and frozen through
`experiments/08_model_selection/ethnobotany_semantic_adapter/`; its outer-fold
result is added only from that model-selection receipt. The frozen protocol
and feature manifest are part of every run receipt.
