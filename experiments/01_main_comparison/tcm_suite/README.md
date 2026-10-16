# TCM-suite main comparison

Prepare a five-fold protocol:

```bash
herbhg benchmark prepare tcm-suite \
  --data-dir data/tcm_suite \
  --negative-file data/tcm_suite/negativehdlist.txt \
  --output-dir artifacts/protocols/tcm_suite_seed42 \
  --seed 42
```

Run HerbHG and the configured baselines on the same folds:

```bash
herbhg benchmark compare \
  --protocol-manifest artifacts/protocols/tcm_suite_seed42/manifest.json \
  --dataset tcm-suite \
  --data-dir data/tcm_suite \
  --embedding-dir artifacts/tcm_suite/features \
  --hghda-embedding-dir artifacts/tcm_suite/hghda_features \
  --methods astra smgcl milgnet lhgce bigi hghda \
  --herb-semantic artifacts/tcm_suite/herb_semantic_embeddings_64d.npy \
  --disease-semantic artifacts/tcm_suite/disease_semantic_embeddings_64d.npy \
  --output-root outputs/runs/main_comparison/tcm_suite_seed42 \
  --seed 42 --resume
```

External baseline sources are fetched and pinned separately under
`third_party/checkouts/`.
