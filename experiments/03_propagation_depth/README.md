# Propagation-depth sensitivity

This experiment varies the number of hypergraph propagation layers while
holding the remaining model and data settings fixed. The supported depths are
1 through 8.

```bash
python src/reproduction/tcm_suite/run_layer_folds.py \
  --tasks 1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1 \
  --protocol-dir artifacts/protocols/tcm_suite_retained_seed42 \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/propagation_depth \
  --seed 42 --resume
```

The task syntax is `depth:fold`; run folds `1` through `5` for each depth used
in the reported sensitivity curve.
