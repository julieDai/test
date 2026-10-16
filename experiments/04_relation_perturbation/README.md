# Relation perturbation

For each requested percentage, preparation removes a deterministic subset of
unique H--C and P--D edges and adds the same number of non-edges. The fold
runner uses the corresponding graph and swaps an equal number of positive and
negative training H--D labels. Test rows remain fixed by the frozen protocol.

```bash
python src/reproduction/tcm_suite/prepare_noise.py \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/relation_perturbation \
  --ratios 10,20,30,40,50,60,70,80,90 \
  --seed 42

python src/reproduction/tcm_suite/run_noise_folds.py \
  --tasks 10:1,20:1,30:1,40:1,50:1,60:1,70:1,80:1,90:1 \
  --protocol-dir artifacts/protocols/tcm_suite_retained_seed42 \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/relation_perturbation \
  --seed 42 --resume
```

The task syntax is `percentage:fold`; run all five folds for each percentage.
