# Leave-one-herb-out evaluation

Five targets are sampled reproducibly from the twenty herbs with the largest
H--C degree. All H--D rows for a target herb are withheld during training, and
the trained model ranks the complete disease set.

```bash
python src/reproduction/tcm_suite/run_cold_start.py \
  --experiments 1,2,3,4,5 \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/cold_start/held_out_herbs \
  --seed 42 --resume
```
