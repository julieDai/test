# Semantic-fusion sensitivity

The prediction representation is

```text
(1 - alpha) * structural_branch + alpha * semantic_branch
```

Run each coefficient on the same five held-out herbs:

```bash
python src/reproduction/tcm_suite/run_text_weight_cold_start.py \
  --tasks 0:1,0.25:1,0.5:1,0.75:1,1:1 \
  --dataset-dir data/tcm_suite \
  --run-dir outputs/runs/cold_start/semantic_fusion \
  --seed 42 --resume
```

Repeat the task list for experiments `2` through `5`, then aggregate and test
paired differences:

```bash
python src/reproduction/tcm_suite/summarize_text_weight_cold_start.py \
  --run-dir outputs/runs/cold_start/semantic_fusion

python src/reproduction/tcm_suite/statistical_test_text_weight.py \
  --task-metrics outputs/runs/cold_start/semantic_fusion/task_metrics.csv \
  --output-dir outputs/runs/cold_start/semantic_fusion/statistics
```
