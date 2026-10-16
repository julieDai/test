# Case study

The case study removes known H--D associations for Chuanxiong or Huangqi,
trains HerbHG on the remaining graph, and ranks diseases for the held-out herb.

```bash
python src/case_study/prepare_case_study_data.py \
  --herb both --data-dir data/tcm_suite \
  --output-dir data/case_study --seed 42

python src/case_study/train_final.py \
  --herb both --data-dir data/case_study \
  --source-data-dir data/tcm_suite \
  --output-dir outputs/runs/case_study --seed 42 --device 0

python src/case_study/predict_correct.py \
  --herb both --data-dir data/case_study \
  --output-dir outputs/runs/case_study
```

The output contains complete rankings, top-100 rankings, and ranking metrics
for the withheld associations.
