# Withheld-association case study

The three entry points form a reproducible pipeline:

1. `prepare_case_study_data.py` removes a target herb's known H--D edges and
   creates a balanced training file.
2. `train_final.py` trains HerbHG and saves herb/disease embeddings with their
   identifier maps.
3. `predict_correct.py` ranks diseases and reports recovery metrics.

All input and output paths can be supplied on the command line. Public
environment variables are `HERBHG_DATA_DIR`, `HERBHG_CASE_DATA_DIR`,
`HERBHG_CASE_RESULT_DIR`, and `HERBHG_SEED`. Backend compatibility aliases are
also accepted.

See `experiments/07_case_study/` for a complete invocation.
