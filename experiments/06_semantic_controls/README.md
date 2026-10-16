# Semantic controls

The controlled-text pipeline separates the effects of text content, encoder
representation, and dimensionality reduction. It supports structured text,
name-only text, generated descriptions, shuffled features, and random
features, with explicit entity-ID alignment at every stage.

Core stages are implemented in `src/reproduction/tcm_suite/`:

1. `build_structured_text_corpora.py` creates deterministic corpora.
2. `generate_controlled_batch.py` produces generated-description conditions.
3. `extract_controlled_embeddings.py` extracts aligned SciBERT embeddings.
4. `train_shared_text_reducer.py` fits a shared 768-to-64 reducer.
5. `build_embedding_controls.py` constructs shuffled and random controls.
6. `run_text_condition_queue.py` evaluates each condition on matched targets.
7. `summarize_controlled_text_experiment.py` exports comparison tables.

Each command exposes its complete input contract through `--help`. A prepared
job directory can be resumed end to end with:

```bash
python src/reproduction/tcm_suite/run_controlled_text_pipeline.py \
  --experiment-root outputs/runs/semantic_controls \
  --dataset-dir data/tcm_suite
```
