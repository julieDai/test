# Data layout and protocol

Datasets are supplied locally and are not embedded in the source package.
Place immutable source files under `data/`, derived reusable features under
`artifacts/`, and run-specific products under `outputs/`.

## Relation inputs

The benchmark protocol accepts the following bipartite relations when they are
available for a dataset:

| Relation | Meaning | Conventional file stem |
|---|---|---|
| H--D | herb--disease labels | `H_D` |
| H--C | herb--compound links | `H_C` |
| C--P | compound--protein links | `C_P` |
| P--D | protein--disease links | `P_D` |

Ethnobotany entity mapping files use the names
`herb_id_mapping.csv`, `compound_id_mapping.csv`,
`protein_id_mapping.csv`, and `disease_id_mapping.csv`. TCM-suite experiments
also require an explicit frozen negative H--D file. Input paths are bound into
the generated protocol manifest and verified by digest when loaded.

## Frozen evaluation protocol

`astra.benchmark.protocol` is the single conversion and split boundary shared
by HerbHG and all protocol-matched baselines. It performs relation
de-duplication, identifier compaction, negative-set construction, and the
five-fold split. For each outer fold, 10% of each class in the outer-training
partition is deterministically reserved for validation.

Model selection, checkpoint selection, and operating-threshold selection use
only the inner validation partition. The outer test labels are used only for
the final evaluation. AUROC and AUPRC are calculated from continuous scores;
threshold-dependent metrics use the threshold selected on validation data.

The frozen protocol consists of:

- `manifest.json`: dataset, seed, split policy, counts, and input digests;
- `protocol_arrays.npz`: compact relations, labels, row identities, and folds;
- `entity_mappings.json`: the compact-to-source identifier correspondence.

Every result directory should retain the exact protocol manifest or a digest
and relative reference to it.

## Semantic features

Prompt templates are versioned under `configs/prompts/`. Generated text,
embeddings, reduced feature matrices, and row-identity sidecars are derived
artifacts. A feature matrix is accepted only with the row identity expected by
the associated protocol so that feature rows cannot be silently reordered.
