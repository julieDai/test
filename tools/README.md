# Repository utilities

| Utility | Purpose |
|---|---|
| `build_source_manifest.py` | Build or verify the deterministic SHA-256 source inventory |
| `build_ethnobotany_protein_sequence_set_cache.py` | Construct the versioned Ethnobotany protein sequence-set cache from a frozen candidate snapshot |
| `kpgt_official_worker.py` | Execute KPGT feature extraction inside its pinned standalone environment |
| `prepare_tcm_astra_pca64_deterministic_v2.sh` | Build the TCM-suite PCA64 control feature bundle and its frozen execution plan |
| `repair_qwen_response_duplicates.py` | Deduplicate resumable semantic-generation JSONL artifacts while retaining a repair receipt |

Each utility exposes its complete arguments through `--help`. Generated files
belong under `artifacts/` or `outputs/`, not in the source package.
