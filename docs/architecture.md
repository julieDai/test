# Architecture

HerbHG separates the public interface, numerical backends, experiment recipes,
and externally maintained comparison methods.

## Repository layers

| Layer | Location | Responsibility |
|---|---|---|
| Public package | `src/herbhg/` | Stable commands, paths, registries, and lightweight orchestration |
| Model backends | `src/herbhg/models/backends/` | Dataset-specific HerbHG implementations and their local runtime support |
| Experiment engine | `src/astra/` | Frozen protocol construction, feature preparation, model selection, baseline adapters, and metric aggregation |
| Experiment recipes | `experiments/` | Numbered, paper-oriented entry points and experiment-specific configuration |
| Configuration | `configs/` | Model, protocol, and semantic-prompt configuration |
| External methods | `third_party/` | Commit lock, fetch utility, and retained license texts |
| Runtime state | `data/`, `artifacts/`, `outputs/` | Inputs, derived assets, checkpoints, predictions, and logs |
| Curated results | `results/reference/` | Compact, versioned summaries and provenance records |

## Backend isolation

The TCM-suite and Ethnobotany backends retain dataset-specific configuration
and module conventions. HerbHG launches a backend in its own Python process,
with only that backend placed at the front of its import path. This provides a
single public namespace while preserving the numerical behavior of each
implementation.

Baseline methods use the same process boundary. The lightweight registry in
`src/herbhg/baselines/` records adapter modules without importing TensorFlow,
PyTorch, DGL, or author repositories during ordinary package inspection.

## Compatibility layer

The `astra` namespace contains the established experiment engine used by the
current benchmark artifacts. New user-facing entry points live under
`herbhg`; compatibility modules remain importable so existing frozen manifests
and result receipts retain their source identity.

## Dependency direction

```text
configs + data
      |
      v
herbhg public interface
      |
      +--> dataset backend process
      |
      +--> astra experiment engine --> third-party adapter process
      |
      v
artifacts + outputs --> results/reference
```
