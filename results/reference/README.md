# Reference results

This directory contains compact, versioned result summaries selected from
completed HerbHG experiments. Large checkpoints, prediction tensors, feature
matrices, and logs remain under `outputs/` or external artifact storage.

A reference result directory should contain:

```text
<experiment>/<dataset>/<run_id>/
├── protocol.json
├── configuration.json
├── metrics.json
├── cross_fold_summary.json
└── provenance.json
```

`provenance.json` should record the source commit or source-manifest digest,
the central protocol digest, feature-bundle digests, and any pinned external
implementation commits. Reported cross-fold values must be derivable from the
per-fold metric artifacts retained with the complete run.
