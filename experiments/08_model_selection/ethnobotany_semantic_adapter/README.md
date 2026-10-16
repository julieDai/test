# Ethnobotany semantic-adapter selection

Candidate files in `configs/` define linear, residual-linear, residual-MLP,
and gated-residual adapters under matched training settings. The search is
executed with:

```bash
herbhg experiment ethnobotany-semantic-adapter --help
```

Candidate identifiers and backend option keys remain stable so existing run
receipts can be verified against the migrated implementation.
