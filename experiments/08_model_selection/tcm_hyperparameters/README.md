# TCM-suite hyperparameter selection

The TCM-suite selection modules cover learning rate, epoch boundary, PCA-64
controls, and deterministic semantic-adapter candidates:

```bash
herbhg experiment tcm-hyperparameters --help
herbhg experiment tcm-epoch-boundary --help
herbhg experiment tcm-pca64-control --help
herbhg experiment tcm-semantic-adapter --help
```

All selectors write their candidate scores, tie-break decisions, and frozen
fold choices to the requested output directory.
