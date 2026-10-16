# Model selection

Model-selection experiments use only the inner validation rows of each outer
fold. The chosen configuration is frozen before the corresponding outer test
evaluation.

Available public launchers include:

```bash
herbhg experiment ethnobotany-alpha --help
herbhg experiment ethnobotany-refined --help
herbhg experiment ethnobotany-semantic-adapter --help
herbhg experiment tcm-epoch-boundary --help
herbhg experiment tcm-hyperparameters --help
herbhg experiment tcm-pca64-control --help
herbhg experiment tcm-semantic-adapter --help
```

Dataset-specific notes and frozen candidate definitions are kept in the two
subdirectories.
