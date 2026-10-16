# Main comparison

Main comparisons use one immutable protocol package per dataset. The package
contains compact entity mappings, row identifiers, train/validation/test
indices, input hashes, and five outer folds. Candidate selection uses only the
inner validation subset; each frozen candidate is evaluated once on its outer
test fold.

See the dataset-specific recipes in `tcm_suite/` and `ethnobotany/`.
