# Reference results

The repository versions one compact experimental-results snapshot:

- [`manuscript_current/`](manuscript_current/) contains the values used by the
  current HerbHG manuscript, identified by the manuscript SHA-256 digest in
  its provenance record.

The snapshot preserves manuscript-facing tables and the compact evidence
needed to identify their source runs. Candidate sweeps and superseded result
sets are intentionally excluded. Large checkpoints, predictions, feature
matrices, and logs remain outside version control under `outputs/`,
`artifacts/`, or archival storage.

Model and protocol configurations elsewhere in the repository are executable
research inputs rather than additional result sets.
