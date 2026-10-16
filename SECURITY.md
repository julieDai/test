# Security Policy

## Supported versions

Security updates are applied to the current development branch. Research
snapshots and third-party repositories retain their pinned upstream versions
for reproducibility.

## Reporting a vulnerability

Please use the repository host's private vulnerability-reporting channel. A
report should include the affected file or command, a minimal reproduction,
the expected impact, and any suggested mitigation. Do not include private
datasets, credentials, access tokens, or unpublished participant information.

Reports concerning code fetched into `third_party/checkouts/` should also be
sent to the corresponding upstream project.

## Research data

Keep datasets, generated embeddings, model checkpoints, and credentials out
of version control. The default ignore rules cover the standard local asset
directories; contributors should inspect staged files before every commit.
