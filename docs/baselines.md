# Comparison methods

HerbHG integrates author implementations through explicit adapters and a
shared frozen protocol. The source repositories and exact commits are listed
in `third_party/baselines.lock.json`.

## Protocol-matched methods

The public registry exposes the following comparison methods:

| Identifier | Display name | Author-code source |
|---|---|---|
| `bigi` | BiGI | BiGI repository |
| `hghda` | HGHDA | HGHDA repository |
| `lhgce` | LHGCE | LHGCE repository |
| `milgnet` | MilGNet | MilGNet repository |
| `smgcl` | SMGCL | SMGCL repository |

Adapters consume identical train, validation, and test rows from the central
manifest. They write validation and test predictions together with the
effective hyperparameters, source revision, runtime identity, and metric
summary.

## Contextual methods

`htinet-kn` and `htinet-rf` are registered separately under
`herbhg.baselines.contextual`. Keeping contextual references distinct from the
protocol-matched registry lets result aggregation state its comparison set
unambiguously.

## Local setup

Fetch all pinned author repositories with:

```bash
bash third_party/fetch_baselines.sh
```

The checkouts are created under `third_party/checkouts/` and are not part of
the HerbHG source tree. Run the preflight command before a formal comparison:

```bash
herbhg baseline preflight \
  --protocol-manifest artifacts/protocols/<dataset>/manifest.json \
  --require-ready
```

The aggregate launcher is available as:

```bash
herbhg baseline run --help
```

The corresponding compatibility modules remain directly executable under
`astra.official_baselines` for frozen workflows.
