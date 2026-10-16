# Pinned baseline implementations

HerbHG evaluates comparison methods through project-maintained adapters and
commit-pinned author repositories. The exact revisions are recorded in
`baselines.lock.json`:

| Method | Commit | Publication |
|---|---|---|
| SMGCL | `09fa09795bc610018f906d26bea5b4259233400a` | `10.1093/bioinformatics/btad357` |
| MilGNet | `0149c707e70b039e11ba8c5b88b869aed38e152d` | `10.1109/BIBM55620.2022.9995152` |
| LHGCE | `eeedb4cd6cf64166bd0f8a0ebc2cca96c60874dc` | `10.1016/j.imu.2023.101177` |
| BiGI | `ed54c20523a5b3f295b90a9c08f7c54e8258d04a` | `10.1145/3437963.3441783` |
| HGHDA | `c176902f4206f45c2d23bc1c684eeb0c229984cb` | `10.1093/bib/bbae067` |

Run `bash third_party/fetch_baselines.sh` from any directory to create detached
checkouts under `third_party/checkouts/`. The checkout directory is a local
dependency and is excluded from source distributions.

The adapters under `src/herbhg/baselines/` expose stable method identifiers.
The compatibility implementations under `src/astra/official_baselines/`
provide protocol loading, model-specific integration, and result
serialization. Every comparison run receives the same explicit protocol
manifest; adapters do not create private outer-test splits.

License texts available from the pinned repositories are retained under
`third_party/licenses/`. See `THIRD_PARTY_NOTICES.md` for attribution and
redistribution information.
