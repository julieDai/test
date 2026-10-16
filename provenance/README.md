# Source provenance

`origins.json` records the repositories and base commits used to assemble this
source tree. `migration_map.tsv` maps retained implementation areas to their
HerbHG locations. `source_manifest.json` provides a deterministic SHA-256
inventory of versioned source material.

Regenerate the inventory after an intentional source change:

```bash
python tools/build_source_manifest.py
```

Verify a checkout against the recorded inventory:

```bash
python tools/build_source_manifest.py --check
```

The manifest excludes generated data, outputs, model artifacts, local
third-party checkouts, virtual environments, caches, and the manifest itself.
