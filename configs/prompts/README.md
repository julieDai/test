# Prompt configurations

This directory contains the prompt templates used to construct semantic
features for herbs and diseases.

- `original/` contains the prompts used for the original semantic-feature
  generation workflow.
- `target_neutral/` contains the controlled prompts used to generate intrinsic
  descriptions without directly naming the opposite prediction endpoint.

Templates use `[Herb]` and `[Disease]` as substitution fields. Generated text,
model responses, API credentials, and provider-specific request metadata are
runtime artifacts and are not stored here.

`manifest.json` assigns stable condition names and records the SHA-256 digest
of every template.
