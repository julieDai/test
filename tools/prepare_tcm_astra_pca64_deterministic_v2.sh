#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s PARENT_ROOT PROTOCOL_MANIFEST SOURCE_BUNDLE_MANIFEST OUTPUT_ROOT\n' "$0" >&2
  exit 2
fi

parent_root=$1
protocol_manifest=$2
source_bundle_manifest=$3
output_root=$4
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}

# PARENT_ROOT is the completed semantic_adapter_deterministic_v3 directory.
parent_plan="${parent_root}/plan.json"
parent_selection="${parent_root}/selection.json"
parent_completion="${parent_root}/.astra_semantic_adapter_deterministic_v3_complete"

for required in "${parent_plan}" "${parent_selection}" "${parent_completion}"; do
  [[ -f "${required}" ]] || {
    printf 'missing prerequisite: %s\n' "${required}" >&2
    exit 3
  }
done

for fold in 1 2 3 4 5; do
  receipt="${parent_root}/pair_receipts/fold_${fold}.json"
  [[ -f "${receipt}" ]] || {
    printf 'missing fold receipt: %s\n' "${receipt}" >&2
    exit 3
  }
done

pca64_bundle="${output_root}/pca64_bundle_v1"
control_plan="${output_root}/plan.json"
[[ ! -e "${pca64_bundle}" && ! -e "${control_plan}" ]] || {
  printf 'refusing to overwrite an existing PCA64 bundle or plan\n' >&2
  exit 4
}

PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -m astra.benchmark.tcm_pca64_feature_bundle \
  --protocol-manifest "${protocol_manifest}" \
  --source-bundle-manifest "${source_bundle_manifest}" \
  --output-dir "${pca64_bundle}"

PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -m astra.benchmark.tcm_astra_pca64_deterministic_plan \
  --adapter-selection "${parent_selection}" \
  --pca64-bundle-manifest "${pca64_bundle}/bundle_manifest.json" \
  --output "${control_plan}"

printf 'wrote %s and %s\n' "${pca64_bundle}" "${control_plan}"
