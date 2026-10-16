#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock_file="${repo_root}/third_party/baselines.lock.json"

checkout_root="$(${PYTHON:-python3} -c '
import json
import pathlib
import sys

lock_path = pathlib.Path(sys.argv[1])
project = pathlib.Path(sys.argv[2]).resolve()
payload = json.loads(lock_path.read_text(encoding="utf-8"))
relative = pathlib.Path(payload["checkout_root"])
if relative.is_absolute() or relative == pathlib.Path(".") or ".." in relative.parts:
    raise SystemExit("checkout_root must be a non-empty repository-relative path")
resolved = (project / relative).resolve()
if project not in resolved.parents:
    raise SystemExit("checkout_root escapes the HerbHG repository")
print(resolved)
' "${lock_file}" "${repo_root}")"
mkdir -p "${checkout_root}"

fetch_at_commit() {
  local directory="$1"
  local url="$2"
  local commit="$3"
  local destination="${checkout_root}/${directory}"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --no-checkout "${url}" "${destination}"
  else
    local configured_url
    configured_url="$(git -C "${destination}" remote get-url origin)"
    if [[ "${configured_url}" != "${url}" ]]; then
      echo "origin URL mismatch for ${directory}: ${configured_url}" >&2
      exit 1
    fi
  fi
  git -C "${destination}" fetch --depth=1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"

  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "checkout verification failed for ${directory}" >&2
    exit 1
  fi
}

while IFS=$'\t' read -r directory url commit; do
  fetch_at_commit "${directory}" "${url}" "${commit}"
done < <(${PYTHON:-python3} -c '
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, record in payload["sources"].items():
    directory = pathlib.Path(record["directory"])
    if directory.is_absolute() or directory == pathlib.Path(".") or ".." in directory.parts:
        raise SystemExit(f"invalid checkout directory for {name}: {directory}")
    print("\t".join((directory.as_posix(), record["url"], record["commit"])))
' "${lock_file}")
