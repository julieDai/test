#!/usr/bin/env python3
"""Repair concurrent Qwen JSONL appends while preserving full audit evidence.

The first successfully appended record for each ``(entity, row_id)`` identity
is retained. The original file is moved intact to a caller-selected backup;
the repaired file replaces the canonical path atomically, and a manifest binds
both artifacts and every dropped line number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    source = args.responses.resolve()
    backup = args.backup.resolve()
    audit = args.audit.resolve()
    if not source.is_file():
        raise SystemExit(f"response file is absent: {source}")
    for target in (backup, audit):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing audit artifact: {target}")

    retained: list[tuple[int, str, dict[str, Any]]] = []
    first_by_identity: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise SystemExit(f"line {line_number} is not a JSON object")
            identity = (str(value.get("entity")), str(value.get("row_id")))
            if identity[0] not in {"herb", "disease"} or identity[1] == "None":
                raise SystemExit(f"line {line_number} has invalid identity {identity}")
            prior = first_by_identity.get(identity)
            if prior is None:
                first_by_identity[identity] = (line_number, value)
                retained.append((line_number, raw, value))
            else:
                dropped.append(
                    {
                        "entity": identity[0],
                        "row_id": identity[1],
                        "retained_line": prior[0],
                        "dropped_line": line_number,
                        "records_identical": prior[1] == value,
                        "retained_text_sha256": prior[1].get("text_sha256"),
                        "dropped_text_sha256": value.get("text_sha256"),
                    }
                )

    if not dropped:
        raise SystemExit("response file has no duplicate identities; no repair needed")

    source.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=source.parent,
            prefix=f".{source.name}.deduplicated.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for _, raw, _ in retained:
                handle.write(raw if raw.endswith("\n") else raw + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        source.replace(backup)
        temporary.replace(source)
        temporary = None

        payload = {
            "schema": "astra-ethnobotany-qwen-duplicate-repair/v1",
            "policy": "retain_first_successfully_appended_record_per_entity_row_id",
            "reason": "two resumable generators concurrently appended overlapping pending identities",
            "original": artifact(backup),
            "repaired": artifact(source),
            "original_nonempty_rows": len(retained) + len(dropped),
            "retained_unique_rows": len(retained),
            "dropped_duplicate_rows": len(dropped),
            "conflicting_duplicate_rows": sum(
                1 for record in dropped if not record["records_identical"]
            ),
            "dropped": dropped,
        }
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({key: payload[key] for key in (
            "original_nonempty_rows",
            "retained_unique_rows",
            "dropped_duplicate_rows",
            "conflicting_duplicate_rows",
        )}, indent=2))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
