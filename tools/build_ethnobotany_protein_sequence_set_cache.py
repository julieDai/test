#!/usr/bin/env python3
"""Build the audited Ethnobotany protein sequence-set import and cache.

This tool re-adjudicates the frozen candidate snapshot without network access.
It never chooses a first or longest biologically distinct sequence.  At the
first non-empty evidence tier it collapses byte-identical amino-acid sequences
by SHA-256 and retains every distinct sequence as an equal-weight entity set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from astra.benchmark.ethnobotany_feature_bundle import (
    AUDITED_MISSING_SEQUENCE_SCHEMA,
    FROZEN_SEQUENCE_TIERS,
    SEQUENCE_SET_AGGREGATION,
    SEQUENCE_SET_RESOLUTION_SCHEMA,
    FeatureBundleError,
    _protein_candidate,
    read_entity_mapping,
    update_retrieval_cache,
)

IMPORT_SCHEMA = "astra-ethnobotany-protein-sequence-set-import/v1"
EXPECTED_COUNTS = {
    "resolved_sequence_set": 311,
    "target_not_protein": 37,
    "audited_missing_sequence": 7,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureBundleError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FeatureBundleError(f"JSON root must be an object: {path}")
    return payload


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _audit_rows(
    audit: Mapping[str, Any],
) -> dict[str, tuple[str, int, dict[str, Any]]]:
    rows: dict[str, tuple[str, int, dict[str, Any]]] = {}

    def add(section: str, values: object) -> None:
        if not isinstance(values, list):
            raise FeatureBundleError(f"audit section {section} is not a list")
        for index, raw in enumerate(values):
            if not isinstance(raw, dict):
                raise FeatureBundleError(
                    f"audit section {section}[{index}] is not an object"
                )
            symbol = str(raw.get("gene_symbol", "")).strip()
            if not symbol:
                raise FeatureBundleError(
                    f"audit section {section}[{index}] lacks gene_symbol"
                )
            if symbol in rows:
                raise FeatureBundleError(f"duplicate audit gene_symbol {symbol}")
            rows[symbol] = (section, index, raw)

    resolved = audit.get("resolved_by_tier")
    if not isinstance(resolved, Mapping):
        raise FeatureBundleError("audit resolved_by_tier is malformed")
    for tier, values in resolved.items():
        add(f"resolved_by_tier.{tier}", values)
    add("ambiguous", audit.get("ambiguous"))
    add("missing", audit.get("missing"))
    add("ineligible", audit.get("ineligible"))
    return rows


def _candidate_groups(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate["sequence_sha256"]), []).append(candidate)
    return groups


def _tier_candidates(
    tier: str, candidates: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    if tier == "mane_select":
        return [
            candidate
            for candidate in candidates
            if candidate["select_category"] == "MANE_SELECT"
        ]
    if tier == "refseq_select":
        return [
            candidate
            for candidate in candidates
            if candidate["select_category"] == "REFSEQ_SELECT"
        ]
    if tier == "reviewed_uniprot":
        return [
            candidate
            for candidate in candidates
            if candidate["source"] == "uniprot" and candidate["reviewed"]
        ]
    if tier == "known_refseq_np":
        return [
            candidate
            for candidate in candidates
            if candidate["source"] == "refseq"
            and str(candidate["accession"]).startswith("NP_")
        ]
    if tier == "unreviewed_uniprot":
        return [
            candidate
            for candidate in candidates
            if candidate["source"] == "uniprot" and not candidate["reviewed"]
        ]
    if tier == "complete_refseq":
        return [
            candidate for candidate in candidates if candidate["source"] == "refseq"
        ]
    raise FeatureBundleError(f"unsupported frozen sequence tier {tier}")


def _sequence_set_record(
    *,
    symbol: str,
    source_record: Mapping[str, Any],
    audit_location: tuple[str, int, Mapping[str, Any]],
    audit_id: str,
    audit_sha256: str,
    audit_artifact: Mapping[str, Any],
    candidate_import_artifact: Mapping[str, Any],
    mapping_sha256: str,
) -> dict[str, Any] | None:
    expected_taxid = str(source_record.get("expected_taxid", "")).strip()
    if not expected_taxid:
        raise FeatureBundleError(f"protein-coding record {symbol} lacks expected_taxid")
    raw_candidates = source_record.get("candidates")
    if not isinstance(raw_candidates, list):
        raise FeatureBundleError(f"protein-coding record {symbol} has malformed candidates")
    candidates: list[dict[str, Any]] = []
    invalid: list[str] = []
    for raw_candidate in raw_candidates:
        candidate, failure = _protein_candidate(
            raw_candidate,
            expected_taxid=expected_taxid,
        )
        if candidate is not None:
            candidates.append(candidate)
        elif failure is not None:
            invalid.append(failure)
    if invalid:
        raise FeatureBundleError(
            f"normalized strict import unexpectedly contains invalid candidates for "
            f"{symbol}: {invalid[:3]}"
        )
    if not candidates:
        return None

    tier_evidence: list[dict[str, Any]] = []
    selected_tier: str | None = None
    selected_groups: dict[str, list[Mapping[str, Any]]] = {}
    for tier in FROZEN_SEQUENCE_TIERS:
        selected = _tier_candidates(tier, candidates)
        groups = _candidate_groups(selected)
        tier_evidence.append(
            {
                "tier": tier,
                "candidate_count": len(selected),
                "sequence_group_count": len(groups),
                "sequence_sha256s": sorted(groups),
                "accessions": sorted(
                    {str(candidate["accession"]) for candidate in selected}
                ),
            }
        )
        if groups:
            selected_tier = tier
            selected_groups = groups
            break
    if selected_tier is None:
        return None

    members: list[dict[str, Any]] = []
    for digest in sorted(selected_groups):
        group = selected_groups[digest]
        sequences = {str(candidate["sequence"]) for candidate in group}
        taxids = {str(candidate["taxid"]) for candidate in group}
        if len(sequences) != 1 or taxids != {expected_taxid}:
            raise FeatureBundleError(
                f"sequence-SHA group for {symbol} is internally inconsistent"
            )
        members.append(
            {
                "sequence": next(iter(sequences)),
                "sequence_sha256": digest,
                "accessions": sorted(
                    {str(candidate["accession"]) for candidate in group}
                ),
                "taxid": expected_taxid,
                "annotation_releases": sorted(
                    {
                        str(candidate["annotation_release"]).strip()
                        for candidate in group
                        if isinstance(candidate.get("annotation_release"), str)
                        and str(candidate["annotation_release"]).strip()
                    }
                ),
            }
        )

    section, index, audit_row = audit_location
    source_gene_id = str(source_record.get("source_gene_id", "")).strip()
    if not source_gene_id:
        raise FeatureBundleError(f"protein-coding record {symbol} lacks source_gene_id")
    action = "single_sequence" if len(members) == 1 else "equal_weight_mean"
    evidence = {
        "schema": SEQUENCE_SET_RESOLUTION_SCHEMA,
        "mapping_sha256": mapping_sha256,
        "audit_id": audit_id,
        "source": "Ethnobotany sequence-resolution audit",
        "gene_symbol": symbol,
        "source_gene_id": source_gene_id,
        "current_gene_id": source_record.get("current_gene_id"),
        "expected_taxid": expected_taxid,
        "audit_artifact_sha256": audit_sha256,
        "audit_artifact": dict(audit_artifact),
        "audit_section": section,
        "audit_row_index": index,
        "audit_row": dict(audit_row),
        "audit_row_sha256": _value_sha256(audit_row),
        "audit_row_hash_basis": (
            "SHA-256 of the canonical UTF-8 JSON object at audit_section/"
            "audit_row_index (sorted keys, compact separators)"
        ),
        "candidate_import_record_sha256": _value_sha256(source_record),
        "candidate_import_record": dict(source_record),
        "candidate_import_artifact": dict(candidate_import_artifact),
        "selected_tier": selected_tier,
        "resolution_hierarchy": list(FROZEN_SEQUENCE_TIERS),
        "tier_evidence": tier_evidence,
        "representation_action": action,
    }
    return {
        "status": "resolved_sequence_set",
        "expected_taxid": expected_taxid,
        "source_gene_id": source_gene_id,
        "current_gene_id": source_record.get("current_gene_id"),
        "gene_type": "PROTEIN_CODING",
        "organism": source_record.get("organism"),
        "sequence_set_tier": selected_tier,
        "sequence_set_action": action,
        "sequence_set_aggregation": SEQUENCE_SET_AGGREGATION,
        "sequence_set_count": len(members),
        "sequence_set": members,
        "sequence_resolution_evidence": evidence,
    }


def _audited_missing_record(
    *,
    symbol: str,
    source_record: Mapping[str, Any],
    audit_location: tuple[str, int, Mapping[str, Any]],
    audit_id: str,
    audit_sha256: str,
    audit_artifact: Mapping[str, Any],
    candidate_import_artifact: Mapping[str, Any],
    mapping_sha256: str,
    queried_at: str,
) -> dict[str, Any]:
    section, index, audit_row = audit_location
    gene_type = str(audit_row.get("gene_type", source_record.get("gene_type", "")))
    if gene_type == "UNKNOWN":
        target_class = "unknown_target_class"
        resolution_status = "ineligible"
    elif gene_type == "PROTEIN_CODING":
        target_class = "protein_coding_no_valid_sequence"
        resolution_status = "missing"
    else:
        raise FeatureBundleError(
            f"audited missing row {symbol} has unsupported gene_type {gene_type!r}"
        )
    source_gene_id = str(audit_row.get("source_gene_id", "")).strip()
    taxid = str(audit_row.get("taxid", "")).strip()
    reason = str(audit_row.get("reason", "")).strip()
    if not source_gene_id or not taxid or not reason:
        raise FeatureBundleError(f"audited missing row {symbol} lacks exact evidence")
    current_gene_id = audit_row.get("current_gene_id")
    evidence = {
        "schema": AUDITED_MISSING_SEQUENCE_SCHEMA,
        "mapping_sha256": mapping_sha256,
        "audit_id": audit_id,
        "source": "Ethnobotany sequence-resolution audit",
        "gene_symbol": symbol,
        "source_gene_id": source_gene_id,
        "current_gene_id": (
            str(current_gene_id).strip() if current_gene_id is not None else None
        ),
        "taxid": taxid,
        "gene_type": gene_type,
        "resolution_status": resolution_status,
        "queried_at": queried_at,
        "audit_artifact_sha256": audit_sha256,
        "audit_artifact": dict(audit_artifact),
        "audit_section": section,
        "audit_row_index": index,
        "audit_row": dict(audit_row),
        "audit_row_sha256": _value_sha256(audit_row),
        "audit_row_hash_basis": (
            "SHA-256 of the canonical UTF-8 JSON object at audit_section/"
            "audit_row_index (sorted keys, compact separators)"
        ),
        "candidate_import_record_sha256": _value_sha256(source_record),
        "candidate_import_record": dict(source_record),
        "candidate_import_artifact": dict(candidate_import_artifact),
    }
    return {
        "status": "audited_missing_sequence",
        "target_class": target_class,
        "reason": reason,
        "source_gene_id": source_gene_id,
        "current_gene_id": evidence["current_gene_id"],
        "gene_type": gene_type,
        "taxid": taxid,
        "organism": audit_row.get("organism"),
        "annotation_release": queried_at,
        "missing_sequence_evidence": evidence,
    }


def _target_not_protein_record(
    *,
    symbol: str,
    source_record: Mapping[str, Any],
    audit_location: tuple[str, int, Mapping[str, Any]],
    audit_sha256: str,
    audit_artifact: Mapping[str, Any],
    candidate_import_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach independently recomputable audit-row evidence to a zero-UNK row."""

    record = dict(source_record)
    raw_evidence = record.get("classification_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise FeatureBundleError(
            f"target-not-protein record {symbol} lacks classification_evidence"
        )
    section, index, audit_row = audit_location
    evidence = dict(raw_evidence)
    evidence.update(
        {
            "audit_artifact_sha256": audit_sha256,
            "audit_artifact": dict(audit_artifact),
            "audit_section": section,
            "audit_row_index": index,
            "audit_row": dict(audit_row),
            "audit_row_sha256": _value_sha256(audit_row),
            "audit_row_hash_basis": (
                "SHA-256 of the canonical UTF-8 JSON object at audit_section/"
                "audit_row_index (sorted keys, compact separators)"
            ),
            "candidate_import_record_sha256": _value_sha256(source_record),
            "candidate_import_record": dict(source_record),
            "candidate_import_artifact": dict(candidate_import_artifact),
        }
    )
    record["classification_evidence"] = evidence
    return record


def build_import(
    *,
    mapping_path: Path,
    strict_import_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    mapping = read_entity_mapping(mapping_path, "protein")
    strict_import = _read_json(strict_import_path)
    audit = _read_json(audit_path)
    records = strict_import.get("records")
    if not isinstance(records, Mapping):
        raise FeatureBundleError("strict import records must be an object")
    if set(records) != set(mapping.source_ids):
        raise FeatureBundleError(
            "strict import source IDs do not exactly equal the protein mapping"
        )
    audit_id = str(audit.get("audit_id", "")).strip()
    queried_at = str(audit.get("query_completed_at", "")).strip()
    if not audit_id or not queried_at:
        raise FeatureBundleError("audit lacks ID or completion timestamp")
    audit_sha256 = _file_sha256(audit_path)
    audit_artifact = _artifact(audit_path)
    candidate_import_artifact = _artifact(strict_import_path)
    audit_rows = _audit_rows(audit)
    if set(audit_rows) != set(mapping.source_ids):
        raise FeatureBundleError(
            "audit entity rows do not exactly equal the protein mapping"
        )

    output_records: dict[str, dict[str, Any]] = {}
    selected_tiers: Counter[str] = Counter()
    set_sizes: Counter[int] = Counter()
    actions: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    missing_classes: Counter[str] = Counter()
    distinct_sequence_count = 0
    selected_accession_count = 0

    for row_id, symbol in zip(mapping.row_ids, mapping.source_ids):
        raw = records[symbol]
        if not isinstance(raw, Mapping):
            raise FeatureBundleError(f"strict import record {symbol} is not an object")
        audit_location = audit_rows[symbol]
        gene_type = str(raw.get("gene_type", ""))
        if gene_type == "PROTEIN_CODING":
            record = _sequence_set_record(
                symbol=symbol,
                source_record=raw,
                audit_location=audit_location,
                audit_id=audit_id,
                audit_sha256=audit_sha256,
                audit_artifact=audit_artifact,
                candidate_import_artifact=candidate_import_artifact,
                mapping_sha256=mapping.sha256,
            )
            if record is None:
                record = _audited_missing_record(
                    symbol=symbol,
                    source_record=raw,
                    audit_location=audit_location,
                    audit_id=audit_id,
                    audit_sha256=audit_sha256,
                    audit_artifact=audit_artifact,
                    candidate_import_artifact=candidate_import_artifact,
                    mapping_sha256=mapping.sha256,
                    queried_at=queried_at,
                )
            else:
                selected_tiers[str(record["sequence_set_tier"])] += 1
                set_sizes[int(record["sequence_set_count"])] += 1
                actions[str(record["sequence_set_action"])] += 1
                distinct_sequence_count += int(record["sequence_set_count"])
                selected_accession_count += sum(
                    len(member["accessions"])
                    for member in record["sequence_set"]
                )
        elif raw.get("status") == "target_not_protein":
            record = _target_not_protein_record(
                symbol=symbol,
                source_record=raw,
                audit_location=audit_location,
                audit_sha256=audit_sha256,
                audit_artifact=audit_artifact,
                candidate_import_artifact=candidate_import_artifact,
            )
        elif gene_type == "UNKNOWN":
            record = _audited_missing_record(
                symbol=symbol,
                source_record=raw,
                audit_location=audit_location,
                audit_id=audit_id,
                audit_sha256=audit_sha256,
                audit_artifact=audit_artifact,
                candidate_import_artifact=candidate_import_artifact,
                mapping_sha256=mapping.sha256,
                queried_at=queried_at,
            )
        else:
            raise FeatureBundleError(
                f"record {row_id}:{symbol} has unsupported class/status"
            )
        output_records[symbol] = record
        statuses[str(record["status"])] += 1
        if record["status"] == "audited_missing_sequence":
            missing_classes[str(record["target_class"])] += 1

    if dict(statuses) != EXPECTED_COUNTS:
        raise FeatureBundleError(
            f"unexpected output status counts: {dict(statuses)} != {EXPECTED_COUNTS}"
        )
    if missing_classes != Counter(
        {"unknown_target_class": 6, "protein_coding_no_valid_sequence": 1}
    ):
        raise FeatureBundleError(
            f"unexpected audited missing classes: {dict(missing_classes)}"
        )

    return {
        "schema": IMPORT_SCHEMA,
        "audit_id": audit_id,
        "source_audit_completed_at": queried_at,
        "metadata": {
            "purpose": (
                "Formal, identity-bound Ethnobotany protein sequence-set import. "
                "All distinct sequences at the first non-empty frozen evidence tier "
                "are retained; no first/longest accession tie-break is used."
            ),
            "mapping": _artifact(mapping_path),
            "strict_candidate_import": _artifact(strict_import_path),
            "sequence_resolution_audit": _artifact(audit_path),
            "policy": {
                "resolution_hierarchy": list(FROZEN_SEQUENCE_TIERS),
                "tier_rule": (
                    "At the first non-empty tier, deduplicate candidates only by "
                    "exact amino-acid sequence SHA-256."
                ),
                "single_sequence_action": "single_sequence",
                "multiple_sequence_action": "equal_weight_mean",
                "cache_aggregation_contract": SEQUENCE_SET_AGGREGATION,
                "forbidden_tiebreaks": [
                    "first result",
                    "longest sequence",
                    "lexicographically first biologically distinct accession",
                ],
                "known_refseq_interpretation": (
                    "NP_ is an explicitly disclosed known-RefSeq provenance tier; "
                    "the prefix alone is not called canonical, principal, Select, "
                    "VALIDATED, or REVIEWED."
                ),
            },
            "audit_row_hash_contract": (
                "Each evidence object's audit_row_sha256 hashes the exact JSON object "
                "at its recorded audit_section/audit_row_index using sorted keys, "
                "compact separators, UTF-8, and ensure_ascii=false."
            ),
            "counts": {
                **EXPECTED_COUNTS,
                "selected_tiers": dict(sorted(selected_tiers.items())),
                "sequence_set_actions": dict(sorted(actions.items())),
                "sequence_set_size_distribution": {
                    str(key): value for key, value in sorted(set_sizes.items())
                },
                "distinct_sequence_count": distinct_sequence_count,
                "selected_accession_count": selected_accession_count,
                "audited_missing_classes": dict(sorted(missing_classes.items())),
            },
            "official_accession_semantics": [
                {
                    "url": "https://www.ncbi.nlm.nih.gov/refseq/about/",
                    "use": (
                        "NCBI defines NP_ as known RefSeq and XP_ as model RefSeq; "
                        "this supports provenance ordering, not a canonical claim."
                    ),
                },
                {
                    "url": (
                        "https://www.ncbi.nlm.nih.gov/sites/books/NBK21101/pdf/"
                        "Bookshelf_NBK21101.pdf"
                    ),
                    "use": "NCBI RefSeq accession-prefix semantics.",
                },
            ],
        },
        "records": output_records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--strict-import", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-import", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = {
        "mapping": args.mapping.resolve(),
        "strict_import": args.strict_import.resolve(),
        "audit": args.audit.resolve(),
        "output_import": args.output_import.resolve(),
        "output_cache": args.output_cache.resolve(),
    }
    if paths["output_import"] in {
        paths["mapping"], paths["strict_import"], paths["audit"]
    } or paths["output_cache"] in {
        paths["mapping"], paths["strict_import"], paths["audit"]
    }:
        raise FeatureBundleError("output paths must not overwrite frozen inputs")
    for key in ("output_import", "output_cache"):
        if paths[key].exists() and not args.force:
            raise FeatureBundleError(
                f"refusing to overwrite existing {key}: {paths[key]}"
            )
        if paths[key].exists() and args.force:
            if not paths[key].is_file():
                raise FeatureBundleError(
                    f"refusing to replace non-file {key}: {paths[key]}"
                )
            paths[key].unlink()

    payload = build_import(
        mapping_path=paths["mapping"],
        strict_import_path=paths["strict_import"],
        audit_path=paths["audit"],
    )
    _atomic_json(paths["output_import"], payload)
    cache = update_retrieval_cache(
        entity="protein",
        mapping_path=paths["mapping"],
        output_path=paths["output_cache"],
        import_path=paths["output_import"],
        missing_sequence_policy="audited-missing-zero",
        ambiguous_sequence_policy="mean-distinct-first-tier",
        require_complete=True,
    )
    expected_cache_counts = {
        "resolved": 0,
        "resolved_sequence_set": 311,
        "target_not_protein": 37,
        "audited_missing_sequence": 7,
        "pending": 0,
        "failed": 0,
    }
    if cache["status_counts"] != expected_cache_counts or not cache["formal_eligible"]:
        raise FeatureBundleError(
            f"generated cache failed its count/formal contract: "
            f"{cache['status_counts']} formal={cache['formal_eligible']}"
        )
    print(
        json.dumps(
            {
                "import": _artifact(paths["output_import"]),
                "cache": _artifact(paths["output_cache"]),
                "status_counts": cache["status_counts"],
                "formal_eligible": cache["formal_eligible"],
                "resolved_sequence_count": cache["resolved_sequence_count"],
                "masked_missing_count": cache["masked_missing_count"],
                "import_counts": payload["metadata"]["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
