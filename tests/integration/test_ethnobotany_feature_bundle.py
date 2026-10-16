from __future__ import annotations

import json
import io
import hashlib
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astra.benchmark.ethnobotany_feature_bundle import (
    BUNDLE_SCHEMA,
    CACHE_SCHEMA,
    EMBEDDING_SHARD_SCHEMA,
    EXTRACTION_MANIFEST_SCHEMA,
    TARGET_CLASSIFICATION_SCHEMA,
    FeatureBundleError,
    IncompleteFeatureBundleError,
    _validate_biological_model_provenance,
    assemble_embedding_shards,
    finalize_feature_bundle,
    main,
    reduce_embedding_assembly,
    update_retrieval_cache,
)
from astra.benchmark.features import feature_row_ids_path, load_aligned_feature_matrix
from astra.benchmark.astra_runner import (
    ASTRARunnerError,
    _validate_ethnobotany_feature_bundle,
)


def _write_mapping(root: Path, entity: str, count: int) -> Path:
    path = root / f"{entity}_id_mapping.csv"
    lines = [f"{entity}_id,{entity}_source_id"]
    lines.extend(f"{index},{entity}-source-{index}" for index in range(1, count + 1))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_shard(
    path: Path,
    row_ids: list[str],
    embeddings: np.ndarray,
    *,
    failures: dict[str, str] | None = None,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "row_ids": np.asarray(row_ids, dtype=str),
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }
    if failures is not None:
        arrays["failed_row_ids"] = np.asarray(list(failures), dtype=str)
        arrays["failure_reasons"] = np.asarray(list(failures.values()), dtype=str)
    np.savez(path, **arrays)
    return path


def _artifact_for_test(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_for_test(path),
    }


def _target_not_protein_record(
    source_id: str,
    mapping: Path,
    *,
    target_class: str = "ncrna",
    gene_type: str = "ncRNA",
) -> dict[str, object]:
    reason = f"NCBI Gene classifies {source_id} as {gene_type}"
    audit_row = {
        "gene_symbol": source_id,
        "source_gene_id": "1001",
        "current_gene_id": "1001",
        "taxid": "9606",
        "gene_type": gene_type,
        "status": "ineligible",
        "reason": reason,
    }
    audit_path = mapping.parent / (
        "unit_test_audit_" + hashlib.sha256(source_id.encode()).hexdigest()[:12] + ".json"
    )
    _write_json(audit_path, {"ineligible": [audit_row]})
    candidate_record = {
        "status": "target_not_protein",
        "gene_type": gene_type,
        "source_gene_id": "1001",
        "current_gene_id": "1001",
        "taxid": "9606",
    }
    candidate_path = mapping.parent / (
        "unit_test_candidates_"
        + hashlib.sha256(source_id.encode()).hexdigest()[:12]
        + ".json"
    )
    _write_json(candidate_path, {"records": {source_id: candidate_record}})
    return {
        "status": "target-not-protein",
        "target_class": target_class,
        "reason": reason,
        "classification_evidence": {
            "schema": TARGET_CLASSIFICATION_SCHEMA,
            "mapping_sha256": _sha256_for_test(mapping),
            "source": "NCBI Datasets Gene report",
            "audit_id": "unit-test-ncbi-gene-audit",
            "gene_symbol": source_id,
            "source_gene_id": "1001",
            "current_gene_id": "1001",
            "taxid": "9606",
            "gene_type": gene_type,
            "annotation_release": "unit-test-release",
            "source_record_sha256": "a" * 64,
            "audit_artifact_sha256": _sha256_for_test(audit_path),
            "audit_artifact": _artifact_for_test(audit_path),
            "audit_section": "ineligible",
            "audit_row_index": 0,
            "audit_row": audit_row,
            "audit_row_sha256": hashlib.sha256(
                json.dumps(
                    audit_row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "candidate_import_record_sha256": hashlib.sha256(
                json.dumps(
                    candidate_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "candidate_import_record": candidate_record,
            "candidate_import_artifact": _artifact_for_test(candidate_path),
        },
    }


def _mapping_ids(mapping: Path) -> list[str]:
    return [
        line.split(",", 1)[0]
        for line in mapping.read_text(encoding="utf-8").splitlines()[1:]
    ]


def _write_formal_biological_shard(
    root: Path,
    *,
    entity: str,
    mapping: Path,
    cache: Path,
    start: int,
    stop: int,
    embeddings_by_row: dict[str, np.ndarray],
    failures: dict[str, str] | None = None,
    label: str = "shard",
) -> Path:
    """Write a locally self-consistent v2 formal shard fixture.

    Live source-tree/runtime attestation is patched in this unit-test module;
    the bundle's independent recursive validator still checks every recorded
    artifact, range, row, hash, dimension, and extraction-manifest contract.
    """

    failures = failures or {}
    row_ids = _mapping_ids(mapping)
    expected = row_ids[start:stop]
    if set(embeddings_by_row).union(failures) != set(expected):
        raise AssertionError("fixture shard does not exactly cover its range")
    dimension = 2304 if entity == "compound" else 640
    successful = [row_id for row_id in expected if row_id in embeddings_by_row]
    failed = [row_id for row_id in expected if row_id in failures]
    matrix = np.stack(
        [np.asarray(embeddings_by_row[row_id], dtype=np.float32) for row_id in successful]
    ) if successful else np.empty((0, dimension), dtype=np.float32)
    if matrix.shape != (len(successful), dimension):
        raise AssertionError("fixture uses the wrong raw extractor dimension")
    shard_dir = root / f"{entity}_{label}_{start}_{stop}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"{entity}_{start:06d}_{stop - 1:06d}.npz"
    np.savez(
        shard,
        row_ids=np.asarray(successful, dtype=str),
        embeddings=matrix,
        failed_row_ids=np.asarray(failed, dtype=str),
        failure_reasons=np.asarray([failures[row_id] for row_id in failed], dtype=str),
    )
    checkpoint = shard_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"unit-test-checkpoint")
    checkpoint_artifact = _artifact_for_test(checkpoint)
    if entity == "compound":
        worker = shard_dir / "kpgt_worker.py"
        worker.write_text("# unit-test worker\n", encoding="utf-8")
        worker_artifact = _artifact_for_test(worker)
        extractor = "kpgt-base-generate-fps"
        provenance: dict[str, object] = {
            "implementation": "official KPGT LiGhT generate_fps",
            "config": "base",
            "repository": str((shard_dir / "KPGT").resolve()),
            "repository_commit": "47dc1646c70b2138a157de481d24a1ac35d174cd",
            "expected_repository_commit": "47dc1646c70b2138a157de481d24a1ac35d174cd",
            "repository_tree_sha256": "b" * 64,
            "checkpoint": checkpoint_artifact,
            "expected_checkpoint_sha256": checkpoint_artifact["sha256"],
            "worker": worker_artifact,
            "runtime": {
                "python_version": "3.7.12",
                "packages": {"torch": "1.10.0", "dgl": "0.7.2", "rdkit": "2021.09.5"},
                "worker_sha256": worker_artifact["sha256"],
            },
            "expected_runtime": {
                "python_version": "3.7.12",
                "torch": "1.10.0",
                "dgl": "0.7.2",
                "rdkit": "2021.09.5",
            },
            "raw_dimension": dimension,
        }
    else:
        extractor = "esm2_t30_150M_UR50D"
        versions = {"fair-esm": "2.0.0", "torch": "unit-test", "numpy": np.__version__}
        provenance = {
            "implementation": "fair-esm local model loader",
            "model_name": extractor,
            "representation_layer": 30,
            "source_tree_root": str((shard_dir / "fair_esm").resolve()),
            "source_tree_sha256": "c" * 64,
            "package_version": "2.0.0",
            "dependency_versions": versions,
            "runtime": {"packages": versions},
            "checkpoint": checkpoint_artifact,
            "expected_checkpoint_sha256": checkpoint_artifact["sha256"],
            "sequence_set_aggregation": "equal_weight_mean_of_distinct_sequence_embeddings",
            "raw_dimension": dimension,
        }
    contract: dict[str, object] = {
        "schema": EMBEDDING_SHARD_SCHEMA,
        "extractor": extractor,
        "entity": entity,
        "mapping_sha256": _sha256_for_test(mapping),
        "retrieval_cache_sha256": _sha256_for_test(cache),
        "row_start": start,
        "row_stop": stop,
        "row_count": stop - start,
        "batch_size": max(1, stop - start),
        "input_dimension": dimension,
        "embedding_dimension": dimension,
        "provenance": provenance,
        "success_count": len(successful),
        "failure_count": len(failed),
        "row_ids": successful,
        "failure_row_ids": failed,
        "artifact": _artifact_for_test(shard),
    }
    if entity == "protein":
        contract["sequence_set_rows"] = []
    sidecar = shard.with_suffix(".manifest.json")
    _write_json(sidecar, contract)
    extraction: dict[str, object] = {
        "schema": EXTRACTION_MANIFEST_SCHEMA,
        "extractor": extractor,
        "entity": entity,
        "mapping_sha256": _sha256_for_test(mapping),
        "mapping": _artifact_for_test(mapping),
        "retrieval_cache": _artifact_for_test(cache),
        "retrieval_cache_formal_eligible": True,
        "range": [start, stop],
        "shard_size": stop - start,
        "batch_size": max(1, stop - start),
        "provenance": provenance,
        "success_count": len(successful),
        "failure_count": len(failed),
        "shards": [{"sidecar": _artifact_for_test(sidecar), "contract": contract}],
    }
    if entity == "protein":
        extraction["sequence_set_rows"] = []
    _write_json(shard_dir / "extraction_manifest.json", extraction)
    return shard


def _protein_candidate(
    accession: str,
    sequence: str,
    *,
    source: str = "refseq",
    select_category: str | None = None,
    reviewed: bool = False,
    taxid: int = 9606,
) -> dict:
    candidate = {
        "accession": accession,
        "sequence": sequence,
        "reported_length": len(sequence),
        "source": source,
        "reviewed": reviewed,
        "taxid": taxid,
        "annotation_release": "test-release-1",
        "partial": False,
        "low_quality": False,
    }
    if select_category is not None:
        candidate["select_category"] = select_category
    return candidate


def _protein_record(candidates: list[dict], *, taxid: int = 9606) -> dict:
    return {"expected_taxid": taxid, "candidates": candidates}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class EthnobotanyFeatureBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="astra_ethno_bundle_")
        self.root = Path(self.temporary.name)
        self.formal_source_patch = patch(
            "astra.benchmark.ethnobotany_feature_extractors.validate_formal_extractor_shard",
            side_effect=lambda **kwargs: _read(Path(kwargs["sidecar_path"])),
        )
        self.formal_source_validator = self.formal_source_patch.start()

    def tearDown(self) -> None:
        self.formal_source_patch.stop()
        self.temporary.cleanup()

    def test_protein_cache_never_picks_first_ambiguous_candidate(self) -> None:
        mapping = _write_mapping(self.root, "protein", 5)
        imported = {
            "protein-source-1": _protein_record(
                [_protein_candidate("P1-A", "MAAA")]
            ),
            "protein-source-2": _protein_record(
                [
                    _protein_candidate("P2-A", "MCCC"),
                    _protein_candidate(
                        "P2-B", "MDDD", select_category="MANE_SELECT"
                    ),
                ]
            ),
            "protein-source-3": _protein_record(
                [
                    _protein_candidate("P3-A", "MDDD"),
                    _protein_candidate("P3-B", "MEEE"),
                ]
            ),
            "protein-source-4": _target_not_protein_record(
                "protein-source-4", mapping
            ),
            "protein-source-5": _protein_record(
                [
                    _protein_candidate("P5-A", "MFFF"),
                    _protein_candidate("P5-B", "MGGG"),
                ]
            ),
        }
        imported_path = _write_json(self.root / "protein_import.json", imported)
        allowlist_path = _write_json(
            self.root / "allowlist.json", {"protein-source-5": "P5-B"}
        )
        cache_path = self.root / "protein_cache.json"

        strict = update_retrieval_cache(
            entity="protein",
            mapping_path=mapping,
            output_path=cache_path,
            import_path=imported_path,
            accession_allowlist=allowlist_path,
        )

        self.assertEqual(strict["schema"], CACHE_SCHEMA)
        self.assertFalse(strict["formal_eligible"])
        self.assertEqual(strict["status_counts"]["resolved"], 3)
        self.assertEqual(strict["status_counts"]["target_not_protein"], 1)
        self.assertEqual(strict["status_counts"]["failed"], 1)
        rows = {row["row_id"]: row for row in strict["rows"]}
        self.assertEqual(rows["2"]["accession"], "P2-B")
        self.assertEqual(rows["2"]["resolution_rule"], "mane_select")
        self.assertEqual(rows["2"]["select_category"], "MANE_SELECT")
        self.assertEqual(len(rows["2"]["sequence_sha256"]), 64)
        self.assertEqual(rows["5"]["accession"], "P5-B")
        self.assertEqual(rows["5"]["resolution_rule"], "accession_allowlist")
        self.assertEqual(rows["3"]["status"], "failed")
        self.assertIn("ambiguous protein candidates", rows["3"]["error"])
        with self.assertRaises(IncompleteFeatureBundleError):
            update_retrieval_cache(
                entity="protein",
                mapping_path=mapping,
                output_path=cache_path,
                require_complete=True,
            )

        repaired_import = dict(imported)
        repaired_import["protein-source-3"] = _protein_record(
            [_protein_candidate("P3-A", "MDDD")]
        )
        repaired_path = _write_json(self.root / "protein_repaired.json", repaired_import)
        resumed = update_retrieval_cache(
            entity="protein",
            mapping_path=mapping,
            output_path=cache_path,
            import_path=repaired_path,
            accession_allowlist=allowlist_path,
            missing_sequence_policy="zero-unk",
            require_complete=True,
        )
        self.assertTrue(resumed["formal_eligible"])
        self.assertEqual(resumed["target_not_protein_count"], 1)
        self.assertEqual(resumed["resolved_sequence_count"], 4)
        self.assertAlmostEqual(resumed["overall_sequence_resolution_rate"], 0.8)
        self.assertAlmostEqual(resumed["protein_target_resolution_rate"], 1.0)

    def test_cli_saves_incomplete_cache_state_before_failing_closed(self) -> None:
        mapping = _write_mapping(self.root, "compound", 2)
        imported = _write_json(
            self.root / "one_smiles.json", {"compound-source-1": "CCO"}
        )
        cache_path = self.root / "cli_cache.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main(
                [
                    "cache",
                    "--entity",
                    "compound",
                    "--mapping",
                    str(mapping),
                    "--output",
                    str(cache_path),
                    "--import-json",
                    str(imported),
                    "--require-complete",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertTrue(cache_path.is_file())
        payload = _read(cache_path)
        self.assertFalse(payload["formal_eligible"])
        self.assertEqual(payload["status_counts"]["pending"], 1)

    def test_protein_resolution_hierarchy_and_sequence_equivalence_are_recorded(
        self,
    ) -> None:
        mapping = _write_mapping(self.root, "protein", 4)
        partial = _protein_candidate("R4-BAD", "MQQQ")
        partial["partial"] = True
        imported = {
            "protein-source-1": _protein_record(
                [
                    _protein_candidate("R1-A", "MAAA"),
                    _protein_candidate(
                        "R1-B", "MCCC", select_category="REFSEQ_SELECT"
                    ),
                ]
            ),
            "protein-source-2": _protein_record(
                [
                    _protein_candidate("R2-A", "MDDD"),
                    _protein_candidate(
                        "U2-A", "MEEE", source="uniprot", reviewed=True
                    ),
                ]
            ),
            "protein-source-3": _protein_record(
                [
                    _protein_candidate("R3-A", "MFFF"),
                    _protein_candidate(
                        "U3-A", "MGGG", source="uniprot", reviewed=False
                    ),
                ]
            ),
            "protein-source-4": _protein_record(
                [
                    _protein_candidate("NP_4.1", "MHHH"),
                    _protein_candidate("NP_4.2", "MHHH"),
                    _protein_candidate(
                        "U4-A", "MIII", source="uniprot", reviewed=False
                    ),
                    _protein_candidate(
                        "U4-B", "MKKK", source="uniprot", reviewed=False
                    ),
                    partial,
                ]
            ),
        }
        imported_path = _write_json(self.root / "hierarchy.json", imported)
        payload = update_retrieval_cache(
            entity="protein",
            mapping_path=mapping,
            output_path=self.root / "hierarchy_cache.json",
            import_path=imported_path,
            require_complete=True,
        )
        rows = {row["row_id"]: row for row in payload["rows"]}
        self.assertEqual(rows["1"]["resolution_rule"], "refseq_select")
        self.assertEqual(rows["2"]["resolution_rule"], "reviewed_uniprot")
        self.assertEqual(rows["3"]["resolution_rule"], "unreviewed_uniprot")
        self.assertEqual(
            rows["4"]["resolution_rule"],
            "known_refseq_np",
        )
        self.assertEqual(rows["4"]["equivalent_accessions"], ["NP_4.1", "NP_4.2"])
        known_np_attempt = next(
            attempt
            for attempt in rows["4"]["resolution_attempts"]
            if attempt["rule"] == "known_refseq_np"
        )
        self.assertEqual(known_np_attempt["candidate_count"], 2)
        self.assertIn("marked partial", rows["4"]["candidate_failures"][0])
        self.assertEqual(payload["taxids"], ["9606"])
        self.assertEqual(payload["annotation_releases"], ["test-release-1"])
        self.assertEqual(payload["rejected_candidate_count"], 1)

    def test_shard_merge_is_canonical_resumable_and_fail_closed(self) -> None:
        mapping = _write_mapping(self.root, "compound", 4)
        cache_import = _write_json(
            self.root / "smiles.json",
            {f"compound-source-{index}": f"C{index}" for index in range(1, 5)},
        )
        cache_path = self.root / "compound_cache.json"
        update_retrieval_cache(
            entity="compound",
            mapping_path=mapping,
            output_path=cache_path,
            import_path=cache_import,
            require_complete=True,
        )
        # Historical/formal compound caches predate the protein-only
        # resolved_sequence_set status.  Its absence means zero for compounds,
        # while a present malformed value must still fail strict type checks.
        legacy_compound_cache = _read(cache_path)
        legacy_compound_cache["status_counts"].pop("resolved_sequence_set")
        _write_json(cache_path, legacy_compound_cache)
        vectors = {
            str(index): np.full(2304, float(index), dtype=np.float32)
            for index in range(1, 5)
        }
        shard_one = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=mapping,
            cache=cache_path,
            start=0,
            stop=2,
            embeddings_by_row={"1": vectors["1"], "2": vectors["2"]},
            label="first",
        )
        assembly_dir = self.root / "assembly"

        with self.assertRaises(IncompleteFeatureBundleError):
            assemble_embedding_shards(
                entity="compound",
                mapping_path=mapping,
                shard_paths=[shard_one],
                output_dir=assembly_dir,
                retrieval_cache=cache_path,
            )
        partial = _read(assembly_dir / "assembly_manifest.json")
        self.assertEqual(partial["missing_count"], 2)
        self.assertFalse(partial["formal_eligible"])
        np.testing.assert_array_equal(
            np.load(assembly_dir / "missing_mask.npy"),
            [False, False, True, True],
        )

        shard_two = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=mapping,
            cache=cache_path,
            start=2,
            stop=4,
            embeddings_by_row={"3": vectors["3"], "4": vectors["4"]},
            label="second",
        )
        manifest_path = assemble_embedding_shards(
            entity="compound",
            mapping_path=mapping,
            shard_paths=[shard_two],
            output_dir=assembly_dir,
            retrieval_cache=cache_path,
        )
        complete = _read(manifest_path)
        self.assertTrue(complete["formal_eligible"])
        self.assertEqual(complete["missing_count"], 0)
        np.testing.assert_allclose(
            np.load(assembly_dir / "raw_embeddings.npy"),
            np.stack([vectors[str(index)] for index in range(1, 5)]),
        )
        sidecar = _read(feature_row_ids_path(assembly_dir / "raw_embeddings.npy"))
        self.assertEqual(sidecar["row_ids"], ["1", "2", "3", "4"])

    def test_shared_extraction_manifest_is_validated_once_per_assembly(self) -> None:
        mapping = _write_mapping(self.root, "compound", 4)
        cache_path = self.root / "shared_manifest_compound_cache.json"
        update_retrieval_cache(
            entity="compound",
            mapping_path=mapping,
            output_path=cache_path,
            import_path=_write_json(
                self.root / "shared_manifest_smiles.json",
                {
                    f"compound-source-{index}": f"C{index}"
                    for index in range(1, 5)
                },
            ),
            require_complete=True,
        )
        vectors = {
            str(index): np.full(2304, float(index), dtype=np.float32)
            for index in range(1, 5)
        }
        shard_one = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=mapping,
            cache=cache_path,
            start=0,
            stop=2,
            embeddings_by_row={"1": vectors["1"], "2": vectors["2"]},
            label="shared_first",
        )
        shard_two = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=mapping,
            cache=cache_path,
            start=2,
            stop=4,
            embeddings_by_row={"3": vectors["3"], "4": vectors["4"]},
            label="shared_second",
        )
        sidecar_one = shard_one.with_suffix(".manifest.json")
        sidecar_two = shard_two.with_suffix(".manifest.json")
        contract_one = _read(sidecar_one)
        contract_two = _read(sidecar_two)
        # Formal extraction shards from one run share one immutable provenance
        # closure even when their array files live in different directories.
        contract_two["provenance"] = contract_one["provenance"]
        _write_json(sidecar_two, contract_two)
        combined = _read(shard_one.parent / "extraction_manifest.json")
        combined.update(
            {
                "range": [0, 4],
                "provenance": contract_one["provenance"],
                "success_count": 4,
                "failure_count": 0,
                "shards": [
                    {
                        "sidecar": _artifact_for_test(sidecar_one),
                        "contract": contract_one,
                    },
                    {
                        "sidecar": _artifact_for_test(sidecar_two),
                        "contract": contract_two,
                    },
                ],
            }
        )
        # Byte-identical manifests intentionally exercise the content-addressed
        # cache even though each shard discovers it through its own directory.
        _write_json(shard_one.parent / "extraction_manifest.json", combined)
        _write_json(shard_two.parent / "extraction_manifest.json", combined)

        self.formal_source_validator.reset_mock()
        with patch(
            "astra.benchmark.ethnobotany_feature_bundle."
            "_validate_biological_model_provenance",
            wraps=_validate_biological_model_provenance,
        ) as shared_provenance_validator:
            assembled = assemble_embedding_shards(
                entity="compound",
                mapping_path=mapping,
                shard_paths=[shard_one, shard_two],
                output_dir=self.root / "shared_manifest_assembly",
                retrieval_cache=cache_path,
            )
        self.assertTrue(_read(assembled)["formal_eligible"])
        self.assertEqual(self.formal_source_validator.call_count, 1)
        self.assertEqual(shared_provenance_validator.call_count, 1)

        # Caching the manifest never bypasses selected-shard verification.
        tampered = _read(sidecar_two)
        tampered["row_ids"] = ["4", "3"]
        _write_json(sidecar_two, tampered)
        with self.assertRaises(IncompleteFeatureBundleError):
            assemble_embedding_shards(
                entity="compound",
                mapping_path=mapping,
                shard_paths=[shard_one, shard_two],
                output_dir=self.root / "tampered_shared_manifest_assembly",
                retrieval_cache=cache_path,
            )

    def test_reducer_recomputes_mask_flags_and_unmasked_degeneracy(self) -> None:
        mapping = _write_mapping(self.root, "compound", 65)
        cache_import = _write_json(
            self.root / "reducer_smiles.json",
            {
                f"compound-source-{index}": f"C{index}"
                for index in range(1, 66)
            },
        )
        cache = self.root / "reducer_compound_cache.json"
        update_retrieval_cache(
            entity="compound",
            mapping_path=mapping,
            output_path=cache,
            import_path=cache_import,
            require_complete=True,
        )
        compound_raw = np.eye(65, 2304, dtype=np.float32)
        shard = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=mapping,
            cache=cache,
            start=0,
            stop=65,
            embeddings_by_row={
                str(index): compound_raw[index - 1] for index in range(1, 66)
            },
            label="reducer",
        )
        assembly = assemble_embedding_shards(
            entity="compound",
            mapping_path=mapping,
            shard_paths=[shard],
            output_dir=self.root / "reducer_compound_assembly",
            retrieval_cache=cache,
        )
        assembly_payload = _read(assembly)
        pristine = json.loads(json.dumps(assembly_payload))
        assembly_payload["random_imputation"] = True
        _write_json(assembly, assembly_payload)
        with self.assertRaisesRegex(FeatureBundleError, "random imputation"):
            reduce_embedding_assembly(
                assembly_manifest=assembly,
                output_dir=self.root / "random_reducer",
            )

        _write_json(assembly, pristine)
        mask_path = Path(pristine["missing_mask"]["path"])
        np.save(mask_path, np.zeros(65, dtype=np.uint8))
        pristine["missing_mask"] = _artifact_for_test(mask_path)
        _write_json(assembly, pristine)
        with self.assertRaisesRegex(FeatureBundleError, "boolean and row-aligned"):
            reduce_embedding_assembly(
                assembly_manifest=assembly,
                output_dir=self.root / "bad_mask_reducer",
            )

        # A zero-UNK row cannot be the only source of apparent diversity.
        protein_mapping = _write_mapping(self.root, "protein", 66)
        protein_import = {
            f"protein-source-{index}": _protein_record(
                [_protein_candidate(f"PD{index}", "MSEQ")]
            )
            for index in range(1, 66)
        }
        protein_import["protein-source-66"] = _target_not_protein_record(
            "protein-source-66", protein_mapping
        )
        protein_cache = self.root / "degenerate_protein_cache.json"
        update_retrieval_cache(
            entity="protein",
            mapping_path=protein_mapping,
            output_path=protein_cache,
            import_path=_write_json(
                self.root / "degenerate_sequences.json", protein_import
            ),
            missing_sequence_policy="zero-unk",
            require_complete=True,
        )
        protein_shard = _write_formal_biological_shard(
            self.root,
            entity="protein",
            mapping=protein_mapping,
            cache=protein_cache,
            start=0,
            stop=66,
            embeddings_by_row={
                str(index): np.ones(640, dtype=np.float32)
                for index in range(1, 66)
            },
            failures={"66": "target-not-protein has no ESM embedding"},
            label="degenerate",
        )
        protein_assembly = assemble_embedding_shards(
            entity="protein",
            mapping_path=protein_mapping,
            shard_paths=[protein_shard],
            output_dir=self.root / "degenerate_protein_assembly",
            retrieval_cache=protein_cache,
        )
        with self.assertRaisesRegex(FeatureBundleError, "degenerate"):
            reduce_embedding_assembly(
                assembly_manifest=protein_assembly,
                output_dir=self.root / "degenerate_protein_component",
            )

    def test_biological_bundle_keeps_explicit_zero_unk_and_hashes_every_asset(self) -> None:
        components: list[Path] = []

        compound_mapping = _write_mapping(self.root, "compound", 65)
        compound_cache = self.root / "all_compound_cache.json"
        update_retrieval_cache(
            entity="compound",
            mapping_path=compound_mapping,
            output_path=compound_cache,
            import_path=_write_json(
                self.root / "all_smiles.json",
                {f"compound-source-{index}": f"C{index}" for index in range(1, 66)},
            ),
            require_complete=True,
        )
        compound_raw = np.eye(65, 2304, dtype=np.float32)
        compound_shard = _write_formal_biological_shard(
            self.root,
            entity="compound",
            mapping=compound_mapping,
            cache=compound_cache,
            start=0,
            stop=65,
            embeddings_by_row={
                str(index): compound_raw[index - 1] for index in range(1, 66)
            },
            label="complete",
        )
        compound_assembly = assemble_embedding_shards(
            entity="compound",
            mapping_path=compound_mapping,
            shard_paths=[compound_shard],
            output_dir=self.root / "compound_assembly",
            retrieval_cache=compound_cache,
        )
        components.append(
            reduce_embedding_assembly(
                assembly_manifest=compound_assembly,
                output_dir=self.root / "compound_component",
            )
        )

        protein_mapping = _write_mapping(self.root, "protein", 66)
        protein_import = {
            f"protein-source-{index}": _protein_record(
                [_protein_candidate(f"NP_{index}.1", "MSEQ")]
            )
            for index in range(1, 66)
        }
        protein_import["protein-source-66"] = _target_not_protein_record(
            "protein-source-66", protein_mapping
        )
        protein_cache = self.root / "protein_cache.json"
        update_retrieval_cache(
            entity="protein",
            mapping_path=protein_mapping,
            output_path=protein_cache,
            import_path=_write_json(self.root / "protein_sequences.json", protein_import),
            missing_sequence_policy="zero-unk",
            require_complete=True,
        )
        protein_raw = np.eye(65, 640, dtype=np.float32)
        protein_shard = _write_formal_biological_shard(
            self.root,
            entity="protein",
            mapping=protein_mapping,
            cache=protein_cache,
            start=0,
            stop=66,
            embeddings_by_row={
                str(index): protein_raw[index - 1] for index in range(1, 66)
            },
            failures={"66": "target-not-protein has no ESM embedding"},
            label="complete",
        )
        protein_assembly = assemble_embedding_shards(
            entity="protein",
            mapping_path=protein_mapping,
            shard_paths=[protein_shard],
            output_dir=self.root / "protein_assembly",
            retrieval_cache=protein_cache,
        )
        protein_assembly_payload = _read(protein_assembly)
        self.assertTrue(protein_assembly_payload["formal_eligible"])
        self.assertEqual(protein_assembly_payload["declared_zero_unk_count"], 1)
        self.assertEqual(protein_assembly_payload["missing_count"], 1)
        np.testing.assert_array_equal(
            np.load(self.root / "protein_assembly" / "raw_embeddings.npy")[-1],
            np.zeros(640),
        )
        protein_component = reduce_embedding_assembly(
            assembly_manifest=protein_assembly,
            output_dir=self.root / "protein_component",
        )
        components.append(protein_component)
        np.testing.assert_array_equal(
            np.loadtxt(self.root / "protein_component" / "protein_embedding_esm2.txt")[-1],
            np.zeros(64),
        )

        bundle_path = finalize_feature_bundle(
            component_manifests=components,
            output_dir=self.root / "bundle",
            profile="hghda-biological-only",
        )
        bundle = _read(bundle_path)
        self.assertEqual(bundle["schema"], BUNDLE_SCHEMA)
        self.assertEqual(bundle["profile"], "hghda-biological-only")
        self.assertEqual(set(bundle["entities"]), {"compound", "protein"})
        self.assertTrue(bundle["formal_eligible"])
        self.assertFalse(bundle["random_imputation"])
        self.assertEqual(bundle["entities"]["protein"]["declared_zero_unk_count"], 1)
        for entity, record in bundle["entities"].items():
            matrix = Path(record["matrix"]["path"])
            sidecar = Path(record["row_ids_sidecar"]["path"])
            mapping = Path(record["mapping"]["path"])
            loaded = load_aligned_feature_matrix(
                matrix,
                entity=entity,
                canonical_ids=[str(index) for index in range(1, record["shape"][0] + 1)],
                mapping_sha256=record["mapping_sha256"],
                sidecar_path=sidecar,
            )
            self.assertEqual(loaded.matrix.shape[1], 64)
            self.assertEqual(record["mapping"]["sha256"], _sha256_for_test(mapping))

        protocol = SimpleNamespace(
            entity_ids={
                entity: tuple(str(index) for index in range(1, record["shape"][0] + 1))
                for entity, record in bundle["entities"].items()
            },
            source_files={
                f"mapping/{entity}": {"sha256": record["mapping_sha256"]}
                for entity, record in bundle["entities"].items()
            },
        )
        selected_paths = {
            entity: Path(record["matrix"]["path"])
            for entity, record in bundle["entities"].items()
        }
        mask_paths, diagnostics = _validate_ethnobotany_feature_bundle(
            protocol, bundle_path.parent, selected_paths, smoke=False
        )
        self.assertTrue(diagnostics["formal_eligible"])
        self.assertEqual(diagnostics["entities"]["protein"]["missing_count"], 1)
        self.assertEqual(np.flatnonzero(np.load(mask_paths["protein"])).tolist(), [65])

        pristine_bundle = json.loads(json.dumps(bundle))
        for key, bad_value, message in (
            ("random_imputation", True, "random imputation"),
            ("formal_ineligibility_reasons", ["forged"], "ineligibility"),
            ("formal_eligible", "true", "JSON boolean"),
            ("row_id_space", "implicit_row_number", "identity/schema"),
        ):
            malicious = json.loads(json.dumps(pristine_bundle))
            malicious[key] = bad_value
            _write_json(bundle_path, malicious)
            with self.assertRaisesRegex(ASTRARunnerError, message):
                _validate_ethnobotany_feature_bundle(
                    protocol, bundle_path.parent, selected_paths, smoke=False
                )

        malicious = json.loads(json.dumps(pristine_bundle))
        matrix_path = Path(malicious["entities"]["protein"]["matrix"]["path"])
        pristine_matrix = matrix_path.read_bytes()
        values = np.loadtxt(matrix_path)
        values[0, 0] += 1.0
        np.savetxt(matrix_path, values)
        malicious["entities"]["protein"]["matrix"] = _bundle_artifact_for_test(
            bundle_path.parent, matrix_path
        )
        _write_json(bundle_path, malicious)
        with self.assertRaisesRegex(ASTRARunnerError, "recursive provenance"):
            _validate_ethnobotany_feature_bundle(
                protocol, bundle_path.parent, selected_paths, smoke=False
            )
        matrix_path.write_bytes(pristine_matrix)
        _write_json(bundle_path, pristine_bundle)

        protein_component_payload = _read(protein_component)
        protein_component_payload["random_imputation"] = True
        _write_json(protein_component, protein_component_payload)
        with self.assertRaisesRegex(FeatureBundleError, "random"):
            finalize_feature_bundle(
                component_manifests=components,
                output_dir=self.root / "forged_component_bundle",
                profile="hghda-biological-only",
            )


def _sha256_for_test(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_for_test(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_for_test(path),
    }


def _bundle_artifact_for_test(root: Path, path: Path) -> dict[str, object]:
    return {
        **_artifact_for_test(path),
        "relative_path": str(path.resolve().relative_to(root.resolve())),
    }


if __name__ == "__main__":
    unittest.main()
