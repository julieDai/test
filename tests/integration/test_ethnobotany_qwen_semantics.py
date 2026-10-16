from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from astra.benchmark import ethnobotany_feature_bundle as bundle
from astra.benchmark import ethnobotany_qwen_semantics as qwen
from astra.benchmark import ethnobotany_semantic_features as semantic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(path: Path, entity: str, source_ids: list[str]) -> None:
    source_column = "herb_curie" if entity == "herb" else "disease_mondo"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{entity}_id", source_column])
        writer.writerows((index, value) for index, value in enumerate(source_ids, 1))


class EthnobotanyQwenSemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.herb_ids = [f"ncbitaxon:{1000 + index}" for index in range(33)]
        self.disease_ids = [f"MONDO:{index:07d}" for index in range(1, 33)]
        self.herb_mapping = self.root / "herb_id_mapping.csv"
        self.disease_mapping = self.root / "disease_id_mapping.csv"
        _mapping(self.herb_mapping, "herb", self.herb_ids)
        _mapping(self.disease_mapping, "disease", self.disease_ids)

        self.herb_component = self.root / "herb-component.csv"
        with self.herb_component.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["plant_curie", "plant name"])
            for index, source_id in enumerate(self.herb_ids):
                writer.writerow([source_id, f"Plant {index}"])

        nodes = [
            {
                "id": (
                    "http://purl.obolibrary.org/obo/MONDO_"
                    f"{source_id.split(':', 1)[1]}"
                ),
                "lbl": f"Disease {index}",
                "meta": {"definition": {"val": f"Definition {index}"}},
            }
            for index, source_id in enumerate(self.disease_ids)
        ]
        self.mondo = self.root / "mondo-base.json"
        self.mondo.write_text(
            json.dumps({"graphs": [{"nodes": nodes}]}), encoding="utf-8"
        )
        self.source_patches = mock.patch.multiple(
            semantic,
            HERB_COMPONENT_SHA256=_sha(self.herb_component),
            MONDO_BASE_JSON_SHA256=_sha(self.mondo),
        )
        self.source_patches.start()
        self.environment = mock.patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
            },
        )
        self.environment.start()
        self.source_text = semantic.build_semantic_text_manifest(
            herb_mapping_path=self.herb_mapping,
            disease_mapping_path=self.disease_mapping,
            herb_component_csv=self.herb_component,
            mondo_json=self.mondo,
            output_path=self.root / "source_text_manifest.json",
            expected_herb_component_sha256=_sha(self.herb_component),
            expected_mondo_sha256=_sha(self.mondo),
        )

    def tearDown(self) -> None:
        self.environment.stop()
        self.source_patches.stop()
        self.temporary.cleanup()

    @staticmethod
    def _fake_api(_: str, request: dict[str, object]) -> dict[str, object]:
        return {
            "provider_response_id": f"fixture-{request['entity']}-{request['row_id']}",
            "response_model": qwen.QWEN_MODEL,
            "finish_reason": "stop",
            "text": (
                f"Generated target-neutral scientific text for {request['entity']} "
                f"row {request['row_id']}."
            ),
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

    def _build_qwen_text(self) -> tuple[Path, Path, Path, Path]:
        request = qwen.build_qwen_request_manifest(
            source_text_manifest=self.source_text,
            output_path=self.root / "qwen_requests.json",
        )
        responses = self.root / "qwen_responses.jsonl"
        status = qwen.generate_qwen_responses(
            request_manifest=request,
            response_jsonl=responses,
            status_path=self.root / "qwen_status.json",
            workers=3,
            requests_per_minute=100_000,
            max_retries=1,
            api_key="fixture-secret-not-serialized",
            request_fn=self._fake_api,
        )
        text = qwen.finalize_qwen_text_manifest(
            generation_status=status,
            output_path=self.root / "qwen_text_manifest.json",
        )
        return request, responses, status, text

    def test_requests_are_rederived_and_generation_is_complete_and_resumable(self) -> None:
        request, responses, status, text = self._build_qwen_text()
        checked_request = qwen.validate_qwen_request_manifest(request)
        self.assertEqual(len(checked_request["requests"]), 65)
        self.assertNotIn("fixture-secret", request.read_text(encoding="utf-8"))
        self.assertNotIn("fixture-secret", status.read_text(encoding="utf-8"))
        checked_text = semantic.validate_semantic_text_manifest(text)
        self.assertEqual(
            checked_text["manifest"]["semantic_condition"],
            qwen.QWEN_SEMANTIC_CONDITION,
        )
        self.assertEqual(
            checked_text["manifest"]["entities"]["herb"]["row_count"], 33
        )

        partial_status = qwen.generate_qwen_responses(
            request_manifest=request,
            response_jsonl=self.root / "partial_responses.jsonl",
            status_path=self.root / "partial_status.json",
            workers=1,
            requests_per_minute=100_000,
            max_retries=1,
            limit=1,
            api_key="fixture",
            request_fn=self._fake_api,
        )
        self.assertEqual(
            json.loads(partial_status.read_text(encoding="utf-8"))["status"],
            "partial",
        )
        with self.assertRaisesRegex(qwen.QwenSemanticError, "not complete"):
            qwen.finalize_qwen_text_manifest(
                generation_status=partial_status,
                output_path=self.root / "invalid_partial_text.json",
            )

        calls: list[object] = []

        def must_not_call(*args: object) -> dict[str, object]:
            calls.append(args)
            raise AssertionError("resume should not call an already complete request")

        qwen.generate_qwen_responses(
            request_manifest=request,
            response_jsonl=responses,
            status_path=status,
            workers=1,
            requests_per_minute=100_000,
            max_retries=1,
            api_key="fixture",
            request_fn=must_not_call,
        )
        self.assertEqual(calls, [])

        tampered = json.loads(request.read_text(encoding="utf-8"))
        tampered["entities"]["herb"]["requests"][0]["messages"][1][
            "content"
        ] += " Mention indications."
        request.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(qwen.QwenSemanticError, "not re-derived"):
            qwen.validate_qwen_request_manifest(request)

    def test_qwen_text_reuses_scibert_shared_pca_and_bundle_bridge(self) -> None:
        _, _, _, text = self._build_qwen_text()
        snapshot = self.root / semantic.SCIBERT_REVISION
        snapshot.mkdir()
        hashes: dict[str, str] = {}
        for name in semantic.SCIBERT_FILE_SHA256:
            path = snapshot / name
            path.write_bytes(f"fixture:{name}".encode())
            hashes[name] = _sha(path)

        call_index = 0

        def fake_encode(
            texts: list[str], *, snapshot_dir: Path, batch_size: int
        ) -> tuple[np.ndarray, dict[str, object]]:
            nonlocal call_index
            rng = np.random.default_rng(42 + call_index)
            call_index += 1
            return (
                rng.normal(size=(len(texts), semantic.SCIBERT_DIMENSION)).astype(
                    np.float32
                ),
                {
                    "python": "fixture",
                    "python_executable": "fixture",
                    "platform": "fixture",
                    "numpy": np.__version__,
                    "torch": "fixture",
                    "transformers": "fixture",
                    "device": "cpu",
                    "seed": 42,
                    "model_training_flag": False,
                    "deterministic_algorithms": True,
                    "torch_num_threads": 4,
                    "torch_num_interop_threads": 1,
                    "thread_environment": {
                        "OMP_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4",
                    },
                    "batch_size": batch_size,
                },
            )

        with mock.patch.object(semantic, "SCIBERT_FILE_SHA256", hashes), mock.patch.object(
            semantic, "_encode_scibert", fake_encode
        ):
            embeddings = semantic.extract_scibert_semantic_embeddings(
                text_manifest=text,
                snapshot_dir=snapshot,
                output_dir=self.root / "scibert",
                batch_size=8,
            )
            reduction = semantic.reduce_shared_semantic_embeddings(
                embedding_manifest=embeddings,
                output_dir=self.root / "pca64",
            )
            checked = semantic.validate_shared_semantic_reduction(reduction)
            self.assertEqual(
                checked["manifest"]["semantic_condition"],
                qwen.QWEN_SEMANTIC_CONDITION,
            )
            components = bundle.build_verified_semantic_components(
                semantic_reduction_manifest=reduction,
                herb_mapping_path=self.herb_mapping,
                disease_mapping_path=self.disease_mapping,
                output_dir=self.root / "semantic_components",
            )
            for entity in ("herb", "disease"):
                component = bundle.validate_feature_component(components[entity])
                self.assertEqual(
                    component["manifest"]["semantic_condition"],
                    qwen.QWEN_SEMANTIC_CONDITION,
                )

            portable = self.root / "portable"
            portable.mkdir()
            resolver: dict[str, Path] = {}
            for index, dependency in enumerate(checked["dependencies"]):
                target = portable / f"{index:02d}_{dependency.name}"
                shutil.copy2(dependency, target)
                resolver[_sha(dependency)] = target
            portable_reduction = resolver[_sha(reduction)]
            portable_checked = semantic.validate_shared_semantic_reduction(
                portable_reduction, resolver=resolver
            )
            self.assertEqual(
                portable_checked["manifest"]["semantic_condition"],
                qwen.QWEN_SEMANTIC_CONDITION,
            )


if __name__ == "__main__":
    unittest.main()
