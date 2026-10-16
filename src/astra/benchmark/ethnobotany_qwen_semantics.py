#!/usr/bin/env python3
"""Auditable Qwen text generation for Ethnobotany semantic features.

The API stage is deliberately separated from SciBERT and PCA.  A frozen,
ID-keyed request manifest is prepared first; successful provider responses are
appended to JSONL and may be resumed; a final text manifest is emitted only
after exact response coverage and request identity have been verified.  The
final manifest is accepted by ``ethnobotany_semantic_features`` so the existing
pinned SciBERT and shared-PCA implementation is reused unchanged.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .ethnobotany_semantic_features import (
    SemanticFeatureError,
    _artifact,
    _read_json,
    _text_sha256,
    _verify_artifact,
    validate_semantic_text_manifest,
)


QWEN_REQUEST_MANIFEST_SCHEMA = "astra-ethnobotany-qwen-requests/v1"
QWEN_RESPONSE_SCHEMA = "astra-ethnobotany-qwen-response/v1"
QWEN_STATUS_SCHEMA = "astra-ethnobotany-qwen-generation-status/v1"
QWEN_TEXT_MANIFEST_SCHEMA = "astra-ethnobotany-qwen-semantic-text/v1"
QWEN_SEMANTIC_CONDITION = (
    "ethnobotany-id-keyed-qwen-plus-2025-12-01-target-neutral-v1-"
    "scibert-cls-shared-pca64/v1"
)
QWEN_CONDITION_STATUS = "new_qwen_generated_condition_not_legacy_reproduction"
QWEN_MODEL = "qwen-plus-2025-12-01"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_TEMPERATURE = 0.2
QWEN_MAX_TOKENS = 600
PROMPT_VERSION = "ethnobotany-target-neutral-v1"
SYSTEM_PROMPT = (
    "Write a concise, factual scientific entity description for representation "
    "learning. Return one cohesive English paragraph only, without headings, "
    "citations, lists, or disclaimers."
)
HERB_PROMPT_TEMPLATE = """Using the entity identity below, write 150--200 words describing its botanical identity, medicinal part when known, pharmacognostic characteristics, representative phytochemicals, and molecular targets or pathways. Do not mention named diseases, symptoms, clinical indications, therapeutic recommendations, herb--disease associations, or database labels. If a fact is uncertain, omit it rather than inventing it.

Entity identity:
{source_text}"""
DISEASE_PROMPT_TEMPLATE = """Using the entity identity and ontology context below, write 150--200 words describing its biomedical definition, pathological or biological processes, molecular mechanisms, and relevant genes, proteins, targets, or pathways. Do not mention herbs, medicinal plants, botanical sources, phytochemicals as treatments, therapeutic recommendations, or plant--disease associations. If a fact is uncertain, omit it rather than inventing it.

Entity identity and ontology context:
{source_text}"""


class QwenSemanticError(SemanticFeatureError):
    """Raised when Qwen request, response, or text provenance is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_core(
    *, entity: str, row: Mapping[str, Any], source_text: str
) -> dict[str, Any]:
    template = HERB_PROMPT_TEMPLATE if entity == "herb" else DISEASE_PROMPT_TEMPLATE
    user_prompt = template.format(source_text=source_text)
    return {
        "entity": entity,
        "row_id": str(row["row_id"]),
        "source_id": str(row["source_id"]),
        "source_text_sha256": _text_sha256(source_text),
        "model": QWEN_MODEL,
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def _request_rows(source_manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for entity in ("herb", "disease"):
        rows: list[dict[str, Any]] = []
        for source_row in source_manifest["entities"][entity]["rows"]:
            source_text = str(source_row["text"])
            core = _request_core(
                entity=entity, row=source_row, source_text=source_text
            )
            rows.append({**core, "request_sha256": _canonical_sha256(core)})
        result[entity] = rows
    return result


def _request_contract(
    source_text_manifest: Path, source_validation: Mapping[str, Any]
) -> dict[str, Any]:
    rows = _request_rows(source_validation["manifest"])
    return {
        "schema": QWEN_REQUEST_MANIFEST_SCHEMA,
        "semantic_condition": QWEN_SEMANTIC_CONDITION,
        "semantic_condition_status": QWEN_CONDITION_STATUS,
        "formal_eligible": True,
        "association_labels_used": False,
        "target_neutral_prompt": True,
        "source_text_manifest": _artifact(source_text_manifest),
        "api_contract": {
            "base_url": QWEN_BASE_URL,
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": QWEN_MAX_TOKENS,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "templates": {
                "herb": HERB_PROMPT_TEMPLATE,
                "disease": DISEASE_PROMPT_TEMPLATE,
            },
            "credential_source": "DASHSCOPE_API_KEY_environment_only",
        },
        "entities": {
            entity: {"request_count": len(rows[entity]), "requests": rows[entity]}
            for entity in ("herb", "disease")
        },
        "request_count": sum(len(rows[entity]) for entity in ("herb", "disease")),
    }


def build_qwen_request_manifest(
    *, source_text_manifest: Path | str, output_path: Path | str
) -> Path:
    """Freeze all 3,947 production requests before any API call is made."""

    source_path = Path(source_text_manifest).resolve()
    source = validate_semantic_text_manifest(source_path)
    if source["manifest"].get("schema") == QWEN_TEXT_MANIFEST_SCHEMA:
        raise QwenSemanticError("Qwen requests must be built from source texts, not Qwen output")
    payload = {
        **_request_contract(source_path, source),
        "created_at": _utc_now(),
    }
    output = Path(output_path).resolve()
    _atomic_json(output, payload)
    validate_qwen_request_manifest(output)
    return output


def validate_qwen_request_manifest(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != QWEN_REQUEST_MANIFEST_SCHEMA:
        raise QwenSemanticError("Qwen request manifest has a wrong schema")
    if (
        payload.get("semantic_condition") != QWEN_SEMANTIC_CONDITION
        or payload.get("semantic_condition_status") != QWEN_CONDITION_STATUS
        or payload.get("formal_eligible") is not True
        or payload.get("association_labels_used") is not False
        or payload.get("target_neutral_prompt") is not True
    ):
        raise QwenSemanticError("Qwen request eligibility contract is false")
    source_path = _verify_artifact(
        payload.get("source_text_manifest"),
        "Qwen source text manifest",
        resolver=resolver,
    )
    source = validate_semantic_text_manifest(source_path, resolver=resolver)
    if source["manifest"].get("schema") == QWEN_TEXT_MANIFEST_SCHEMA:
        raise QwenSemanticError("Qwen request source recursively points to Qwen output")
    expected = _request_contract(source_path, source)
    if resolver is not None:
        expected["source_text_manifest"]["path"] = payload[
            "source_text_manifest"
        ]["path"]
    observed = {key: value for key, value in payload.items() if key != "created_at"}
    if observed != expected:
        raise QwenSemanticError(
            "Qwen request manifest is not re-derived from the pinned source texts"
        )
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in ("herb", "disease"):
        for record in payload["entities"][entity]["requests"]:
            key = (entity, str(record["row_id"]))
            if key in requests:
                raise QwenSemanticError(f"duplicate Qwen request identity {key}")
            requests[key] = record
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "source": source,
        "requests": requests,
        "dependencies": list(
            dict.fromkeys([manifest_path, *source["dependencies"]])
        ),
    }


def _normalise_usage(value: object) -> dict[str, int]:
    source = value if isinstance(value, Mapping) else {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = source.get(key, 0)
        if type(raw) is not int or raw < 0:
            raise QwenSemanticError(f"provider usage {key} is invalid")
        result[key] = raw
    return result


def _read_responses(
    path: Path,
    *,
    requests: Mapping[tuple[str, str], Mapping[str, Any]],
    require_complete: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        if require_complete:
            raise QwenSemanticError(f"Qwen response JSONL is absent: {path}")
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QwenSemanticError(
                        f"Qwen response {path}:{line_number} is not an object"
                    )
                entity = value.get("entity")
                row_id = value.get("row_id")
                key = (str(entity), str(row_id))
                request_record = requests.get(key)
                if request_record is None:
                    raise QwenSemanticError(
                        f"Qwen response {path}:{line_number} has unknown identity {key}"
                    )
                if key in records:
                    raise QwenSemanticError(f"duplicate Qwen response identity {key}")
                text = value.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise QwenSemanticError(f"Qwen response {key} has empty text")
                text = text.strip()
                provider_id = value.get("provider_response_id")
                if provider_id is not None and (
                    not isinstance(provider_id, str) or not provider_id.strip()
                ):
                    raise QwenSemanticError(
                        f"Qwen response {key} has an invalid provider response ID"
                    )
                expected = {
                    "schema": QWEN_RESPONSE_SCHEMA,
                    "entity": key[0],
                    "row_id": key[1],
                    "source_id": request_record["source_id"],
                    "request_sha256": request_record["request_sha256"],
                    "model_requested": QWEN_MODEL,
                    "response_model": QWEN_MODEL,
                    "finish_reason": "stop",
                    "text": text,
                    "text_sha256": _text_sha256(text),
                    "provider_response_id": provider_id,
                    "usage": _normalise_usage(value.get("usage")),
                }
                if value != expected:
                    raise QwenSemanticError(
                        f"Qwen response {key} differs from its pinned request contract"
                    )
                records[key] = value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenSemanticError(f"cannot parse Qwen response JSONL {path}: {error}") from error
    if require_complete and set(records) != set(requests):
        missing = sorted(set(requests).difference(records))
        raise QwenSemanticError(
            f"Qwen response coverage is incomplete: {len(records)}/{len(requests)}; "
            f"first missing={missing[:3]}"
        )
    return records


class _SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise QwenSemanticError("requests_per_minute must be positive")
        self.limit = requests_per_minute
        self.timestamps: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= 60.0:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.limit:
                    self.timestamps.append(now)
                    return
                delay = 60.0 - (now - self.timestamps[0]) + 0.05
            time.sleep(delay)


def _dashscope_call(api_key: str, request_record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        "model": request_record["model"],
        "messages": request_record["messages"],
        "temperature": request_record["temperature"],
        "max_tokens": request_record["max_tokens"],
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        QWEN_BASE_URL,
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=195) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"DashScope HTTP {error.code}: {detail}") from error
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"DashScope request failed: {error}") from error
    try:
        choice = body["choices"][0]
        return {
            "provider_response_id": body.get("id"),
            "response_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "text": choice["message"]["content"],
            "usage": body.get("usage", {}),
        }
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("DashScope response lacks the expected choice") from error


def generate_qwen_responses(
    *,
    request_manifest: Path | str,
    response_jsonl: Path | str,
    status_path: Path | str,
    workers: int = 6,
    requests_per_minute: int = 50,
    max_retries: int = 5,
    limit: int | None = None,
    api_key: str | None = None,
    request_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Path:
    """Generate or resume Qwen responses; never serialize the API credential."""

    if workers < 1 or max_retries < 1 or (limit is not None and limit < 1):
        raise QwenSemanticError("workers, max_retries, and limit must be positive")
    validated = validate_qwen_request_manifest(request_manifest)
    response_path = Path(response_jsonl).resolve()
    status = Path(status_path).resolve()
    response_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_responses(
        response_path, requests=validated["requests"], require_complete=False
    )
    pending = [
        record
        for entity in ("herb", "disease")
        for record in validated["manifest"]["entities"][entity]["requests"]
        if (entity, str(record["row_id"])) not in existing
    ]
    if limit is not None:
        pending = pending[:limit]
    secret = api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY")
    if not secret:
        raise QwenSemanticError("DASHSCOPE_API_KEY is not configured")
    call = request_fn or _dashscope_call
    limiter = _SlidingWindowLimiter(requests_per_minute)
    failures: list[dict[str, Any]] = []
    completed_this_run = 0
    started = _utc_now()

    def snapshot(state: str) -> None:
        payload: dict[str, Any] = {
            "schema": QWEN_STATUS_SCHEMA,
            "status": state,
            "started_at": started,
            "updated_at": _utc_now(),
            "request_manifest": _artifact(validated["manifest_path"]),
            "response_jsonl_path": str(response_path),
            "request_count": len(validated["requests"]),
            "initial_completed": len(existing),
            "completed_this_run": completed_this_run,
            "completed_total": len(existing) + completed_this_run,
            "workers": workers,
            "requests_per_minute": requests_per_minute,
            "max_retries": max_retries,
            "credential_source": "DASHSCOPE_API_KEY_environment_only",
            "failures": failures,
        }
        if state == "complete":
            payload["response_jsonl"] = _artifact(response_path)
        _atomic_json(status, payload)

    def one(record: Mapping[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            limiter.acquire()
            try:
                raw = call(secret, record)
                text = raw.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("provider returned empty text")
                cleaned = text.strip()
                response = {
                    "schema": QWEN_RESPONSE_SCHEMA,
                    "entity": record["entity"],
                    "row_id": record["row_id"],
                    "source_id": record["source_id"],
                    "request_sha256": record["request_sha256"],
                    "model_requested": QWEN_MODEL,
                    "response_model": raw.get("response_model"),
                    "finish_reason": raw.get("finish_reason"),
                    "text": cleaned,
                    "text_sha256": _text_sha256(cleaned),
                    "provider_response_id": raw.get("provider_response_id"),
                    "usage": _normalise_usage(raw.get("usage")),
                }
                # Apply the strict model/termination checks before it reaches disk.
                if response["response_model"] != QWEN_MODEL:
                    raise RuntimeError("provider response model differs from the pin")
                if response["finish_reason"] != "stop":
                    raise RuntimeError("provider response did not finish with stop")
                return response
            except Exception as error:  # provider/network boundary
                last_error = error
                if attempt + 1 < max_retries:
                    time.sleep(min(30.0, float(2**attempt)))
        raise RuntimeError(str(last_error))

    snapshot("running")
    try:
        with response_path.open("a", encoding="utf-8", buffering=1) as stream:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                work = iter(pending)
                futures: dict[
                    concurrent.futures.Future[dict[str, Any]], Mapping[str, Any]
                ] = {}

                def submit_next() -> bool:
                    try:
                        record = next(work)
                    except StopIteration:
                        return False
                    futures[executor.submit(one, record)] = record
                    return True

                for _ in range(workers * 2):
                    if not submit_next():
                        break
                processed = 0
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        record = futures.pop(future)
                        try:
                            response = future.result()
                        except Exception as error:  # preserve successes for resume
                            failures.append(
                                {
                                    "entity": record["entity"],
                                    "row_id": record["row_id"],
                                    "error": repr(error),
                                }
                            )
                        else:
                            stream.write(json.dumps(response, ensure_ascii=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                            completed_this_run += 1
                        processed += 1
                        submit_next()
                        if processed % 10 == 0:
                            snapshot("running")
                            print(
                                "qwen progress "
                                f"completed={len(existing) + completed_this_run}/"
                                f"{len(validated['requests'])} "
                                f"failed_this_run={len(failures)}",
                                flush=True,
                            )
        if failures:
            snapshot("failed")
            raise QwenSemanticError(
                f"{len(failures)} Qwen requests failed; rerun the same command to resume"
            )
        observed = _read_responses(
            response_path, requests=validated["requests"], require_complete=False
        )
        snapshot(
            "complete" if len(observed) == len(validated["requests"]) else "partial"
        )
    except BaseException:
        if not status.is_file() or _read_json(status).get("status") == "running":
            snapshot("failed")
        raise
    return status


def validate_qwen_generation_status(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    status_path = Path(path).resolve()
    payload = _read_json(status_path)
    if payload.get("schema") != QWEN_STATUS_SCHEMA or payload.get("status") != "complete":
        raise QwenSemanticError("Qwen generation status is not complete")
    if payload.get("credential_source") != "DASHSCOPE_API_KEY_environment_only":
        raise QwenSemanticError("Qwen credential provenance is invalid")
    for field in (
        "request_count",
        "initial_completed",
        "completed_this_run",
        "completed_total",
        "workers",
        "requests_per_minute",
        "max_retries",
    ):
        if type(payload.get(field)) is not int or payload[field] < 0:
            raise QwenSemanticError(f"Qwen status field {field} is invalid")
    if (
        payload["workers"] < 1
        or payload["requests_per_minute"] < 1
        or payload["max_retries"] < 1
        or payload["initial_completed"] + payload["completed_this_run"]
        != payload["completed_total"]
    ):
        raise QwenSemanticError("Qwen generation runtime/count contract is invalid")
    request_path = _verify_artifact(
        payload.get("request_manifest"), "Qwen request manifest", resolver=resolver
    )
    requests = validate_qwen_request_manifest(request_path, resolver=resolver)
    response_path = _verify_artifact(
        payload.get("response_jsonl"), "Qwen response JSONL", resolver=resolver
    )
    if payload.get("response_jsonl_path") != payload["response_jsonl"].get("path"):
        raise QwenSemanticError("Qwen status response path differs from its artifact")
    responses = _read_responses(
        response_path, requests=requests["requests"], require_complete=True
    )
    if (
        payload.get("request_count") != len(requests["requests"])
        or payload.get("completed_total") != len(responses)
        or payload.get("failures") != []
    ):
        raise QwenSemanticError("Qwen generation status counts are inconsistent")
    return {
        "manifest": payload,
        "manifest_path": status_path,
        "request": requests,
        "response_path": response_path,
        "responses": responses,
        "dependencies": list(
            dict.fromkeys(
                [status_path, response_path, *requests["dependencies"]]
            )
        ),
    }


def _text_contract(
    *, status_path: Path, status: Mapping[str, Any]
) -> dict[str, Any]:
    request = status["request"]
    responses = status["responses"]
    source = request["source"]
    entities: dict[str, Any] = {}
    for entity in ("herb", "disease"):
        rows: list[dict[str, Any]] = []
        for request_record in request["manifest"]["entities"][entity]["requests"]:
            key = (entity, str(request_record["row_id"]))
            response = responses[key]
            rows.append(
                {
                    "row_id": request_record["row_id"],
                    "source_id": request_record["source_id"],
                    "source_text_sha256": request_record["source_text_sha256"],
                    "request_sha256": request_record["request_sha256"],
                    "response_model": response["response_model"],
                    "text": response["text"],
                    "text_sha256": response["text_sha256"],
                }
            )
        entities[entity] = {
            "row_count": len(rows),
            "mapping_sha256": source["manifest"]["entities"][entity][
                "mapping_sha256"
            ],
            "rows": rows,
        }
    return {
        "schema": QWEN_TEXT_MANIFEST_SCHEMA,
        "semantic_condition": QWEN_SEMANTIC_CONDITION,
        "semantic_condition_status": QWEN_CONDITION_STATUS,
        "formal_eligible": True,
        "random_imputation": False,
        "association_label_free": True,
        "association_labels_used": False,
        "target_neutral_prompt": True,
        "row_identity_basis": "generation_time_id_keyed_qwen_request_join",
        "generation_status": _artifact(status_path),
        "mappings": {
            entity: source["manifest"]["mappings"][entity]
            for entity in ("herb", "disease")
        },
        "entities": entities,
    }


def finalize_qwen_text_manifest(
    *, generation_status: Path | str, output_path: Path | str
) -> Path:
    status_path = Path(generation_status).resolve()
    status = validate_qwen_generation_status(status_path)
    payload = {**_text_contract(status_path=status_path, status=status), "created_at": _utc_now()}
    output = Path(output_path).resolve()
    _atomic_json(output, payload)
    validate_qwen_text_manifest(output)
    return output


def validate_qwen_text_manifest(
    path: Path | str, *, resolver: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != QWEN_TEXT_MANIFEST_SCHEMA:
        raise QwenSemanticError("Qwen text manifest has a wrong schema")
    if (
        payload.get("semantic_condition") != QWEN_SEMANTIC_CONDITION
        or payload.get("semantic_condition_status") != QWEN_CONDITION_STATUS
        or payload.get("formal_eligible") is not True
        or payload.get("random_imputation") is not False
        or payload.get("association_label_free") is not True
        or payload.get("association_labels_used") is not False
        or payload.get("target_neutral_prompt") is not True
    ):
        raise QwenSemanticError("Qwen text eligibility contract is false")
    status_path = _verify_artifact(
        payload.get("generation_status"), "Qwen generation status", resolver=resolver
    )
    status = validate_qwen_generation_status(status_path, resolver=resolver)
    expected = _text_contract(status_path=status_path, status=status)
    if resolver is not None:
        expected["generation_status"]["path"] = payload["generation_status"]["path"]
        for entity in ("herb", "disease"):
            expected["mappings"][entity]["path"] = payload["mappings"][entity]["path"]
    observed = {key: value for key, value in payload.items() if key != "created_at"}
    if observed != expected:
        raise QwenSemanticError(
            "Qwen text manifest is not re-derived from its requests and responses"
        )
    return {
        "manifest": payload,
        "manifest_path": manifest_path,
        "mappings": status["request"]["source"]["mappings"],
        "dependencies": list(
            dict.fromkeys([manifest_path, *status["dependencies"]])
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze all target-neutral Qwen requests")
    prepare.add_argument("--source-text-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("generate", help="call Qwen with resumable JSONL output")
    generate.add_argument("--request-manifest", type=Path, required=True)
    generate.add_argument("--responses", type=Path, required=True)
    generate.add_argument("--status", type=Path, required=True)
    generate.add_argument("--workers", type=int, default=6)
    generate.add_argument("--requests-per-minute", type=int, default=50)
    generate.add_argument("--max-retries", type=int, default=5)
    generate.add_argument(
        "--limit",
        type=int,
        help="generate only the first N pending rows (for an auditable API smoke)",
    )
    generate.add_argument("--dry-run", action="store_true")
    finalize = commands.add_parser("finalize", help="freeze a complete Qwen text manifest")
    finalize.add_argument("--generation-status", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate final Qwen text provenance")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = build_qwen_request_manifest(
                source_text_manifest=args.source_text_manifest, output_path=args.output
            )
        elif args.command == "generate":
            request = validate_qwen_request_manifest(args.request_manifest)
            response_path = args.responses.resolve()
            completed = _read_responses(
                response_path,
                requests=request["requests"],
                require_complete=False,
            )
            if args.dry_run:
                pending = [
                    record
                    for entity in ("herb", "disease")
                    for record in request["manifest"]["entities"][entity]["requests"]
                    if (entity, str(record["row_id"])) not in completed
                ]
                print(
                    json.dumps(
                        {
                            "request_count": len(request["requests"]),
                            "completed": len(completed),
                            "pending": len(pending),
                            "sample": pending[:3],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            result = generate_qwen_responses(
                request_manifest=args.request_manifest,
                response_jsonl=args.responses,
                status_path=args.status,
                workers=args.workers,
                requests_per_minute=args.requests_per_minute,
                max_retries=args.max_retries,
                limit=args.limit,
            )
        elif args.command == "finalize":
            result = finalize_qwen_text_manifest(
                generation_status=args.generation_status,
                output_path=args.output,
            )
        else:
            validate_qwen_text_manifest(args.manifest)
            result = args.manifest.resolve()
    except SemanticFeatureError as error:
        print(f"Ethnobotany Qwen semantic command failed: {error}", file=os.sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
