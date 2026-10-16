#!/usr/bin/env python3
"""Generate resumable Qwen descriptions for controlled prompt experiments.

The API key is accepted only through DASHSCOPE_API_KEY. Results are appended as
JSONL by the main process, so interrupted runs can resume without losing valid
responses or exposing credentials in command lines.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute):
        self.limit = requests_per_minute
        self.timestamps = collections.deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.limit:
                    self.timestamps.append(now)
                    return
                delay = 60 - (now - self.timestamps[0]) + 0.05
            time.sleep(delay)


def load_entities(entity, mapping_path):
    frame = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype=str)
    if entity == "herb":
        id_column = "herb_id"
        name_columns = ["englishName", "pinyinName", "latinName", "xbChineseName"]
    else:
        id_column = "disease_id"
        name_columns = ["englishName", "xbDiseaseName"]
    entities = []
    for _, row in frame.iterrows():
        name = None
        for column in name_columns:
            value = row.get(column)
            if isinstance(value, str) and value.strip() and value.lower() != "nan":
                name = value.strip()
                break
        # Match the legacy generator: entities without a usable name are skipped
        # and later receive a documented mean embedding rather than fabricated text.
        if name is not None:
            entities.append({"id": str(row[id_column]), "name": name})
    return entities


def completed_ids(path):
    done = set()
    if not path.exists():
        return done
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["entity_id"]))
            except Exception as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return done


def call_api(api_key, limiter, model, temperature, max_tokens, prompt, max_retries):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the requested description without headings."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=(15, 180))
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            return {
                "description": choice["message"]["content"].strip(),
                "response_model": body.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage", {}),
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt + 1 < max_retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", choices=["herb", "disease"], required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen-plus-2025-12-01")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt_template = args.prompt.read_text().strip()
    placeholder = "[Herb]" if args.entity == "herb" else "[Disease]"
    if placeholder not in prompt_template:
        raise ValueError(f"prompt does not contain {placeholder}")
    entities = load_entities(args.entity, args.mapping)
    done = completed_ids(args.output)
    pending = [item for item in entities if item["id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]

    if args.dry_run:
        for item in pending[:3]:
            print(json.dumps({
                "entity": args.entity, "variant": args.variant, "entity_id": item["id"],
                "entity_name": item["name"],
                "prompt": prompt_template.replace(placeholder, item["name"]),
            }, ensure_ascii=False, indent=2))
        print(f"total={len(entities)} done={len(done)} pending_selected={len(pending)}")
        return

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    limiter = SlidingWindowLimiter(args.requests_per_minute)

    def generate(item):
        prompt = prompt_template.replace(placeholder, item["name"])
        result = call_api(
            api_key, limiter, args.model, args.temperature, args.max_tokens,
            prompt, args.max_retries,
        )
        result.update({
            "entity": args.entity, "variant": args.variant,
            "entity_id": item["id"], "entity_name": item["name"],
            "model_requested": args.model, "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        })
        return result

    failures = []
    with args.output.open("a", buffering=1) as output_stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(generate, item): item for item in pending}
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                item = future_map[future]
                try:
                    result = future.result()
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        print(f"completed={completed}/{len(pending)} entity={args.entity} variant={args.variant}", flush=True)
                except Exception as exc:
                    failures.append({"entity_id": item["id"], "entity_name": item["name"],
                                     "error": repr(exc)})
    if failures:
        failure_path = args.output.with_suffix(args.output.suffix + ".failures.json")
        failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
        raise SystemExit(f"{len(failures)} requests failed; see {failure_path}")


if __name__ == "__main__":
    main()
