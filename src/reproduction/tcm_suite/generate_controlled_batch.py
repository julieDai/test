#!/usr/bin/env python3
"""Generate several controlled Qwen corpora with one shared RPM limiter.

This wrapper preserves the JSONL format of generate_controlled_descriptions.py,
interleaves jobs, writes only completed responses, and resumes by entity_id.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import time
from pathlib import Path

from generate_controlled_descriptions import (
    SlidingWindowLimiter,
    call_api,
    completed_ids,
    load_entities,
)

try:
    from .job_paths import resolve_job_path
except ImportError:  # Direct script execution.
    from job_paths import resolve_job_path


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def round_robin(queues):
    active = collections.deque(index for index, queue in enumerate(queues) if queue)
    while active:
        index = active.popleft()
        yield index, queues[index].popleft()
        if queues[index]:
            active.append(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen-plus-2025-12-01")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--requests-per-minute", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    jobs_file = args.jobs_file.expanduser().resolve()
    status_file = args.status_file.expanduser().resolve()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")

    job_specs = json.loads(jobs_file.read_text())
    jobs, queues = [], []
    for spec in job_specs:
        mapping = resolve_job_path(spec["mapping"], jobs_file)
        prompt_path = resolve_job_path(spec["prompt"], jobs_file)
        output = resolve_job_path(spec["output"], jobs_file)
        prompt_template = prompt_path.read_text().strip()
        placeholder = "[Herb]" if spec["entity"] == "herb" else "[Disease]"
        if placeholder not in prompt_template:
            raise ValueError(f"{prompt_path} does not contain {placeholder}")
        entities = load_entities(spec["entity"], mapping)
        done = completed_ids(output)
        pending = collections.deque(item for item in entities if item["id"] not in done)
        output.parent.mkdir(parents=True, exist_ok=True)
        jobs.append({
            **spec, "mapping": mapping, "prompt_path": prompt_path,
            "prompt_template": prompt_template, "placeholder": placeholder,
            "output": output, "total": len(entities), "initial_done": len(done),
            "completed_this_run": 0, "prompt_tokens": 0, "completion_tokens": 0,
        })
        queues.append(pending)

    status_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "status": "running", "model": args.model, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "workers": args.workers,
        "requests_per_minute_shared": args.requests_per_minute,
        "started_unix": started, "jobs": [],
    }
    limiter = SlidingWindowLimiter(args.requests_per_minute)
    streams = [job["output"].open("a", buffering=1, encoding="utf-8") for job in jobs]
    failures = []

    def snapshot(final_status=None):
        status["jobs"] = [{
            "entity": job["entity"], "variant": job["variant"],
            "output": str(job["output"].resolve()), "total": job["total"],
            "initial_done": job["initial_done"],
            "completed_this_run": job["completed_this_run"],
            "completed_total": job["initial_done"] + job["completed_this_run"],
            "prompt_tokens": job["prompt_tokens"],
            "completion_tokens": job["completion_tokens"],
        } for job in jobs]
        status["elapsed_seconds"] = time.time() - started
        status["updated_unix"] = time.time()
        if final_status:
            status["status"] = final_status
        atomic_json(status_file, status)

    def generate(job_index, item):
        job = jobs[job_index]
        prompt = job["prompt_template"].replace(job["placeholder"], item["name"])
        result = call_api(
            api_key, limiter, args.model, args.temperature, args.max_tokens,
            prompt, args.max_retries,
        )
        result.update({
            "entity": job["entity"], "variant": job["variant"],
            "entity_id": item["id"], "entity_name": item["name"],
            "model_requested": args.model, "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        })
        return job_index, result

    try:
        snapshot()
        work = iter(round_robin(queues))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            inflight = {}
            for _ in range(args.workers * 2):
                try:
                    job_index, item = next(work)
                except StopIteration:
                    break
                inflight[executor.submit(generate, job_index, item)] = (job_index, item)
            since_snapshot = 0
            while inflight:
                done_set, _ = concurrent.futures.wait(
                    inflight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done_set:
                    fallback_job, item = inflight.pop(future)
                    try:
                        job_index, result = future.result()
                        streams[job_index].write(json.dumps(result, ensure_ascii=False) + "\n")
                        job = jobs[job_index]
                        job["completed_this_run"] += 1
                        usage = result.get("usage") or {}
                        job["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                        job["completion_tokens"] += int(usage.get("completion_tokens", 0))
                    except Exception as exc:
                        failures.append({
                            "job_index": fallback_job, "entity_id": item["id"],
                            "entity_name": item["name"], "error": repr(exc),
                        })
                    since_snapshot += 1
                    try:
                        next_job, next_item = next(work)
                        inflight[executor.submit(generate, next_job, next_item)] = (next_job, next_item)
                    except StopIteration:
                        pass
                if since_snapshot >= 10:
                    snapshot()
                    progress = " ".join(
                        f"{job['variant']}:{job['entity']}="
                        f"{job['initial_done'] + job['completed_this_run']}/{job['total']}"
                        for job in jobs
                    )
                    print(progress, flush=True)
                    since_snapshot = 0
        if failures:
            failure_path = status_file.with_suffix(".failures.json")
            failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n")
            snapshot("failed")
            raise SystemExit(f"{len(failures)} calls failed; resume will retry them")
        snapshot("success")
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
