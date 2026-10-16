#!/usr/bin/env python3
"""Run controlled SciBERT extraction jobs sequentially on one visible GPU."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from .job_paths import resolve_job_path
except ImportError:  # Direct script execution.
    from job_paths import resolve_job_path


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    jobs_file = args.jobs_file.expanduser().resolve()
    status_file = args.status_file.expanduser().resolve()
    jobs = json.loads(jobs_file.read_text())
    extractor = Path(__file__).with_name("extract_controlled_embeddings.py")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "jobs_file": str(jobs_file), "jobs": []}
    atomic_json(status_file, status)

    for index, job in enumerate(jobs, 1):
        mapping = resolve_job_path(job["mapping"], jobs_file)
        texts = resolve_job_path(job["texts"], jobs_file)
        node_file = resolve_job_path(job["node_file"], jobs_file)
        output = resolve_job_path(job["output"], jobs_file)
        metadata = output.with_suffix(output.suffix + ".metadata.json")
        item = {
            "index": index,
            "entity": job["entity"],
            "texts": str(texts),
            "output": str(output),
        }
        status["jobs"].append(item)
        if output.exists() and metadata.exists():
            item["status"] = "skipped_complete"
            atomic_json(status_file, status)
            continue
        command = [
            sys.executable, str(extractor),
            "--entity", job["entity"],
            "--mapping", str(mapping),
            "--texts", str(texts),
            "--node-file", str(node_file),
            "--node-column", str(job["node_column"]),
            "--output", str(output),
            "--batch-size", str(job.get("batch_size", 32)),
            "--max-length", str(job.get("max_length", 512)),
            "--device", job.get("device", "cuda:0"),
            "--resume",
        ]
        item.update({"status": "running", "started_unix": time.time(), "command": command})
        atomic_json(status_file, status)
        try:
            subprocess.run(command, check=True)
            item.update({"status": "success", "finished_unix": time.time()})
        except Exception as exc:
            item.update({"status": "failed", "finished_unix": time.time(), "error": repr(exc)})
            status["status"] = "failed"
            atomic_json(status_file, status)
            raise
        atomic_json(status_file, status)

    status["status"] = "success"
    status["finished_unix"] = time.time()
    atomic_json(status_file, status)


if __name__ == "__main__":
    main()
