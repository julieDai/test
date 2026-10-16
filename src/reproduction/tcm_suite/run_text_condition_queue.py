#!/usr/bin/env python3
"""Run text-condition cold-start jobs sequentially on one visible GPU."""

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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    jobs_file = args.jobs_file.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    status_file = args.status_file.expanduser().resolve()
    jobs = json.loads(jobs_file.read_text())
    runner = Path(__file__).with_name("run_text_weight_cold_start.py")
    results_root.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "seed": args.seed, "jobs": []}
    atomic_json(status_file, status)

    for index, job in enumerate(jobs, 1):
        run_dir = results_root / job["condition"]
        herb_embeddings = resolve_job_path(job["herb"], jobs_file)
        disease_embeddings = resolve_job_path(job["disease"], jobs_file)
        item = {"index": index, "condition": job["condition"], "run_dir": str(run_dir.resolve())}
        status["jobs"].append(item)
        command = [
            sys.executable, str(runner),
            "--tasks", "0.25:1,0.25:2,0.25:3,0.25:4,0.25:5",
            "--dataset-dir", str(dataset_dir),
            "--run-dir", str(run_dir),
            "--seed", str(args.seed),
            "--condition", job["condition"],
            "--herb-text-embeddings", str(herb_embeddings),
            "--disease-text-embeddings", str(disease_embeddings),
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

    status.update({"status": "success", "finished_unix": time.time()})
    atomic_json(status_file, status)


if __name__ == "__main__":
    main()
