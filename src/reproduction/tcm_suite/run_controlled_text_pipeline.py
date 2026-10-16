#!/usr/bin/env python3
"""Continue the controlled-text experiment after API generation completes.

The service waits without consuming GPU, then performs aligned SciBERT
extraction, one shared reducer fit, derived controls, and 35 cold-start tasks.
Every stage is resumable through its own output/status files.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_parallel(commands, environments, status, status_path, stage):
    status.update({"stage": stage, "stage_started_unix": time.time()})
    atomic_json(status_path, status)
    processes = []
    for command, environment in zip(commands, environments):
        merged = os.environ.copy()
        merged.update(environment)
        processes.append(subprocess.Popen(command, env=merged))
    while any(process.poll() is None for process in processes):
        status["stage_processes"] = [process.poll() for process in processes]
        status["updated_unix"] = time.time()
        atomic_json(status_path, status)
        time.sleep(30)
    return_codes = [process.returncode for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"stage {stage} failed: {return_codes}")
    status["stage_processes"] = return_codes
    atomic_json(status_path, status)


def run_one(command, environment, status, status_path, stage):
    run_parallel([command], [environment], status, status_path, stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    root = args.experiment_root.resolve()
    dataset = args.dataset_dir.resolve()
    status_path = root / "pipeline_status.json"
    status = {"status": "running", "stage": "waiting_for_api", "started_unix": time.time()}
    atomic_json(status_path, status)

    api_status_path = root / "api_generation_status.json"
    template_outputs = [
        root / "embeddings_768d/structured_clean_herb.npy.metadata.json",
        root / "embeddings_768d/structured_clean_disease.npy.metadata.json",
        root / "embeddings_768d/structured_with_indication_herb.npy.metadata.json",
        root / "embeddings_768d/name_only_herb.npy.metadata.json",
        root / "embeddings_768d/name_only_disease.npy.metadata.json",
    ]
    while True:
        api = json.loads(api_status_path.read_text()) if api_status_path.exists() else {}
        status["api_status"] = api.get("status", "not_started")
        status["api_progress"] = [
            {
                "variant": job.get("variant"), "entity": job.get("entity"),
                "completed": job.get("completed_total"), "total": job.get("total"),
            }
            for job in api.get("jobs", [])
        ]
        status["template_embeddings_ready"] = sum(path.exists() for path in template_outputs)
        status["updated_unix"] = time.time()
        atomic_json(status_path, status)
        if api.get("status") == "success" and all(path.exists() for path in template_outputs):
            break
        time.sleep(args.poll_seconds)

    queue_script = project / "src/reproduction/tcm_suite/run_embedding_queue.py"
    clean_command = [
        sys.executable, str(queue_script), "--jobs-file", str(root / "llm_jobs_gpu0.json"),
        "--status-file", str(root / "llm_embed_gpu0_status.json"),
    ]
    original_command = [
        sys.executable, str(queue_script), "--jobs-file", str(root / "llm_jobs_gpu1.json"),
        "--status-file", str(root / "llm_embed_gpu1_status.json"),
    ]
    run_parallel(
        [clean_command, original_command],
        [{"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
         {"CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}],
        status, status_path, "scibert_llm_embeddings",
    )

    reducer_command = [
        sys.executable, str(project / "src/reproduction/tcm_suite/train_shared_text_reducer.py"),
        "--jobs-file", str(root / "shared_reducer_jobs.json"),
        "--output-dir", str(root / "shared_reducer"),
        "--device", "cuda:0", "--epochs", "50", "--batch-size", "128", "--seed", "42", "--resume",
    ]
    run_one(
        reducer_command, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
        status, status_path, "shared_768_to_64_reducer",
    )

    control_command = [
        sys.executable, str(project / "src/reproduction/tcm_suite/build_embedding_controls.py"),
        "--clean-herb", str(root / "embeddings_64d/llm_clean/herb_semantic_embeddings_64d.npy"),
        "--clean-disease", str(root / "embeddings_64d/llm_clean/disease_semantic_embeddings_64d.npy"),
        "--output-root", str(root / "embeddings_64d"), "--seed", "42",
    ]
    run_one(control_command, {"PYTHONUNBUFFERED": "1"}, status, status_path, "derived_controls")

    condition_script = project / "src/reproduction/tcm_suite/run_text_condition_queue.py"
    condition_commands = []
    for gpu in (0, 1):
        condition_commands.append([
            sys.executable, str(condition_script),
            "--jobs-file", str(root / f"condition_jobs_gpu{gpu}.json"),
            "--dataset-dir", str(dataset),
            "--results-root", str(root / "cold_start_results"),
            "--status-file", str(root / f"condition_gpu{gpu}_status.json"),
            "--seed", "42",
        ])
    run_parallel(
        condition_commands,
        [{"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"},
         {"CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}],
        status, status_path, "cold_start_35_tasks",
    )
    status.update({"status": "success", "stage": "complete", "finished_unix": time.time()})
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
