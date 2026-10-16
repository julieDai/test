#!/usr/bin/env python3
"""Run protocol-aligned layer-depth/fold tasks with isolated outputs."""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def parse_metrics(values):
    return {name: float(raw) for name, raw in (value.split(":", 1) for value in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Comma-separated layer:fold tasks")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir", type=Path,
        help="TCM-suite data directory (default: directory recorded in protocol.json).",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol_dir, run_dir = args.protocol_dir.resolve(), args.run_dir.resolve()
    src_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src_dir))
    from herbhg.experiments.runtime import activate_tcm_backend

    backend_dir = activate_tcm_backend(src_dir)
    from astra.runtime import (
        configure_run_environment,
        configure_tcm_data_paths,
        tcm_dataset_directory,
    )

    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    frozen_negative = Path(protocol["frozen_negative_file"])
    if not frozen_negative.is_absolute():
        frozen_negative = (protocol_dir / frozen_negative).resolve()
    data_dir = tcm_dataset_directory(
        protocol,
        protocol_dir=protocol_dir,
        override=args.dataset_dir,
    )
    configure_run_environment(
        negative_file=frozen_negative,
        seed=args.seed,
        allow_missing_semantic=False,
    )

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from sensitivity_analysis_of_network_layers.layer_sensitivity import HGHDALayerSensitivity
    from util.config import ModelConf
    from util.io import FileIO

    config = ModelConf(backend_dir / "ASTRA.conf")
    configure_tcm_data_paths(
        config,
        data_dir,
        negative_file=frozen_negative,
    )
    all_data = FileIO.loadDataSet(config, config["datapath"])
    if len(all_data) != protocol["total_rows"]:
        raise RuntimeError("Loaded sample count differs from frozen protocol")

    for item in args.tasks.split(","):
        layer, fold = (int(x) for x in item.split(":"))
        if layer not in range(1, 9) or fold not in range(1, 6):
            raise ValueError(item)
        task_dir = run_dir / f"layer_{layer}" / f"fold_{fold}"
        status_path = task_dir / "status.json"
        if args.resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "success":
            print(f"{layer}:{fold} already successful, skipping", flush=True)
            continue
        output_dir, log_dir = task_dir / "outputs", task_dir / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        configure_run_environment(output_dir=output_dir, log_dir=log_dir)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)
        modulo = fold - 1
        train = [row for index, row in enumerate(all_data) if index % 5 != modulo]
        test = [row for index, row in enumerate(all_data) if index % 5 == modulo]
        started = time.time()
        status = {"status": "running", "layer": layer, "fold": fold, "seed": args.seed,
                  "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "train_rows": len(train),
                  "test_rows": len(test), "started_unix": started}
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        try:
            model = HGHDALayerSensitivity(config, train, test, f"[{fold}]", n_layers=layer)
            status.update({"status": "success", "metrics": parse_metrics(model.execute()),
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
        except Exception as exc:
            status.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"layer={layer} fold={fold} {status['metrics']}", flush=True)
        del model, train, test


if __name__ == "__main__":
    main()
