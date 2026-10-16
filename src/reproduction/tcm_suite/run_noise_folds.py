#!/usr/bin/env python3
"""Run fixed-test-fold TCM-suite noise experiments with resumable outputs."""

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


def perturb_train_labels(train, ratio, seed, valid_herbs, valid_diseases):
    result = [row[:] for row in train]
    positives = np.array([i for i, row in enumerate(result) if float(row[2]) > 0], dtype=np.int64)
    negatives = np.array([i for i, row in enumerate(result)
                          if float(row[2]) <= 0 and row[0] in valid_herbs and row[1] in valid_diseases],
                         dtype=np.int64)
    n_swap = int(round(min(len(positives), len(negatives)) * ratio))
    rng = np.random.RandomState(seed)
    pos = rng.choice(positives, n_swap, replace=False)
    neg = rng.choice(negatives, n_swap, replace=False)
    for i in pos:
        result[int(i)][2] = 0.0
    for i in neg:
        result[int(i)][2] = 1.0
    return result, n_swap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="Comma-separated percent:fold tasks")
    p.add_argument("--protocol-dir", type=Path, required=True)
    p.add_argument(
        "--dataset-dir", type=Path,
        help="TCM-suite data directory (default: directory recorded in protocol.json).",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.protocol_dir = args.protocol_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src))
    from herbhg.experiments.runtime import activate_tcm_backend

    backend_dir = activate_tcm_backend(src)
    from astra.runtime import (
        configure_run_environment,
        configure_tcm_data_paths,
        tcm_dataset_directory,
    )

    protocol = json.loads((args.protocol_dir / "protocol.json").read_text())
    frozen_negative = Path(protocol["frozen_negative_file"])
    if not frozen_negative.is_absolute():
        frozen_negative = (args.protocol_dir / frozen_negative).resolve()
    data_dir = tcm_dataset_directory(
        protocol,
        protocol_dir=args.protocol_dir,
        override=args.dataset_dir,
    )
    configure_run_environment(
        negative_file=frozen_negative,
        seed=args.seed,
        allow_missing_semantic=False,
    )

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from HerbHG import HerbHG
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
        raise RuntimeError("Loaded rows differ from frozen protocol")
    for item in args.tasks.split(","):
        percent, fold = (int(x) for x in item.split(":"))
        if percent not in range(10, 100, 10) or fold not in range(1, 6):
            raise ValueError(item)
        task = args.run_dir / f"noise_{percent:02d}" / f"fold_{fold}"
        status_path = task / "status.json"
        if args.resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "success":
            continue
        assoc = args.run_dir / f"noise_{percent:02d}" / "associations"
        configure_run_environment(
            hc_file=(assoc / "H_C.txt").resolve(),
            pd_file=(assoc / "P_D.txt").resolve(),
        )
        valid_herbs = {line.split("\t", 1)[0] for line in (data_dir / "H_C.txt").open()}
        valid_diseases = {line.rstrip("\n").split("\t")[1] for line in (data_dir / "P_D.txt").open()}
        (task / "outputs").mkdir(parents=True, exist_ok=True)
        (task / "logs").mkdir(parents=True, exist_ok=True)
        configure_run_environment(
            output_dir=(task / "outputs").resolve(),
            log_dir=(task / "logs").resolve(),
        )
        modulo = fold - 1
        clean_train = [row for i, row in enumerate(all_data) if i % 5 != modulo]
        test = [row for i, row in enumerate(all_data) if i % 5 == modulo]
        train, swapped = perturb_train_labels(clean_train, percent / 100.0,
                                               args.seed + percent * 100 + fold,
                                               valid_herbs, valid_diseases)
        random.seed(args.seed); np.random.seed(args.seed); tf.reset_default_graph(); tf.set_random_seed(args.seed)
        started = time.time()
        status = {"status": "running", "percent": percent, "fold": fold, "seed": args.seed,
                  "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "train_rows": len(train),
                  "test_rows": len(test), "swapped_positive_negative_pairs": swapped,
                  "test_labels_unchanged": True, "started_unix": started}
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        try:
            model = HerbHG(config, train, test, f"[noise{percent}-fold{fold}]")
            status.update(status="success", metrics=parse_metrics(model.execute()),
                          elapsed_seconds=time.time() - started, finished_unix=time.time())
        except Exception as exc:
            status.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                          elapsed_seconds=time.time() - started, finished_unix=time.time())
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"noise={percent} fold={fold} {status['metrics']}", flush=True)
        del model, train, clean_train, test


if __name__ == "__main__":
    main()
