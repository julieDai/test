#!/usr/bin/env python3
"""Five-repeat leave-one-herb-out cold-start experiment for HerbHG."""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def load_rows(path, label=None):
    rows = []
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            rows.append([parts[0], parts[1], float(parts[2]) if label is None else float(label)])
    return rows


def choose_targets(dataset_dir, seed, count=5):
    hc = pd.read_csv(dataset_dir / "H_C.txt", sep="\t", header=None, dtype=str)
    counts = hc.groupby(0)[1].nunique().to_dict()
    top20 = sorted(counts, key=lambda herb: (-counts[herb], int(herb)))[:20]
    return random.Random(seed).sample(top20, count), counts, top20


def evaluate(scores, labels):
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    result = {
        "AUC": float(roc_auc_score(labels, scores)),
        "AUPR": float(auc(recall_curve, precision_curve)),
    }
    best = None
    for threshold in np.linspace(0, 1, 100):
        pred = scores >= threshold
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        tn = int(np.sum((~pred) & (labels == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold, precision, recall, (tp + tn) / len(labels))
        if best is None or candidate[0] > best[0]:
            best = candidate
    result.update({
        "F1-Score": best[0], "Best_Threshold": best[1],
        "Precision": best[2], "Recall": best[3], "Accuracy": best[4],
    })
    order = np.argsort(-scores)
    ranked = labels[order]
    hits10 = int(ranked[:10].sum())
    positive_count = int(labels.sum())
    relevant_ranks = np.flatnonzero(ranked == 1) + 1
    ideal = min(10, positive_count)
    dcg = sum(1 / np.log2(rank + 1) for rank in relevant_ranks if rank <= 10)
    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal + 1))
    result.update({
        "positive_count": positive_count,
        "negative_count": int(len(labels) - positive_count),
        "Hits@10": hits10,
        "Precision@10": hits10 / 10,
        "Recall@10": hits10 / positive_count,
        "MRR": 1 / float(relevant_ranks.min()),
        "NDCG@10": dcg / idcg if idcg else 0.0,
    })
    return result, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", required=True, help="Comma-separated indices 1..5")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src_dir))
    from herbhg.experiments.runtime import activate_tcm_backend

    backend_dir = activate_tcm_backend(src_dir)
    from astra.runtime import configure_run_environment, configure_tcm_data_paths

    configure_run_environment(seed=args.seed, allow_missing_semantic=False)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from HerbHG import HerbHG
    from util.config import ModelConf

    targets, compound_counts, top20 = choose_targets(dataset_dir, args.seed)
    selection = {"seed": args.seed, "top20": top20, "selected": targets}
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    positives = load_rows(dataset_dir / "H_D.txt", 1)
    negative_pool = load_rows(dataset_dir / "negativehdlist.txt", 0)
    disease_ids = list(dict.fromkeys(row[1] for row in load_rows(dataset_dir / "P_D.txt")))
    disease_index = {disease: index for index, disease in enumerate(disease_ids)}
    config = ModelConf(backend_dir / "ASTRA.conf")
    configure_tcm_data_paths(config, dataset_dir)

    for exp_idx in [int(x) for x in args.experiments.split(",")]:
        target = targets[exp_idx - 1]
        task_dir = run_dir / f"exp_{exp_idx}_herb_{target}"
        status_path = task_dir / "status.json"
        if args.resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "success":
            print(f"Experiment {exp_idx} already successful, skipping", flush=True)
            continue
        output_dir, log_dir = task_dir / "outputs", task_dir / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        configure_run_environment(output_dir=output_dir, log_dir=log_dir)

        train_positive = [row for row in positives if row[0] != target]
        train_negative_pool = [row for row in negative_pool if row[0] != target]
        train_negative = random.Random(args.seed + exp_idx).sample(train_negative_pool, len(train_positive))
        train = train_positive + train_negative
        positive_diseases = {row[1] for row in positives if row[0] == target}
        labels = np.asarray([1 if disease in positive_diseases else 0 for disease in disease_ids], dtype=int)

        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)
        started = time.time()
        status = {
            "status": "running", "experiment": exp_idx, "target_herb": target,
            "seed": args.seed, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_compounds": compound_counts[target], "train_rows": len(train),
            "test_rows": len(labels), "started_unix": started,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        try:
            model = HerbHG(config, train, [], f"[cold-{exp_idx}]")
            model.readConfiguration()
            model.initializing_log()
            model.initModel()
            model.trainModel()
            herb_idx = model.data.herb[target]
            raw = model.u[herb_idx].dot(model.i.T)
            scores = 1 / (1 + np.exp(-raw))
            metrics, order = evaluate(scores, labels)
            ranking = pd.DataFrame({
                "rank": np.arange(1, len(labels) + 1),
                "disease_id": np.asarray(disease_ids)[order],
                "score": scores[order], "label": labels[order],
            })
            ranking.to_csv(task_dir / "ranking.csv", index=False)
            status.update({"status": "success", "metrics": metrics,
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
        except Exception as exc:
            status.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"cold:{exp_idx} herb={target} {status['metrics']}", flush=True)
        del model, train, train_positive, train_negative_pool, train_negative


if __name__ == "__main__":
    main()
