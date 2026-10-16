#!/usr/bin/env python3
"""Cold-start sensitivity experiment for the current text-enhanced branch.

Each task trains one model for one leave-one-herb-out target and one fusion
coefficient.  alpha=1 reproduces the current HerbHG prediction branch;
alpha=0 selects the text-free structural branch built in the same graph.
"""

import argparse
import gc
import json
import os
import platform
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


def parse_tasks(value):
    tasks = []
    for item in value.split(","):
        alpha_text, experiment_text = item.split(":", 1)
        alpha = float(alpha_text)
        experiment = int(experiment_text)
        if not 0.0 <= alpha <= 1.0:
            raise argparse.ArgumentTypeError(f"alpha must be in [0,1]: {alpha}")
        if experiment not in range(1, 6):
            raise argparse.ArgumentTypeError(f"experiment must be 1..5: {experiment}")
        tasks.append((alpha, experiment))
    if len(tasks) != len(set(tasks)):
        raise argparse.ArgumentTypeError("duplicate alpha:experiment task")
    return tasks


def alpha_slug(alpha):
    return f"alpha_{alpha:.2f}".replace(".", "p")


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


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
    parser.add_argument(
        "--tasks", required=True,
        help="Comma-separated alpha:experiment pairs, e.g. 0:1,0.25:2,1:5",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", default="legacy_embeddings")
    parser.add_argument("--herb-text-embeddings", type=Path)
    parser.add_argument("--disease-text-embeddings", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    dataset_dir = args.dataset_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src_dir))
    from herbhg.experiments.runtime import activate_tcm_backend

    backend_dir = activate_tcm_backend(src_dir)
    from astra.runtime import configure_run_environment, configure_tcm_data_paths

    configure_run_environment(seed=args.seed, allow_missing_semantic=False)
    if bool(args.herb_text_embeddings) != bool(args.disease_text_embeddings):
        raise ValueError("herb and disease text embedding paths must be provided together")
    if args.herb_text_embeddings:
        herb_text_path = args.herb_text_embeddings.resolve()
        disease_text_path = args.disease_text_embeddings.resolve()
        if not herb_text_path.exists() or not disease_text_path.exists():
            raise FileNotFoundError(f"missing text embeddings: {herb_text_path}, {disease_text_path}")
        configure_run_environment(
            herb_text_embeddings=herb_text_path,
            disease_text_embeddings=disease_text_path,
        )

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from HerbHG import HerbHG
    from util.config import ModelConf

    targets, compound_counts, top20 = choose_targets(dataset_dir, args.seed)
    selection = {
        "seed": args.seed,
        "condition": args.condition,
        "herb_text_embeddings": os.environ.get("ASTRA_HERB_TEXT_EMBEDDINGS", "configured default"),
        "disease_text_embeddings": os.environ.get("ASTRA_DISEASE_TEXT_EMBEDDINGS", "configured default"),
        "top20_by_compound_degree": top20,
        "selected": targets,
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "definition": "(1-alpha)*structural_branch + alpha*current_text_enhanced_branch",
        "alpha_1_anchor": "current HerbHG prediction representation",
        "alpha_0_anchor": "current text-free structural prediction representation",
    }
    selection_path = run_dir / "experiment_design.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        if existing != selection:
            raise RuntimeError(f"experiment design mismatch: {selection_path}")
    else:
        atomic_json(selection_path, selection)

    positives = load_rows(dataset_dir / "H_D.txt", 1)
    negative_pool = load_rows(dataset_dir / "negativehdlist.txt", 0)
    disease_ids = list(dict.fromkeys(row[1] for row in load_rows(dataset_dir / "P_D.txt")))
    config = ModelConf(backend_dir / "ASTRA.conf")
    configure_tcm_data_paths(config, dataset_dir)
    failures = []

    for alpha, exp_idx in tasks:
        target = targets[exp_idx - 1]
        task_dir = run_dir / alpha_slug(alpha) / f"exp_{exp_idx}_herb_{target}"
        status_path = task_dir / "status.json"
        if args.resume and status_path.exists():
            old_status = json.loads(status_path.read_text())
            if old_status.get("status") == "success":
                print(f"alpha={alpha:.2f} experiment={exp_idx} already successful; skipping", flush=True)
                continue
        output_dir, log_dir = task_dir / "outputs", task_dir / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        configure_run_environment(output_dir=output_dir, log_dir=log_dir)

        train_positive = [row for row in positives if row[0] != target]
        train_negative_pool = [row for row in negative_pool if row[0] != target]
        # Keep the sampled negatives identical across all alpha values for a target.
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
            "text_fusion_alpha": alpha, "seed": args.seed,
            "condition": args.condition,
            "herb_text_embeddings": os.environ.get("ASTRA_HERB_TEXT_EMBEDDINGS", "configured default"),
            "disease_text_embeddings": os.environ.get("ASTRA_DISEASE_TEXT_EMBEDDINGS", "configured default"),
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_compounds": compound_counts[target], "train_rows": len(train),
            "test_rows": len(labels), "started_unix": started,
            "python": platform.python_version(), "tensorflow": tf.__version__,
            "fusion_definition": "(1-alpha)*structural_branch + alpha*current_text_enhanced_branch",
        }
        atomic_json(status_path, status)
        model = None
        try:
            model = HerbHG(
                config,
                train,
                [],
                f"[text-alpha-{alpha:.2f}-cold-{exp_idx}]",
                text_fusion_alpha=alpha,
            )
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
            np.savez_compressed(task_dir / "predictions.npz", scores=scores, labels=labels,
                                disease_ids=np.asarray(disease_ids))
            status.update({"status": "success", "metrics": metrics,
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
        except Exception as exc:
            status.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                           "elapsed_seconds": time.time() - started, "finished_unix": time.time()})
            failures.append((alpha, exp_idx, repr(exc)))
        finally:
            atomic_json(status_path, status)
            if model is not None and getattr(model, "sess", None) is not None:
                model.sess.close()
            del model, train, train_positive, train_negative_pool, train_negative
            gc.collect()
        print(f"alpha={alpha:.2f} cold={exp_idx} herb={target} status={status['status']} "
              f"metrics={status.get('metrics')}", flush=True)

    if failures:
        print(f"failed tasks: {failures}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
