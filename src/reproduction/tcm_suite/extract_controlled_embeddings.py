#!/usr/bin/env python3
"""Extract aligned 768-D SciBERT embeddings from controlled JSONL text.

Unlike the legacy Mac scripts, alignment is by explicit entity identifier and
the actual node order used by Rating (first occurrence in H_C/P_D), so missing
descriptions cannot silently shift all later embedding rows.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            entity_id = str(record["entity_id"])
            description = str(record.get("description", "")).strip()
            if not description:
                continue
            if entity_id in records:
                raise ValueError(f"duplicate entity_id {entity_id} at {path}:{line_number}")
            records[entity_id] = description
    return records


def node_order(path, column):
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    return list(dict.fromkeys(frame[int(column)].astype(str).tolist()))


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", choices=["herb", "disease"], required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--node-file", type=Path, required=True)
    parser.add_argument("--node-column", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if args.resume and args.output.exists() and metadata_path.exists():
        print(f"already complete: {args.output}")
        return

    started = time.time()
    id_column = "herb_id" if args.entity == "herb" else "disease_id"
    mapping = pd.read_csv(args.mapping, encoding="utf-8-sig", dtype=str)
    mapping_ids = mapping[id_column].astype(str).tolist()
    ordered_ids = node_order(args.node_file, args.node_column)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("node order unexpectedly contains duplicates")
    if set(mapping_ids) != set(ordered_ids):
        missing_mapping = sorted(set(ordered_ids) - set(mapping_ids))[:10]
        missing_nodes = sorted(set(mapping_ids) - set(ordered_ids))[:10]
        raise ValueError(
            f"mapping/node ID mismatch: missing_mapping={missing_mapping}, missing_nodes={missing_nodes}"
        )

    descriptions = load_jsonl(args.texts)
    present_ids = [entity_id for entity_id in ordered_ids if entity_id in descriptions]
    missing_ids = [entity_id for entity_id in ordered_ids if entity_id not in descriptions]
    if not present_ids:
        raise ValueError(f"no descriptions align with {args.node_file}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    extracted = {}
    with torch.inference_mode():
        for start in range(0, len(present_ids), args.batch_size):
            batch_ids = present_ids[start:start + args.batch_size]
            batch_text = [descriptions[entity_id] for entity_id in batch_ids]
            inputs = tokenizer(
                batch_text, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            pooled = mean_pool(model(**inputs).last_hidden_state, inputs["attention_mask"])
            values = pooled.detach().cpu().numpy().astype(np.float32)
            extracted.update(zip(batch_ids, values))
            done = min(start + args.batch_size, len(present_ids))
            if done % (args.batch_size * 20) == 0 or done == len(present_ids):
                print(f"{args.entity}: {done}/{len(present_ids)}", flush=True)

    mean_embedding = np.mean(np.stack(list(extracted.values())), axis=0)
    aligned = np.stack([extracted.get(entity_id, mean_embedding) for entity_id in ordered_ids]).astype(np.float32)
    if aligned.shape != (len(ordered_ids), model.config.hidden_size):
        raise RuntimeError(f"unexpected output shape: {aligned.shape}")
    if not np.isfinite(aligned).all():
        raise RuntimeError("embedding output contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.npy")
    np.save(temporary, aligned)
    temporary.replace(args.output)
    metadata = {
        "status": "success",
        "entity": args.entity,
        "rows": int(aligned.shape[0]),
        "dimensions": int(aligned.shape[1]),
        "present_descriptions": len(present_ids),
        "missing_descriptions_filled_with_present_mean": len(missing_ids),
        "missing_ids": missing_ids,
        "alignment": "first-occurrence node order from node_file",
        "pooling": "attention-mask mean including model special tokens (legacy-compatible)",
        "model": args.model,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "source_texts": str(args.texts.resolve()),
        "source_texts_sha256": sha256(args.texts),
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256(args.mapping),
        "node_file": str(args.node_file.resolve()),
        "node_file_sha256": sha256(args.node_file),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
