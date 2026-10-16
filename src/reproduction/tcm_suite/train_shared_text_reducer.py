#!/usr/bin/env python3
"""Train one shared 768->64 autoencoder and transform all text conditions."""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .job_paths import resolve_job_path
except ImportError:  # Direct script execution.
    from job_paths import resolve_job_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 768),
        )

    def encode(self, values):
        return F.normalize(self.encoder(values), p=2, dim=1)

    def forward(self, values):
        encoded = self.encode(values)
        return encoded, self.decoder(encoded)


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_array(path):
    values = np.load(path).astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 768:
        raise ValueError(f"expected N x 768 array at {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values at {path}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jobs_file = args.jobs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    jobs = json.loads(jobs_file.read_text())
    fit_paths = [resolve_job_path(path, jobs_file) for path in jobs["fit_embeddings"]]
    transforms = [
        {
            **job,
            "input": resolve_job_path(job["input"], jobs_file),
            "output": resolve_job_path(job["output"], jobs_file),
        }
        for job in jobs["transforms"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    checkpoint_path = output_dir / "shared_reducer.pt"
    if args.resume and status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get("status") == "success" and all(job["output"].exists() for job in transforms):
            print("shared reducer and all transforms already complete")
            return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    started = time.time()
    fit_arrays = [load_array(path) for path in fit_paths]
    fit = np.concatenate(fit_arrays, axis=0)
    dataset = TensorDataset(torch.from_numpy(fit))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    status = {
        "status": "training", "seed": args.seed, "device": str(device),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "fit_rows": int(fit.shape[0]),
        "fit_embeddings": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in fit_paths],
        "losses": [], "started_unix": started,
    }
    atomic_json(status_path, status)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, decoded = model(batch)
            reconstruction = F.mse_loss(decoded, batch)
            l2 = sum(0.5 * parameter.square().sum() for name, parameter in model.named_parameters() if "bias" not in name)
            loss = reconstruction + 0.0001 * l2
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        epoch_loss = total / count
        status["losses"].append(epoch_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.8f}", flush=True)
        status["last_epoch"] = epoch + 1
        atomic_json(status_path, status)

    torch.save({
        "state_dict": model.state_dict(), "architecture": "768-256-128-64-128-256-768",
        "seed": args.seed, "epochs": args.epochs,
    }, checkpoint_path)
    model.eval()
    output_metadata = []
    with torch.inference_mode():
        for job in transforms:
            source = job["input"]
            output = job["output"]
            values = load_array(source)
            encoded_batches = []
            for start in range(0, len(values), 512):
                batch = torch.from_numpy(values[start:start + 512]).to(device)
                encoded_batches.append(model.encode(batch).cpu().numpy().astype(np.float32))
            encoded = np.concatenate(encoded_batches, axis=0)
            if encoded.shape != (len(values), 64) or not np.isfinite(encoded).all():
                raise RuntimeError(f"bad encoded output for {job}: {encoded.shape}")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.tmp.npy")
            np.save(temporary, encoded)
            temporary.replace(output)
            output_metadata.append({
                "condition": job["condition"], "entity": job["entity"],
                "input": str(source.resolve()), "input_sha256": sha256(source),
                "output": str(output.resolve()), "output_sha256": sha256(output),
                "shape": list(encoded.shape),
            })
            print(f"encoded {job['condition']} {job['entity']}: {encoded.shape}", flush=True)

    status.update({
        "status": "success", "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "outputs": output_metadata, "python": platform.python_version(), "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
