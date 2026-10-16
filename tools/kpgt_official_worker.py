#!/usr/bin/env python
"""Python-3.7-compatible worker for the pinned official KPGT environment.

This file deliberately has no dependency on the ``astra`` package. The
maintained Python 3.10 CLI starts it with an explicitly selected interpreter
and communicates over a dedicated control file descriptor. Third-party
stdout/stderr output therefore cannot corrupt the protocol.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


INPUT_SCHEMA = "astra-kpgt-worker-input/v1"
PROTOCOL_SCHEMA = "astra-kpgt-worker-protocol/v1"
OUTPUT_DIMENSION = 2304


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_output(repo, *arguments):
    process = subprocess.Popen(
        ["git", "-C", str(repo)] + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError("git inspection failed: " + stderr.strip())
    return stdout.strip()


def verify_inputs(args):
    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not (repo / "src" / "model" / "light.py").is_file():
        raise RuntimeError("official KPGT source tree is incomplete: " + str(repo))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("KPGT checkpoint is absent or empty: " + str(checkpoint))
    commit = git_output(repo, "rev-parse", "HEAD")
    if commit != args.expected_repo_commit:
        raise RuntimeError(
            "KPGT commit {} != expected {}".format(commit, args.expected_repo_commit)
        )
    if git_output(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("KPGT checkout has modified tracked files")
    checkpoint_digest = sha256(checkpoint)
    if checkpoint_digest != args.expected_checkpoint_sha256:
        raise RuntimeError("KPGT checkpoint SHA-256 mismatch")
    return repo, checkpoint, commit, checkpoint_digest


def dependency_versions(torch, dgl, Chem):
    try:
        from rdkit import rdBase

        rdkit_version = str(rdBase.rdkitVersion)
    except Exception:
        rdkit_version = str(getattr(Chem, "__version__", "unknown"))
    try:
        import scipy

        scipy_version = str(scipy.__version__)
    except Exception:
        scipy_version = "unknown"
    return {
        "python": platform.python_version(),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "dgl": str(getattr(dgl, "__version__", "unknown")),
        "rdkit": rdkit_version,
        "numpy": str(np.__version__),
        "scipy": scipy_version,
    }


class OfficialKPGT(object):
    def __init__(self, repo, checkpoint, device_name, seed):
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import torch
        import dgl
        from rdkit import Chem
        from src.data import collator as collator_module
        from src.data import featurizer
        from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
        from src.model.light import LiGhTPredictor
        from src.model_config import config_dict

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but official torch reports no CUDA")
        self.torch = torch
        self.dgl = dgl
        self.Chem = Chem
        self.smiles_to_graph = featurizer.smiles_to_graph_tune
        self.descriptor = RDKit2DNormalized()
        self.collator = collator_module.Collator_tune(5)
        self.device = torch.device(device_name)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        config = config_dict["base"]
        vocab = featurizer.Vocab(featurizer.N_ATOM_TYPES, featurizer.N_BOND_TYPES)
        model = LiGhTPredictor(
            d_node_feats=config["d_node_feats"],
            d_edge_feats=config["d_edge_feats"],
            d_g_feats=config["d_g_feats"],
            d_hpath_ratio=config["d_hpath_ratio"],
            n_mol_layers=config["n_mol_layers"],
            path_length=config["path_length"],
            n_heads=config["n_heads"],
            n_ffn_dense_layers=config["n_ffn_dense_layers"],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size,
        )
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError("KPGT checkpoint is not a state dictionary")
        cleaned = {}
        for key, value in state.items():
            key = str(key)
            cleaned[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(cleaned, strict=True)
        self.model = model.to(self.device).eval()
        self.versions = dependency_versions(torch, dgl, Chem)

    def embed(self, items):
        torch = self.torch
        samples = []
        for item in items:
            smiles = item["smiles"]
            molecule = self.Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError("RDKit rejected row " + item["row_id"])
            graph = self.smiles_to_graph(smiles, max_length=5, n_virtual_nodes=2)
            if graph is None:
                raise RuntimeError(
                    "official graph construction rejected row " + item["row_id"]
                )
            fingerprint = np.asarray(
                list(
                    self.Chem.RDKFingerprint(
                        molecule, minPath=1, maxPath=7, fpSize=512
                    )
                ),
                dtype=np.float32,
            )
            processed = self.descriptor.process(smiles)
            if processed is None:
                raise RuntimeError("descriptor generation failed for row " + item["row_id"])
            descriptor = np.asarray(processed[1:], dtype=np.float32)
            descriptor = np.where(np.isnan(descriptor), 0.0, descriptor)
            if descriptor.shape != (200,) or not np.isfinite(descriptor).all():
                raise RuntimeError("invalid descriptors for row " + item["row_id"])
            samples.append(
                (
                    smiles,
                    graph,
                    torch.from_numpy(fingerprint),
                    torch.from_numpy(descriptor),
                    torch.zeros(1, dtype=torch.float32),
                )
            )
        _, graph, fingerprint, descriptor, _ = self.collator(samples)
        with torch.no_grad():
            result = self.model.generate_fps(
                graph.to(self.device),
                fingerprint.to(self.device),
                descriptor.to(self.device),
            )
        matrix = np.asarray(result.detach().cpu().numpy(), dtype=np.float32)
        if matrix.shape != (len(items), OUTPUT_DIMENSION):
            raise RuntimeError("KPGT returned invalid shape " + str(matrix.shape))
        if not np.isfinite(matrix).all():
            raise RuntimeError("KPGT returned NaN or infinity")
        return matrix


def read_request(path):
    with open(str(path), "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != INPUT_SCHEMA:
        raise RuntimeError("worker request has a wrong schema")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise RuntimeError("worker request has no items")
    items = []
    seen = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise RuntimeError("worker request item is not an object")
        row_id = str(raw.get("row_id", "")).strip()
        smiles = str(raw.get("smiles", "")).strip()
        if not row_id or row_id in seen or not smiles:
            raise RuntimeError("worker request has invalid row IDs or SMILES")
        seen.add(row_id)
        items.append({"row_id": row_id, "smiles": smiles})
    return items


def atomic_npz(path, row_ids, matrix):
    path = Path(path).resolve()
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)
    temporary = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path.parent), prefix="." + path.name + ".", delete=False
        )
        temporary = Path(handle.name)
        with handle:
            np.savez_compressed(
                handle,
                row_ids=np.asarray(row_ids, dtype=str),
                embeddings=np.asarray(matrix, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def emit(handle, payload):
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def serve(args):
    repo, checkpoint, commit, checkpoint_digest = verify_inputs(args)
    engine = OfficialKPGT(repo, checkpoint, args.device, args.seed)
    protocol = os.fdopen(args.protocol_fd, "w", 1)
    emit(
        protocol,
        {
            "schema": PROTOCOL_SCHEMA,
            "status": "ready",
            "worker_path": str(Path(__file__).resolve()),
            "worker_sha256": sha256(Path(__file__).resolve()),
            "interpreter": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "packages": engine.versions,
            "repo": str(repo),
            "repo_commit": commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "device": str(engine.device),
            "embedding_dimension": OUTPUT_DIMENSION,
        },
    )
    for line in sys.stdin:
        command = None
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise RuntimeError("worker command is not an object")
            request_id = str(command.get("request_id", ""))
            if command.get("command") == "shutdown":
                emit(
                    protocol,
                    {
                        "schema": PROTOCOL_SCHEMA,
                        "status": "shutdown",
                        "request_id": request_id,
                    },
                )
                return 0
            if command.get("command") != "embed":
                raise RuntimeError("unsupported worker command")
            input_path = Path(str(command.get("input_json", ""))).resolve()
            output_path = Path(str(command.get("output_npz", ""))).resolve()
            items = read_request(input_path)
            matrix = engine.embed(items)
            atomic_npz(output_path, [item["row_id"] for item in items], matrix)
            emit(
                protocol,
                {
                    "schema": PROTOCOL_SCHEMA,
                    "status": "resolved",
                    "request_id": request_id,
                    "row_count": len(items),
                    "embedding_dimension": OUTPUT_DIMENSION,
                    "output_sha256": sha256(output_path),
                },
            )
        except Exception as error:
            emit(
                protocol,
                {
                    "schema": PROTOCOL_SCHEMA,
                    "status": "error",
                    "request_id": (
                        str(command.get("request_id", ""))
                        if isinstance(command, dict)
                        else ""
                    ),
                    "error": "{}: {}".format(type(error).__name__, error),
                },
            )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol-fd", type=int, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return serve(args)
    except Exception as error:
        try:
            protocol = os.fdopen(args.protocol_fd, "w", 1)
            emit(
                protocol,
                {
                    "schema": PROTOCOL_SCHEMA,
                    "status": "startup-error",
                    "error": "{}: {}".format(type(error).__name__, error),
                },
            )
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    sys.exit(main())
