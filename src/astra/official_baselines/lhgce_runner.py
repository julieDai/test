#!/usr/bin/env python3
"""Run pinned official LHGCE with exact, sparse identity-feature algebra."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

try:
    from .common import (
        DEFAULT_THIRD_PARTY_DIR,
        balanced_prefix_indices,
        compact_compound_protein_pairs,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        legacy_protocol_provenance,
        load_protocol,
        outer_training_split,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )
    from .lhgce_identity import (
        ExactIdentityLHGCE,
        chunked_bce_with_logits_backward,
        chunked_sigmoid_scores,
    )
except ImportError:  # Direct script execution.
    from common import (
        DEFAULT_THIRD_PARTY_DIR,
        balanced_prefix_indices,
        compact_compound_protein_pairs,
        fold_split,
        formal_runtime_module_origins,
        formal_runtime_versions,
        legacy_protocol_provenance,
        load_protocol,
        outer_training_split,
        sha256,
        validate_frozen_input_manifest,
        verify_checkout_commit,
        write_json,
    )
    from lhgce_identity import (
        ExactIdentityLHGCE,
        chunked_bce_with_logits_backward,
        chunked_sigmoid_scores,
    )

try:
    from ..benchmark.metrics import (
        evaluate_test_scores,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )
except ImportError:  # Direct script execution with ``src`` on PYTHONPATH.
    from astra.benchmark.metrics import (
        evaluate_test_scores,
        evaluate_test_oracle,
        evaluate_validation_and_test,
        save_predictions,
    )


UPSTREAM_COMMIT = "eeedb4cd6cf64166bd0f8a0ebc2cca96c60874dc"
INNER_VALIDATION_REQUESTED_EPOCH_CANDIDATES = (20, 50, 100, 200, 300, 500, 1000)


def _requested_epochs_to_optimizer_steps(requested_epochs: int) -> int:
    """Map the released requested-epoch convention to executed updates.

    The upstream loop is ``range(1, EPOCHS)``.  A requested budget of ``E``
    therefore denotes the state after exactly ``E - 1`` optimizer steps.
    """

    requested_epochs = int(requested_epochs)
    if requested_epochs < 2:
        raise ValueError("LHGCE requested epochs must be at least 2")
    return requested_epochs - 1


def _validation_ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the locked continuous validation metrics for budget selection."""

    metrics = evaluate_test_scores(labels, scores, threshold=0.5)
    return {
        "Validation_AUROC": float(metrics["AUROC"]),
        "Validation_AUPRC": float(metrics["AUPRC"]),
        "Validation_AveragePrecision": float(metrics["AveragePrecision"]),
    }


def _select_requested_budget(
    validation_records: list[dict[str, object]],
) -> int:
    """Select maximum validation AUPRC; exact ties use the lower budget."""

    if not validation_records:
        raise ValueError("at least one LHGCE validation checkpoint is required")
    ordered = sorted(
        validation_records,
        key=lambda record: int(record["Requested_Epochs"]),
    )
    return int(
        max(
            ordered,
            key=lambda record: float(record["Validation_AUPRC"]),
        )["Requested_Epochs"]
    )


def _score_sha256(scores: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(scores, dtype=np.float32).view(np.uint8)
    ).hexdigest()


def _cpu_clone_tree(value):
    """Detach all tensors into immutable host copies for checkpoint writing."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(item) for item in value)
    return value


def _save_training_checkpoint(
    torch,
    path: Path,
    *,
    model,
    optimizer,
    requested_epochs: int,
    optimizer_steps: int,
) -> None:
    """Atomically persist model and optimizer state for one locked candidate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "Requested_Epochs": int(requested_epochs),
            "Optimizer_Steps": int(optimizer_steps),
            "Model_State_Dict": _cpu_clone_tree(model.state_dict()),
            "Optimizer_State_Dict": _cpu_clone_tree(optimizer.state_dict()),
        },
        temporary,
    )
    temporary.replace(path)


def _load_training_checkpoint(torch, path: Path, *, device):
    """Load a locally generated tensor-only checkpoint on the requested device."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 has no ``weights_only`` keyword.
        return torch.load(path, map_location=device)


def _load_official_model(source_dir: Path):
    previous_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_metrics = sys.modules.pop("metrics", None)
    with tempfile.TemporaryDirectory(prefix="herbhg_lhgce_pycache_") as pycache:
        try:
            sys.pycache_prefix = pycache
            sys.dont_write_bytecode = True
            metrics_spec = importlib.util.spec_from_file_location(
                "metrics", source_dir / "metrics.py"
            )
            metrics_module = importlib.util.module_from_spec(metrics_spec)
            assert metrics_spec.loader is not None
            metrics_spec.loader.exec_module(metrics_module)
            sys.modules["metrics"] = metrics_module
            model_spec = importlib.util.spec_from_file_location(
                "astar_lhgce_official_model", source_dir / "model.py"
            )
            model_module = importlib.util.module_from_spec(model_spec)
            assert model_spec.loader is not None
            model_spec.loader.exec_module(model_module)
        finally:
            if previous_metrics is not None:
                sys.modules["metrics"] = previous_metrics
            else:
                sys.modules.pop("metrics", None)
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    return model_module.Model


def _edge_index(torch, pairs: np.ndarray):
    if not len(pairs):
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(np.asarray(pairs, dtype=np.int64).T.copy()).long()


def _build_identity_graph(
    protocol,
    fold_rows,
    include_cp,
    *,
    training_positive_pairs: np.ndarray | None = None,
):
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.data import HeteroData

    data = HeteroData()
    # Preserve the node-type insertion order of the previous dense adapter and
    # declare only counts.  ExactIdentityLHGCE supplies the identity-feature
    # computation without ever allocating the dense matrices.
    data["drug"].num_nodes = protocol.n_herbs
    data["disease"].num_nodes = protocol.n_diseases
    data["compound"].num_nodes = protocol.n_compounds
    data["protein"].num_nodes = protocol.n_proteins

    hc = np.unique(protocol.herb_compound_pairs, axis=0)
    pd_pairs = np.unique(protocol.protein_disease_pairs, axis=0)
    data["drug", "has", "compound"].edge_index = _edge_index(torch, hc)
    data["protein", "associated_with", "disease"].edge_index = _edge_index(
        torch, pd_pairs
    )
    data["drug", "treats", "disease"].edge_index = _edge_index(
        torch,
        (
            fold_rows.train_positive_pairs
            if training_positive_pairs is None
            else np.unique(training_positive_pairs, axis=0)
        ),
    )
    mapped_cp = compact_compound_protein_pairs(protocol)
    if include_cp and len(mapped_cp):
        data["compound", "targets", "protein"].edge_index = _edge_index(
            torch, mapped_cp
        )

    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    # Upstream applies NormalizeFeatures to dense identities.  For every
    # actual benchmark node type (size > 1), this leaves I_N unchanged.
    return data, mapped_cp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        required=True,
        help="Central fair-protocol manifest file or directory.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_THIRD_PARTY_DIR / "LHGCE_official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--neurons", type=int, default=168)
    parser.add_argument("--agg-linear", default="mul", choices=("concatenate", "mean", "sum", "mul"))
    parser.add_argument("--agg-conv", default="var", choices=("sum", "var", "mean", "std"))
    parser.add_argument("--agg-hetero", default="sum")
    parser.add_argument("--include-mapped-cp", action="store_true")
    parser.add_argument(
        "--train-chunk-size",
        type=int,
        default=262_144,
        help="Decoder rows per exact streamed BCE chunk.",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=262_144,
        help="Decoder rows per streamed inference chunk.",
    )
    parser.add_argument(
        "--legacy-test-oracle",
        action="store_true",
        help=(
            "Restore all four outer-training folds and select descriptive "
            "threshold metrics on the held-out test fold."
        ),
    )
    parser.add_argument(
        "--inner-validation-budget-selection",
        action="store_true",
        help=(
            "Train one requested-1000-epoch trajectory, checkpoint the states "
            "corresponding to requested budgets 20/50/100/200/300/500/1000, "
            "select by inner-validation trapezoidal AUPRC, freeze that choice, "
            "then score the outer test once with only the selected checkpoint."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Run one update on a deterministic prefix")
    parser.add_argument("--run-generation-id")
    args = parser.parse_args()

    exact_configuration = (
        args.layers == 1
        and args.agg_linear == "mul"
        and args.agg_conv == "var"
        and args.agg_hetero == "sum"
    )
    if not exact_configuration:
        parser.error(
            "the formal memory-safe path is proven equivalent only for "
            "--layers 1 --agg-conv var --agg-hetero sum --agg-linear mul"
        )
    if not args.smoke and args.epochs < 2:
        parser.error("formal LHGCE runs require --epochs >= 2 (the author loop executes EPOCHS-1 updates)")
    if args.train_chunk_size < 1 or args.eval_chunk_size < 1:
        parser.error("chunk sizes must be positive")
    if args.inner_validation_budget_selection:
        if args.smoke or args.legacy_test_oracle:
            parser.error(
                "--inner-validation-budget-selection is formal validation only "
                "and cannot be combined with smoke or legacy-test-oracle"
            )
        if args.epochs != 1000:
            parser.error(
                "--inner-validation-budget-selection requires --epochs 1000"
            )

    started = time.time()
    args.protocol_manifest = args.protocol_manifest.resolve()
    args.output_root = args.output_root.resolve()
    args.source_dir = args.source_dir.resolve()
    if not (args.source_dir / "model.py").exists():
        raise FileNotFoundError(
            f"LHGCE checkout missing at {args.source_dir}; run third_party/fetch_baselines.sh"
        )
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)

    # Required by torch deterministic mode for CUDA matrix multiplications.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch
    import torch_geometric

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch_geometric.seed_everything(args.seed)
    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)

    protocol = load_protocol(args.protocol_manifest)
    if args.inner_validation_budget_selection and protocol.dataset != "tcm-suite":
        parser.error(
            "--inner-validation-budget-selection is locked to the TCM-suite protocol"
        )
    if args.legacy_test_oracle and (
        protocol.dataset != "tcm-suite" or protocol.pair_order != "source"
    ):
        parser.error(
            "--legacy-test-oracle is defined only for the source-order "
            "TCM-suite manifest"
        )
    initial_input_manifest = protocol.input_manifest()
    frozen_input_lock_sha256 = validate_frozen_input_manifest(initial_input_manifest)
    fold_rows = protocol.fold_rows(args.fold)
    legacy_provenance = (
        legacy_protocol_provenance(
            protocol,
            fold_rows,
            args.protocol_manifest,
            require_clean_project=not args.smoke,
        )
        if args.legacy_test_oracle
        else None
    )
    if args.legacy_test_oracle:
        train_pairs_full, train_labels_full, train_row_id_full = outer_training_split(
            fold_rows
        )
    else:
        train_pairs_full, train_labels_full, train_row_id_full = fold_split(
            fold_rows, "train"
        )
    data, mapped_cp = _build_identity_graph(
        protocol,
        fold_rows,
        args.include_mapped_cp,
        training_positive_pairs=train_pairs_full[train_labels_full == 1],
    )
    OfficialModel = _load_official_model(args.source_dir)
    official_model = OfficialModel(
        data=data,
        neurons=args.neurons,
        layers=args.layers,
        aggregator_lin=args.agg_linear,
        aggregator_conv=args.agg_conv,
        aggregator_hetero=args.agg_hetero,
    )
    node_counts = {
        "drug": protocol.n_herbs,
        "disease": protocol.n_diseases,
        "compound": protocol.n_compounds,
        "protein": protocol.n_proteins,
    }
    model = ExactIdentityLHGCE(
        official_model,
        data.edge_index_dict,
        node_counts,
        data.edge_types,
    ).to(device)
    # Match the lazy-parameter initialization order of the official dense
    # forward without creating any identity input tensor.
    model.initialize_parameters()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    validation_pairs_full, validation_labels_full, validation_row_id_full = fold_split(
        fold_rows, "validation"
    )
    test_pairs_full, test_labels_full, test_row_id_full = fold_split(fold_rows, "test")
    train_selection = np.arange(len(train_pairs_full), dtype=np.int64)
    validation_selection = np.arange(len(validation_pairs_full), dtype=np.int64)
    test_selection = np.arange(len(test_pairs_full), dtype=np.int64)
    if args.smoke:
        # Keep all model smoke artifacts on the same 64-positive/64-negative
        # rows so the independent cross-method identity audit is meaningful.
        train_selection = balanced_prefix_indices(train_labels_full, 64)
        validation_selection = balanced_prefix_indices(validation_labels_full, 64)
        test_selection = balanced_prefix_indices(test_labels_full, 64)
        epochs = 1
    else:
        # The released main.py iterates ``range(1, EPOCHS)``.  Preserve the
        # resulting EPOCHS-1 optimizer updates while retaining the paper's
        # requested-epoch value in the provenance record.
        epochs = max(1, args.epochs - 1)
    train_pairs = train_pairs_full[train_selection]
    train_labels = train_labels_full[train_selection]
    validation_pairs = validation_pairs_full[validation_selection]
    validation_labels = validation_labels_full[validation_selection]
    validation_row_id = validation_row_id_full[validation_selection]
    test_pairs = test_pairs_full[test_selection]
    test_labels = test_labels_full[test_selection]
    test_row_id = test_row_id_full[test_selection]

    train_index = _edge_index(torch, train_pairs).to(device)
    train_targets = torch.from_numpy(train_labels).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positive = float(train_labels.sum())
    negative = float(len(train_labels) - positive)
    pos_weight = torch.tensor(
        negative / positive, dtype=torch.float32, device=device
    )

    result_dir = args.output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    validation_index = _edge_index(torch, validation_pairs).to(device)
    history = []
    optimizer_step_seconds = []
    checkpoint_validation_scores: dict[int, np.ndarray] = {}
    checkpoint_validation_records: list[dict[str, object]] = []
    checkpoint_validation_runtime_seconds: list[float] = []
    for epoch in range(epochs):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = chunked_bce_with_logits_backward(
            model,
            train_index,
            train_targets,
            pos_weight=pos_weight,
            chunk_size=args.train_chunk_size,
        )
        optimizer.step()
        history.append(loss_value)
        optimizer_step_seconds.append(time.perf_counter() - step_started)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1}/{epochs} loss={history[-1]:.8f} "
                f"seconds={optimizer_step_seconds[-1]:.3f}",
                flush=True,
            )
        completed_updates = epoch + 1
        requested_epochs = completed_updates + 1
        if (
            args.inner_validation_budget_selection
            and requested_epochs in INNER_VALIDATION_REQUESTED_EPOCH_CANDIDATES
        ):
            checkpoint_path = (
                result_dir
                / "training_checkpoints"
                / f"requested_epochs_{requested_epochs}.pt"
            )
            _save_training_checkpoint(
                torch,
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                requested_epochs=requested_epochs,
                optimizer_steps=completed_updates,
            )
            model.eval()
            checkpoint_started = time.perf_counter()
            checkpoint_scores = chunked_sigmoid_scores(
                model,
                validation_index,
                chunk_size=args.eval_chunk_size,
            ).numpy()
            checkpoint_runtime = time.perf_counter() - checkpoint_started
            checkpoint_validation_runtime_seconds.append(checkpoint_runtime)
            checkpoint_validation_scores[requested_epochs] = checkpoint_scores
            prediction_path = (
                result_dir
                / "validation_checkpoints"
                / f"requested_epochs_{requested_epochs}.npz"
            )
            save_predictions(
                prediction_path,
                row_id=validation_row_id,
                pair=validation_pairs,
                label=validation_labels,
                score=checkpoint_scores,
            )
            checkpoint_validation_records.append(
                {
                    "Requested_Epochs": requested_epochs,
                    "Optimizer_Steps": completed_updates,
                    **_validation_ranking(validation_labels, checkpoint_scores),
                    "Validation_Score_SHA256": _score_sha256(checkpoint_scores),
                    "Validation_Runtime_Seconds": checkpoint_runtime,
                    "Validation_Predictions": str(
                        prediction_path.relative_to(args.output_root)
                    ),
                    "Validation_Predictions_SHA256": sha256(prediction_path),
                    "Training_Checkpoint": str(
                        checkpoint_path.relative_to(args.output_root)
                    ),
                    "Training_Checkpoint_SHA256": sha256(checkpoint_path),
                }
            )

    epoch_selection: dict[str, object]
    if args.inner_validation_budget_selection:
        if tuple(checkpoint_validation_scores) != (
            INNER_VALIDATION_REQUESTED_EPOCH_CANDIDATES
        ):
            raise RuntimeError(
                "LHGCE inner-validation selection did not evaluate every locked "
                "requested-epoch checkpoint"
            )
        selected_requested_epochs = _select_requested_budget(
            checkpoint_validation_records
        )
        selected_record = next(
            record
            for record in checkpoint_validation_records
            if record["Requested_Epochs"] == selected_requested_epochs
        )
        epoch_selection = {
            "Mode": "inner-validation",
            "Candidates": list(INNER_VALIDATION_REQUESTED_EPOCH_CANDIDATES),
            "Candidate_Semantics": (
                "requested EPOCHS values; upstream range(1, EPOCHS) executes EPOCHS-1 updates"
            ),
            "Selection_Metric": "validation trapezoidal AUPRC",
            "Selection_Tie_Break": (
                "lowest requested epoch budget on an exact AUPRC tie"
            ),
            "Selected_Requested_Epochs": selected_requested_epochs,
            "Selected_Optimizer_Steps": _requested_epochs_to_optimizer_steps(
                selected_requested_epochs
            ),
            "Selected_Validation_AUPRC": selected_record["Validation_AUPRC"],
            "Selected_Training_Checkpoint": selected_record[
                "Training_Checkpoint"
            ],
            "Selected_Training_Checkpoint_SHA256": selected_record[
                "Training_Checkpoint_SHA256"
            ],
            "Checkpoint_Validation_Metrics": checkpoint_validation_records,
            "Outer_Test_Access": (
                "selection decision persisted before outer-test index construction; "
                "only the selected checkpoint is evaluated on the outer test"
            ),
            "Decision_Frozen_Before_Test": True,
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }
        # The selection decision is the commit point: persist it before the
        # outer-test index is even constructed.  A later failure leaves no
        # complete metrics or top-level test prediction artifact.
        write_json(result_dir / "epoch_selection.json", epoch_selection)
        selected_checkpoint_path = args.output_root / str(
            epoch_selection["Selected_Training_Checkpoint"]
        )
        selected_checkpoint = _load_training_checkpoint(
            torch, selected_checkpoint_path, device=torch.device("cpu")
        )
        if (
            int(selected_checkpoint["Requested_Epochs"])
            != selected_requested_epochs
            or int(selected_checkpoint["Optimizer_Steps"])
            != _requested_epochs_to_optimizer_steps(selected_requested_epochs)
        ):
            raise RuntimeError("selected LHGCE training checkpoint metadata differs")
        model.load_state_dict(selected_checkpoint["Model_State_Dict"])
        optimizer.load_state_dict(selected_checkpoint["Optimizer_State_Dict"])
        del selected_checkpoint
        validation_scores = checkpoint_validation_scores[
            selected_requested_epochs
        ]
        validation_inference_seconds = float(
            sum(checkpoint_validation_runtime_seconds)
        )
    else:
        model.eval()
        validation_started = time.perf_counter()
        validation_scores = chunked_sigmoid_scores(
            model, validation_index, chunk_size=args.eval_chunk_size
        ).numpy()
        validation_inference_seconds = time.perf_counter() - validation_started
        selected_requested_epochs = args.epochs
        epoch_selection = {
            "Mode": "fixed-budget",
            "Candidates": [args.epochs],
            "Selected_Requested_Epochs": args.epochs,
            "Selected_Optimizer_Steps": epochs,
            "Test_Checkpoints_Evaluated": 1,
            "Unselected_Test_Checkpoints_Evaluated": 0,
        }

    # Do not construct or score the outer-test index until the budget decision
    # above has been written and the selected state has been restored.
    model.eval()
    test_index = _edge_index(torch, test_pairs).to(device)
    test_started = time.perf_counter()
    test_scores = chunked_sigmoid_scores(
        model, test_index, chunk_size=args.eval_chunk_size
    ).numpy()
    test_inference_seconds = time.perf_counter() - test_started
    inference_seconds = validation_inference_seconds + test_inference_seconds
    metrics_started = time.perf_counter()
    if args.legacy_test_oracle:
        metrics = evaluate_test_oracle(test_labels, test_scores)
    else:
        metrics = evaluate_validation_and_test(
            validation_labels,
            validation_scores,
            test_labels,
            test_scores,
        )
    metric_seconds = time.perf_counter() - metrics_started
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    metrics.update(
        {
            "Model": "LHGCE-official-identity-exact",
            "Fold": args.fold,
            "Dataset": str(protocol.dataset),
            "Seed": args.seed,
            "Device": str(device),
            "Runtime_Seconds": time.time() - started,
            "Epochs_Requested": args.epochs,
            "Epochs_Executed": epochs,
            "Upstream_Commit": UPSTREAM_COMMIT,
            "Upstream_Source": "https://github.com/shahinghasemi/LHGCE",
            "Adapter": (
                "official model.py parameters and mul decoder with exact sparse "
                "one-layer identity-feature algebra and exact chunked BCE"
            ),
            "Protocol_Kind": "official one-layer identity architecture, algebraically exact",
            "Auxiliary_Features": "identity for herbs, diseases, compounds, and proteins",
            "Mapped_CP_Rows": int(len(mapped_cp)),
            "CP_Included": bool(args.include_mapped_cp),
            "Smoke": bool(args.smoke),
            "Formal_Eligible": bool(not args.smoke),
            "Optimizer_Steps": epochs,
            "Epoch_Selection_Mode": epoch_selection["Mode"],
            "Epoch_Candidates": epoch_selection["Candidates"],
            "Selected_Requested_Epochs": selected_requested_epochs,
            "Selected_Checkpoint_Optimizer_Steps": (
                _requested_epochs_to_optimizer_steps(selected_requested_epochs)
                if not args.smoke
                else epochs
            ),
            "Checkpoint_Validation_Metrics": checkpoint_validation_records,
            "Checkpoint_Trajectory_Equivalent_To_Independent_Fixed_Budget": bool(
                args.inner_validation_budget_selection
            ),
            "Checkpoint_Trajectory_Equivalence_Basis": (
                "same seeded initialization and uninterrupted deterministic Adam prefix; "
                "validation inference and tensor-only checkpoint serialization do not "
                "update model, optimizer, scheduler, or RNG state"
                if args.inner_validation_budget_selection
                else None
            ),
            "Epoch_Selection_Metric": epoch_selection.get("Selection_Metric"),
            "Epoch_Selection_Tie_Break": epoch_selection.get(
                "Selection_Tie_Break"
            ),
            "Decision_Frozen_Before_Test": epoch_selection.get(
                "Decision_Frozen_Before_Test", False
            ),
            "Test_Checkpoints_Evaluated": epoch_selection[
                "Test_Checkpoints_Evaluated"
            ],
            "Unselected_Test_Checkpoints_Evaluated": epoch_selection[
                "Unselected_Test_Checkpoints_Evaluated"
            ],
            "Evaluation_Mode": (
                "legacy-test-oracle" if args.legacy_test_oracle else "validation"
            ),
            "Training_Rows": int(len(train_pairs)),
            "Training_Chunk_Size": int(args.train_chunk_size),
            "Evaluation_Chunk_Size": int(args.eval_chunk_size),
            "Chunked_Loss_Exact": True,
            "Optimizer_Step_Seconds_Mean": float(
                np.mean(optimizer_step_seconds)
            ),
            "Inference_Seconds": inference_seconds,
            "Validation_Inference_Seconds": validation_inference_seconds,
            "Test_Inference_Seconds": test_inference_seconds,
            "Metric_Seconds": metric_seconds,
            "Peak_CUDA_Memory_Bytes": peak_cuda_memory_bytes,
            "Run_Parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    if legacy_provenance is not None:
        metrics["Legacy_Protocol_Provenance"] = legacy_provenance
    verify_checkout_commit(args.source_dir, UPSTREAM_COMMIT)
    final_input_manifest = protocol.input_manifest()
    if final_input_manifest != initial_input_manifest:
        raise RuntimeError("central benchmark inputs changed during the LHGCE run")
    validate_frozen_input_manifest(final_input_manifest)
    save_predictions(
        result_dir / "validation_predictions.npz",
        row_id=validation_row_id,
        pair=validation_pairs,
        label=validation_labels,
        score=validation_scores,
    )
    save_predictions(
        result_dir / "test_predictions.npz",
        row_id=test_row_id,
        pair=test_pairs,
        label=test_labels,
        score=test_scores,
    )
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        args.output_root / "protocol.json",
        {
            **initial_input_manifest,
            "frozen_input_lock_sha256": frozen_input_lock_sha256,
            "protocol_manifest": str(args.protocol_manifest),
            "dataset": str(protocol.dataset),
            "run_generation_id": args.run_generation_id,
            "legacy_protocol_provenance": legacy_provenance,
            "model": metrics,
            "adapter_sources": {
                "common.py": sha256(Path(__file__).with_name("common.py")),
                "lhgce_identity.py": sha256(Path(__file__).with_name("lhgce_identity.py")),
                "lhgce_runner.py": sha256(Path(__file__)),
            },
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "training_loss": history,
            "optimizer_step_seconds": optimizer_step_seconds,
            "optimizer_steps": epochs,
            "epoch_selection": epoch_selection,
            "evaluation_mode": (
                "legacy-test-oracle" if args.legacy_test_oracle else "validation"
            ),
            "training_split": (
                "complete source-order outer training fold"
                if args.legacy_test_oracle
                else "inner training fold"
            ),
            "prediction_artifacts": {
                "validation": "results/validation_predictions.npz",
                "test": "results/test_predictions.npz",
                "validation_checkpoints": (
                    "results/validation_checkpoints/requested_epochs_{E}.npz"
                    if args.inner_validation_budget_selection
                    else None
                ),
                "training_checkpoints": (
                    "results/training_checkpoints/requested_epochs_{E}.pt"
                    if args.inner_validation_budget_selection
                    else None
                ),
                "fields": ["row_id", "pair", "label", "score"],
            },
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": ":16:8",
            },
            "identity_native_exact": {
                "upstream_commit": UPSTREAM_COMMIT,
                "configuration": {
                    "layers": 1,
                    "sage_aggregation": "VarAggregation",
                    "relation_aggregation": "sum",
                    "decoder": "official Linears aggregator=mul",
                    "sage_normalize": True,
                    "sage_root_weight": True,
                    "sage_project": False,
                },
                "node_feature_semantics": {
                    node_type: f"identity({count})"
                    for node_type, count in node_counts.items()
                },
                "dense_identity_tensors_allocated": False,
                "variance_algebra": "p_j - p_j^2, where p_j=count(j)/target_degree",
                "root_algebra": "lin_r(I) = lin_r.weight.T",
                "pyg_normalize_features": (
                    "omitted without approximation: NormalizeFeatures leaves each "
                    "one-hot identity matrix unchanged"
                ),
                "equivalence_test": "tests/test_lhgce_identity_exact.py",
            },
            "exact_chunking": {
                "train_chunk_size": args.train_chunk_size,
                "eval_chunk_size": args.eval_chunk_size,
                "loss_identity": (
                    "sum of per-chunk BCEWithLogits sums divided by the full "
                    "row count equals full-batch BCEWithLogits mean"
                ),
                "encoder_evaluations_per_optimizer_step": 1,
                "optimizer_updates_match_upstream": epochs,
                "equivalence_test": "tests/test_lhgce_identity_exact.py",
            },
            "runtime_versions": formal_runtime_versions(),
            "runtime_module_origins": formal_runtime_module_origins(
                require_environment=not args.smoke
            ),
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
