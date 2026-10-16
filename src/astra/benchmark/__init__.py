"""Shared evaluation and audit utilities for controlled benchmarks."""

from .metrics import (
    BenchmarkAuditError,
    PredictionTable,
    ThresholdSelection,
    aggregate_fold_metrics,
    audit_cross_method_predictions,
    audit_milgnet_run,
    evaluate_test_scores,
    evaluate_test_oracle,
    evaluate_validation_and_test,
    load_predictions,
    save_predictions,
    select_f1_threshold,
    validate_prediction_coverage,
)
from .protocol import (
    BenchmarkProtocol,
    FoldRows,
    ProtocolError,
    build_protocol,
    load_protocol,
    load_protocol_manifest,
)

__all__ = [
    "BenchmarkAuditError",
    "BenchmarkProtocol",
    "FoldRows",
    "PredictionTable",
    "ProtocolError",
    "ThresholdSelection",
    "aggregate_fold_metrics",
    "audit_cross_method_predictions",
    "audit_milgnet_run",
    "build_protocol",
    "evaluate_test_scores",
    "evaluate_test_oracle",
    "evaluate_validation_and_test",
    "load_predictions",
    "load_protocol",
    "load_protocol_manifest",
    "save_predictions",
    "select_f1_threshold",
    "validate_prediction_coverage",
]
