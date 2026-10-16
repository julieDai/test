"""Stable metadata for protocol-matched comparison methods."""

from .registry import (
    BASELINE_MODULES,
    CONTEXTUAL_BASELINES,
    FORMAL_BASELINES,
    BaselineSpec,
    get_baseline,
)

__all__ = [
    "BASELINE_MODULES",
    "CONTEXTUAL_BASELINES",
    "FORMAL_BASELINES",
    "BaselineSpec",
    "get_baseline",
]
