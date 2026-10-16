"""Declarative registry for baseline adapter entry points.

The registry intentionally stores module names instead of importing adapter
implementations. This keeps metadata inspection lightweight and lets each
framework execute in an isolated Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


BaselineRole = Literal["protocol-matched", "contextual"]


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata required to locate one comparison-method adapter."""

    identifier: str
    display_name: str
    source_key: str
    runner_module: str
    wrapper_module: str
    role: BaselineRole


def _spec(
    identifier: str,
    display_name: str,
    source_key: str,
    runner_module: str,
    wrapper_module: str,
    role: BaselineRole,
) -> BaselineSpec:
    return BaselineSpec(
        identifier=identifier,
        display_name=display_name,
        source_key=source_key,
        runner_module=runner_module,
        wrapper_module=wrapper_module,
        role=role,
    )


FORMAL_BASELINES: Mapping[str, BaselineSpec] = MappingProxyType(
    {
        "bigi": _spec(
            "bigi",
            "BiGI",
            "BiGI",
            "astra.official_baselines.bigi_runner",
            "herbhg.baselines.adapters.bigi",
            "protocol-matched",
        ),
        "hghda": _spec(
            "hghda",
            "HGHDA",
            "HGHDA",
            "astra.official_baselines.hghda_runner",
            "herbhg.baselines.adapters.hghda",
            "protocol-matched",
        ),
        "lhgce": _spec(
            "lhgce",
            "LHGCE",
            "LHGCE",
            "astra.official_baselines.lhgce_runner",
            "herbhg.baselines.adapters.lhgce",
            "protocol-matched",
        ),
        "milgnet": _spec(
            "milgnet",
            "MilGNet",
            "MilGNet",
            "astra.official_baselines.milgnet_runner",
            "herbhg.baselines.adapters.milgnet",
            "protocol-matched",
        ),
        "smgcl": _spec(
            "smgcl",
            "SMGCL",
            "SMGCL",
            "astra.official_baselines.smgcl_runner",
            "herbhg.baselines.adapters.smgcl",
            "protocol-matched",
        ),
    }
)


CONTEXTUAL_BASELINES: Mapping[str, BaselineSpec] = MappingProxyType(
    {
        "htinet-kn": _spec(
            "htinet-kn",
            "HTInet-KN",
            "HGHDA",
            "astra.official_baselines.htinet_runner",
            "herbhg.baselines.contextual.htinet_kn",
            "contextual",
        ),
        "htinet-rf": _spec(
            "htinet-rf",
            "HTInet-RF",
            "HGHDA",
            "astra.official_baselines.htinet_runner",
            "herbhg.baselines.contextual.htinet_rf",
            "contextual",
        ),
    }
)


BASELINE_MODULES = MappingProxyType(
    {
        "run": "astra.official_baselines.run",
        "audit": "astra.official_baselines.audit_results",
        "preflight": "astra.official_baselines.preflight",
    }
)


def get_baseline(identifier: str, *, include_contextual: bool = True) -> BaselineSpec:
    """Return the registered baseline named by ``identifier``."""

    key = identifier.strip().lower()
    if key in FORMAL_BASELINES:
        return FORMAL_BASELINES[key]
    if include_contextual and key in CONTEXTUAL_BASELINES:
        return CONTEXTUAL_BASELINES[key]
    choices = list(FORMAL_BASELINES)
    if include_contextual:
        choices.extend(CONTEXTUAL_BASELINES)
    raise KeyError(f"unknown baseline {identifier!r}; choose from {', '.join(choices)}")
