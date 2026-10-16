"""Public TCM-suite model name backed by the established implementation."""

from HGHDA import HGHDA as _TCMHerbHG


class HerbHG(_TCMHerbHG):
    """HerbHG model for the TCM-suite benchmark."""


__all__ = ["HerbHG"]
