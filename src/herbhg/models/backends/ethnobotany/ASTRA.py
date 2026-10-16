"""Public ASTRA model name with backward-compatible HGHDA internals."""

from HGHDA import HGHDA


class ASTRA(HGHDA):
    """Ethnobotany ASTRA implementation.

    The historical implementation class remains available as ``HGHDA`` so
    old experiment scripts continue to run. New entry points and outputs use
    the paper-facing ``ASTRA`` name.
    """


__all__ = ["ASTRA"]
