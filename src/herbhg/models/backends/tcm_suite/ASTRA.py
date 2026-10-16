"""Public ASTRA model name backed by the validated legacy implementation."""

from HGHDA import HGHDA


class ASTRA(HGHDA):
    """Compatibility wrapper for the implementation historically named HGHDA.

    Keeping ``HGHDA`` unchanged allows archived reproduction scripts to import
    their original class, while new commands and outputs use the paper name
    ASTRA.
    """

    pass
