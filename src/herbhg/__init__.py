"""HerbHG: structure--semantic hypergraph learning for HDA prediction.

The package root intentionally avoids importing numerical backends. This
keeps metadata inspection, command discovery, and smoke checks independent of
TensorFlow and the dataset-local backend namespaces.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
