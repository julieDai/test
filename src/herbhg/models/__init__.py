"""Model backends.

Backends are loaded only by dataset-specific child processes.  Importing this
module therefore never initializes TensorFlow or a GPU runtime.
"""

from __future__ import annotations

__all__: list[str] = []
