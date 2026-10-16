"""Fast, dependency-light validation of the public package contract."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
import tempfile


def _pairs(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] not in {"0", "1"}:
            raise ValueError(f"Invalid relation row at {path}:{line_number}")
        rows.append((fields[0], fields[1], int(fields[2])))
    return rows


def run_smoke() -> dict[str, object]:
    """Exercise packaged backends and deterministic fold coverage on synthetic data."""

    backends = files("herbhg.models.backends")
    entries = {
        "tcm-suite": ("tcm_suite", "main.py"),
        "ethnobotany": ("ethnobotany", "run_training.py"),
    }
    for dataset, (directory, filename) in entries.items():
        backend = backends.joinpath(directory)
        entry = backend.joinpath(filename)
        if not backend.is_dir():
            raise FileNotFoundError(str(backend))
        if not entry.is_file():
            raise FileNotFoundError(str(entry))

    with tempfile.TemporaryDirectory(prefix="herbhg-smoke-") as temporary:
        relation = Path(temporary) / "H_D.txt"
        relation.write_text(
            "h0\td0\t1\n"
            "h1\td1\t1\n"
            "h2\td2\t1\n"
            "h3\td3\t1\n"
            "h0\td2\t0\n"
            "h1\td3\t0\n"
            "h2\td0\t0\n"
            "h3\td1\t0\n",
            encoding="utf-8",
        )
        rows = _pairs(relation)
        positives = sum(label == 1 for _, _, label in rows)
        negatives = sum(label == 0 for _, _, label in rows)
        if positives != negatives:
            raise AssertionError("Synthetic class balance changed")
        folds = [rows[0::2], rows[1::2]]
        if sum(map(len, folds)) != len(rows):
            raise AssertionError("Synthetic fold coverage changed")

    return {
        "status": "ok",
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "folds": len(folds),
        "backends": ["tcm-suite", "ethnobotany"],
    }


def main() -> int:
    print(json.dumps(run_smoke(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
