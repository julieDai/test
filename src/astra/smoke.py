"""Run a fast, dependency-free smoke check on synthetic relation data."""

from __future__ import annotations

from pathlib import Path
import tempfile

from .config import ModelConfig
from .data_validation import modulo_folds, read_pairs, validate_binary_pairs


def run_smoke() -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="astra-smoke-") as temporary:
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        files = {
            "H_D.txt": "h0\td0\t1\nh1\td1\t1\nh2\td2\t1\nh3\td3\t1\n",
            "negativehdlist.txt": "h0\td2\t0\nh1\td3\t0\nh2\td0\t0\nh3\td1\t0\n",
            "H_C.txt": "h0\tc0\nh1\tc1\nh2\tc0\nh3\tc1\n",
            "C_P.txt": "c0\tp0\nc1\tp1\n",
            "P_D.txt": "p0\td0\np1\td1\np0\td2\np1\td3\n",
        }
        for name, content in files.items():
            (data / name).write_text(content, encoding="utf-8")
        config_path = root / "ASTRA.conf"
        config_path.write_text(
            "datapath=./data/H_D.txt\n"
            "ratings.setup=-columns 0 1 2\n"
            "model.name=ASTRA\n"
            "evaluation.setup=-cv 2\n"
            "output.setup=on -dir ./outputs\n",
            encoding="utf-8",
        )

        config = ModelConfig(config_path)
        positives = read_pairs(config.resolve("datapath"))
        negatives = read_pairs(data / "negativehdlist.txt")
        validate_binary_pairs(positives, negatives)
        folds = modulo_folds(positives + negatives, folds=2)
        if sorted(index for fold in folds for index in fold) != list(range(8)):
            raise AssertionError("Synthetic fold assignment lost or duplicated rows")
        for relation in ("H_C.txt", "C_P.txt", "P_D.txt"):
            if not read_pairs(data / relation):
                raise AssertionError(f"Synthetic relation is empty: {relation}")
        if config.resolve_option("output.setup", "-dir") != root / "outputs":
            raise AssertionError("Config-relative output path resolution failed")

    return {"positive_pairs": 4, "negative_pairs": 4, "folds": 2}


def main() -> int:
    summary = run_smoke()
    print(
        "ASTRA smoke test passed: "
        f"{summary['positive_pairs']} positives, "
        f"{summary['negative_pairs']} negatives, "
        f"{summary['folds']} folds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
