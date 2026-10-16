"""Small, dependency-free parser for ASTRA's legacy key/value configs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import shlex

from .paths import PathLike, resolve_project_path


class ModelConfig(Mapping[str, str]):
    """Read a ``key=value`` model config while retaining its source path."""

    def __init__(self, path: PathLike):
        self.path = resolve_project_path(path, must_exist=True)
        self._values: dict[str, str] = {}
        for line_number, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"Malformed configuration line {line_number} in {self.path}"
                )
            key, value = line.split("=", 1)
            self._values[key.strip()] = value.strip()

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def directory(self) -> Path:
        return self.path.parent

    def resolve(self, key: str, *, must_exist: bool = False) -> Path:
        """Resolve a plain path-valued key relative to the config file."""

        return resolve_project_path(
            self[key], base=self.directory, must_exist=must_exist
        )

    def option(self, key: str, flag: str, default: str | None = None) -> str | None:
        """Return one value from a legacy option string such as ``-dir out``."""

        tokens = shlex.split(self[key])
        try:
            index = tokens.index(flag)
        except ValueError:
            return default
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
            return ""
        return tokens[index + 1]

    def resolve_option(
        self,
        key: str,
        flag: str,
        *,
        default: PathLike | None = None,
    ) -> Path | None:
        value = self.option(key, flag)
        if value is None:
            return resolve_project_path(default) if default is not None else None
        return resolve_project_path(value, base=self.directory)


def load_model_config(path: PathLike) -> ModelConfig:
    return ModelConfig(path)
