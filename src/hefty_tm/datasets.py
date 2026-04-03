from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TABLE_PATTERNS = ("*.dat", "*.txt", "*.tsv")
SKIP_BASENAMES = {"README.txt", "00README.json"}


@dataclass(frozen=True)
class Table:
    path: Path
    header: tuple[str, ...]
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)


def _nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _looks_like_header(line: str) -> bool:
    return bool(re.search(r"[A-Za-z]", line))


def _split_header(line: str) -> tuple[str, ...]:
    if "\t" in line:
        parts = [part.strip() for part in line.split("\t") if part.strip()]
    else:
        parts = [part.strip() for part in re.split(r"\s+", line) if part.strip()]
    return tuple(parts)


def load_table(path: str | Path) -> Table:
    table_path = Path(path)
    lines = _nonempty_lines(table_path)
    if not lines:
        raise ValueError(f"Empty table: {table_path}")

    header: tuple[str, ...]
    data_lines = lines
    if _looks_like_header(lines[0]):
        header = _split_header(lines[0])
        data_lines = lines[1:]
    else:
        ncols = len(_split_header(lines[0]))
        header = tuple(f"col{i}" for i in range(ncols))

    data = np.loadtxt(io.StringIO("\n".join(data_lines)))
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(-1, len(header))
    return Table(path=table_path, header=header, data=data)


def discover_tables(root: str | Path) -> list[Path]:
    root_path = Path(root)
    tables: list[Path] = []
    for pattern in TABLE_PATTERNS:
        for path in sorted(root_path.rglob(pattern)):
            if path.name in SKIP_BASENAMES:
                continue
            tables.append(path)
    return tables


def summarize_tables(root: str | Path) -> list[dict[str, object]]:
    summaries = []
    for path in discover_tables(root):
        try:
            table = load_table(path)
        except Exception as exc:
            summaries.append(
                {
                    "path": str(path),
                    "status": f"error: {exc}",
                }
            )
            continue

        summaries.append(
            {
                "path": str(path),
                "status": "ok",
                "rows": int(table.data.shape[0]),
                "cols": int(table.data.shape[1]),
                "header": list(table.header),
            }
        )
    return summaries
