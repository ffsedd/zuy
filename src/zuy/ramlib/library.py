"""Spectral library container.

Layout on disk (no pickle):

    <name>.npz        axis (M,), Y (N, M) float32 with NaN outside each entry's
                      own coverage, ids (N,), config (JSON string)
    <name>.meta.csv   one row per entry, key column ``id``; hand-editable
                      (e.g. fix the ``group`` column to merge duplicates)

All spectra live on one common axis, so scoring one query against the whole
library is a matrix operation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_VERSION = 1


def meta_path_for(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".meta.csv")


@dataclass
class Library:
    axis: np.ndarray  # (M,) wavenumbers in cm-1, strictly increasing
    Y: np.ndarray  # (N, M) float32, NaN where an entry has no data
    meta: pd.DataFrame  # index = id (unique), N rows, same order as Y
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        self.axis = np.asarray(self.axis)
        self.Y = np.asarray(self.Y, dtype=np.float32)
        if self.Y.ndim != 2 or self.Y.shape != (len(self.meta), len(self.axis)):
            raise ValueError(
                f"shape mismatch: Y {self.Y.shape}, meta {len(self.meta)}, axis {len(self.axis)}"
            )
        if not np.all(np.diff(self.axis) > 0):
            raise ValueError("axis must be strictly increasing")
        if not self.meta.index.is_unique:
            raise ValueError("ids must be unique")

    # ---- access -----------------------------------------------------------
    def __len__(self) -> int:
        return self.Y.shape[0]

    def __repr__(self) -> str:
        lo, hi = self.axis[0], self.axis[-1]
        return f"Library({len(self)} spectra, axis {lo:g}-{hi:g} cm-1, {len(self.axis)} points)"

    @property
    def ids(self) -> list[str]:
        return self.meta.index.tolist()

    @property
    def coverage(self) -> np.ndarray:
        """Boolean (N, M) mask: True where an entry has data."""
        return ~np.isnan(self.Y)

    def index_of(self, id_: str) -> int:
        return self.meta.index.get_loc(id_)

    def spectrum(self, id_: str) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of one entry, trimmed to its own coverage."""
        row = self.Y[self.index_of(id_)]
        m = ~np.isnan(row)
        return self.axis[m], row[m]

    def crop(self, lo: float | None = None, hi: float | None = None) -> "Library":
        """New library restricted to axis in [lo, hi]. ``None`` means no limit."""
        m = np.ones(len(self.axis), bool)
        if lo is not None:
            m &= self.axis >= lo
        if hi is not None:
            m &= self.axis <= hi
        return Library(self.axis[m], self.Y[:, m], self.meta.copy(), dict(self.config))

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(self.config, format_version=FORMAT_VERSION)
        np.savez_compressed(
            path.with_suffix(".npz"),
            axis=self.axis,
            Y=self.Y,
            ids=np.array(self.ids, dtype=str),
            config=np.array(json.dumps(cfg)),
        )
        self.meta.to_csv(
            meta_path_for(path.with_suffix(".npz")), index_label="id", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, meta_path: str | Path | None = None) -> "Library":
        path = Path(path).with_suffix(".npz")
        meta_path = Path(meta_path) if meta_path else meta_path_for(path)
        with np.load(path, allow_pickle=False) as z:
            axis, Y = z["axis"], z["Y"]
            ids = [str(i) for i in z["ids"]]
            config = json.loads(str(z["config"]))
        meta = pd.read_csv(meta_path, index_col="id", keep_default_na=False, encoding="utf-8")
        missing = set(ids) - set(meta.index)
        if missing:
            raise ValueError(
                f"{len(missing)} ids in {path.name} missing from {meta_path.name}, "
                f"e.g. {sorted(missing)[:3]}"
            )
        meta = meta.loc[ids]  # align to the matrix row order; extra csv rows are ignored
        return cls(axis, Y, meta, config)
