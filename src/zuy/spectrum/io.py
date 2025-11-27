from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from zuy.spectrum.models import Spectrum


def parse_msa_file(fpath: Path) -> Spectrum:
    """
    Parse a text-based MSA file into a `Spectrum`.

    Format:
        #KEY: VALUE       (optional metadata)
        #SPECTRUM         (marks start of XY numeric block)
        x, y              float pairs

    Ignores any non-comment text before #SPECTRUM.
    """
    meta: dict[str, str | float] = {}
    rows: list[str] = []
    in_data = False

    with fpath.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                name = s.upper()
                if name.startswith("#SPECTRUM"):
                    in_data = True
                    continue
                _parse_meta(s, meta)
                continue

            if in_data:
                rows.append(s)

    if not rows:
        raise ValueError(f"{fpath}: no spectrum data found (#SPECTRUM missing or empty)")

    data = _parse_xy_block(rows, fpath)
    return Spectrum(x=data[:, 0], y=data[:, 1], metadata=meta)


def _parse_meta(line: str, meta: dict[str, str | float]) -> None:
    t = line.lstrip("#").strip()
    if ":" not in t:
        return
    k, v = (x.strip() for x in t.split(":", maxsplit=1))
    meta[k.upper()] = _autotype(v)


def _parse_xy_block(rows: list[str], fpath: Path) -> np.ndarray:
    try:
        arr = np.array([_parse_xy(r) for r in rows], dtype=float)
    except ValueError as e:
        raise ValueError(f"{fpath}: malformed numeric line -> {e}") from e

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{fpath}: expected Nx2 numeric array, got shape {arr.shape}")

    return arr


def _parse_xy(line: str) -> Tuple[float, float]:
    x, y = (s.strip() for s in line.split(",", maxsplit=1))
    return float(x), float(y)


def _autotype(v: str) -> str | float:
    try:
        return float(v)
    except ValueError:
        return v
