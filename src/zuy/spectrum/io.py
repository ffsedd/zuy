from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .models import Spectrum


def parse_msa_file(fpath: Path) -> Spectrum:
    """
    Parse a text-based MSA file into a Spectrum.

    Expected structure:
        * Header metadata: lines beginning with one or more '#'
          Format optionally `#KEY: VALUE` (case-normalized key preservation).
        * A line starting with '#SPECTRUM' begins numeric XY data.
        * Data lines: 'x, y' float pairs.

    Parameters
    ----------
    fpath : Path
        File path to `.msa` file.

    Returns
    -------
    Spectrum
        Parsed spectrum object.

    Raises
    ------
    ValueError
        On malformed numeric lines or missing spectrum data.
    """

    metadata: dict[str, str] = {}
    xs: list[float] = []
    ys: list[float] = []
    in_data = False

    with fpath.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                upper = line.upper()
                if upper.startswith("#SPECTRUM"):
                    in_data = True
                    continue
                _parse_metadata_line(line, metadata)
                continue

            if not in_data:
                # non-comment before SPECTRUM marker -> ignore silently
                continue

            try:
                xi, yi = _parse_xy(line)
            except Exception as e:
                raise ValueError(f"Invalid numeric XY pair in {fpath}: {line}") from e

            xs.append(xi)
            ys.append(yi)

    if not xs:
        raise ValueError(f"No numeric spectrum data found in {fpath}")

    return Spectrum(x=np.array(xs), y=np.array(ys), metadata=metadata)


def _parse_metadata_line(line: str, meta: dict[str, str]) -> None:
    """Parse and store metadata if it contains a key/value."""
    stripped = line.lstrip("#").strip()
    if ":" not in stripped:
        return
    key, val = stripped.split(":", maxsplit=1)
    meta[key.strip().upper()] = val.strip()


def _parse_xy(line: str) -> Tuple[float, float]:
    """Parse a single `x, y` pair from a line."""
    xi, yi = line.split(",", maxsplit=1)
    return float(xi), float(yi)
