#!/usr/bin/env python3
"""Convert the old pickled library (dict[str, Spectrum]) to the new Library format.

Does NOT need zumpy / zuy: the spectrum classes are replaced by inert stubs, and
only a small allowlist of numpy / pandas / builtin names may be unpickled.
Still, only run it on pickles you made yourself.

    python -m ramanlib2.convert_pkl RamanLibIrugSop.pkl data/RamanLibIrugSop

The stored y values are whatever the old pipeline produced (baseline corrected,
smoothed, 1 cm-1 grid, rescaled 0-1). Use build_library.py (from the source
.jdx files) when you want control over the preprocessing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:  # works as module and as plain script
    from .library import Library
except ImportError:  # pragma: no cover
    from library import Library

log = logging.getLogger(__name__)

_SAFE_BUILTINS = {
    "set",
    "frozenset",
    "dict",
    "list",
    "tuple",
    "slice",
    "range",
    "int",
    "float",
    "complex",
    "bool",
    "str",
    "bytes",
    "bytearray",
    "object",
}
_SAFE_PREFIXES = ("numpy", "pandas", "pathlib", "collections", "datetime", "copyreg", "_codecs")


class _Stub:
    """Stands in for any non-library class found in the pickle (spectrum objects)."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        top = module.split(".")[0]
        if module == "builtins":
            if name in _SAFE_BUILTINS:
                return getattr(__import__("builtins"), name)
            raise pickle.UnpicklingError(f"blocked builtin {name}")
        if top in _SAFE_PREFIXES:
            return super().find_class(module, name)
        if top in ("qq", "zumpy", "zuy"):  # the old spectrum classes
            return type(name, (_Stub,), {})
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


# ---- metadata extraction ---------------------------------------------------
_RE_ID = re.compile(r"^(?P<src>IRUG|SOPR)_(?P<rest>.*)$")
_RE_IRUG_EID = re.compile(r"^(?P<eid>[A-Z]{3}\d+)\s+(?P<rest>.*)$")
_RE_PCODE = re.compile(r"\(\s*([A-Z][A-Za-z]{0,2}\d+(?::\d+)?)\s*\)")  # (PR105) (PB15:1) (NB1)
_RE_CI = re.compile(r"C\.?\s?I\.?\s*(\d{4,6}(?::\d+)?)")
_RE_NM = re.compile(r"(\d{3,4})")


def _clean(v) -> str:
    return " ".join(str(v).split()) if v not in (None,) else ""


def _laser_nm(header: dict) -> float:
    cands = [header.get("LASER", "")]
    m = re.search(r"source:\s*(\d{3,4})\s*nm", str(header.get("INSTRUMENT PARAMETERS", "")))
    if m:
        cands.append(m.group(1))
    m = re.search(r";\s*(\d{3,4})\s*;", str(header.get("TITLE", "")))
    if m:
        cands.append(m.group(1))
    for c in cands:
        m = _RE_NM.search(str(c))
        if m and 400 <= int(m.group(1)) <= 1100:
            return float(m.group(1))
    return float("nan")


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_meta(key: str, spec) -> dict:
    """Metadata row for one entry. Heuristic fields (name, group, code) are meant
    to be corrected by hand in the csv when needed."""
    h = spec._header
    m = _RE_ID.match(key)
    source = m.group("src") if m else ""
    rest = m.group("rest") if m else key
    names = str(h.get("NAMES", ""))

    if source == "IRUG":
        e = _RE_IRUG_EID.match(rest)
        entry_id = e.group("eid") if e else ""
        label = e.group("rest") if e else rest
        name = label.split(",")[0].strip()
        pc = _RE_PCODE.search(label) or _RE_PCODE.search(str(h.get("TITLE", "")))
        code = pc.group(1) if pc else ""
        ci = _RE_CI.search(label) or _RE_CI.search(names)
    else:  # SOPR: TITLE is the pigment code
        entry_id = rest.split("_")[0]
        code = _clean(h.get("TITLE", "")) or entry_id
        name = code
        ci = _RE_CI.search(names)

    group = code if code else _norm_name(name)

    idx = spec._data.index.values
    return {
        "source": source,
        "entry_id": entry_id,
        "name": name,
        "pigment_code": code,
        "group": group,
        "ci": ci.group(1) if ci else "",
        "cas": _clean(h.get("CAS REGISTRY NO", "")),
        "color": _clean(h.get("$COLOUR", h.get("COLOR", ""))),
        "laser_nm": _laser_nm(h),
        "title": _clean(h.get("TITLE", "")),
        "x_min": float(idx.min()),
        "x_max": float(idx.max()),
        "n_points": int(len(idx)),
        "orig_x_min": float(h.get("MINX", np.nan)),
        "orig_x_max": float(h.get("MAXX", np.nan)),
        "orig_n_points": int(h.get("NPOINTS", 0) or 0),
        "orig_step": abs(float(h.get("DELTAX", np.nan))),
        "origin_url": str(h.get("ORIGIN", ""))
        if str(h.get("ORIGIN", "")).startswith("http")
        else "",
        "file_stem": Path(str(spec._path)).stem,
    }


# ---- conversion --------------------------------------------------------------
def to_library(d: dict, source_name: str = "") -> Library:
    keys = list(d)
    idx_all = [np.asarray(d[k]._data.index.values, dtype=float) for k in keys]
    for k, ix in zip(keys, idx_all):
        if not np.allclose(ix, np.round(ix)) or not np.allclose(np.diff(ix), 1.0):
            raise ValueError(f"{k}: not on an integer 1 cm-1 grid")
    lo = int(min(ix.min() for ix in idx_all))
    hi = int(max(ix.max() for ix in idx_all))
    axis = np.arange(lo, hi + 1, dtype=np.int32)

    Y = np.full((len(keys), len(axis)), np.nan, dtype=np.float32)
    rows = []
    for i, (k, ix) in enumerate(zip(keys, idx_all)):
        y = d[k]._data.iloc[:, 0].to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError(f"{k}: contains NaN")
        Y[i, (ix.astype(int) - lo)] = y
        rows.append(parse_meta(k, d[k]))

    meta = pd.DataFrame(rows, index=pd.Index(keys, name="id"))
    config = {
        "source_pickle": source_name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "grid_step_cm1": 1,
        "y_scale": "each spectrum scaled to [0, 1]",
        "processing": (
            "as stored in the old pickle: baseline corrected, smoothed "
            "(window not recorded), resampled to 1 cm-1, rescaled 0-1"
        ),
        "sources": dict(Counter(meta["source"])),
    }
    return Library(axis, Y, meta, config)


def convert(pkl: str | Path, dst: str | Path) -> Library:
    pkl = Path(pkl)
    lib = to_library(load_pickle(pkl), source_name=pkl.name)
    lib.save(dst)
    log.info(
        "wrote %s and %s",
        Path(dst).with_suffix(".npz"),
        Path(dst).with_suffix("").name + ".meta.csv",
    )
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkl", help="old library pickle")
    ap.add_argument("dst", help="output path without extension (writes .npz and .meta.csv)")
    args = ap.parse_args()
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    lib = convert(args.pkl, args.dst)
    print(lib)


if __name__ == "__main__":
    main()
