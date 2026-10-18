from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ramanlib2.library import Library

REAL = Path(__file__).resolve().parents[1] / "data" / "RamanLibIrugSop.npz"


def _toy() -> Library:
    axis = np.arange(100, 200)
    Y = np.full((3, 100), np.nan, np.float32)
    Y[0, :] = np.linspace(0, 1, 100)
    Y[1, 20:80] = 0.5
    Y[2, 50:] = 1.0
    meta = pd.DataFrame(
        {"name": ["a", "b", "c"], "group": ["g1", "g1", "g2"]},
        index=pd.Index(["x0", "x1", "x2"], name="id"),
    )
    return Library(axis, Y, meta, {"note": "toy"})


def test_roundtrip(tmp_path):
    lib = _toy()
    lib.save(tmp_path / "toy")
    back = Library.load(tmp_path / "toy.npz")
    assert back.ids == lib.ids
    assert np.array_equal(back.axis, lib.axis)
    assert np.array_equal(back.Y, lib.Y, equal_nan=True)
    assert back.meta["group"].tolist() == ["g1", "g1", "g2"]
    assert back.config["note"] == "toy" and back.config["format_version"] == 1


def test_meta_reordered_csv_is_aligned(tmp_path):
    lib = _toy()
    lib.save(tmp_path / "toy")
    p = tmp_path / "toy.meta.csv"
    df = pd.read_csv(p).iloc[::-1]
    df.to_csv(p, index=False)
    assert Library.load(tmp_path / "toy.npz").ids == lib.ids


def test_missing_meta_rows_raise(tmp_path):
    lib = _toy()
    lib.save(tmp_path / "toy")
    p = tmp_path / "toy.meta.csv"
    pd.read_csv(p).iloc[:2].to_csv(p, index=False)
    with pytest.raises(ValueError):
        Library.load(tmp_path / "toy.npz")


def test_spectrum_and_crop():
    lib = _toy()
    x, y = lib.spectrum("x1")
    assert x[0] == 120 and x[-1] == 179 and len(y) == 60
    c = lib.crop(150, 170)
    assert c.axis[0] == 150 and c.axis[-1] == 170 and len(c) == 3


def test_rejects_bad_input():
    meta = pd.DataFrame(index=pd.Index(["a", "a"], name="id"))
    with pytest.raises(ValueError):
        Library(np.arange(5), np.zeros((2, 5)), meta)  # duplicate ids
    meta = pd.DataFrame(index=pd.Index(["a"], name="id"))
    with pytest.raises(ValueError):
        Library(np.array([3, 2, 1]), np.zeros((1, 3)), meta)  # axis not increasing


@pytest.mark.skipif(not REAL.exists(), reason="converted library not present")
def test_real_library_invariants():
    lib = Library.load(REAL)
    assert len(lib) == 1322
    assert lib.meta["source"].value_counts().to_dict() == {"IRUG": 1003, "SOPR": 319}
    assert (lib.meta["group"] != "").all()
    finite = lib.Y[~np.isnan(lib.Y)]
    assert finite.min() == 0.0 and finite.max() == pytest.approx(1.0)
    # every entry's coverage matches its recorded range
    for i, id_ in enumerate(lib.ids[::97]):
        x, _ = lib.spectrum(id_)
        row = lib.meta.loc[id_]
        assert x[0] == row["x_min"] and x[-1] == row["x_max"] and len(x) == row["n_points"]
