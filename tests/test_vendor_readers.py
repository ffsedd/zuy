"""Internal readers must work without the obsolete zumpy package."""

from datetime import datetime
from pathlib import Path
import struct

import numpy as np
import pytest

from zuy.spectrum.jdx import Jdx
from zuy.spectrum.scn import read_scn


@pytest.fixture
def scn_path(tmp_path: Path) -> Path:
    path = tmp_path / "buck.scn"
    path.write_text(
        "BUCK Scientific Model 500 IR\nLead white\n2/14/2018\n3\n0,8\n4000\n3997,6\n0\n"
        "4000, 21,59625\n3999,2, 21,645\n3998,4, 21,79125\n",
        encoding="utf-8",
    )
    return path


def test_scn_decimal_commas_and_exclusive_axis_limit(scn_path: Path, tmp_path: Path) -> None:
    source = read_scn(scn_path)
    np.testing.assert_array_equal(source.x, [4000, 3999.2, 3998.4])
    np.testing.assert_array_equal(source.y, [21.59625, 21.645, 21.79125])
    assert source.instrument == "BUCK Scientific Model 500 IR"
    assert source.title == "Lead white"
    assert source.date == datetime(2018, 2, 14)
    spec = Jdx(scn_path)
    np.testing.assert_array_equal(spec.x, source.x)
    np.testing.assert_array_equal(spec.y, source.y)
    assert spec.datatype == "INFRARED SPECTRUM"
    assert (spec.xunits, spec.yunits) == ("1/CM", "ARBITRARY")
    output = spec.save(tmp_path / "buck.jdx")
    loaded = Jdx(output, auto_tidy=False)
    np.testing.assert_array_equal(loaded.x, source.x[::-1])
    np.testing.assert_array_equal(loaded.y, source.y[::-1])
    assert loaded.date == source.date
    assert loaded.spectrometerdatasystem == source.instrument


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("\n3\n", "\n4\n", "declares 4 points"),
        ("\n0,8\n", "\ninvalid\n", "numeric header"),
        ("3999,2, 21,645", "3999,2, invalid", "line 10"),
        ("\n3\n", "\n0\n", "point count"),
    ],
)
def test_scn_rejects_invalid_data(scn_path: Path, old: str, new: str, message: str) -> None:
    scn_path.write_text(scn_path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        read_scn(scn_path)


def test_scn_short_header_and_unknown_date(tmp_path: Path, scn_path: Path) -> None:
    truncated = tmp_path / "short.scn"
    truncated.write_text("instrument\ntitle\n")
    with pytest.raises(ValueError, match="eight-line header"):
        read_scn(truncated)
    scn_path.write_text(scn_path.read_text().replace("2/14/2018", "unknown acquisition date"))
    source = read_scn(scn_path)
    assert source.date is None
    assert source.acquired == "unknown acquisition date"


def test_internal_spa_reader_and_legacy_alias(tmp_path: Path) -> None:
    path = tmp_path / "legacy.spa"
    contents = bytearray(652)
    contents[30:37] = b"Example"
    struct.pack_into("<HH", contents, 288, 3, 640)
    struct.pack_into("<i", contents, 564, 3)
    struct.pack_into("<ff", contents, 576, 4000, 600)
    struct.pack_into("<fff", contents, 640, 0.1, 0.2, 0.3)
    path.write_bytes(contents)
    spec = Jdx(path, auto_tidy=False)
    np.testing.assert_array_equal(spec.x, [4000, 2300, 600])
    np.testing.assert_allclose(spec.y, [0.1, 0.2, 0.3])
    assert spec.title == "Example"
    assert spec.yunits == "ABSORBANCE"
    alias = Jdx()
    alias.from_spa2(path)
    np.testing.assert_array_equal(alias.y, spec.y)
