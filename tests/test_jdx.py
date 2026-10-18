"""Regression tests for jdx.py: uv run --with pytest --with numpy pytest -q."""

from datetime import datetime

import numpy as np
import pytest

from zuy.spectrum.jdx import Jdx, _read_jcamp, decode_asdf, interpolate_linear, signif


def spectrum(x: list[float], y: list[float], units: str = "ABSORBANCE") -> Jdx:
    return Jdx(x=np.array(x), y=np.array(y), xunits="1/CM", yunits=units)


def test_asdf_sqz_dif_dup_and_nan() -> None:
    values, ends_with_dif = decode_asdf("A2LV")
    assert values == [12, 15, 18, 21, 24]
    assert ends_with_dif
    assert decode_asdf("a2B3")[0] == [-12, 23]
    assert decode_asdf("10 20 -4 ?")[0][:3] == [10, 20, -4]
    assert np.isnan(decode_asdf("?")[0][0])
    assert decode_asdf("1E+2 2E-2")[0] == [100, 0.02]
    assert decode_asdf("A3T")[0] == [13, 13]


def test_asdf_invalid() -> None:
    with pytest.raises(ValueError, match="DUP must follow"):
        decode_asdf("T")
    with pytest.raises(ValueError, match="DIF must follow"):
        decode_asdf("J")
    with pytest.raises(ValueError, match="Unexpected JCAMP"):
        decode_asdf("A$")


def test_xydata_with_checkpoint_and_factors() -> None:
    data = """##TITLE=Compressed
##FIRSTX=0
##LASTX=4
##NPOINTS=5
##XFACTOR=2
##YFACTOR=0.1
##XYDATA=(X++(Y..Y))
0 A J J
2 C J J
##END=
"""
    headers, x, y = _read_jcamp(data)
    assert headers["title"] == "Compressed"
    np.testing.assert_allclose(x, [0, 2, 4, 6, 8])
    np.testing.assert_allclose(y, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_xydata_point_count_validation() -> None:
    data = "##LASTX=2\n##NPOINTS=4\n##XYDATA=(X++(Y..Y))\n0 1 2 3\n##END="
    with pytest.raises(ValueError, match="NPOINTS"):
        _read_jcamp(data)


def test_xy_pairs_scientific_notation(tmp_path) -> None:
    path = tmp_path / "a.jdx"
    path.write_text("""##TITLE=A
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##NPOINTS=3
##XYDATA=(XY..XY)
3 -1e-3
2,2.5E-2
1 ; ?
##END=
""")
    obj = Jdx(path, auto_tidy=False)
    np.testing.assert_equal(obj.x, [3, 2, 1])
    np.testing.assert_allclose(obj.y[:2], [-0.001, 0.025])
    assert np.isnan(obj.y[-1])


def test_save_roundtrip_preserves_original_arrays_and_metadata(tmp_path) -> None:
    obj = spectrum([4000, 2000, 1000], [0.1, 0.2, np.nan])
    obj.date = datetime(2026, 10, 18, 12, 34, 56)
    obj.title = "Example"
    obj.datatype = "INFRARED SPECTRUM"
    before = obj.x.copy()
    path = tmp_path / "spectrum.jdx"
    assert obj.save(path) == path
    np.testing.assert_equal(obj.x, before)
    loaded = Jdx(path, auto_tidy=False)
    np.testing.assert_equal(loaded.x, [1000, 2000, 4000])
    np.testing.assert_allclose(loaded.y[:2], [np.nan, 0.2], equal_nan=True)
    assert loaded.title == "Example"
    assert loaded.date == obj.date
    loaded.save(path)
    assert path.with_suffix(".bak").exists()


def test_arithmetic_returns_independent_spectra() -> None:
    a = spectrum([1, 2], [2, 3])
    b = spectrum([1, 2], [4, 5])
    c = a + b
    assert c is not a
    np.testing.assert_equal(c.y, [6, 8])
    np.testing.assert_equal((b - a).y, [2, 2])
    np.testing.assert_equal((a * 2).y, [4, 6])
    np.testing.assert_equal((2 * a).y, [4, 6])
    np.testing.assert_equal((a / 2).y, [1, 1.5])
    c.y[0] = 999
    assert a.y[0] == 2
    with pytest.raises(ZeroDivisionError):
        a / 0
    with pytest.raises(ValueError, match="matching X"):
        a + spectrum([1, 3], [2, 3])
    with pytest.raises(ValueError, match="matching X and Y units"):
        a + spectrum([1, 2], [2, 3], units="TRANSMITTANCE")


@pytest.mark.parametrize("reverse", [False, True])
def test_clear_in_both_x_directions(reverse: bool) -> None:
    x = np.arange(2000.0, 2600.0, 50)
    y = np.ones_like(x)
    y[(x >= 2250) & (x <= 2450)] = 25
    if reverse:
        x, y = x[::-1], y[::-1]
    obj = spectrum(x.tolist(), y.tolist())
    obj.clear()
    np.testing.assert_allclose(obj.y, 1)
    obj.clear(interp=False)
    assert np.isnan(obj.y[(obj.x >= 2250) & (obj.x <= 2450)]).all()


def test_tidy_preserves_unknown_units() -> None:
    obj = Jdx(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), xunits="Raman shift", yunits="COUNTS")
    obj.tidy()
    assert obj.xunits == "Raman shift"
    np.testing.assert_equal(obj.y, [1, 2, 3])


def test_tidy_wavelength_and_transmittance() -> None:
    obj = Jdx(
        x=np.array([10, 5, 2.5]), y=np.array([1, 0.1, 0.01]), xunits="um", yunits="TRANSMITTANCE"
    )
    obj.tidy(clear_co2=False)
    np.testing.assert_equal(obj.x, [1000, 2000, 4000])
    np.testing.assert_allclose(obj.y, [0, 1, 2])


def test_two_point_calibration_and_interpolation() -> None:
    obj = spectrum([0, 1, 2], [0, 10, 20])
    obj.calibrate_x((0, 2), (1, 3), resample=False)
    np.testing.assert_equal(obj.x, [1, 2, 3])
    np.testing.assert_allclose(
        interpolate_linear(np.array([0, 0.5, 3]), [1, 2], [5, 7]), [3, 4, 9]
    )
    old = spectrum([0, 1, 2], [0, 10, 20])
    old.calibrate_x((0, 2), (1, 3))
    np.testing.assert_equal(old.x, [0, 1, 2])
    np.testing.assert_allclose(old.y, [-10, 0, 10])


def test_text_input_ignores_comments_and_header(tmp_path) -> None:
    path = tmp_path / "test.txt"
    path.write_text("X Y\n# comment\n1 ; 2\n3,4\n5 6\n")
    spec = Jdx(path)
    np.testing.assert_equal(spec.x, [1, 3, 5])
    np.testing.assert_equal(spec.y, [2, 4, 6])


def test_signif_handles_nan_and_zero() -> None:
    values = signif([0, 0.123456, np.nan], 3)
    assert values[0] == 0
    assert values[1] == 0.123
    assert np.isnan(values[2])


def test_scipy_peak_and_smooth() -> None:
    pytest.importorskip("scipy")
    x = np.arange(101, dtype=float)
    y = np.exp(-0.5 * ((x - 50) / 5) ** 2)
    obj = spectrum(x.tolist(), y.tolist())
    peaks = obj.find_peaks(distance=5, prominence=0.1, width=1)
    assert len(peaks) == 1 and peaks[0][0] == 50
    obj.smooth(window_size=11)
    assert np.isfinite(obj.y).all()


def test_xy_pairs_without_delimiter_before_negative(tmp_path) -> None:
    path = tmp_path / "compact.dx"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##XYDATA=(XY..XY)\n4000-2e-3 3999-3e-3\n##END=\n"
    )
    spec = Jdx(path, auto_tidy=False)
    np.testing.assert_array_equal(spec.x, [4000, 3999])
    np.testing.assert_allclose(spec.y, [-0.002, -0.003])


def test_tidy_never_fails_for_short_ir_spectrum() -> None:
    spec = Jdx(
        x=np.array([2200, 2300, 2400]), y=np.array([1, 2, 3]), xunits="1/CM", yunits="ABSORBANCE"
    )
    spec.tidy()
    np.testing.assert_array_equal(spec.y, [1, 2, 3])


def test_save_reload_keeps_input_unmodified(tmp_path) -> None:
    spec = spectrum([3, 2, 1], [1, 2, 3])
    original_y = spec.y.copy()
    fpath = tmp_path / "roundtrip.dx"
    spec.save(fpath)
    np.testing.assert_array_equal(spec.y, original_y)
    other = Jdx(fpath, auto_tidy=False)
    np.testing.assert_array_equal(other.x, [1, 2, 3])
    np.testing.assert_array_equal(other.y, [3, 2, 1])


def test_raman_tidy_does_not_erase_co2_region() -> None:
    obj = Jdx(
        x=np.array([2100, 2200, 2300, 2400, 2500]),
        y=np.array([1, 1, 10, 1, 1]),
        xunits="1/CM",
        yunits="COUNTS",
        datatype="RAMAN SPECTRUM",
    )
    obj.tidy()
    assert obj.y[2] == 10


def test_repeated_tidy_does_not_relabel_unsupported_xunits() -> None:
    obj = Jdx(x=np.array([100, 110]), y=np.array([5, 6]), xunits="channel", yunits="COUNTS")
    obj.tidy().tidy()
    assert obj.xunits == "channel"
