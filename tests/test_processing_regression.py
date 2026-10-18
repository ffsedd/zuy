"""Characterize legacy processing before extracting it from Jdx."""

from pathlib import Path

import numpy as np
import pytest

from zuy.spectrum.jdx import Jdx, interpolate_linear


def test_rmp00084_baseline_then_smooth_preserves_metadata_and_axis(tmp_path: Path) -> None:
    reference = Path(__file__).parent / "fixtures" / "raman_processing_reference.npz"
    with np.load(reference) as data:
        input_x, input_y = data["x"], data["y"]
        obj = Jdx(x=input_x, y=input_y, title="RMP00084", auto_tidy=False)
        obj.datatype, obj.xunits, obj.yunits = "RAMAN SPECTRUM", "1/CM", "RELATIVE INTENSITY"
        obj.dataprocessing = "Instrument processing history"
        assert obj.baseline_correction() is obj
        np.testing.assert_allclose(obj.y, data["corrected"], rtol=1e-8, atol=1e-9)
        assert obj.smooth() is obj
        np.testing.assert_allclose(obj.y, data["smoothed"], rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(obj.x, data["x"])
        np.testing.assert_array_equal(input_y, data["y"])
        np.testing.assert_array_equal(input_x, data["x"])
        assert obj.title == "RMP00084"
        assert obj.dataprocessing == "Instrument processing history"
        saved = obj.save(tmp_path / "processed.jdx")
        loaded = Jdx(saved, auto_tidy=False)
        np.testing.assert_allclose(loaded.y, data["smoothed"], rtol=1e-8, atol=1e-9)
        assert loaded.dataprocessing == obj.dataprocessing
        assert (loaded.datatype, loaded.xunits, loaded.yunits) == (
            "RAMAN SPECTRUM",
            "1/CM",
            "RELATIVE INTENSITY",
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_calibration_with_resampling_and_extrapolation(reverse: bool) -> None:
    x = np.arange(5, dtype=float)
    y = 2 * x + 1
    if reverse:
        x, y = x[::-1], y[::-1]
    obj = Jdx(x=x, y=y)
    assert obj.calibrate_x((0, 4), (1, 5)) is obj
    np.testing.assert_array_equal(obj.x, x)
    np.testing.assert_allclose(obj.y, 2 * x - 1)
    np.testing.assert_array_equal(y, 2 * x + 1)


def test_clear_preserves_unbracketed_regions_and_nonselected_values() -> None:
    obj = Jdx(x=np.arange(5, dtype=float), y=np.array([9, 2, 100, 6, 8], dtype=float))
    assert obj.clear((2, 2)) is obj
    np.testing.assert_array_equal(obj.y, [9, 2, 4, 6, 8])
    obj.clear((0, 1))
    np.testing.assert_array_equal(obj.y, [9, 2, 4, 6, 8])
    obj.clear((3, 3), interp=False)
    assert np.isnan(obj.y[3])
    np.testing.assert_array_equal(obj.y[[0, 1, 2, 4]], [9, 2, 4, 8])


def test_interpolation_rejects_duplicate_reference_coordinates() -> None:
    with pytest.raises(ValueError, match="unique"):
        interpolate_linear([0, 1], [1, 1], [3, 4])


@pytest.mark.parametrize(("lam", "p", "niter"), [(0, 0.005, 10), (2e7, 1, 10), (2e7, 0.005, 0)])
def test_baseline_invalid_settings_do_not_mutate(lam: float, p: float, niter: int) -> None:
    obj = Jdx(x=np.arange(5, dtype=float), y=np.ones(5))
    with pytest.raises(ValueError, match="Require"):
        obj.baseline_correction(lam=lam, p=p, niter=niter)
    np.testing.assert_array_equal(obj.y, np.ones(5))


def test_baseline_rejects_nonfinite_y_without_mutation() -> None:
    obj = Jdx(x=np.arange(3, dtype=float), y=np.array([1, np.nan, 3]))
    with pytest.raises(ValueError, match="finite Y"):
        obj.baseline_correction()
    np.testing.assert_array_equal(obj.y, [1, np.nan, 3])


def test_smoothing_rejects_short_spectra_without_mutation() -> None:
    obj = Jdx(x=np.arange(5, dtype=float), y=np.arange(5, dtype=float))
    with pytest.raises(ValueError, match="31 points"):
        obj.smooth()
    np.testing.assert_array_equal(obj.y, np.arange(5, dtype=float))


def test_intensity_conversion_preserves_legacy_invalid_value_policy() -> None:
    obj = Jdx(x=np.arange(4, dtype=float), y=np.array([1, 0.1, 0, -1]), yunits="TRANSMITTANCE")
    assert obj.to_absorbance() is obj
    np.testing.assert_allclose(obj.y, [0, 1, np.nan, np.nan], equal_nan=True)
    assert obj.yunits == "ABSORBANCE"
    assert obj.to_transmittance() is obj
    np.testing.assert_allclose(obj.y, [1, 0.1, np.nan, np.nan], equal_nan=True)
