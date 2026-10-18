"""Shared numerical functions operate independently of Jdx and never mutate inputs."""

from pathlib import Path

import numpy as np
import pytest

from zuy.spectrum import processing
from zuy.spectrum.jdx import Jdx, interpolate_linear, signif


def test_reference_arrays_match_shared_processing_without_mutation() -> None:
    with np.load(Path(__file__).parent / "fixtures" / "raman_processing_reference.npz") as data:
        x, y = data["x"], data["y"]
        baseline = processing.estimate_baseline_legacy_asls(y)
        corrected = y - baseline
        np.testing.assert_allclose(corrected, data["corrected"], rtol=1e-8, atol=1e-9)
        smoothed = processing.smooth_savgol(corrected)
        np.testing.assert_allclose(smoothed, data["smoothed"], rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(x, data["x"])
        np.testing.assert_array_equal(y, data["y"])
        np.testing.assert_allclose(corrected, data["corrected"], rtol=1e-8, atol=1e-9)
        assert not np.shares_memory(smoothed, corrected)


def test_band_replacement_is_pure_but_jdx_preserves_inplace_array_identity() -> None:
    x = np.arange(5, dtype=float)
    y = np.array([0, 1, 99, 3, 4], dtype=float)
    corrected = processing.replace_region(x, y, (2, 2))
    np.testing.assert_array_equal(corrected, x)
    np.testing.assert_array_equal(y, [0, 1, 99, 3, 4])
    assert not np.shares_memory(corrected, y)
    obj = Jdx(x=x, y=y)
    alias = obj.y
    obj.clear((2, 2))
    assert obj.y is alias
    np.testing.assert_array_equal(alias, x)


def test_shared_interpolation_and_rounding_remain_importable_from_jdx() -> None:
    assert interpolate_linear is processing.interpolate_linear
    assert signif is processing.signif


def test_calibration_preserves_input_and_descending_order() -> None:
    x = np.array([3, 2, 1], dtype=float)
    calibrated = processing.calibrate_coordinates(x, (1, 3), (2, 6))
    np.testing.assert_array_equal(calibrated, [6, 4, 2])
    np.testing.assert_array_equal(x, [3, 2, 1])
    identity = processing.calibrate_coordinates(x)
    np.testing.assert_array_equal(identity, x)
    assert not np.shares_memory(identity, x)


def test_intensity_conversion_and_peak_detection_leave_inputs_unchanged() -> None:
    y = np.array([1, 0.1, 0.01])
    absorbance = processing.transmittance_to_absorbance(y)
    np.testing.assert_allclose(absorbance, [0, 1, 2])
    np.testing.assert_allclose(processing.absorbance_to_transmittance(absorbance), y)
    np.testing.assert_array_equal(y, [1, 0.1, 0.01])
    np.testing.assert_array_equal(
        processing.find_peak_indices([0, 1, 0], distance=1, prominence=0.1, width=0), [1]
    )
    np.testing.assert_array_equal(processing.smooth_moving_average([0, 3, 0], n=3), [1, 1, 1])


@pytest.mark.parametrize("window_size", [0, 2, 4])
def test_savgol_rejects_invalid_windows(window_size: int) -> None:
    with pytest.raises(ValueError, match="odd and >= 3"):
        processing.smooth_savgol(np.arange(5), window_size=window_size)


def test_array_processing_rejects_mismatched_shapes_and_nonfinite_baseline() -> None:
    with pytest.raises(ValueError, match="equal length"):
        processing.replace_region([1, 2], [3])
    with pytest.raises(ValueError, match="finite Y"):
        processing.estimate_baseline_legacy_asls([1, np.inf, 2])
