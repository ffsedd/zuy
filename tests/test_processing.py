# tests/test_processing.py
import numpy as np
import pytest
from zuy.spectrum.processing import smooth, baseline_correction


def test_smooth_basic():
    y = np.array([0, 0, 0, 10, 0, 0, 0], dtype=float)
    y_sm = smooth(y, window_size=3)
    # output should be smoothed, not equal to input
    assert y_sm.shape == y.shape
    assert np.any(y_sm != y)
    # edges should be smaller than central peak
    assert y_sm[3] > y_sm[0]


def test_smooth_invalid_window():
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        smooth(y, 0)


def test_baseline_correction_monotone():
    # simple ramp + baseline
    y = np.linspace(0, 10, 50) + 2
    y_corr = baseline_correction(y, niter=5, lam=1e5, p=0.01)
    assert y_corr.shape == y.shape
    # baseline subtraction should reduce values but keep positive
    assert np.all(y_corr >= 1e-10)
    assert np.mean(y_corr) < np.mean(y)


def test_baseline_correction_constant_offset():
    y = np.ones(10) * 5
    y_corr = baseline_correction(y, y_offset=2)
    assert np.all(y_corr >= 2)
