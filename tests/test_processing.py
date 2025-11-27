# tests/test_processing.py
import numpy as np
import pytest

from zuy.spectrum.processing import smooth


def test_smooth_basic():
    y = np.array([0, 0, 0, 10, 0, 0, 0], dtype=float)
    y_sm = smooth(y, n=3)
    # output should be smoothed, not equal to input
    assert y_sm.shape == y.shape
    assert np.any(y_sm != y)
    # edges should be smaller than central peak
    assert y_sm[3] > y_sm[0]


def test_smooth_invalid_window():
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        smooth(y, 0)
