"""Format-independent spectrum processing; array functions never mutate inputs."""

from __future__ import annotations

import numpy as np
import logging
from typing import Any

import numpy.typing as npt

from .models import Spectrum


Array = npt.NDArray[np.float64]
LOG = logging.getLogger(__name__)


def smooth(y: np.ndarray, n: int = 5) -> np.ndarray:
    """
    Simple moving average smoothing.

    Parameters
    ----------
    y : np.ndarray
        Input 1D signal.
    n : int
        Window size for moving average.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.convolve(y, np.ones(n) / n, mode="same")


def baseline_correct(s: Spectrum, lam: float = 1e6, p: float = 0.01) -> Spectrum:
    """
    Baseline correction using pybaselines ASLS method.

    Parameters
    ----------
    s : Spectrum
        Input spectrum to correct.
    lam : float
        Smoothness parameter for ASLS. Higher values produce smoother baselines.
    p : float
        Asymmetry parameter for ASLS. Small values favor fitting the baseline under peaks.

    Returns
    -------
    Spectrum
        New Spectrum with baseline-corrected y.
    """
    from pybaselines import Baseline

    bl = Baseline(x_data=s.x)
    baseline, _ = bl.asls(s.y, lam=lam, p=p)
    return Spectrum(x=s.x.copy(), y=s.y - baseline, metadata=s.metadata.copy())


def baseline_correct_arpls(s: Spectrum, lam: float = 1e2, **kwargs) -> Spectrum:
    from pybaselines import Baseline

    bl = Baseline(x_data=s.x)
    baseline, _ = bl.arpls(s.y, lam=lam)
    return Spectrum(x=s.x.copy(), y=s.y - baseline, metadata=s.metadata.copy())


def tidy_spectrum(
    s: Spectrum,
    smooth_window: int = 5,
    lam: float = 1e6,
    p: float = 0.01,
) -> Spectrum:
    """
    Smooth and baseline-correct a Spectrum object.

    Parameters
    ----------
    s : Spectrum
        Input spectrum.
    smooth_window : int
        Window size for moving average smoothing.
    lam : float
        Smoothness parameter for ASLS baseline correction.
    p : float
        Asymmetry parameter for ASLS baseline correction.

    Returns
    -------
    Spectrum
        New Spectrum with smoothed and baseline-corrected y.
    """
    y_smooth = smooth(s.y, n=smooth_window)
    s_smooth = Spectrum(x=s.x.copy(), y=y_smooth, metadata=s.metadata.copy())
    return baseline_correct(s_smooth, lam=lam, p=p)


def signif(values: npt.ArrayLike, digits: int) -> Array:
    """Round to significant figures without modifying the input."""
    if digits < 1:
        raise ValueError("digits must be positive")
    arr = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        magnitudes = np.where(np.isfinite(arr) & (arr != 0), np.abs(arr), 1.0)
        powers = digits - 1 - np.floor(np.log10(magnitudes))
        scale = np.power(10.0, powers)
        return np.round(arr * scale) / scale


def interpolate_linear(x: npt.ArrayLike, xp: npt.ArrayLike, yp: npt.ArrayLike) -> Array:
    """Linear interpolation and edge extrapolation; XP can be descending."""
    xi = np.asarray(x, dtype=float)
    xs = np.asarray(xp, dtype=float)
    ys = np.asarray(yp, dtype=float)
    if len(xs) < 2:
        raise ValueError("At least two reference points are required")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.any(np.diff(xs) == 0):
        raise ValueError("X points must be unique")
    result = np.interp(xi, xs, ys)
    left = xi < xs[0]
    right = xi > xs[-1]
    result[left] = ys[0] + (xi[left] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    result[right] = ys[-1] + (xi[right] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return result


def _signal(y: npt.ArrayLike, minimum: int = 1) -> Array:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("Y must be one-dimensional")
    if len(values) < minimum:
        raise ValueError(f"Spectrum requires at least {minimum} points")
    return values


def _xy(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[Array, Array]:
    xs, ys = _signal(x), _signal(y)
    if xs.shape != ys.shape:
        raise ValueError("X and Y must be one-dimensional arrays of equal length")
    return xs, ys


def smooth_moving_average(y: npt.ArrayLike, n: int = 5) -> Array:
    """Explicit name for the historical moving-average smoother."""
    return smooth(np.asarray(y, dtype=float), n=n)


def smooth_savgol(
    y: npt.ArrayLike, window_size: int = 31, polyorder: int = 3, **kwargs: Any
) -> Array:
    """Savitzky–Golay smoothing; return new values without changing the input."""
    from scipy.signal import savgol_filter

    values = _signal(y, window_size)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3")
    return np.asarray(
        savgol_filter(values, window_length=window_size, polyorder=polyorder, **kwargs),
        dtype=float,
    )


def estimate_baseline_legacy_asls(
    y: npt.ArrayLike, lam: float = 2e7, p: float = 0.005, niter: int = 10
) -> Array:
    """Estimate Jdx's fixed-iteration ASLS baseline, preserving its algorithm.

    This differs from pybaselines ASLS in baseline_correct(), including defaults
    and iteration/stopping behavior. The input is never modified.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    values = _signal(y, 3)
    if not 0 < p < 1 or lam <= 0 or niter < 1:
        raise ValueError("Require 0 < p < 1, lam > 0 and niter >= 1")
    if not np.isfinite(values).all():
        raise ValueError("Baseline correction needs finite Y values")
    size = len(values)
    differences = sparse.diags([1, -2, 1], [0, -1, -2], shape=(size, size - 2))
    curvature = lam * (differences @ differences.T)
    weights = np.ones(size)
    baseline = np.zeros(size)
    for _ in range(niter):
        weights_matrix = sparse.diags(weights)
        baseline = np.asarray(spsolve(weights_matrix + curvature, weights * values))
        weights = np.where(values > baseline, p, 1 - p)
    return baseline


def find_peak_indices(
    y: npt.ArrayLike,
    distance: int = 40,
    prominence: float = 0.005,
    width: float = 5,
    **kwargs: Any,
) -> npt.NDArray[np.intp]:
    """Find maxima using SciPy; callers choose whether to invert their signal."""
    from scipy.signal import find_peaks

    indices, _ = find_peaks(
        _signal(y), distance=distance, prominence=prominence, width=(width, None), **kwargs
    )
    return indices


def replace_region(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    bounds: tuple[float, float] = (2250, 2450),
    interp: bool = True,
) -> Array:
    """Replace a band with interpolated values or NaNs, preserving axis order.

    Bands without two finite bracketing samples are left unchanged. Return an
    independent Y array even when there is no band to replace.
    """
    xs, ys = _xy(x, y)
    result = ys.copy()
    lo, hi = sorted(bounds)
    selected = (xs >= lo) & (xs <= hi)
    if not selected.any():
        return result
    if not interp:
        result[selected] = np.nan
        return result
    valid = ~selected & np.isfinite(xs) & np.isfinite(ys)
    if np.count_nonzero(valid) < 2:
        LOG.warning("Skipping interpolation: fewer than two points outside %s", bounds)
        return result
    x_valid = xs[valid]
    if min(x_valid) > min(xs[selected]) or max(x_valid) < max(xs[selected]):
        LOG.warning("Skipping interpolation: %s is not bracketed by samples", bounds)
        return result
    order = np.argsort(x_valid)
    result[selected] = np.interp(xs[selected], x_valid[order], ys[valid][order])
    return result


def absorbance_to_transmittance(y: npt.ArrayLike) -> Array:
    """Convert absorbance to fractional transmittance without changing input."""
    return np.power(10.0, -np.asarray(y, dtype=float))


def transmittance_to_absorbance(y: npt.ArrayLike) -> Array:
    """Convert fractional transmittance; nonpositive values become NaNs."""
    values = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0, -np.log10(values), np.nan)


def calibrate_coordinates(
    x: npt.ArrayLike,
    old: tuple[float, float] = (906.7, 2850.7),
    new: tuple[float, float] = (906.7, 2850.7),
) -> Array:
    """Return X coordinates calibrated from two old/new reference points."""
    xs = _signal(x, 2)
    old_a, old_b = old
    new_a, new_b = new
    if old_a == old_b:
        raise ValueError("Calibration reference X points must differ")
    if old == new:
        return xs.copy()
    return new_a + (xs - old_a) * (new_b - new_a) / (old_b - old_a)
