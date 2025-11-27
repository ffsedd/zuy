# zuy/spectrum/processing.py
from __future__ import annotations

import numpy as np
from pybaselines import Baseline  # type: ignore

from .models import Spectrum


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
    bl = Baseline(x_data=s.x)
    baseline, _ = bl.asls(s.y, lam=lam, p=p)
    return Spectrum(x=s.x.copy(), y=s.y - baseline, metadata=s.metadata.copy())


def baseline_correct_arpls(s: Spectrum, lam: float = 1e2, **kwargs) -> Spectrum:
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
