"""Named processing orders, independent of spectrum file formats.

Raman conversion historically corrects the baseline before Savitzky–Golay
smoothing. MSA processing smooths first with a moving average, then applies
pybaselines ASLS. Keep those workflows distinct to preserve numerical behavior.
"""

import numpy as np
import numpy.typing as npt

from . import processing
from .models import Spectrum


def raman_baseline_then_smooth(
    y: npt.ArrayLike,
    *,
    lam: float = 2e7,
    p: float = 0.005,
    niter: int = 10,
    window_size: int = 31,
    polyorder: int = 3,
) -> processing.Array:
    """Return baseline-corrected, Savitzky–Golay-smoothed Raman intensities."""
    values = np.asarray(y, dtype=float)
    baseline = processing.estimate_baseline_legacy_asls(values, lam, p, niter)
    return processing.smooth_savgol(values - baseline, window_size, polyorder)


def msa_smooth_then_baseline(
    s: Spectrum,
    *,
    smooth_window: int = 5,
    lam: float = 1e6,
    p: float = 0.01,
) -> Spectrum:
    """Return a new MSA spectrum using the historical tidy_spectrum order."""
    y_smooth = processing.smooth(s.y, n=smooth_window)
    smoothed = Spectrum(x=s.x.copy(), y=y_smooth, metadata=s.metadata.copy())
    return processing.baseline_correct(smoothed, lam=lam, p=p)
