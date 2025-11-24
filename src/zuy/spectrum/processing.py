# zuy/spectrum/processing.py
import numpy as np
from scipy.ndimage import uniform_filter1d  # type: ignore
from .baseline import baseline_subtract_classic


def smooth(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be >=1")
    return uniform_filter1d(y, size=window_size)


def baseline_correction(
    y: np.ndarray,
    niter: int = 10,
    lam: float = 2e7,
    p: float = 0.05,
    y_offset: float = 0,
) -> np.ndarray:
    y_corr = baseline_subtract_classic(y, niter=niter, lam=lam, p=p)
    y_corr += y_offset
    return np.maximum(y_corr, 1e-10)
