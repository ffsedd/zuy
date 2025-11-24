#!/usr/bin/env python3
"""Detect peaks in data based on their amplitude and other features."""

from __future__ import division, print_function
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from matplotlib.axes import Axes

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.4"
__license__ = "MIT"


def detect_peaks(
    x: ArrayLike,
    mph: Optional[float] = None,
    mpd: int = 1,
    threshold: float = 0,
    edge: Optional[str] = "rising",
    kpsh: bool = False,
    valley: bool = False,
    show: bool = False,
    ax: Optional[Axes] = None,
) -> NDArray[np.int_]:
    """
    Detect peaks in data based on amplitude and other criteria.

    Parameters
    ----------
    x : 1D array_like
        Data array.
    mph : float or None, optional
        Minimum peak height. Peaks below this are ignored.
    mpd : int, optional
        Minimum peak distance in number of samples.
    threshold : float, optional
        Threshold relative to neighbors for peak detection.
    edge : {'rising', 'falling', 'both', None}, optional
        Which edge of flat peaks to keep.
    kpsh : bool, optional
        Keep peaks with same height even if closer than mpd.
    valley : bool, optional
        Detect valleys (local minima) instead of peaks.
    show : bool, optional
        Show plot of data and detected peaks.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.

    Returns
    -------
    numpy.ndarray
        Indices of detected peaks.
    """
    x = np.atleast_1d(x).astype(np.float64)
    if x.size < 3:
        return np.array([], dtype=int)

    if valley:
        x = -x

    dx = np.diff(x)
    indnan = np.where(np.isnan(x))[0]

    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    ine, ire, ife = (
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )

    if edge is None:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        edge = edge.lower()
        if edge in ("rising", "both"):
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge in ("falling", "both"):
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    if ind.size and indnan.size:
        invalid = np.unique(np.hstack((indnan, indnan - 1, indnan + 1)))
        ind = ind[~np.in1d(ind, invalid)]

    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    if ind.size and threshold > 0:
        diff_min = np.minimum(x[ind] - x[ind - 1], x[ind] - x[ind + 1])
        ind = np.delete(ind, np.where(diff_min < threshold)[0])

    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])[::-1]]
        idel = np.zeros(ind.size, dtype=bool)

        for i in range(ind.size):
            if not idel[i]:
                window = (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)
                if kpsh:
                    mask = window & (x[ind[i]] > x[ind])
                else:
                    mask = window
                idel = idel | mask
                idel[i] = False

        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot_detect_peaks(x, mph, mpd, threshold, edge, valley, ind, ax=ax)

    return ind


def _plot_detect_peaks(
    x: NDArray[np.float64],
    mph: Optional[float],
    mpd: int,
    threshold: float,
    edge: Optional[str],
    valley: bool,
    ind: NDArray[np.int_],
    ax: Optional[Axes] = None,
) -> None:
    """Plot the detected peaks and the data."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, "b", lw=1)

    if ind.size:
        label = "valley" if valley else "peak"
        label += "s" if ind.size > 1 else ""
        ax.plot(ind, x[ind], "+", mec="r", mew=2, ms=8, label=f"{ind.size} {label}")
        ax.legend(loc="best", framealpha=0.5, numpoints=1)

    ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
    finite_x = x[np.isfinite(x)]
    ymin, ymax = finite_x.min(), finite_x.max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)

    ax.set_xlabel("Data #", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)

    mode = "Valley detection" if valley else "Peak detection"
    ax.set_title(f"{mode} (mph={mph}, mpd={mpd}, threshold={threshold}, edge='{edge}')")

    plt.show()
