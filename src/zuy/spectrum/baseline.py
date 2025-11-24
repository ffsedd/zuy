#!/usr/bin/env python3
"""Example script for comparing baseline correction methods on a spectrum."""

from __future__ import annotations

import numpy as np
from numpy import ndarray
import scipy as sp  # type: ignore


def baseline_subtract_classic(
    y: ndarray, niter: int = 10, lam: float = 2e7, p: float = 0.005
) -> ndarray:
    """Simple asymmetric least squares baseline correction."""
    if niter < 1:
        raise ValueError("niter must be >= 1")

    L: int = len(y)
    D = sp.sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))  # type: ignore
    w = np.ones(L)

    for _ in range(niter):
        W = sp.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z: ndarray = np.array(sp.sparse.linalg.spsolve(Z, w * y))
        w = p * (y > z) + (1 - p) * (y < z)

    corr = y - z  # type: ignore
    if np.isnan(corr).all():
        raise ValueError("baseline correction failed")

    return corr


def baseline_subtract_arpls(
    y: ndarray, lam: float = 100, ratio: float = 1e-6, niter: int = 10
) -> ndarray:
    """
    Perform baseline correction on a 1D signal using the Asymmetric
    Reweighted Penalized Least Squares (arPLS) algorithm.

    Parameters
    ----------
    y : ndarray
        Input signal (1D array) to correct the baseline.
    lam : float, optional
        Smoothness parameter that controls the baseline flexibility.
        Larger values result in smoother baselines. Default is 100.
    ratio : float, optional
        Convergence threshold for the iterative reweighting.
        Iterations stop if relative change in weights is below this value.
        Default is 1e-6.
    niter : int, optional
        Maximum number of iterations to perform. Must be >= 1.
        Default is 10.

    Returns
    -------
    ndarray
        Estimated baseline of the input signal `y`.

    Raises
    ------
    ValueError
        If `niter` is less than 1.

    Notes
    -----
    This implementation uses sparse matrix operations for efficiency.
    The baseline is iteratively estimated by reweighting residuals to
    emphasize negative deviations, which correspond to the baseline.
    """
    if niter < 1:
        raise ValueError("niter must be >= 1")

    L: int = len(y)
    diag = np.ones(L - 2)
    D = sp.sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D @ D.T

    w = np.ones(L)
    W = sp.sparse.spdiags(w, 0, L, L)

    for _ in range(niter):
        z: ndarray = np.array(sp.sparse.linalg.spsolve(W + H, W @ y))
        d = y - z
        dn = d[d < 0]
        m, s = np.mean(dn), np.std(dn)
        w_new = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        crit: float = float(np.linalg.norm(w_new - w) / np.linalg.norm(w))
        w = w_new
        W.setdiag(w)

        if crit < ratio:
            break
    else:
        print("arPLS: Max iterations exceeded")

    return z  # type: ignore


def baseline_subtract_arpls_cholesky(
    y: ndarray, lam: float = 1e4, ratio: float = 0.05, itermax: int = 100
) -> ndarray:
    """arPLS with Cholesky solver (as in Baek et al. 2015)."""

    N: int = len(y)
    D = sp.sparse.eye(N, format="csc")
    D = D[1:] - D[:-1]  # type: ignore
    H = lam * D.T @ D

    w = np.ones(N)
    for _ in range(itermax):
        W = sp.sparse.diags(w, 0, shape=(N, N))
        WH = sp.linalg.csc_matrix(W + H)
        C = sp.linalg.cholesky(WH.todense())
        z: ndarray = np.array(
            sp.sparse.linalg.spsolve(C, sp.sparse.linalg.spsolve(C.T, w * y))
        )
        d = y - z
        dn = d[d < 0]
        m, s = np.mean(dn), np.std(dn)
        w_new = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - w_new) / np.linalg.norm(w) < ratio:
            break
        w = w_new

    return z  # type: ignore


if __name__ == "__main__":
    ...
