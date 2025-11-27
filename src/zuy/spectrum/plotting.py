# zuy/spectrum/plotting.py
from matplotlib import pyplot as plt  # type: ignore

from .detect_peaks import detect_peaks
from .models import Spectrum


def plot_spectrum(
    s: Spectrum,
    ax: plt.Axes | None = None,
    mph_perc: float = 1,
    mpd: int = 5,
    thl_perc: float = 0.05,
) -> None:
    """
    Plot a single spectrum and annotate peaks.

    Parameters
    ----------
    s : Spectrum
        Spectrum to plot.
    ax : matplotlib.axes.Axes | None, optional
        Axes to plot on. If None, a new figure is created.
    mph_perc : float, optional
        Minimum peak height as a percentage of the maximum amplitude.
    mpd : int, optional
        Minimum peak distance in samples.
    thl_perc : float, optional
        Threshold relative to neighbors for peak detection.

    Returns
    -------
    None
    """
    y = s.y
    x = s.x
    y_max = y.max()

    peak_idx = detect_peaks(
        y,
        mph=y_max * mph_perc / 100,
        edge="both",
        mpd=mpd,
        kpsh=True,
        threshold=y_max * thl_perc / 100,
    )

    ax = ax or plt.gca()
    ax.plot(x, y, label=s.metadata.get("TITLE", "Spectrum"), linewidth=1)
    ax.scatter(x[peak_idx], y[peak_idx], s=3)

    for px, py in zip(x[peak_idx].round(2), y[peak_idx]):
        ax.annotate(
            f"{px:.2f}",
            xy=(px, py),
            xytext=(px, py * 1),
            fontsize=6,
            color="#666666",
        )


def plot_multiple_spectra(spectra: list[Spectrum], figsize=(10, 5), **kwargs):
    plt.figure(figsize=figsize)
    for s in spectra:
        plot_spectrum(s, **kwargs)
    ax = plt.gca()
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf(), ax
