from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Any
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import natsort
from zuy.common.logger import setup_logger
from zuy.spectrum.detect_peaks import detect_peaks
from zuy.spectrum.baseline import (
    baseline_subtract_classic,
    baseline_subtract_arpls,
    baseline_subtract_arpls_cholesky,
)

from scipy.ndimage import uniform_filter1d

logger = setup_logger(__name__)


class SpectrumProtocol(Protocol):
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    metadata: dict[str, Any]


class ProcessMixin:
    def smooth(self, window_size: int = 5) -> None:
        """Smooth the spectrum data in-place using a moving average filter."""
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.y = uniform_filter1d(self.y, size=window_size)

    def baseline_correction(
        self, niter: int = 10, lam: float = 2e7, p: float = 0.05, y_offset: float = 0
    ) -> None:
        """
        Apply baseline correction in-place using asymmetric least squares.

        Parameters
        ----------
        niter : int
            Number of iterations for the ALS algorithm.
        lam : float
            Smoothness parameter for baseline correction.
        p : float
            Asymmetry parameter for baseline correction.
        y_offset : float
            Constant offset added after baseline correction to avoid
            negative or zero values.

        Notes
        -----
        Negative values after baseline subtraction are clamped to a small positive number.
        """
        y = baseline_subtract_classic(self.y, niter=niter, lam=lam, p=p)
        # y = baseline_subtract_arpls(self.y, lam=1e-3, ratio=1e-6, niter=10)
        # y = baseline_subtract_arpls_cholesky(self.y, lam=1e-2, ratio=0.05, itermax=10)
        y += y_offset
        self.y = np.maximum(y, 1e-10)


class PlotMixin:
    def plot(
        self: SpectrumProtocol,
        mph_perc: float = 1,
        mpd: int = 5,
        thl_perc: float = 0.05,
        ax: plt.Axes | None = None,
    ) -> None:

        y = np.copy(self.y)
        x = self.x
        y_max = y.max()
        peak_idx = detect_peaks(
            y,
            mph=y_max * mph_perc / 100,
            edge="both",
            mpd=mpd,
            kpsh=True,
            threshold=y_max * thl_perc / 100,
        )

        peaks_x = x[peak_idx]
        peaks_y = y[peak_idx]

        ax = ax or plt.gca()
        ax.plot(x, y, label=self.metadata.get("TITLE", "Spectrum"), linewidth=1)
        ax.scatter(peaks_x, peaks_y, s=3)

        for px, py in zip(peaks_x.round(2), peaks_y):
            ax.annotate(
                f"{px:.2f}",
                xy=(px, py),
                xytext=(px, py * 1),
                fontsize=6,
                color="#666666",
            )


@dataclass
class BaseSpectrum:
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y shapes must match, got {self.x.shape} and {self.y.shape}"
            )


@dataclass
class EmsaSpectrum(BaseSpectrum, PlotMixin, ProcessMixin):
    @classmethod
    def from_msa(cls, fpath: Path) -> "EmsaSpectrum":
        x, y, metadata = parse_msa_file(fpath)
        return cls(x=x, y=y, metadata=metadata)


def parse_msa_file(fpath: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    x: list[float] = []
    y: list[float] = []
    data_started = False
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not data_started and line.upper().startswith("#SPECTRUM"):
                data_started = True
                continue
            if line.startswith("#"):
                if ":" in line:
                    key, val = line[1:].split(":", maxsplit=1)
                    metadata[key.strip().upper()] = val.strip()
                continue
            if data_started:
                xi, yi = line.split(",", maxsplit=1)
                x.append(float(xi))
                y.append(float(yi))
    return np.array(x), np.array(y), metadata


def plot_emsa_files(fpaths: list[Path], **kwargs) -> None:
    if not fpaths:
        logger.warning(f"No .msa files to process")
        return

    for f in fpaths:
        s = EmsaSpectrum.from_msa(f)
        logger.info(f"Processing: {f.name}")
        logger.info(f"Metadata: {s.metadata}")
        s.smooth(5)
        s.baseline_correction(niter=10, lam=2e7, p=0.05)
        s.plot()

    logger.info(f"Completed")


if __name__ == "__main__":
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("linestyle", ["-", "--", ":", "-."])
            * plt.rcParams["axes.prop_cycle"]
        ),
    )

    indir = Path(__file__).parent.parent / "data"
    fpaths = natsort.natsorted(indir.rglob("*.msa"), key=str)
    logger.info(fpaths)

    plt.figure(figsize=(6, 4))

    plot_emsa_files(fpaths)

    ax = plt.gca()
    from zuy.spectrum.squre_root_scale import SqrtScale

    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50, top=None)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
