# zuy/spectrum/cli.py
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import natsort  # type: ignore
from cycler import cycler  # type: ignore

from zuy.common.logger import setup_logger

from .io import parse_msa_file
from .plotting import plot_multiple_spectra
from .processing import tidy_spectrum
from .squre_root_scale import register_sqrt_scale

register_sqrt_scale()

logger = setup_logger(__name__)


def main(data_dir: str | Path = ".") -> None:
    data_dir = Path(data_dir)
    fpaths = natsort.natsorted(data_dir.rglob("*.msa"), key=str)
    spectra = [parse_msa_file(f) for f in fpaths]
    tidy_spectra = [tidy_spectrum(s, smooth_window=5) for s in spectra]

    plt.rc(
        "axes",
        prop_cycle=cycler("linestyle", ["-", "--", ":", "-."]) * plt.rcParams["axes.prop_cycle"],
    )
    plot_multiple_spectra(spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)

    plot_multiple_spectra(tidy_spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)
    plt.show()


if __name__ == "__main__":
    main(Path(__file__).parent.parent / "data")
