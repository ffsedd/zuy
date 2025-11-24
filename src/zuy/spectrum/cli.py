# zuy/spectrum/cli.py
from pathlib import Path
import natsort
import matplotlib.pyplot as plt
from .processing import smooth, baseline_correction
from .plotting import plot_multiple_spectra
from .io import parse_msa_file
from zuy.common.logger import setup_logger

from .squre_root_scale import register_sqrt_scale

register_sqrt_scale()

logger = setup_logger(__name__)


def main(data_dir: str | Path):
    data_dir = Path(data_dir)
    fpaths = natsort.natsorted(data_dir.rglob("*.msa"), key=str)
    spectra = []

    for f in fpaths:
        s = parse_msa_file(f)
        logger.info(f"Processing {f.name} with metadata {s.metadata}")
        s.y = smooth(s.y, 5)
        s.y = baseline_correction(s.y, niter=10, lam=2e7, p=0.05)
        spectra.append(s)

    plt.rc(
        "axes",
        prop_cycle=plt.cycler("linestyle", ["-", "--", ":", "-."])
        * plt.rcParams["axes.prop_cycle"],
    )
    plot_multiple_spectra(spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)
    plt.show()


if __name__ == "__main__":
    main(Path(__file__).parent.parent / "data")
