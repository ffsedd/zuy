<<<<<<< HEAD
# zuy/spectrum/cli.py
=======
#!/usr/bin/env python3
from __future__ import annotations

import argparse
>>>>>>> 9a9d1ef (spectrumcli)
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import natsort  # type: ignore
from cycler import cycler  # type: ignore

from zuy.common.logger import setup_logger
from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.plotting import plot_multiple_spectra
from zuy.spectrum.processing import tidy_spectrum
from zuy.spectrum.squre_root_scale import register_sqrt_scale

register_sqrt_scale()
<<<<<<< HEAD

logger = setup_logger(__name__)


def main(data_dir: str | Path = ".") -> None:
    data_dir = Path(data_dir)
    fpaths = natsort.natsorted(data_dir.rglob("*.msa"), key=str)
    spectra = [parse_msa_file(f) for f in fpaths]
    tidy_spectra = [tidy_spectrum(s, smooth_window=5) for s in spectra]
=======
logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot raw and tidied spectra from .msa files")
    p.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory to search for .msa files (recursive)",
    )
    p.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Plot only a single .msa file",
    )
    p.add_argument(
        "-w",
        "--smooth-window",
        type=int,
        default=5,
        help="Smoothing window for tidy_spectrum()",
    )
    return p.parse_args()


def load_from_dir(dpath: Path) -> list:
    fpaths = natsort.natsorted(dpath.rglob("*.msa"), key=str)
    return [parse_msa_file(f) for f in fpaths]


def load_single(fpath: Path) -> list:
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    return [parse_msa_file(fpath)]


def main(data_dir: Path, fpath: Path | None, smooth_window: int) -> None:
    if fpath is not None:
        spectra = load_single(fpath)
    else:
        spectra = load_from_dir(data_dir)

    tidy_spectra = [tidy_spectrum(s, smooth_window=smooth_window) for s in spectra]
>>>>>>> 9a9d1ef (spectrumcli)

    plt.rc(
        "axes",
        prop_cycle=cycler("linestyle", ["-", "--", ":", "-."]) * plt.rcParams["axes.prop_cycle"],
    )
<<<<<<< HEAD
    plot_multiple_spectra(spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)
=======

    plot_multiple_spectra(spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(0, 14)
    ax.set_ylim(-50)
>>>>>>> 9a9d1ef (spectrumcli)

    plot_multiple_spectra(tidy_spectra)
    ax = plt.gca()
    ax.set_yscale("sqrt")
<<<<<<< HEAD
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)
=======
    ax.set_xlim(0, 14)
    ax.set_ylim(-50)

>>>>>>> 9a9d1ef (spectrumcli)
    plt.show()


if __name__ == "__main__":
<<<<<<< HEAD
    main(Path(__file__).parent.parent / "data")
=======
    args = parse_args()
    main(args.data_dir, args.file, args.smooth_window)
>>>>>>> 9a9d1ef (spectrumcli)
