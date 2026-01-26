from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import natsort  # type: ignore
from cycler import cycler  # type: ignore

from zuy.common.logger import setup_logger
from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.plotting import plot_multiple_spectra
from zuy.spectrum.processing import tidy_spectrum
from zuy.spectrum.square_root_scale import register_sqrt_scale

register_sqrt_scale()
logger = setup_logger(__name__)


def _find_msa_files(data_dir: Path) -> list[Path]:
    return natsort.natsorted(data_dir.rglob("*.msa"), key=str)


def _plot_spectra(spectra: Iterable, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.rc(
        "axes",
        prop_cycle=cycler("linestyle", ["-", "--", ":", "-."]) * plt.rcParams["axes.prop_cycle"],
    )
    plot_multiple_spectra(list(spectra))
    ax = plt.gca()
    ax.set_yscale("sqrt")
    ax.set_xlim(left=0, right=14)
    ax.set_ylim(bottom=-50)
    ax.set_title(title)
    plt.tight_layout()


def main(data_dir: str | Path = ".") -> None:
    data_dir = Path(data_dir)

    fpaths = _find_msa_files(data_dir)
    if not fpaths:
        logger.warning("No .msa files found in %s", data_dir)
        return

    spectra = [parse_msa_file(f) for f in fpaths]
    tidy_spectra = [tidy_spectrum(s, smooth_window=5) for s in spectra]

    _plot_spectra(spectra, title="Raw spectra")
    _plot_spectra(tidy_spectra, title="Tidy spectra")

    plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MSA spectra")
    parser.add_argument("data_dir", nargs="?", default=".", help="Directory containing .msa files")
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    main(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
