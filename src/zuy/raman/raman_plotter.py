#!/usr/bin/env python3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from zuy.spectrum.jdx import Jdx
from zuy.spectrum.square_root_scale import register_sqrt_scale

register_sqrt_scale()

SPEC_EXTS = [".jdx"]
STD_DIRNAME = "_STD"


class RamanPlotter:
    def __init__(self, dpath, std_spectra=[], *a, **kw):
        logging.info(f"RamanPlotter {dpath} {a} {kw}")

        self.dpath = Path(dpath).resolve()
        self.spectra = self.load_spectra(self.dpath)
        if not len(self.spectra):
            raise FileNotFoundError(f"No spectra found in {self.dpath}")
        self.standards = std_spectra + self.load_spectra(
            self.dpath.parent / STD_DIRNAME
        )  # common standards
        self.standards += self.load_spectra(self.dpath / STD_DIRNAME)  # sample standards
        print("standards", self.standards)
        # ~ print(self)
        self.process_spectra(*a, **kw)
        self.plot_spectra(*a, **kw)

    def __repr__(self):
        return f"{len(self.spectra)} spectra in {self.dpath} \n" + "\n".join(
            f"fname:{s.filepath}\ttitle:{s.title}\tlen:{len(s.x)}" for s in self.spectra
        )

    def load_spectra(self, dpath):
        spectra = []
        fps = sorted(f for f in dpath.glob("*.*") if f.suffix.lower() in SPEC_EXTS)
        logging.debug(f"dpath: {dpath} found files: {fps}")
        if fps:
            spectra = [Jdx(filepath=fp, auto_tidy=False) for fp in fps]
        return spectra

    def process_spectra(self, smooth=11, *a, **kw):
        logging.info("Process spectra...")

        for s in self.spectra:
            s.baseline_correction()
            s.smooth(window_size=smooth)
            s.y[s.y < 0] = np.nan

        for s in self.standards:
            s.smooth(window_size=smooth)
            s.y[s.y < 0] = np.nan

        logging.info("... done")

    def plot_spectra(
        self,
        std_spectra=None,
        minx=0,
        maxx=None,
        maxy=None,
        maxz=None,
        jpegs=False,
        *args,
        **kwargs,
    ):
        """Plots Raman spectra along with standard spectra.

        Args:
            std_spectra (list, optional): List of standard spectra to plot. Defaults to None.
            maxx (float, optional): Maximum x-axis value.
            maxy (float, optional): Maximum y-axis value.
            maxz (float, optional): Maximum z-axis value.
            jpegs (bool, optional): Whether to save JPEGs. Defaults to False.
            *args, **kwargs: Additional arguments for customization.
        """
        std_spectra = std_spectra or []  # Ensure it's a list if None
        xlim = (minx, maxx) if maxx is not None else None
        ylim = (0, maxy) if maxy is not None else None
        ylim2 = (0, maxz) if maxz is not None else None

        fig, ax1 = plt.subplots(figsize=(20, 10))
        highest_maxy = 0

        for spectrum in self.spectra:
            sp_max_y = np.nanmax(spectrum.y)
            logging.debug(f"Process spectrum {spectrum.filepath} max: {sp_max_y}")

            # Scale down if the maximum value exceeds maxy
            if maxy is not None and sp_max_y > maxy:
                # ~ print(f"rescale y from {sp_max_y} to {maxy}")

                spectrum.y = spectrum.y * maxy / sp_max_y

            highest_maxy = max(highest_maxy, np.nanmax(spectrum.y))
            # ~ print(highest_maxy)

            spectrum.plot(ax=ax1, linewidth=1)
            spectrum.plot_peaks()

        # Adjust y-axis based on the highest value found
        ax1.set_ylim(0, highest_maxy)

        # Plot standard spectra on a secondary axis
        ax2 = ax1.twinx()
        for standard in self.standards:
            standard.plot(linestyle="dashed", linewidth=1, ax=ax2)
            standard.plot_peaks()

        # Configure plot settings
        self.plot_config(ax1, ax2, xlim, ylim, ylim2)

        # Save the plot
        output_path = self.dpath.with_suffix(".pdf")
        try:
            self.saveplot(output_path)
        finally:
            plt.close(fig)
        self.plot_single_spectra(jpegs=jpegs, minx=minx, maxx=maxx, maxy=maxy, maxz=maxz)

    def plot_single_spectra(self, jpegs=False, minx=0, maxx=None, maxy=None, maxz=None, **kwargs):
        if not jpegs:
            return
        xlim = (minx, maxx) if maxx is not None else None
        ylim = (0, maxy) if maxy is not None else None
        ylim2 = (0, maxz) if maxz is not None else None
        for spectrum in self.spectra:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            try:
                spectrum.plot(color="red", ax=ax1)
                spectrum.plot_peaks()
                ax2 = ax1.twinx()
                for standard in self.standards:
                    standard.plot(linestyle="dashed", linewidth=1, ax=ax2)
                    standard.plot_peaks()
                self.plot_config(ax1, ax2, xlim, ylim, ylim2)
                self.saveplot(Path(spectrum.filepath).with_suffix(".jpg"))
            finally:
                plt.close(fig)

    def saveplot(self, fpath, *a, **kw):
        fpath = Path(fpath).resolve()
        logging.info(f"saving plot {fpath}")
        plt.savefig(
            fpath,
            dpi=600,
            facecolor="w",
            edgecolor="k",
            orientation="landscape",
            format=None,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.1,
            **kw,
        )
        logging.info("... saved")

    def plot_config(self, ax1, ax2, xlim, ylim, ylim2):
        ax1.set_yscale("sqrt")
        ax2.set_yscale("sqrt")

        ax1.set_xlim(xlim)  #   X AXIS
        # ~ plt.gca().set_ylim([2, 1e5])
        ax1.set_ylim(ylim)  #   Y AXIS
        ax2.set_ylim(ylim2)  #   Y AXIS
        ax1.grid(color="#cccccc")
        ax1.legend(fontsize=10, loc=2)

        ax2.legend(fontsize=10, loc=1)
        ax1.set_ylabel("Raman intensity")
        ax2.set_ylabel("Raman intensity")
        ax1.set_xlabel("Wavenumber (cm$^{-1}$)")

        ax1.xaxis.set_major_locator(MultipleLocator(100))
        ax1.xaxis.set_minor_locator(MultipleLocator(10))


def parse_args():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description=f"{__doc__} path:{Path(__file__).resolve()}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-x", "--maxx", default=None, type=int, help="xlim max")
    p.add_argument("--minx", default=0, type=int, help="xlim min")
    p.add_argument("-y", "--maxy", default=None, type=int, help="ylim max")
    p.add_argument("-z", "--maxz", default=None, type=int, help="ylim2 max")
    p.add_argument("-s", "--smooth", default=11, type=int, help="smooth window")
    p.add_argument("-j", "--jpegs", action="store_true", default=False)
    p.add_argument("-f", "--flag", action="store_true", default=False)
    p.add_argument("-r", "--recursive", action="store_true", default=False)
    p.add_argument(
        "-p", "--path", type=str, default=str(Path(__file__).parent.parent / "data/raman")
    )
    p.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable the debug mode for logging debug statements.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbose (CRITICAL, ERROR, WARN, INFO, DEBUG), eg. -vvvv",
    )
    args = p.parse_args()
    return args


def main():
    args = vars(parse_args())
    print("args:", args)

    root = Path(args["path"]).expanduser().resolve()
    print(f"{__file__} started in {root}")

    logging.basicConfig(
        format="!%(levelno)s [%(module)10s%(lineno)4d]	%(message)s",
        # ~ filename=Path(__file__).with_suffix(".log"), filemode='w'
        level=10,
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    print(f"process dir: {root} \n args:{args}")
    dpaths = [root]
    if args["recursive"]:
        dpaths += sorted(d for d in root.rglob("*") if d.is_dir())

    for dpath in dpaths:
        try:
            RamanPlotter(dpath, **args)
        except FileNotFoundError as e:
            logging.error(e)


if __name__ == "__main__":
    main()
