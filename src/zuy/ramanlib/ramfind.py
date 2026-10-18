import logging
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from zumpy.qq.spectrum.spectrum import Spectrum
from matplotlib import pyplot as plt
import pickle
from IPython.display import display

pd.options.display.width = 500
pd.set_option("display.max_colwidth", 100)


class LibrarySearch:
    def __init__(self, library_path):
        self.library_path = Path(library_path)
        self.library_step = 1
        self.result = None
        self.spectrum = None

        self.data = self.load_library(self.library_path)

    def load_library(self, library_file):
        with open(library_file, "rb") as f:
            return pickle.load(f)

    def find(self, spectrum, roi=(None, None), process=True):
        # ~ self.spectrum_path = Path(spectrum_path)
        # ~ self.spectrum = self.process(self.spectrum_path)
        # ~ sp1 = self.spectrum.data.copy()
        self.spectrum = self.process(spectrum) if process else spectrum
        sp1 = spectrum.data.copy()
        a, b = roi
        if a:
            sp1 = sp1.loc[(sp1.index >= a)]
        if b:
            sp1 = sp1.loc[(sp1.index <= b)]

        logging.info(f"FIND {spectrum.title} x: {min(spectrum.x)} - {max(spectrum.x)}")

        results = []
        for stem, sp0 in self.data.items():
            logging.debug(f"compare with {stem} {sp0}")
            logging.debug(f"compare with {sp0.title} x: {min(sp0.x)} - {max(sp0.x)}")
            y1, y0 = self.intersected_spectra(sp1.y, sp0.y)
            # ~ print(y1, y0)

            if len(y1) < 100:
                logging.info(f"...skip, insufficient x overlap: {len(y1)}")
                continue
            try:
                r2 = self.calculate_correlation_coefficient(y1, y0)
                results.append((round(r2, 4), stem, sp0.title, sp0.path))

            except Exception as e:
                logging.error(f"Error comparing spectrum {sp0.path}: {e}")

        df = pd.DataFrame(data=results, columns=["r2", "stem", "title", "path"])
        df = df.sort_values("r2", ascending=False)
        self.result = df
        return df

    def plot(self, number=5):
        if self.result is None:
            logging.error("Nothing to plot")
        else:
            plt.figure(figsize=(25, 12))

            self.spectrum.plot(linewidth=2, color="red")
            self.spectrum.plot_peaks()

            best = [self.data[stem] for stem in self.result.stem.iloc[:number].tolist()]
            print("plot best fit", [s.path.stem for s in best])
            for s in best:
                s.plot(linestyle="dashed", linewidth=1)
                s.plot_peaks()

            plt.tight_layout()
            plt.show(block=False)

    def intersected_spectra_slow(self, spectrum1, spectrum2):
        df = pd.concat([spectrum1, spectrum2], axis=1).dropna()
        df.reset_index(drop=True, inplace=True)  # Reset index for proper alignment
        return df.iloc[:, 0], df.iloc[:, 1]

    def intersected_spectra(self, s1, s2):
        """
        Compare only overlapping parts of spectra.
        Faster numpy alternative to concat in pandas
        """
        common_indices = np.intersect1d(s1.index, s2.index)
        intersected_s1 = s1.values[s1.index.isin(common_indices)]
        intersected_s2 = s2.values[s2.index.isin(common_indices)]
        return intersected_s1, intersected_s2

    def calculate_correlation_coefficient(self, s1, s2):
        correlation_matrix = np.corrcoef(s1, s2)
        correlation_coefficient = correlation_matrix[0, 1]
        return correlation_coefficient

    def calculate_difference(self, s1, s2):
        difference = np.sum((s1 - s2) ** 2)
        return -difference

    def process(self, s):
        logging.info(f"Process {s.path.stem}")
        # ~ logging.info(f"Load {f.stem}")
        # ~ s = Spectrum(f)
        s.baseline_correction()
        s.smooth(21)
        s.resample(1)
        s.rescale_intensity()
        print(s)
        return s


def search_recursive(spectrum, library, roi, view, plot, process=True, results=[]):
    df = library.find(spectrum, roi=roi, process=process).head(view)
    display(df[["r2", "stem", "title"]])
    if plot:
        library.plot(plot)

    spectrum = library.spectrum  # processed

    fit = library.data[df.stem.iloc[0]]
    results.append(fit.copy())
    if len(results) > 5:
        return results

    fit = fit.smooth(71)  # wider peaks

    ignore_indexes_list = fit.data[fit.y > 0.2].index.tolist()
    spectrum.data.loc[spectrum.data.index.isin(ignore_indexes_list), "y"] = np.nan
    # ~ spectrum.plot()

    # ~ plt.show()

    if df.r2.fillna(0).iloc[0] < 0.2:
        return results
    # ~ if input("finish?"):
    # ~ return

    search_recursive(spectrum, library, roi, view, plot, process=False)
    return results


def ramfind(spectrum_path, lib_path=None, plot=7, view=50, roi=(None, None)):
    lib_path = Path(lib_path)
    assert lib_path.is_file(), f"library not found {lib_path}"
    # ~ lib_path = "/home/m/DATA/RamanLibs/RAMAN_IRUG_SOPR_sm71.pkl"

    library = LibrarySearch(lib_path)

    spectrum = Spectrum(spectrum_path)

    results = search_recursive(spectrum, library, roi, view, plot)

    print(results)

    if plot:
        plt.figure(figsize=(20, 10))
        library.process(Spectrum(spectrum_path)).plot()
        for s in results:
            s.plot()
        plt.tight_layout()
        plt.show()


def multisearch(spectrum_path, lib_paths, plot=0, view=50, roi=(None, None)):
    for i, lib in enumerate(reversed(lib_paths)):
        print(lib)
        ramfind(spectrum_path, lib, plot=plot, view=view, roi=roi)


def main():
    p = argparse.ArgumentParser(
        description="Find similar spectra in a library based on a given spectrum."
    )
    p.add_argument(
        "spectrum",
        nargs="?",
        type=str,
        default="/home/m/Dropbox/appy/zumpy/zumpy/raman/ramanlib/samples/2269v2-bod-1--532-1200_0001.jdx",
        help="Path to the spectrum",
    )
    p.add_argument(
        "-l",
        "--library",
        type=str,
        default=Path(__file__).parent / "RamanLibIrugSop.pkl",
        help="Path to the library (PKL).",
    )
    p.add_argument(
        "-p", "--plot", default=7, type=int, help="how many found spectra to plot (default=7)"
    )
    p.add_argument(
        "-v", "--view", default=50, type=int, help="how many found spectra to list (default=50)"
    )
    p.add_argument("-a", default=100, type=int, help="left x limit")
    p.add_argument("-b", default=3000, type=int, help="right x limit")

    args = p.parse_args()

    logging.basicConfig(format="%(levelname)s [%(module)s:%(lineno)d] %(message)s", level=20)
    logging.info(f"Script started: {Path(__file__).resolve()}")

    ramfind(args.spectrum, args.library, args.plot, args.view, (args.a, args.b))


if __name__ == "__main__":
    main()
