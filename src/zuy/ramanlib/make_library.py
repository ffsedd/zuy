#!/usr/bin/env python3
""" """

import logging

from pathlib import Path
import pickle
from zumpy.qq.spectrum.spectrum import Spectrum


class LibraryManager:
    def __init__(self, path):
        self.path = Path(path)
        self.library_step = 1
        self.data = dict()

        if self.path.is_file():
            self.load_data()

        self.errors = []

    def __repr__(self):
        return f"LibraryManager {self.path} - {len(self.data)} spectra"

    def save_data(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def load_data(self):
        logging.info(f"load data  {self.path}")
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)

    def make(self, source_dir, source_mask="*.jdx", smooth=21):
        self.data = {
            f.stem: self.process(f, smooth=smooth) for f in sorted(source_dir.glob(source_mask))
        }
        self.save_data()
        logging.info(f"Library made and saved: {self.path}, errors: {self.errors}")

    def process(self, f, smooth=21):
        try:
            logging.info(f"Load {f.stem}")
            s = Spectrum(f)
            s.baseline_correction()
            s.smooth(smooth)
            s.resample(1)
            s.rescale_intensity()
        except Exception:
            logging.error(f"loading spectrum {f.name} failed")
            self.errors.append(f.name)
        return s


def make_library(src="/home/m/IRUG/", source_mask="*.jdx", smooth=21):
    src = Path(src)
    lib_manager = LibraryManager(src.parent / f"{src.stem}_sm{smooth}.pkl")
    lib_manager.make(source_dir=src, source_mask=source_mask, smooth=smooth)


def main():
    logging.basicConfig(
        format="%(levelname)s [%(module)s:%(lineno)d] %(message)s", level=logging.INFO
    )
    logging.info(f"Script started: {Path(__file__).resolve()}")

    # ~ make_library(src="/home/m/SOPRANO/",)
    # ~ make_library(src="/home/m/IRUG/",source_mask="R*.jdx")

    lib = "/home/m/DATA/RamanLibs/RAMAN_IRUG_SOPR"

    make_library(src=lib, smooth=71)

    # ~ lib = "/home/m/IRUG.pkl"
    # ~ lib = "/home/m/SOPRANO.pkl"
    # ~ lib = "/home/m/IRUG.pkl"

    LM = LibraryManager(lib)
    print(LM)


if __name__ == "__main__":
    main()
