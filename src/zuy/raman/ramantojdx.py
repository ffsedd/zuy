#!/usr/bin/env python3
""" """

from pathlib import Path
from pprint import pprint

from zuy.spectrum.jdx import Jdx

ROOT = Path().resolve()  # current dir - ./2034v1/spec.csv  ./2034v1/standardy/anatas.jdx
SPEC_EXTS = [".csv", ".txt"]
archive_orig = False


def main():
    fs = sorted(f for f in ROOT.rglob("*.*") if f.suffix in SPEC_EXTS)
    pprint(fs)

    # convert all files recursivelly
    for f in fs:
        print(f)
        try:
            s = Jdx(filepath=f)
            s.datatype = "RAMAN SPECTRUM"
            s.title = f.stem
            s.xunits = "1/CM"
            s.yunits = "INTENSITY"
            s.baseline_correction()
            s.smooth()
            s.save(f.with_suffix(".jdx"))  # save as jdx
            if archive_orig:
                dpath = f.parent / "orig"
                dpath.mkdir(exist_ok=True)
                f.rename(dpath / (f.name + ".bak"))
        except ValueError as e:
            print(f"Spectrum conversion failed: {f}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
