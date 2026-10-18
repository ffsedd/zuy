#!/usr/bin/env python3


from pprint import pprint

from pathlib import Path

from collections import defaultdict

# ~ from zumpy.ir.jdx import Jdx
from zumpy.qq.spectrum.spectrum import RamanSpectrum


def main():
    # args = vars(parse_args())

    dpath = Path("/home/m/Dropbox/zumi_2/ANALYZY/RAMAN/RAMANPY/2395v4/")
    outdp = dpath / "averge"
    outdp.mkdir(exist_ok=True)

    fpaths = sorted(f for f in dpath.glob("*.jdx"))
    fpdic = defaultdict(list)
    for f in fpaths:
        fpdic[f.stem.split("_")[0]].append(f)

    pprint(f"fpaths: {fpdic}")
    for name, fs in fpdic.items():
        print(name)
        ss = [RamanSpectrum(f) for f in fs]
        ss[0].y = sum(s.y for s in ss) / len(ss)
        fpout = outdp / f"{name}.jdx"
        ss[0].save(fpout)


def parse_args():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description=f"{__doc__} path:{Path(__file__).resolve()}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--dpath", type=str, help="directory with spectra")
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbose (CRITICAL, ERROR, WARN, INFO, DEBUG), eg. -vvvv",
    )
    args = p.parse_args()
    return args


if __name__ == "__main__":
    main()
