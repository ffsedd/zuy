#!/usr/bin/env python3
from pprint import pprint
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

from zuy.raman.raman_plotter import RamanPlotter


ROOT = Path.home().joinpath("Dropbox/zumi_2/ANALYZY/RAMAN/RAMANPY").resolve()


def main():
    args = vars(parse_args())

    root = Path(args.pop("path")).expanduser().resolve()
    print(f"{__file__} started in {root}")
    logging.basicConfig(
        format="!%(levelno)s [%(module)10s%(lineno)4d]	%(message)s",
        # ~ filename=Path(__file__).with_suffix(".log"), filemode='w'
        level=20,
    )

    dirs = sorted(d for d in root.glob("*") if d.is_dir() and not d.stem.startswith("_"))

    pprint(f"process dirs: {dirs} \n args:{args}")

    if not dirs:
        return

    pool = Pool(processes=min(len(dirs), cpu_count()))
    results = []

    for dpath in dirs:
        print(dpath)
        r = pool.apply_async(RamanPlotter, (dpath,), args)
        results.append(r)
    pool.close()
    pool.join()  # Wait for all child processes to close.

    for r in results:
        r.get()  # print errors if any


def parse_args():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description=f"{__doc__} path:{Path(__file__).resolve()}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--path", default=str(ROOT), help="Parent directory of spectrum folders")
    p.add_argument("-x", "--maxx", default=2500, type=int, help="xlim max")
    p.add_argument("-y", "--maxy", default=10000, type=int, help="ylim max")
    p.add_argument("-z", "--maxz", default=10, type=int, help="ylim2 max")
    p.add_argument("-s", "--smooth", default=11, type=int, help="smooth window")
    p.add_argument("-j", "--jpegs", action="store_true", default=False)
    p.add_argument("-f", "--flag", action="store_true", default=False)
    p.add_argument("-c", "--copyresults", action="store_true", default=False)
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


if __name__ == "__main__":
    main()
