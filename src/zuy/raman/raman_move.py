#!/usr/bin/env python3
import logging
from pathlib import Path
import shutil

# ~ from zumpy.qq.ntool.smooth import smooth_golay


from zumpy import zlib

# ~ ZDIR = zlib.find_zakazky_dir()      # all zakazky dir
ZDIC = zlib.get_zak_dict()  # dict {zakazka_no : Path}

ROOT = Path.home().joinpath("Dropbox/zumi_2/ANALYZY/RAMAN/RAMANPY").resolve()


def mkdir(dpath, exit_ok=True):
    from os import makedirs

    makedirs(dpath, exist_ok=exit_ok)


def main():
    logging.basicConfig(
        format="!%(levelno)s [%(module)10s%(lineno)4d]	%(message)s",
        # ~ filename=Path(__file__).with_suffix(".log"), filemode='w'
        level=20,
    )

    root = Path(ROOT)
    done_dir = root / "_DONE"
    done_dir.mkdir(exist_ok=True)

    fs = root.glob("*.pdf")  # find pdf

    for pdf_path in fs:
        try:
            zakno = int(pdf_path.stem[:4])
        except ValueError:
            logging.error(f"pdf nezačíná číslem zakázky: {pdf_path}")
            continue

        trg_root = ZDIC[zakno] / "pytex"

        trg = trg_root / "raman"

        trg2 = trg / "spektra"
        trg2.mkdir(exist_ok=True, parents=True)
        shutil.copy2(pdf_path, trg2 / pdf_path.name.replace("%", "_"))
        pdf_path.rename(done_dir / pdf_path.name)

        print(pdf_path, trg)

        src2 = pdf_path.parent / pdf_path.stem

        trg2 = trg / "jdx"
        trg2.mkdir(exist_ok=True, parents=True)

        for jdx_path in src2.glob("*.jdx"):
            shutil.copy2(jdx_path, trg2 / jdx_path.name.replace("%", "_"))

        trg2 = trg / "lokalizace"
        trg2.mkdir(exist_ok=True, parents=True)

        for jpg_path in src2.glob("*.jpg"):
            shutil.copy2(jpg_path, trg2 / jpg_path.name.replace("%", "_"))

        done2 = done_dir / src2.name
        if done2.exists():
            shutil.rmtree(done2)

        src2.rename(done_dir / src2.name)


if __name__ == "__main__":
    main()
