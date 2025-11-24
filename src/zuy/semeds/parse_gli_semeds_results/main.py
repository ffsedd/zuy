"""
Usage: uv run src/zuy/semeds/parse_gli_semeds_results/main.py /path/to/directory

"""

from pathlib import Path
from typing import Dict
import argparse
import natsort
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MultipleLocator

from zuy.common.logger import setup_logger
from zuy.semeds.parse_gli_semeds_results.merge_xlsx import merge_xlsx
from zuy.semeds.parse_gli_semeds_results.df_cleanup import clean_df
from zuy.semeds.parse_gli_semeds_results.split_samples import split_samples
from zuy.semeds.parse_gli_semeds_results.save_outputs import save_output
from zuy.semeds.parse_gli_semeds_results.copy_result import copy_results
from zuy.semeds.parse_gli_semeds_results.convert_spectra import (
    convert_spectra_txt_to_msa,
)
from zuy.common.zlib import find_zakazky_dir, zak_dict
from zuy.spectrum.emsa import EmsaSpectrum
from zuy.spectrum.squre_root_scale import register_sqrt_scale

register_sqrt_scale()

logger = setup_logger(__name__)

COPY_RESULT = True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and process SEM/EDS results from GLI."
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Path to the directory containing XLSX and spectra files",
        default=".",
    )
    # parser.add_argument(
    #     "--copy_result",
    #     type=bool,
    #     help="Path to the target directory in pytex folder",
    #     default=None,
    # )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dpath: Path = args.src.resolve()
    # trg_dpath: Path | None = args.trg.resolve() if args.trg is not None else None 

    # if trg_dpath is not None:
    # logger.info(f"Target directory: {trg_dpath}")
    zak_dir = find_zakazky_dir()
    zmap = zak_dict(zak_dir)

    
    # if 1:
    #     args = parse_args()
    #     src_dpath: Path = args.src
    #     trg_dpath: Path = args.trg
    # else:
    #     src_dpath = Path("/home/m/Dropbox/ZUMI/zakazky/ZADANI-SEM/sem-25-09")
    #     trg_dpath = Path("/home/m/Dropbox/ZUMI/zakazky/2536_Fogaš/pytex/sem/")


    
    outdir = src_dpath / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting processing in directory: {src_dpath}")

    # %% Merge XLSX files
    logger.info("Merge XLSX files into DataFrames by category...")
    merged: Dict[str, pd.DataFrame] = merge_xlsx(src_dpath)
    if "weight" not in merged:
        logger.warning("No 'weight' tables found. Exiting.")
        return

    # %% Save raw merged weight DataFrame
    df_weight = merged["weight"]
    weight_merged_path = outdir / "weight_merged.xlsx"
    df_weight.to_excel(weight_merged_path, index=False)
    logger.info(f"...saved: {weight_merged_path} (shape={df_weight.shape})")

    # %% Clean and normalize merged DataFrame
    logger.info("Clean and normalize merged DataFrame...")
    df_clean = clean_df(df_weight)
    weight_clean_path = outdir / "weight_clean.xlsx"
    df_clean.to_excel(weight_clean_path, index=False)
    logger.info(f"...saved: {weight_clean_path} (shape={df_clean.shape})")

    # %% Split cleaned data into samples and save outputs
    logger.info("Split cleaned data into samples...")
    samples: Dict[str, pd.DataFrame] = split_samples(df_clean)
    save_output(samples, outdir)
    logger.info(f"...saved outputs to {outdir}")

    # %% Convert spectra files
    logger.info("Convert spectra files (.txt → .msa)...")
    convert_spectra_txt_to_msa(src_dpath)

    # %% Plot spectra
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("linestyle", ["-", "--", ":", "-."])
            * plt.rcParams["axes.prop_cycle"]
        ),
    )

    fpaths = src_dpath.rglob("*.msa")
    src_dpaths = set([f.parent for f in fpaths])

    for src_dpath in src_dpaths:

        fpaths = natsort.natsorted([f for f in src_dpath.glob("*.msa")], key=str)
        logger.info(f"Plotting spectra in directory: {src_dpath}")

        plt.figure(figsize=(10, 4.9))

        for f in fpaths:
            s = EmsaSpectrum.from_msa(f)
            s.smooth(5)
            s.baseline_correction(y_offset=15)
            logger.info(f"Plotting: {f.name}")
            s.plot(mph_perc=0.1, mpd=2, thl_perc=0.03)

        ax = plt.gca()
        ax.set_title(src_dpath.name)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        ax.set_yscale("sqrt")
        ax.set_xlim(left=0, right=14)
        ax.set_ylim(bottom=0, top=None)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(src_dpath / f"{src_dpath.name}_spectra.pdf", dpi=300)

        # %% Copy result
        if COPY_RESULT:
            try:
                zak = int(src_dpath.name[:4])
                trg_dpath = zmap[zak] / "pytex/sem"
                trg_dpath.mkdir(parents=True, exist_ok=True)
                copy_results(src_dpath, trg_dpath)
            except KeyError:
                logger.warning(f"Zakazka {src_dpath.name[:4]} not found.")


if __name__ == "__main__":
    main()
