"""
Usage: uv run src/zuy/semeds/main.py /path/to/directory

"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict

import natsort  # type: ignore
import pandas as pd  # type: ignore  # type: ignore
from cycler import cycler  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.ticker import MultipleLocator  # type: ignore

from zuy.common.dftools import save_formatted_xlsx
from zuy.common.logger import setup_logger
from zuy.common.zlib import find_zakazky_dir, zak_dict
from zuy.semeds.clean_df import clean_df
from zuy.semeds.convert_spectra import convert_gli_txt_spectra_to_msa
from zuy.semeds.df_split import df_split
from zuy.semeds.merge_xlsx import merge_xlsx
from zuy.semeds.models import ZakazkaSample
from zuy.semeds.plot_element_correlations import plot_correlations_from_tsv
from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.plotting import plot_spectrum
from zuy.spectrum.processing import baseline_correction, smooth
from zuy.spectrum.squre_root_scale import register_sqrt_scale

register_sqrt_scale()

logger = setup_logger(__name__)


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
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy results to zakazka directories"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dpath: Path = args.src.resolve()
    # trg_dpath: Path | None = args.trg.resolve() if args.trg is not None else None

    # if trg_dpath is not None:
    # logger.info(f"Target directory: {trg_dpath}")
    zak_dir = find_zakazky_dir()
    zmap = zak_dict(zak_dir)

    outdir = src_dpath / "processed"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting processing in directory: {src_dpath}")

    # %% Rename files   # TODO
    # 1111_1 -> 1111v1

    # %% Merge XLSX files  --------------------------------------------------
    weight_merged_path = outdir / "weight_merged.xlsx"
    if weight_merged_path.exists() and not args.overwrite:
        weight_df = pd.read_excel(weight_merged_path)  # Load if exists
        logger.info(f"DataFrame loaded: {weight_merged_path} (shape={weight_df.shape})")
    else:
        logger.info("Merge XLSX files into DataFrames by category...")
        merged_df: Dict[str, pd.DataFrame] = merge_xlsx(src_dpath)
        if "weight" not in merged_df:
            raise ValueError("No 'weight' tables found. Exiting.")
        # Save raw merged weight DataFrame
        weight_df = merged_df["weight"]
        weight_df.to_excel(weight_merged_path, index=False)
        logger.info(f"...saved: {weight_merged_path} (shape={weight_df.shape})")

    # %% Clean and normalize merged DataFrame --------------------------------------------------
    logger.info("Clean and normalize merged DataFrame...")

    weight_clean_path = outdir / "weight_clean.xlsx"
    if weight_clean_path.exists() and not args.overwrite:
        df_clean = pd.read_excel(weight_clean_path, index_col=(0, 1, 2), header=0)
        logger.info(f"DataFrame loaded: {weight_clean_path} (shape={df_clean.shape})")
    else:
        df_clean = clean_df(weight_df)
        df_clean.to_excel(weight_clean_path)
        logger.info(f"...saved: {weight_clean_path} (shape={df_clean.shape})")

    # %% Split cleaned DataFrame into samples  --------------------------------------------------
    samples: Dict[ZakazkaSample, pd.DataFrame] = df_split(df_clean)
    logger.info(f"...split into {len(samples)} samples: {samples.keys()}")

    # %% Save outputs  --------------------------------------------------
    def save_output(
        samples: dict[ZakazkaSample, pd.DataFrame],
        outdir: Path,
        overwrite: bool = False,
    ) -> None:
        """
        Save each sample's DataFrame to LaTeX, TSV, XLS, and PNG heatmap.
        """
        outdir.mkdir(exist_ok=True, parents=True)

        for (zakazka, sample), df in samples.items():
            zak_sample = f"{zakazka}v{sample}"
            # print(zak_sample)
            logger.info(f"Saving outputs for sample: {zak_sample}")

            # 1. Save .xls
            fp_xls = outdir / f"{zak_sample}.xls"
            if fp_xls.exists() and not overwrite:
                logger.info(
                    f"File {fp_xls} exists and overwrite is False. Skipping XLS, TEX, TSV save."
                )
            else:
                save_formatted_xlsx(fp_xls, {"results": df}, widths=[(0, 50, 5)])

                df2 = df.droplevel(["Zakazka", "Sample"]).drop(
                    columns=["source_file", "Site", "Project"]
                )

                # drop unneeded index levels
                # 2. Save .tex
                fp_tex = outdir / f"{zak_sample}.tex"

                tex = df2.to_latex(index=True, na_rep="", float_format="%.1f")
                tex = re.sub(r" &", "\t&", tex)  # tab-align
                fp_tex.write_text(tex)

                # 3. Save .tsv
                fp_tsv = outdir / f"{zak_sample}.tsv"
                df2.to_csv(fp_tsv, sep="\t", index=True, na_rep="", float_format="%.1f")
                plot_correlations_from_tsv(fp_tsv)

        logger.info(f"Saved outputs to {outdir}")

    save_output(samples, outdir, overwrite=args.overwrite)
    logger.info(f"...saved outputs to {outdir}")

    # %% Convert spectra files  --------------------------------------------------
    logger.info("Convert spectra files (.txt → .msa)...")
    convert_gli_txt_spectra_to_msa(src_dpath)

    # %% Plot spectra --------------------------------------------------

    def plot_dir(src_dpath: Path) -> None:
        fpaths = natsort.natsorted([f for f in src_dpath.glob("*.msa")], key=str)  # type: ignore
        logger.info(f"Plotting spectra in directory: {src_dpath}")
        spectra_fp = src_dpath / f"{src_dpath.name}_spectra.pdf"

        if spectra_fp.exists() and not args.overwrite:
            logger.info(f"Skipping existing file: {spectra_fp}")
            return
        else:
            plt.figure(figsize=(10, 4.9))

            for f in fpaths:
                s = parse_msa_file(f)
                s.y = smooth(s.y, 5)
                s.y = baseline_correction(s.y, y_offset=15)
                logger.info(f"Plotting: {f.name}")
                plot_spectrum(s, mph_perc=0.1, mpd=2, thl_perc=0.03)

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
            plt.savefig(spectra_fp, dpi=300)

    plt.rc(
        "axes",
        prop_cycle=(
            cycler("linestyle", ["-", "--", ":", "-."])
            * plt.rcParams["axes.prop_cycle"]
        ),
    )

    fpaths = src_dpath.rglob("*.msa")
    src_dpaths = set(f.parent for f in fpaths)

    for src_dpath in src_dpaths:
        plot_dir(src_dpath)
        logger.info(f"...saved spectra plot for {src_dpath.name}")

    # %% Copy result --------------------------------------------------
    if args.copy:
        logger.info("Copying results to zakazka directories...")
        for src_dpath in src_dpaths:

            zak = int(src_dpath.name[:4])
            if zak not in zmap:
                logger.warning(f"Zakazka {src_dpath.name[:4]} not found.")
                continue

            trg_dpath = zmap[zak] / "pytex/sem"
            for fp in src_dpath.iterdir():
                if fp.suffix in (
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tex",
                    ".xlsx",
                    ".msa",
                ):
                    shutil.copy(fp, trg_dpath)
                    logger.info(f"copied: {fp.name}")
                # else:
                #     logger.warning(f"Unknown file type: {fp}")


if __name__ == "__main__":
    main()
