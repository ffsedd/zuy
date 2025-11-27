import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Set

import natsort  # type: ignore
import pandas as pd  # type: ignore
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
from zuy.semeds.models import Sample
from zuy.semeds.plot_element_correlations import plot_correlations_from_tsv
from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.plotting import plot_spectrum
from zuy.spectrum.processing import tidy_spectrum
from zuy.spectrum.squre_root_scale import register_sqrt_scale

register_sqrt_scale()

logger = setup_logger(__name__)

# Matplotlib style
propcy = cycler("linestyle", ["-", "--", ":", "-."]) * plt.rcParams["axes.prop_cycle"]
plt.rc("axes", prop_cycle=propcy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process SEM/EDS results from GLI.")
    parser.add_argument(
        "src", type=Path, help="Directory with XLSX and spectra files", default="."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--copy", action="store_true", help="Copy results to zakazka directories")
    return parser.parse_args()


# --- XLSX pipeline ---
def merge_and_clean_xlsx(root: Path, outdir: Path, overwrite: bool = False) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    # Merge
    merged_path = outdir / "weight_merged.xlsx"
    if merged_path.exists() and not overwrite:
        df = pd.read_excel(merged_path)
        logger.info(f"Loaded merged XLSX: {merged_path} ({df.shape})")
    else:
        merged = merge_xlsx(root)
        df = merged["weight"]
        df.to_excel(merged_path, index=False)
        logger.info(f"Merged XLSX saved: {merged_path} ({df.shape})")

    # Clean
    clean_path = outdir / "weight_clean.xlsx"
    if clean_path.exists() and not overwrite:
        df_clean = pd.read_excel(clean_path, index_col=(0, 1, 2))
        logger.info(f"Loaded cleaned XLSX: {clean_path} ({df_clean.shape})")
    else:
        df_clean = clean_df(df)
        df_clean.to_excel(clean_path)
        logger.info(f"Cleaned XLSX saved: {clean_path} ({df_clean.shape})")

    return df_clean


# --- Save sample outputs ---
def save_sample_outputs(
    samples: Dict[Sample, pd.DataFrame], outdir: Path, overwrite: bool = False
) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    for sample, df in samples.items():
        name = f"{sample.zakazka}v{sample.sample_no}"
        xls_path = outdir / f"{name}.xls"
        if not xls_path.exists() or overwrite:
            save_formatted_xlsx(xls_path, {"results": df}, widths=[(0, 50, 5)])
            df2 = df.droplevel(["Zakazka", "Sample"]).drop(
                columns=["source_file", "Site", "Project"]
            )
            (outdir / f"{name}.tex").write_text(
                re.sub(r" &", "\t&", df2.fillna("").to_latex(float_format="%.1f"))
            )
            df2.to_csv(outdir / f"{name}.tsv", sep="\t", index=True, float_format="%.1f")
            plot_correlations_from_tsv(outdir / f"{name}.tsv")
        else:
            logger.info(f"Skipped sample {name} (exists, no overwrite)")


# --- Spectra plotting ---
def plot_spectra_dir(dpath: Path, overwrite: bool = False, smooth_window: int = 5) -> None:
    """Load, tidy (smooth + baseline), and plot all .msa spectra in a directory."""
    msa_files = natsort.natsorted(dpath.glob("*.msa"), key=str)
    if not msa_files:
        return

    out_pdf = dpath / f"{dpath.name}_spectra.pdf"
    if out_pdf.exists() and not overwrite:
        logger.info(f"Skipped existing plot: {out_pdf.name}")
        return

    fig, ax = plt.subplots(figsize=(10, 4.9))
    for f in msa_files:
        s = parse_msa_file(f)
        s_tidy = tidy_spectrum(s, smooth_window=smooth_window)
        plot_spectrum(s_tidy, ax=ax, mph_perc=0.1, mpd=2, thl_perc=0.03)

    ax.set_title(dpath.name)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_yscale("sqrt")  # type: ignore
    ax.set_xlim(0, 14)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)
    plt.close(fig)
    logger.info(f"Saved spectra plot: {out_pdf}")


# --- Copy results ---
def copy_to_zakazky(fp: Path, zmap: Dict[int, Path], rename: Optional[str] = None) -> None:
    pattern = r"(\d{4})v\d+"
    if m := re.match(pattern, fp.stem) or re.match(pattern, fp.parent.stem):
        zak = int(m.group(1))
    else:
        logger.warning(f"Cannot guess zakazka from {fp}")
        return

    if zak not in zmap:
        logger.warning(f"Zakazka {zak} not found in mapping.")
        return

    trg_dir = zmap[zak] / "pytex/sem"
    trg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp, trg_dir / (rename or fp.name))
    logger.info(f"Copied {fp.name} -> {trg_dir}")


# --- Main pipeline ---
def main() -> None:
    args = parse_args()
    root = args.src.resolve()
    outdir = root / "processed"
    zak_dir = find_zakazky_dir()
    zmap = zak_dict(zak_dir)

    df_clean = merge_and_clean_xlsx(root, outdir, args.overwrite)
    samples = df_split(df_clean)
    logger.info(f"Split into {len(samples)} samples")

    save_sample_outputs(samples, outdir, args.overwrite)
    convert_gli_txt_spectra_to_msa(root)

    spectra_dirs: Set[Path] = {f.parent for f in root.rglob("*.msa")}
    for d in spectra_dirs:
        plot_spectra_dir(d, args.overwrite)

    if args.copy:
        logger.info("Copying results to zakazka directories...")
        for d in [outdir] + list(spectra_dirs):
            for fp in d.iterdir():
                if fp.suffix.lower() in {".xls", ".tex", ".tsv", ".pdf", ".png"}:
                    copy_to_zakazky(fp, zmap)
                elif fp.suffix.lower() == ".msa":
                    copy_to_zakazky(fp, zmap, rename=f"{d.name}_{fp.name}")
    else:
        logger.info("Copy flag not set, skipping copy.")


if __name__ == "__main__":
    main()
