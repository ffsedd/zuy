from pathlib import Path
import re
import natsort
from zuy.common.logger import setup_logger
from zuy.spectrum.emsa import plot_emsa_files


logger = setup_logger(__name__)


def convert_spectra_txt_to_msa(base_path: Path) -> None:
    """
    Convert all spectra .txt files starting with digits in base_path (recursive)
    to .msa files by substituting ': EDS' with ': EDS_SEM'.
    Original .txt files are renamed to .bak after conversion.
    """

    txt_files = [f for f in base_path.rglob("*.txt") if f.stem and f.stem[0].isdigit()]
    txt_files.sort()
    if not txt_files:
        logger.warning(f"No .txt spectra found in {base_path}")
        return

    for txt_file in txt_files:
        msa_file = txt_file.with_suffix(".msa")

        if msa_file.exists():
            print(f"Skipping existing .msa file: {msa_file}")
            continue

        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {txt_file}: {e}")
            continue

        # Substitute ': EDS' → ': EDS_SEM'
        text_modified = re.sub(r":\s*EDS", ": EDS_SEM", text)

        try:
            msa_file.write_text(text_modified, encoding="utf-8")
            # Rename original .txt to .bak
            backup_file = txt_file.with_suffix(".bak")
            txt_file.rename(backup_file)
            print(f"Converted {txt_file} -> {msa_file}, backup: {backup_file}")
        except Exception as e:
            print(f"Failed to write {msa_file} or rename {txt_file}: {e}")


def plot_spectra(base_path: Path) -> None:

    # PLOT MSA
    fs = [f for f in base_path.rglob("*.msa")]

    dirs = list(set([f.parent.stem for f in fs]))
    logger.info("plot spectra in dirs: {dirs}")

    for d in dirs:
        fsi = natsort.natsorted([f for f in fs if f.parent.stem == d], key=str)
        logger.info(fsi)
        if len(fsi):
            plot_emsa_files(fsi, squared=True, figsize=(26, 10), overwrite=False)
