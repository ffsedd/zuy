from pathlib import Path
import re
from zuy.common.logger import setup_logger

logger = setup_logger(__name__)


def convert_gli_txt_spectra_to_msa(base_path: Path) -> None:
    """
    Convert all spectra .txt files starting with digits in base_path (recursive)
    to .msa files by substituting ': EDS' with ': EDS_SEM'.
    Original .txt files are renamed to .bak after conversion.
    """

    fpaths = [f for f in base_path.rglob("*.txt") if f.stem and f.stem[0].isdigit()]
    fpaths.sort()
    if not fpaths:
        logger.warning(f"No .txt spectra found in {base_path}")
        return

    for in_fpath in fpaths:
        out_fpath = in_fpath.with_suffix(".msa")

        if out_fpath.exists():
            print(f"Skipping existing .msa file: {out_fpath}")
            continue

        try:
            text = in_fpath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {in_fpath}: {e}")
            continue

        # Substitute ': EDS' → ': EDS_SEM'
        text_modified = re.sub(r":\s*EDS", ": EDS_SEM", text)

        try:
            out_fpath.write_text(text_modified, encoding="utf-8")
            # Rename original .txt to .bak
            backup_file = in_fpath.with_suffix(".bak")
            in_fpath.rename(backup_file)
            print(f"Converted {in_fpath} -> {out_fpath}, backup: {backup_file}")
        except Exception as e:
            print(f"Failed to write {out_fpath} or rename {in_fpath}: {e}")
