import shutil
from pathlib import Path
from typing import Dict, Optional
from zuy.common.logger import setup_logger
import re

logger = setup_logger(__name__)


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


if __name__ == "__main__":
    pass
