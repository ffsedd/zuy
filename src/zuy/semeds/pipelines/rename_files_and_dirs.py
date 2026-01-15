import re
from pathlib import Path
from zuy.common.logger import setup_logger

logger = setup_logger(__name__)

_PATTERN = re.compile(r"(\d{4})_(\d{1,2})")


def rename_files_and_dirs(root: Path) -> None:
    """
    Recursively rename files and directories matching (\d{4})_(\d{1,2}) → \1v\2.
    """
    # First rename directories (bottom-up)
    for d in sorted(root.rglob("*"), key=lambda p: -p.parts.__len__()):
        if d.is_dir() and (_m := _PATTERN.fullmatch(d.name)):
            new_name = f"{_m.group(1)}v{_m.group(2)}"
            new_path = d.parent / new_name
            if new_path.exists():
                logger.warning(f"Cannot rename {d} → {new_path}, already exists")
            else:
                d.rename(new_path)
                logger.info(f"Renamed dir: {d} → {new_path}")

    # Then rename files
    for f in root.rglob("*"):
        if f.is_file() and (_m := _PATTERN.fullmatch(f.stem)):
            new_name = f"{_m.group(1)}v{_m.group(2)}{f.suffix}"
            new_path = f.parent / new_name
            if new_path.exists():
                logger.warning(f"Cannot rename {f} → {new_path}, already exists")
            else:
                f.rename(new_path)
                logger.info(f"Renamed file: {f} → {new_path}")
