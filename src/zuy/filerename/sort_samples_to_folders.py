#!/usr/bin/env python3
"""Sort files into subdirectories based on sample prefixes."""

import argparse
import logging
import re
import shutil
from pathlib import Path
from collections.abc import Iterator

SAMPLE_RE = re.compile(r"\d{4}[v_-]\d{1,2}")

logger = logging.getLogger(__name__)


def extract_prefix(name: str) -> str | None:
    """Extract sample prefix from a filename."""
    match = SAMPLE_RE.search(name)
    return match.group() if match else None


def plan_moves(root: Path) -> Iterator[tuple[Path, Path]]:
    """Generate source and destination paths without changing files."""
    for src in sorted(root.iterdir()):
        if not src.is_file():
            continue

        prefix = extract_prefix(src.name)
        if prefix:
            yield src, root / prefix / src.name


def sort_files(root: Path, dry_run: bool = False) -> tuple[int, int]:
    """Move matching files into sample subdirectories.

    Returns:
        Number of moved files and skipped collisions.
    """
    moved = 0
    skipped = 0

    for src, dst in plan_moves(root):
        if dst.exists():
            logger.warning("Destination exists, skipping: %s", dst)
            skipped += 1
            continue

        logger.info("%s -> %s", src.name, dst.parent.name)

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)

        moved += 1

    return moved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Directory to sort (default: current directory)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview file moves without changing anything",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")

    moved, skipped = sort_files(root, dry_run=args.dry_run)

    action = "Planned" if args.dry_run else "Moved"
    logger.info("%s %d files, skipped %d collisions", action, moved, skipped)


if __name__ == "__main__":
    main()
