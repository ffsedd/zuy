"""Reader for Buck Scientific Model 500 infrared SCN text exports.

The eight-line header is followed by XY pairs separated by a comma and spaces.
Individual numbers can use decimal commas, e.g. ``3999,2, 21,59625``.
The header's last-X value can be an exclusive limit; measured X values are kept.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ScnData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    instrument: str
    title: str
    acquired: str
    date: datetime | None


def read_scn(filepath: str | Path) -> ScnData:
    """Read SCN data without processing or changing axis order.

    Based on the legacy zumpy Buck SCN reader. Validate header numbers and point
    count, and include the source line number in malformed-record errors.
    """
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 8:
        raise ValueError(f"{path}: SCN requires an eight-line header")
    try:
        count = int(lines[3].strip())
        step, first, last = (float(line.strip().replace(",", ".")) for line in lines[4:7])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid SCN numeric header") from exc
    if count < 1 or not np.isfinite([step, first, last]).all():
        raise ValueError(f"{path}: invalid SCN point count or axis bounds")

    pairs: list[tuple[float, float]] = []
    for number, line in enumerate(lines[8:], 9):
        if not line.strip():
            continue
        try:
            left, right = re.split(r",\s+", line.strip())
            pairs.append((float(left.replace(",", ".")), float(right.replace(",", "."))))
        except ValueError as exc:
            raise ValueError(f"{path}: invalid SCN record on line {number}: {line!r}") from exc
    if len(pairs) != count:
        raise ValueError(f"{path}: SCN declares {count} points, found {len(pairs)}")

    acquired = lines[2].strip()
    date = None
    for pattern in ("%m/%d/%Y", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d"):
        try:
            date = datetime.strptime(acquired, pattern)
            break
        except ValueError:
            continue
    data = np.asarray(pairs, dtype=float)
    return ScnData(data[:, 0], data[:, 1], lines[0].strip(), lines[1].strip(), acquired, date)
