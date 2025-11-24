# zuy/spectrum/io.py
from pathlib import Path
import numpy as np
from .models import Spectrum


def parse_msa_file(fpath: Path) -> Spectrum:
    metadata: dict[str, str] = {}
    x: list[float] = []
    y: list[float] = []
    data_started = False

    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not data_started and line.upper().startswith("#SPECTRUM"):
                data_started = True
                continue
            if line.startswith("#"):
                if ":" in line:
                    key, val = line[1:].split(":", maxsplit=1)
                    metadata[key.strip().upper()] = val.strip()
                continue
            if data_started:
                xi, yi = line.split(",", maxsplit=1)
                x.append(float(xi))
                y.append(float(yi))

    return Spectrum(np.array(x), np.array(y), metadata)
