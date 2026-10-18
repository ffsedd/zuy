#!/usr/bin/env python3
"""JCAMP-DX spectra: load, inspect, process, plot and save.

Dependencies: numpy. Optional: matplotlib (plotting), scipy (peak finding,
Savitzky-Golay smoothing and baseline correction). SPA and SCN readers are internal.

``Jdx`` retains the historical mutable processing API. Binary arithmetic
returns independent spectra; operations such as ``clear`` and ``subtract``
modify ``self`` for backwards compatibility.
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, time
from numbers import Real
from pathlib import Path
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.typing as npt

from zuy.spectrum import processing
from zuy.spectrum.processing import interpolate_linear as interpolate_linear, signif as signif

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

Array = npt.NDArray[np.float64]
LOG = logging.getLogger(__name__)

SQZ_DIGITS = {
    "@": 0,
    **{chr(65 + i): i + 1 for i in range(9)},
    **{chr(97 + i): -(i + 1) for i in range(9)},
}
DIF_DIGITS = {
    "%": 0,
    **{chr(74 + i): i + 1 for i in range(9)},
    **{chr(106 + i): -(i + 1) for i in range(9)},
}
DUP_DIGITS = {**{chr(83 + i): i + 1 for i in range(8)}, "s": 9}

# Historical spellings for code that imports these names.
SQZ_digits = SQZ_DIGITS
DIF_digits = DIF_DIGITS
DUP_digits = DUP_DIGITS

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_X_PREFIX = re.compile(rf"^\s*({_FLOAT})\s*,?\s*(.*)$")
_TOKEN = re.compile(
    rf"(?P<number>{_FLOAT})|(?P<sqz>[@A-Ia-i][0-9.]*)|"
    r"(?P<dif>[%J-Rj-r][0-9.]*)|(?P<dup>[S-Zs][0-9]*)|(?P<nan>\?)"
)
_TEXT_SPLIT = re.compile(r"[;,\s]+")
_X_WAVENUMBER = {"1/cm", "cm-1", "cm^-1", "cm ^ -1", "cm⁻¹"}
_X_MICROMETER = {"micrometers", "micrometer", "um", "µm", "wavelength (um)"}
_X_NANOMETER = {"nanometers", "nanometer", "nm", "wavelength (nm)"}
_Y_TRANSMITTANCE = {"transmittance", "transmitance", "reflectance"}


def is_float(value: str) -> bool:
    """Return whether a string can be converted to float."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _code_number(token: str, mapping: dict[str, int]) -> float:
    """Replace a JCAMP pseudo-digit with the leading signed digit."""
    first = mapping[token[0]]
    tail = token[1:]
    sign = -1 if first < 0 else 1
    return float(sign * float(f"{abs(first)}{tail}"))


def decode_asdf(data: str) -> tuple[list[float], bool]:
    """Decode one line's ordinate values; return (values, ends_with_DIF).

    Supports uncompressed values and SQZ, DIF and DUP pseudo-digits. DUP
    repeats the previous *code/value*, not just the previous character.
    DIF differences are relative to the previous Y value in this data line.
    The X checkpoint must be removed before calling this function.
    """
    values: list[float] = []
    previous: tuple[float, bool] | None = None
    last_is_dif = False
    index = 0

    def append(value: float, is_dif: bool) -> None:
        if is_dif:
            if not values:
                raise ValueError("DIF must follow an absolute Y value")
            value += values[-1]
        values.append(value)

    while index < len(data):
        if data[index] in " \t,;\r\n":
            index += 1
            continue
        token = _TOKEN.match(data, index)
        if token is None:
            raise ValueError(f"Unexpected JCAMP data near {data[index : index + 20]!r}")
        index = token.end()
        word = token.group()
        mode = token.lastgroup

        if mode == "dup":
            if previous is None:
                raise ValueError("DUP must follow an ordinate")
            count = int(f"{DUP_DIGITS[word[0]]}{word[1:]}")
            for _ in range(count - 1):
                append(*previous)
            last_is_dif = previous[1]
            continue

        if mode == "number":
            current = (float(word), False)
        elif mode == "sqz":
            current = (_code_number(word, SQZ_DIGITS), False)
        elif mode == "dif":
            current = (_code_number(word, DIF_DIGITS), True)
        else:
            current = (float("nan"), False)

        append(*current)
        previous = current
        last_is_dif = current[1]

    return values, last_is_dif


def _numeric_pairs(line: str) -> tuple[list[float], list[float]]:
    """Read XY pairs, including JCAMP's delimiter-free negative numbers.

    For example, ``4000-0.002 3999-0.003`` contains two pairs. Scientific
    notation is also accepted without splitting the exponent sign.
    """
    tokens = list(re.finditer(rf"{_FLOAT}|\?", line))
    pos = 0
    numbers: list[float] = []
    for match in tokens:
        if line[pos : match.start()].strip(" \t,;"):
            raise ValueError(f"Invalid XY data near {line[pos : match.start()]!r}")
        token = match.group()
        numbers.append(float("nan") if token == "?" else float(token))
        pos = match.end()
    if line[pos:].strip(" \t,;"):
        raise ValueError(f"Invalid XY trailing data {line[pos:]!r}")
    if not numbers or len(numbers) % 2:
        raise ValueError(f"Missing or odd number of XY values in {line!r}")
    return numbers[::2], numbers[1::2]


def _header_key(name: str) -> str:
    return re.sub(r"\W+", "", name).lower()


def _header_value(value: str) -> str | float | int:
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if is_float(value):
        return float(value)
    return value


def _read_jcamp(text: str) -> tuple[dict[str, str], Array, Array]:
    """Parse a single-block JCAMP spectrum without modifying an object."""
    headers: dict[str, str] = {}
    xpoints: list[float] = []
    ypoints: list[float] = []
    row_starts: list[float] = []
    mode = ""
    previous_was_dif = False
    found_data = False

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("$$", 1)[0].strip()
        if not line:
            continue
        if line.startswith("##"):
            if "=" not in line:
                raise ValueError(f"Invalid JCAMP header on line {line_number}")
            key, value = line[2:].split("=", 1)
            key = _header_key(key)
            value = value.strip()
            if key == "end":
                mode = ""
                continue
            if key in {"xydata", "xypoints", "peaktable"}:
                if found_data:
                    raise NotImplementedError("Multiple JCAMP data blocks are not supported")
                found_data = True
                mode = value.replace(" ", "").upper()
                if mode not in {"(X++(Y..Y))", "(XY..XY)"}:
                    raise NotImplementedError(f"Unsupported JCAMP data layout {value!r}")
            elif found_data and mode:
                mode = ""
            headers[key] = value
            continue
        if not mode:
            continue

        try:
            if mode == "(XY..XY)":
                xs, ys = _numeric_pairs(line)
                xpoints.extend(xs)
                ypoints.extend(ys)
            else:
                match = _X_PREFIX.match(line)
                if match is None or not match[2].strip():
                    raise ValueError("Missing X checkpoint or Y data")
                row_starts.append(float(match[1]))
                values, ends_with_dif = decode_asdf(match[2])
                if previous_was_dif and ypoints and values:
                    if np.isclose(ypoints[-1], values[0], equal_nan=True):
                        # DIF checkpoint at the end of the previous data line.
                        ypoints.pop()
                    else:
                        LOG.warning("JCAMP DIF checkpoint mismatch on line %d", line_number)
                ypoints.extend(values)
                previous_was_dif = ends_with_dif
        except ValueError as exc:
            raise ValueError(f"Invalid JCAMP line {line_number}: {exc}") from exc

    if not found_data or not ypoints:
        raise ValueError("No supported JCAMP spectrum data found")

    if row_starts:
        if "lastx" not in headers:
            raise ValueError("Compressed XYDATA requires ##LASTX")
        npoints = int(headers.get("npoints", len(ypoints)))
        if npoints != len(ypoints):
            raise ValueError(f"JCAMP NPOINTS={npoints}, decoded {len(ypoints)} Y points")
        firstx = float(headers.get("firstx", row_starts[0]))
        xpoints = np.linspace(firstx, float(headers["lastx"]), npoints).tolist()
    elif "npoints" in headers and len(ypoints) != int(headers["npoints"]):
        raise ValueError("JCAMP NPOINTS differs from number of XY pairs")

    x = np.asarray(xpoints, dtype=float) * float(headers.get("xfactor", 1))
    y = np.asarray(ypoints, dtype=float) * float(headers.get("yfactor", 1))
    return headers, x, y


@dataclass
class Jdx:
    """Spectrum with JCAMP header metadata and mutable processing methods.

    Loading .jdx/.spa automatically calls ``tidy()`` by default, preserving
    historical behavior. Pass ``auto_tidy=False`` for raw, unmodified data.
    """

    filepath: str | Path = ""
    title: str | None = None
    txt_sep: str = ";"
    auto_tidy: bool = True
    x: Array = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    y: Array = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    xunits: str = ""
    yunits: str = ""
    datatype: str = ""
    dataprocessing: str = ""
    samplingprocedure: str = ""
    spectrometerdatasystem: str = ""
    origin: str = ""
    owner: str = ""
    jcampdx: str = "5.01"
    xydata: str = ""
    end: str = ""
    xfactor: float = 1.0
    yfactor: float = 1.0
    minx: float = 0.0
    maxx: float = 0.0
    miny: float = 0.0
    maxy: float = 0.0
    date: datetime | None = None
    time: time | None = None
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    ax: Axes | None = field(default=None, init=False, repr=False)
    fig: Figure | SubFigure | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).copy()
        self.y = np.asarray(self.y, dtype=float).copy()
        if self.filepath:
            self._jdx_from_file(self.filepath)
        else:
            self.filepath = Path("new.jdx")
            if self.date is None:
                self.date = datetime.now()

    @property
    def filename(self) -> str:
        return Path(self.filepath).name

    @property
    def filestem(self) -> str:
        return Path(self.filepath).stem

    @property
    def fileext(self) -> str:
        return Path(self.filepath).suffix.lower()

    def _require_data(self, minimum: int = 1) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) != len(self.y):
            raise ValueError("X and Y must be one-dimensional arrays of equal length")
        if len(self.x) < minimum:
            raise ValueError(f"Spectrum requires at least {minimum} points")

    def get_firstx(self) -> float:
        self._require_data()
        return float(self.x[0])

    def get_lastx(self) -> float:
        self._require_data()
        return float(self.x[-1])

    def get_firsty(self) -> float:
        self._require_data()
        return float(self.y[0])

    def get_maxx(self) -> float:
        self._require_data()
        return float(np.nanmax(self.x))

    def get_minx(self) -> float:
        self._require_data()
        return float(np.nanmin(self.x))

    def get_maxy(self) -> float:
        self._require_data()
        return float(np.nanmax(self.y))

    def get_miny(self) -> float:
        self._require_data()
        return float(np.nanmin(self.y))

    def get_npoints(self) -> int:
        return len(self.x)

    def get_deltax(self) -> float:
        self._require_data(2)
        return round(abs(float(self.x[-1] - self.x[0])) / (len(self.x) - 1), 3)

    def copy(self) -> Jdx:
        """Copy metadata and both arrays without reloading the source file."""
        result = copy.copy(self)
        result.x = self.x.copy()
        result.y = self.y.copy()
        result.headers = self.headers.copy()
        result.ax = None
        result.fig = None
        return result

    def _check_compatible(self, other: Jdx) -> None:
        if not isinstance(other, Jdx):
            raise TypeError("Expected another Jdx spectrum")
        self._require_data()
        other._require_data()
        if self.x.shape != other.x.shape or not np.allclose(self.x, other.x):
            raise ValueError("Spectra must have matching X coordinates")
        same_xunits = self.xunits.lower() == other.xunits.lower()
        same_yunits = self.yunits.lower() == other.yunits.lower()
        if not (same_xunits and same_yunits):
            raise ValueError("Spectra must have matching X and Y units")

    def __add__(self, other: Jdx) -> Jdx | NotImplementedType:
        if not isinstance(other, Jdx):
            return NotImplemented
        self._check_compatible(other)
        result = self.copy()
        result.y = self.y + other.y
        return result

    def __sub__(self, other: Jdx) -> Jdx | NotImplementedType:
        if not isinstance(other, Jdx):
            return NotImplemented
        self._check_compatible(other)
        result = self.copy()
        result.y = self.y - other.y
        return result

    def __mul__(self, value: float) -> Jdx | NotImplementedType:
        if not isinstance(value, Real):
            return NotImplemented
        result = self.copy()
        result.y *= float(value)
        return result

    __rmul__ = __mul__

    def __truediv__(self, value: float) -> Jdx | NotImplementedType:
        if not isinstance(value, Real):
            return NotImplemented
        if value == 0:
            raise ZeroDivisionError("Cannot divide spectrum by zero")
        result = self.copy()
        result.y /= float(value)
        return result

    @staticmethod
    def _is_float(value: str | Sequence[str]) -> bool | list[bool]:
        if isinstance(value, str):
            return is_float(value)
        return [is_float(part) for part in value]

    @staticmethod
    def _jcamp_parse(line: str) -> list[float]:
        """Compatibility method for decoding individual JCAMP data lines."""
        values, _ = decode_asdf(line)
        return values

    def _jdx_from_file(self, filepath: str | Path) -> None:
        path = Path(filepath).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.filepath = path
        match path.suffix.lower():
            case ".jdx" | ".dx":
                self.loadjdx(path)
                if self.auto_tidy:
                    self.tidy()
            case ".spa":
                self.from_spa(path)
                if self.auto_tidy:
                    self.tidy()
            case ".scn":
                self.from_scn(path)
            case _:
                self.loadtxt(path)

    def loadjdx(self, filepath: str | Path) -> None:
        """Load JCAMP data as-is (without automatic cleanup)."""
        path = Path(filepath)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        headers, self.x, self.y = _read_jcamp(text)
        self.filepath = path.resolve()
        self.headers = headers
        for key, value in headers.items():
            if key in {"date", "longdate", "time", "xydata", "xypoints", "peaktable"}:
                continue
            if key in {"xfactor", "yfactor", "minx", "maxx", "miny", "maxy"}:
                setattr(self, key, float(value))
            elif key in self.__dataclass_fields__ and key not in {"x", "y", "filepath"}:
                setattr(self, key, value)
            elif key.isidentifier() and not hasattr(self, key):
                # Preserve access to extra, vendor-specific header attributes.
                setattr(self, key, _header_value(value))
        self.xydata = headers.get("xydata", headers.get("xypoints", ""))
        dates = (("longdate", "%Y/%m/%d"), ("date", "%y/%m/%d"))
        parsed_date: datetime | None = None
        for key, pattern in dates:
            if key in headers and headers[key]:
                try:
                    parsed_date = datetime.strptime(headers[key], pattern)
                    break
                except ValueError:
                    LOG.warning("Invalid JCAMP %s: %r", key, headers[key])
        if "time" in headers and headers["time"]:
            try:
                self.time = datetime.strptime(headers["time"], "%H:%M:%S").time()
            except ValueError:
                LOG.warning("Invalid JCAMP time: %r", headers["time"])
        if parsed_date is not None:
            self.date = datetime.combine(parsed_date, self.time or time())
        LOG.debug("Loaded %s: %d points", path, len(self.x))

    def loadtxt(self, filepath: str | Path) -> None:
        """Load two-column numeric text, skipping header and comment rows."""
        path = Path(filepath)
        xs: list[float] = []
        ys: list[float] = []
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith(("#", "//")):
                continue
            fields = [field for field in _TEXT_SPLIT.split(line.strip()) if field]
            if len(fields) < 2:
                LOG.debug("Ignoring short row %d in %s", number, path)
                continue
            try:
                xi, yi = float(fields[0]), float(fields[1])
            except ValueError:
                LOG.debug("Ignoring nonnumeric row %d in %s", number, path)
                continue
            xs.append(xi)
            ys.append(yi)
        if not xs:
            raise ValueError(f"No numeric XY pairs in {path}")
        self.x, self.y = np.asarray(xs), np.asarray(ys)
        self.filepath = path.resolve()

    def tidy(self, clear_co2: bool = True) -> Jdx:
        """Normalize IR units and optionally interpolate over the CO2 band.

        Preserves historical behavior for ordinary absorbance/transmittance
        units. Unknown X units are left unchanged, rather than mislabelled.
        """
        self._require_data()
        units = self.xunits.strip().lower()
        if units in _X_WAVENUMBER:
            self.xunits = "1/CM"
        elif units in _X_MICROMETER | _X_NANOMETER:
            scale = 10_000.0 if units in _X_MICROMETER else 10_000_000.0
            with np.errstate(divide="ignore", invalid="ignore"):
                self.x = np.divide(scale, self.x)
            self.xunits = "1/CM"
        else:
            LOG.debug("Not converting unknown X unit %r", self.xunits)

        yunits = self.yunits.strip().lower()
        is_ir = self.datatype.strip().lower() in {"infrared spectrum", "ftir spectrum"}
        absorbance_like = yunits == "absorbance" or yunits in _Y_TRANSMITTANCE
        absorbance_like |= yunits in {"%t", "percent transmittance", "% transmittance"}
        if yunits == "absorbance":
            self.yunits = "ABSORBANCE"
        elif yunits in _Y_TRANSMITTANCE:
            self.to_absorbance()
        elif yunits in {"%t", "percent transmittance", "% transmittance"}:
            self.y = self.y / 100.0
            self.yunits = "TRANSMITTANCE"
            self.to_absorbance()
        if clear_co2 and self.xunits.lower() in _X_WAVENUMBER and (is_ir or absorbance_like):
            self.clear()
        self.x = signif(self.x, 6)
        self.y = signif(self.y, 8)
        return self

    def subtract(self, bg: Jdx) -> Jdx:
        """Subtract absorbance spectra, or ratio matching intensity spectra.

        Mutates self for compatibility with the original implementation.
        """
        self._check_compatible(bg)
        if self.yunits.lower() == "absorbance":
            self.y = self.y - bg.y
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.y = self.y / bg.y
            self.yunits = "TRANSMITTANCE"
        return self.tidy()

    def clear(self, bounds: tuple[float, float] = (2250, 2450), interp: bool = True) -> Jdx:
        """Replace values in a given X region, independent of axis direction."""
        self._require_data()
        self.y[...] = processing.replace_region(self.x, self.y, bounds, interp)
        return self

    def to_transmittance(self) -> Jdx:
        if self.yunits.lower() == "absorbance":
            self.y = processing.absorbance_to_transmittance(self.y)
            self.yunits = "TRANSMITTANCE"
        return self

    def to_absorbance(self) -> Jdx:
        if self.yunits.lower() in _Y_TRANSMITTANCE:
            self.y = processing.transmittance_to_absorbance(self.y)
            self.yunits = "ABSORBANCE"
        return self

    def _sort_xy(self) -> Jdx:
        """Sort X and Y in place, retaining the historical helper."""
        self._require_data()
        order = np.argsort(self.x)
        self.x, self.y = self.x[order], self.y[order]
        return self

    def save(self, out_fp: str | Path | None = None, backup: bool = True) -> Path:
        """Save uncompressed XY JCAMP; write sorted data without changing self."""
        self._require_data()
        output = Path(out_fp) if out_fp is not None else Path(self.filepath).with_suffix(".jdx")
        order = np.argsort(self.x)
        xs, ys = self.x[order], self.y[order]
        stamp = self.date
        time_value = self.time or (stamp.time() if stamp else None)
        headers = {
            "TITLE": self.title or self.filestem,
            "JCAMP-DX": self.jcampdx,
            "DATA TYPE": self.datatype.upper(),
            "SAMPLING PROCEDURE": self.samplingprocedure,
            "ORIGIN": self.origin,
            "OWNER": self.owner,
            "DATE": stamp.strftime("%y/%m/%d") if stamp else "",
            "TIME": time_value.strftime("%H:%M:%S") if time_value else "",
            "LONGDATE": stamp.strftime("%Y/%m/%d") if stamp else "",
            "DATA PROCESSING": self.dataprocessing,
            "SPECTROMETER/DATA SYSTEM": self.spectrometerdatasystem,
            "XUNITS": self.xunits.upper(),
            "YUNITS": self.yunits.upper(),
            "MAXY": f"{np.nanmax(ys):.12g}",
            "MINY": f"{np.nanmin(ys):.12g}",
            "MAXX": f"{np.nanmax(xs):.12g}",
            "MINX": f"{np.nanmin(xs):.12g}",
            "XFACTOR": "1",
            "YFACTOR": "1",
            "FIRSTX": f"{xs[0]:.12g}",
            "LASTX": f"{xs[-1]:.12g}",
            "FIRSTY": f"{ys[0]:.12g}",
            "NPOINTS": str(len(xs)),
            "DELTAX": f"{abs(xs[-1] - xs[0]) / (len(xs) - 1):.12g}" if len(xs) > 1 else "0",
            "XYDATA": "(XY..XY)",
        }
        lines = [f"##{key}={value}" for key, value in headers.items()]
        lines.extend(f"{x:.12g} {'?' if np.isnan(y) else f'{y:.12g}'}" for x, y in zip(xs, ys))
        lines.append("##END=")
        if output.exists() and backup:
            shutil.copy2(output, output.with_suffix(".bak"))
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOG.info("Saved %s", output)
        return output

    def smooth(self, window_size: int = 31, polyorder: int = 3, **kwargs: Any) -> Jdx:
        """Smooth in place with the shared Savitzky–Golay implementation."""
        self._require_data(window_size)
        self.y = processing.smooth_savgol(self.y, window_size, polyorder, **kwargs)
        return self

    def find_peaks(
        self, distance: int = 40, prominence: float = 0.005, width: float = 5, **kwargs: Any
    ) -> list[tuple[float, float]]:
        """Find spectral peaks; transmittance/reflectance are inverted."""
        self._require_data()
        values = -self.y if self.yunits.lower() in _Y_TRANSMITTANCE else self.y
        indices = processing.find_peak_indices(values, distance, prominence, width, **kwargs)
        return [(float(self.x[i]), float(self.y[i])) for i in indices]

    def find_peaks2(
        self, mph: float = -0.5, mpd: int = 20, threshold: float = 0.000001, **kwargs: Any
    ) -> list[tuple[float, float]]:
        """Compatibility wrapper for the previous peak-finding method."""
        return self.find_peaks(
            distance=mpd, prominence=0, width=0, height=mph, threshold=threshold, **kwargs
        )

    def new_plot(
        self,
        figsize: tuple[float, float] = (10, 5),
        ax: Axes | None = None,
        fig: Figure | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        fontsize: float | None = None,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            if fig is not None:
                ax = fig.gca()
            elif plt.get_fignums():
                ax = plt.gca()
            else:
                fig, ax = plt.subplots(figsize=figsize)
        self.ax, self.fig = ax, ax.figure
        if fontsize is not None:
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if self.xunits:
            ax.set_xlabel(
                "Wavenumber (cm$^{-1}$)" if self.xunits.lower() in _X_WAVENUMBER else self.xunits
            )
        if self.yunits:
            ax.set_ylabel(self.yunits.upper())
        if self.datatype.lower() == "infrared spectrum" and not ax.xaxis_inverted():
            ax.invert_xaxis()
        return ax

    def plot(
        self, title: str | None = None, linewidth: float = 1, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        axes = self.new_plot(ax=ax)
        axes.plot(
            self.x,
            self.y,
            label=title or self.title or self.filestem,
            linewidth=linewidth,
            **kwargs,
        )
        return axes

    def plot_peaks(
        self,
        rng: tuple[float, float] = (0, 3000),
        xoffset: float = 0,
        fontsize: float = 8,
        **kwargs: Any,
    ) -> list[tuple[float, float]]:
        import matplotlib.pyplot as plt

        peaks = self.find_peaks(**kwargs)
        axes = self.ax if self.ax is not None else plt.gca()
        for px, py in peaks:
            if rng[0] < px < rng[1]:
                axes.annotate(
                    f"{px:.2f}", xy=(px, py), xytext=(px + xoffset, py), fontsize=fontsize
                )
        return peaks

    def baseline_correction(
        self, plot: bool = False, lam: float = 2e7, p: float = 0.005, niter: int = 10
    ) -> Jdx:
        """Asymmetric least-squares baseline correction (Eilers/Boelens)."""
        self._require_data(3)
        values = self.y
        baseline = processing.estimate_baseline_legacy_asls(values, lam, p, niter)
        corrected = values - baseline
        if plot:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.plot(self.x, values, label="Original")
            ax.plot(self.x, baseline, label="Baseline")
            ax.plot(self.x, corrected, label="Corrected")
            if self.datatype.lower() == "infrared spectrum":
                ax.invert_xaxis()
            ax.legend()
        self.y = corrected
        return self

    def calibrate_x(
        self,
        x_values_old: tuple[float, float] = (906.7, 2850.7),
        x_values_new: tuple[float, float] = (906.7, 2850.7),
        resample: bool = True,
    ) -> Jdx:
        """Calibrate X, optionally resampling onto the original grid (legacy default)."""
        self._require_data(2)
        corrected_x = processing.calibrate_coordinates(self.x, x_values_old, x_values_new)
        if x_values_old == x_values_new:
            return self
        if resample:
            self.y = interpolate_linear(self.x, corrected_x, self.y)
        else:
            self.x = corrected_x
        return self

    def from_spa2(self, fpath: str | Path) -> None:
        """Load legacy Thermo SPA data with the standalone reader."""
        self.y, self.x, self.title = read_spa(fpath)
        self.datatype = "INFRARED SPECTRUM"
        self.xunits = "1/CM"
        self.yunits = "ABSORBANCE"
        self.filepath = Path(fpath).resolve()

    def from_spa(self, fpath: str | Path) -> None:
        """Load the supported legacy SPA layout using the internal reader."""
        self.from_spa2(fpath)

    def from_scn(self, fpath: str | Path) -> None:
        """Load Buck 500 infrared text data using the internal SCN reader."""
        from zuy.spectrum.scn import read_scn

        source = read_scn(fpath)
        self.x, self.y = source.x, source.y
        self.title = source.title
        self.spectrometerdatasystem = source.instrument
        self.datatype = "INFRARED SPECTRUM"
        self.xunits = "1/CM"
        self.yunits = "ARBITRARY"
        self.dataprocessing = ""
        self.date = source.date
        self.time = None
        self.filepath = Path(fpath).resolve()
        self.headers = {"longdate": source.acquired}
        self._require_data()

    def _copy_reader(self, source: Any) -> None:
        for name, value in vars(source).items():
            if name in {"x", "y"}:
                setattr(self, name, np.asarray(value, dtype=float).copy())
            elif name not in {"ax", "fig"}:
                # Preserve additional vendor-specific metadata (legacy behavior).
                setattr(self, name, copy.deepcopy(value))
        self._require_data()

    def info(self) -> None:
        """Print the spectrum metadata without dumping the full sample vectors."""
        print(f"\n*** SPECTRUM INFO: {self.title} ***")
        for name in self.__dataclass_fields__:
            if name not in {"x", "y", "ax", "fig", "headers"}:
                print(f"{name:24} {getattr(self, name)}")
        print(f"{'npoints':24} {len(self.x)}")


def read_spa(filepath: str | Path) -> tuple[Array, Array, str]:
    """Read the legacy SPA layout; other SPA variants need a dedicated reader."""
    import struct

    path = Path(filepath)
    with path.open("rb") as stream:
        stream.seek(564)
        raw = stream.read(4)
        if len(raw) != 4:
            raise ValueError(f"Truncated SPA file: {path}")
        count = struct.unpack("<i", raw)[0]
        if not 1 <= count <= 10_000_000:
            raise ValueError(f"Invalid SPA sample count {count}")
        stream.seek(30)
        title = stream.read(255).split(b"\x00", 1)[0].decode("latin-1", errors="replace")
        stream.seek(576)
        maximum, minimum = struct.unpack("<ff", stream.read(8))
        stream.seek(288)
        offset: int | None = None
        for _ in range(100_000):
            raw = stream.read(2)
            if len(raw) != 2:
                break
            if struct.unpack("<H", raw)[0] == 3:
                raw = stream.read(2)
                if len(raw) == 2:
                    offset = struct.unpack("<H", raw)[0]
                break
        if offset is None:
            raise ValueError("SPA data offset marker not found")
        stream.seek(offset)
        y = np.frombuffer(stream.read(count * 4), dtype="<f4").astype(float)
        if len(y) != count:
            raise ValueError("SPA file ends before all samples can be read")
    x = np.linspace(maximum, minimum, count)
    return y, x, title


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot one or more JCAMP or XY spectra")
    parser.add_argument("files", nargs="+", type=Path, help="JDX, DX, SPA, SCN or text files")
    parser.add_argument("--raw", action="store_true", help="Disable automatic IR cleanup")
    args = parser.parse_args(argv)
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(10, 5))
    for filepath in args.files:
        Jdx(filepath, auto_tidy=not args.raw).plot(ax=ax)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
