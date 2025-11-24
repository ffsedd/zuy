#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List
import tempfile


HDIR = Path.home().resolve()
ZAK_DIR_CANDIDATES: List[Path] = [
    Path(r"d:\Dropbox\ZUMI\zakazky"),
    HDIR / "Dropbox/ZUMI/zakazky",
    HDIR / "ZUMI/zakazky",
    HDIR / "zakazky",
    HDIR / "DATA/DISK/Dropbox/ZUMI/zakazaky",
    HDIR / "DATA/DISK/Dropbox/ZUMI/zakazky",
    Path("/media/m/w750/zakazky/"),
]

RE_ZAK = re.compile(r"(\d{4}).*")


def find_zakazky_dir(dpaths: Iterable[Path] = ZAK_DIR_CANDIDATES) -> Path:
    """
    Return the first existing directory from the candidates.

    Raises:
        FileNotFoundError: if none of the paths exist.

    >>> tmp = Path(".").resolve()
    >>> find_existing_dir([tmp]) == tmp
    True
    """
    for d in dpaths:
        p = d.resolve()
        if p.is_dir():
            return p
    raise FileNotFoundError("Zakazky directory not found")


def zak_dict(zdir: Path) -> Dict[int, Path]:
    """
    Build dict: zak-no → folder path. Expects folder names starting with YYYY.

    Example folder names:
        1988 Novak - Policka
        2023-Kral-Vcelnice
        2010_Stastny

    >>> with tempfile.TemporaryDirectory() as td:
    ...     base = Path(td)
    ...     _ = (base / "2020_Test").mkdir()
    ...     _ = (base / "bad").mkdir()
    ...     result = zak_dict(base)
    ...     list(result.keys()) == [2020]
    True

    Returns:
        Dict[int, Path]: ordered mapping from integer year to folder.
    """
    root = Path(zdir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Zakazky directory not found: {root}")

    d: Dict[int, Path] = {}
    for p in sorted(root.glob("*")):
        if not p.is_dir():
            continue
        m = RE_ZAK.match(p.stem)
        if m:
            d[int(m.group(1))] = p
    return d


@dataclass(frozen=True)
class ImgParams:
    """Microscope file parameters.

    Example filename format:
        1234v2_MOD_T1_M40_01_note-extra.png

    >>> p = parse_img_name("1234v2_MO_T1_M40_01_hello-world.jpg")
    >>> (p.zak, p.vz, p.obj, p.note)
    ('1234', '2', '40', 'hello-world')
    """
    zak: str
    vz: str
    mod: str
    typ: str
    mik: str
    obj: str
    fileid: str
    note: str


def parse_img_name(fp: Path | str) -> ImgParams:
    """
    Parse microscope filename into ImgParams.

    Format:
        ZAKvVZ_MOD_TYP_MOBJ_FILEID(_NOTE).ext

    >>> parse_img_name("1111v1_MOD_TYP_M20_00.png").zak
    '1111'
    >>> parse_img_name("1111v1_MOD_TYP_M20_00.png").obj
    '20'
    """
    fp = Path(fp)
    logging.debug("parse %s", fp)

    parts = fp.stem.replace("-", "_").split("_")
    if len(parts) < 5:
        raise ValueError(f"invalid image filename format: {fp.name}")

    zak_vz, mod, typ, mik_obj, fileid, *note = parts
    try:
        zak, vz = zak_vz.split("v")
        mik, obj = mik_obj[0], mik_obj[1:]
    except Exception as e:
        raise ValueError(f"failed parsing microscope token: {e}") from e

    return ImgParams(
        zak=zak,
        vz=vz,
        mod=mod,
        typ=typ,
        mik=mik,
        obj=obj,
        fileid=fileid,
        note="_".join(note) if note else "",
    )


def main() -> None:
    root = find_zakazky_dir()
    print(f"root: {root}")
    zmap = zak_dict(root)
    for k, p in zmap.items():
        print(k, p.stem)

    if zmap:
        last_key = next(reversed(zmap))
        print(last_key, zmap[last_key])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
