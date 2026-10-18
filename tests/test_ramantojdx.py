"""Exercise file loading, JCAMP export, and source archiving through the CLI."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from zuy.spectrum.jdx import Jdx


@pytest.mark.parametrize("archive_orig", [False, True])
def test_raman_conversion_preserves_or_archives_successful_inputs(
    tmp_path: Path, archive_orig: bool
) -> None:
    source = tmp_path / "rutile.csv"
    x = np.arange(41, dtype=float)
    y = 5 + 0.1 * x + np.exp(-(((x - 20) / 3) ** 2))
    np.savetxt(source, np.column_stack((x, y)), delimiter=",")
    original = source.read_bytes()
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("not a spectrum\n")

    command = [sys.executable, "-m", "zuy.raman.ramantojdx"]
    if archive_orig:
        command = [
            sys.executable,
            "-c",
            "from zuy.raman import ramantojdx; ramantojdx.archive_orig = True; ramantojdx.main()",
        ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    converted = Jdx(source.with_suffix(".jdx"), auto_tidy=False)
    assert converted.title == "rutile"
    assert converted.datatype == "RAMAN SPECTRUM"
    assert (converted.xunits, converted.yunits) == ("1/CM", "INTENSITY")
    np.testing.assert_array_equal(converted.x, x)
    assert np.isfinite(converted.y).all()
    assert not np.allclose(converted.y, y)
    if archive_orig:
        assert not source.exists()
        assert (tmp_path / "orig" / "rutile.csv.bak").read_bytes() == original
    else:
        assert source.read_bytes() == original
        assert not (tmp_path / "orig").exists()
    assert invalid.read_text() == "not a spectrum\n"
    assert not invalid.with_suffix(".jdx").exists()
    assert not (tmp_path / "orig" / "invalid.csv.bak").exists()
    assert "Spectrum conversion failed:" in result.stdout
