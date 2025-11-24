import numpy as np
import tempfile
from pathlib import Path
from zuy.spectrum.emsa import EmsaSpectrum, parse_msa_file


def sample_msa_text() -> str:
    return """\
#TITLE: Test Spectrum
#NPOINTS: 3
#XUNITS: keV
#YUNITS: counts
#SPECTRUM
0.1,100
0.2,200
0.3,150
"""


def test_parse_msa_file_creates_valid_arrays():
    with tempfile.NamedTemporaryFile("w+", suffix=".msa", delete=False) as f:
        f.write(sample_msa_text())
        f.flush()
        path = Path(f.name)

    x, y, metadata = parse_msa_file(path)

    np.testing.assert_allclose(x, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(y, [100, 200, 150])
    assert metadata["TITLE"] == "Test Spectrum"
    assert metadata["NPOINTS"] == "3"


def test_from_msa_creates_spectrum_instance():
    with tempfile.NamedTemporaryFile("w+", suffix=".msa", delete=False) as f:
        f.write(sample_msa_text())
        f.flush()
        path = Path(f.name)

    spectrum = EmsaSpectrum.from_msa(path)

    assert isinstance(spectrum, EmsaSpectrum)
    assert spectrum.x.shape == (3,)
    assert spectrum.y.shape == (3,)
    assert spectrum.metadata["TITLE"] == "Test Spectrum"
