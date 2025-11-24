# tests/test_io.py
import pytest
from pathlib import Path
import numpy as np
from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.models import Spectrum


@pytest.fixture
def sample_msa(tmp_path: Path) -> Path:
    """Create a small .msa file for testing."""
    content = """#FORMAT      : EMSA/MAS Spectral Data File
#VERSION     : 1.0
#TITLE       : Spectrum name
#DATE        : 15-FEB-2022
#TIME        : 13:54
#OWNER       : 
#NPOINTS     : 2048.0
#NCOLUMNS    : 1.0
#XUNITS      : keV
#YUNITS      : counts
#DATATYPE    : XY
#XPERCHAN    : 0.01
#OFFSET      : -0.2
#SIGNALTYPE  : EDS_SEM
#CHOFFSET    : 20.0
#LIVETIME    : 20.0
#REALTIME    : 30.527103424
#BEAMKV      : 20.0
#PROBECUR    : 1.005128
#MAGCAM      : 300.0
#XTILTSTGE   : 0.0
#AZIMANGLE   : 0.0
#ELEVANGLE   : 35.0
#XPOSITION mm: -8.4092
#YPOSITION mm: -11.5192
#ZPOSITION mm: 12.3310
##OXINSTPT   : 6
##OXINSTSTROB: 33.77
##OXINSTNDET : 1
##OXINSTELEMS: 6,8,11,13,14,16,19,20,24,26,30,56,82,38,51
##OXINSTLABEL: 6, 0.277, C K?1,2
##OXINSTLABEL: 8, 0.525, O K?1
##OXINSTLABEL: 11, 1.041, Na K?1,2
##OXINSTLABEL: 13, 1.487, Al K?1
#SPECTRUM    : Spectral Data Starts Here
-0.20000, 0.0
-0.19000, 0.0
-0.18000, 0.0
-0.17000, 0.0
-0.16000, 0.0
-0.15000, 0.0
-0.14000, 0.0
-0.13000, 0.0
-0.12000, 0.0
-0.11000, 0.0
-0.10000, 0.0
-0.09000, 0.0
-0.08000, 0.0
-0.07000, 1.0
-0.06000, 1.0
-0.05000, 5.0
-0.04000, 21.0
-0.03000, 79.0
-0.02000, 273.0
-0.01000, 619.0
0.00000, 928.0
0.01000, 1097.0
0.02000, 1118.0
0.03000, 931.0
0.04000, 628.0
0.05000, 342.0
0.06000, 205.0
0.07000, 134.0
0.08000, 134.0
0.09000, 135.0
0.10000, 133.0
0.11000, 122.0
0.12000, 120.0
0.13000, 150.0
0.14000, 157.0
0.15000, 159.0

"""
    fpath = tmp_path / "test.msa"
    fpath.write_text(content)
    return fpath


def test_parse_msa_file(sample_msa: Path):
    spec = parse_msa_file(sample_msa)
    assert isinstance(spec, Spectrum)
    # check first few x/y values
    np.testing.assert_allclose(spec.x[:5], [-0.2, -0.19, -0.18, -0.17, -0.16])
    np.testing.assert_allclose(spec.y[:5], [0, 0, 0, 0, 0])
    # check a non-zero point
    assert spec.y[14] == 1.0
    assert spec.metadata["TITLE"] == "Spectrum name"
