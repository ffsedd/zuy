"""Verify processing orders, plot content, and legacy loading behavior."""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from zuy.spectrum.jdx import Jdx
from zuy.spectrum.models import Spectrum
from zuy.spectrum.plotting import plot_baseline_diagnostics
from zuy.spectrum.processing import tidy_spectrum
from zuy.spectrum.workflows import msa_smooth_then_baseline, raman_baseline_then_smooth


FIXTURES = Path(__file__).parent / "fixtures"


def test_named_raman_workflow_matches_original_order() -> None:
    with np.load(FIXTURES / "raman_processing_reference.npz") as data:
        y = data["y"]
        result = raman_baseline_then_smooth(y)
        np.testing.assert_allclose(result, data["smoothed"], rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(y, data["y"])


def test_msa_workflow_and_legacy_wrapper_match_original_order() -> None:
    with np.load(FIXTURES / "msa_processing_reference.npz") as data:
        source = Spectrum(x=data["x"], y=data["y"], metadata={"TITLE": "MSA reference"})
        for result in (msa_smooth_then_baseline(source), tidy_spectrum(source)):
            np.testing.assert_allclose(result.y, data["processed"], rtol=1e-8, atol=1e-9)
            np.testing.assert_array_equal(result.x, source.x)
            assert result.metadata == source.metadata
            assert result.metadata is not source.metadata
            assert not np.shares_memory(result.y, source.y)
            assert not np.shares_memory(result.x, source.x)
        np.testing.assert_array_equal(source.y, data["y"])


def test_diagnostic_plot_uses_supplied_data_without_modification() -> None:
    x = np.arange(3, dtype=float)
    y = np.array([2, 4, 3], dtype=float)
    baseline = np.ones(3)
    axes = Figure().subplots()
    assert plot_baseline_diagnostics(x, y, baseline, invert_x=True, ax=axes) is axes
    assert [line.get_label() for line in axes.lines] == ["Original", "Baseline", "Corrected"]
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), y)
    np.testing.assert_array_equal(axes.lines[1].get_ydata(), baseline)
    np.testing.assert_array_equal(axes.lines[2].get_ydata(), [1, 3, 2])
    assert axes.xaxis_inverted()
    np.testing.assert_array_equal(y, [2, 4, 3])
    np.testing.assert_array_equal(baseline, [1, 1, 1])


@pytest.mark.parametrize(
    ("datatype", "inverted"), [("INFRARED SPECTRUM", True), ("RAMAN SPECTRUM", False)]
)
def test_jdx_baseline_plot_preserves_corrected_values(datatype: str, inverted: bool) -> None:
    plt.switch_backend("Agg")
    x = np.arange(41, dtype=float)
    y = 2 + 0.01 * x + np.exp(-(((x - 20) / 3) ** 2))
    spec = Jdx(x=x, y=y, datatype=datatype)
    reference = spec.copy().baseline_correction()
    try:
        assert spec.baseline_correction(plot=True) is spec
        axes = plt.gca()
        assert bool(axes.xaxis_inverted()) == inverted
        np.testing.assert_array_equal(spec.y, reference.y)
        np.testing.assert_array_equal(axes.lines[2].get_ydata(), reference.y)
    finally:
        plt.close("all")


def test_auto_tidy_and_background_ratio_remain_legacy_workflows(tmp_path: Path) -> None:
    path = tmp_path / "transmittance.jdx"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=TRANSMITTANCE\n##XYDATA=(XY..XY)\n"
        "1000 1\n1100 0.1\n1200 0.01\n##END=\n"
    )
    raw = Jdx(path, auto_tidy=False)
    tidy = Jdx(path)
    np.testing.assert_allclose(raw.y, [1, 0.1, 0.01])
    np.testing.assert_allclose(tidy.y, [0, 1, 2])
    assert tidy.yunits == "ABSORBANCE"
    background = Jdx(x=raw.x, y=np.ones(3), xunits=raw.xunits, yunits=raw.yunits)
    assert raw.subtract(background) is raw
    np.testing.assert_allclose(raw.y, tidy.y)
