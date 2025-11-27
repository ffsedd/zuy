import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Slider  # type: ignore

from zuy.spectrum.io import parse_msa_file
from zuy.spectrum.models import Spectrum
from zuy.spectrum.processing import tidy_spectrum
from zuy.spectrum.squre_root_scale import register_sqrt_scale

register_sqrt_scale()


def interactive_baseline_vs_code(s: Spectrum):
    """
    Interactive ASLS baseline tuning with smoothing, lambda, and p sliders.
    Works in VS Code or any standard Python script.
    """
    lam0, p0, smooth0 = 1e6, 0.01, 5

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(bottom=0.35)

    s_tidy = tidy_spectrum(s, smooth_window=smooth0, lam=lam0, p=p0)
    (line,) = ax.plot(s.x, s_tidy.y, lw=1)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts")
    plt.ylim(0, None)
    plt.yscale("sqrt")  # type: ignore
    # ax.set_yscale("function", functions=(np.sqrt, lambda v: v**2))

    ax.grid(True)
    ax.set_title(f"λ={lam0:.1e}, p={p0:.3f}, smooth={smooth0}")

    ax_lam = plt.axes((0.15, 0.15, 0.7, 0.03))
    ax_p = plt.axes((0.15, 0.1, 0.7, 0.03))
    ax_smooth = plt.axes((0.15, 0.05, 0.7, 0.03))

    slider_lam = Slider(ax_lam, "λ", valmin=1, valmax=1e5, valinit=lam0, valfmt="%1.1e")
    slider_p = Slider(ax_p, "p", valmin=0.001, valmax=0.1, valinit=p0, valfmt="%1.3f")
    slider_smooth = Slider(ax_smooth, "smooth", valmin=1, valmax=51, valinit=smooth0, valstep=1)

    def update(val):
        lam = slider_lam.val
        p = slider_p.val
        smooth_win = int(slider_smooth.val)
        s_tidy = tidy_spectrum(s, smooth_window=smooth_win, lam=lam, p=p)
        line.set_ydata(s_tidy.y)
        ax.set_title(f"λ={lam:.1e}, p={p:.3f}, smooth={smooth_win}")
        fig.canvas.draw_idle()

    slider_lam.on_changed(update)
    slider_p.on_changed(update)
    slider_smooth.on_changed(update)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive ASLS baseline tuning for a spectrum")
    parser.add_argument("msa_file", type=Path, help="Path to the .msa spectrum file")
    args = parser.parse_args()

    if not args.msa_file.exists():
        raise FileNotFoundError(f"{args.msa_file} not found")

    s = parse_msa_file(args.msa_file)
    interactive_baseline_vs_code(s)


if __name__ == "__main__":
    main()
