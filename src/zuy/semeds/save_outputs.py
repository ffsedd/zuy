from pathlib import Path
import re
import logging

import pandas as pd  # type: ignore

from zuy.common.dftools import save_formatted_xlsx
from zuy.semeds.models import ZakazkaSample
from zuy.semeds.plot_element_correlations import plot_correlations_from_tsv


logger = logging.getLogger(__name__)


def save_output(
    samples: dict[ZakazkaSample, pd.DataFrame], outdir: Path, overwrite: bool = False
) -> None:
    """
    Save each sample's DataFrame to LaTeX, TSV, XLS, and PNG heatmap.
    """
    outdir.mkdir(exist_ok=True, parents=True)

    for (zakazka, sample), df in samples.items():
        zak_sample = f"{zakazka}v{sample}"
        # print(zak_sample)
        logger.info(f"Saving outputs for sample: {zak_sample}")

        # 1. Save .xls
        fp_xls = outdir / f"{zak_sample}.xls"
        if fp_xls.exists() and not overwrite:
            logger.info(
                f"File {fp_xls} exists and overwrite is False. Skipping XLS, TEX, TSV save."
            )
        else:
            save_formatted_xlsx(fp_xls, {"results": df}, widths=[(0, 50, 5)])

            df2 = df.droplevel(["Zakazka", "Sample"]).drop(
                columns=["source_file", "Site", "Project"]
            )

            # drop unneeded index levels
            # 2. Save .tex
            fp_tex = outdir / f"{zak_sample}.tex"

            tex = df2.to_latex(index=True, na_rep="", float_format="%.1f")
            tex = re.sub(r" &", "\t&", tex)  # tab-align
            fp_tex.write_text(tex)

            # 3. Save .tsv
            fp_tsv = outdir / f"{zak_sample}.tsv"
            df2.to_csv(fp_tsv, sep="\t", index=True, na_rep="", float_format="%.1f")
            plot_correlations_from_tsv(fp_tsv)

    logger.info(f"Saved outputs to {outdir}")
