from pathlib import Path
import re
import logging

import pandas as pd  # type: ignore

from zuy.common.dftools import save_formatted_xlsx


logger = logging.getLogger(__name__)


def save_output(samples: dict[str, pd.DataFrame], outdir: Path) -> None:
    """
    Save each sample's DataFrame to LaTeX, TSV, XLS, and PNG heatmap.
    """
    outdir.mkdir(exist_ok=True, parents=True)

    for sample, df in samples.items():
        logger.info(f"Saving outputs for sample: {sample}")

        # 1. Save .tex
        tex = df.to_latex(index=False, na_rep="", float_format="%.1f")
        tex = re.sub(r" &", "\t&", tex)  # tab-align
        fp_tex = outdir / f"{sample}.tex"
        fp_tex.write_text(tex)

        # 2. Save .tsv
        fp_tsv = outdir / f"{sample}.tsv"
        df.to_csv(fp_tsv, sep="\t", index=False)

        # 3. Save .xls
        fp_xls = outdir / f"{sample}.xls"
        save_formatted_xlsx(fp_xls, {"results": df}, widths=[(0, 50, 5)], index=False)

    logger.info(f"Saved outputs to {outdir}")
