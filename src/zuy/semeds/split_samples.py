from typing import Dict
import pandas as pd  #   type: ignore
import logging

logger = logging.getLogger(__name__)


def split_samples(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame into per-sample DataFrames.

    Steps:
    - Expect 'Project Path' column already split into 'Sample'.
    - Clean and normalize columns.
    - Split into dict {sample_name: DataFrame}.
    """
    samples = {}

    if "Sample" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Sample' column")

    # Clean 'No' column: remove non-digits
    if "No" in df.columns:
        df["No"] = df["No"].astype(str).str.extract(r"(\d+)", expand=False)

    # Drop rows missing 'Sample'
    df = df.dropna(subset=["Sample"])

    # Process each sample
    for s in sorted(df["Sample"].unique()):
        dfi = df[df["Sample"] == s].copy()

        # Drop helper columns if present
        for col in ["Project Path", "Path", "Sample", "Site"]:
            if col in dfi.columns:
                dfi = dfi.drop(columns=[col])

        # Remove any NaN columns

        dfi = dfi.loc[:, dfi.columns.dropna()]
        dfi = dfi.drop(columns=["source_file", "Zakazka", "Project"])

        # Convert to numeric where possible
        try:
            dfi = dfi.apply(pd.to_numeric, errors="coerce")
        except ValueError:
            logger.warning("Could not convert all columns to numeric")

        # Sort by measurement number
        if "No" in dfi.columns:
            dfi = dfi.sort_values(by="No")

        samples[s] = dfi.reset_index(drop=True)

    return samples
