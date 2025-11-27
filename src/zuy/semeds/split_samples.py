from typing import Dict
import pandas as pd  # type: ignore
from .models import Sample


def split_samples(
    df: pd.DataFrame,
) -> Dict[Sample, pd.DataFrame]:
    """
    Split a MultiIndex DataFrame with ['Zakazka','Sample'] into separate DataFrames
    per Sample key.

    Returns
    -------
    dict
        {Sample: DataFrame}, where each DataFrame contains rows for that
        specific Zakazka and measurement Sample.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex ['Zakazka', 'Sample']")
    if df.index.names != ["Zakazka", "Sample"]:
        raise ValueError(
            f"Expected MultiIndex names ['Zakazka', 'Sample'], got {df.index.names}"
        )

    split_dfs: Dict[Sample, pd.DataFrame] = {}
    # groupby returns tuples matching the levels
    for (zakazka, sample), dfi in df.groupby(level=["Zakazka", "Sample"]):
        # Drop columns that are all NaN
        dfi_clean = dfi.dropna(axis=1, how="all")
        # Optional: convert numeric columns if desired
        dfi_clean = dfi_clean.apply(pd.to_numeric, errors="coerce")

        key = Sample(zakazka=zakazka, sample_no=int(sample))
        split_dfs[key] = dfi_clean.copy()

    return split_dfs
