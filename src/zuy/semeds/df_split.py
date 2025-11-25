from typing import Dict
import pandas as pd  # type: ignore

from zuy.semeds.models import ZakazkaSample


def df_split(
    df: pd.DataFrame,
) -> Dict[ZakazkaSample, pd.DataFrame]:
    """
    Given df indexed by ['Zakazka','Sample','No'], return a dict
    mapping each (Zakazka, Sample, No) to its sub‑DataFrame.
    """
    # quick sanity checks
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"DataFrame must have a MultiIndex ['Zakazka', 'Sample', 'No']. It is: {df.index}"
        )
    if df.index.names != ["Zakazka", "Sample", "No"]:
        raise ValueError(
            f"Expected MultiIndex names ['Zakazka', 'Sample', 'No'], got {df.index.names}"
        )

    out: Dict[ZakazkaSample, pd.DataFrame] = {}

    # groupby on all levels yields tuples matching that order
    for (zak, samp), sub in df.groupby(level=["Zakazka", "Sample"]):
        # drop columns that are all NaN
        sub_clean = sub.dropna(axis=1, how="all")
        # optional: ensure numeric where reasonable
        sub_clean = sub_clean.apply(pd.to_numeric, errors="coerce")
        key = ZakazkaSample(int(zak), int(samp))
        out[key] = sub_clean.copy()

    return out
