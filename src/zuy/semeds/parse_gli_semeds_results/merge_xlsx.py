from pathlib import Path
import pandas as pd
import re
import logging
from typing import Pattern
from collections import defaultdict

logger = logging.getLogger(__name__)


def split_on_header_row(
    df: pd.DataFrame, pattern: str | Pattern, column: int = 0
) -> list[pd.DataFrame]:
    """
    Split a DataFrame into sub-DataFrames using a regex-matched header row.
    Each matching row becomes the header for the following sub-DataFrame.
    """
    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    col_values = df.iloc[:, column].astype(str)
    row_ix = col_values[col_values.str.match(regex)].index.tolist() + [len(df)]

    tables = []
    for i in range(len(row_ix) - 1):
        header = df.iloc[row_ix[i]]
        data = df.iloc[row_ix[i] + 1 : row_ix[i + 1]].copy()
        data.columns = header
        tables.append(data.reset_index(drop=True))

    return tables


def tables_from_xlsx(fp: str | Path) -> list[pd.DataFrame]:
    """Load and split multiple tables from an XLSX file."""
    logger.info(f"Loading file: {fp}")
    df = pd.read_excel(fp, engine="openpyxl", header=None)
    tables = split_on_header_row(df, pattern=r".*Label.*", column=0)

    # Add source filename to each table for traceability
    for t in tables:
        t["source_file"] = Path(fp).name

    return tables


def sort_tables(tables: list[pd.DataFrame]) -> dict[str, list[pd.DataFrame]]:
    """
    Sort tables into categories based on presence of specific column headers.

    Returns:
        A dictionary with keys like 'weight' and 'oxide', mapping to lists of DataFrames.
    """
    result = defaultdict(list)

    for tbl in tables:
        key = "weight" if "O" in tbl.columns else "oxide"
        if key == "weight":
            logger.info("Table contains column 'O' – likely weight percent data")
        result[key].append(tbl)

    return dict(result)


def merge_xlsx(dpath: Path) -> dict[str, pd.DataFrame]:
    """
    Load all XLSX files in a directory, extract labeled tables, classify them,
    and merge all tables in each category.

    Returns:
        Dictionary of merged DataFrames by category.
    """
    files = sorted(dpath.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx result files found in {dpath}")
        return {}

    tables = defaultdict(list)

    for fp in files:
        logger.info(f"Processing: {fp.name}")
        t = tables_from_xlsx(fp)
        sorted_tables = sort_tables(t)

        for key, group in sorted_tables.items():
            tables[key].extend(group)

    merged_tables = {
        key: pd.concat(group, ignore_index=True)
        for key, group in tables.items()
        if group
    }

    for key, df in merged_tables.items():
        logger.info(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")

    return merged_tables


def main() -> None:
    dpath = Path("/home/m/Dropbox/ZUMI/zakazky/ZADANI-SEM/sem-25-07/100725/")
    outdir = dpath / "merged"
    outdir.mkdir(parents=True, exist_ok=True)

    merged = merge_xlsx(dpath)

    if not merged:
        logger.info("No tables were merged.")
        return

    for key, df in merged.items():
        out_fp = outdir / f"{key}.xlsx"
        df.to_excel(out_fp, index=False)
        logger.info(f"Saved '{key}' table to {out_fp}")


if __name__ == "__main__":
    main()
