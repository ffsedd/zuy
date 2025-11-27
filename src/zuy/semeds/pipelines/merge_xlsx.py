from pathlib import Path
import pandas as pd  # type: ignore
from pandas import DataFrame  # type: ignore
import re
from typing import Pattern, Union
from zuy.common.logger import setup_logger

logger = setup_logger(__name__)


def put_columns_first(df: DataFrame, first_columns: list[str]) -> DataFrame:
    return df.reindex(columns=[*first_columns, *df.columns.difference(first_columns)])


def split_on_header_row(
    df: DataFrame, fname: str, pattern: Union[str, Pattern], column: int = 0
) -> list[DataFrame]:
    """
    Split a DataFrame into sub-DataFrames using a regex-matched header row.
    Each matching row becomes the header for the following sub-DataFrame.
    """
    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    col_values = df.iloc[:, column].astype(str)  # extract column
    row_indices = col_values[col_values.str.match(regex)].index.tolist() + [
        len(df)
    ]  # find "Label" rows

    tables = []
    for start, end in zip(row_indices, row_indices[1:]):
        header = df.iloc[start]
        data = df.iloc[start + 1 : end].copy()
        data.columns = header

        # add filename column
        data["source_file"] = fname
        print(data.columns)

        # put columns first
        data = put_columns_first(
            data,
            [
                "source_file",
                "Project Path (1)",
                "Project Path (2)",
                "Project Path (3)",
                "Label",
                "Total",
            ],
        )

        tables.append(data.reset_index(drop=True))

    return tables


def tables_from_xlsx(fp: Union[str, Path]) -> list[DataFrame]:
    """Load an XLSX file and split it into multiple labeled tables."""
    logger.info(f"Loading file: {fp}")
    fp = Path(fp)
    df = pd.read_excel(fp, engine="openpyxl", header=None)

    print(df)

    tables = split_on_header_row(df, fname=fp.name, pattern=r".*Label.*", column=0)
    logger.info(f"Loaded {len(tables)} tables from {fp}")
    logger.debug(f"First table: {tables[0].head()}")
    return tables


def categorize_tables(tables: list[DataFrame]) -> dict[str, list[DataFrame]]:
    """
    Classify tables by type based on the presence of column headers.
    Returns a dictionary mapping category -> list of DataFrames.
    """
    categorized: dict[str, list[DataFrame]] = {"weight": [], "oxide": []}

    for tbl in tables:
        key = "weight" if "O" in tbl.columns else "oxide"
        logger.info(f"Table categorized as '{key}'")
        categorized[key].append(tbl)

    return {k: v for k, v in categorized.items() if v}


def merge_xlsx(dpath: Path) -> dict[str, DataFrame]:
    """
    Merge all XLSX tables in a directory by category.
    Returns a dict mapping category -> merged DataFrame.
    """
    fpaths = sorted(dpath.glob("*.xlsx"))
    if not fpaths:
        raise FileNotFoundError(f"No .xlsx files found in {dpath}")

    tables_by_category: dict[str, list[DataFrame]] = {}

    for fp in fpaths:
        logger.info(f"Processing: {fp.name}")
        tables = tables_from_xlsx(fp)
        categorized = categorize_tables(tables)

        for key, group in categorized.items():
            tables_by_category.setdefault(key, []).extend(group)

    merged = {
        key: pd.concat(group, ignore_index=True)
        for key, group in tables_by_category.items()
    }

    for key, df in merged.items():
        logger.info(f"{key}: {df.shape[0]} rows x {df.shape[1]} columns")

    return merged


def main() -> None:
    from zuy.semeds.main import parse_args

    args = parse_args()
    dpath = Path(args.src).resolve()
    outdir = dpath / "merged"
    outdir.mkdir(parents=True, exist_ok=True)

    merged_tables = merge_xlsx(dpath)

    if not merged_tables:
        logger.info("No tables were merged.")
        return

    for key, df in merged_tables.items():
        out_fp = outdir / f"{key}.xlsx"
        if not out_fp.exists() or args.overwrite:
            df.to_excel(out_fp, index=False)
            logger.info(f"Saved '{key}' table to {out_fp}")
        else:
            logger.info(f"Skipped saving '{key}' table to {out_fp} (exists)")


if __name__ == "__main__":
    main()
