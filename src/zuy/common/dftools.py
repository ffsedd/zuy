#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def dataframe_to_latex(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    headers: list[str] | None = None,
    header_format: str = r"\rowcolor{gray!25}",
    column_format: str | None = None,
    index: bool = False,
    table_type: str = "tabular",
    newline: str = "\n",
    hline: str = r"\hline",
) -> str:
    """
    Convert DataFrame to a LaTeX tabular string.

    Parameters:
        df: DataFrame to convert.
        columns: Optional subset of columns to include.
        headers: Optional replacement header row.
        header_format: LaTeX prefix for the header row.
        column_format: e.g. 'ccc' or 'c*3'. Defaults to 'c' repeated for each column.
        index: Include DataFrame index?
        table_type: e.g. 'tabular' or 'longtable'.
        newline: Line ending character.
        hline: Horizontal line command.

    Returns:
        LaTeX string of the table.
    """
    if columns:
        df = df[columns]

    col_fmt = column_format or "c" * df.shape[1]
    begin = f"\\begin{{{table_type}}}{{{col_fmt}}}"
    end = f"\\end{{{table_type}}}"

    header_line = (
        ""
        if headers is None
        else f"{header_format} {' & '.join(headers)} \\\\{newline}"
    )

    tex = df.to_csv(
        sep="đ",
        lineterminator=r" \\ ",
        escapechar="",
        encoding="utf-8",
        index=index,
        header=False,
        quotechar="Đ",
        na_rep="~",
    )

    tex = tex.replace("Đ", "").replace("&", r"\&").replace("đ", " & ").replace("~", "")

    tex_lines = tex.splitlines()
    table = newline.join([begin, hline, header_line] + tex_lines + [hline, end])
    return table


def save_formatted_xlsx(
    fpath: Path,
    dataframes: dict[str, pd.DataFrame],
    widths: list[tuple[int, int, int]] = [],
    index: bool = True,
    freeze_panes: tuple[int, int] = (1, 1),
    color_scale: dict | None = None,
) -> None:
    """
    Save multiple DataFrames to a formatted Excel file using xlsxwriter.

    Parameters:
        fpath: Output file path.
        dataframes: Dict of sheet_name -> DataFrame.
        widths: List of (first_col, last_col, width) tuples per sheet.
        index: Write index column?
        freeze_panes: Excel freeze panes coordinates.
        color_scale: Optional formatting config.
    """
    color_scale = color_scale or {
        "type": "3_color_scale",
        "min_color": "#ffffff",
        "mid_type": "percentile",
        "mid_value": 80,
        "mid_color": "#ffff99",
        "max_color": "#ff9999",
    }

    with pd.ExcelWriter(fpath, engine="xlsxwriter") as writer:
        for i, (sheet, df) in enumerate(dataframes.items()):
            df.to_excel(
                writer, sheet_name=sheet, index=index, freeze_panes=freeze_panes
            )
            ws = writer.sheets[sheet]

            if i < len(widths):
                from_, to_, width = widths[i]
                ws.set_column(from_, to_, width)

            if color_scale:
                max_row, max_col = len(df) + 1, len(df.columns)
                col_range = f"A1:{xl_col_to_name(max_col - 1)}{max_row}"
                ws.conditional_format(col_range, color_scale)


def flatten_nested_dicts(dicts: list[dict]) -> pd.DataFrame:
    """
    Convert list of nested dicts into a DataFrame with MultiIndex columns.
    """
    records = [
        {(k1, k2): v for k1, sub in d.items() for k2, v in sub.items()} for d in dicts
    ]
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "Name": ["John", "Alice", "Bob"],
            "Age": [30, 25, 35],
            "City": ["New York", "LA", "Chicago"],
        }
    )

    latex = dataframe_to_latex(df, columns=["Name", "Age"], headers=["Name", "Age"])
    print(latex)

    save_formatted_xlsx(Path("test_output.xlsx"), {"Sheet1": df}, widths=[(0, 2, 12)])
