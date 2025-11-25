import pandas as pd  # type: ignore

ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U"
).split()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a spectral DataFrame and create MultiIndex
    ['Zakazka', 'Sample', 'No'].
    Assumes `Sample` may be integer-only (no embedded Zakazka).
    """
    # --- Parse Project Sample Site ---
    if "Project Path" in df.columns:
        df["Project Path"] = df["Project Path"].astype(str)
        df[["Project", "Sample", "Site"]] = df["Project Path"].str.split(
            "/", expand=True
        )
    elif all(
        col in df.columns
        for col in ["Project Path (1)", "Project Path (2)", "Project Path (3)"]
    ):
        df.rename(
            columns={
                "Project Path (1)": "Project",
                "Project Path (2)": "Sample",
                "Project Path (3)": "Site",
            },
            inplace=True,
        )

    # --- Sample name split  ---
    # If Sample is integer-only, skip splitting; otherwise keep original logic.
    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].astype(str)

        # Detect if Sample contains a separator that implies Zakazka+Sample
        # If not, do nothing (uses integer-only Sample).
        sample0 = df["Sample"].iloc[0]
        if any(sep in sample0 for sep in ["_", "-", "v", "V"]):
            # previous logic: split into Zakazka and Sample
            split_char = None
            for ch in ["_", "-", "v", "V"]:
                if ch in sample0:
                    split_char = ch
                    break
            if split_char:
                df[["Zakazka", "Sample"]] = df["Sample"].str.split(
                    split_char, expand=True
                )
        else:
            # integer-only Sample; ensure we keep Sample and create a placeholder Zakazka if needed
            # If you have a separate column or logic for Zakazka, use that here.
            # For now, if Zakazka not present, set it to a default or NaN
            if "Zakazka" not in df.columns:
                df["Zakazka"] = pd.NA  # or df["Zakazka"] = -1, or df["Project"], etc.

    # --- Rename Total to Sum ---
    if "Total" in df.columns:
        df.rename(columns={"Total": "Sum"}, inplace=True)

    # --- Normalize label column ---
    label_col = next(
        (col for col in df.columns if col in ["Spectrum Label", "Label"]), None
    )
    if label_col:
        df.rename(columns={label_col: "No"}, inplace=True)
        df["No"] = df["No"].astype(str).str.extract(r"(\d+)", expand=False)
    else:
        df["No"] = pd.NA

    # --- Drop rows with missing Sample ---
    df = df.dropna(subset=["Sample"]).copy()

    # --- Convert 'No' to numeric ---
    df["No"] = pd.to_numeric(df["No"], errors="coerce")

    # --- Ensure no duplicate column names before further processing ---
    # If there happen to be duplicates (e.g., Sample existed already and was recreated),
    # keep the first occurrence and drop the rest.
    if df.columns.duplicated().any():
        # df = df.loc[:, ~df.columns.duplicated()]
        raise ValueError(f"Duplicate column names found {df.columns.duplicated()}")

    # --- Column ordering ---
    columns = df.columns.tolist()
    elements = [e for e in ELEMENT_SYMBOLS if e in columns]
    others = [
        col
        for col in columns
        if col not in elements and col not in ["Zakazka", "Sample", "No"]
    ]
    ordered_columns = elements + others

    # Include Zakazka, Sample, No at the front in a known order
    df = df[["Zakazka", "Sample", "No"] + ordered_columns]

    # --- Final sort ---
    # Some rows may have NaN Zakazka if integer-only Sample; they will go last.
    df = df.sort_values(["Zakazka", "Sample", "No"], na_position="last").reset_index(
        drop=True
    )

    # --- Create MultiIndex ---
    df.set_index(["Zakazka", "Sample", "No"], inplace=True)

    return df
