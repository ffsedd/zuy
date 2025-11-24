import pandas as pd  # type: ignore

ELEMENT_SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U".split()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a spectral DataFrame:
    - Split 'Project Path' into Project / Sample / Site.
    - Split 'Sample' into Zakazka and refined Sample.
    - Normalize label column into 'No' with numeric values.
    - Drop rows with missing Sample.
    - Convert 'No' to numeric.
    - Sort element columns by periodic table order.
    - Reorder columns: 'No', ELEMENTS, then remaining.
    - Sort by 'No' and 'Sample'.
    """

    # --- Path parsing ---
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
    print(df.head())    
    print(df.columns)

    # --- Sample name split ---
    if "Sample" in df.columns:

        df["Sample"] = df["Sample"].astype(str)

        if "_" in df["Sample"][0]:
            split_char = "_"
        elif "-" in df["Sample"][0]:
            split_char = "-"
        elif "v" in df["Sample"][0]:
            split_char = "v"
        elif "V" in df["Sample"][0]:
            split_char = "V"    
        else:
            raise ValueError(f"Failed to split 'Sample' column to Zakazka and Sample: {df.Sample.unique()}")

        print(f"Split char: {split_char}    ")

        try:        
            df[["Zakazka", "Sample"]] = df["Sample"].astype(str).str.split(split_char, expand=True)
            
        except ValueError:
            raise ValueError(f"Failed to split 'Sample' column to Zakazka and Sample: {df.Sample.unique()}")
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

    # --- Column ordering ---
    columns = df.columns.tolist()
    elements = [e for e in ELEMENT_SYMBOLS if e in columns]
    others = [col for col in columns if col not in elements and col != "No"]
    ordered_columns = ["No"] + elements + others
    df = df[ordered_columns]

    # --- Final sort ---
    df = df.sort_values(["No", "Sample"], na_position="last").reset_index(drop=True)

    return df
