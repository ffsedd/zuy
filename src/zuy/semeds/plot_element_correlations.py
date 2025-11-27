#!/usr/bin/env python3
""" """

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

# from IPython.core.display import display, HTML
from pathlib import Path

IGNORE_PAIRS = [
    ["Al", "Mg"],
    ["Al", "Na"],
    ["Al", "O"],
    ["Al", "Si"],
    ["Al", "K"],
    ["Al", "Ti"],
    ["Al", "P"],
    ["Al", "Fe"],
    ["Al", "Ca"],
    ["Ca", "K"],
    ["Ca", "Mg"],
    ["Ca", "Na"],
    ["Ca", "Fe"],
    ["Ca", "Si"],
    ["Ca", "O"],
    ["Cl", "Na"],
    ["Cl", "P"],
    ["Cl", "K"],
    ["Cl", "Pb"],
    ["Cl", "Fe"],
    ["Cl", "Zn"],
    ["Fe", "Na"],
    ["Fe", "O"],
    ["Fe", "Si"],
    ["K", "Mg"],
    ["K", "Na"],
    ["K", "Si"],
    ["K", "O"],
    ["Mg", "Na"],
    ["Mg", "O"],
    ["Mg", "Fe"],
    ["Mg", "Si"],
    ["Mg", "Ti"],
    ["Mn", "O"],
    ["Mn", "Na"],
    ["Na", "O"],
    ["Na", "Si"],
    ["Na", "Mn"],
    ["Na", "Pb"],
    ["Na", "Zn"],
    ["O", "P"],
    ["O", "S"],
    ["O", "Si"],
    ["O", "Ti"],
    ["P", "Ti"],
    ["P", "Si"],
]


def df2corrs(df, rmin=0):

    dataCorr = df.corr(method="pearson")
    dataCorr = dataCorr.stack().reset_index()
    dataCorr = dataCorr[
        dataCorr["level_0"].astype(str) != dataCorr["level_1"].astype(str)
    ]

    # filtering out lower/upper triangular duplicates
    dataCorr["ordered-cols"] = dataCorr.apply(
        lambda x: "-".join(sorted([x["level_0"], x["level_1"]])), axis=1
    )
    dataCorr = dataCorr.drop_duplicates(["ordered-cols"])
    dataCorr.drop(["ordered-cols"], axis=1, inplace=True)
    dataCorr.columns = ["a", "b", "r"]
    # print(dataCorr)
    return dataCorr


def plot_corr(df, rmin=0.5, rmax=1):
    # ~ print(df.Hg[~df.Hg.isna()])
    # get correlations
    tc = df2corrs(df)

    # filter by r
    tc = tc[(tc.r >= rmin) & (tc.r <= rmax)].sort_values("r", ascending=False)
    print(tc)

    pairs = tc.loc[:, ["a", "b"]].values.tolist()

    pairs = [p for p in pairs if sorted(p) not in IGNORE_PAIRS]  # filter ignored pairs
    # print(pairs)
    npairs = len(pairs)

    fig = plt.figure(figsize=(8, npairs / 2.5))

    for i, pair in enumerate(pairs):
        ax = fig.add_subplot(npairs // 5 + 1, 5, i + 1)
        ax.scatter(df[pair[0]], df[pair[1]], c="red", s=4)
        r = tc.r[(tc.a == pair[0]) & (tc.b == pair[1])].tolist()[0]
        ax.set_title("  ".join(pair) + f"      r:{r:.2f}")
        # ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.xticks([])
        plt.yticks([])

        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])

    fig.tight_layout()


def plot_correlations_from_tsv(fpath):

    df = pd.read_csv(fpath, sep="\t")
    drop_cols = (
        c
        for c in df.columns
        if c
        in (
            "Total",
            "No",
            "Sum",
        )
    )
    df = df.drop(columns=drop_cols)
    df = df.fillna(0)
    plot_corr(df, rmin=0.5)

    fpath_out = fpath.parent / f"{fpath.stem}_corr.png"
    plt.savefig(fpath_out)


def main():

    fp = Path("/home/m/dev/zuy/src/zuy/data/2543v3.tsv")

    plot_correlations_from_tsv(fp)


if __name__ == "__main__":
    main()
