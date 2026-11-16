"""
Visualise the data for predicting undrained shear strength (su) at the five
target boreholes (Benchmark Problem #1 of Otake et al. 2025).

Files (one row per test record; columns as in the notes):
    BM1_five_borehole_data.xlsx       the 5 target boreholes B1-B5. LL, PL, w are
                                      known at every depth; su is OBSERVED at one
                                      depth per borehole only. The su cells shaded
                                      yellow are the hidden true values, kept for
                                      checking predictions - do not use them to
                                      make a prediction.
    BM1_Cluster_BID_4_minus_5BHs.xlsx the other 46 boreholes of the verification
                                      site with a `cluster` label 1-5; target Bn
                                      lies in cluster n ("Cluster-BID/4")
    BM1_Local_BID_V_4_minus_5BHs.xlsx the same 46 boreholes without the label
                                      ("Local-BID-V/4")
    drilling_dates.csv                year/month each borehole was drilled

Figures written to ./figures:
    1_plan_view.png             boreholes coloured by cluster, targets as stars
    2_target_B1.png ... B5      profiles of each target over its cluster's data
    3_cross_plots.png           su against depth, LL, PL, w (log su)

Set SHOW_HIDDEN = True to draw the hidden su values as open circles (for
marking / checking only).

Requires: pandas, openpyxl, matplotlib   ->   pip install pandas openpyxl matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook

HERE = Path(__file__).parent
TARGET_PATH = HERE / "BM1_five_borehole_data.xlsx"
CLUSTER_PATH = HERE / "BM1_Cluster_BID_4_minus_5BHs.xlsx"
LOCAL_V_PATH = HERE / "BM1_Local_BID_V_4_minus_5BHs.xlsx"
DATES_PATH = HERE / "drilling_dates.csv"
OUT_DIR = HERE / "figures"

SHOW_HIDDEN = False
VERTICAL = "elevation"  # or "z_depth"
TARGETS = {682: "B1", 905: "B2", 845: "B3", 712: "B4", 685: "B5"}  # target -> cluster n of Bn
COLS = ["global_id", "borehole_id", "x", "y", "z_depth", "elevation", "LL", "PL", "w", "su"]
PARAMS = [("LL", "LL (%)"), ("PL", "PL (%)"), ("w", "w (%)"), ("su", "$s_u$ (kPa)")]
CLUSTER_COLOURS = dict(zip(range(1, 6), plt.cm.tab10.colors))


def load_targets():
    """Target records with `su_observed` = True where su is given (cell not shaded)."""
    df = pd.read_excel(TARGET_PATH, usecols=COLS)
    ws = load_workbook(TARGET_PATH).active
    su_col = COLS.index("su") + 1
    df["su_observed"] = [ws.cell(r, su_col).fill.fill_type is None for r in range(2, len(df) + 2)]
    df["target"] = df["borehole_id"].map(TARGETS)
    df["cluster"] = df["target"].str[1].astype(int)
    return df


def load_bid():
    cluster = pd.read_excel(CLUSTER_PATH)
    local_v = pd.read_excel(LOCAL_V_PATH)
    return cluster, local_v


def _vlabel():
    return "Elevation (m)" if VERTICAL == "elevation" else "Depth (m)"


def _set_vaxis(ax):
    ax.set_ylabel(_vlabel())
    if VERTICAL == "z_depth":
        ax.invert_yaxis()


def plot_plan(cluster, targets):
    bh = cluster.groupby("borehole_id").first().reset_index()
    tb = targets.groupby("borehole_id").first().reset_index()
    fig, ax = plt.subplots(figsize=(7, 7))
    for c, col in CLUSTER_COLOURS.items():
        s = bh[bh["cluster"] == c]
        ax.scatter(s["x"], s["y"], color=col, s=35, edgecolor="k", lw=0.4, label=f"C{c} ({len(s)} BH)")
        t = tb[tb["cluster"] == c]
        ax.scatter(t["x"], t["y"], color=col, marker="*", s=350, edgecolor="k", lw=0.8)
    for _, r in tb.iterrows():
        ax.annotate(f"{r['target']} ({r['borehole_id']})", (r["x"], r["y"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Verification site: clusters C1-C5 and target boreholes (stars)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_target(cluster, local_v, targets, bh):
    t = targets[targets["borehole_id"] == bh]
    n = t["cluster"].iloc[0]
    c = cluster[cluster["cluster"] == n]
    fig, axes = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
    for ax, (col, label) in zip(axes, PARAMS):
        ax.scatter(local_v[col], local_v[VERTICAL], s=5, c="lightgrey", label="Local-BID-V/4")
        ax.scatter(c[col], c[VERTICAL], s=10, color=CLUSTER_COLOURS[n], alpha=0.7, label=f"Cluster-BID/4 (C{n})")
        if col == "su":
            obs = t[t["su_observed"]]
            ax.scatter(obs["su"], obs[VERTICAL], s=90, c="k", marker="D", zorder=5, label="observed $s_u$")
            if SHOW_HIDDEN:
                hid = t[~t["su_observed"]]
                ax.scatter(hid["su"], hid[VERTICAL], s=30, facecolor="none", edgecolor="k",
                           zorder=4, label="hidden $s_u$ (true)")
            else:
                for y in t.loc[~t["su_observed"], VERTICAL]:
                    ax.axhline(y, color="k", lw=0.4, ls=":", alpha=0.6)
        else:
            ax.plot(t[col], t[VERTICAL], "ko-", ms=4, lw=1, zorder=5, label=f"target {bh}")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _set_vaxis(axes[0])
    r = t.iloc[0]
    note = "" if SHOW_HIDDEN else "   (dotted lines: depths where $s_u$ is to be predicted)"
    fig.suptitle(f"{r['target']} = borehole {bh}  (x = {r['x']:.0f} m, y = {r['y']:.0f} m), "
                 f"{len(t)} depths, 1 observed $s_u${note}")
    fig.tight_layout()
    return fig


def plot_cross(cluster, targets):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
    obs = targets[targets["su_observed"]]
    for ax, col in zip(axes, ["z_depth", "LL", "PL", "w"]):
        for n, colour in CLUSTER_COLOURS.items():
            s = cluster[cluster["cluster"] == n]
            ax.scatter(s[col], s["su"], s=8, color=colour, alpha=0.6, label=f"C{n}")
        ax.scatter(obs[col], obs["su"], s=90, c="k", marker="D", label="observed target $s_u$")
        ax.set_xlabel({"z_depth": "Depth (m)"}.get(col, f"{col} (%)"))
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("$s_u$ (kPa, log scale)")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Cross-correlation of $s_u$ with depth and index properties (Cluster-BID/4)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    targets = load_targets()
    cluster, local_v = load_bid()
    print(f"targets: {targets['borehole_id'].nunique()} boreholes, {len(targets)} records, "
          f"{targets['su_observed'].sum()} observed su")
    print(f"Cluster-BID/4 / Local-BID-V/4: {cluster['borehole_id'].nunique()} boreholes, {len(cluster)} records")
    print(targets[targets["su_observed"]][["target", "borehole_id", "z_depth", "elevation", "su"]]
          .to_string(index=False), "\n")

    OUT_DIR.mkdir(exist_ok=True)
    figs = [("1_plan_view", plot_plan(cluster, targets))]
    figs += [(f"2_target_{TARGETS[bh]}", plot_target(cluster, local_v, targets, bh)) for bh in TARGETS]
    figs += [("3_cross_plots", plot_cross(cluster, targets))]
    for name, fig in figs:
        fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved -> figures/{name}.png")
