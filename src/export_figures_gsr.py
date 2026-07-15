"""
Export final submission-ready figures for the GSR Hypothesis Paper.

GSR's Instructions to Authors require: figures submitted separately (not
embedded in the manuscript), as .tif or .eps files, illustration text no
smaller than 10-point Helvetica at final print size (column = 8 cm, page
= 16.5 cm width), filenames as <FirstAuthorLastName>Fig<N>.<ext>. Vector
EPS is used here (rather than 1200-dpi TIFF) since these are simple boxplot
and bar-chart figures with no raster content -- EPS meets the same
line-art resolution requirement without the very large file sizes that a
1200-dpi TIFF would produce for a 39-row bar chart.

Run directly:
    python src/export_figures_gsr.py
Outputs (results/figures/gsr_submission/):
    RahmanFig1.eps  (OSD-977 cancellous BV/TV by group/mission, page width)
    RahmanFig2.eps  (OSD-342 39-marker screen, page width)
    RahmanFig3.eps  (OSD-342 plasma IGFBP2, column width)
    RahmanFig4.eps  (OSD-342 treatment/recovery specificity check, page width)
    RahmanFig5.eps  (OSD-977 forest plot, 5 metrics x 2 missions, page width)
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import osd342_pipeline as m2
import osd977_pipeline as m1

OUT_DIR = Path("results/figures/gsr_submission")

# GSR final print sizes, converted cm -> inches
COLUMN_W = 8 / 2.54
PAGE_W = 16.5 / 2.54

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10  # >= 10pt requirement at final size
matplotlib.rcParams["ps.fonttype"] = 42  # embed as TrueType, not Type 3, for compatibility
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["savefig.transparent"] = False
# NOTE: GSR also asks for color figures saved as CMYK. Matplotlib's EPS backend
# writes RGB only; there is no Ghostscript available in this environment to do
# an RGB->CMYK conversion pass, so these EPS files remain RGB. Flagged for the
# author to confirm with the editorial office whether RGB EPS is acceptable, or
# to run the files through Ghostscript/Acrobat for CMYK conversion before
# upload.


def export_fig1():
    merged = m1.load_merged()
    sub = merged[merged["Tissue"] == "Femur Head"].dropna(subset=["bone_volume_over_tissue_volumn"])
    g = sns.catplot(
        data=sub, x="Group", y="bone_volume_over_tissue_volumn", col="Mission",
        kind="box", order=m1.GROUP_ORDER, height=PAGE_W / 2, aspect=1.0,
        showfliers=False,
    )
    g.set_axis_labels("", "Cancellous BV/TV (%)")
    g.set_titles("{col_name}")
    for mission, ax in zip(g.col_names, g.axes.flat):
        mission_sub = sub[sub["Mission"] == mission]
        sns.stripplot(
            data=mission_sub, x="Group", y="bone_volume_over_tissue_volumn", order=m1.GROUP_ORDER,
            color="black", size=4, jitter=0.15, ax=ax,
        )
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
    g.figure.set_size_inches(PAGE_W, PAGE_W / 2.1)
    g.figure.set_facecolor("white")
    out = OUT_DIR / "RahmanFig1.eps"
    g.figure.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(g.figure)
    print("Wrote", out)


def export_fig2():
    annot, data, groups, _sample_ids = m2.load_somascan()
    res = m2.compare_bone_markers(annot, data, groups, m2.GROUP_A, m2.GROUP_B, scale="log2", dedup=True)
    df = res.sort_values("p", ascending=False).reset_index(drop=True)
    neglogp = -np.log10(df["p"])
    accent, neutral = "#2a78d6", "#c3c2b7"
    colors = [accent if sig else neutral for sig in df["significant_FDR05"]]

    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W * 1.35))
    y = np.arange(len(df))
    ax.barh(y, neglogp, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Gene"], fontsize=10)
    ax.set_xlabel("-log10(p), log2 RFU, Flight vs Ground Control", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    xref = -np.log10(0.05)
    ax.axvline(xref, color="#898781", linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    ax.text(xref + 0.05, len(df) - 1.2, "Unadjusted p=0.05 reference", fontsize=8,
            color="#5a5850", ha="left", va="top")
    igfbp2_row = df.index[df["Gene"] == "IGFBP2"][0]
    igfbp2_q = df.loc[igfbp2_row, "p_adj_BH"]
    ax.text(neglogp[igfbp2_row] + 0.08, igfbp2_row, f"BH q={igfbp2_q:.3f}", fontsize=8,
            color=accent, ha="left", va="center", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "RahmanFig2.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig3():
    annot, data, groups, _sample_ids = m2.load_somascan()
    idx = annot.index[annot["EntrezGeneSymbol"] == "IGFBP2"][0]
    import pandas as pd
    df = pd.DataFrame({"value": data.iloc[idx].values.astype(float), "group": groups})
    df = df[df["group"].isin([m2.GROUP_A, m2.GROUP_B])]
    df["group"] = df["group"].map({m2.GROUP_B: "Ground", m2.GROUP_A: "Flight"})

    fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 1.25))
    sns.boxplot(data=df, x="group", y="value", order=["Ground", "Flight"], ax=ax,
                showfliers=False)
    sns.stripplot(data=df, x="group", y="value", order=["Ground", "Flight"], color="black", size=4, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("IGFBP2-associated SomaScan RFU", fontsize=10)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    out = OUT_DIR / "RahmanFig3.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig4():
    """Two-panel treatment/recovery figure for IGFBP2, testing specificity of the flight signal."""
    import pandas as pd

    annot, data, groups, _sample_ids = m2.load_somascan()
    idx = annot.index[annot["EntrezGeneSymbol"] == "IGFBP2"][0]
    df = pd.DataFrame({"value": data.iloc[idx].values.astype(float), "group": groups})

    panel_a_map = {
        "ground uninjected": ("Ground", "Untreated"),
        "flight uninjected": ("Flight", "Untreated"),
        "ground + ACVR2B/Fc": ("Ground", "ACVR2B/Fc"),
        "flight + ACVR2B/Fc": ("Flight", "ACVR2B/Fc"),
    }
    panel_b_map = {
        "vivarium uninjected 2 weeks": ("Vivarium", "Untreated"),
        "flight recovery uninjected": ("Flight recovery", "Untreated"),
        "vivarium + ACVR2B/Fc 2 weeks": ("Vivarium", "ACVR2B/Fc"),
        "flight recovery + ACVR2B/Fc": ("Flight recovery", "ACVR2B/Fc"),
    }

    def prep(mapping):
        sub = df[df["group"].isin(mapping)].copy()
        sub["condition"] = sub["group"].map(lambda g: mapping[g][0])
        sub["treatment"] = sub["group"].map(lambda g: mapping[g][1])
        return sub

    a_df = prep(panel_a_map)
    b_df = prep(panel_b_map)

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, PAGE_W * 0.62))

    shared_ylim = (10000, 67000)
    for ax, sub, order, title, letter in [
        (axes[0], a_df, ["Ground", "Flight"], "Early postflight", "A"),
        (axes[1], b_df, ["Vivarium", "Flight recovery"], "14-day recovery", "B"),
    ]:
        sns.boxplot(data=sub, x="condition", y="value", hue="treatment", order=order,
                    hue_order=["Untreated", "ACVR2B/Fc"], ax=ax, showfliers=False,
                    palette={"Untreated": "#c3c2b7", "ACVR2B/Fc": "#2a78d6"})
        sns.stripplot(data=sub, x="condition", y="value", hue="treatment", order=order,
                      hue_order=["Untreated", "ACVR2B/Fc"], dodge=True, color="black", size=3,
                      ax=ax, legend=False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(labelsize=9)
        ax.set_ylim(*shared_ylim)
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        if ax is axes[0]:
            ax.set_ylabel("Plasma IGFBP2-associated\nSomaScan RFU", fontsize=10)
            ax.legend(fontsize=8, title=None, loc="upper right")
        else:
            ax.set_ylabel("")
            ax.legend_.remove() if ax.legend_ else None

    fig.tight_layout()
    out = OUT_DIR / "RahmanFig4.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig5():
    """Forest plot: standardized flight-vs-ground effect sizes for all 5 cancellous metrics, both missions.
    Uses the same 30-comparison-family BH-FDR q-values reported in the manuscript (via
    m1.run_all_stats), so significance markers here always match the text."""
    metric_labels = {
        "bone_volume_over_tissue_volumn": "BV/TV",
        "connectivity_density": "Connectivity density",
        "trabecular_number": "Trabecular number",
        "trabecular_thickness": "Trabecular thickness",
        "trabecular_spacing": "Trabecular spacing",
    }

    merged = m1.load_merged()
    _summary, pw_cancellous, _pw_cortical, _anova = m1.run_all_stats(merged)
    df = pw_cancellous[pw_cancellous["Comparison"] == "Space Flight vs Ground Control"].copy()
    df["label"] = df["Metric"].map(metric_labels)
    df["mission"] = df["Mission"]
    df["g"] = df["hedges_g"]
    df["ci_lo"] = df["hedges_g_ci_lo"]
    df["ci_hi"] = df["hedges_g_ci_hi"]
    df["sig"] = df["q_BH"] < 0.05

    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W * 0.75))
    y_labels = list(metric_labels.values())[::-1]
    offset = {"STS-57": 0.15, "STS-62": -0.15}
    color = {"STS-57": "#2a78d6", "STS-62": "#e34948"}
    marker = {"STS-57": "o", "STS-62": "s"}
    for mission in ["STS-57", "STS-62"]:
        sub = df[df["mission"] == mission]
        y_pos = [y_labels.index(l) + offset[mission] for l in sub["label"]]
        ax.errorbar(sub["g"], y_pos, xerr=[sub["g"] - sub["ci_lo"], sub["ci_hi"] - sub["g"]],
                    fmt="none", ecolor=color[mission], capsize=3, zorder=2)
        for x, y, sig in zip(sub["g"], y_pos, sub["sig"]):
            ax.plot(x, y, marker=marker[mission], color=color[mission],
                    markerfacecolor=color[mission] if sig else "white",
                    markeredgecolor=color[mission], markeredgewidth=1.2, markersize=6, zorder=3)
        # dummy handle for the legend showing mission color+shape only
        ax.plot([], [], marker=marker[mission], color=color[mission], linestyle="none",
                 markersize=6, label=mission)
    ax.axvline(0, color="#898781", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Hedges' g (Flight vs Ground Control)", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    xlim = ax.get_xlim()
    ax.text(xlim[0], -0.9, "Lower in flight", fontsize=8, color="#5a5850", ha="left", va="top")
    ax.text(xlim[1], -0.9, "Higher in flight", fontsize=8, color="#5a5850", ha="right", va="top")
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    sig_handles = [
        Line2D([0], [0], marker="o", color="#5a5850", markerfacecolor="#5a5850", linestyle="none", markersize=6, label="BH q<0.05"),
        Line2D([0], [0], marker="o", color="#5a5850", markerfacecolor="white", markeredgewidth=1.2, linestyle="none", markersize=6, label="BH q≥0.05"),
    ]
    ax.legend(handles=handles + sig_handles, fontsize=8, loc="lower left", ncol=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "RahmanFig5.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_fig1()
    export_fig2()
    export_fig3()
    export_fig4()
    export_fig5()


if __name__ == "__main__":
    main()
