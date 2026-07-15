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
    RahmanFig1.eps  (study-logic workflow diagram, page width)
    RahmanFig2.eps  (OSD-977 cancellous BV/TV by group/mission, page width)
    RahmanFig3.eps  (OSD-342 39-marker screen, page width)
    RahmanFig4.eps  (OSD-342 plasma IGFBP2, column width)
    RahmanFig5.eps  (OSD-342 treatment/recovery specificity check, page width)
    RahmanFig6.eps  (OSD-977 forest plot, 5 metrics x 2 missions, page width)
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
    """Study-logic workflow diagram, question-first design: the central question
    frames two parallel, independent analyses (structural reference from OSD-977,
    molecular candidate from OSD-342), which converge on an explicit caveat (no
    within-animal relationship was measured), then a specificity stress test,
    the current interpretation, and the proposed decisive prospective test.
    Box heights are computed from actual line counts so text never overflows."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    W_REF = 8.0  # coordinate width CHARS_PER_UNIT_BASE was calibrated against
    W = 9.4  # actual working coordinate width; wider than the main flow (0.4-7.6) to fit
    # the side annotation box. Physical page width (PAGE_W inches) stays fixed regardless
    # of W, so 1 data unit = PAGE_W/W inches -- CHARS_PER_UNIT below is rescaled by W/W_REF
    # so text always renders at the same real (inches-based) size no matter how wide W is.
    edge = "#5a594f"
    arrow_color = "#3a3a34"
    _s = W / W_REF  # rescales every data-unit spacing constant so physical (inch) size stays
    # constant regardless of W -- font sizes are fixed in points, so the coordinate-to-inch
    # ratio must stay fixed too, or text overlaps (horizontally and vertically) as W grows.
    title_h = 0.34 * _s
    line_h = 0.255 * _s
    pad_top = 0.12 * _s
    pad_bot = 0.14 * _s
    CHARS_PER_UNIT_BASE = 10.2  # empirical: characters per data unit at body_size 9.3, W=W_REF.
    # Figure width in inches is fixed (PAGE_W); as W (data-unit span) grows, each data unit
    # covers LESS physical space, so fewer characters fit per unit -- inverse of _s, not _s.
    CHARS_PER_UNIT = CHARS_PER_UNIT_BASE / _s
    MARGIN_CHARS = 4 / _s  # safety margin, scaled the same direction as CHARS_PER_UNIT

    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W))  # figsize fixed up after layout
    ax.axis("off")

    def wrap_lines(lines, w, bullets, body_size=9.3):
        """Word-wrap each logical line to fit box width w, returning (text, is_bullet_start)
        pairs so only the first physical line of a wrapped bullet gets the bullet marker --
        continuation lines must not look like separate list items. Wrap width scales
        inversely with font size and always keeps a flat character margin."""
        prefix_len = 3 if bullets else 0
        scale = 9.3 / body_size
        wrap_w = int(max(8, int(w * CHARS_PER_UNIT * scale) - prefix_len - MARGIN_CHARS))
        out = []
        for line in lines:
            if line == "":
                out.append(("", False))
                continue
            wrapped = textwrap.wrap(line, width=wrap_w, break_long_words=False) or [""]
            for i, sub in enumerate(wrapped):
                out.append((sub, i == 0))
        return out

    def title_lines_for(title, w, title_size):
        if not title:
            return []
        scale = 9.3 / title_size
        wrap_w = int(max(6, int(w * CHARS_PER_UNIT * scale) - MARGIN_CHARS))
        out = []
        for part in title.split("\n"):
            out.extend(textwrap.wrap(part, width=wrap_w, break_long_words=False) or [""])
        return out

    def box_height(title_line_count, n_lines):
        h = pad_top + pad_bot
        h += title_line_count * title_h
        h += n_lines * line_h
        return h

    def draw_box(x, y_top, w, title, raw_lines, facecolor, dashed=False,
                 title_size=10.5, body_size=9.3, bullets=True, force_h=None):
        lines = wrap_lines(raw_lines, w, bullets, body_size=body_size)
        t_lines = title_lines_for(title, w, title_size)
        h = force_h if force_h is not None else box_height(len(t_lines), len(lines))
        b = FancyBboxPatch(
            (x, y_top - h), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.3, edgecolor=edge, facecolor=facecolor,
            linestyle=(0, (4.5, 2.5)) if dashed else "solid", zorder=2,
        )
        ax.add_patch(b)
        ty = y_top - pad_top
        for t_line in t_lines:
            ax.text(x + w / 2, ty - title_h * 0.55, t_line, fontsize=title_size, fontweight="bold",
                     ha="center", va="center", zorder=3)
            ty -= title_h
        for text, is_start in lines:
            if text == "":
                ty -= line_h * 0.5
                continue
            prefix = "•  " if (bullets and is_start) else ""
            ax.text(x + w / 2, ty - line_h / 2, prefix + text, fontsize=body_size,
                     ha="center", va="center", zorder=3)
            ty -= line_h
        return {"x": x, "y_top": y_top, "w": w, "h": h}

    def bottom_mid(b):
        return (b["x"] + b["w"] / 2, b["y_top"] - b["h"])

    def top_mid(b):
        return (b["x"] + b["w"] / 2, b["y_top"])

    def right_mid(b):
        return (b["x"] + b["w"], b["y_top"] - b["h"] / 2)

    def left_mid(b):
        return (b["x"], b["y_top"] - b["h"] / 2)

    def arrow(p0, p1):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                             linewidth=1.4, color=arrow_color, zorder=4,
                             shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    gap = 0.30 * _s
    y = 12.0  # running top-of-next-box cursor; trimmed to actual content at the end (started
    # high enough that content never runs past y=0 regardless of W)

    # ---- Question ----
    q = draw_box(0.4, y, 7.2, None,
                 ["QUESTION: Can postflight plasma IGFBP2 indicate cancellous bone loss?"],
                 "#f4f3ee", title_size=11, body_size=10.5, bullets=False)
    y = bottom_mid(q)[1] - gap - 0.55 * _s  # extra clearance for the branching arrows

    # ---- Two parallel boxes (compute both at same height: max of the two) ----
    struct_lines = ["OSD-977 — rat micro-CT", "",
                    "Lower cancellous BV/TV",
                    "STS-62: thinner, more widely spaced trabeculae",
                    "Defines the structural outcome to be predicted"]
    mol_lines = ["OSD-342 — mouse plasma", "",
                 "IGFBP2 signal 36% lower",
                 "Only significant candidate in 39-protein panel",
                 "Robust within this dataset; not proteome-wide significant"]
    side_w = 3.5
    n_struct = len(wrap_lines(struct_lines, side_w, True))
    n_mol = len(wrap_lines(mol_lines, side_w, True))
    n_title2 = len(title_lines_for("STRUCTURAL REFERENCE", side_w, 10.5))
    row2_h = box_height(n_title2, max(n_struct, n_mol))
    structural = draw_box(0.4, y, side_w, "STRUCTURAL REFERENCE", struct_lines, "#dce8f7", force_h=row2_h)
    molecular = draw_box(4.1, y, side_w, "MOLECULAR CANDIDATE", mol_lines, "#e2f0e2", force_h=row2_h)
    row2_bottom = bottom_mid(structural)[1]

    # ---- Side annotation: external concordance check, tied to the molecular candidate ----
    concordance_lines = [
        "Kimura et al. 30-protein signature",
        "23/30 mapped to OSD-342",
        "17/23 concordant",
        "Exploratory evidence",
    ]
    side_x = molecular["x"] + molecular["w"] + 0.35 * _s
    side_w2 = W - side_x - 0.15 * _s
    conc_title = "SIGNATURE\nCONCORDANCE"
    n_conc = len(wrap_lines(concordance_lines, side_w2, True, body_size=7.3))
    n_conc_title = len(title_lines_for(conc_title, side_w2, 8.5))
    conc_h = box_height(n_conc_title, n_conc)
    conc_y_top = molecular["y_top"] - (row2_h - conc_h) / 2  # vertically centered on the molecular box
    concordance = draw_box(side_x, conc_y_top, side_w2, conc_title,
                            concordance_lines, "#fdf1d6", title_size=8.5, body_size=7.3)
    conn = FancyArrowPatch(right_mid(molecular), left_mid(concordance), arrowstyle="-|>",
                            mutation_scale=11, linewidth=1.1, color=arrow_color,
                            linestyle=(0, (3, 2)), zorder=4, shrinkA=1, shrinkB=1)
    ax.add_patch(conn)

    y = row2_bottom - gap - 0.5 * _s

    # ---- Caveat (dashed) ----
    caveat = draw_box(0.7, y, 6.6, None,
                       ["DIFFERENT SPECIES, MISSIONS, AND ANIMALS",
                        "No within-animal IGFBP2-BV/TV relationship was measured"],
                       "#fbfaf6", dashed=True, title_size=10, body_size=9.3, bullets=False)
    y = bottom_mid(caveat)[1] - gap

    # ---- Specificity stress test ----
    stress = draw_box(0.4, y, 7.2, "SPECIFICITY STRESS TEST",
                       ["ACVR2B/Fc also lowered IGFBP2 in ground animals, challenging a "
                        "flight-specific interpretation"],
                       "#f7e3d8", bullets=False)
    y = bottom_mid(stress)[1] - gap

    # ---- Current interpretation ----
    interp = draw_box(0.7, y, 6.6, "CURRENT INTERPRETATION",
                       ["IGFBP2 is a gravity- and treatment-responsive candidate indicator "
                        "— not a validated biomarker."],
                       "#e7e6df", bullets=False)
    y = bottom_mid(interp)[1] - gap

    # ---- Decisive prospective test ----
    decisive = draw_box(0.4, y, 7.2, "DECISIVE PROSPECTIVE TEST",
                         ["Same animals measured longitudinally",
                          "ΔIGFBP2 versus ΔBV/TV",
                          "Independent IGFBP2 assay",
                          "Test the ACVR2B/Fc interaction"],
                         "#dff0d8")
    y_bottom = bottom_mid(decisive)[1]

    # ---- arrows (drawn after all boxes so coordinates are final) ----
    qx, qy = bottom_mid(q)
    clear = 0.02 * _s
    arrow((qx - 1.6, qy), (top_mid(structural)[0], top_mid(structural)[1] + clear))
    arrow((qx + 1.6, qy), (top_mid(molecular)[0], top_mid(molecular)[1] + clear))
    arrow((top_mid(structural)[0], bottom_mid(structural)[1]), (caveat["x"] + caveat["w"] * 0.28, top_mid(caveat)[1] + clear))
    arrow((top_mid(molecular)[0], bottom_mid(molecular)[1]), (caveat["x"] + caveat["w"] * 0.72, top_mid(caveat)[1] + clear))
    arrow(bottom_mid(caveat), top_mid(stress))
    arrow(bottom_mid(stress), top_mid(interp))
    arrow(bottom_mid(interp), top_mid(decisive))

    # ---- trim canvas exactly to content ----
    top_edge = 12.0 + 0.15 * _s
    bottom_edge = y_bottom - 0.15 * _s
    ax.set_xlim(0, W)
    ax.set_ylim(bottom_edge, top_edge)
    content_h = top_edge - bottom_edge
    fig.set_size_inches(PAGE_W, PAGE_W * (content_h / W))

    out = OUT_DIR / "RahmanFig1.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig2():
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
    out = OUT_DIR / "RahmanFig2.eps"
    g.figure.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(g.figure)
    print("Wrote", out)


def export_fig3():
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
    out = OUT_DIR / "RahmanFig3.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig4():
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
    out = OUT_DIR / "RahmanFig4.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig5():
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
    out = OUT_DIR / "RahmanFig5.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig6():
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
    out = OUT_DIR / "RahmanFig6.eps"
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
    export_fig6()


if __name__ == "__main__":
    main()
