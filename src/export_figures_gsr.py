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


def add_sig_bracket(ax, x1, x2, y, tick_h, text, color="#2a2a2a", fontsize=8):
    """Draw a horizontal significance bracket between data-coordinates x1 and x2 at
    height y, with a small vertical tick of height tick_h at each end, and a text
    label centered above it. Used to put the manuscript's own p/q-values directly
    on figure panels rather than leaving significance only in captions/text."""
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y],
            lw=1.0, color=color, clip_on=False, zorder=5)
    ax.text((x1 + x2) / 2, y + tick_h, text, ha="center", va="bottom",
            fontsize=fontsize, color=color, clip_on=False, zorder=5)


def fmt_q_fig(q):
    return f"q={q:.1e}" if q < 0.0001 else f"q={q:.4f}"


def fmt_p_fig(p):
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.4f}"


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
    W = 12.0  # three-column evidence row with a visible molecular-to-concordance connector
    # Physical page width (PAGE_W inches) stays fixed regardless
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
            if bullets:
                # List text is left aligned; wrapped continuation lines receive a small indent.
                tx = x + 0.18 * _s + (0 if is_start else 0.22 * _s)
                ha = "left"
            else:
                tx = x + w / 2
                ha = "center"
            ax.text(tx, ty - line_h / 2, prefix + text, fontsize=body_size,
                     ha=ha, va="center", zorder=3)
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

    def arrow(p0, p1, dashed=False):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                             linewidth=1.4, color=arrow_color, zorder=4,
                             linestyle=(0, (4, 3)) if dashed else "solid",
                             shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    gap = 0.30 * _s
    y = 12.0  # running top-of-next-box cursor; trimmed to actual content at the end (started
    # high enough that content never runs past y=0 regardless of W)

    # ---- Question ----
    q = draw_box(0.4, y, 11.2, None,
                 ["QUESTION: Can postflight plasma IGFBP2 track cancellous bone loss?"],
                 "#f4f3ee", title_size=11, body_size=10.5, bullets=False)
    y = bottom_mid(q)[1] - gap - 0.55 * _s  # extra clearance for the branching arrows

    # ---- Three parallel evidence boxes (compute all at the same height) ----
    struct_lines = ["OSD-977: rat micro-CT",
                    "Flight-associated cancellous deterioration",
                    "Defines the structural outcome"]
    mol_lines = ["OSD-342: mouse plasma",
                 "IGFBP2 signal 36% lower after flight",
                 "Only panel-FDR-significant candidate"]
    conc_lines = ["17/23 proteins directionally concordant",
                  "Direction-adjusted score p=0.027",
                  "Concordant-count p=0.151"]
    side_w = 3.3
    n_struct = len(wrap_lines(struct_lines, side_w, True))
    n_mol = len(wrap_lines(mol_lines, side_w, True))
    n_conc = len(wrap_lines(conc_lines, side_w, True, body_size=8.6))
    title_counts = [len(title_lines_for(t, side_w, 9.2)) for t in
                    ["STRUCTURAL\nREFERENCE", "MOLECULAR\nCANDIDATE", "SUGGESTIVE\nCONCORDANCE"]]
    row2_h = box_height(max(title_counts), max(n_struct, n_mol, n_conc))
    structural = draw_box(0.4, y, side_w, "STRUCTURAL\nREFERENCE", struct_lines,
                          "#dce8f7", title_size=9.2, body_size=8.6, force_h=row2_h)
    molecular = draw_box(4.35, y, side_w, "MOLECULAR\nCANDIDATE", mol_lines,
                         "#e2f0e2", title_size=9.2, body_size=8.6, force_h=row2_h)
    concordance = draw_box(8.3, y, side_w, "SUGGESTIVE\nCONCORDANCE", conc_lines,
                           "#fff2cc", title_size=9.2, body_size=8.6, force_h=row2_h)
    row2_bottom = bottom_mid(structural)[1]

    y = row2_bottom - gap - 0.5 * _s

    # ---- Caveat (dashed) ----
    caveat = draw_box(0.7, y, 10.6, None,
                       ["NO ANIMAL-LEVEL PLASMA-BONE PAIRING",
                        "The public data cannot test IGFBP2 versus BV/TV directly"],
                       "#fbfaf6", dashed=True, title_size=10, body_size=9.3, bullets=False)
    y = bottom_mid(caveat)[1] - gap

    # ---- Specificity stress test ----
    stress = draw_box(0.4, y, 11.2, "SPECIFICITY STRESS TEST",
                       ["ACVR2B/Fc also lowered IGFBP2 in ground animals"],
                       "#f7e3d8", bullets=False)
    y = bottom_mid(stress)[1] - gap

    # ---- Current interpretation ----
    interp = draw_box(0.7, y, 10.6, "CURRENT INTERPRETATION",
                       ["Candidate signal; specificity and bone association remain untested"],
                       "#e7e6df", bullets=False)
    y = bottom_mid(interp)[1] - gap

    # ---- Decisive prospective test ----
    decisive = draw_box(0.4, y, 11.2, "DECISIVE PROSPECTIVE TEST",
                          ["Measure IGFBP2 and BV/TV in the same animals",
                           "Confirm IGFBP2 with an independent assay",
                           "Test whether treatment modifies the association"],
                         "#dff0d8")
    y_bottom = bottom_mid(decisive)[1]

    # ---- arrows (drawn after all boxes so coordinates are final) ----
    qx, qy = bottom_mid(q)
    clear = 0.02 * _s
    arrow((qx - 2.4 * _s, qy), (top_mid(structural)[0], top_mid(structural)[1] + clear))
    arrow((qx, qy), (top_mid(molecular)[0], top_mid(molecular)[1] + clear))
    # External concordance is supporting evidence for the molecular result,
    # not a third independent branch from the study question.
    arrow((right_mid(molecular)[0] + clear, right_mid(molecular)[1]),
          (left_mid(concordance)[0] - clear, left_mid(concordance)[1]), dashed=True)
    arrow(bottom_mid(structural), (caveat["x"] + caveat["w"] * 0.18, top_mid(caveat)[1] + clear))
    arrow(bottom_mid(molecular), (caveat["x"] + caveat["w"] * 0.50, top_mid(caveat)[1] + clear))
    arrow(bottom_mid(concordance),
          (caveat["x"] + caveat["w"] * 0.82, top_mid(caveat)[1] + clear),
          dashed=True)
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
    fig.savefig(OUT_DIR / "png_previews_for_review" / "RahmanFig1.png", dpi=300,
                bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print("Wrote", out)


def export_fig2():
    merged = m1.load_merged()
    sub = merged[merged["Tissue"] == "Femur Head"].dropna(subset=["bone_volume_over_tissue_volumn"])
    metric = "bone_volume_over_tissue_volumn"
    _summary, pw_cancellous, _pw_cortical, _anova = m1.run_all_stats(merged)
    pw = pw_cancellous[pw_cancellous["Metric"] == metric]

    g = sns.catplot(
        data=sub, x="Group", y=metric, col="Mission",
        kind="box", order=m1.GROUP_ORDER, height=PAGE_W / 2, aspect=1.0,
        showfliers=False,
    )
    g.set_axis_labels("", "Cancellous BV/TV (%)")
    g.set_titles("{col_name}")

    x_pos = {grp: i for i, grp in enumerate(m1.GROUP_ORDER)}
    # (comparison, tier): adjacent-group contrasts share the lower tier since their
    # brackets don't overlap in x; the full-span contrast sits above both.
    bracket_plan = [
        ("Space Flight vs Ground Control", 1),
        ("Ground Control vs Basal Control", 1),
        ("Space Flight vs Basal Control", 2),
    ]

    for mission, ax in zip(g.col_names, g.axes.flat):
        mission_sub = sub[sub["Mission"] == mission]
        sns.stripplot(
            data=mission_sub, x="Group", y=metric, order=m1.GROUP_ORDER,
            color="black", size=4, jitter=0.15, ax=ax,
        )
        ax.tick_params(labelsize=10)
        ax.set_xticks(range(len(m1.GROUP_ORDER)),
                      ["Basal\nControl", "Ground\nControl", "Space\nFlight"])
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)

        ymax = mission_sub[metric].max()
        ymin = mission_sub[metric].min()
        yrange = ymax - ymin
        step = yrange * 0.16
        base = ymax + yrange * 0.10
        mission_pw = pw[pw["Mission"] == mission]
        max_tier = 1
        for comparison, tier in bracket_plan:
            row = mission_pw[mission_pw["Comparison"] == comparison]
            if row.empty:
                continue
            q = row["q_BH"].iloc[0]
            a, b = comparison.split(" vs ")
            x1, x2 = sorted([x_pos[a], x_pos[b]])
            y = base + (tier - 1) * step
            color = "#2a78d6" if q < 0.05 else "#5a5850"
            add_sig_bracket(ax, x1, x2, y, step * 0.22, fmt_q_fig(q), color=color)
            max_tier = max(max_tier, tier)
        ax.set_ylim(top=base + max_tier * step + step * 0.6)

    g.figure.set_size_inches(PAGE_W, PAGE_W / 1.9)
    g.figure.set_facecolor("white")
    out = OUT_DIR / "RahmanFig2.eps"
    g.figure.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    g.figure.savefig(OUT_DIR / "RahmanFig2.png", dpi=300, bbox_inches="tight",
                     facecolor="white", transparent=False)
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
    annot, data, groups, sample_ids = m2.load_somascan()
    idx = annot.index[annot["EntrezGeneSymbol"] == "IGFBP2"][0]
    import pandas as pd
    df = pd.DataFrame({"value": data.iloc[idx].values.astype(float), "group": groups,
                       "sample_id": sample_ids})
    df = df[df["group"].isin([m2.GROUP_A, m2.GROUP_B])]
    df["group"] = df["group"].map({m2.GROUP_B: "Ground", m2.GROUP_A: "Flight"})
    df["id_block"] = df["sample_id"].str.extract(r"(\d+)$")[0].astype(int).map(
        lambda n: "IDs 1-4" if n <= 4 else "IDs 9-12"
    )

    res = m2.compare_bone_markers(annot, data, groups, m2.GROUP_A, m2.GROUP_B, scale="log2", dedup=True)
    q = res.loc[res["Gene"] == "IGFBP2", "p_adj_BH"].iloc[0]

    fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 1.25))
    sns.boxplot(data=df, x="group", y="value", order=["Ground", "Flight"], ax=ax,
                showfliers=False)
    markers = {"IDs 1-4": "o", "IDs 9-12": "^"}
    x_pos = {"Ground": 0, "Flight": 1}
    offsets = {"IDs 1-4": -0.055, "IDs 9-12": 0.055}
    for block, marker in markers.items():
        sub = df[df["id_block"] == block]
        ax.scatter([x_pos[g] + offsets[block] for g in sub["group"]], sub["value"],
                   marker=marker, s=30, color="black", edgecolor="white", linewidth=0.4,
                   label=block, zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("IGFBP2-associated SomaScan RFU", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", title="Numeric ID block",
              title_fontsize=8)

    ymax, ymin = df["value"].max(), df["value"].min()
    yrange = ymax - ymin
    y = ymax + yrange * 0.10
    add_sig_bracket(ax, 0, 1, y, yrange * 0.03, f"BH {fmt_q_fig(q)}",
                     color="#2a78d6" if q < 0.05 else "#5a5850")
    ax.set_ylim(top=y + yrange * 0.18)

    fig.tight_layout()
    out = OUT_DIR / "RahmanFig4.eps"
    fig.savefig(out, format="eps", bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(OUT_DIR / "RahmanFig4.png", dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
    fig.savefig(OUT_DIR / "png_previews_for_review" / "RahmanFig4.png", dpi=300,
                bbox_inches="tight", facecolor="white", transparent=False)
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

    tr = m2.treatment_recovery_models(annot, data, groups, gene="IGFBP2", scale="log2")

    def term_p(arm, term, cov_type="nonrobust"):
        row = tr[(tr["arm"] == arm) & (tr["cov_type"] == cov_type) & (tr["term"] == term)]
        return row["p"].iloc[0] if len(row) else None

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, PAGE_W * 0.68))

    shared_ylim = (10000, 80000)
    panel_specs = [
        (axes[0], a_df, ["Ground", "Flight"], "Early postflight", "A", "immediate_postflight", "Flight", True),
        (axes[1], b_df, ["Vivarium", "Flight recovery"], "14-day recovery", "B", "recovery_14day", "Flight recovery", False),
    ]
    for ax, sub, order, title, letter, arm, cond_alt, show_treatment_bracket in panel_specs:
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

        if arm == "immediate_postflight":
            # Methods/Results: immediate postflight arm uses the log2 two-factor model.
            # Label precisely -- this reference-coded term is the untreated flight-vs-ground
            # contrast, not a generic "condition effect" (the treatment term below is the
            # ground-animals treatment contrast, not a generic "treatment effect").
            p_cond = term_p(arm, f"C(condition)[T.{cond_alt}]")
            cond_label = f"Untreated flight vs ground\n{fmt_p_fig(p_cond)}" if p_cond is not None else None
        else:
            # Methods: recovery comparisons use a raw-scale Welch test on the
            # untreated-only subgroups (exploratory specificity check), not the
            # log2 two-factor model -- matches the manuscript's "-10.7%, p=0.30".
            viv = df.loc[df["group"] == "vivarium uninjected 2 weeks", "value"].values
            frec = df.loc[df["group"] == "flight recovery uninjected", "value"].values
            p_cond = m2.welch_stats(frec, viv)["p"]
            cond_label = f"Untreated flight-recovery vs vivarium\n{fmt_p_fig(p_cond)}"
        if p_cond is not None:
            color = "#2a78d6" if p_cond < 0.05 else "#5a5850"
            add_sig_bracket(ax, -0.2, 0.8, 62000, 1800, cond_label, color=color, fontsize=6.5)

        if show_treatment_bracket:
            p_treat = term_p(arm, "C(treatment)[T.ACVR2B/Fc]")
            if p_treat is not None:
                color = "#2a78d6" if p_treat < 0.05 else "#5a5850"
                add_sig_bracket(ax, -0.2, 0.2, 52000, 1500,
                                f"Treatment in ground animals\n{fmt_p_fig(p_treat)}",
                                color=color, fontsize=6)
            interaction_term = f"C(condition)[T.{cond_alt}]:C(treatment)[T.ACVR2B/Fc]"
            p_int_ord = term_p(arm, interaction_term, "nonrobust")
            p_int_hc3 = term_p(arm, interaction_term, "HC3")
            if p_int_ord is not None:
                ax.text(0.03, 0.90,
                        f"Flight×treatment interaction:\n{fmt_p_fig(p_int_ord)}"
                        + (f" (HC3 {fmt_p_fig(p_int_hc3)})" if p_int_hc3 is not None else ""),
                        transform=ax.transAxes, fontsize=6.5, color="#5a5850", ha="left", va="top")

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
    # Standardize the Welch-Satterthwaite mean-difference CI so the interval
    # and the reported Welch test have the same zero-crossing decision.
    df["ci_lo"] = df["hedges_g_welch_ci_lo"]
    df["ci_hi"] = df["hedges_g_welch_ci_hi"]
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
    fig.savefig(OUT_DIR / "RahmanFig6.png", dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
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
