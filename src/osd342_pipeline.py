"""
OSD-342: mouse plasma SomaScan proteomics pipeline.

Parses the raw SomaScan Excel export (multi-row header with group labels
and per-mouse sample IDs), screens the candidate bone-remodeling marker
panel (src/bone_markers.py), and runs Welch t-tests + Cohen's d (with
95% CI) + Benjamini-Hochberg FDR correction for Space Flight vs Ground
Control (untreated arms only, to avoid the ACVR2B/Fc drug-treatment
confound).

Run directly:
    python src/osd342_pipeline.py
Outputs:
    results/tables/osd342_bone_marker_stats.csv
    results/figures/osd342_igfbp2_boxplot.png
    results/figures/osd342_marker_screen_summary.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bone_markers import BONE_MARKERS
from effect_size import welch_stats

RAW_DIR = Path("data/raw/OSD-342")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

GROUP_A = "flight uninjected"
GROUP_B = "ground uninjected"


def load_somascan() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Return (annotation_df, numeric_data_df, group_label_per_column)."""
    raw = pd.read_excel(RAW_DIR / "somascan_extracted" / "GSE156321_SomaScan_data_file_raw_data.xlsx", header=None)

    annot = raw.iloc[3:, 0:8].reset_index(drop=True)
    annot.columns = ["SeqId", "SeqIdVersion", "SomaId", "TargetFullName", "Target", "UniProt", "EntrezGeneID", "EntrezGeneSymbol"]

    group_row = raw.iloc[0, 8:91].tolist()
    sample_row = raw.iloc[1, 8:91].tolist()
    groups, last = [], None
    for g in group_row:
        if pd.notna(g):
            last = g
        groups.append(last)

    data = raw.iloc[3:, 8:91].reset_index(drop=True)
    data.columns = range(data.shape[1])
    data = data.apply(pd.to_numeric, errors="coerce")

    # spacer columns between groups have no sample id -- drop them so group
    # membership doesn't get contaminated by an all-NaN column
    valid_idx = [i for i, s in enumerate(sample_row) if pd.notna(s)]
    data = data[valid_idx]
    data.columns = range(data.shape[1])
    groups = [groups[i] for i in valid_idx]

    return annot, data, groups


def compare_bone_markers(annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str) -> pd.DataFrame:
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    rows = []
    seen = set()
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene in BONE_MARKERS and gene not in seen:
            seen.add(gene)
            a_vals = data.iloc[i, a_idx].values.astype(float)
            b_vals = data.iloc[i, b_idx].values.astype(float)
            r = welch_stats(a_vals, b_vals)
            fc = a_vals.mean() / b_vals.mean()
            r.update(
                {
                    "Gene": gene,
                    "Target": annot.loc[i, "Target"],
                    "Role": BONE_MARKERS[gene],
                    f"{group_b}_mean": b_vals.mean(),
                    f"{group_a}_mean": a_vals.mean(),
                    "log2FC": np.log2(fc),
                }
            )
            rows.append(r)

    res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    _, p_adj, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_adj_BH"] = p_adj
    res["significant_FDR05"] = p_adj < 0.05
    cols = ["Gene", "Target", "Role", f"{group_b}_mean", f"{group_a}_mean", "log2FC",
            "mean_diff", "ci_lo", "ci_hi", "cohens_d", "t", "p", "p_adj_BH", "significant_FDR05"]
    return res[cols]


def plot_top_hit(annot: pd.DataFrame, data: pd.DataFrame, groups: list, gene: str, out_path: Path) -> None:
    idx = annot.index[annot["EntrezGeneSymbol"] == gene][0]
    df = pd.DataFrame({"value": data.iloc[idx].values.astype(float), "group": groups})
    df = df[df["group"].isin([GROUP_A, GROUP_B])]

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(data=df, x="group", y="value", order=[GROUP_B, GROUP_A], ax=ax)
    sns.stripplot(data=df, x="group", y="value", order=[GROUP_B, GROUP_A], color="black", size=5, ax=ax)
    ax.set_title(f"OSD-342: plasma {gene}")
    ax.set_xlabel("")
    ax.set_ylabel("SomaScan RFU")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marker_summary(res: pd.DataFrame, out_path: Path) -> None:
    """Ranked bar chart of -log10(p) across the full marker panel, FDR-significant hit highlighted."""
    df = res.sort_values("p", ascending=False).reset_index(drop=True)
    neglogp = -np.log10(df["p"])
    accent, neutral = "#2a78d6", "#c3c2b7"
    colors = [accent if sig else neutral for sig in df["significant_FDR05"]]

    fig, ax = plt.subplots(figsize=(7, 9.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(df))
    ax.barh(y, neglogp, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Gene"], fontsize=9)
    ax.set_xlabel("-log10(p), Flight vs Ground Control", fontsize=10)
    n_sig = int(df["significant_FDR05"].sum())
    top_q = df.loc[df["significant_FDR05"], "p_adj_BH"].min() if n_sig else float("nan")
    ax.set_title(
        f"OSD-342 plasma bone-marker screen (39 candidate proteins screened; n=8 mice/group)\n"
        f"{n_sig} of {len(df)} survive Benjamini-Hochberg FDR correction" + (f" (q={top_q:.3f})" if n_sig else ""),
        fontsize=11, loc="left", pad=14,
    )
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.axvline(-np.log10(0.05), color="#898781", linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    annot, data, groups = load_somascan()
    res = compare_bone_markers(annot, data, groups, GROUP_A, GROUP_B)
    res.to_csv(TABLES_DIR / "osd342_bone_marker_stats.csv", index=False)

    top_gene = res.iloc[0]["Gene"]
    plot_top_hit(annot, data, groups, top_gene, FIGURES_DIR / f"osd342_{top_gene.lower()}_boxplot.png")
    plot_marker_summary(res, FIGURES_DIR / "osd342_marker_screen_summary.png")

    print(res.to_string(index=False))
    print(f"\n{res['significant_FDR05'].sum()} of {len(res)} bone markers survive BH-FDR q<0.05")
    print(f"Top hit: {top_gene} (q={res.iloc[0]['p_adj_BH']:.4f})")


if __name__ == "__main__":
    main()
