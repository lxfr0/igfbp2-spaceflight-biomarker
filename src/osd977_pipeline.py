"""
OSD-977: rat femur microCT bone-microarchitecture pipeline.

Merges the ISA sample sheet (group/mission/sex/tissue labels) with the
transformed microCT measurement CSV, then runs group comparisons
(Baseline vs Ground Control vs Space Flight) per mission and tissue,
with Cohen's d + 95% CI per comparison and Benjamini-Hochberg FDR
correction applied across all 12 pairwise tests (2 missions x 2
skeletal metrics x 3 group contrasts) -- the same correction family
size used for the OSD-342 marker screen.

Run directly:
    python src/osd977_pipeline.py
Outputs:
    data/processed/merged_OSD-977.csv
    results/tables/osd977_group_stats.csv
    results/tables/osd977_pairwise_effectsizes.csv
    results/figures/osd977_bvtv_by_mission.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from effect_size import welch_stats

RAW_DIR = Path("data/raw/OSD-977")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

GROUP_ORDER = ["Basal Control", "Ground Control", "Space Flight"]
NUMERIC_COLS = [
    "bone_volume_over_tissue_volumn",
    "connectivity_density",
    "trabecular_number",
    "trabecular_thickness",
    "trabecular_spacing",
    "cortical_thickness",
]


def load_merged() -> pd.DataFrame:
    """Merge the ISA sample sheet with the microCT measurement CSV on Sample Name."""
    meta = pd.read_csv(RAW_DIR / "metadata_extracted" / "s_OSD-977.txt", sep="\t")
    meta = meta[
        [
            "Sample Name",
            "Characteristics[Mission]",
            "Factor Value[Sex]",
            "Factor Value[Material Type]",
            "Factor Value[Spaceflight]",
        ]
    ]
    meta.columns = ["Sample Name", "Mission", "Sex", "Tissue", "Group"]

    data = pd.read_csv(RAW_DIR / "LSDS-175_microCT_Gamboa2021_uCT_TRANSFORMED.csv")
    for c in NUMERIC_COLS:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data[["Sample Name"] + NUMERIC_COLS]

    merged = pd.merge(meta, data, on="Sample Name", how="inner")
    merged["Group"] = pd.Categorical(merged["Group"], categories=GROUP_ORDER, ordered=True)
    return merged


def group_stats(merged: pd.DataFrame, metric: str, tissue: str, mission: str) -> pd.DataFrame:
    """One-way ANOVA + pairwise Welch t-tests for a metric within one mission/tissue."""
    sub = merged[(merged["Mission"] == mission) & (merged["Tissue"] == tissue)].dropna(subset=[metric])
    groups = {g: sub[sub["Group"] == g][metric].values for g in GROUP_ORDER}

    rows = []
    present = [v for v in groups.values() if len(v)]
    f, p_anova = stats.f_oneway(*present) if len(present) > 1 else (float("nan"), float("nan"))
    for g, vals in groups.items():
        rows.append(
            {
                "Mission": mission,
                "Tissue": tissue,
                "Metric": metric,
                "Group": g,
                "n": len(vals),
                "mean": vals.mean() if len(vals) else float("nan"),
                "sd": vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                "ANOVA_F": f,
                "ANOVA_p": p_anova,
            }
        )
    pairwise = []
    pairs = [("Space Flight", "Ground Control"), ("Space Flight", "Basal Control"), ("Ground Control", "Basal Control")]
    for a, b in pairs:
        if len(groups[a]) > 1 and len(groups[b]) > 1:
            r = welch_stats(groups[a], groups[b])
            r.update({"Mission": mission, "Tissue": tissue, "Metric": metric, "Comparison": f"{a} vs {b}"})
            pairwise.append(r)
    pairwise_df = pd.DataFrame(pairwise)
    if len(pairwise_df):
        cols = ["Mission", "Tissue", "Metric", "Comparison", "n1", "n2", "mean_diff", "ci_lo", "ci_hi", "cohens_d", "t", "p"]
        pairwise_df = pairwise_df[cols]
    return pd.DataFrame(rows), pairwise_df


def run_all_stats(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary stats per group, plus pairwise tests with BH-FDR correction applied
    across all 12 comparisons (2 missions x 2 metrics x 3 pairwise contrasts)."""
    summaries, pairwises = [], []
    for mission in merged["Mission"].dropna().unique():
        summaries.append(group_stats(merged, "bone_volume_over_tissue_volumn", "Femur Head", mission)[0])
        pairwises.append(group_stats(merged, "bone_volume_over_tissue_volumn", "Femur Head", mission)[1])
        summaries.append(group_stats(merged, "cortical_thickness", "diaphysis of femur", mission)[0])
        pairwises.append(group_stats(merged, "cortical_thickness", "diaphysis of femur", mission)[1])

    pairwise_all = pd.concat(pairwises, ignore_index=True)
    _, p_adj, _, _ = multipletests(pairwise_all["p"], method="fdr_bh")
    pairwise_all["q_BH"] = p_adj

    return pd.concat(summaries, ignore_index=True), pairwise_all


def plot_bvtv(merged: pd.DataFrame, out_path: Path) -> None:
    sub = merged[merged["Tissue"] == "Femur Head"].dropna(subset=["bone_volume_over_tissue_volumn"])
    g = sns.catplot(
        data=sub, x="Group", y="bone_volume_over_tissue_volumn", col="Mission",
        kind="box", order=GROUP_ORDER, height=4, aspect=1.1,
    )
    g.set_axis_labels("", "Cancellous BV/TV (%)")
    g.figure.suptitle("OSD-977: Femur head cancellous bone volume fraction", y=1.05)
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    merged = load_merged()
    merged.to_csv(PROCESSED_DIR / "merged_OSD-977.csv", index=False)

    summary, pairwise = run_all_stats(merged)
    summary.to_csv(TABLES_DIR / "osd977_group_stats.csv", index=False)
    pairwise.to_csv(TABLES_DIR / "osd977_pairwise_effectsizes.csv", index=False)

    plot_bvtv(merged, FIGURES_DIR / "osd977_bvtv_by_mission.png")

    print(summary.to_string(index=False))
    print()
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
