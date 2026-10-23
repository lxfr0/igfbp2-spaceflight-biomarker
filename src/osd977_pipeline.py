"""
OSD-977: rat femur microCT bone-microarchitecture pipeline.

Merges the ISA sample sheet (group/mission/sex/tissue labels) with the
transformed microCT measurement CSV, then runs group comparisons
(Baseline vs Ground Control vs Space Flight) per mission, with Welch's
ANOVA (unequal-variance omnibus test) and pairwise Welch t-tests +
Hedges' g (95% CI) per comparison.

Two separate, prespecified Benjamini-Hochberg FDR correction families are
used:
  - Cancellous family: 5 cancellous metrics (BV/TV, connectivity density,
    trabecular number, trabecular thickness, trabecular spacing) x 2
    missions x 3 pairwise contrasts = 30 tests.
  - Cortical family: cortical thickness alone, treated as a separate
    prespecified secondary compartment, x 2 missions x 3 pairwise
    contrasts = 6 tests.
These are correction families, not measurement compartments that
biologically interact -- cancellous and cortical bone respond to
unloading differently, so pooling them into one family would let a
strong cortical result inflate apparent significance for a weak
cancellous one, or vice versa.

Run directly:
    python src/osd977_pipeline.py
Outputs:
    data/processed/merged_OSD-977.csv
    results/tables/osd977_group_stats.csv
    results/tables/osd977_welch_anova.csv
    results/tables/osd977_pairwise_effectsizes_cancellous.csv
    results/tables/osd977_pairwise_effectsizes_cortical.csv
    results/figures/osd977_bvtv_by_mission.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
CANCELLOUS_METRICS = [
    "bone_volume_over_tissue_volumn",
    "connectivity_density",
    "trabecular_number",
    "trabecular_thickness",
    "trabecular_spacing",
]
CORTICAL_METRICS = ["cortical_thickness"]
NUMERIC_COLS = CANCELLOUS_METRICS + CORTICAL_METRICS


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


def welch_anova(groups_dict: dict) -> dict:
    """Welch's ANOVA (unequal-variance omnibus test) for k independent groups."""
    present = {g: np.asarray(v, dtype=float) for g, v in groups_dict.items() if len(v) > 1}
    k = len(present)
    if k < 2:
        return {"F_welch": float("nan"), "df1": float("nan"), "df2": float("nan"), "p_welch": float("nan")}

    ni = {g: len(v) for g, v in present.items()}
    mi = {g: v.mean() for g, v in present.items()}
    vi = {g: v.var(ddof=1) for g, v in present.items()}
    wi = {g: ni[g] / vi[g] for g in present}
    w_sum = sum(wi.values())
    grand_mean = sum(wi[g] * mi[g] for g in present) / w_sum

    numerator = sum(wi[g] * (mi[g] - grand_mean) ** 2 for g in present) / (k - 1)
    lam = sum((1 - wi[g] / w_sum) ** 2 / (ni[g] - 1) for g in present)
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * lam
    f_welch = numerator / denom

    df1 = k - 1
    df2 = (k**2 - 1) / (3 * lam)
    p_welch = stats.f.sf(f_welch, df1, df2)

    return {"F_welch": f_welch, "df1": df1, "df2": df2, "p_welch": p_welch}


def group_stats(merged: pd.DataFrame, metric: str, tissue: str, mission: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Welch's ANOVA + pairwise Welch t-tests for a metric within one mission/tissue."""
    sub = merged[(merged["Mission"] == mission) & (merged["Tissue"] == tissue)].dropna(subset=[metric])
    groups = {g: sub[sub["Group"] == g][metric].values for g in GROUP_ORDER}

    anova = welch_anova(groups)
    rows = []
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
                "ANOVA_F_welch": anova["F_welch"],
                "ANOVA_df1": anova["df1"],
                "ANOVA_df2": anova["df2"],
                "ANOVA_p_welch": anova["p_welch"],
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
        cols = ["Mission", "Tissue", "Metric", "Comparison", "n1", "n2", "mean_diff", "ci_lo", "ci_hi",
                "hedges_g", "hedges_g_ci_lo", "hedges_g_ci_hi",
                "hedges_g_welch_ci_lo", "hedges_g_welch_ci_hi", "t", "df", "p"]
        pairwise_df = pairwise_df[cols]
    return pd.DataFrame(rows), pairwise_df, anova


def run_all_stats(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summary stats per group, plus pairwise tests with BH-FDR correction applied
    separately within two prespecified families: 30 cancellous tests (5 metrics x 2
    missions x 3 contrasts) and 6 cortical tests (1 metric x 2 missions x 3 contrasts)."""
    summaries = []
    pairwise_cancellous, pairwise_cortical = [], []
    anova_rows = []

    tissue_for = {**{m: "Femur Head" for m in CANCELLOUS_METRICS}, **{m: "diaphysis of femur" for m in CORTICAL_METRICS}}

    for mission in merged["Mission"].dropna().unique():
        for metric in CANCELLOUS_METRICS:
            s, pw, anova = group_stats(merged, metric, tissue_for[metric], mission)
            summaries.append(s)
            pairwise_cancellous.append(pw)
            anova_rows.append({"Mission": mission, "Metric": metric, "family": "cancellous", **anova})
        for metric in CORTICAL_METRICS:
            s, pw, anova = group_stats(merged, metric, tissue_for[metric], mission)
            summaries.append(s)
            pairwise_cortical.append(pw)
            anova_rows.append({"Mission": mission, "Metric": metric, "family": "cortical", **anova})

    pw_cancellous = pd.concat(pairwise_cancellous, ignore_index=True)
    pw_cortical = pd.concat(pairwise_cortical, ignore_index=True)

    _, q_can, _, _ = multipletests(pw_cancellous["p"], method="fdr_bh")
    pw_cancellous["q_BH"] = q_can
    assert len(pw_cancellous) == 30, f"expected 30 cancellous tests, got {len(pw_cancellous)}"

    _, q_cort, _, _ = multipletests(pw_cortical["p"], method="fdr_bh")
    pw_cortical["q_BH"] = q_cort
    assert len(pw_cortical) == 6, f"expected 6 cortical tests, got {len(pw_cortical)}"

    return pd.concat(summaries, ignore_index=True), pw_cancellous, pw_cortical, pd.DataFrame(anova_rows)


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

    summary, pw_cancellous, pw_cortical, anova = run_all_stats(merged)
    summary.to_csv(TABLES_DIR / "osd977_group_stats.csv", index=False)
    anova.to_csv(TABLES_DIR / "osd977_welch_anova.csv", index=False)
    pw_cancellous.to_csv(TABLES_DIR / "osd977_pairwise_effectsizes_cancellous.csv", index=False)
    pw_cortical.to_csv(TABLES_DIR / "osd977_pairwise_effectsizes_cortical.csv", index=False)

    plot_bvtv(merged, FIGURES_DIR / "osd977_bvtv_by_mission.png")

    print("Welch's ANOVA (omnibus, per metric/mission):")
    print(anova.to_string(index=False))
    print("\nCancellous pairwise (30-test family):")
    print(pw_cancellous.to_string(index=False))
    print("\nCortical pairwise (6-test family):")
    print(pw_cortical.to_string(index=False))


if __name__ == "__main__":
    main()
