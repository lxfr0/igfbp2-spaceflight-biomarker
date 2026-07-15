"""
OSD-342: mouse plasma SomaScan proteomics pipeline.

Parses the raw SomaScan Excel export (multi-row header with group labels
and per-mouse sample IDs), screens the candidate bone-remodeling marker
panel (src/bone_markers.py), and runs the full set of analyses reported
in the manuscript for Space Flight vs Ground Control (untreated arms):

  - Welch t-test + Hedges' g (95% CI) + Benjamini-Hochberg FDR, on both
    log2 RFU (primary) and raw RFU (sensitivity), for the 39 unique
    candidate genes and for the full 44-reagent panel (some genes have
    multiple SomaScan aptamer reagents).
  - Full-proteome sensitivity: same Welch test run against every reagent
    on the SomaScan panel (not just the candidate list), to report where
    the top candidate ranks and whether it survives unrestricted
    proteome-wide FDR correction.
  - Exact (not sampled) Westfall-Young max-T permutation test over all
    C(16,8)=12870 group-label permutations of the 16 untreated animals,
    on both raw and log2 scales.
  - Numeric sample-ID block-restricted permutation test: the 16 untreated
    animals fall into two ID blocks (1-4 and 9-12) within each condition;
    this test permutes flight/ground labels only within each block
    (preserving 4 flight + 4 ground per block), to check whether the
    overall result depends on that unexplained stratification.
  - Stratified bootstrap CI (resampling within group) for IGFBP2.
  - Leave-one-out sensitivity: drop each of the 16 animals in turn and
    recompute IGFBP2's BH-adjusted q-value.
  - Two-factor (flight x treatment) OLS models for the immediate
    postflight and 14-day recovery arms, with OLS, HC1, HC2 and HC3
    robust standard errors on the interaction term.

Run directly:
    python src/osd342_pipeline.py
Outputs (results/tables/):
    osd342_bone_marker_stats_log2.csv       (39 unique genes, log2 primary)
    osd342_bone_marker_stats_raw.csv        (39 unique genes, raw sensitivity)
    osd342_44reagent_stats_log2.csv         (all 44 reagents, log2)
    osd342_44reagent_stats_raw.csv          (all 44 reagents, raw)
    osd342_full_proteome_sensitivity.csv    (IGFBP2 rank among all reagents)
    osd342_robustness_igfbp2.csv            (max-T, block-perm, bootstrap, LOO summary)
    osd342_treatment_recovery_models.csv    (two-factor model coefficients, 3 SE types)
Outputs (results/figures/):
    osd342_igfbp2_boxplot.png
    osd342_marker_screen_summary.png
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bone_markers import BONE_MARKERS
from effect_size import welch_stats

RAW_DIR = Path("data/raw/OSD-342")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

GROUP_A = "flight uninjected"
GROUP_B = "ground uninjected"

BOOTSTRAP_SEED = 20260713
BOOTSTRAP_N = 10000


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
    sample_ids = [sample_row[i] for i in valid_idx]
    groups = [groups[i] for i in valid_idx]

    return annot, data, groups, sample_ids


def _scaled(vals: np.ndarray, scale: str) -> np.ndarray:
    return np.log2(vals) if scale == "log2" else vals


def compare_bone_markers(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    scale: str = "log2", dedup: bool = True,
) -> pd.DataFrame:
    """Welch + Hedges' g + BH-FDR for the candidate panel.

    scale: "log2" (primary) or "raw" (sensitivity).
    dedup: if True, keep only the first-listed reagent per gene (39 unique
        genes, matching the original panel definition). If False, report
        every one of the 44 SomaScan reagents mapping to a candidate gene.
    """
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    rows = []
    seen = set()
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene not in BONE_MARKERS:
            continue
        if dedup:
            if gene in seen:
                continue
            seen.add(gene)
        a_raw = data.iloc[i, a_idx].values.astype(float)
        b_raw = data.iloc[i, b_idx].values.astype(float)
        a_vals = _scaled(a_raw, scale)
        b_vals = _scaled(b_raw, scale)
        r = welch_stats(a_vals, b_vals)
        fc = a_raw.mean() / b_raw.mean()
        r.update(
            {
                "Gene": gene,
                "SeqId": annot.loc[i, "SeqId"],
                "Target": annot.loc[i, "Target"],
                "Role": BONE_MARKERS[gene],
                f"{group_b}_mean_raw": b_raw.mean(),
                f"{group_a}_mean_raw": a_raw.mean(),
                "log2FC": np.log2(fc),
            }
        )
        rows.append(r)

    res = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    _, p_adj, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_adj_BH"] = p_adj
    res["significant_FDR05"] = p_adj < 0.05
    cols = ["Gene", "SeqId", "Target", "Role", f"{group_b}_mean_raw", f"{group_a}_mean_raw", "log2FC",
            "mean_diff", "hedges_g", "hedges_g_ci_lo", "hedges_g_ci_hi", "cohens_d", "t", "df", "p", "p_adj_BH",
            "significant_FDR05"]
    return res[cols]


def full_proteome_sensitivity(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    gene: str, scale: str = "log2",
) -> dict:
    """Run Welch's t-test against every reagent on the panel (not just candidates);
    report the target gene's best reagent rank and whether it survives unrestricted
    proteome-wide BH-FDR correction."""
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    pvals, gene_labels = [], []
    for i, g in annot["EntrezGeneSymbol"].items():
        a_raw = data.iloc[i, a_idx].values.astype(float)
        b_raw = data.iloc[i, b_idx].values.astype(float)
        if np.any(a_raw <= 0) or np.any(b_raw <= 0):
            continue
        a_vals = _scaled(a_raw, scale)
        b_vals = _scaled(b_raw, scale)
        _, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        if np.isnan(p):
            continue
        pvals.append(p)
        gene_labels.append(g)

    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    ranked_genes = np.asarray(gene_labels)[order]
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")

    hits = np.where(ranked_genes == gene)[0]
    rank = int(hits[0]) + 1 if len(hits) else None
    q_full = float(p_adj[order][hits[0]]) if len(hits) else None

    return {
        "scale": scale,
        "n_total_reagents": len(pvals),
        f"{gene}_rank": rank,
        f"{gene}_q_proteome_wide": q_full,
        f"{gene}_survives_proteome_wide_FDR05": (q_full < 0.05) if q_full is not None else None,
    }


def exact_max_t_permutation(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    genes: list, scale: str = "log2", target_gene: str = "IGFBP2",
) -> dict:
    """Exact Westfall-Young max-T permutation over all C(n1+n2, n1) label
    permutations, restricted to the given candidate gene index set."""
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]
    all_idx = a_idx + b_idx
    n1, n2 = len(a_idx), len(b_idx)

    gene_row = {}
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene in genes and gene not in gene_row:
            gene_row[gene] = i
    rows = [gene_row[g] for g in genes]

    mat = data.iloc[rows, all_idx].values.astype(float)
    mat = _scaled(mat, scale)  # (n_genes, n_animals)

    def welch_t(vals_a, vals_b):
        m1, m2 = vals_a.mean(axis=1), vals_b.mean(axis=1)
        v1, v2 = vals_a.var(axis=1, ddof=1), vals_b.var(axis=1, ddof=1)
        na, nb = vals_a.shape[1], vals_b.shape[1]
        se = np.sqrt(v1 / na + v2 / nb)
        return (m1 - m2) / se

    obs_t = welch_t(mat[:, :n1], mat[:, n1:])
    target_idx = genes.index(target_gene)
    obs_abs_t = np.abs(obs_t[target_idx])
    obs_max_abs_t = np.abs(obs_t).max()

    n_total = n1 + n2
    perm_ge_target = 0
    perm_ge_max = 0
    n_perms = 0
    for combo in itertools.combinations(range(n_total), n1):
        combo = list(combo)
        rest = [j for j in range(n_total) if j not in combo]
        t = welch_t(mat[:, combo], mat[:, rest])
        if np.abs(t[target_idx]) >= obs_abs_t - 1e-12:
            perm_ge_target += 1
        if np.abs(t).max() >= obs_max_abs_t - 1e-12:
            perm_ge_max += 1
        n_perms += 1

    return {
        "scale": scale,
        "n_permutations_exact": n_perms,
        "target_gene": target_gene,
        "observed_abs_t": obs_abs_t,
        "single_marker_exact_p": perm_ge_target / n_perms,
        "maxT_adjusted_p": perm_ge_max / n_perms,
    }


def block_restricted_permutation(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, sample_ids: list, group_a: str, group_b: str,
    gene: str = "IGFBP2", scale: str = "log2",
) -> dict:
    """Exact permutation restricted within each numeric-ID block (1-4, 9-12),
    preserving 4 flight + 4 ground animals per block. Tests whether the
    IGFBP2 result depends on the (biologically unexplained) ID stratification."""
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    def block_of(sid: str) -> str:
        num = int("".join(ch for ch in sid if ch.isdigit()))
        return "1-4" if num <= 4 else "9-12"

    blocks = {"1-4": {"a": [], "b": []}, "9-12": {"a": [], "b": []}}
    for i in a_idx:
        blocks[block_of(sample_ids[i])]["a"].append(i)
    for i in b_idx:
        blocks[block_of(sample_ids[i])]["b"].append(i)

    idx = annot.index[annot["EntrezGeneSymbol"] == gene][0]
    vals_by_pos = {}
    all_positions = blocks["1-4"]["a"] + blocks["1-4"]["b"] + blocks["9-12"]["a"] + blocks["9-12"]["b"]
    raw_vals = data.iloc[idx, all_positions].values.astype(float)
    scaled_vals = _scaled(raw_vals, scale)
    for pos, val in zip(all_positions, scaled_vals):
        vals_by_pos[pos] = val

    def welch_t_from_lists(flight_positions, ground_positions):
        fv = np.array([vals_by_pos[p] for p in flight_positions])
        gv = np.array([vals_by_pos[p] for p in ground_positions])
        r = welch_stats(fv, gv)
        return r["t"]

    obs_flight = blocks["1-4"]["a"] + blocks["9-12"]["a"]
    obs_ground = blocks["1-4"]["b"] + blocks["9-12"]["b"]
    obs_t = welch_t_from_lists(obs_flight, obs_ground)

    block1_all = blocks["1-4"]["a"] + blocks["1-4"]["b"]
    block2_all = blocks["9-12"]["a"] + blocks["9-12"]["b"]

    n_ge, n_perms = 0, 0
    for flight1 in itertools.combinations(block1_all, 4):
        ground1 = [p for p in block1_all if p not in flight1]
        for flight2 in itertools.combinations(block2_all, 4):
            ground2 = [p for p in block2_all if p not in flight2]
            t = welch_t_from_lists(list(flight1) + list(flight2), ground1 + ground2)
            if abs(t) >= abs(obs_t) - 1e-12:
                n_ge += 1
            n_perms += 1

    return {
        "gene": gene,
        "scale": scale,
        "n_block_permutations_exact": n_perms,
        "observed_t": obs_t,
        "block_restricted_exact_p": n_ge / n_perms,
    }


def stratified_bootstrap_ci(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    gene: str = "IGFBP2", scale: str = "log2", n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Stratified (within-group) bootstrap percentile CI for the mean difference / Hedges' g."""
    rng = np.random.default_rng(seed)
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]
    idx = annot.index[annot["EntrezGeneSymbol"] == gene][0]

    a_vals = _scaled(data.iloc[idx, a_idx].values.astype(float), scale)
    b_vals = _scaled(data.iloc[idx, b_idx].values.astype(float), scale)
    na, nb = len(a_vals), len(b_vals)

    boot_g = np.empty(n_boot)
    for k in range(n_boot):
        a_s = a_vals[rng.integers(0, na, na)]
        b_s = b_vals[rng.integers(0, nb, nb)]
        boot_g[k] = welch_stats(a_s, b_s)["hedges_g"]

    return {
        "gene": gene,
        "scale": scale,
        "n_boot": n_boot,
        "seed": seed,
        "hedges_g_boot_ci_lo": float(np.percentile(boot_g, 2.5)),
        "hedges_g_boot_ci_hi": float(np.percentile(boot_g, 97.5)),
    }


def leave_one_out(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, genes: list, group_a: str, group_b: str,
    target_gene: str = "IGFBP2", scale: str = "log2",
) -> pd.DataFrame:
    """Drop each of the 16 untreated animals in turn; recompute BH-FDR q for target_gene
    among the candidate panel each time."""
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    gene_row = {}
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene in genes and gene not in gene_row:
            gene_row[gene] = i

    rows = []
    for drop_group, drop_list in [("flight", a_idx), ("ground", b_idx)]:
        for drop_i in drop_list:
            a_keep = [i for i in a_idx if i != drop_i]
            b_keep = [i for i in b_idx if i != drop_i]
            pvals = {}
            for g, row in gene_row.items():
                a_vals = _scaled(data.iloc[row, a_keep].values.astype(float), scale)
                b_vals = _scaled(data.iloc[row, b_keep].values.astype(float), scale)
                pvals[g] = welch_stats(a_vals, b_vals)["p"]
            gene_order = list(pvals.keys())
            _, p_adj, _, _ = multipletests(list(pvals.values()), method="fdr_bh")
            q_target = p_adj[gene_order.index(target_gene)]
            rows.append({"dropped_group": drop_group, "dropped_index": drop_i,
                          "target_gene": target_gene, "q_BH": q_target, "significant": q_target < 0.05})
    return pd.DataFrame(rows)


def treatment_recovery_models(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, gene: str = "IGFBP2", scale: str = "log2",
) -> pd.DataFrame:
    """Two-factor (condition x treatment) OLS models for the immediate postflight arm
    and the 14-day recovery arm, with OLS/HC1/HC2/HC3 standard errors."""
    idx = annot.index[annot["EntrezGeneSymbol"] == gene][0]
    raw_vals = data.iloc[idx].values.astype(float)
    df_all = pd.DataFrame({"value_raw": raw_vals, "group": groups})
    df_all["value"] = _scaled(df_all["value_raw"].values, scale)

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

    rows = []
    for label, mapping, cond_ref, cond_alt in [
        ("immediate_postflight", panel_a_map, "Ground", "Flight"),
        ("recovery_14day", panel_b_map, "Vivarium", "Flight recovery"),
    ]:
        sub = df_all[df_all["group"].isin(mapping)].copy()
        sub["condition"] = pd.Categorical(sub["group"].map(lambda g: mapping[g][0]), categories=[cond_ref, cond_alt])
        sub["treatment"] = pd.Categorical(sub["group"].map(lambda g: mapping[g][1]), categories=["Untreated", "ACVR2B/Fc"])

        model = smf.ols("value ~ C(condition) * C(treatment)", data=sub).fit()
        for cov_type in ["nonrobust", "HC1", "HC2", "HC3"]:
            fit = model if cov_type == "nonrobust" else smf.ols("value ~ C(condition) * C(treatment)", data=sub).fit(cov_type=cov_type)
            for term in fit.params.index:
                rows.append({
                    "arm": label, "cov_type": cov_type, "term": term,
                    "coef": fit.params[term], "p": fit.pvalues[term],
                })
    return pd.DataFrame(rows)


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
    """Ranked bar chart of -log10(p) across the full marker panel (log2-scale primary
    analysis), FDR-significant hit highlighted."""
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
    ax.set_xlabel("-log10(p), log2 RFU, Flight vs Ground Control", fontsize=10)
    n_sig = int(df["significant_FDR05"].sum())
    top_q = df.loc[df["significant_FDR05"], "p_adj_BH"].min() if n_sig else float("nan")
    ax.set_title(
        f"OSD-342 plasma bone-marker screen (39 candidate genes, log2 RFU; n=8 mice/group)\n"
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

    annot, data, groups, sample_ids = load_somascan()
    genes_39 = sorted(set(BONE_MARKERS) & set(annot["EntrezGeneSymbol"]))

    res_log2 = compare_bone_markers(annot, data, groups, GROUP_A, GROUP_B, scale="log2", dedup=True)
    res_raw = compare_bone_markers(annot, data, groups, GROUP_A, GROUP_B, scale="raw", dedup=True)
    res_log2.to_csv(TABLES_DIR / "osd342_bone_marker_stats_log2.csv", index=False)
    res_raw.to_csv(TABLES_DIR / "osd342_bone_marker_stats_raw.csv", index=False)

    res_44_log2 = compare_bone_markers(annot, data, groups, GROUP_A, GROUP_B, scale="log2", dedup=False)
    res_44_raw = compare_bone_markers(annot, data, groups, GROUP_A, GROUP_B, scale="raw", dedup=False)
    res_44_log2.to_csv(TABLES_DIR / "osd342_44reagent_stats_log2.csv", index=False)
    res_44_raw.to_csv(TABLES_DIR / "osd342_44reagent_stats_raw.csv", index=False)

    fp_log2 = full_proteome_sensitivity(annot, data, groups, GROUP_A, GROUP_B, "IGFBP2", scale="log2")
    fp_raw = full_proteome_sensitivity(annot, data, groups, GROUP_A, GROUP_B, "IGFBP2", scale="raw")
    pd.DataFrame([fp_log2, fp_raw]).to_csv(TABLES_DIR / "osd342_full_proteome_sensitivity.csv", index=False)

    maxt_log2 = exact_max_t_permutation(annot, data, groups, GROUP_A, GROUP_B, genes_39, scale="log2")
    maxt_raw = exact_max_t_permutation(annot, data, groups, GROUP_A, GROUP_B, genes_39, scale="raw")
    block_log2 = block_restricted_permutation(annot, data, groups, sample_ids, GROUP_A, GROUP_B, scale="log2")
    boot_log2 = stratified_bootstrap_ci(annot, data, groups, GROUP_A, GROUP_B, scale="log2")
    loo_log2 = leave_one_out(annot, data, groups, genes_39, GROUP_A, GROUP_B, scale="log2")

    robustness_rows = [
        {"check": "maxT_adjusted_p_raw", "value": maxt_raw["maxT_adjusted_p"]},
        {"check": "maxT_adjusted_p_log2", "value": maxt_log2["maxT_adjusted_p"]},
        {"check": "single_marker_exact_p_raw", "value": maxt_raw["single_marker_exact_p"]},
        {"check": "single_marker_exact_p_log2", "value": maxt_log2["single_marker_exact_p"]},
        {"check": "block_restricted_exact_p_log2", "value": block_log2["block_restricted_exact_p"]},
        {"check": "bootstrap_g_ci_lo_log2", "value": boot_log2["hedges_g_boot_ci_lo"]},
        {"check": "bootstrap_g_ci_hi_log2", "value": boot_log2["hedges_g_boot_ci_hi"]},
        {"check": "loo_n_significant_of_16_log2", "value": int(loo_log2["significant"].sum())},
    ]
    pd.DataFrame(robustness_rows).to_csv(TABLES_DIR / "osd342_robustness_igfbp2.csv", index=False)
    loo_log2.to_csv(TABLES_DIR / "osd342_leave_one_out_log2.csv", index=False)

    tr_models = treatment_recovery_models(annot, data, groups, gene="IGFBP2", scale="log2")
    tr_models.to_csv(TABLES_DIR / "osd342_treatment_recovery_models.csv", index=False)

    top_gene = res_log2.iloc[0]["Gene"]
    plot_top_hit(annot, data, groups, top_gene, FIGURES_DIR / f"osd342_{top_gene.lower()}_boxplot.png")
    plot_marker_summary(res_log2, FIGURES_DIR / "osd342_marker_screen_summary.png")

    print(res_log2.to_string(index=False))
    print(f"\n{res_log2['significant_FDR05'].sum()} of {len(res_log2)} bone markers survive BH-FDR q<0.05 (log2, primary)")
    print(f"Top hit: {top_gene} (q={res_log2.iloc[0]['p_adj_BH']:.4f})")
    print("\nRobustness checks:")
    print(pd.DataFrame(robustness_rows).to_string(index=False))


if __name__ == "__main__":
    main()
