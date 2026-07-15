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


def select_primary_reagents(annot: pd.DataFrame, data: pd.DataFrame, groups: list,
                             group_a: str, group_b: str) -> dict:
    """For genes with more than one SomaScan reagent, select the reagent with the
    lowest coefficient of variation (SD/mean) across all 16 untreated animals
    (both groups pooled). This is a pre-specified, outcome-independent selection
    rule -- it never looks at the flight-vs-ground group labels, only at overall
    measurement reproducibility -- replacing an earlier "first-listed reagent"
    rule that had no principled justification. Returns {gene: row_index}."""
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]
    pooled_idx = a_idx + b_idx

    candidates = {}
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene in BONE_MARKERS:
            candidates.setdefault(gene, []).append(i)

    selected = {}
    for gene, row_idx in candidates.items():
        best_row, best_cv = None, None
        for i in row_idx:
            vals = data.iloc[i, pooled_idx].values.astype(float)
            cv = vals.std(ddof=1) / vals.mean()
            if best_cv is None or cv < best_cv:
                best_row, best_cv = i, cv
        selected[gene] = best_row
    return selected


def compare_bone_markers(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    scale: str = "log2", dedup: bool = True,
) -> pd.DataFrame:
    """Welch + Hedges' g + BH-FDR for the candidate panel.

    scale: "log2" (primary) or "raw" (sensitivity).
    dedup: if True, keep only the lowest-CV reagent per gene (39 unique genes;
        see select_primary_reagents for the selection rule). If False, report
        every one of the 44 SomaScan reagents mapping to a candidate gene.
    """
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]

    primary_rows = set(select_primary_reagents(annot, data, groups, group_a, group_b).values()) if dedup else None

    rows = []
    for i, gene in annot["EntrezGeneSymbol"].items():
        if gene not in BONE_MARKERS:
            continue
        if dedup and i not in primary_rows:
            continue
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


# Kimura et al. (2024a), Table 1: the complete list of 30 serum proteins whose levels
# changed in the same direction in spaceflight (mu-g vs A1-g) and hindlimb-unloading
# (HU vs control) mice, taken directly from the paper's own Table 1 (Proteomics 24(9):
# e2300214, Open Access) -- not re-derived or approximated. UniProt is the MOUSE
# accession Kimura reports; direction is the sign of their reported mu-g-vs-A1-g log2
# change (identical in sign to their HU-vs-control column for every listed protein).
# NOTE: SomaScan's own UniProt annotation column (in `annot`) records each aptamer's
# HUMAN target accession, even in this mouse assay (aptamers are raised against human
# proteins) -- confirmed by direct comparison (e.g. IGFBP2: Kimura's mouse accession is
# P47877, SomaScan lists the human accession P18065). Cross-species UniProt accessions
# therefore cannot be matched directly without an additional human-mouse ortholog table,
# so mapping here uses gene symbol (case-normalized), which is unambiguous for every
# protein below -- not a fallback of convenience.
KIMURA_2024A_TABLE = [
    # (UniProt_mouse, Gene, ProteinName, log2_muG_vs_A1g, log2_HU_vs_control)
    ("P12246", "APCS", "Serum amyloid P-component", 0.91, 1.72),
    ("P00688", "AMY2", "Pancreatic alpha-amylase", 0.27, 0.32),
    ("P29533", "VCAM1", "Vascular cell adhesion protein 1", -0.20, -0.40),
    ("P97290", "SERPING1", "Plasma protease C1 inhibitor", -0.20, -0.48),
    ("P98064", "MASP1", "Mannan-binding lectin serine protease 1", -0.21, -0.35),
    ("Q06890", "CLU", "Clusterin", -0.23, -0.36),
    ("P97298", "SERPINF1", "Pigment epithelium-derived factor", -0.29, -0.36),
    ("O09043", "NAPSA", "Napsin-A", -0.30, -0.36),
    ("O54819", "TFPI", "Tissue factor pathway inhibitor", -0.30, -0.51),
    ("P13020", "GSN", "Gelsolin", -0.32, -0.33),
    ("O89103", "CD93", "Complement component C1q receptor", -0.38, -0.72),
    ("P10605", "CTSB", "Cathepsin B", -0.42, -0.28),
    ("Q8R2Y2", "MCAM", "Cell surface glycoprotein MUC18", -0.43, -1.36),
    ("P42703", "LIFR", "Leukemia inhibitory factor receptor", -0.45, -0.34),
    ("Q61207", "PSAP", "Prosaposin", -0.48, -0.41),
    ("P82198", "TGFBI", "TGF-beta-induced protein ig-h3", -0.57, -0.63),
    ("P47877", "IGFBP2", "Insulin-like growth factor-binding protein 2", -0.58, -0.51),
    ("P51885", "LUM", "Lumican", -0.58, -0.30),
    ("P43025", "CLEC3B", "Tetranectin", -0.59, -0.68),
    ("Q8BK48", "CES2E", "Pyrethroid hydrolase Ces2e", -0.61, -1.28),
    ("Q5FW60", "MUP20", "Major urinary protein 20", -0.65, -0.70),
    ("Q03401", "CRISP1", "Cysteine-rich secretory protein 1", -0.66, -1.19),
    ("Q80YX1", "TNC", "Tenascin", -0.69, -0.34),
    ("Q6DFX2", "ANTXR2", "Anthrax toxin receptor 2", -0.72, -0.60),
    ("Q8QZR4", "OAF", "Out at first protein homolog", -0.92, -2.81),
    ("Q9QUN9", "DKK3", "Dickkopf-related protein 3", -0.93, -1.33),
    ("P33434", "MMP2", "72 kDa type IV collagenase", -1.11, -0.54),
    ("P70194", "CLEC4F", "C-type lectin domain family 4 member F", -1.53, -0.91),
    ("P55284", "CDH5", "Cadherin-5", -1.60, -2.63),
    ("Q60696", "PMEL", "Melanocyte protein PMEL", -2.10, -3.54),
]
KIMURA_2024A_DIRECTIONS = {g: (1 if d > 0 else -1) for _, g, _, d, _ in KIMURA_2024A_TABLE}


def kimura_concordance_analysis(
    annot: pd.DataFrame, data: pd.DataFrame, groups: list, group_a: str, group_b: str,
    scale: str = "log2", n_boot_loo: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Exploratory external concordance check against Kimura et al. (2024a)'s complete
    30-protein gravity-responsive serum signature (see KIMURA_2024A_TABLE). Genes with
    multiple OSD-342 reagents are resolved genome-wide with the same lowest-CV,
    outcome-independent rule used for the primary 39-marker panel
    (select_primary_reagents), not restricted to the candidate panel. This is a
    concordance check, not independent validation: OSD-342 (untreated spaceflight vs
    ground) and Kimura's own comparisons (spaceflight mu-g vs onboard artificial-1g;
    hindlimb-unloading vs control) are different contrasts in different animals.

    Reports, in order of preference (least to most assumption-laden):
      1. PRIMARY: exact permutation test of the direction-adjusted signature score,
         enumerating all C(16,8)=12,870 possible 8-vs-8 relabelings of the untreated
         animals -- preserves the correlation structure among proteins and does not
         assume a 50% null concordance rate.
      2. SECONDARY (descriptive): proportion concordant, with exact one-sided AND
         two-sided binomial tests against a null of 0.5, plus an exact (Clopper-Pearson)
         95% CI for the proportion.
      3. SECONDARY (rank): Mann-Whitney U comparing the Kimura-oriented OSD-342
         statistic for mapped proteins against the rest of the full proteome, with the
         U statistic and rank-biserial correlation effect size reported explicitly.
      4. Sensitivity: the above repeated with IGFBP2 excluded, since it is both the
         manuscript's headline result and the strongest single contributor.
    """
    a_idx = [i for i, g in enumerate(groups) if g == group_a]
    b_idx = [i for i, g in enumerate(groups) if g == group_b]
    all_idx = a_idx + b_idx
    pooled_idx = all_idx
    n1 = len(a_idx)

    best_row = {}
    for i, gene in annot["EntrezGeneSymbol"].items():
        if pd.isna(gene):
            continue
        a_raw = data.iloc[i, a_idx].values.astype(float)
        b_raw = data.iloc[i, b_idx].values.astype(float)
        if np.any(a_raw <= 0) or np.any(b_raw <= 0):
            continue
        all_vals = data.iloc[i, pooled_idx].values.astype(float)
        cv = all_vals.std(ddof=1) / all_vals.mean()
        if gene not in best_row or cv < best_row[gene][1]:
            best_row[gene] = (i, cv)

    rows = []
    for gene, (i, _cv) in best_row.items():
        a_raw = data.iloc[i, a_idx].values.astype(float)
        b_raw = data.iloc[i, b_idx].values.astype(float)
        a_vals = _scaled(a_raw, scale)
        b_vals = _scaled(b_raw, scale)
        t, _p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        if np.isnan(t):
            continue
        rows.append({"Gene": gene, "t": t, "row": i})
    full_df = pd.DataFrame(rows)

    # ---- Full 30-protein table, mapped + unmapped, with mapping notes ----
    kimura_df = pd.DataFrame(KIMURA_2024A_TABLE,
                              columns=["UniProt_mouse_Kimura", "Gene", "ProteinName",
                                       "Kimura_log2_muG_vs_A1g", "Kimura_log2_HU_vs_control"])
    kimura_df["kimura_direction"] = kimura_df["Gene"].map(KIMURA_2024A_DIRECTIONS)
    t_lookup = dict(zip(full_df["Gene"], full_df["t"]))
    row_lookup = dict(zip(full_df["Gene"], full_df["row"]))
    kimura_df["mapped"] = kimura_df["Gene"].isin(t_lookup)
    kimura_df["osd342_SeqId"] = kimura_df["Gene"].map(
        lambda g: annot.loc[row_lookup[g], "SeqId"] if g in row_lookup else None)
    kimura_df["osd342_UniProt_human"] = kimura_df["Gene"].map(
        lambda g: annot.loc[row_lookup[g], "UniProt"] if g in row_lookup else None)
    kimura_df["osd342_t"] = kimura_df["Gene"].map(t_lookup)
    kimura_df["osd342_direction"] = np.sign(kimura_df["osd342_t"])
    kimura_df["concordant"] = kimura_df["kimura_direction"] == kimura_df["osd342_direction"]
    ranked = full_df.sort_values("t").reset_index(drop=True)
    rank_lookup = {g: i + 1 for i, g in enumerate(ranked["Gene"])}
    kimura_df["osd342_full_proteome_rank"] = kimura_df["Gene"].map(
        lambda g: rank_lookup.get(g))
    kimura_df["n_total_reagents_for_gene"] = kimura_df["Gene"].map(
        lambda g: int((annot["EntrezGeneSymbol"] == g).sum()))
    kimura_df["mapping_note"] = kimura_df.apply(
        lambda r: (
            "Not present on SomaScan panel" if not r["mapped"]
            else (f"{r['n_total_reagents_for_gene']} reagents on panel; lowest-CV reagent "
                  f"retained (prespecified, outcome-independent rule)"
                  if r["n_total_reagents_for_gene"] > 1 else "Single reagent")
        ), axis=1)
    kimura_df = kimura_df.sort_values("osd342_t").reset_index(drop=True)

    mapped = kimura_df[kimura_df["mapped"]].copy()
    n_mapped = len(mapped)
    n_concordant = int(mapped["concordant"].sum())
    missing = sorted(kimura_df.loc[~kimura_df["mapped"], "Gene"])

    # ---- Secondary (descriptive): exact one- and two-sided binomial + exact CI ----
    binom_1s = stats.binomtest(n_concordant, n_mapped, 0.5, alternative="greater")
    binom_2s = stats.binomtest(n_concordant, n_mapped, 0.5, alternative="two-sided")
    ci = binom_2s.proportion_ci(confidence_level=0.95, method="exact")

    # ---- Secondary (rank): Mann-Whitney U, explicit definition + effect size ----
    mapped["oriented_t"] = mapped["osd342_t"] * mapped["kimura_direction"]
    rest = full_df[~full_df["Gene"].isin(KIMURA_2024A_DIRECTIONS)]
    mwu = stats.mannwhitneyu(mapped["oriented_t"], rest["t"], alternative="greater")
    n_x, n_y = len(mapped), len(rest)
    rank_biserial = (2 * mwu.statistic) / (n_x * n_y) - 1  # common-language / rank-biserial effect size;
    # +1 = mapped set entirely above rest, -1 = entirely below, 0 = no separation

    # ---- Directional percentile (alternative, distribution-free framing of the same idea) ----
    full_sorted_t = np.sort(full_df["t"].values)
    n_full = len(full_sorted_t)

    def directional_percentile(gene_t, direction):
        # pct = ascending-order percentile (low t -> low pct). A predicted-DECREASE protein
        # is concordant when its t is very negative (low raw pct), so reverse for direction=-1
        # so that "concordant/extreme in the predicted direction" always maps to a HIGH value.
        pct = np.searchsorted(full_sorted_t, gene_t, side="right") / n_full
        return (1 - pct) if direction == -1 else pct

    mapped["directional_percentile"] = [
        directional_percentile(t, d) for t, d in zip(mapped["osd342_t"], mapped["kimura_direction"])
    ]
    mean_dir_pctile = float(mapped["directional_percentile"].mean())
    pctile_wilcoxon = stats.wilcoxon(mapped["directional_percentile"] - 0.5, alternative="greater")

    # ---- PRIMARY: exact permutation test of the direction-adjusted signature score ----
    # Enumerates all C(16,8)=12,870 relabelings of the untreated animals; for each,
    # recomputes the 23 mapped genes' Welch t-statistics, multiplies each by its Kimura-
    # predicted sign, and compares two summary statistics (mean adjusted t; count
    # concordant) against their exact permutation null distributions.
    mapped_genes = list(mapped["Gene"])
    mapped_rows = [row_lookup[g] for g in mapped_genes]
    mapped_signs = np.array([KIMURA_2024A_DIRECTIONS[g] for g in mapped_genes])
    mat = data.iloc[mapped_rows, all_idx].values.astype(float)
    mat = _scaled(mat, scale)

    def welch_t_matrix(vals_a, vals_b):
        m1, m2 = vals_a.mean(axis=1), vals_b.mean(axis=1)
        v1, v2 = vals_a.var(axis=1, ddof=1), vals_b.var(axis=1, ddof=1)
        na, nb = vals_a.shape[1], vals_b.shape[1]
        se = np.sqrt(v1 / na + v2 / nb)
        return (m1 - m2) / se

    n_total = mat.shape[1]
    obs_t = welch_t_matrix(mat[:, :n1], mat[:, n1:])
    obs_score = float(np.mean(obs_t * mapped_signs))
    obs_n_concordant = int(np.sum(np.sign(obs_t) == mapped_signs))

    n_perm = 0
    score_ge = 0
    concordant_ge = 0
    for combo in itertools.combinations(range(n_total), n1):
        combo = list(combo)
        rest_idx = [j for j in range(n_total) if j not in combo]
        t = welch_t_matrix(mat[:, combo], mat[:, rest_idx])
        score = np.mean(t * mapped_signs)
        if score >= obs_score - 1e-12:
            score_ge += 1
        if np.sum(np.sign(t) == mapped_signs) >= obs_n_concordant:
            concordant_ge += 1
        n_perm += 1

    perm_score_p = score_ge / n_perm
    perm_concordant_p = concordant_ge / n_perm

    # ---- Sensitivity: exclude IGFBP2 (the headline result) ----
    igfbp2_excl = mapped[mapped["Gene"] != "IGFBP2"]
    n_excl = len(igfbp2_excl)
    n_conc_excl = int(igfbp2_excl["concordant"].sum())
    binom_excl_1s = stats.binomtest(n_conc_excl, n_excl, 0.5, alternative="greater")
    binom_excl_2s = stats.binomtest(n_conc_excl, n_excl, 0.5, alternative="two-sided")

    summary = {
        "scale": scale,
        "n_kimura_proteins_total": len(KIMURA_2024A_TABLE),
        "n_mapped_to_osd342": n_mapped,
        "n_not_on_panel": len(missing),
        "genes_not_on_panel": ", ".join(missing),
        "n_concordant": n_concordant,
        "proportion_concordant": n_concordant / n_mapped,
        "binomial_p_one_sided": binom_1s.pvalue,
        "binomial_p_two_sided": binom_2s.pvalue,
        "binomial_exact_ci_95_lo": ci.low,
        "binomial_exact_ci_95_hi": ci.high,
        "mannwhitney_U": float(mwu.statistic),
        "mannwhitney_p_one_sided": mwu.pvalue,
        "mannwhitney_rank_biserial": rank_biserial,
        "mean_directional_percentile": mean_dir_pctile,
        "directional_percentile_wilcoxon_p": pctile_wilcoxon.pvalue,
        "n_exact_permutations": n_perm,
        "perm_score_observed": obs_score,
        "perm_score_p_one_sided": perm_score_p,
        "perm_concordant_p_one_sided": perm_concordant_p,
        "igfbp2_excluded_n_mapped": n_excl,
        "igfbp2_excluded_n_concordant": n_conc_excl,
        "igfbp2_excluded_proportion": n_conc_excl / n_excl,
        "igfbp2_excluded_binomial_p_one_sided": binom_excl_1s.pvalue,
        "igfbp2_excluded_binomial_p_two_sided": binom_excl_2s.pvalue,
    }
    return kimura_df, summary


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

    gene_row = select_primary_reagents(annot, data, groups, group_a, group_b)
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

    gene_row = select_primary_reagents(annot, data, groups, group_a, group_b)

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

    kimura_genes, kimura_summary = kimura_concordance_analysis(annot, data, groups, GROUP_A, GROUP_B, scale="log2")
    kimura_genes.to_csv(TABLES_DIR / "osd342_kimura_concordance_genes.csv", index=False)
    pd.DataFrame([kimura_summary]).to_csv(TABLES_DIR / "osd342_kimura_concordance_summary.csv", index=False)

    top_gene = res_log2.iloc[0]["Gene"]
    plot_top_hit(annot, data, groups, top_gene, FIGURES_DIR / f"osd342_{top_gene.lower()}_boxplot.png")
    plot_marker_summary(res_log2, FIGURES_DIR / "osd342_marker_screen_summary.png")

    print(res_log2.to_string(index=False))
    print(f"\n{res_log2['significant_FDR05'].sum()} of {len(res_log2)} bone markers survive BH-FDR q<0.05 (log2, primary)")
    print(f"Top hit: {top_gene} (q={res_log2.iloc[0]['p_adj_BH']:.4f})")
    print("\nRobustness checks:")
    print(pd.DataFrame(robustness_rows).to_string(index=False))
    print("\nKimura et al. (2024a) concordance:")
    print(pd.DataFrame([kimura_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
