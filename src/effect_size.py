"""Shared Welch's t-test + Hedges' g + 95% CI helper, used by both pipelines.

Hedges' g (small-sample bias-corrected Cohen's d) is the primary effect
size reported throughout; Cohen's d is retained in the output dict only
as an intermediate (pre-correction) value for reference. The CI on g
uses the standard Hedges & Olkin (1985) delta-method variance estimate,
Var(g) = J^2 * Var(d), Var(d) = (n1+n2)/(n1*n2) + d^2/(2*(n1+n2)),
with a normal-approximation 95% interval (g +/- 1.96*SE(g)).
"""
import numpy as np
from scipy import stats


def welch_stats(a, b) -> dict:
    """Welch's t-test plus Hedges' g (bias-corrected pooled-SD effect size,
    with delta-method 95% CI) and a separate 95% CI for the raw mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / sp
    J = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
    hedges_g = cohens_d * J
    var_d = (n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2))
    se_g = J * np.sqrt(var_d)

    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    dfw = (s1**2 / n1 + s2**2 / n2) ** 2 / ((s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1))
    tcrit = stats.t.ppf(0.975, dfw)
    diff = m1 - m2

    t, p = stats.ttest_ind(a, b, equal_var=False)

    return {
        "n1": n1,
        "n2": n2,
        "mean_diff": diff,
        "ci_lo": diff - tcrit * se,
        "ci_hi": diff + tcrit * se,
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "hedges_g_ci_lo": hedges_g - 1.96 * se_g,
        "hedges_g_ci_hi": hedges_g + 1.96 * se_g,
        "hedges_g_se": se_g,
        "t": t,
        "p": p,
        "df": dfw,
    }
