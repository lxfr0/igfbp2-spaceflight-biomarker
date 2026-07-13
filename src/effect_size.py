"""Shared Welch's t-test + Cohen's d + 95% CI helper, used by both pipelines."""
import numpy as np
from scipy import stats


def welch_stats(a, b) -> dict:
    """Welch's t-test plus Cohen's d (pooled SD) and a 95% CI for the mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / sp

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
        "t": t,
        "p": p,
    }
