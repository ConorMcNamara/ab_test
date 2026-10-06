from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.stats as ss

from ab_test.frequentist_normal.utils import mle_under_null, validate_two_group

__all__ = [
    "welch_test",
    "score_test",
    "likelihood_ratio_test"
]


def welch_test(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Welch's t-test for 2 experiment groups.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The variances for each group.
    trials : array_like
        Number of trials in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline mean,
        or in absolute terms.
    crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
    pval : float
        P-value. Returned if ``crit`` is None.
    stat_sig : boolean
        True if the result is statistically significant, i.e. if the absolute
        test statistic is >= ``crit``. Returned if ``crit`` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    validate_two_group(means, trials, variances, null_lift, lift)
    mean1, mean2 = means[0], means[1]
    var1, var2 = variances[0], variances[1]
    trial1, trial2 = trials[0], trials[1]
    s1 = var1 / trial1
    s2 = var2 / trial2
    se_sq = s1 + s2
    df = se_sq * se_sq / (s1 * s1 / (trial1 - 1) + s2 * s2 / (trial2 - 1))
    se = math.sqrt(se_sq)
    expected_diff = null_lift * mean1 if lift == "relative" else null_lift
    t_value = ((mean2 - mean1) - expected_diff) / se
    if crit is None:
        pval = 2 * (1.0 - ss.t.cdf(abs(t_value), df))  # type: ignore[no-untyped-call]
        return float(pval)
    return abs(t_value) >= crit


def score_test(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Rao's score test for 2 experiment groups.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The sample variances (``ddof=1``) for each group.
    trials : array_like
        Number of trials in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline mean,
        or in absolute terms. See Notes in `mle_under_null`.
    crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
    pval : float
        P-value. Returned if ``crit`` is None.
    stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= ``crit``. Returned if ``crit`` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.

    Assumes both groups share a common variance; use `welch_test` when they
    do not. The test statistic is compared against a chi-squared
    distribution with 1 degree of freedom, so ``crit`` must be on that scale
    (e.g. ``ss.chi2.isf(alpha, df=1)``). For an absolute null it equals
    ``N * t**2 / (N - 2 + t**2)``, where ``t`` is the pooled two-sample
    t-statistic, so the exact small-sample version of this test is the
    pooled t-test.
    """
    validate_two_group(means, trials, variances, null_lift, lift)

    mu, sigma2 = mle_under_null(means, variances, trials, null_lift=null_lift, lift=lift)

    if sigma2 <= 1e-12:
        return 1.0 if crit is None else False

    ts = (trials[0] * (means[0] - mu[0]) ** 2 + trials[1] * (means[1] - mu[1]) ** 2) / sigma2

    if crit is None:
        pval = ss.chi2.sf(ts, df=1)  # type: ignore[no-untyped-call]
        return float(pval)
    return bool(ts >= crit)

def likelihood_ratio_test(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Likelihood Ratio test for 2 experiment groups.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The variances for each group.
    trials : array_like
        Number of trials in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline mean,
        or in absolute terms.
    crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
    pval : float
        P-value. Returned if ``crit`` is None.
    stat_sig : boolean
        True if the result is statistically significant, i.e. if the absolute
        test statistic is >= ``crit``. Returned if ``crit`` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    ...