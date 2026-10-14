"""Statistical tests to determine significance for normally distributed metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.stats as ss

from ab_test.frequentist_normal.utils import mle_under_equal_means, mle_under_null, validate_two_group

__all__ = ["welch_test", "score_test", "likelihood_ratio_test", "welch_anova", "score_omnibus_test"]


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
    equal_var: bool = True,
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
    equal_var : bool, default=True
        If True, assume both groups share a common variance. If False, give
        each group its own variance, analogous to `welch_test`. See Notes.

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

    The test statistic is ``sum(n_i * (mean_i - mu_i)**2 / sigma2_i)``, where
    ``mu_i`` and ``sigma2_i`` are the MLEs under H0 from `mle_under_null`. It
    is compared against a chi-squared distribution with 1 degree of freedom,
    so ``crit`` must be on that scale (e.g. ``ss.chi2.isf(alpha, df=1)``).

    With ``equal_var=True`` and an absolute null, the statistic equals
    ``N * t**2 / (N - 2 + t**2)``, where ``t`` is the pooled two-sample
    t-statistic, so the exact small-sample version of this test is the
    pooled t-test.

    With ``equal_var=False`` and an absolute null, the statistic equals
    ``D**2 / (sigma2_a / n_a + sigma2_b / n_b)`` with
    ``D = mean_b - mean_a - null_lift``: Welch's statistic squared, but with
    each variance estimated under H0. There is no exact small-sample version
    (this is the Behrens-Fisher problem), so for small samples prefer
    `welch_test`.
    """
    validate_two_group(means, trials, variances, null_lift, lift)

    mu, sigma2 = mle_under_null(means, variances, trials, null_lift=null_lift, lift=lift, equal_var=equal_var)
    sigma2_by_group = sigma2 if isinstance(sigma2, list) else [sigma2, sigma2]

    # A group's null variance is only zero when its mean is fit exactly, so it
    # contributes nothing to the statistic.
    ts = sum(n * (m - u) ** 2 / s2 for n, m, u, s2 in zip(trials[:2], means[:2], mu, sigma2_by_group) if s2 > 1e-12)

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


def _validate_many_groups(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
) -> None:
    if not len(means) == len(variances) == len(trials):
        raise ValueError("means, variances and trials must have the same length")
    if len(means) < 2:
        raise ValueError(f"An omnibus test needs at least 2 groups, got {len(means)}")


def welch_anova(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
) -> tuple[float, float, float, float]:
    """Welch's one-way ANOVA that two or more groups share one mean.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The sample variances (``ddof=1``) for each group.
    trials : array_like
        Number of trials in each group.

    Returns
    -------
    statistic : float
        Welch's F statistic.
    df_num, df_denom : float
        Numerator (``k - 1``) and approximate denominator degrees of freedom.
    pval : float
        P-value.

    Raises
    ------
    ValueError
        If a group has zero variance, which leaves its weight undefined.

    Notes
    -----
    Like `welch_test`, this does not assume the groups share a variance. With
    two groups the statistic is the square of `welch_test`'s and the p-values
    agree.

    Welch, B. L. (1951). "On the comparison of several mean values: an
    alternative approach." Biometrika, 38(3/4), 330-336.
    """
    _validate_many_groups(means, variances, trials)
    if any(v <= 0 for v in variances):
        raise ValueError("Welch's ANOVA needs a positive variance in every group")
    k = len(means)
    weights = [n / v for n, v in zip(trials, variances)]
    total_weight = sum(weights)
    weighted_mean = sum(w * m for w, m in zip(weights, means)) / total_weight
    between = sum(w * (m - weighted_mean) ** 2 for w, m in zip(weights, means)) / (k - 1)
    tmp = sum((1 - w / total_weight) ** 2 / (n - 1) for w, n in zip(weights, trials))
    statistic = between / (1 + 2 * (k - 2) / (k * k - 1) * tmp)
    df_denom = (k * k - 1) / (3 * tmp)
    pval = ss.f.sf(statistic, k - 1, df_denom)  # type: ignore[no-untyped-call]
    return float(statistic), float(k - 1), float(df_denom), float(pval)


def score_omnibus_test(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    equal_var: bool = True,
) -> tuple[float, int, float]:
    """Rao's score test that two or more groups share one mean.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The sample variances (``ddof=1``) for each group.
    trials : array_like
        Number of trials in each group.
    equal_var : bool, default=True
        If True, assume every group shares a common variance. If False, give
        each group its own variance.

    Returns
    -------
    statistic : float
        The score statistic.
    df : int
        Degrees of freedom, ``k - 1``.
    pval : float
        P-value.

    Notes
    -----
    The statistic is ``sum(n_i * (mean_i - mu)**2 / sigma2_i)``, where ``mu``
    and ``sigma2_i`` are the MLEs under H0 from `mle_under_equal_means`,
    compared against a chi-squared distribution with ``k - 1`` degrees of
    freedom. With two groups it equals `score_test` with a null lift of 0.
    """
    _validate_many_groups(means, variances, trials)
    mu, sigma2 = mle_under_equal_means(means, variances, trials, equal_var=equal_var)
    sigma2_by_group = sigma2 if isinstance(sigma2, list) else [sigma2] * len(means)
    # As in score_test, a group whose null variance is zero contributes nothing.
    statistic = sum(n * (m - mu) ** 2 / s2 for n, m, s2 in zip(trials, means, sigma2_by_group) if s2 > 1e-12)
    df = len(means) - 1
    pval = ss.chi2.sf(statistic, df=df)  # type: ignore[no-untyped-call]
    return float(statistic), df, float(pval)
