"""Calculates confidence intervals for AB Tests with Normal Data."""

import math
from typing import Any

import numpy as np
import scipy.stats as ss

from ab_test._binary_search import binary_search_interval
from ab_test.frequentist_normal.stats_tests import welch_test
from ab_test.frequentist_normal.utils import observed_lift

__all__ = [
    "confidence_interval",
    "individual_confidence_interval",
    "welch_interval",
    "z_interval",
    "delta_interval",
]


def confidence_interval(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    test: Any = welch_test,
    method: str = "welch",
    alpha: float = 0.05,
    lift: str = "relative",
    tol: float = 1e-06,
) -> tuple[Any, ...]:
    """Calculate confidence intervals using the chosen method.

    Parameters
    ----------
    means : array_like
        Means of each group.
    variances : array_like
        Variances of each group.
    trials : array_like
        Number of trials in each group.
    test : function
        The significance test inverted by ``'binary_search'``, e.g.
        `welch_test` or `score_test`. Ignored by the other methods.
    method : {'welch', 'z', 'binary_search', 'delta'}
        How we want to calculate the confidence interval.
        ``'welch'`` is a t-interval on the combined standard error with
        Welch-Satterthwaite degrees of freedom.  ``'z'`` constructs
        individual z-intervals per group and combines them (large
        sample).  ``'binary_search'`` inverts ``test``.
        ``'delta'`` uses the delta method directly.
    alpha : float
        Threshold for significance. The confidence interval will have
        level 100(1-alpha)%. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    lift : {"relative", "absolute"}
        Whether to compute the interval for relative or absolute lift.
    tol : float, default=1e-06
        The tolerance for binary search. Lower values means narrower CIs.

    Returns
    -------
    ci_low, ci_high : float
        Lower and upper bounds on a confidence interval.
    """
    ote = observed_lift(means, trials, lift=lift)
    lb: float
    ub: float
    if method == "binary_search":
        lb_found, ub_found = binary_search_interval(
            lambda d: test(means, variances, trials, null_lift=d, lift=lift),
            (ote - 0.01, ote),
            (ote, ote + 0.01),
            alpha=alpha,
            lower_limit=-1.0 if lift == "relative" else -math.inf,
            upper_limit=100.0 if lift == "relative" else math.inf,
            tol=tol,
        )
        lb = lb_found if lb_found is not None else -1.0
        ub = ub_found if ub_found is not None else math.inf
    elif method == "welch":
        lb, ub = _welch_satterthwaite_interval(means, variances, trials, alpha, lift)
    elif method == "z":
        z_crit = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
        lower1, upper1 = z_interval(means[0], variances[0], trials[0], alpha, z_crit)
        lower2, upper2 = z_interval(means[1], variances[1], trials[1], alpha, z_crit)
        var_mean_a = ((upper1 - lower1) / 2) ** 2
        var_mean_b = ((upper2 - lower2) / 2) ** 2
        if lift == "relative":
            mean_a = means[0]
            mean_b = means[1]
            var_g = var_mean_b / (mean_a**2) + (mean_b**2) * var_mean_a / (mean_a**4)
        else:
            var_g = var_mean_a + var_mean_b
        lb = ote - math.sqrt(var_g)
        ub = ote + math.sqrt(var_g)
    elif method == "delta":
        lb, ub = delta_interval(means, variances, trials, alpha, lift)
    else:
        raise NotImplementedError(f"No support for {method} method of generating confidence intervals")
    return lb, ub


def individual_confidence_interval(
    m: float,
    v: float,
    n: int,
    alpha: float = 0.05,
    method: str = "welch",
) -> tuple[Any, ...]:
    """Calculate confidence intervals for individual cells.

    Parameters
    ----------
    m : float
        The sample mean.
    v : float
        The sample variance.
    n : int
        The number of observations.
    alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    method : {"welch", "z"}
        The method for calculating individual confidence intervals.

    Returns
    -------
    A confidence interval for our individual cell.
    """
    method = method.casefold()
    if method == "welch":
        lb, ub = welch_interval(m, v, n, alpha)
    elif method == "z":
        lb, ub = z_interval(m, v, n, alpha)
    else:
        raise ValueError(f"No support for calculating confidence interval using {method}")
    return lb, ub


def welch_interval(
    m: float,
    v: float,
    n: int,
    alpha: float = 0.05,
    t_crit: float | None = None,
) -> tuple[Any, ...]:
    """t-interval for a single group mean.

    Parameters
    ----------
    m : float
        The sample mean.
    v : float
        The sample variance.
    n : int
        The number of observations.
    alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    t_crit : float or None
        Pre-computed critical value. When ``None``, computed from ``n - 1``
        degrees of freedom.

    Returns
    -------
    lb, ub : float
        Lower and upper bounds of a 100(1-``alpha``)% confidence interval
        on the mean.
    """
    if t_crit is None:
        t_crit = float(ss.t.ppf(1 - alpha / 2, n - 1))  # type: ignore[no-untyped-call]
    se = math.sqrt(v / n)
    lb = m - t_crit * se
    ub = m + t_crit * se
    return lb, ub


def z_interval(
    m: float,
    v: float,
    n: int,
    alpha: float = 0.05,
    z_crit: float | None = None,
) -> tuple[Any, ...]:
    """z-interval for a single group mean (large-sample approximation).

    Parameters
    ----------
    m : float
        The sample mean.
    v : float
        The sample variance.
    n : int
        The number of observations.
    alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    z_crit : float or None
        Pre-computed critical value. When ``None``, computed from the
        standard normal distribution.

    Returns
    -------
    lb, ub : float
        Lower and upper bounds of a 100(1-``alpha``)% confidence interval
        on the mean.
    """
    if z_crit is None:
        z_crit = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
    se = math.sqrt(v / n)
    lb = m - z_crit * se
    ub = m + z_crit * se
    return lb, ub


def delta_interval(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    alpha: float = 0.05,
    lift: str = "relative",
) -> tuple[Any, ...]:
    """Confidence interval using the delta method.

    Parameters
    ----------
    means : array_like
        Means of each group.
    variances : array_like
        Variances of each group.
    trials : array_like
        Number of trials in each group.
    alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    lift : {"relative", "absolute"}
        Whether to compute the interval for relative or absolute lift.

    Returns
    -------
    lb, ub : float
        Lower and upper bounds on a confidence interval.
    """
    mean_a, mean_b = means[0], means[1]
    var_mean_a = variances[0] / trials[0]
    var_mean_b = variances[1] / trials[1]
    if lift == "relative":
        diff = (mean_b - mean_a) / mean_a
        var_g = var_mean_b / (mean_a**2) + (mean_b**2) * var_mean_a / (mean_a**4)
    else:
        diff = mean_b - mean_a
        var_g = var_mean_a + var_mean_b
    z = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
    se_g = math.sqrt(var_g)
    lb = diff - z * se_g
    ub = diff + z * se_g
    return lb, ub


def _welch_satterthwaite_interval(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    alpha: float = 0.05,
    lift: str = "relative",
) -> tuple[float, float]:
    """Welch t-interval for the difference or ratio of two means.

    The standard error combines both groups, and a single t critical value
    is taken at the Welch-Satterthwaite degrees of freedom. For absolute lift
    this is the interval that inverts `welch_test`. For relative lift the
    standard error comes from the delta method, and the degrees of freedom
    from the same Satterthwaite approximation applied to its two terms.
    """
    mean_a, mean_b = means[0], means[1]
    s_a = variances[0] / trials[0]
    s_b = variances[1] / trials[1]
    if lift == "relative":
        diff = (mean_b - mean_a) / mean_a
        term_a = mean_b**2 * s_a / mean_a**4
        term_b = s_b / mean_a**2
    else:
        diff = mean_b - mean_a
        term_a, term_b = s_a, s_b
    var_g = term_a + term_b
    df = var_g**2 / (term_a**2 / (trials[0] - 1) + term_b**2 / (trials[1] - 1))
    t_crit = float(ss.t.isf(alpha / 2, df))  # type: ignore[no-untyped-call]
    half_width = t_crit * math.sqrt(var_g)
    return float(diff - half_width), float(diff + half_width)
