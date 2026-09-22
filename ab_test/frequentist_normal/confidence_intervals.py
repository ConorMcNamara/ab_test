"""Calculates confidence intervals for AB Tests with Normal Data."""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import scipy.stats as ss

from ab_test.frequentist_normal.stats_tests import welch_test
from ab_test.frequentist_normal.utils import observed_lift

__all__ = [
    "confidence_interval",
    "individual_confidence_interval",
    "welch_interval",
    "z_interval",
    "delta_interval",
]


def _search_lower_bound(
    means: Any,
    variances: Any,
    trials: Any,
    lb_lb: float,
    lb_ub: float,
    alpha: float,
    lift: str,
    tol: float,
) -> float:
    eps = 0.01
    while True:
        if lift == "relative" and lb_lb < -1:
            return -1.0
        pval = welch_test(means, variances, trials, null_lift=lb_lb, lift=lift)
        if pval >= alpha:
            lb_ub = lb_lb
            lb_lb -= eps
            eps *= 2
        else:
            break
    while (lb_ub - lb_lb) > tol:
        lb = 0.5 * (lb_lb + lb_ub)
        pval = welch_test(means, variances, trials, null_lift=lb, lift=lift)
        if pval >= alpha:
            lb_ub = lb
        else:
            lb_lb = lb
    return 0.5 * (lb_lb + lb_ub)


def _search_upper_bound(
    means: Any,
    variances: Any,
    trials: Any,
    ub_lb: float,
    ub_ub: float,
    alpha: float,
    lift: str,
    tol: float,
) -> float:
    eps = 0.01
    while True:
        if ub_ub > 100:
            return math.inf
        pval = welch_test(means, variances, trials, null_lift=ub_ub, lift=lift)
        if pval >= alpha:
            ub_lb = ub_ub
            ub_ub += eps
            eps *= 2
        else:
            break
    while (ub_ub - ub_lb) > tol:
        ub = 0.5 * (ub_lb + ub_ub)
        pval = welch_test(means, variances, trials, null_lift=ub, lift=lift)
        if pval >= alpha:
            ub_lb = ub
        else:
            ub_ub = ub
    return 0.5 * (ub_lb + ub_ub)


def confidence_interval(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
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
    method : {'welch', 'z', 'binary_search', 'delta'}
        How we want to calculate the confidence interval.
        ``'welch'`` constructs individual t-intervals per group and
        combines them.  ``'z'`` does the same with z-intervals (large
        sample).  ``'binary_search'`` inverts the Welch test.
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
        lb_lb = ote - 0.01
        lb_ub = ote
        ub_lb = ote
        ub_ub = ote + 0.01
        with ThreadPoolExecutor(max_workers=2) as executor:
            lb_future = executor.submit(
                _search_lower_bound, means, variances, trials,
                lb_lb, lb_ub, alpha, lift, tol,
            )
            ub_future = executor.submit(
                _search_upper_bound, means, variances, trials,
                ub_lb, ub_ub, alpha, lift, tol,
            )
            lb = lb_future.result()
            ub = ub_future.result()
    elif method in ["welch", "z"]:
        if method == "welch":
            t_crit_a = float(ss.t.ppf(1 - alpha / 2, trials[0] - 1))  # type: ignore[no-untyped-call]
            t_crit_b = float(ss.t.ppf(1 - alpha / 2, trials[1] - 1))  # type: ignore[no-untyped-call]
            lower1, upper1 = welch_interval(means[0], variances[0], trials[0], alpha, t_crit_a)
            lower2, upper2 = welch_interval(means[1], variances[1], trials[1], alpha, t_crit_b)
        else:
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
