"""Calculates confidence intervals for AB Tests."""

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import scipy.stats as ss

from ab_test._lift import _SCALED_LIFTS, scale_bounds
from ab_test.frequentist_binomial.stats_tests import score_test
from ab_test.frequentist_binomial.utils import observed_lift

__all__ = [
    "confidence_interval",
    "individual_confidence_interval",
    "wilson_interval",
    "agresti_coull_interval",
    "jeffrey_interval",
    "clopper_pearson_interval",
    "wald_interval",
    "delta_interval",
]


# Searches stop this far inside a lift's limit: at the limit itself (a ratio of
# 0, or a difference of +/-1) the constrained MLE is degenerate. It matches the
# default bisection tolerance.
_LIMIT_PROBE = 1e-6


def _search_lower_bound(
    test: Any,
    trials: Any,
    successes: Any,
    lb_lb: float,
    lb_ub: float,
    alpha: float,
    lift: str,
    tol: float,
) -> float:
    # Both lifts are bounded below by -1. The observed control rate is not a
    # bound for absolute lift, since the true rate can exceed it.
    floor = -1.0 + _LIMIT_PROBE
    if lb_ub <= floor:
        return -1.0
    eps = 0.01
    while True:
        # Clamp rather than step past the floor, so a bound between the last
        # accepted value and the floor is still found.
        lb_lb = max(lb_lb, floor)
        pval = test(trials, successes, null_lift=lb_lb, lift=lift)
        if pval < alpha:
            break
        if lb_lb <= floor:
            return -1.0
        lb_ub = lb_lb
        lb_lb -= eps
        eps *= 2
    while (lb_ub - lb_lb) > tol:
        lb = 0.5 * (lb_lb + lb_ub)
        pval = test(trials, successes, null_lift=lb, lift=lift)
        if pval >= alpha:
            lb_ub = lb
        else:
            lb_lb = lb
    return 0.5 * (lb_lb + lb_ub)


def _search_upper_bound(
    test: Any,
    trials: Any,
    successes: Any,
    ub_lb: float,
    ub_ub: float,
    alpha: float,
    lift: str,
    tol: float,
    upper_bound_exists: bool,
) -> float:
    unbounded = math.inf if lift == "relative" else 1.0
    if not upper_bound_exists:
        return unbounded
    # Absolute lift is bounded above by 1; relative lift has no bound, so the
    # search gives up past a 100x lift.
    ceiling = 100.0 if lift == "relative" else 1.0 - _LIMIT_PROBE
    if ub_lb >= ceiling:
        return unbounded
    eps = 0.01
    while True:
        # Clamp rather than step past the ceiling, so a bound between the last
        # accepted value and the ceiling is still found.
        ub_ub = min(ub_ub, ceiling)
        pval = test(trials, successes, null_lift=ub_ub, lift=lift)
        if pval < alpha:
            break
        if ub_ub >= ceiling:
            return unbounded
        ub_lb = ub_ub
        ub_ub += eps
        eps *= 2
    while (ub_ub - ub_lb) > tol:
        ub = 0.5 * (ub_lb + ub_ub)
        pval = test(trials, successes, null_lift=ub, lift=lift)
        if pval >= alpha:
            ub_lb = ub
        else:
            ub_ub = ub
    return 0.5 * (ub_lb + ub_ub)


def _mover_ratio_interval(
    p_A: float, lower1: float, upper1: float, p_B: float, lower2: float, upper2: float
) -> tuple[float, float]:
    """MOVER interval for the ratio p_B / p_A from single-proportion intervals.

    Donner, A. & Zou, G. Y. (2012). "Closed-form confidence intervals for
    functions of the normal mean and standard deviation." Section 3.
    """
    prod = p_A * p_B
    lb_denom = upper1 * (2 * p_A - upper1)
    ub_denom = lower1 * (2 * p_A - lower1)
    lb_disc = max(prod**2 - lower2 * upper1 * (2 * p_B - lower2) * (2 * p_A - upper1), 0.0)
    ub_disc = max(prod**2 - upper2 * lower1 * (2 * p_B - upper2) * (2 * p_A - lower1), 0.0)
    lb = (prod - math.sqrt(lb_disc)) / lb_denom if lb_denom != 0 else 0.0
    ub = (prod + math.sqrt(ub_disc)) / ub_denom if ub_denom > 0 else math.inf
    return max(lb, 0.0), ub


def confidence_interval(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    test: Any = score_test,
    alpha: float = 0.05,
    lift: str = "relative",
    method: str = "binary_search",
    tol: float = 1e-06,
    *,
    spend: float | None = None,
    msrp: float | None = None,
) -> tuple[Any, ...]:
    """Calculate confidence intervals using the chosen method.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     test : function
        A function implementing a significance test. Defaults to
        `score_test`. This function should in turn have arguments,
        trials, successes, null_lift, and return a p-value.
     alpha : float
        Threshold for significance. The confidence interval will have
        level 100(1-alpha)%. Defaults to 0.05, corresponding to a 95%
        confidence interval.
     lift : ["relative", "absolute", "incremental", "roas", "revenue", "cpa"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`. The scaled lifts (incremental, roas,
        revenue, cpa) are computed on the absolute scale and then scaled by
        ``max(trials)`` and converted with ``spend`` or ``msrp``.
    method : {'binary_search', "wilson", "jeffrey", "agresti-coull", "clopper-pearson", 'wald', 'delta'}
        How we want to calculate the confidence interval
    tol : float, default=1e-06
        The tolerance for our binary search. Lower values means narrower CIs
    spend : float, optional
        Campaign spend. Required for ``lift="roas"`` and ``lift="cpa"``.
    msrp : float, optional
        Revenue per unit. Required for ``lift="revenue"``.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a confidence interval.

    Notes
    -----
    ``binary_search`` inverts ``test``. ``wilson``, ``jeffrey``,
    ``agresti-coull`` and ``clopper-pearson`` combine the two single-group
    intervals with the MOVER method: Newcombe's hybrid interval for absolute
    lift, and Donner & Zou's ratio interval for relative lift. ``wald`` and
    ``delta`` use the delta method.
    """
    if lift in _SCALED_LIFTS:
        # Build the interval where the variance lives (proportions), then convert its bounds.
        abs_lb, abs_ub = confidence_interval(trials, successes, test, alpha, "absolute", method, tol)
        scale = max(trials)
        return scale_bounds(abs_lb * scale, abs_ub * scale, lift, spend, msrp)
    try:
        ote = observed_lift(trials, successes, lift=lift)
        upper_bound_exists = True
    except ZeroDivisionError:
        ote = 1.0
        upper_bound_exists = False
    lb: float
    ub: float
    if method == "binary_search":
        if test.__name__ in ["score_test", "likelihood_ratio_test", "z_test", "wald_test", "msprt_test"]:
            if lift == "relative":
                lb_lb = ote - 0.01
                lb_ub = ote
                ub_lb = ote
                ub_ub = ote + 0.01
            else:
                lb_lb = max(ote - 0.01, -1.0)
                lb_ub = ote
                ub_lb = ote
                ub_ub = min(ote + 0.01, 1.0)

            with ThreadPoolExecutor(max_workers=2) as executor:
                lb_future = executor.submit(
                    _search_lower_bound, test, trials, successes, lb_lb, lb_ub, alpha, lift, tol
                )
                ub_future = executor.submit(
                    _search_upper_bound, test, trials, successes, ub_lb, ub_ub, alpha, lift, tol, upper_bound_exists
                )
                lb = lb_future.result()
                ub = ub_future.result()
        else:
            raise NotImplementedError(f"binary_search is not implemented for {test}")
    else:
        if method in ["wilson", "jeffrey", "agresti-coull", "clopper-pearson"]:
            single_intervals: dict[str, Callable[..., tuple[Any, ...]]] = {
                "wilson": wilson_interval,
                "jeffrey": jeffrey_interval,
                "agresti-coull": agresti_coull_interval,
                "clopper-pearson": clopper_pearson_interval,
            }
            single_interval = single_intervals[method]
            p_A = successes[0] / trials[0]
            p_B = successes[1] / trials[1]
            lower1, upper1 = single_interval(successes[0], trials[0], alpha)
            lower2, upper2 = single_interval(successes[1], trials[1], alpha)
            if lift == "relative":
                lb, ub = _mover_ratio_interval(p_A, lower1, upper1, p_B, lower2, upper2)
                lb, ub = lb - 1, ub - 1
            else:
                # Newcombe's hybrid score interval (MOVER) for the difference
                lb = ote - math.sqrt((p_B - lower2) ** 2 + (upper1 - p_A) ** 2)
                ub = ote + math.sqrt((upper2 - p_B) ** 2 + (p_A - lower1) ** 2)
        elif method == "wald":
            # Wald intervals are symmetric, so combining half-widths is the delta method.
            lower1, upper1 = wald_interval(successes[0], trials[0], alpha)
            lower2, upper2 = wald_interval(successes[1], trials[1], alpha)
            var_pA = math.pow((upper1 - lower1) / 2, 2)
            var_pB = math.pow((upper2 - lower2) / 2, 2)
            if lift == "relative":
                p_A = successes[0] / trials[0]
                p_B = successes[1] / trials[1]
                var_g = var_pB / (p_A**2) + (p_B**2) * var_pA / (p_A**4)
            else:
                var_g = var_pA + var_pB
            lb = ote - math.sqrt(var_g)
            ub = ote + math.sqrt(var_g)
        elif method == "delta":
            lb, ub = delta_interval(trials, successes, alpha, lift)
        else:
            raise NotImplementedError(f"No support for {method} method of generating confidence intervals")
    return lb, ub


def individual_confidence_interval(s: int, n: int, alpha: float = 0.05, method: str = "wilson") -> tuple[Any, ...]:
    """Calculate confidence intervals for individual cells.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    method : {"wilson", "agresti-coull", "jeffrey", "clopper-pearson", "wald"}
            The method for calculating individual confidence intervals

    Returns
    -------
    A confidence interval for our individual cell
    """
    method = method.casefold()
    if method == "wilson":
        lb, ub = wilson_interval(s, n, alpha)
    elif method == "agresti-coull":
        lb, ub = agresti_coull_interval(s, n, alpha)
    elif method == "jeffrey":
        lb, ub = jeffrey_interval(s, n, alpha)
    elif method == "clopper-pearson":
        lb, ub = clopper_pearson_interval(s, n, alpha)
    elif method == "wald":
        lb, ub = wald_interval(s, n, alpha)
    else:
        raise ValueError(f"No support for calculating confidence interval using {method}")
    return lb, ub


def agresti_coull_interval(s: int, n: int, alpha: float = 0.05, z: float | None = None) -> tuple[Any, ...]:
    """Agresti-Coull Interval on Binomial Proportions.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    This returns a confidence interval on `p_tilde`
    """
    if z is None:
        z = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
    z_squared = z * z
    n_tilde = n + z_squared
    p_tilde = (s + z_squared / 2) / n_tilde
    half_width = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(p_tilde - half_width, 0.0), min(p_tilde + half_width, 1.0)


def jeffrey_interval(s: int, n: int, alpha: float = 0.05) -> tuple[Any, ...]:
    """Jeffrey's Interval on Binomial Proportions.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    This assumes a Beta distribution of (1 / 2, 1 / 2)
    """
    lb = ss.beta.ppf(alpha / 2, s + 1 / 2, n - s + 1 / 2)  # type: ignore[no-untyped-call]
    ub = ss.beta.ppf(1 - alpha / 2, s + 1 / 2, n - s + 1 / 2)  # type: ignore[no-untyped-call]
    return lb, ub


def clopper_pearson_interval(s: int, n: int, alpha: float = 0.05) -> tuple[Any, ...]:
    """Clopper-Pearson's Interval on Binomial Proportions.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    This is based off the Beta representation of Clopper-Pearson's formula. Note that Clopper-Pearson is
    an exact method, meaning that its intervals can be wider than other methods like Wilson or Jeffrey
    """
    lb = 0.0 if s == 0 else ss.beta.ppf(alpha / 2, s, n - s + 1)  # type: ignore[no-untyped-call]
    ub = 1.0 if s == n else ss.beta.ppf(1 - alpha / 2, s + 1, n - s)  # type: ignore[no-untyped-call]
    return lb, ub


def wald_interval(s: int, n: int, alpha: float = 0.05, z: float | None = None) -> tuple[Any, ...]:
    """Compute the Wald Interval on Binomial Proportions.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    This is based off the Wald formula. Note that this formula is fragile to proportions near 0 or 1.
    """
    p_hat = s / n
    if z is None:
        z = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
    lb = p_hat - z * math.sqrt(p_hat * (1 - p_hat) / n)
    ub = p_hat + z * math.sqrt(p_hat * (1 - p_hat) / n)
    return lb, ub


def delta_interval(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    alpha: float,
    lift: str = "relative",
) -> tuple[Any, ...]:
    """Compute the confidence interval for Binomial Proportions using the Delta Method.

    Parameters
    ----------
    trials : array_like
        Number of trials in each group.
    successes : array_like
        Number of successes in each group.
    alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.
    lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.

    Returns
    -------
    lb, ub : float
        Lower and upper bounds on a confidence interval.
    """
    p1_hat = successes[1] / trials[1]
    p2_hat = successes[0] / trials[0]
    if lift == "relative":
        diff = (p1_hat - p2_hat) / p2_hat

        def dg_dp1(p2: float) -> float:
            return float(1 / p2)

        def dg_dp2(p1: float, p2: float) -> float:
            return float(-p1 / math.pow(p2, 2))
    else:
        diff = p1_hat - p2_hat

        def dg_dp1(p2: float) -> float:  # To maintain compatibility with relative
            return 1.0

        def dg_dp2(p1: float, p2: float) -> float:  # To maintain compatibility with relative
            return -1.0

    var_p1 = p1_hat * (1 - p1_hat) / trials[1]
    var_p2 = p2_hat * (1 - p2_hat) / trials[0]
    cov_p1_p2 = 0  # Covariance is 0 for independent samples
    var_g = (
        math.pow(dg_dp1(p2_hat), 2) * var_p1
        + math.pow(dg_dp2(p1_hat, p2_hat), 2) * var_p2
        + 2 * dg_dp1(p2_hat) * dg_dp2(p1_hat, p2_hat) * cov_p1_p2
    )
    se_g = np.sqrt(var_g)
    z = ss.norm.isf(alpha / 2)  # type: ignore[no-untyped-call]  # Calculate the z-score
    lb = diff - z * se_g
    ub = diff + z * se_g
    return lb, ub


def wilson_interval(s: int, n: int, alpha: float = 0.05, z: float | None = None) -> tuple[Any, ...]:
    """Wilson Confidence Interval on Binomial Proportion.

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    Assuming s ~ Binom(n, p), this function returns a confidence interval on p.
    """
    if z is None:
        z = float(ss.norm.isf(alpha / 2))  # type: ignore[no-untyped-call]
    z_squared = z * z
    p_hat = s / n
    ctr = p_hat + z_squared / (2 * n)
    inner_width = (p_hat * (1 - p_hat) + z_squared / (4 * n)) / n
    denom = 1 + z_squared / n
    wdth = z * math.sqrt(inner_width)
    # Alternative implementation using n_s and n_f
    # ctr = (s + 0.5 * z_squared) / (n + z_squared)
    # inner_wdth = s * (n - s) / n + z_squared / 4
    # wdth = (z / (n + z_squared)) * math.sqrt(inner_wdth)
    return (ctr - wdth) / denom, (ctr + wdth) / denom
