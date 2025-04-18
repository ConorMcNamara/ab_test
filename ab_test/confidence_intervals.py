"""Calculates confidence intervals for AB Tests."""

import math
from typing import Union

import numpy as np
import scipy.stats as ss

from ab_test.stats_tests import score_test
from ab_test.utils import observed_lift

__all__ = [
    "wilson_interval",
    "confidence_interval",
    "agresti_coull_interval",
    "jeffrey_interval",
    "clopper_pearson_interval",
]


def wilson_interval(s: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson Confidence Interval on binomial proportion

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
    z = ss.norm.isf(alpha / 2)
    z_squared = math.pow(z, 2)
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


def confidence_interval(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    test=score_test,
    alpha: float = 0.05,
    lift: str = "relative",
) -> tuple:
    """Confidence interval for relative lift.

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
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a confidence interval.

    Notes
    -----
    Uses binary search to compute a confidence interval.
    """

    tol = 1e-6
    upper_bound_exists = True
    try:
        ote = observed_lift(trials, successes, lift=lift)
    except ZeroDivisionError:
        ote = 1.0
        upper_bound_exists = False

    if lift == "relative":
        lb_lb = ote - 0.01
        lb_ub = ote
        ub_lb = ote
        ub_ub = ote + 0.01
    else:
        pa = successes[0] / trials[0]
        lb_lb = max(ote - 0.01, -pa)
        lb_ub = ote
        ub_lb = ote
        ub_ub = min(ote + 0.01, 1.0 - pa)

    # Initial search for a lower bound on the lower bound of the
    # confidence interval.
    eps = 0.01
    lower_bound_exists = True
    while True:
        if (lift == "relative" and lb_lb < -1) or (lift == "absolute" and lb_lb < -pa):
            lower_bound_exists = False
            break

        pval = test(trials, successes, null_lift=lb_lb, lift=lift)
        if pval >= alpha:
            # lb_lb is consistent with data; decrease it
            lb_ub = lb_lb
            lb_lb -= eps
            eps *= 2
        else:
            break

    if lower_bound_exists:
        # (lb_lb, lb_ub) is a bound on the lower bound of the confidence interval
        while (lb_ub - lb_lb) > tol:
            lb = 0.5 * (lb_lb + lb_ub)
            pval = test(trials, successes, null_lift=lb, lift=lift)
            if pval >= alpha:
                # lb is consistent with data; expand the interval by
                # decreasing lb.
                lb_ub = lb
            else:
                # lb is rejected by the test; shrink the interval by
                # increasing lb.
                lb_lb = lb

        lb = 0.5 * (lb_lb + lb_ub)
    elif successes[0] > 0:
        lb = -1.0
    else:
        lb = "-Infinity"

    # Initial search for an upper bound on the upper bound of the
    # confidence interval.
    if upper_bound_exists:
        eps = 0.01
        while True:
            if ub_ub > 100:
                upper_bound_exists = False
                break

            pval = test(trials, successes, null_lift=ub_ub, lift=lift)
            if pval >= alpha:
                # ub_ub is consistent with data; increase it
                ub_lb = ub_ub
                ub_ub += eps
                eps *= 2
            else:
                break

    if upper_bound_exists:
        # (ub_lb, ub_ub) is a bound on the upper bound of the confidence interval
        while (ub_ub - ub_lb) > tol:
            ub = 0.5 * (ub_lb + ub_ub)
            pval = test(trials, successes, null_lift=ub, lift=lift)
            if pval >= alpha:
                # ub is consistent with data; expand the interval by
                # increasing ub.
                ub_lb = ub
            else:
                # ub is rejected by the test; shrink the interval by
                # decreasing ub.
                ub_ub = ub

        ub = 0.5 * (ub_lb + ub_ub)
    elif lift == "relative":
        ub = "Infinity"
    else:
        ub = 1.0

    return lb, ub


def agresti_coull_interval(s: int, n: int, alpha: float = 0.05) -> tuple:
    """Agresti-Coull Interval on binomial proportions

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
    z = ss.norm.isf(alpha / 2)
    z_squared = math.pow(z, 2)
    n_tilde = n + z_squared
    p_tilde = (s + z_squared / 2) / n_tilde
    return p_tilde - z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde), p_tilde + z * math.sqrt(
        p_tilde * (1 - p_tilde) / n_tilde
    )


def jeffrey_interval(s: int, n: int, alpha: float = 0.05) -> tuple:
    """Jeffrey's Interval on Binomial Proportions

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
    lb = ss.beta.ppf(alpha / 2, s + 1 / 2, n - s + 1 / 2)
    ub = ss.beta.ppf(1 - alpha / 2, s + 1 / 2, n - s + 1 / 2)
    return lb, ub


def clopper_pearson_interval(s: int, n: int, alpha: float = 0.05) -> tuple:
    """Clopper-Pearson's Interval on Binomial Proportions

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
    lb = ss.beta.ppf(alpha / 2, s, n - s + 1)
    ub = ss.beta.ppf(1 - alpha / 2, s + 1, n - s)
    return lb, ub
