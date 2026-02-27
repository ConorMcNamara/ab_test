"""Calculates confidence intervals for AB Tests."""

import math

import numpy as np
import scipy.stats as ss

from ab_test.binomial.stats_tests import score_test
from ab_test.binomial.utils import observed_lift

__all__ = [
    "confidence_interval",
    "individual_confidence_interval",
    "wilson_interval",
    "agresti_coull_interval",
    "jeffrey_interval",
    "clopper_pearson_interval",
    "wald_interval",
]


def confidence_interval(
    trials: np.ndarray | list,
    successes: np.ndarray | list,
    test=score_test,
    alpha: float = 0.05,
    lift: str = "relative",
    method: str = "binary_search",
    tol: float = 1e-06,
) -> tuple:
    """Wrapper for calculating confidence intervals

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
     lift : ["relative", "absolute", "incremental", "roas", "revenue"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
    method : {'binary_search', "wilson", "jeffrey", "agresti-coull", "clopper-pearson", 'wald'}
        How we want to calculate the confidence interval
    tol : float, default=1e-06
        The tolerance for our binary search. Lower values means narrower CIs

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a confidence interval.

    Notes
    -----
    Uses binary search to compute a confidence interval.
    """
    try:
        ote = observed_lift(trials, successes, lift=lift)
        upper_bound_exists = True
    except ZeroDivisionError:
        ote = 1.0
        upper_bound_exists = False
    lb: float | str
    ub: float | str
    if method == "binary_search":
        if test.__name__ in ["score_test", "likelihood_ratio_test", "z_test"]:
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
        else:
            raise NotImplementedError(f"binary_search is not implemented for {test}")
    else:
        if method in ["wilson", "jeffrey", "agresti-coull", "clopper-pearson", "wald"]:
            if method == "wilson":
                lower1, upper1 = wilson_interval(successes[0], trials[0], alpha)
                lower2, upper2 = wilson_interval(successes[1], trials[1], alpha)
            elif method == "jeffrey":
                lower1, upper1 = jeffrey_interval(successes[0], trials[0], alpha)
                lower2, upper2 = jeffrey_interval(successes[1], trials[1], alpha)
            elif method == "agresti-coull":
                lower1, upper1 = agresti_coull_interval(successes[0], trials[0], alpha)
                lower2, upper2 = agresti_coull_interval(successes[1], trials[1], alpha)
            elif method == "clopper-pearson":
                lower1, upper1 = clopper_pearson_interval(successes[0], trials[0], alpha)
                lower2, upper2 = clopper_pearson_interval(successes[1], trials[1], alpha)
            else:
                lower1, upper1 = wald_interval(successes[0], trials[0], alpha)
                lower2, upper2 = wald_interval(successes[1], trials[1], alpha)
            if lift == "relative" and method != "delta":
                lower1 /= successes[0] / trials[0]
                lower2 /= successes[0] / trials[0]
                upper1 /= successes[0] / trials[0]
                upper2 /= successes[0] / trials[0]
            var_p1 = math.pow((upper1 - lower1) / 2, 2)
            var_p2 = math.pow((upper2 - lower2) / 2, 2)
            lb = ote - math.sqrt(var_p1 + var_p2)
            ub = ote + math.sqrt(var_p1 + var_p2)
        elif method == "delta":
            lb, ub = delta_interval(trials, successes, alpha, lift)
        else:
            raise NotImplementedError(f"No support for {method} method of generating confidence intervals")
    return lb, ub


def individual_confidence_interval(s: int, n: int, alpha: float = 0.05, method: str = "wilson") -> tuple:
    """A wrapper for calculating confidence intervals for individual cells

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


def wald_interval(s: int, n: int, alpha: float = 0.05) -> tuple:
    """The Wald Interval on Binomial Proportions

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
    z = ss.norm.isf(alpha / 2)
    lb = p_hat - z * math.sqrt(p_hat * (1 - p_hat) / n)
    ub = p_hat + z * math.sqrt(p_hat * (1 - p_hat) / n)
    return lb, ub


def delta_interval(
    trials: np.ndarray | list, successes: np.ndarray | list, alpha: float, lift: str = "relative"
) -> tuple:
    """The confidence interval for Binomial Proportions using the Delta Method

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

        def dg_dp1(p2):
            return 1 / p2

        def dg_dp2(p1, p2):
            return -p1 / math.pow(p2, 2)
    else:
        diff = p1_hat - p2_hat

        def dg_dp1(p2):  # To maintain compatibility with relative
            return 1

        def dg_dp2(p1, p2):  # To maintain compatibility with relative
            return -1

    var_p1 = p1_hat * (1 - p1_hat) / trials[1]
    var_p2 = p2_hat * (1 - p2_hat) / trials[0]
    cov_p1_p2 = 0  # Covariance is 0 for independent samples
    var_g = (
        math.pow(dg_dp1(p2_hat), 2) * var_p1
        + math.pow(dg_dp2(p1_hat, p2_hat), 2) * var_p2
        + 2 * dg_dp1(p2_hat) * dg_dp2(p1_hat, p2_hat) * cov_p1_p2
    )
    se_g = np.sqrt(var_g)
    z = ss.norm.isf(alpha / 2)  # Calculate the z-score
    lb = diff - z * se_g
    ub = diff + z * se_g
    return lb, ub


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
