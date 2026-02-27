"""Methods to calculate the power of a test"""

import numpy as np
import scipy.stats as ss

from ab_test.binomial.utils import simple_hypothesis_from_composite

__all__ = [
    "score_power",
    "abtest_power",
    "minimum_detectable_lift",
    "required_sample_size",
]


def score_power(
    n: np.ndarray | list,
    p_null: np.ndarray | list,
    p_alt: np.ndarray | list,
    alpha: float = 0.05,
) -> float:
    """Power of Rao's Score Test

    Parameters
    ----------
     n : array_like
        Number of experimental units in each group.
     p_null : array_like
        Probability of success in each group under the null
        hypothesis.
     p_alt : array_like
        Probability of success in each group under the alternative
        hypothesis.
     alpha : float
        Type-I error rate. Defaults to 0.05

    Returns
    -------
     power : float
        The power of the test.

    Notes
    -----
    Rao's score test is the same as Pearson's chi-squared test for 2x2
    contingency tables, so the power has a nice simple form.
    """
    nc = 0.0
    for ni, null, alt in zip(n, p_null, p_alt):
        nc += ni * (null - alt) * (null - alt) / (null * (1.0 - null))
    return ss.ncx2.sf(ss.chi2.isf(alpha, df=1), df=1, nc=nc)


def abtest_power(
    group_sizes: np.ndarray | list,
    baseline: float,
    alt_lift: float,
    alpha: float = 0.05,
    null_lift: float = 0.0,
    power=score_power,
    lift: str = "relative",
):
    """Power associated with an A/B Test

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Lift associated with alternative hypothesis.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     power : float
        The power of the test.
    """
    if len(group_sizes) > 2:
        # Get two smallest groups -- this governs the overall power
        a, b, *_ = np.partition(group_sizes, 1)
        group_sizes = [a, b]

    p_null, p_alt = simple_hypothesis_from_composite(group_sizes, baseline, null_lift, alt_lift, lift=lift)
    return power(group_sizes, p_null, p_alt, alpha=alpha)


def minimum_detectable_lift(
    group_sizes: np.ndarray | list,
    baseline: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    power=score_power,
    drop: bool = False,
    lift: str = "relative",
):
    """Minimum detectable lift.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     beta : float, optional
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     drop : boolean, optional
        If True, the minimum detectable drop will be returned.
        Defaults to False, returning the minimum detectable lift.
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     mdl : float
        Minimum detectable lift/drop associated with test. If `lift` is
        "relative", this will be in relative terms, otherwise it will be in
        absolute terms.

    Notes
    -----
    Uses binary search to compute the smallest lift/drop with adequate
    power.
    """

    tol = 1e-6

    # Find an extremum bound on the MDL
    mdl_inner = 0.0
    if drop:
        if lift == "relative":
            mdl_extremum = -0.2
        else:
            mdl_extremum = -0.99 * baseline
    else:
        if lift == "relative":
            mdl_extremum = 0.2
        else:
            mdl_extremum = 0.99 * (1 - baseline)

    pwr = abtest_power(
        group_sizes,
        baseline,
        mdl_extremum,
        alpha=alpha,
        null_lift=null_lift,
        power=power,
        lift=lift,
    )

    while pwr < 1 - beta:
        mdl_inner = mdl_extremum
        if lift == "relative":
            mdl_extremum *= 2
        else:
            mdl_extremum = 0.5 * (-baseline + mdl_extremum)

        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl_extremum,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )

    while abs(mdl_extremum - mdl_inner) > tol:
        mdl = 0.5 * (mdl_inner + mdl_extremum)
        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase mdl
            mdl_inner = mdl
        else:
            # Adequate power, decrease mdl
            mdl_extremum = mdl

    if drop:
        mdl_extremum *= -1.0

    return mdl_extremum


def required_sample_size(
    baseline: float,
    alt_lift: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    group_proportions: np.ndarray | list | None = None,
    null_lift: float = 0.0,
    power=score_power,
    lift: str = "relative",
):
    """Required sample size.

    Parameters
    ----------
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Relative lift (second group relative to first) associated with
        alternative hypothesis.
     alpha : float
        Type-I error rate threshold. Defaults to 0.05.
     beta : float
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     group_proportions : array_like or None
        Fraction of experimental units in each group. If None
        (default), will use an even split.
     null_lift : float
        Relative lift (second group relative to first) associated with
        null hypothesis. Defaults to 0.0.
     power : function
        Function that computes power, such as `score_power` (default).
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     sample_size : int
        Minimum sample size, across all experiment groups, required to
        have desired sensitivity.

    Notes
    -----
    Uses binary search to compute the smallest sample size with
    adequate power.
    """

    tol = 0.01

    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    def sample_size_to_group_sizes(ss):
        return [int(ss * g) for g in group_proportions]

    # Find an upper bound on the required sample size
    ss_lower = 0
    ss_upper = 1000

    pwr = abtest_power(
        sample_size_to_group_sizes(ss_upper),
        baseline,
        alt_lift,
        alpha=alpha,
        null_lift=null_lift,
        power=power,
        lift=lift,
    )

    while pwr < 1 - beta:
        ss_lower = ss_upper
        ss_upper *= 2
        pwr = abtest_power(
            sample_size_to_group_sizes(ss_upper),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )

    while ss_upper - ss_lower > tol * ss_lower:
        ss = int(0.5 * (ss_lower + ss_upper))
        pwr = abtest_power(
            sample_size_to_group_sizes(ss),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase ss
            ss_lower = ss
        else:
            # Adequate power, decrease ss
            ss_upper = ss

    return ss_upper
