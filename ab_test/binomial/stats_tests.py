"""Statistical tests to determine significance"""

import math
from typing import Optional, Union

import numpy as np
import scipy.stats as ss

from ab_test.binomial.utils import mle_under_null, mle_under_alternative

__all__ = [
    "ab_test",
    "score_test",
    "likelihood_ratio_test",
    "z_test",
    "fisher_test",
    "barnard_exact_test",
    "boschloo_exact_test",
    "modified_log_likelihood_test",
    "freeman_tukey_test",
    "neyman_test",
    "cressie_read_test",
]


def ab_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.array, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
    method: str = "score_test",
) -> Union[float, bool]:
    """A wrapper for our different statistical tests

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
     crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.
    method : {'score', 'likelihood', 'z', 'fisher', 'barnard', 'boschloo', 'modified_likelihood', 'freeman-tukey', 'neyman', 'cressie-read'}
        How we plan on calculating the p_value or critical value of our experiment

    Returns
    -------
     pval : float
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")
    method = method.casefold()
    if method == "score":
        val = score_test(trials, successes, null_lift, lift, crit)
    elif method == "likelihood":
        val = likelihood_ratio_test(trials, successes, null_lift, lift, crit)
    elif method == "z":
        val = z_test(trials, successes, null_lift, lift, crit)
    elif method == "fisher":
        val = fisher_test(trials, successes, null_lift, lift, crit)
    elif method == "barnard":
        val = barnard_exact_test(trials, successes, null_lift, lift, crit)
    elif method == "boschloo":
        val = boschloo_exact_test(trials, successes, null_lift, lift, crit)
    elif method == "modified_likelihood":
        val = modified_log_likelihood_test(trials, successes, null_lift, lift, crit)
    elif method == "freeman-tukey":
        val = freeman_tukey_test(trials, successes, null_lift, lift, crit)
    elif method == "neyman":
        val = neyman_test(trials, successes, null_lift, lift, crit)
    elif method == "cressie-read":
        val = cressie_read_test(trials, successes, null_lift, lift, crit)
    else:
        raise ValueError(f"No support for calculating the p-value and critical value of {method}")
    return val


def score_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.array, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Rao's score test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    p = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)

    if min(p) <= 1e-12 or max(p) + 1e-12 >= 1.0:
        return 1.0

    # Score
    u = [(si - ni * pi) / (pi * (1 - pi)) for ni, si, pi in zip(trials, successes, p)]

    # Fisher information
    diagI = [ni / (pi * (1 - pi)) for ni, pi in zip(trials, p)]

    # Test statistic
    ts = sum([ui * ui / ii for ui, ii in zip(u, diagI)])

    if crit is None:
        # Note: this line takes 80% of the time for the score test, including the
        # MLE, which is really fast! So there's no real point in optimizing
        # anything else here. On the other hand, if we can optimize this line, then
        # great!
        pval = ss.chi2.sf(ts, df=1)
        return pval
    return ts >= crit


def likelihood_ratio_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Likelihood ratio test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    if min(p0) <= 1e-12 or max(p0) + 1e-12 >= 1.0:
        return 1.0

    if min(p1) <= 1e-12 or max(p1) + 1e-12 >= 1.0:
        return 1.0

    def log_likelihood(p):
        return sum([si * np.log(pi) + (ti - si) * np.log(1 - pi) for (si, ti, pi) in zip(successes, trials, p)])

    ts = 2 * (log_likelihood(p1) - log_likelihood(p0))
    if crit is None:
        pval = ss.chi2.sf(ts, df=1)
        return pval
    return ts >= crit


def z_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Z test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    sigma2 = sum([p_i * (1 - p_i) / t_i for (p_i, t_i) in zip(p0, trials)])
    z = (p1[1] - p1[0] - null_lift) / math.sqrt(sigma2)
    if crit is None:
        return 2.0 * ss.norm.cdf(-abs(z))
    return abs(z) >= crit


def fisher_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Fisher's Exact Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    statistic, pval = ss.fisher_exact(contingency_table)

    if crit is None:
        return pval
    return abs(statistic) >= crit


def barnard_exact_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Barnard's Exact Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    barnard = ss.barnard_exact(contingency_table)
    statistic, pval = barnard.statistic, barnard.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit


def boschloo_exact_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Boschloo's Exact Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    boschloo = ss.boschloo_exact(contingency_table)
    statistic, pval = boschloo.statistic, boschloo.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit


def modified_log_likelihood_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Modified Log Likelihood Ratio Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    mod_like = ss.chi2_contingency(contingency_table, correction=False, lambda_="mod-log-likelihood")
    statistic, pval = mod_like.statistic, mod_like.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit


def freeman_tukey_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Freeman-Tukey's Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    freeman_tukey = ss.chi2_contingency(contingency_table, correction=False, lambda_="freeman-tukey")
    statistic, pval = freeman_tukey.statistic, freeman_tukey.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit


def neyman_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Neyman's Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    neyman = ss.chi2_contingency(contingency_table, correction=False, lambda_="neyman")
    statistic, pval = neyman.statistic, neyman.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit


def cressie_read_test(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: Optional[float] = None,
) -> Union[float, bool]:
    """Cressie-Read's Test for a 2x2 Contingency Table

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
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
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")

    non_successes = [(n_i - s_i) for n_i, s_i in zip(trials, successes)]
    contingency_table = [successes, non_successes]
    mod_like = ss.chi2_contingency(contingency_table, correction=False, lambda_="cressie-read")
    statistic, pval = mod_like.statistic, mod_like.pvalue

    if crit is None:
        return pval
    return abs(statistic) >= crit
