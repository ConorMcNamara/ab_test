"""Statistical tests to determine significance."""

import math
from typing import Any, Literal

import numpy as np
import scipy.stats as ss

from ab_test.frequentist_binomial.utils import mle_under_null, mle_under_alternative

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


def _validate_two_group(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    allow_relative_null: bool = True,
) -> None:
    """Validate the inputs shared by every significance test.

    Parameters
    ----------
    trials, successes : array_like
        Per-group trial and success counts. At most two groups are supported.
    null_lift : float
        Lift associated with the null hypothesis.
    lift : str
        Whether ``null_lift`` is interpreted in relative or absolute terms.
    allow_relative_null : bool
        If False, a nonzero relative ``null_lift`` is rejected (only tests that
        support a nonzero relative null pass True).

    Raises
    ------
    NotImplementedError
        If more than two groups are supplied, or a nonzero relative ``null_lift``
        is given when ``allow_relative_null`` is False.
    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Only supports a 2x2 contingency table")
    if not allow_relative_null and lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")


def _contingency_table(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
) -> np.ndarray[Any, Any]:
    non_successes = np.asarray(trials) - np.asarray(successes)
    return np.array([successes, non_successes])


def _test_result(statistic: Any, pval: Any, crit: float | None) -> float | bool:
    """Return the p-value, or a significance boolean when a critical value is given."""
    if crit is None:
        return float(pval)
    return bool(abs(statistic) >= crit)


def _power_divergence_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    lambda_: Literal["mod-log-likelihood", "freeman-tukey", "neyman", "cressie-read"],
    null_lift: float,
    lift: str,
    crit: float | None,
) -> float | bool:
    """Run a power-divergence test on the 2x2 table via ``scipy.stats.chi2_contingency``.

    Shared by the Freeman-Tukey, Neyman, Cressie-Read, and modified
    log-likelihood tests, which differ only in the ``lambda_`` parameter.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    contingency_table = _contingency_table(trials, successes)
    result = ss.chi2_contingency(contingency_table, correction=False, lambda_=lambda_)  # type: ignore[no-untyped-call, attr-defined, var-annotated]
    return _test_result(result.statistic, result.pvalue, crit)


def ab_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
    method: str = "score",
) -> float | bool:
    """Dispatch to our different statistical tests.

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
    method : {'score', 'likelihood', 'z', 'fisher', 'barnard', 'boschloo',
              'modified_likelihood', 'freeman-tukey', 'neyman', 'cressie-read'}
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
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
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
    _validate_two_group(trials, successes, null_lift, lift)

    p = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)

    if min(p) <= 1e-12 or max(p) + 1e-12 >= 1.0:
        return 1.0

    p0, p1 = p[0], p[1]
    pq0, pq1 = p0 * (1 - p0), p1 * (1 - p1)
    ts = (successes[0] - trials[0] * p0) ** 2 / (trials[0] * pq0) + (successes[1] - trials[1] * p1) ** 2 / (
        trials[1] * pq1
    )

    if crit is None:
        # Note: this line takes 80% of the time for the score test, including the
        # MLE, which is really fast! So there's no real point in optimizing
        # anything else here. On the other hand, if we can optimize this line, then
        # great!
        pval = ss.chi2.sf(ts, df=1)  # type: ignore[no-untyped-call]
        return float(pval)
    return ts >= crit


def likelihood_ratio_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
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
    _validate_two_group(trials, successes, null_lift, lift)

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    if min(p0) <= 1e-12 or max(p0) + 1e-12 >= 1.0:
        return 1.0

    if min(p1) <= 1e-12 or max(p1) + 1e-12 >= 1.0:
        return 1.0

    def log_likelihood(p: list[Any] | np.ndarray[Any, Any]) -> float:
        return (
            successes[0] * math.log(p[0])
            + (trials[0] - successes[0]) * math.log(1 - p[0])
            + successes[1] * math.log(p[1])
            + (trials[1] - successes[1]) * math.log(1 - p[1])
        )

    ts = 2 * (log_likelihood(p1) - log_likelihood(p0))
    if crit is None:
        pval = ss.chi2.sf(ts, df=1)  # type: ignore[no-untyped-call]
        return float(pval)
    return ts >= crit


def z_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
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
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    p0_arr = np.asarray(p0)
    sigma2 = float(np.sum(p0_arr * (1 - p0_arr) / np.asarray(trials)))
    z = (p1[1] - p1[0] - null_lift) / math.sqrt(sigma2)
    if crit is None:
        return float(2.0 * ss.norm.cdf(-abs(z)))  # type: ignore[no-untyped-call]
    return bool(abs(z) >= crit)


def fisher_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Fisher's Exact Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    contingency_table = _contingency_table(trials, successes)
    statistic, pval = ss.fisher_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined, arg-type]
    return _test_result(statistic, pval, crit)


def barnard_exact_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Barnard's Exact Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    contingency_table = _contingency_table(trials, successes)
    barnard = ss.barnard_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined]
    return _test_result(barnard.statistic, barnard.pvalue, crit)


def boschloo_exact_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Boschloo's Exact Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    contingency_table = _contingency_table(trials, successes)
    boschloo = ss.boschloo_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined]
    return _test_result(boschloo.statistic, boschloo.pvalue, crit)


def modified_log_likelihood_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Compute the Modified Log-Likelihood Ratio Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    return _power_divergence_test(trials, successes, "mod-log-likelihood", null_lift, lift, crit)


def freeman_tukey_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Freeman-Tukey's Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    return _power_divergence_test(trials, successes, "freeman-tukey", null_lift, lift, crit)


def neyman_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Neyman's Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    return _power_divergence_test(trials, successes, "neyman", null_lift, lift, crit)


def cressie_read_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Cressie-Read's Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.

    Notes
    -----
    Only supports two experiment groups, and only an absolute lift (or a
    relative lift with a null of 0%).
    """
    return _power_divergence_test(trials, successes, "cressie-read", null_lift, lift, crit)
