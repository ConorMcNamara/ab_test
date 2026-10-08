"""Statistical tests to determine significance."""

import math
from typing import Any, Literal

import numpy as np
import scipy.stats as ss
from scipy.special import xlogy

from ab_test.frequentist_binomial.msprt import msprt_test
from ab_test.frequentist_binomial.randomization_inference import randomization_test
from ab_test.frequentist_binomial.utils import mle_under_null, mle_under_alternative, validate_two_group

__all__ = [
    "ab_test",
    "score_test",
    "likelihood_ratio_test",
    "z_test",
    "wald_test",
    "fisher_test",
    "barnard_exact_test",
    "boschloo_exact_test",
    "modified_log_likelihood_test",
    "freeman_tukey_test",
    "neyman_test",
    "cressie_read_test",
    "msprt_test",
    "randomization_test",
]


_validate_two_group = validate_two_group


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


def _pvalue_decision(pval: Any, crit: float | None) -> float | bool:
    """Return the p-value, or a significance boolean when an alpha threshold is given.

    Use this for exact tests (Fisher, Barnard, Boschloo), whose decision must
    come from the exact p-value rather than an asymptotic critical value.
    """
    if crit is None:
        return float(pval)
    if not 0 < crit < 1:
        raise ValueError(
            f"crit is the significance level for exact tests and must be in (0, 1), got {crit}. "
            "A z or chi-squared critical value would make every result significant."
        )
    return bool(pval <= crit)


def _require_zero_null(null_lift: float, test_name: str) -> None:
    """Reject a nonzero null for tests that can only test for no difference."""
    if null_lift != 0:
        raise NotImplementedError(f"{test_name} only supports a null lift of 0")


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
    _require_zero_null(null_lift, f"The {lambda_} test")
    contingency_table = _contingency_table(trials, successes)
    if np.any(np.asarray(contingency_table) == 0):
        if lambda_ in ("neyman", "mod-log-likelihood"):
            raise ValueError(
                f"The {lambda_} statistic is undefined when a cell has zero observed count "
                "(it divides by, or takes the log of, the observed count). "
                "Use the score test or Fisher's exact test instead."
            )
        if lambda_ == "freeman-tukey":
            # scipy evaluates 0 * inf here; the statistic's limit is 4 * sum((sqrt(O) - sqrt(E))**2).
            observed = np.asarray(contingency_table, dtype=float)
            expected = ss.contingency.expected_freq(observed)
            statistic = float(4 * np.sum((np.sqrt(observed) - np.sqrt(expected)) ** 2))
            return _test_result(statistic, ss.chi2.sf(statistic, df=1), crit)
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
        This makes such simulations about 5x faster. For the exact tests
        (``'fisher'``, ``'barnard'``, ``'boschloo'``), ``crit`` is instead the
        significance level alpha and is compared with the exact p-value.
     method : str
        How we plan on calculating the p_value or critical value of our
        experiment.  One of ``'score'``, ``'likelihood'``, ``'z'``,
        ``'wald'``, ``'fisher'``, ``'barnard'``, ``'boschloo'``,
        ``'modified_likelihood'``, ``'freeman-tukey'``, ``'neyman'``,
        ``'cressie-read'``, or ``'msprt'``.

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
    elif method == "wald":
        val = wald_test(trials, successes, null_lift, lift, crit)
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
    elif method == "msprt":
        val = msprt_test(trials, successes, null_lift, lift, crit)
    elif method == "randomization":
        val = randomization_test(trials, successes, null_lift, lift, crit)
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

    # The constrained MLE only reaches 0 or 1 for a group with no successes or
    # no failures, which it then fits exactly, so that group adds nothing.
    ts = sum(
        (s_i - n_i * p_i) ** 2 / (n_i * p_i * (1 - p_i))
        for n_i, s_i, p_i in zip(trials[:2], successes[:2], p)
        if 1e-12 < p_i < 1.0 - 1e-12
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

    # xlogy treats 0 * log(0) as 0, so groups with no successes or no failures
    # (where an MLE sits at 0 or 1) are handled without special-casing.
    def log_likelihood(p: list[Any] | np.ndarray[Any, Any]) -> float:
        return float(
            sum(xlogy(s_i, p_i) + xlogy(n_i - s_i, 1 - p_i) for n_i, s_i, p_i in zip(trials[:2], successes[:2], p))
        )

    ts = max(2 * (log_likelihood(p1) - log_likelihood(p0)), 0.0)
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
    """Score test for a 2x2 contingency table, expressed as a z-statistic.

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
        `mle_under_null`.
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
        True if the result is statistically significant, i.e. if the absolute
        z-statistic is >= `crit`. Returned if `crit` is not None. Note that
        `crit` is on the standard normal scale (e.g. 1.96), unlike
        `score_test`, whose `crit` is on the chi-squared scale.

    Notes
    -----
    Only supports two experiment groups at this time.

    Despite the name, this is not a Wald test. The statistic is::

        z = (pb_hat - pa_hat - null_lift) / sqrt(pa*(1 - pa)/na + pb*(1 - pb)/nb)

    where ``pa`` and ``pb`` are the MLEs under H0 (see `mle_under_null`),
    not the observed rates. Estimating the variance under the null makes this
    the score test: ``z**2`` equals the `score_test` statistic, so the two
    return the same p-value. A Wald test would use the observed rates in the
    variance instead.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    # The constrained MLE can overshoot [0, 1] by rounding error near the boundary.
    p0_arr = np.clip(np.asarray(p0), 0.0, 1.0)
    sigma2 = float(np.sum(p0_arr * (1 - p0_arr) / np.asarray(trials)))
    if sigma2 <= 0:
        return 1.0 if crit is None else False
    z = (p1[1] - p1[0] - null_lift) / math.sqrt(sigma2)
    if crit is None:
        return float(2.0 * ss.norm.cdf(-abs(z)))  # type: ignore[no-untyped-call]
    return bool(abs(z) >= crit)


def wald_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Wald test for a 2x2 contingency table, expressed as a z-statistic.

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
        equivalent to an absolute lift with null_lift 0.
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
        True if the result is statistically significant, i.e. if the absolute
        z-statistic is >= `crit`. Returned if `crit` is not None. Note that
        `crit` is on the standard normal scale (e.g. 1.96), unlike
        `score_test`, whose `crit` is on the chi-squared scale.

    Notes
    -----
    Only supports two experiment groups at this time.

    The statistic is::

        z = (pb_hat - pa_hat - null_lift) / sqrt(pa_hat*(1 - pa_hat)/na + pb_hat*(1 - pb_hat)/nb)

    where ``pa_hat`` and ``pb_hat`` are the observed success rates. `z_test`
    has the same numerator but estimates the variance under H0, which makes
    it the score test; the two agree closely in large samples.

    Because the variance comes from the observed rates, the Wald test is
    fragile near 0 and 1: a group with no successes (or no failures)
    contributes no variance. When both groups do, the variance is zero, so
    the test returns a p-value of 1 if the observed difference equals
    ``null_lift`` and 0 otherwise. Inverting it then gives a zero-width
    confidence interval, as with ``confidence_interval(method="wald")``. Prefer
    `score_test` when rates are close to 0 or 1.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)

    pa, pb = mle_under_alternative(trials, successes)
    diff = pb - pa - null_lift
    sigma2 = float(pa * (1 - pa) / trials[0] + pb * (1 - pb) / trials[1])
    if sigma2 <= 0:
        # Zero estimated variance makes z infinite unless the difference is exactly the null.
        significant = bool(abs(diff) > 1e-12)
        if crit is None:
            return 0.0 if significant else 1.0
        return significant
    z = diff / math.sqrt(sigma2)
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
    Unlike chi-squared or z tests, ``crit`` is compared against the p-value
    (i.e. treated as an alpha threshold) because Fisher's test has no
    test statistic amenable to a critical-value comparison.

    Notes
    -----
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    _require_zero_null(null_lift, "fisher_test")
    contingency_table = _contingency_table(trials, successes)
    _, pval = ss.fisher_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined, arg-type]
    return _pvalue_decision(pval, crit)


def barnard_exact_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Barnard's Exact Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.
    As with Fisher's and Boschloo's tests, ``crit`` is compared against the
    exact p-value (i.e. treated as an alpha threshold). Comparing Barnard's
    Wald statistic with a normal critical value would give an asymptotic
    decision that can disagree with the exact p-value.

    Notes
    -----
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    _require_zero_null(null_lift, "barnard_exact_test")
    contingency_table = _contingency_table(trials, successes)
    barnard = ss.barnard_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined]
    return _pvalue_decision(barnard.pvalue, crit)


def boschloo_exact_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Boschloo's Exact Test for a 2x2 Contingency Table.

    See :func:`score_test` for the shared parameter and return semantics.
    Unlike chi-squared or z tests, ``crit`` is compared against the p-value
    (i.e. treated as an alpha threshold) because Boschloo's test has no
    test statistic amenable to a critical-value comparison.

    Notes
    -----
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
    """
    _validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    _require_zero_null(null_lift, "boschloo_exact_test")
    contingency_table = _contingency_table(trials, successes)
    boschloo = ss.boschloo_exact(contingency_table)  # type: ignore[no-untyped-call, attr-defined]
    return _pvalue_decision(boschloo.pvalue, crit)


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
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
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
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
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
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
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
    Only supports two experiment groups and a null lift of 0 (no difference
    between groups), on either lift scale. These tests cannot be inverted
    into ``binary_search`` confidence intervals.
    """
    return _power_divergence_test(trials, successes, "cressie-read", null_lift, lift, crit)
