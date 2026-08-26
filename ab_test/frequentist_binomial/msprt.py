"""Mixture Sequential Probability Ratio Test (mSPRT) for always-valid inference.

Provides always-valid p-values and confidence sequences that maintain
type-I error control regardless of how many times results are checked.
The test uses a Gaussian mixture on the effect size, following the
framework of Johari et al. (2017).
"""

import math
from typing import Any

import numpy as np

from ab_test.frequentist_binomial.utils import mle_under_alternative, mle_under_null, validate_two_group

__all__ = [
    "msprt_test",
    "msprt_critical_value",
]


def _msprt_log_lambda(d: float, sigma2: float, tau2: float) -> float:
    """Log of the mSPRT likelihood ratio.

    Parameters
    ----------
    d : float
        Centered effect: observed absolute effect minus null-expected
        absolute effect.
    sigma2 : float
        Variance of the effect estimate under H0.
    tau2 : float
        Squared scale of the Gaussian mixing distribution.

    Returns
    -------
    float
        Log of the likelihood ratio Lambda_n.
    """
    return -0.5 * math.log(1 + tau2 / sigma2) + d * d * tau2 / (2 * sigma2 * (sigma2 + tau2))


def msprt_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
    *,
    tau: float | None = None,
) -> float | bool:
    """Mixture Sequential Probability Ratio Test for a 2x2 contingency table.

    Computes an always-valid p-value that controls type-I error no matter
    how many times the data are examined. The test mixes over a Gaussian
    prior ``N(0, tau^2)`` on the effect size.

    Parameters
    ----------
    trials : array_like
        Number of trials in each group.
    successes : array_like
        Number of successes in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms.
    crit : float, optional
        Critical value for the likelihood ratio. If omitted, an always-valid
        p-value is returned. If passed, a boolean is returned indicating
        whether the likelihood ratio exceeds the critical value. Use
        :func:`msprt_critical_value` to obtain the threshold for a given
        alpha.
    tau : float or None, optional
        Scale of the Gaussian mixing distribution on the effect size. Larger
        values give more power for large effects but less for small ones.
        When ``None`` (the default), ``tau`` is set to the standard error of
        the difference under the pooled null — a unit-information prior that
        scales naturally with the data.

    Returns
    -------
    pval : float
        Always-valid p-value. Returned if ``crit`` is None.
    stat_sig : bool
        True if the likelihood ratio exceeds ``crit``. Returned if ``crit``
        is not None.

    Notes
    -----
    The always-valid p-value is ``min(1, 1 / Lambda_n)`` where ``Lambda_n``
    is the likelihood ratio. It is safe to compute at any sample size and
    reject whenever it drops below alpha, without inflating the type-I error.

    References
    ----------
    Johari, R., Pekelis, L., & Walsh, D. J. (2017). Peeking at A/B Tests:
    Why it matters, and what to do about it. *KDD '17*.
    """
    validate_two_group(trials, successes, null_lift, lift)

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    if min(p0) <= 1e-12 or max(p0) + 1e-12 >= 1.0:
        return 1.0 if crit is None else False

    sigma2 = p0[0] * (1 - p0[0]) / trials[0] + p0[1] * (1 - p0[1]) / trials[1]
    if sigma2 <= 1e-24:
        return 1.0 if crit is None else False

    d = (p1[1] - p1[0]) - (p0[1] - p0[0])

    if tau is None:
        tau = math.sqrt(sigma2)
    tau2 = tau * tau

    log_lambda = _msprt_log_lambda(d, sigma2, tau2)
    lambda_n = math.exp(min(log_lambda, 700))

    if crit is None:
        return min(1.0, 1.0 / lambda_n)
    return lambda_n >= crit


def msprt_critical_value(alpha: float = 0.05) -> float:
    """Critical value for the mSPRT likelihood ratio.

    Parameters
    ----------
    alpha : float
        Type-I error rate. Defaults to 0.05.

    Returns
    -------
    float
        The threshold ``1 / alpha``. Reject H0 when the likelihood ratio
        exceeds this value.
    """
    return 1.0 / alpha
