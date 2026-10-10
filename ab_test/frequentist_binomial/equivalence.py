"""Equivalence testing via the TOST (Two One-Sided Tests) procedure."""

from typing import Any

import numpy as np

from ab_test.frequentist_binomial.stats_tests import ab_test
from ab_test.frequentist_binomial.utils import observed_lift

__all__ = ["tost_test"]

_SUPPORTED_METHODS = frozenset({"score", "likelihood", "z"})


def tost_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    delta: float,
    alpha: float = 0.05,
    lift: str = "absolute",
    method: str = "score",
) -> dict[str, float | bool]:
    """Two One-Sided Tests (TOST) for equivalence of two proportions.

    Tests whether the true difference between group B and group A falls
    within the equivalence margin ``[-delta, delta]``.  The null hypothesis
    is that the groups differ by *at least* ``delta``; rejecting it
    (``p_value <= alpha``) is evidence of practical equivalence.

    Parameters
    ----------
    trials : array_like
        Number of trials in each group, length 2.
    successes : array_like
        Number of successes in each group, length 2.
    delta : float
        Equivalence margin.  Must be positive.  Groups are deemed equivalent
        when their difference (in the scale given by ``lift``) falls within
        ``[-delta, delta]``.
    alpha : float, optional
        Significance level.  Default is 0.05.
    lift : {"absolute", "relative"}, optional
        How ``delta`` is interpreted.  ``"absolute"`` means an additive
        difference in proportions; ``"relative"`` means a multiplicative
        factor of the control rate.  Default is ``"absolute"``.
    method : {"score", "likelihood", "z"}, optional
        Which underlying test to use for each one-sided test.  Only tests
        that support a non-zero ``null_lift`` are allowed, and ``"z"``
        supports only ``lift="absolute"``.
        Default is ``"score"`` (Rao score / Farrington-Manning).

    Returns
    -------
    dict[str, float | bool]
        Dictionary with the following keys:

        - ``"p_value"`` : float — TOST p-value (max of the two one-sided).
        - ``"p_lower"`` : float — one-sided p-value for the lower-bound
          test (H0: diff <= -delta).
        - ``"p_upper"`` : float — one-sided p-value for the upper-bound
          test (H0: diff >= delta).
        - ``"equivalent"`` : bool — ``True`` when ``p_value <= alpha``.

    Raises
    ------
    ValueError
        If ``delta <= 0``, ``method`` is unsupported, ``method="z"`` is used
        with ``lift="relative"``, or any input validation from the underlying
        test fails.

    Notes
    -----
    The procedure runs two one-sided tests at level ``alpha``:

    1. Non-inferiority:  H0: diff <= -delta  vs  H1: diff > -delta
    2. Non-superiority:  H0: diff >=  delta  vs  H1: diff <  delta

    Equivalence is concluded when **both** reject, which is equivalent to
    checking that ``max(p_lower, p_upper) <= alpha``.

    With ``lift="relative"`` and ``delta >= 1``, the lower null hypothesis
    needs a treatment rate of 0 or less, so ``p_lower`` is 0 (or 1 at
    ``delta = 1`` when the treatment group has no successes) and the decision
    rests on the upper test.

    One-sided p-values are derived from the two-sided p-values returned by
    the underlying test.  Because the score and likelihood-ratio tests
    produce chi-squared statistics (1 df), the two-sided p-value is halved
    in the direction consistent with the observed difference.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    method_lower = method.casefold()
    if method_lower not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Method '{method}' does not support non-zero null_lift. Supported methods: {sorted(_SUPPORTED_METHODS)}"
        )
    if method_lower == "z" and lift.casefold() == "relative":
        raise ValueError(
            "method='z' supports only lift='absolute' for TOST. Use method='score' or 'likelihood' for relative lift."
        )

    d_hat = observed_lift(trials, successes, lift=lift)

    p_two_lower = float(ab_test(trials, successes, null_lift=-delta, lift=lift, method=method_lower))
    p_two_upper = float(ab_test(trials, successes, null_lift=delta, lift=lift, method=method_lower))

    p_lower = p_two_lower / 2 if d_hat > -delta else 1 - p_two_lower / 2
    if lift.casefold() == "relative" and delta >= 1:
        # A relative lift below -1 needs a negative treatment rate, so the lower
        # null is impossible; at delta = 1 it is a treatment rate of exactly 0,
        # which any treatment success refutes. The constrained MLE behind
        # p_two_lower is invalid here (it gave p_lower = 0.5).
        p_lower = 1.0 if delta == 1 and successes[1] == 0 else 0.0
    p_upper = p_two_upper / 2 if d_hat < delta else 1 - p_two_upper / 2

    p_value = max(p_lower, p_upper)
    return {
        "p_value": p_value,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "equivalent": p_value <= alpha,
    }
