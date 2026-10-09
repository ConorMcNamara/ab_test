"""General utility functions."""

import math
from typing import Any

import numpy as np

__all__ = [
    "validate_two_group",
    "simple_hypothesis_from_composite",
    "mle_under_null",
    "mle_under_alternative",
    "wilson_significance",
    "observed_lift",
]


def validate_two_group(
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


def simple_hypothesis_from_composite(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    null_lift: float,
    alt_lift: float,
    lift: str = "relative",
) -> tuple[Any, ...]:
    """Translate a composite hypothesis into a simple hypothesis.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     null_lift : float
        Lift associated with null hypothesis.
     alt_lift : float
        Lift associated with alternative hypothesis.
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     p_null, p_alt: array
        Success rate in each group under the null and alternative hypotheses,
        respectively.

    Notes
    -----
    The power formula relies on simple null and alternative hypotheses of the
    form: "under the null hypothesis, the success rate in the first group is x,
    and the success rate in the second group is y". We care more about
    composite hypotheses like "under the alternative hypothesis, the success
    rate in the second group is 10% higher than in the first group".

    The alternative is taken at face value: the first group has the baseline
    rate and the second has ``baseline * (1 + alt_lift)`` (relative) or
    ``baseline + alt_lift`` (absolute). The null rates are the MLE under H0
    for the counts expected under that alternative -- the pooled rate when
    ``null_lift`` is 0 -- which is where the score test's restricted estimate
    settles when the alternative is true. The noncentrality of the score
    statistic is then

                 na * (pa - pi_a)^2     nb * (pb - pi_b)^2
      lambda =   ------------------  +  ------------------ ,
                  pi_a * (1 - pi_a)      pi_b * (1 - pi_b)

    with (pa, pb) the alternative and (pi_a, pi_b) the null rates. Choosing
    the alternative to minimise lambda instead (as this function used to)
    shrinks a relative lift towards zero rates and understates power badly:
    10% -> 40% with 50 per arm gave 0.40 for a test whose power is 0.95.

    Raises
    ------
    ValueError
        If the alternative implies a success rate outside (0, 1).
    """
    na = group_sizes[0]
    nb = group_sizes[1]

    p_alt_a = baseline
    p_alt_b = baseline * (1 + alt_lift) if lift == "relative" else baseline + alt_lift
    if not (0 < p_alt_a < 1 and 0 < p_alt_b < 1):
        raise ValueError(
            f"The alternative implies success rates of {p_alt_a:.4g} and {p_alt_b:.4g}; both must be in (0, 1)"
        )

    p_null = mle_under_null(group_sizes, [na * p_alt_a, nb * p_alt_b], null_lift=null_lift, lift=lift)
    p_alt = [p_alt_a, p_alt_b]
    return [float(p_null[0]), float(p_null[1])], p_alt


def observed_lift(
    trials: np.ndarray[Any, Any] | list[Any], successes: np.ndarray[Any, Any] | list[Any], lift: str = "relative"
) -> float:
    """Calculate the lift from our experiment.

    Parameters
    ----------
    trials : numpy array
        The number of trials for each iteration of an AB test
    successes : numpy array
        The number of successes for each iteration of an AB test
    lift : {'relative', 'absolute', 'incremental'}
        The lift we are measuring

    Returns
    -------
    ote : float
        The observed treatment effect, i.e., lift of our experiment

    Raises
    ------
    ZeroDivisionError
        If ``lift="relative"`` and the control group has no successes.
    """
    pa = successes[0] / trials[0]
    pb = successes[1] / trials[1]
    if lift == "relative":
        # Checked explicitly: numpy returns inf with a warning rather than raising.
        if pa == 0:
            raise ZeroDivisionError("Relative lift is undefined with no control successes")
        ote = (pb - pa) / pa
    else:
        if lift == "incremental":
            if trials[0] > trials[1]:
                pb = successes[1] * (trials[0] / trials[1])
                pa = successes[0]
            else:
                pa = successes[0] * (trials[1] / trials[0])
                pb = successes[1]
        ote = pb - pa
    return float(ote)


def mle_under_null(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
) -> list[Any]:
    """Maximum Likelihood Estimation under H0.

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
        rate, or in absolute terms. See Notes.

    Returns
    -------
     p : array
        Array [pa_star, pb_star], corresponding to the MLE of pa and pb under
        H0.

    Notes
    -----
    Solves the following optimization problem::

        maximize ll(pa, pb)
        s.t.     H0

    where H0 is of the form A*(p-a) = 0.

    When the null lift is zero, that corresponds to pa = pb, or A = [1 -1] and
    a = [0 0]'.

    Otherwise, the form of A and a depends on whether we are using relative
    lift (pb = pa * (1 + d)) or absolute lift (pb = pa + d). With relative
    lift, a = [0 0]' and A = [(1 + d) -1]. With absolute lift, a = [d 0]' and
    A = [1 -1].

    In all cases, H0 takes the form of a linear equality constraint. In all
    cases, the log-likelihood is concave, so the problem is efficiently
    solvable.

    When the null lift is zero, the solution is trivial. When we are using
    relative lift, there is still a fixed formula for the solution, but it is
    complicated! It involves solving a quadratic equation. When using absolute
    lift, we need to find the root of a cubic polynomial. It is easiest to use
    Newton's method for this.
    """
    if null_lift == 0:
        p_hat = (successes[0] + successes[1]) / (trials[0] + trials[1])
        p = [p_hat, p_hat]
    elif lift == "relative":
        S = np.sum(successes)
        T = np.sum(trials)
        neg_b = T + S + null_lift * (S + trials[1] - successes[1])
        a = T * (1 + null_lift)
        c = S
        radical = neg_b * neg_b - 4 * a * c
        pstar_a = (neg_b - math.sqrt(radical)) / (2.0 * a)
        pstar_b = pstar_a * (1.0 + null_lift)
        p = [pstar_a, pstar_b]
    else:
        # Find the root of the equation:
        #    A * x^3 + B * x^2 + C * x + D = 0
        val_tol = 1e-12
        step_tol = 1e-12

        sa = successes[0]
        sb = successes[1]
        fa = trials[0] - successes[0]
        fb = trials[1] - successes[1]
        d = null_lift

        A = sa + sb + fa + fb
        B = -2 * sa * (1 - d) + sb * (d - 2) - fa * (1 - 2 * d) - fb * (1 - d)
        C = sa * ((1 - d) ** 2 - d) + sb * (1 - d) - fa * d * (1 - d) - fb * d
        D = sa * d * (1 - d)

        pcrit = B * B - 3 * A * C
        if pcrit > 0:
            sqrt_pcrit = math.sqrt(pcrit)
            one_over_6A = 1.0 / (6 * A)
            pcrit_minus = (-B - sqrt_pcrit) * one_over_6A
            pcrit_plus = (-B + sqrt_pcrit) * one_over_6A

            if pcrit_minus < 0:
                pcrit_minus = 0.0

            if pcrit_plus > 1:
                pcrit_plus = 1.0
        else:
            pcrit_minus = 0
            pcrit_plus = 1

        x0 = 0.5 * (pcrit_minus + pcrit_plus)
        converged = False
        for _ in range(50):
            fn = D + x0 * (C + x0 * (B + x0 * A))
            fpn = C + x0 * (2 * B + x0 * 3 * A)
            x0 -= fn / fpn

            if abs(fn) < val_tol and abs(fn / fpn) < step_tol:
                converged = True
                break

        # pa must keep both rates in [0, 1]. When the maximum sits on that
        # boundary (e.g. a group with no successes or no failures), the cubic has
        # no root inside it: Newton lands outside, or on a spurious root at the
        # boundary introduced by clearing denominators. Fall back to bisection.
        lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
        if not converged or not lo + 1e-9 < x0 < hi - 1e-9:
            x0 = _bisect_absolute_mle(sa, fa, sb, fb, d, lo, hi)

        pstar_a = x0
        pstar_b = pstar_a + null_lift
        p = [pstar_a, pstar_b]

    return p


def _bisect_absolute_mle(sa: float, fa: float, sb: float, fb: float, d: float, lo: float, hi: float) -> float:
    """Maximise the absolute-lift constrained log-likelihood over ``[lo, hi]``.

    The log-likelihood is concave in ``pa``, so its derivative is decreasing and
    bisection on its sign converges to the interior root, or to whichever end of
    the interval the maximum lies at.
    """

    def ratio(count: float, prob: float) -> float:
        if count == 0:
            return 0.0
        return count / prob if prob > 0 else math.inf

    def slope(pa: float) -> float:
        pb = pa + d
        return ratio(sa, pa) - ratio(fa, 1 - pa) + ratio(sb, pb) - ratio(fb, 1 - pb)

    left, right = lo, hi
    for _ in range(200):
        mid = 0.5 * (left + right)
        if mid <= left or mid >= right:
            break
        if slope(mid) > 0:
            left = mid
        else:
            right = mid
    pa = 0.5 * (left + right)
    if pa - lo < 1e-12:
        return lo
    if hi - pa < 1e-12:
        return hi
    return pa


def mle_under_alternative(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    alt_lift: float | None = None,
    lift: str = "relative",
) -> np.ndarray[Any, Any] | list[Any]:
    """Maximum Likelihood Estimation under H1.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     alt_lift : float, optional
        Lift associated with alternative hypothesis. If None (default),
        alternative is unconstrained.
     lift : ["relative", "absolute"], optional
        Whether to interpret `alt_lift` relative to the baseline success
        rate, or in absolute terms. See Notes.

    Returns
    -------
     p : array
        Array [pa_star, pb_star], corresponding to the MLE of pa and pb under
        H1.

    Notes
    -----
    The most common alternative hypothesis considered is unconstrained, in
    which case p is simply successes / trials. But we also support an
    alternative hypothesis of the same form as H0, in case we ever want that.
    """
    if alt_lift is None:
        return np.asarray(successes) / np.asarray(trials)
    return mle_under_null(trials, successes, null_lift=alt_lift, lift=lift)


def wilson_significance(pval: float, alpha: float) -> float:
    """Wilson significance.

    Parameters
    ----------
     pval : float
        P-value.
     alpha : float
        Type-I error threshold.

    Returns
    -------
     W : float
        Wilson significance: log10(alpha / pval).

    Notes
    -----
    The Wilson significance is defined to be log10(alpha / pval),
    where pval is the p-value and alpha is the Type-I error rate. It
    has the following properties:

    - When the result is statistically significant, W > 0.
    - The larger W, the stronger the evidence.
    - An increase in W of 1 corresponds to a 10x decrease in p-value.
    """
    try:
        W = math.log10(alpha) - math.log10(pval)
    except ValueError:
        W = 310.0

    return W
