"""General utility functions."""

import math
from typing import Optional, Union

import numpy as np

__all__ = ["simple_hypothesis_from_composite", "mle_under_null", "mle_under_alternative", "wilson_significance"]


def simple_hypothesis_from_composite(
    group_sizes: Union[np.ndarray, list],
    baseline: float,
    null_lift: float,
    alt_lift: float,
    lift: str = "relative",
) -> np.array:
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
    specifying composite hypotheses like, "under the null hypothesis, the
    success rates in the two groups are equal", or "under the alternative
    hypothesis, the success rate in the second group is 10% higher than in the
    first group".

    This function translates such composite null and alternative hypotheses to
    simple hypotheses. We need the baseline success rate as well. Consider this
    translation mechanism:
       H0: pa = baseline, pb = baseline * (1 + null_lift)
       H1: pa = baseline, pb = baseline * (1 + alt_lift)
    The success rate in the first group is the same either way, but the success
    rate in the second group depends on the null/alt lift. This seems innocent
    enough, but consider the noncentrality parameter of the chi-2 distribution
    under the alternative hypothesis:

                 na * (pa - pi_a)^2     nb * (pb - pi_b)^2
      lambda =   ------------------  +  ------------------ ,
                  pi_a * (1 - pi_a)      pi_b * (1 - pi_b)

    where pi_a, pi_b are the success rates in the first and second groups under
    H0, pa and pb are for H1, and na and nb are the group sizes. In our naive
    approach, pa is always equal to pi_a (is equal to the baseline), so the
    first term is always zero. No matter how big or small na is, the
    noncentrality parameter is always the same, so the power is always the
    same. That's not right.

    The power of the test increases with lambda, so we might reasonably ask,
    what is the lowest lambda could be while still being aligned with the
    information given? If we leave the success rates under H0 alone (pi_a =
    baseline and pi_b = pi_a * (1 + null_lift)), which seems reasonable enough,
    we can minimize lambda subject to the constraint pb = pa * (1 + alt_lift).
    Treating na, nb, pi_a, and pi_b as data, this is a convex optimization
    problem. The solution (pa, pb) is the simple alternative hypothesis.
    """
    na = group_sizes[0]
    nb = group_sizes[1]

    p_null_a = baseline
    if lift == "relative":
        p_null_b = (1 + null_lift) * baseline
    else:
        p_null_b = baseline + null_lift

    if lift == "relative":
        p_alt_a = (2 * na / (1 - p_null_a)) + (2 * nb * (1 + alt_lift) / (1 - p_null_b))
        p_alt_a /= (2 * na) / (p_null_a * (1 - p_null_a)) + (2 * nb * (1 + alt_lift) ** 2) / (p_null_b * (1 - p_null_b))
    else:
        p_alt_a = 2 * na / (1 - p_null_a)
        p_alt_a += 2 * nb * (p_null_b - alt_lift) / (p_null_b * (1 - p_null_b))
        p_alt_a /= 2 * na / (p_null_a * (1 - p_null_a)) + 2 * nb / (p_null_b * (1 - p_null_b))

    if lift == "relative":
        p_alt_b = (1 + alt_lift) * p_alt_a
    else:
        p_alt_b = p_alt_a + alt_lift

    p_null = [p_null_a, p_null_b]
    p_alt = [p_alt_a, p_alt_b]
    return p_null, p_alt


def observed_lift(trials: Union[np.array, list], successes: Union[np.array, list], lift: str = "relative") -> float:
    """Calculates the lift from our experiment

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
    """
    pa = successes[0] / trials[0]
    pb = successes[1] / trials[1]
    if lift == "relative":
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
    return ote


def mle_under_null(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    null_lift: float = 0.0,
    lift: str = "relative",
) -> np.ndarray:
    """Maximum Likelihood Estimation under H0

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
    Solves the following optimization problem:
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
        # MLE of parameters under null hypothesis
        p = [sum(successes) / sum(trials) for _ in range(2)]
    elif lift == "relative":
        S = sum(successes)
        T = sum(trials)
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
        for _ in range(50):
            fn = A * x0 * x0 * x0 + B * x0 * x0 + C * x0 + D
            fpn = 3 * A * x0 * x0 + 2 * B * x0 + C
            x0 -= fn / fpn

            if abs(fn) < val_tol and abs(fn / fpn) < step_tol:
                break
        else:
            raise ValueError("MLE did not converge")

        pstar_a = x0
        pstar_b = pstar_a + null_lift
        p = [pstar_a, pstar_b]

    return p


def mle_under_alternative(
    trials: Union[np.ndarray, list],
    successes: Union[np.ndarray, list],
    alt_lift: Optional[float] = None,
    lift: str = "relative",
) -> np.ndarray:
    """Maximum Likelihood Estimation under H1

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
        successes = np.array(successes)
        trials = np.array(trials)
        return successes / trials
    return mle_under_null(trials, successes, null_lift=alt_lift, lift=lift)


def wilson_significance(pval: float, alpha: float) -> float:
    """Wilson significance

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
