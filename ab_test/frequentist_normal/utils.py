import math
from typing import Any

import numpy as np

__all__ = [
    "validate_two_group",
    "observed_lift",
    "mle_under_null",
    "mle_under_alternative",
]


def validate_two_group(
    means: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    allow_relative_null: bool = True,
) -> None:
    """Validate the inputs shared by every significance test.

    Parameters
    ----------
    means, trials, variances : array_like
        Per-group means, trial counts and variances. At most two groups are supported.
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
    if len(trials) > 2 or len(means) > 2 or len(variances) > 2:
        raise NotImplementedError("Only supports a 2x2 continuous table")
    if not allow_relative_null and lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")


def observed_lift(
    means: np.ndarray[Any, Any] | list[Any], trials: np.ndarray[Any, Any] | list[Any], lift: str = "relative"
) -> float:
    """Calculate the lift from our experiment.

    Parameters
    ----------
    means : numpy array
        The mean for each iteration of an AB test
    trials : numpy array
        The number of trials for each iteration of an AB test
    lift : {'relative', 'absolute', 'incremental'}
        The lift we are measuring

    Returns
    -------
    ote : float
        The observed treatment effect, i.e., lift of our experiment
    """
    mean_a, mean_b = means[0], means[1]
    if lift == "relative":
        ote = (mean_b - mean_a) / mean_a
    elif lift == "incremental":
        scale = max(trials[0], trials[1])
        ote = (mean_b - mean_a) * scale
    else:
        ote = mean_b - mean_a
    return float(ote)


def _mle_under_null_unequal_var(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float,
    lift: str,
) -> tuple[list[float], list[float]]:
    """Constrained MLE when each group has its own variance.

    Writing H0 as ``mu_b = k * mu_a + c``, profiling out both variances leaves
    a one-dimensional likelihood in ``mu_a`` whose stationary points are the
    real roots of a cubic. The likelihood can have more than one local
    maximum, so every real root is evaluated and the best one is kept.
    """
    mean_a, mean_b = float(means[0]), float(means[1])
    n_a, n_b = trials[0], trials[1]
    ml_var_a = (n_a - 1) / n_a * variances[0]
    ml_var_b = (n_b - 1) / n_b * variances[1]
    if lift == "relative":
        k, c = 1.0 + null_lift, 0.0
    else:
        k, c = 1.0, null_lift

    x = np.polynomial.Polynomial([0.0, 1.0])
    resid_a = mean_a - x
    resid_b = (mean_b - c) - k * x
    stationary = (n_a * resid_a * (ml_var_b + resid_b**2) + n_b * k * resid_b * (ml_var_a + resid_a**2)).trim()
    # Near-repeated roots come back slightly complex, so keep every root's real
    # part; the zero-residual points cover groups with zero variance.
    candidates = [float(r.real) for r in stationary.roots()] + [mean_a]
    if k != 0:
        candidates.append((mean_b - c) / k)

    def profile_ll(mu_a: float) -> float:
        var_a = ml_var_a + (mean_a - mu_a) ** 2
        var_b = ml_var_b + (mean_b - k * mu_a - c) ** 2
        if var_a <= 0 or var_b <= 0:
            return math.inf
        return float(-0.5 * n_a * math.log(var_a) - 0.5 * n_b * math.log(var_b))

    mu_a = max(candidates, key=profile_ll)
    mu = [mu_a, k * mu_a + c]
    sigma2 = [float(ml_var_a + (mean_a - mu[0]) ** 2), float(ml_var_b + (mean_b - mu[1]) ** 2)]
    return [float(mu[0]), float(mu[1])], sigma2


def mle_under_null(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    equal_var: bool = True,
) -> tuple[list[float], float | list[float]]:
    """Maximum Likelihood Estimation under H0.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The sample variances (``ddof=1``) for each group.
    trials : array_like
        Number of trials in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline mean,
        or in absolute terms. See Notes.
    equal_var : bool, default=True
        Whether both groups share a common variance. See Notes.

    Returns
    -------
    mu : list of float
        ``[mu_a_star, mu_b_star]``, the MLE of each group mean under H0.
    sigma2 : float or list of float
        The MLE of the common variance under H0 if ``equal_var`` is True,
        otherwise ``[sigma2_a_star, sigma2_b_star]``, one per group.

    Notes
    -----
    Assumes both groups are normal and solves::

        maximize ll(mu_a, mu_b, sigma2)
        s.t.     H0

    where H0 is either ``mu_b = mu_a + d`` (absolute lift) or
    ``mu_b = mu_a * (1 + d)`` (relative lift).

    With a common variance, both constraints are linear so the solution is
    closed form: ``mu_a_star`` is a weighted least-squares fit of the two
    sample means under the constraint, and ``sigma2`` is the pooled
    within-group sum of squares plus the squared deviation of each sample
    mean from its constrained estimate, divided by the total number of trials.

    With separate variances (the Behrens-Fisher setting), each group's
    variance is its within-group sum of squares divided by its trials, plus
    the squared deviation of its sample mean from its constrained estimate.
    Because those variances in turn weight the constrained means, there is no
    closed form; the constrained mean is found among the real roots of a cubic.
    """
    if not equal_var:
        return _mle_under_null_unequal_var(means, variances, trials, null_lift, lift)
    mean_a, mean_b = means[0], means[1]
    n_a, n_b = trials[0], trials[1]
    if lift == "relative":
        k = 1.0 + null_lift
        mu_a = (n_a * mean_a + n_b * k * mean_b) / (n_a + n_b * k * k)
        mu = [mu_a, k * mu_a]
    else:
        mu_a = (n_a * mean_a + n_b * (mean_b - null_lift)) / (n_a + n_b)
        mu = [mu_a, mu_a + null_lift]
    sum_sq = (
        (n_a - 1) * variances[0]
        + (n_b - 1) * variances[1]
        + n_a * (mean_a - mu[0]) ** 2
        + n_b * (mean_b - mu[1]) ** 2
    )
    sigma2 = sum_sq / (n_a + n_b)
    return [float(mu[0]), float(mu[1])], float(sigma2)


def mle_under_alternative(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    alt_lift: float | None = None,
    lift: str = "relative",
    equal_var: bool = True,
) -> tuple[list[float], float | list[float]]:
    """Maximum Likelihood Estimation under H1.

    Parameters
    ----------
    means : array_like
        The average for each group.
    variances : array_like
        The sample variances (``ddof=1``) for each group.
    trials : array_like
        Number of trials in each group.
    alt_lift : float, optional
        Lift associated with alternative hypothesis. If None (default),
        alternative is unconstrained.
    lift : {"relative", "absolute"}
        Whether to interpret ``alt_lift`` relative to the baseline mean,
        or in absolute terms. See Notes in `mle_under_null`.
    equal_var : bool, default=True
        Whether both groups share a common variance.

    Returns
    -------
    mu : list of float
        ``[mu_a_star, mu_b_star]``, the MLE of each group mean under H1.
    sigma2 : float or list of float
        The MLE of the common variance under H1 if ``equal_var`` is True,
        otherwise ``[sigma2_a_star, sigma2_b_star]``, one per group.

    Notes
    -----
    The most common alternative hypothesis considered is unconstrained, in
    which case ``mu`` is simply the sample means and ``sigma2`` is the
    within-group sum of squares divided by the number of trials, pooled
    across groups if ``equal_var`` is True. But we also support an alternative
    hypothesis of the same form as H0, in case we ever want that.
    """
    if alt_lift is None:
        mu = [float(means[0]), float(means[1])]
        if not equal_var:
            return mu, [float((n - 1) / n * v) for n, v in zip(trials[:2], variances[:2])]
        sum_sq = (trials[0] - 1) * variances[0] + (trials[1] - 1) * variances[1]
        return mu, float(sum_sq / (trials[0] + trials[1]))
    return mle_under_null(means, variances, trials, null_lift=alt_lift, lift=lift, equal_var=equal_var)
