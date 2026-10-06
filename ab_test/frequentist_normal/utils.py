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


def mle_under_null(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
) -> tuple[list[float], float]:
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

    Returns
    -------
    mu : list of float
        ``[mu_a_star, mu_b_star]``, the MLE of each group mean under H0.
    sigma2 : float
        The MLE of the common variance under H0.

    Notes
    -----
    Assumes both groups are normal with a common variance and solves::

        maximize ll(mu_a, mu_b, sigma2)
        s.t.     H0

    where H0 is either ``mu_b = mu_a + d`` (absolute lift) or
    ``mu_b = mu_a * (1 + d)`` (relative lift). Both are linear equality
    constraints, so unlike the binomial case the solution is closed form:
    ``mu_a_star`` is a weighted least-squares fit of the two sample means
    under the constraint, and ``sigma2`` is the pooled within-group sum of
    squares plus the squared deviation of each sample mean from its
    constrained estimate, divided by the total number of trials.
    """
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
) -> tuple[list[float], float]:
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

    Returns
    -------
    mu : list of float
        ``[mu_a_star, mu_b_star]``, the MLE of each group mean under H1.
    sigma2 : float
        The MLE of the common variance under H1.

    Notes
    -----
    The most common alternative hypothesis considered is unconstrained, in
    which case ``mu`` is simply the sample means and ``sigma2`` is the pooled
    within-group sum of squares divided by the total number of trials. But we
    also support an alternative hypothesis of the same form as H0, in case we
    ever want that.
    """
    if alt_lift is None:
        sum_sq = (trials[0] - 1) * variances[0] + (trials[1] - 1) * variances[1]
        return [float(means[0]), float(means[1])], float(sum_sq / (trials[0] + trials[1]))
    return mle_under_null(means, variances, trials, null_lift=alt_lift, lift=lift)
