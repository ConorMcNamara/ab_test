from math import sqrt
from typing import Any, Literal

import numpy as np
from scipy.stats import beta, norm

from ab_test.bayesian_binomial.utils import sample_beta

__all__: list[str] = [
    "credible_interval",
    "individual_credible_interval",
    "calculate_interval",
    "calculate_hdi",
    "calculate_hdi_from_samples",
]


def credible_interval(
    successes: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    prior_alphas: np.ndarray[Any, Any] | list[Any],
    prior_betas: np.ndarray[Any, Any] | list[Any],
    confidence_level: float = 0.95,
    lift: Literal["relative", "absolute"] = "relative",
    is_sample: bool = True,
    n_samples: int = 100_000,
    method: Literal["credible", "hdi"] = "credible",
) -> tuple[float, float]:
    """Compute the credible interval for the lift between two binomial variants.

    Either samples from the Beta posterior of each variant and derives the
    interval from the resulting lift distribution, or uses a normal
    approximation via the delta method when ``is_sample=False``.

    Parameters
    ----------
    successes : np.ndarray or list
        Number of successes for each variant. Index 0 is the control,
        index 1 is the treatment.
    trials : np.ndarray or list
        Total number of trials for each variant. Index 0 is the control,
        index 1 is the treatment.
    prior_alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    prior_betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    confidence_level : float, optional
        Desired probability mass within the interval, by default 0.95.
    lift : {"relative", "absolute"}, optional
        Type of lift to compute. ``"relative"`` returns ``(theta_2 - theta_1)
        / theta_1``; ``"absolute"`` returns ``theta_2 - theta_1``, by default
        ``"relative"``.
    is_sample : bool, optional
        Whether to use Monte Carlo sampling to derive the posterior. If False,
        relies on the normal approximation, by default True.
    n_samples : int, optional
        Number of posterior samples to draw per variant, by default 100_000.
    method : {"credible", "hdi"}, optional
        Method used to compute the interval. ``"credible"`` uses equal-tailed
        percentiles; ``"hdi"`` uses the Highest Density Interval. Note that if_sample = False
        creates a symmetric interval, thereby both ``"credible"`` and ``"hdi"``
        return the same interval. By default, ``"credible"``.

    Returns
    -------
    tuple[float, float]
        A ``(lower_bound, upper_bound)`` pair representing the credible interval
        of the lift between the two variants.

    Raises
    ------
    NotImplementedError
        If ``method`` is not ``"credible"`` or ``"hdi"``.
    """
    lower_bound = (1 - confidence_level) / 2
    upper_bound = 1 - lower_bound
    if is_sample:
        theta_1 = sample_beta(successes[0], trials[0], prior_alphas[0], prior_betas[0], n_samples)
        theta_2 = sample_beta(successes[1], trials[1], prior_alphas[1], prior_betas[1], n_samples)
        abs_lift = theta_2 - theta_1
        lift_samples = abs_lift / np.where(theta_1 == 0, 1e-09, theta_1) if lift == "relative" else abs_lift
        if method == "credible":
            lb, ub = np.percentile(lift_samples, [lower_bound * 100, upper_bound * 100])
        elif method == "hdi":
            lb, ub = calculate_hdi_from_samples(lift_samples, confidence_level)
        else:
            raise NotImplementedError(f"No support for {method} method of generating credible intervals")
    else:
        mu_1 = (successes[0] + prior_alphas[0]) / (trials[0] + prior_alphas[0] + prior_betas[0])
        var_1 = (mu_1 * (1 - mu_1)) / (trials[0] + prior_alphas[0] + prior_betas[0] + 1)

        mu_2 = (successes[1] + prior_alphas[1]) / (trials[1] + prior_alphas[1] + prior_betas[1])
        var_2 = (mu_2 * (1 - mu_2)) / (trials[1] + prior_alphas[1] + prior_betas[1] + 1)

        if lift == "relative":
            lift_mu = mu_2 / mu_1 - 1
            lift_std = sqrt((mu_2**2 / mu_1**4) * var_1 + (1 / mu_1**2) * var_2)
        else:
            lift_mu = mu_2 - mu_1
            lift_std = sqrt(var_2 + var_1)

        z = norm.ppf(upper_bound)
        lb, ub = lift_mu - z * lift_std, lift_mu + z * lift_std
    return lb, ub


def individual_credible_interval(
    s: int,
    n: int,
    confidence_level: float = 0.95,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    n_samples: int = 100_000,
    method: Literal["credible", "hdi"] = "credible",
) -> tuple[float, float]:
    """Compute the credible interval for a single binomial proportion.

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    confidence_level : float, optional
        Desired probability mass within the interval, by default 0.95.
    prior_alpha : int, optional
        Alpha parameter of the Beta prior distribution, by default 1.
    prior_beta : int, optional
        Beta parameter of the Beta prior distribution, by default 1.
    n_samples : int, optional
        Number of samples to draw when using the HDI method, by default 100_000.
    method : {"credible", "hdi"}, optional
        Method used to compute the interval. ``"credible"`` uses equal-tailed
        quantiles; ``"hdi"`` uses the Highest Density Interval, by default
        ``"credible"``.

    Returns
    -------
    tuple[float, float]
        A ``(lower_bound, upper_bound)`` pair representing the credible interval.

    Raises
    ------
    NotImplementedError
        If ``method`` is not ``"credible"`` or ``"hdi"``.
    """
    if method == "credible":
        lb, ub = calculate_interval(s, n, prior_alpha, prior_beta, confidence_level)
    elif method == "hdi":
        lb, ub = calculate_hdi(s, n, prior_alpha, prior_beta, confidence_level, n_samples)
    else:
        raise NotImplementedError(f"No support for {method} method of generating confidence intervals")
    return lb, ub


def calculate_interval(
    s: int,
    n: int,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate the equal-tailed credible interval using the Beta posterior.

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    prior_alpha : int, optional
        Alpha parameter of the Beta prior distribution, by default 1.
    prior_beta : int, optional
        Beta parameter of the Beta prior distribution, by default 1.
    confidence_level : float, optional
        Desired probability mass within the interval, by default 0.95.

    Returns
    -------
    tuple[float, float]
        A ``(lower_bound, upper_bound)`` pair derived from the posterior
        Beta distribution quantiles.
    """
    lower_bound = (1 - confidence_level) / 2
    upper_bound = 1 - lower_bound

    ci_lower = beta.ppf(lower_bound, prior_alpha + s, prior_beta + n - s)
    ci_upper = beta.ppf(upper_bound, prior_alpha + s, prior_beta + n - s)

    return ci_lower, ci_upper


def calculate_hdi(
    s: int,
    n: int,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    confidence_level: float = 0.95,
    n_samples: int = 100_000,
) -> tuple[float, float]:
    """Calculate the Highest Density Interval (HDI) via Monte Carlo sampling.

    Draws samples from the Beta posterior and finds the shortest interval
    that contains the specified probability mass.

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    prior_alpha : int, optional
        Alpha parameter of the Beta prior distribution, by default 1.
    prior_beta : int, optional
        Beta parameter of the Beta prior distribution, by default 1.
    confidence_level : float, optional
        Desired probability mass within the interval, by default 0.95.
    n_samples : int, optional
        Number of samples to draw from the posterior, by default 100_000.

    Returns
    -------
    tuple[float, float]
        A ``(lower_bound, upper_bound)`` pair representing the HDI.
    """
    samples = sample_beta(s, n, prior_alpha, prior_beta, n_samples)
    return calculate_hdi_from_samples(samples, confidence_level)


def calculate_hdi_from_samples(
    samples: np.ndarray[Any, Any] | list[Any],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Highest Density Interval (HDI) from an array of samples.

    Finds the shortest contiguous interval that contains the specified
    probability mass by scanning all candidate intervals of the sorted sample.

    Parameters
    ----------
    samples : np.ndarray or list
        Posterior samples from any distribution.
    confidence_level : float, optional
        Desired probability mass within the interval, by default 0.95.

    Returns
    -------
    tuple[float, float]
        A ``(lower_bound, upper_bound)`` pair representing the HDI.
    """
    n_samples = len(samples)
    sorted_samples: np.ndarray[Any, Any] = np.sort(np.asarray(samples))

    interval_idx_inc = int(np.floor(confidence_level * n_samples))
    n_intervals = n_samples - interval_idx_inc
    interval_widths = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_widths)

    return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx_inc]
