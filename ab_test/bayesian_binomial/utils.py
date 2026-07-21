"""General utility functions."""

from typing import Any

import numpy as np

__all__ = [
    "sample_beta",
    "posterior_mean",
]


def sample_beta(s: int, n: int, alpha: float, beta: float, n_samples: int) -> np.ndarray[Any, Any]:
    """Draw samples from the Beta posterior given observed binomial data.

    Combines the observed data with the Beta prior to form the posterior
    Beta(alpha + s, beta + n - s) and draws samples from it.

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    alpha : float
        Alpha parameter of the Beta prior distribution.
    beta : float
        Beta parameter of the Beta prior distribution.
    n_samples : int
        Number of samples to draw from the posterior.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples,)`` containing draws from the posterior.
    """
    return np.random.beta(alpha + s, beta + n - s, n_samples)


def posterior_mean(s: int, n: int, alpha: float, beta: float) -> float:
    """Compute the mean of the Beta posterior given observed binomial data.

    Combines the observed data with the Beta prior to form the posterior
    Beta(alpha + s, beta + n - s) and returns its mean: (alpha + s) / (alpha + beta + n).

    Parameters
    ----------
    s : int
        Number of successes observed.
    n : int
        Total number of trials.
    alpha : float
        Alpha parameter of the Beta prior distribution.
    beta : float
        Beta parameter of the Beta prior distribution.

    Returns
    -------
    float
        The mean of the posterior Beta distribution.
    """
    alpha_post = alpha + s
    beta_post = beta + (n - s)
    return alpha_post / (alpha_post + beta_post)
