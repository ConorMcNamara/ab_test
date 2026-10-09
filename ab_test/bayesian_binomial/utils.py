"""General utility functions."""

from typing import Any

import numpy as np

from ab_test._lift import from_absolute

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


def _default_rope_half_width(
    control_rate: float,
    lift: str,
    scale: int | float,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Half-width of the default ROPE: 10% of the control rate, in ``lift`` units.

    For relative lift that is 0.1; ``"cpa"`` also returns 0.1, though its ROPE
    is not computed. ``scale`` is the number of units scaled lifts are
    expressed over, as used for the reported lift.
    """
    if lift in ("relative", "cpa"):
        return 0.1
    return float(from_absolute(0.1 * control_rate, lift, int(scale), spend, msrp))


def _between_group_sd_samples(
    estimates: np.ndarray[Any, Any] | list[float],
    variances: np.ndarray[Any, Any] | list[float],
    n_draws: int,
    n_grid: int = 2000,
) -> np.ndarray[Any, Any]:
    """Posterior draws of the between-group standard deviation of an effect.

    Uses the normal random-effects model: each group's estimate is
    ``y_k ~ N(theta_k, v_k)`` with ``theta_k ~ N(mu, tau^2)``. ``mu`` has a
    flat prior and is integrated out analytically; ``tau`` has a half-Cauchy
    prior whose scale is the median within-group standard error, and its
    posterior is evaluated on a grid (Gelman et al., *Bayesian Data
    Analysis*, 3rd ed., section 5.4; Gelman, 2006). Unlike the spread of
    the group estimates, this does not count within-group noise as
    heterogeneity, so it concentrates near zero when the groups agree.

    Parameters
    ----------
    estimates : array_like
        Posterior mean effect in each group.
    variances : array_like
        Posterior variance of the effect in each group.
    n_draws : int
        Number of posterior draws of ``tau`` to return.
    n_grid : int, default=2000
        Number of grid points for ``tau``.

    Returns
    -------
    np.ndarray
        Draws from the posterior of ``tau``, in the units of ``estimates``.
    """
    y = np.asarray(estimates, dtype=float)
    v = np.asarray(variances, dtype=float)
    spread = float(np.ptp(y))
    scale = float(np.sqrt(np.median(v))) or max(spread, 1e-12)
    tau = np.linspace(0.0, 20.0 * max(scale, spread), n_grid)
    w = 1.0 / (v[None, :] + tau[:, None] ** 2)
    mu_hat = (w * y).sum(axis=1) / w.sum(axis=1)
    log_post = (
        0.5 * np.log(w).sum(axis=1)
        - 0.5 * np.log(w.sum(axis=1))
        - 0.5 * (w * (y - mu_hat[:, None]) ** 2).sum(axis=1)
        - np.log1p((tau / scale) ** 2)
    )
    probs = np.exp(log_post - log_post.max())
    probs /= probs.sum()
    step = tau[1] - tau[0]
    draws = tau[np.random.choice(n_grid, n_draws, p=probs)] + np.random.uniform(-step / 2, step / 2, n_draws)
    return np.abs(draws)
