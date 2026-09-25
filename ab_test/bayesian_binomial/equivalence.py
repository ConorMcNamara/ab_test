"""Bayesian equivalence testing via the ROPE (Region of Practical Equivalence)."""

from typing import Any

import numpy as np

__all__ = ["bayes_equivalence_test"]


def _compute_lift(
    samples_a: np.ndarray[Any, Any],
    samples_b: np.ndarray[Any, Any],
    lift: str,
    trials: tuple[int, int] | None,
    spend: float | None,
    msrp: float | None,
) -> np.ndarray[Any, Any]:
    """Compute the posterior lift distribution between two sets of samples."""
    if lift == "relative":
        return (samples_b - samples_a) / samples_a
    if lift == "absolute":
        return samples_b - samples_a

    if trials is None:
        raise ValueError(f"trials must be provided for lift='{lift}'")
    max_n = max(trials)
    diff = samples_b - samples_a

    if lift == "incremental":
        return diff * max_n
    if lift == "revenue":
        if msrp is None:
            raise ValueError("msrp must be provided for lift='revenue'")
        return diff * max_n * msrp
    if lift == "roas":
        if spend is None:
            raise ValueError("spend must be provided for lift='roas'")
        return diff * max_n / spend
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be provided for lift='cpa'")
        incremental = diff * max_n
        return np.where(np.abs(incremental) > 1e-12, spend / incremental, np.inf)

    raise NotImplementedError(f"lift '{lift}' not implemented")


def bayes_equivalence_test(
    successes: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    delta: float,
    n_samples: int = 10_000,
    lift: str = "absolute",
    threshold: float = 0.95,
    spend: float | None = None,
    msrp: float | None = None,
    *,
    seed: int | None = None,
) -> dict[str, float | bool]:
    """Bayesian equivalence test using a Region of Practical Equivalence (ROPE).

    Draws posterior samples for each variant's success rate and computes
    the probability that the lift between B and A falls within
    ``[-delta, delta]``.  If that probability exceeds ``threshold``, the
    variants are declared practically equivalent.

    Parameters
    ----------
    successes : array_like
        Number of successes for variants A and B, length 2.
    trials : array_like
        Number of trials for variants A and B, length 2.
    alphas : array_like
        Alpha parameters of the Beta prior for variants A and B, length 2.
    betas : array_like
        Beta parameters of the Beta prior for variants A and B, length 2.
    delta : float
        Half-width of the equivalence region.  Must be positive.  The ROPE
        is ``[-delta, delta]`` in the scale specified by ``lift``.
    n_samples : int, optional
        Number of posterior samples to draw per variant.  Default is 10 000.
    lift : {"absolute", "relative", "incremental", "revenue", "roas", "cpa"}
        How to compute the lift between variants.  Default is ``"absolute"``.
    threshold : float, optional
        Posterior probability threshold for declaring equivalence.
        Default is 0.95.
    spend : float, optional
        Total ad spend — required when ``lift`` is ``"roas"`` or ``"cpa"``.
    msrp : float, optional
        Average product price — required when ``lift`` is ``"revenue"``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float | bool]
        Dictionary with the following keys:

        - ``"prob_equivalent"`` : float — posterior probability that the
          lift falls within ``[-delta, delta]``.
        - ``"equivalent"`` : bool — ``True`` when
          ``prob_equivalent >= threshold``.
        - ``"prob_superior"`` : float — posterior probability that the
          lift exceeds ``delta`` (B is meaningfully better).
        - ``"prob_inferior"`` : float — posterior probability that the
          lift is below ``-delta`` (B is meaningfully worse).

    Raises
    ------
    ValueError
        If ``delta <= 0``, or required parameters for the chosen ``lift``
        are missing.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    successes_arr = np.asarray(successes)
    trials_arr = np.asarray(trials)
    alphas_arr = np.asarray(alphas)
    betas_arr = np.asarray(betas)

    rng = np.random.default_rng(seed)

    alpha_post_a = alphas_arr[0] + successes_arr[0]
    beta_post_a = betas_arr[0] + trials_arr[0] - successes_arr[0]
    alpha_post_b = alphas_arr[1] + successes_arr[1]
    beta_post_b = betas_arr[1] + trials_arr[1] - successes_arr[1]

    samples_a = rng.beta(alpha_post_a, beta_post_a, size=n_samples)
    samples_b = rng.beta(alpha_post_b, beta_post_b, size=n_samples)

    trial_pair = (int(trials_arr[0]), int(trials_arr[1]))
    lift_arr = _compute_lift(samples_a, samples_b, lift, trial_pair, spend, msrp)

    prob_equivalent = float(np.mean((lift_arr >= -delta) & (lift_arr <= delta)))
    prob_superior = float(np.mean(lift_arr > delta))
    prob_inferior = float(np.mean(lift_arr < -delta))

    return {
        "prob_equivalent": prob_equivalent,
        "equivalent": prob_equivalent >= threshold,
        "prob_superior": prob_superior,
        "prob_inferior": prob_inferior,
    }
