from typing import Any

import numpy as np

from ab_test.bayesian_binomial.utils import sample_beta


def calculate_rope(
    sample_a: np.ndarray[Any, Any] | list[Any],
    sample_b: np.ndarray[Any, Any] | list[Any],
    lift: str = "relative",
    low: float = -0.01,
    high: float = 0.01,
) -> dict:
    """Calculate the probability that the lift between B and A falls within the ROPE.

    The Region of Practical Equivalence (ROPE) is the interval [low, high]
    within which the difference between variants is considered negligible.

    Parameters
    ----------
    sample_a : np.ndarray
        Posterior samples for variant A.
    sample_b : np.ndarray
        Posterior samples for variant B.
    lift : {"relative", "absolute"}, optional
        How to compute the lift between variants. "relative" computes
        (B - A) / A; "absolute" computes B - A. Default is "relative".
    low : float, optional
        Lower bound of the ROPE. Default is -0.01 (-1% lift).
    high : float, optional
        Upper bound of the ROPE. Default is 0.01 (+1% lift).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``prob_in_rope`` : float
            Probability that the lift falls within [low, high].
        - ``prob_lift_exceeds`` : float
            Probability that B is practically better than A (lift > high).
        - ``prob_lift_drops`` : float
            Probability that B is practically worse than A (lift < low).

    Raises
    ------
    NotImplementedError
        If ``lift`` is not "relative" or "absolute".
    """
    a = np.asarray(sample_a)
    b = np.asarray(sample_b)
    if lift == "relative":
        lift_arr = (b - a) / a
    elif lift == "absolute":
        lift_arr = b - a
    else:
        raise NotImplementedError(f"lift {lift} not implemented")

    # Calculate what % of samples fall within the ROPE
    prob_in_rope = np.mean((lift_arr >= low) & (lift_arr <= high))

    # Also useful: Prob B is practically better (above ROPE)
    prob_better = np.mean(lift_arr > high)

    # Also useful: Prob B is practically worse (below ROPE)
    prob_worse = np.mean(lift_arr < low)

    return {"prob_in_rope": prob_in_rope, "prob_lift_exceeds": prob_better, "prob_lift_drops": prob_worse}


def probability_b_greater_than_a(sample_a: np.ndarray[Any] | list[Any], sample_b: np.ndarray[Any] | list[Any]) -> float:
    """Estimate the probability that variant B is greater than variant A.

    Parameters
    ----------
    sample_a : np.ndarray or list
        Posterior samples for variant A.
    sample_b : np.ndarray or list
        Posterior samples for variant B.

    Returns
    -------
    float
        Proportion of samples where B exceeds A.
    """
    return calculate_rope(sample_a, sample_b, lift="absolute", high=0.0)["prob_lift_exceeds"]


def expected_loss_b(sample_a: np.ndarray[Any], sample_b: np.ndarray[Any]) -> float:
    """Compute the expected loss from choosing variant B over variant A.

    The expected loss is the average amount by which A would exceed B across
    posterior samples, i.e. E[max(A - B, 0)]. A low value indicates that
    choosing B carries little risk of underperforming A.

    Parameters
    ----------
    sample_a : np.ndarray
        Posterior samples for variant A.
    sample_b : np.ndarray
        Posterior samples for variant B.

    Returns
    -------
    float
        Expected loss incurred by selecting variant B.
    """
    loss_b = np.maximum(sample_a - sample_b, 0).mean()
    return loss_b


def prob_lift_exceeds(
    sample_a: np.ndarray[Any], sample_b: np.ndarray[Any], lift: str = "relative", threshold: float = 0.01
) -> float:
    """Calculate the probability that the lift of B over A exceeds a threshold.

    Parameters
    ----------
    sample_a : np.ndarray
        Posterior samples for variant A.
    sample_b : np.ndarray
        Posterior samples for variant B.
    lift : {"relative", "absolute"}, optional
        How to compute the lift between variants. "relative" computes
        (B - A) / A; "absolute" computes B - A. Default is "relative".
    threshold : float, optional
        The minimum lift value to exceed. Default is 0.01 (+1% lift).

    Returns
    -------
    float
        Probability that the lift of B over A is greater than ``threshold``.
    """
    return calculate_rope(sample_a, sample_b, lift, high=threshold)["prob_lift_exceeds"]


def calculate_metrics(
    successes: np.ndarray[Any, Any],
    trials: np.ndarray[Any, Any],
    alphas: np.ndarray[Any, Any],
    betas: np.ndarray[Any, Any],
    n_samples: int,
    lift: str = "relative",
    low_threshold: float = -0.01,
    high_threshold: float = 0.01,
) -> dict[str, float]:
    """Compute a suite of Bayesian metrics comparing variant B against variant A.

    Draws posterior samples for each variant and returns the probability that B
    beats A, the expected loss from choosing B, and a full ROPE breakdown.

    Parameters
    ----------
    successes : np.ndarray
        Array of length 2 containing the number of successes for variants A and B.
    trials : np.ndarray
        Array of length 2 containing the number of trials for variants A and B.
    alphas : np.ndarray
        Array of length 2 containing the alpha prior parameters for variants A and B.
    betas : np.ndarray
        Array of length 2 containing the beta prior parameters for variants A and B.
    n_samples : int
        Number of posterior samples to draw for each variant.
    lift : {"relative", "absolute"}, optional
        How to compute the lift between variants. "relative" computes
        (B - A) / A; "absolute" computes B - A. Default is "relative".
    low_threshold : float, optional
        Lower bound of the ROPE. Default is -0.01 (-1% lift).
    high_threshold : float, optional
        Upper bound of the ROPE. Default is 0.01 (+1% lift).

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        - ``"Proportion of samples where B exceeds A"`` : float
        - ``"Expected loss"`` : float
        - ``"Probability of ROPE"`` : float
        - ``f"Probability {lift} exceeds {high_threshold}"`` : float
        - ``f"Probability {lift} is below {low_threshold}"`` : float
    """
    sample_a = sample_beta(successes[0], trials[0], alphas[0], betas[0], n_samples)
    sample_b = sample_beta(successes[1], trials[1], alphas[1], betas[1], n_samples)
    prob_b_greater_a = float(np.mean(sample_b > sample_a))
    expected_loss = expected_loss_b(sample_a, sample_b)
    rope = calculate_rope(sample_a, sample_b, lift=lift, low=low_threshold, high=high_threshold)
    return {
        "Proportion of samples where B exceeds A": prob_b_greater_a,
        "Expected loss": expected_loss,
        "Probability of ROPE": rope["prob_in_rope"],
        f"Probability {lift} exceeds {high_threshold}": rope["prob_lift_exceeds"],
        f"Probability {lift} is below {low_threshold}": rope["prob_lift_drops"],
    }
