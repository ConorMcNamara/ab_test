"""Statistical tests to determine significance."""

from typing import Any

import numpy as np

from ab_test.bayesian_binomial.utils import sample_beta


def calculate_rope(
    sample_a: np.ndarray[Any, Any] | list[Any],
    sample_b: np.ndarray[Any, Any] | list[Any],
    lift: str = "relative",
    low: float = -0.01,
    high: float = 0.01,
    trials: tuple[int, int] | None = None,
    spend: float | None = None,
    msrp: float | None = None,
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
    lift : {"relative", "absolute", "incremental", "revenue", "roas"}, optional
        How to compute the lift between variants:

        - ``"relative"``: ``(B - A) / A``
        - ``"absolute"``: ``B - A``
        - ``"incremental"``: ``(B - A) * max(trials)`` — count difference at scale.
          Requires ``trials``.
        - ``"revenue"``: ``(B - A) * max(trials) * msrp`` — incremental revenue.
          Requires ``trials`` and ``msrp``.
        - ``"roas"``: ``spend / (A * max_n) - spend / (B * max_n)`` — difference
          in cost-per-acquisition; positive means B is cheaper. Requires ``trials``
          and ``spend``.

        Default is ``"relative"``.
    low : float, optional
        Lower bound of the ROPE. Default is -0.01.
    high : float, optional
        Upper bound of the ROPE. Default is 0.01.
    trials : tuple[int, int], optional
        ``(trials_a, trials_b)`` — required for ``"incremental"``, ``"revenue"``,
        and ``"roas"``.
    spend : float, optional
        Total ad spend — required for ``"roas"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

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
        If ``lift`` is not one of the supported types.
    ValueError
        If a required parameter (``trials``, ``spend``, or ``msrp``) is missing
        for the chosen lift type.
    """
    a = np.asarray(sample_a)
    b = np.asarray(sample_b)
    if lift == "relative":
        lift_arr = (b - a) / a
    elif lift == "absolute":
        lift_arr = b - a
    elif lift in ("incremental", "revenue", "roas"):
        if trials is None:
            raise ValueError(f"trials must be provided for lift='{lift}'")
        max_n = max(trials)
        if lift == "incremental":
            lift_arr = (b - a) * max_n
        elif lift == "revenue":
            if msrp is None:
                raise ValueError("msrp must be provided for lift='revenue'")
            lift_arr = (b - a) * max_n * msrp
        else:  # roas
            if spend is None:
                raise ValueError("spend must be provided for lift='roas'")
            # CPA_A - CPA_B: positive means B has lower cost-per-conversion (better)
            lift_arr = spend / (a * max_n) - spend / (b * max_n)
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
    successes: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    n_samples: int,
    lift: str = "relative",
    low_threshold: float = -0.01,
    high_threshold: float = 0.01,
    spend: float | None = None,
    msrp: float | None = None,
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
    lift : {"relative", "absolute", "incremental", "revenue", "roas"}, optional
        How to compute the lift between variants. Default is ``"relative"``.
        See :func:`calculate_rope` for full semantics of each mode.
    low_threshold : float, optional
        Lower bound of the ROPE. Default is -0.01.
    high_threshold : float, optional
        Upper bound of the ROPE. Default is 0.01.
    spend : float, optional
        Total ad spend — required when ``lift="roas"``.
    msrp : float, optional
        Average product price — required when ``lift="revenue"``.

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
    rope = calculate_rope(
        sample_a,
        sample_b,
        lift=lift,
        low=low_threshold,
        high=high_threshold,
        trials=(int(trials[0]), int(trials[1])),
        spend=spend,
        msrp=msrp,
    )
    return {
        "Proportion of samples where B exceeds A": prob_b_greater_a,
        "Expected loss": expected_loss,
        "Probability of ROPE": rope["prob_in_rope"],
        f"Probability {lift} exceeds {high_threshold}": rope["prob_lift_exceeds"],
        f"Probability {lift} is below {low_threshold}": rope["prob_lift_drops"],
    }
