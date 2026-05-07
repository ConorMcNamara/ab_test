from typing import Any, Literal

import numpy as np


def bayes_power_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    n_samples: int = 100_000,
    mc_samples: int = 1_000,
    confidence_level: float = 0.95,
) -> float:
    """Estimate the Bayesian power of a two-variant binomial experiment via simulation.

    Simulates ``n_samples`` experiments under the alternative hypothesis and returns
    the proportion in which P(B > A) meets or exceeds ``confidence_level``.

    The treatment rate can be specified in two mutually exclusive ways:

    - Pass ``alt_lift`` and ``lift`` to derive the rate from the baseline.
    - Pass ``alt_rate`` directly as the raw treatment conversion rate.

    Parameters
    ----------
    group_sizes : np.ndarray or list
        Trial counts for each variant. When more than two are provided the two
        smallest are used, as they govern overall power.
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    alt_lift : float, optional
        Expected lift of the treatment over the control. Interpreted according to
        ``lift``: a relative multiplier (e.g. ``0.10`` → +10%) or an absolute
        addition (e.g. ``0.02`` → +2 pp). Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline`` to derive the treatment rate.
        Ignored when ``alt_rate`` is provided. Default is ``"relative"``.
    n_samples : int, optional
        Number of simulated experiments, by default 100_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate
        P(B > A), by default 1_000.
    confidence_level : float, optional
        Posterior probability threshold that defines a "win". Power is the
        fraction of simulations where P(B > A) >= this value, by default 0.95.

    Returns
    -------
    float
        Estimated Bayesian power in [0, 1].

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or
        ``"absolute"``.
    """
    if len(group_sizes) > 2:
        # Get two smallest groups — they govern the overall power
        a, b, *_ = np.partition(group_sizes, 1)
        group_sizes = [a, b]

    if alt_rate is None:
        if alt_lift is None:
            raise ValueError("Provide either alt_lift or alt_rate")
        if lift == "relative":
            alt_rate = baseline * (1 + alt_lift)
        elif lift == "absolute":
            alt_rate = baseline + alt_lift
        else:
            raise NotImplementedError(f"lift '{lift}' not implemented")

    successes_null = np.random.binomial(group_sizes[0], baseline, size=n_samples)
    successes_alt = np.random.binomial(group_sizes[1], alt_rate, size=n_samples)
    null_alpha, null_beta = alphas[0] + successes_null, betas[0] + group_sizes[0] - successes_null
    alt_alpha, alt_beta = alphas[1] + successes_alt, betas[1] + group_sizes[1] - successes_alt

    samples_null = np.random.beta(null_alpha[:, np.newaxis], null_beta[:, np.newaxis], size=(n_samples, mc_samples))
    samples_alt = np.random.beta(alt_alpha[:, np.newaxis], alt_beta[:, np.newaxis], size=(n_samples, mc_samples))

    prob_b_better = np.mean(samples_alt > samples_null, axis=1)
    power = float(np.mean(prob_b_better >= confidence_level))
    return power


def bayes_power_loss(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    n_samples: int = 100_000,
    mc_samples: int = 1_000,
    loss_threshold: float = 0.001,
) -> float:
    """Estimate the Bayesian power of a two-variant binomial experiment via expected loss.

    Simulates ``n_samples`` experiments under the alternative hypothesis and returns
    the proportion in which the expected loss of choosing B over A falls at or below
    ``loss_threshold``. A simulation is counted as a "win" when
    E[max(A − B, 0)] ≤ ``loss_threshold``, meaning the downside risk of picking B
    is acceptably small.

    The loss is computed in rate units (percentage points). A ``loss_threshold`` of
    ``0.001`` means "acceptable to lose at most 0.1 pp of conversion rate." Choose
    this value relative to your baseline — for a 10% baseline, 0.001 represents 1%
    of the baseline rate.

    The treatment rate can be specified in two mutually exclusive ways:

    - Pass ``alt_lift`` and ``lift`` to derive the rate from the baseline.
    - Pass ``alt_rate`` directly as the raw treatment conversion rate.

    Parameters
    ----------
    group_sizes : np.ndarray or list
        Trial counts for each variant. When more than two are provided the two
        smallest are used, as they govern overall power.
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    alt_lift : float, optional
        Expected lift of the treatment over the control. Interpreted according to
        ``lift``: a relative multiplier (e.g. ``0.10`` → +10%) or an absolute
        addition (e.g. ``0.02`` → +2 pp). Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline`` to derive the treatment rate.
        Ignored when ``alt_rate`` is provided. Default is ``"relative"``.
    n_samples : int, optional
        Number of simulated experiments, by default 100_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate the
        expected loss, by default 1_000.
    loss_threshold : float, optional
        Maximum acceptable expected loss in rate units. A simulation counts as a
        "win" when E[max(A − B, 0)] <= this value, by default 0.001.

    Returns
    -------
    float
        Estimated Bayesian power in [0, 1].

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or
        ``"absolute"``.
    """
    if len(group_sizes) > 2:
        # Get two smallest groups — they govern the overall power
        a, b, *_ = np.partition(group_sizes, 1)
        group_sizes = [a, b]

    if alt_rate is None:
        if alt_lift is None:
            raise ValueError("Provide either alt_lift or alt_rate")
        if lift == "relative":
            alt_rate = baseline * (1 + alt_lift)
        elif lift == "absolute":
            alt_rate = baseline + alt_lift
        else:
            raise NotImplementedError(f"lift '{lift}' not implemented")

    successes_null = np.random.binomial(group_sizes[0], baseline, size=n_samples)
    successes_alt = np.random.binomial(group_sizes[1], alt_rate, size=n_samples)
    null_alpha, null_beta = alphas[0] + successes_null, betas[0] + group_sizes[0] - successes_null
    alt_alpha, alt_beta = alphas[1] + successes_alt, betas[1] + group_sizes[1] - successes_alt

    samples_null = np.random.beta(null_alpha[:, np.newaxis], null_beta[:, np.newaxis], size=(n_samples, mc_samples))
    samples_alt = np.random.beta(alt_alpha[:, np.newaxis], alt_beta[:, np.newaxis], size=(n_samples, mc_samples))

    expected_loss = np.mean(np.maximum(samples_null - samples_alt, 0), axis=1)
    power = float(np.mean(expected_loss <= loss_threshold))
    return power


def bayes_minimum_sample_size(
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_n: int = 1_000_000,
) -> int:
    """Find the minimum per-group sample size that achieves a target Bayesian power.

    Uses a two-phase search: first doubles a candidate size from 100 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest n that suffices.

    Because power estimates are stochastic, results may vary slightly between
    calls. Increase ``n_samples`` for a more stable (but slower) result.

    Parameters
    ----------
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    alt_lift : float, optional
        Expected lift of the treatment over the control. Interpreted according to
        ``lift``: a relative multiplier (e.g. ``0.10`` → +10%) or an absolute
        addition (e.g. ``0.02`` → +2 pp). Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline`` to derive the treatment rate.
        Ignored when ``alt_rate`` is provided. Default is ``"relative"``.
    target_power : float, optional
        Minimum acceptable Bayesian power, by default 0.80.
    confidence_level : float, optional
        Posterior probability threshold that defines a "win" inside each power
        simulation, by default 0.95.
    n_samples : int, optional
        Number of simulated experiments per power evaluation, by default 10_000.
        Higher values reduce noise at the cost of speed.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate
        P(B > A), by default 500.
    max_n : int, optional
        Upper bound on the per-group sample size search. A ``ValueError`` is raised
        if ``target_power`` cannot be reached within this limit, by default
        1_000_000.

    Returns
    -------
    int
        Smallest per-group sample size estimated to reach ``target_power``.

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    ValueError
        If ``target_power`` cannot be reached within ``max_n`` samples per group.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or
        ``"absolute"``.
    """

    def _power(n: int) -> float:
        return bayes_power_lift(
            group_sizes=[n, n],
            alphas=alphas,
            betas=betas,
            baseline=baseline,
            alt_lift=alt_lift,
            alt_rate=alt_rate,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            confidence_level=confidence_level,
        )

    # Phase 1: double from 100 until power exceeds target or we hit the cap
    low, high = 100, 200
    while high <= max_n:
        if _power(high) >= target_power:
            break
        low, high = high, high * 2
    else:
        raise ValueError(
            f"Could not reach target power of {target_power} within "
            f"{max_n:,} samples per group. "
            "Consider a larger effect size."
        )

    # Phase 2: binary search within [low, high]
    while high - low > 1:
        mid = (low + high) // 2
        if _power(mid) >= target_power:
            high = mid
        else:
            low = mid

    return high
