"""Methods to calculate the power of a test."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

__all__ = [
    "bayes_power_lift",
    "bayes_power_loss",
    "bayes_minimum_sample_size",
    "bayes_minimum_sample_size_loss",
    "bayes_minimum_detectable_lift",
    "bayes_minimum_detectable_lift_loss",
    "plot_bayes_power_curve",
    "plot_bayes_sensitivity_curve",
]

_SCALED_LIFTS = {"incremental", "roas", "revenue", "cpa"}


def _to_absolute(
    lift_value: float,
    lift: str,
    scale: int,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Convert a lift value from the given lift type to absolute."""
    if lift in ("relative", "absolute"):
        return lift_value
    if lift == "incremental":
        return lift_value / scale
    if lift == "roas":
        if spend is None:
            raise ValueError("spend must be set for ROAS calculations")
        return lift_value * spend / scale
    if lift == "revenue":
        if msrp is None:
            raise ValueError("msrp must be set for revenue calculations")
        return lift_value / (scale * msrp)
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be set for CPA calculations")
        return spend / (lift_value * scale)
    raise ValueError(f"Unsupported lift type: {lift}")


def _from_absolute(
    abs_value: float,
    lift: str,
    scale: int,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Convert an absolute lift value to the given lift type."""
    if lift in ("relative", "absolute"):
        return abs_value
    if lift == "incremental":
        return abs_value * scale
    if lift == "roas":
        if spend is None:
            raise ValueError("spend must be set for ROAS calculations")
        return abs_value * scale / spend
    if lift == "revenue":
        if msrp is None:
            raise ValueError("msrp must be set for revenue calculations")
        return abs_value * scale * msrp
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be set for CPA calculations")
        return spend / (abs_value * scale) if abs_value != 0 else np.inf
    raise ValueError(f"Unsupported lift type: {lift}")


def _resolve_alt_rate(
    baseline: float,
    alt_lift: float | None,
    alt_rate: float | None,
    lift: Literal["relative", "absolute"],
) -> float:
    """Resolve the treatment conversion rate from either ``alt_rate`` or ``alt_lift``.

    Parameters
    ----------
    baseline : float
        Expected conversion rate of the control variant.
    alt_lift : float or None
        Expected lift of the treatment over the control, interpreted per ``lift``.
    alt_rate : float or None
        Treatment conversion rate specified directly. Takes precedence over ``alt_lift``.
    lift : {"relative", "absolute"}
        How ``alt_lift`` is applied to ``baseline``. Ignored when ``alt_rate`` is given.

    Returns
    -------
    float
        The treatment conversion rate.

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or ``"absolute"``.
    """
    if alt_rate is not None:
        return alt_rate
    if alt_lift is None:
        raise ValueError("Provide either alt_lift or alt_rate")
    if lift == "relative":
        return baseline * (1 + alt_lift)
    if lift == "absolute":
        return baseline + alt_lift
    raise NotImplementedError(f"lift '{lift}' not implemented")


def _two_smallest_group_sizes(group_sizes: np.ndarray[Any, Any] | list[Any]) -> np.ndarray[Any, Any] | list[Any]:
    """Return the two smallest group sizes, which govern overall power."""
    if len(group_sizes) > 2:
        a, b, *_ = np.partition(group_sizes, 1)
        return [a, b]
    return group_sizes


def _simulate_posterior_draws(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_rate: float,
    n_samples: int,
    mc_samples: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Simulate posterior draws for the control and treatment across ``n_samples`` experiments.

    Draws ``n_samples`` simulated experiments (one binomial outcome per arm) then,
    for each, ``mc_samples`` posterior draws from the resulting Beta posterior.

    Returns
    -------
    tuple of np.ndarray
        ``(samples_null, samples_alt)``, each of shape ``(n_samples, mc_samples)``.
    """
    successes_null = np.random.binomial(group_sizes[0], baseline, size=n_samples)
    successes_alt = np.random.binomial(group_sizes[1], alt_rate, size=n_samples)
    null_alpha, null_beta = alphas[0] + successes_null, betas[0] + group_sizes[0] - successes_null
    alt_alpha, alt_beta = alphas[1] + successes_alt, betas[1] + group_sizes[1] - successes_alt

    samples_null = np.random.beta(null_alpha[:, np.newaxis], null_beta[:, np.newaxis], size=(n_samples, mc_samples))
    samples_alt = np.random.beta(alt_alpha[:, np.newaxis], alt_beta[:, np.newaxis], size=(n_samples, mc_samples))
    return samples_null, samples_alt


def _search_min_sample_size(
    power_fn: Callable[[int], float],
    target_power: float,
    max_n: int,
    error_message: str,
) -> int:
    """Find the smallest per-group sample size reaching ``target_power``.

    Doubles a candidate size from 100 until ``power_fn`` meets ``target_power``,
    then binary-searches the resulting bracket.

    Raises
    ------
    ValueError
        With ``error_message`` if ``target_power`` is not reached within ``max_n``.
    """
    low, high = 100, 200
    while high <= max_n:
        if power_fn(high) >= target_power:
            break
        low, high = high, high * 2
    else:
        raise ValueError(error_message)

    while high - low > 1:
        mid = (low + high) // 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def _search_min_lift(
    power_fn: Callable[[float], float],
    target_power: float,
    max_lift: float,
    tol: float,
    error_message: str,
) -> float:
    """Find the smallest lift reaching ``target_power``.

    Doubles a candidate lift from 0.01 until ``power_fn`` meets ``target_power``,
    then binary-searches the resulting bracket to within ``tol``.

    Raises
    ------
    ValueError
        With ``error_message`` if ``target_power`` is not reached within ``max_lift``.
    """
    low, high = 0.0, 0.01
    while high <= max_lift:
        if power_fn(high) >= target_power:
            break
        low, high = high, high * 2
    else:
        raise ValueError(error_message)

    while high - low > tol:
        mid = (low + high) / 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def bayes_power_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: str = "relative",
    n_samples: int = 100_000,
    mc_samples: int = 1_000,
    confidence_level: float = 0.95,
    spend: float | None = None,
    msrp: float | None = None,
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
        ``lift``. Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
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
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.

    Returns
    -------
    float
        Estimated Bayesian power in [0, 1].

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not supported.
    """
    group_sizes = _two_smallest_group_sizes(group_sizes)
    if alt_rate is None and alt_lift is not None and lift in _SCALED_LIFTS:
        scale = max(group_sizes)
        alt_lift = _to_absolute(alt_lift, lift, scale, spend, msrp)
        lift = "absolute"
    alt_rate = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    samples_null, samples_alt = _simulate_posterior_draws(
        group_sizes, alphas, betas, baseline, alt_rate, n_samples, mc_samples
    )

    prob_b_better = np.mean(samples_alt > samples_null, axis=1)
    return float(np.mean(prob_b_better >= confidence_level))


def bayes_power_loss(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: str = "relative",
    n_samples: int = 100_000,
    mc_samples: int = 1_000,
    loss_threshold: float = 0.001,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Estimate the Bayesian power of a two-variant binomial experiment via expected loss.

    Simulates ``n_samples`` experiments under the alternative hypothesis and returns
    the proportion in which the expected loss of choosing B over A falls at or below
    ``loss_threshold``. A simulation is counted as a "win" when
    E[max(A − B, 0)] ≤ ``loss_threshold``, meaning the downside risk of picking B
    is acceptably small.

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
        ``lift``. Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
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
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.

    Returns
    -------
    float
        Estimated Bayesian power in [0, 1].

    Raises
    ------
    ValueError
        If neither ``alt_lift`` nor ``alt_rate`` is provided.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not supported.
    """
    group_sizes = _two_smallest_group_sizes(group_sizes)
    if alt_rate is None and alt_lift is not None and lift in _SCALED_LIFTS:
        scale = max(group_sizes)
        alt_lift = _to_absolute(alt_lift, lift, scale, spend, msrp)
        lift = "absolute"
    alt_rate = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    samples_null, samples_alt = _simulate_posterior_draws(
        group_sizes, alphas, betas, baseline, alt_rate, n_samples, mc_samples
    )

    expected_loss = np.mean(np.maximum(samples_null - samples_alt, 0), axis=1)
    return float(np.mean(expected_loss <= loss_threshold))


def bayes_minimum_sample_size_loss(
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: str = "relative",
    target_power: float = 0.80,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_n: int = 1_000_000,
) -> int:
    """Find the minimum per-group sample size that achieves a target Bayesian power via expected loss.

    Uses a two-phase search: first doubles a candidate size from 100 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest n that suffices.

    A simulation counts as a "win" when E[max(A − B, 0)] <= ``loss_threshold``,
    meaning the downside risk of picking B is acceptably small.

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
        ``lift``. Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline`` to derive the treatment rate.
        Ignored when ``alt_rate`` is provided. Default is ``"relative"``.
        Scaled lift types (incremental, roas, revenue, cpa) are not
        supported because the effect size depends on the unknown sample size.
    target_power : float, optional
        Minimum acceptable Bayesian power, by default 0.80.
    loss_threshold : float, optional
        Maximum acceptable expected loss in rate units used inside each power
        simulation, by default 0.001.
    n_samples : int, optional
        Number of simulated experiments per power evaluation, by default 10_000.
        Higher values reduce noise at the cost of speed.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate the
        expected loss, by default 500.
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
        If neither ``alt_lift`` nor ``alt_rate`` is provided, or if a scaled
        lift type is used.
    ValueError
        If ``target_power`` cannot be reached within ``max_n`` samples per group.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or
        ``"absolute"``.
    """
    if lift in _SCALED_LIFTS:
        raise ValueError(
            f"lift={lift!r} is not supported for bayes_minimum_sample_size_loss "
            f"because the absolute effect size depends on the group sizes being "
            f"solved for. Convert to 'relative' or 'absolute' lift first."
        )

    def _power(n: int) -> float:
        return bayes_power_loss(
            group_sizes=[n, n],
            alphas=alphas,
            betas=betas,
            baseline=baseline,
            alt_lift=alt_lift,
            alt_rate=alt_rate,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            loss_threshold=loss_threshold,
        )

    return _search_min_sample_size(
        _power,
        target_power,
        max_n,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"{max_n:,} samples per group. "
            "Consider a larger effect size."
        ),
    )


def bayes_minimum_sample_size(
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: str = "relative",
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
        ``lift``. Mutually exclusive with ``alt_rate``.
    alt_rate : float, optional
        Treatment conversion rate specified directly, bypassing the lift
        calculation. Mutually exclusive with ``alt_lift``.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline`` to derive the treatment rate.
        Ignored when ``alt_rate`` is provided. Default is ``"relative"``.
        Scaled lift types (incremental, roas, revenue, cpa) are not
        supported because the effect size depends on the unknown sample size.
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
        If neither ``alt_lift`` nor ``alt_rate`` is provided, or if a scaled
        lift type is used.
    ValueError
        If ``target_power`` cannot be reached within ``max_n`` samples per group.
    NotImplementedError
        If ``alt_lift`` is provided but ``lift`` is not ``"relative"`` or
        ``"absolute"``.
    """
    if lift in _SCALED_LIFTS:
        raise ValueError(
            f"lift={lift!r} is not supported for bayes_minimum_sample_size "
            f"because the absolute effect size depends on the group sizes being "
            f"solved for. Convert to 'relative' or 'absolute' lift first."
        )

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

    return _search_min_sample_size(
        _power,
        target_power,
        max_n,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"{max_n:,} samples per group. "
            "Consider a larger effect size."
        ),
    )


def bayes_minimum_detectable_lift(
    group_size: int,
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    lift: str = "relative",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_lift: float = 10.0,
    tol: float = 0.0001,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Find the minimum lift detectable at a target Bayesian power via P(B > A).

    Uses a two-phase search: first doubles a candidate lift from 0.01 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest lift that suffices.

    Because power estimates are stochastic, results may vary slightly between
    calls. Increase ``n_samples`` for a more stable (but slower) result.

    For ``lift="absolute"``, ensure ``baseline + max_lift <= 1.0``; otherwise
    the implied treatment rate exceeds 1.

    Parameters
    ----------
    group_size : int
        Per-group sample size (equal allocation assumed).
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How the searched lift is applied to ``baseline``. Default is
        ``"relative"``.
    target_power : float, optional
        Minimum acceptable Bayesian power, by default 0.80.
    confidence_level : float, optional
        Posterior probability threshold that defines a "win" inside each power
        simulation, by default 0.95.
    n_samples : int, optional
        Number of simulated experiments per power evaluation, by default 10_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate
        P(B > A), by default 500.
    max_lift : float, optional
        Upper bound on the lift search. A ``ValueError`` is raised if
        ``target_power`` cannot be reached within this value, by default 10.0.
    tol : float, optional
        Convergence tolerance for the binary search. The returned lift is
        accurate to within this value, by default 0.0001.
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.

    Returns
    -------
    float
        Smallest lift estimated to reach ``target_power``, in the units
        specified by ``lift``.

    Raises
    ------
    ValueError
        If ``target_power`` cannot be reached within ``max_lift``.
    NotImplementedError
        If ``lift`` is not supported.
    """
    if lift in _SCALED_LIFTS:
        internal_lift = "absolute"
    else:
        internal_lift = lift

    def _power(alt_lift_val: float) -> float:
        return bayes_power_lift(
            group_sizes=[group_size, group_size],
            alphas=alphas,
            betas=betas,
            baseline=baseline,
            alt_lift=alt_lift_val,
            lift=internal_lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            confidence_level=confidence_level,
        )

    abs_mdl = _search_min_lift(
        _power,
        target_power,
        max_lift,
        tol,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"a lift of {max_lift}. "
            "Consider a smaller target power or larger group size."
        ),
    )
    if lift in _SCALED_LIFTS:
        return _from_absolute(abs_mdl, lift, group_size, spend, msrp)
    return abs_mdl


def bayes_minimum_detectable_lift_loss(
    group_size: int,
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    lift: str = "relative",
    target_power: float = 0.80,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_lift: float = 10.0,
    tol: float = 0.0001,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Find the minimum lift detectable at a target Bayesian power via expected loss.

    Uses a two-phase search: first doubles a candidate lift from 0.01 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest lift that suffices.

    A simulation counts as a "win" when E[max(A − B, 0)] <= ``loss_threshold``.

    Because power estimates are stochastic, results may vary slightly between
    calls. Increase ``n_samples`` for a more stable (but slower) result.

    For ``lift="absolute"``, ensure ``baseline + max_lift <= 1.0``; otherwise
    the implied treatment rate exceeds 1.

    Parameters
    ----------
    group_size : int
        Per-group sample size (equal allocation assumed).
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How the searched lift is applied to ``baseline``. Default is
        ``"relative"``.
    target_power : float, optional
        Minimum acceptable Bayesian power, by default 0.80.
    loss_threshold : float, optional
        Maximum acceptable expected loss in rate units used inside each power
        simulation, by default 0.001.
    n_samples : int, optional
        Number of simulated experiments per power evaluation, by default 10_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment used to estimate the
        expected loss, by default 500.
    max_lift : float, optional
        Upper bound on the lift search. A ``ValueError`` is raised if
        ``target_power`` cannot be reached within this value, by default 10.0.
    tol : float, optional
        Convergence tolerance for the binary search. The returned lift is
        accurate to within this value, by default 0.0001.
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.

    Returns
    -------
    float
        Smallest lift estimated to reach ``target_power``, in the units
        specified by ``lift``.

    Raises
    ------
    ValueError
        If ``target_power`` cannot be reached within ``max_lift``.
    NotImplementedError
        If ``lift`` is not supported.
    """
    if lift in _SCALED_LIFTS:
        internal_lift = "absolute"
    else:
        internal_lift = lift

    def _power(alt_lift_val: float) -> float:
        return bayes_power_loss(
            group_sizes=[group_size, group_size],
            alphas=alphas,
            betas=betas,
            baseline=baseline,
            alt_lift=alt_lift_val,
            lift=internal_lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            loss_threshold=loss_threshold,
        )

    abs_mdl = _search_min_lift(
        _power,
        target_power,
        max_lift,
        tol,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"a lift of {max_lift}. "
            "Consider a smaller target power or larger group size."
        ),
    )
    if lift in _SCALED_LIFTS:
        return _from_absolute(abs_mdl, lift, group_size, spend, msrp)
    return abs_mdl


def plot_bayes_power_curve(
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: str = "relative",
    decision: Literal["lift", "loss"] = "lift",
    confidence_level: float = 0.95,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    n_points: int = 50,
    spend: float | None = None,
    msrp: float | None = None,
) -> go.Figure:
    """Plot Bayesian power as a function of per-group sample size.

    Parameters
    ----------
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    alt_lift : float, optional
        Expected lift of the treatment over the control.
    alt_rate : float, optional
        Treatment conversion rate specified directly.
    lift : {"relative", "absolute"}, optional
        How ``alt_lift`` is applied to ``baseline``. Default is ``"relative"``.
    decision : {"lift", "loss"}, optional
        Decision rule to use: ``"lift"`` uses P(B > A) >= ``confidence_level``,
        ``"loss"`` uses E[max(A-B, 0)] <= ``loss_threshold``. Default is
        ``"lift"``.
    confidence_level : float, optional
        Posterior probability threshold when ``decision="lift"``. Default is 0.95.
    loss_threshold : float, optional
        Maximum acceptable expected loss when ``decision="loss"``. Default is
        0.001.
    n_samples : int, optional
        Number of simulated experiments per power evaluation. Default is 10_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment. Default is 500.
    sample_sizes : array_like or None, optional
        Explicit per-group sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 50.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with per-group sample size on the x-axis
        and Bayesian power on the y-axis.
    """
    power_fn = bayes_power_lift if decision == "lift" else bayes_power_loss

    if sample_sizes is None:
        if lift in _SCALED_LIFTS:
            raise ValueError(
                f"lift={lift!r} requires explicit sample_sizes because the "
                f"automatic range uses minimum sample size search, which does "
                f"not support scaled lift types."
            )
        search_fn = bayes_minimum_sample_size if decision == "lift" else bayes_minimum_sample_size_loss
        common: dict[str, Any] = {
            "alphas": alphas,
            "betas": betas,
            "baseline": baseline,
            "alt_lift": alt_lift,
            "alt_rate": alt_rate,
            "lift": lift,
            "target_power": 0.8,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
        }
        if decision == "lift":
            common["confidence_level"] = confidence_level
        else:
            common["loss_threshold"] = loss_threshold
        target_n = search_fn(**common)
        max_n = int(target_n * 2)
        sample_sizes = np.linspace(max(20, max_n // n_points), max_n, n_points, dtype=int)

    powers = []
    for n in sample_sizes:
        kwargs: dict[str, Any] = {
            "group_sizes": [int(n), int(n)],
            "alphas": alphas,
            "betas": betas,
            "baseline": baseline,
            "alt_lift": alt_lift,
            "alt_rate": alt_rate,
            "lift": lift,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
            "spend": spend,
            "msrp": msrp,
        }
        if decision == "lift":
            kwargs["confidence_level"] = confidence_level
        else:
            kwargs["loss_threshold"] = loss_threshold
        powers.append(power_fn(**kwargs))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=powers,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name="Power",
        )
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="80% power",
        annotation_position="top left",
    )

    rule = f"P(B>A) ≥ {confidence_level}" if decision == "lift" else f"E[loss] ≤ {loss_threshold}"
    fig.update_layout(
        title=f"Bayesian Power Curve ({rule})",
        xaxis_title="Per-group sample size",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def plot_bayes_sensitivity_curve(
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    lift: str = "relative",
    decision: Literal["lift", "loss"] = "lift",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    n_points: int = 50,
    spend: float | None = None,
    msrp: float | None = None,
) -> go.Figure:
    """Plot minimum detectable lift as a function of per-group sample size.

    Parameters
    ----------
    alphas : np.ndarray or list
        Alpha parameters of the Beta prior for each variant.
    betas : np.ndarray or list
        Beta parameters of the Beta prior for each variant.
    baseline : float
        Expected conversion rate of the control variant.
    lift : {"relative", "absolute"}, optional
        How the lift is applied to ``baseline``. Default is ``"relative"``.
    decision : {"lift", "loss"}, optional
        Decision rule to use: ``"lift"`` uses P(B > A) >= ``confidence_level``,
        ``"loss"`` uses E[max(A-B, 0)] <= ``loss_threshold``. Default is
        ``"lift"``.
    target_power : float, optional
        Minimum acceptable Bayesian power. Default is 0.80.
    confidence_level : float, optional
        Posterior probability threshold when ``decision="lift"``. Default is 0.95.
    loss_threshold : float, optional
        Maximum acceptable expected loss when ``decision="loss"``. Default is
        0.001.
    n_samples : int, optional
        Number of simulated experiments per power evaluation. Default is 10_000.
    mc_samples : int, optional
        Number of posterior draws per simulated experiment. Default is 500.
    sample_sizes : array_like or None, optional
        Explicit per-group sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 50.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with per-group sample size on the x-axis
        and minimum detectable lift on the y-axis.
    """
    mdl_fn = bayes_minimum_detectable_lift if decision == "lift" else bayes_minimum_detectable_lift_loss

    if sample_sizes is None:
        if lift in _SCALED_LIFTS:
            raise ValueError(
                f"lift={lift!r} requires explicit sample_sizes because the "
                f"automatic range uses minimum sample size search, which does "
                f"not support scaled lift types."
            )
        search_fn = bayes_minimum_sample_size if decision == "lift" else bayes_minimum_sample_size_loss
        common: dict[str, Any] = {
            "alphas": alphas,
            "betas": betas,
            "baseline": baseline,
            "alt_lift": 0.05,
            "lift": lift,
            "target_power": target_power,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
        }
        if decision == "lift":
            common["confidence_level"] = confidence_level
        else:
            common["loss_threshold"] = loss_threshold
        target_n = search_fn(**common)
        min_n = max(100, target_n // 10)
        max_n = target_n * 5
        sample_sizes = np.linspace(min_n, max_n, n_points, dtype=int)

    mdls = []
    for n in sample_sizes:
        kwargs: dict[str, Any] = {
            "group_size": int(n),
            "alphas": alphas,
            "betas": betas,
            "baseline": baseline,
            "lift": lift,
            "target_power": target_power,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
            "spend": spend,
            "msrp": msrp,
        }
        if decision == "lift":
            kwargs["confidence_level"] = confidence_level
        else:
            kwargs["loss_threshold"] = loss_threshold
        mdls.append(mdl_fn(**kwargs))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=mdls,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name="MDL",
        )
    )

    _lift_labels = {
        "relative": "Minimum detectable relative lift",
        "absolute": "Minimum detectable absolute lift",
        "incremental": "Minimum detectable incremental lift",
        "roas": "Minimum detectable ROAS",
        "revenue": "Minimum detectable revenue",
        "cpa": "Minimum detectable CPA",
    }
    y_label = _lift_labels.get(lift, f"Minimum detectable {lift} lift")
    y_format = ",.0%" if lift in ("relative", "absolute") else ",."
    rule = f"P(B>A) ≥ {confidence_level}" if decision == "lift" else f"E[loss] ≤ {loss_threshold}"
    fig.update_layout(
        title=f"Bayesian Sensitivity Curve ({rule})",
        xaxis_title="Per-group sample size",
        yaxis_title=y_label,
        yaxis_tickformat=y_format,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
