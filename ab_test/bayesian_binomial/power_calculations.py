"""Methods to calculate the power of a test."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import scipy.stats as ss
from joblib import Parallel, delayed

from ab_test._display import apply_dark_mode
from ab_test._lift import _SCALED_LIFTS, from_absolute, to_absolute

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


def _resolve_alt_rate(
    baseline: float,
    alt_lift: float | None,
    alt_rate: float | None,
    lift: str,
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


# Posterior draws per arm held in memory at once. Simulations run in blocks of
# about this many draws, so memory stays near 10 MB per arm whatever n_samples is.
_DRAWS_PER_BLOCK = 1_000_000

Seed = int | np.random.Generator | np.random.SeedSequence | None


def _entropy(seed: Seed) -> int:
    """Resolve ``seed`` to the integer every simulation block is derived from.

    ``None`` draws fresh entropy (unseeded), an int is used as is, and a
    ``Generator`` is advanced once to produce one.
    """
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1, np.uint64)[0])
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(2**63))
    if isinstance(seed, np.random.SeedSequence):
        return int(seed.generate_state(1, np.uint64)[0])
    return int(seed)


def _block_rng(entropy: int, *key: int) -> np.random.Generator:
    """Return the generator for one block, determined by ``entropy`` and the block's key alone."""
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=key))


def _simulate_block(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_rate: float,
    size: int,
    mc_samples: int,
    entropy: int,
    block: int,
    statistic: Literal["prob", "loss"],
) -> np.ndarray[Any, Any]:
    """Simulate one block of experiments and return each one's decision statistic.

    The experiment outcomes come from the binomial inverse CDF of uniforms drawn
    first, a fixed number per block, so they depend smoothly on the group sizes
    and rates, and the posterior draws that follow start from the same point of
    the stream. Every probe of a search therefore replays common random numbers.
    """
    rng = _block_rng(entropy, block)
    u = rng.random((2, size))
    n_null, n_alt = int(group_sizes[0]), int(group_sizes[1])
    successes_null = ss.binom.ppf(u[0], n_null, baseline)
    successes_alt = ss.binom.ppf(u[1], n_alt, alt_rate)
    samples_null = rng.beta(
        (alphas[0] + successes_null)[:, np.newaxis],
        (betas[0] + n_null - successes_null)[:, np.newaxis],
        (size, mc_samples),
    )
    samples_alt = rng.beta(
        (alphas[1] + successes_alt)[:, np.newaxis],
        (betas[1] + n_alt - successes_alt)[:, np.newaxis],
        (size, mc_samples),
    )
    if statistic == "prob":
        return np.mean(samples_alt > samples_null, axis=1)  # type: ignore[no-any-return]
    return np.mean(np.maximum(samples_null - samples_alt, 0), axis=1)  # type: ignore[no-any-return]


def _block_sizes(n_samples: int, mc_samples: int) -> list[int]:
    """Split ``n_samples`` simulations into blocks of at most ``_DRAWS_PER_BLOCK`` draws per arm."""
    block = max(1, _DRAWS_PER_BLOCK // mc_samples)
    sizes = [block] * (n_samples // block)
    if n_samples % block:
        sizes.append(n_samples % block)
    return sizes


def _simulated_statistics(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    alphas: np.ndarray[Any, Any] | list[Any],
    betas: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_rate: float,
    n_samples: int,
    mc_samples: int,
    statistic: Literal["prob", "loss"],
    seed: Seed = None,
    n_jobs: int = 1,
) -> np.ndarray[Any, Any]:
    """Decision statistic for each of ``n_samples`` simulated experiments.

    ``"prob"`` is P(B > A) and ``"loss"`` is E[max(A - B, 0)], each estimated
    from ``mc_samples`` posterior draws. Blocks are seeded from ``seed`` and
    their index only, so results do not depend on ``n_jobs``.
    """
    entropy = _entropy(seed)
    sizes = _block_sizes(n_samples, mc_samples)
    args = (group_sizes, alphas, betas, baseline, alt_rate)
    results: list[np.ndarray[Any, Any]]
    if n_jobs == 1:
        results = [_simulate_block(*args, size, mc_samples, entropy, i, statistic) for i, size in enumerate(sizes)]
    else:
        results = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
            delayed(_simulate_block)(*args, size, mc_samples, entropy, i, statistic) for i, size in enumerate(sizes)
        )
    return np.concatenate(results)


def _search_min_sample_size(
    power_fn: Callable[[int], float],
    target_power: float,
    max_n: int,
    error_message: str,
) -> int:
    """Find the smallest per-group sample size reaching ``target_power``.

    Checks 100 first (searching below it if that is already enough), otherwise
    doubles a candidate size, capped at ``max_n``, until ``power_fn`` meets
    ``target_power``, then binary-searches the resulting bracket. ``max_n``
    itself is always evaluated.

    Raises
    ------
    ValueError
        With ``error_message`` if ``target_power`` is not reached at ``max_n``.
    """
    low, high = 0, min(100, max_n)
    while power_fn(high) < target_power:
        if high >= max_n:
            raise ValueError(error_message)
        low, high = high, min(high * 2, max_n)

    while high - low > 1:
        mid = (low + high) // 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


# Largest treatment rate a lift search may imply: a rate of exactly 1 breaks the Beta
# parameterizations used in the simulations.
_MAX_RATE = 1 - 1e-9


def _max_feasible_lift(baseline: float, lift: str) -> float:
    """Largest lift whose implied treatment rate stays below 1."""
    if lift == "relative":
        return _MAX_RATE / baseline - 1
    return _MAX_RATE - baseline


def _search_min_lift(
    power_fn: Callable[[float], float],
    target_power: float,
    max_lift: float,
    tol: float,
    error_message: str,
) -> float:
    """Find the smallest lift reaching ``target_power``.

    Doubles a candidate lift from 0.01, capped at ``max_lift``, until
    ``power_fn`` meets ``target_power``, then binary-searches the resulting
    bracket to within ``tol``. ``max_lift`` itself is always evaluated.

    Raises
    ------
    ValueError
        With ``error_message`` if ``target_power`` is not reached at ``max_lift``.
    """
    low, high = 0.0, min(0.01, max_lift)
    while power_fn(high) < target_power:
        if high >= max_lift:
            raise ValueError(error_message)
        low, high = high, min(high * 2, max_lift)

    while high - low > tol:
        mid = (low + high) / 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def _unreachable_lift_message(target_power: float, max_lift: float, baseline: float, lift: str) -> str:
    """Error message for a lift search that cannot reach ``target_power``."""
    feasible = _max_feasible_lift(baseline, lift)
    if feasible < max_lift:
        bound = f"any lift that keeps the treatment rate below 1 (up to {feasible:.4g})"
    else:
        bound = f"a lift of {max_lift}"
    return (
        f"Could not reach target power of {target_power} within {bound}. "
        "Consider a smaller target power or larger group size."
    )


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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulation. ``None`` (default) is unseeded. The result
        does not depend on ``n_jobs``.
    n_jobs : int, optional
        Number of parallel jobs for the simulation. ``1`` (default) runs
        sequentially; ``-1`` uses all available cores.

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
        alt_lift = to_absolute(alt_lift, lift, scale, spend, msrp)
        lift = "absolute"
    alt_rate = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    prob_b_better = _simulated_statistics(
        group_sizes, alphas, betas, baseline, alt_rate, n_samples, mc_samples, "prob", seed=seed, n_jobs=n_jobs
    )
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulation. ``None`` (default) is unseeded. The result
        does not depend on ``n_jobs``.
    n_jobs : int, optional
        Number of parallel jobs for the simulation. ``1`` (default) runs
        sequentially; ``-1`` uses all available cores.

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
        alt_lift = to_absolute(alt_lift, lift, scale, spend, msrp)
        lift = "absolute"
    alt_rate = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    expected_loss = _simulated_statistics(
        group_sizes, alphas, betas, baseline, alt_rate, n_samples, mc_samples, "loss", seed=seed, n_jobs=n_jobs
    )
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
) -> int:
    """Find the minimum per-group sample size that achieves a target Bayesian power via expected loss.

    Uses a two-phase search: first doubles a candidate size from 100 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest n that suffices.

    A simulation counts as a "win" when E[max(A − B, 0)] <= ``loss_threshold``,
    meaning the downside risk of picking B is acceptably small.

    Power estimates are stochastic. Pass ``seed`` for a reproducible result,
    and increase ``n_samples`` for a more precise (but slower) one.

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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every power evaluation in the search reuses
        the same random numbers, so the search is reproducible for a given seed
        and estimated power changes smoothly with the search variable. ``None``
        (default) draws one fresh seed for the whole search.

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

    # One seed for the whole search: every probe replays the same random numbers.
    entropy = _entropy(seed)

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
            seed=entropy,
            n_jobs=n_jobs,
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
) -> int:
    """Find the minimum per-group sample size that achieves a target Bayesian power.

    Uses a two-phase search: first doubles a candidate size from 100 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest n that suffices.

    Power estimates are stochastic. Pass ``seed`` for a reproducible result,
    and increase ``n_samples`` for a more precise (but slower) one.

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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every power evaluation in the search reuses
        the same random numbers, so the search is reproducible for a given seed
        and estimated power changes smoothly with the search variable. ``None``
        (default) draws one fresh seed for the whole search.

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

    # One seed for the whole search: every probe replays the same random numbers.
    entropy = _entropy(seed)

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
            seed=entropy,
            n_jobs=n_jobs,
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
) -> float:
    """Find the minimum lift detectable at a target Bayesian power via P(B > A).

    Uses a two-phase search: first doubles a candidate lift from 0.01 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest lift that suffices.

    Power estimates are stochastic. Pass ``seed`` for a reproducible result,
    and increase ``n_samples`` for a more precise (but slower) one.

    The search never implies a treatment rate of 1 or more: it stops at the
    largest lift that keeps the rate below 1 if that is smaller than ``max_lift``.

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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every power evaluation in the search reuses
        the same random numbers, so the search is reproducible for a given seed
        and estimated power changes smoothly with the search variable. ``None``
        (default) draws one fresh seed for the whole search.

    Returns
    -------
    float
        Smallest lift estimated to reach ``target_power``, in the units
        specified by ``lift``. For ``"cpa"`` this is the largest detectable
        CPA: a smaller effect means fewer incremental conversions, so a
        higher cost per acquisition.

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

    # One seed for the whole search: every probe replays the same random numbers.
    entropy = _entropy(seed)

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
            seed=entropy,
            n_jobs=n_jobs,
        )

    abs_mdl = _search_min_lift(
        _power,
        target_power,
        min(max_lift, _max_feasible_lift(baseline, internal_lift)),
        tol,
        error_message=_unreachable_lift_message(target_power, max_lift, baseline, internal_lift),
    )
    if lift in _SCALED_LIFTS:
        return from_absolute(abs_mdl, lift, group_size, spend, msrp)
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
) -> float:
    """Find the minimum lift detectable at a target Bayesian power via expected loss.

    Uses a two-phase search: first doubles a candidate lift from 0.01 until the
    estimated power meets ``target_power``, then binary-searches within the
    resulting bracket to pinpoint the smallest lift that suffices.

    A simulation counts as a "win" when E[max(A − B, 0)] <= ``loss_threshold``.

    Power estimates are stochastic. Pass ``seed`` for a reproducible result,
    and increase ``n_samples`` for a more precise (but slower) one.

    The search never implies a treatment rate of 1 or more: it stops at the
    largest lift that keeps the rate below 1 if that is smaller than ``max_lift``.

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
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every power evaluation in the search reuses
        the same random numbers, so the search is reproducible for a given seed
        and estimated power changes smoothly with the search variable. ``None``
        (default) draws one fresh seed for the whole search.

    Returns
    -------
    float
        Smallest lift estimated to reach ``target_power``, in the units
        specified by ``lift``. For ``"cpa"`` this is the largest detectable
        CPA: a smaller effect means fewer incremental conversions, so a
        higher cost per acquisition.

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

    # One seed for the whole search: every probe replays the same random numbers.
    entropy = _entropy(seed)

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
            seed=entropy,
            n_jobs=n_jobs,
        )

    abs_mdl = _search_min_lift(
        _power,
        target_power,
        min(max_lift, _max_feasible_lift(baseline, internal_lift)),
        tol,
        error_message=_unreachable_lift_message(target_power, max_lift, baseline, internal_lift),
    )
    if lift in _SCALED_LIFTS:
        return from_absolute(abs_mdl, lift, group_size, spend, msrp)
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
    dark_mode: bool = False,
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
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every point on the curve reuses the same
        random numbers, so the curve is smooth and reproducible. ``None``
        (default) draws one fresh seed for the whole curve.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with per-group sample size on the x-axis
        and Bayesian power on the y-axis.
    """
    # One seed for the whole curve: every point reuses the same random numbers.
    entropy = _entropy(seed)
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
        target_n = search_fn(**common, seed=entropy, n_jobs=n_jobs)
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
            "seed": entropy,
            "n_jobs": n_jobs,
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
    apply_dark_mode(fig, dark_mode)
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
    *,
    seed: Seed = None,
    n_jobs: int = 1,
    dark_mode: bool = False,
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
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).
    seed : int, numpy.random.Generator or None, optional
        Seed for the simulations. Every point on the curve reuses the same
        random numbers, so the curve is smooth and reproducible. ``None``
        (default) draws one fresh seed for the whole curve.

    Returns
    -------
    go.Figure
        An interactive Plotly figure with per-group sample size on the x-axis
        and minimum detectable lift on the y-axis.
    """
    # One seed for the whole curve: every point reuses the same random numbers.
    entropy = _entropy(seed)
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
        target_n = search_fn(**common, seed=entropy, n_jobs=n_jobs)
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
            "seed": entropy,
            "n_jobs": n_jobs,
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
        # A smaller detectable effect means fewer incremental conversions, so a higher CPA.
        "cpa": "Maximum detectable CPA",
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
    apply_dark_mode(fig, dark_mode)
    return fig
