"""Exact sequential e-test for comparing two Bernoulli proportions.

Implements the e-process of Turner, Ly & Grünwald (2024) for testing that two
groups share the same success rate. Unlike the mSPRT, it needs no normal
approximation and no mixing scale: it is exact at every sample size and stays
valid under optional stopping and continuous monitoring.

The data are processed as a sequence of looks. At each look, the new
observations since the previous look form a block, and the block's e-value
compares a plug-in alternative (each group's posterior-mean rate from the
earlier blocks) against the null rate closest to it in the reverse
information projection sense (the block-size-weighted mean of the plug-in
rates). The e-process is the running product of the block e-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import xlogy

__all__ = [
    "bernoulli_e_process",
    "bernoulli_e_test",
]


def _validate_looks(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return cumulative trials and successes as ``(n_looks, 2)`` integer arrays."""
    trials_arr = np.atleast_2d(np.asarray(trials))
    successes_arr = np.atleast_2d(np.asarray(successes))
    if trials_arr.shape != successes_arr.shape or trials_arr.ndim != 2 or trials_arr.shape[1] != 2:
        raise ValueError(
            "trials and successes must have the same shape: (n_looks, 2) cumulative counts, or (2,) for one look"
        )
    if not (np.all(np.mod(trials_arr, 1) == 0) and np.all(np.mod(successes_arr, 1) == 0)):
        raise ValueError("trials and successes must be whole numbers")
    trials_arr = trials_arr.astype(np.int64)
    successes_arr = successes_arr.astype(np.int64)
    if np.any(successes_arr < 0) or np.any(successes_arr > trials_arr):
        raise ValueError("successes must be between 0 and trials")
    new_trials = np.diff(trials_arr, axis=0, prepend=0)
    new_successes = np.diff(successes_arr, axis=0, prepend=0)
    if np.any(new_trials < 0) or np.any(new_successes < 0) or np.any(new_successes > new_trials):
        raise ValueError("counts must be cumulative: trials and successes cannot decrease between looks")
    return trials_arr, successes_arr


def bernoulli_e_process(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    *,
    prior: float = 0.5,
) -> np.ndarray[Any, Any]:
    """E-process for equal success rates in two groups, one value per look.

    Parameters
    ----------
    trials : array_like, shape (n_looks, 2)
        Cumulative number of trials in each group at each look. A single look
        may be given as shape ``(2,)``.
    successes : array_like, shape (n_looks, 2)
        Cumulative number of successes in each group at each look.
    prior : float, default=0.5
        Parameter ``a`` of the ``Beta(a, a)`` prior behind the plug-in rates.
        The default is the Jeffreys prior; power is insensitive to it.

    Returns
    -------
    np.ndarray
        The e-process at each look: the product of the block e-values so far.
        Under the null, the probability that it ever reaches ``1 / alpha`` is
        at most ``alpha``.

    Raises
    ------
    ValueError
        If the counts are not cumulative, successes exceed trials, or
        ``prior`` is not positive.

    Notes
    -----
    For a block of ``m_g`` new trials with ``s_g`` successes in group ``g``,
    and plug-in rates ``theta_g`` from the earlier blocks, the block e-value is

    ``prod_g (theta_g / theta_0)^s_g ((1 - theta_g) / (1 - theta_0))^(m_g - s_g)``

    with ``theta_0 = sum_g m_g theta_g / sum_g m_g``. For any common rate
    ``theta``, its expectation is ``prod_g f_g(theta)^m_g`` with ``f_g`` linear
    in ``theta``; the log of this is concave in ``theta``, and ``theta_0``
    makes its derivative zero at ``theta = theta_0``, where it equals 1. So the
    expectation is at most 1 under every null rate.

    That argument needs each block's sizes to be fixed before its outcomes
    are seen, which holds when the allocation and the look schedule do not
    depend on the outcomes. The first look has no earlier data, so its
    e-value is 1: the test learns only from the second look on, and more
    frequent looks give more power.

    References
    ----------
    Turner, R. J., Ly, A., & Grünwald, P. D. (2024). Generic E-variables for
    exact sequential k-sample tests that allow for optional stopping.
    *Statistics & Probability Letters*, 207, 110003.

    Examples
    --------
    >>> trials = [[1000, 1000], [2000, 2000], [3000, 3000]]
    >>> successes = [[100, 130], [205, 262], [300, 395]]
    >>> [round(float(e), 2) for e in bernoulli_e_process(trials, successes)]
    [1.0, 5.56, 155.26]
    """
    if prior <= 0:
        raise ValueError(f"prior must be positive, got {prior}")
    trials_arr, successes_arr = _validate_looks(trials, successes)

    # Each look's block, and the cumulative counts before it.
    new_trials = np.diff(trials_arr, axis=0, prepend=0)
    new_successes = np.diff(successes_arr, axis=0, prepend=0)
    prev_trials = trials_arr - new_trials
    prev_successes = successes_arr - new_successes

    theta = (prev_successes + prior) / (prev_trials + 2 * prior)
    block_size = new_trials.sum(axis=1, keepdims=True)
    theta_0 = np.divide(
        (new_trials * theta).sum(axis=1, keepdims=True),
        block_size,
        out=np.full_like(block_size, 0.5, dtype=float),
        where=block_size > 0,
    )
    new_failures = new_trials - new_successes
    log_e = (
        xlogy(new_successes, theta)
        + xlogy(new_failures, 1 - theta)
        - xlogy(new_successes, theta_0)
        - xlogy(new_failures, 1 - theta_0)
    ).sum(axis=1)
    # Cap as msprt_test does: beyond e^700 the float overflows, and the decision is long settled.
    return np.exp(np.minimum(np.cumsum(log_e), 700.0))


def bernoulli_e_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    alpha: float = 0.05,
    *,
    prior: float = 0.5,
) -> dict[str, Any]:
    """Exact sequential test that two groups share the same success rate.

    Parameters
    ----------
    trials : array_like, shape (n_looks, 2)
        Cumulative number of trials in each group at each look, in time
        order. A single look may be given as shape ``(2,)``.
    successes : array_like, shape (n_looks, 2)
        Cumulative number of successes in each group at each look.
    alpha : float, default=0.05
        Significance level. The null is rejected once the e-process reaches
        ``1 / alpha``.
    prior : float, default=0.5
        Parameter of the ``Beta(prior, prior)`` prior behind the plug-in
        rates (see :func:`bernoulli_e_process`).

    Returns
    -------
    dict
        - ``"e_value"`` : float — the e-process at the last look.
        - ``"max_e_value"`` : float — its largest value so far.
        - ``"p_value"`` : float — anytime-valid p-value,
          ``min(1, 1 / max_e_value)``. It never increases from look to look.
        - ``"rejected"`` : bool — ``True`` once ``max_e_value >= 1 / alpha``.
        - ``"rejected_at"`` : int or None — index of the first look at which
          the null was rejected.
        - ``"e_values"`` : np.ndarray — the e-process at each look.
        - ``"p_values"`` : np.ndarray — the anytime-valid p-value at each look.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1), or the counts are invalid (see
        :func:`bernoulli_e_process`).

    Notes
    -----
    The test is exact: it needs no normal approximation, and the type-I error
    is at most ``alpha`` however often you look and whenever you stop. It
    tests only the null of equal rates, so it does not support a nonzero
    ``null_lift`` or give a confidence interval; use
    :func:`~ab_test.frequentist_binomial.msprt.msprt_test` for those.

    Examples
    --------
    >>> trials = [[1000, 1000], [2000, 2000], [3000, 3000]]
    >>> successes = [[100, 130], [205, 262], [300, 395]]
    >>> result = bernoulli_e_test(trials, successes, alpha=0.05)
    >>> result["rejected"], result["rejected_at"], round(result["p_value"], 4)
    (True, 2, 0.0064)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    e_values = bernoulli_e_process(trials, successes, prior=prior)
    running_max = np.maximum.accumulate(e_values)
    p_values = np.minimum(1.0, 1.0 / running_max)
    crossed = np.flatnonzero(running_max >= 1 / alpha)
    return {
        "e_value": float(e_values[-1]),
        "max_e_value": float(running_max[-1]),
        "p_value": float(p_values[-1]),
        "rejected": bool(crossed.size),
        "rejected_at": int(crossed[0]) if crossed.size else None,
        "e_values": e_values,
        "p_values": p_values,
    }
