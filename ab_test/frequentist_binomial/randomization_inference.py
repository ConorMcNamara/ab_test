"""Randomization inference for A/B tests.

Provides assumption-free p-values by permuting treatment assignments and
comparing the observed test statistic to the permutation distribution.

Two test functions are provided:

* :func:`randomization_test` — individual-level permutation test for 2×2
  contingency tables.  Uses the hypergeometric distribution to generate
  the permutation distribution vectorized.
* :func:`cluster_randomization_test` — cluster-level permutation test
  that permutes cluster assignments rather than individual units.
  Supports both exact enumeration and Monte Carlo approximation.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ab_test.frequentist_binomial.utils import validate_two_group

__all__ = [
    "randomization_test",
    "cluster_randomization_test",
]

_MAX_EXACT_COMBINATIONS = 1_000_000


def randomization_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
    *,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> float | bool:
    """Randomization inference test for a 2×2 contingency table.

    Under the sharp null hypothesis of no treatment effect, the total
    number of successes is fixed and group assignment is the only source
    of randomness.  The permutation distribution is the hypergeometric,
    so all permutations are drawn in a single vectorized call.

    Parameters
    ----------
    trials : array_like
        Number of trials in each group (length 2).
    successes : array_like
        Number of successes in each group (length 2).
    null_lift : float
        Must be 0.  Non-zero sharp nulls are not supported with
        aggregate data.
    lift : str
        Lift type (``"relative"`` or ``"absolute"``).  Only affects
        validation; the test statistic is always the difference in
        proportions.
    crit : float or None
        When ``None``, the p-value is returned.  When set, it is
        treated as an alpha threshold and a boolean is returned.
    n_permutations : int
        Number of Monte Carlo permutations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    float or bool
        P-value when ``crit`` is None, otherwise whether the result is
        significant at the ``crit`` level.
    """
    validate_two_group(trials, successes, null_lift, lift, allow_relative_null=False)
    if null_lift != 0.0:
        raise ValueError(
            "Randomization inference only supports null_lift=0 with aggregate data. "
            "A non-zero sharp null cannot be evaluated without individual-level outcomes."
        )

    n = np.asarray(trials, dtype=int)
    s = np.asarray(successes, dtype=int)
    total_n = int(n.sum())
    total_s = int(s.sum())

    observed = s[1] / n[1] - s[0] / n[0]

    rng = np.random.default_rng(seed)
    perm_s0 = rng.hypergeometric(total_s, total_n - total_s, n[0], size=n_permutations)
    perm_s1 = total_s - perm_s0
    perm_diffs = perm_s1 / n[1] - perm_s0 / n[0]

    p_value = float((np.sum(np.abs(perm_diffs) >= np.abs(observed)) + 1) / (n_permutations + 1))

    if crit is None:
        return p_value
    return bool(p_value <= crit)


def _cluster_perm_chunk(
    all_props: np.ndarray[Any, Any],
    k_ctrl: int,
    abs_obs: float,
    chunk_size: int,
    seed: int,
) -> int:
    """Run a chunk of Monte Carlo cluster permutations and return the extreme count."""
    k_total = len(all_props)
    rng = np.random.default_rng(seed)
    random_keys = rng.random((chunk_size, k_total))
    indices = np.argsort(random_keys, axis=1)
    ctrl_means = all_props[indices[:, :k_ctrl]].mean(axis=1)
    treat_means = all_props[indices[:, k_ctrl:]].mean(axis=1)
    perm_diffs = treat_means - ctrl_means
    return int(np.sum(np.abs(perm_diffs) >= abs_obs - 1e-12))


def cluster_randomization_test(
    successes_ctrl: np.ndarray[Any, Any] | list[int],
    trials_ctrl: np.ndarray[Any, Any] | list[int],
    successes_treat: np.ndarray[Any, Any] | list[int],
    trials_treat: np.ndarray[Any, Any] | list[int],
    *,
    n_permutations: int = 10_000,
    seed: int | None = None,
    exact: bool = False,
    n_jobs: int = 1,
) -> float:
    """Randomization inference test for cluster-randomized trials.

    Under the sharp null, cluster-level outcomes are fixed regardless of
    assignment.  The test permutes which clusters belong to which arm and
    recomputes the difference in (unweighted) mean cluster proportions.

    Parameters
    ----------
    successes_ctrl, trials_ctrl : array_like
        Per-cluster successes and trials for the control arm.
    successes_treat, trials_treat : array_like
        Per-cluster successes and trials for the treatment arm.
    n_permutations : int
        Number of Monte Carlo permutations (ignored when ``exact=True``).
    seed : int or None
        Random seed for reproducibility (ignored when ``exact=True``).
    exact : bool
        If True, enumerate all possible cluster assignments.  Raises
        ``ValueError`` when the number of combinations exceeds
        1,000,000.
    n_jobs : int
        Number of parallel jobs for Monte Carlo permutations.  ``1``
        (default) runs sequentially; ``-1`` uses all available cores.
        Ignored when ``exact=True``.

    Returns
    -------
    float
        Two-sided p-value.
    """
    s_c = np.asarray(successes_ctrl, dtype=float)
    m_c = np.asarray(trials_ctrl, dtype=float)
    s_t = np.asarray(successes_treat, dtype=float)
    m_t = np.asarray(trials_treat, dtype=float)

    k_ctrl = len(s_c)
    k_treat = len(s_t)
    if k_ctrl < 2:
        raise ValueError(f"Control arm has {k_ctrl} cluster(s); need at least 2")
    if k_treat < 2:
        raise ValueError(f"Treatment arm has {k_treat} cluster(s); need at least 2")

    all_props = np.concatenate([s_c / m_c, s_t / m_t])
    k_total = k_ctrl + k_treat

    observed = float(np.mean(all_props[k_ctrl:]) - np.mean(all_props[:k_ctrl]))

    if exact:
        n_combos = math.comb(k_total, k_ctrl)
        if n_combos > _MAX_EXACT_COMBINATIONS:
            raise ValueError(
                f"Exact enumeration requires {n_combos:,} combinations "
                f"(max {_MAX_EXACT_COMBINATIONS:,}). Use exact=False for Monte Carlo."
            )
        count = 0
        total = 0
        for ctrl_idx in itertools.combinations(range(k_total), k_ctrl):
            treat_idx = [i for i in range(k_total) if i not in ctrl_idx]
            diff = float(np.mean(all_props[list(treat_idx)]) - np.mean(all_props[list(ctrl_idx)]))
            if abs(diff) >= abs(observed) - 1e-12:
                count += 1
            total += 1
        return count / total

    abs_obs = abs(observed)
    rng = np.random.default_rng(seed)

    if n_jobs == 1:
        child_seed = int(rng.integers(2**31))
        extreme_count = _cluster_perm_chunk(all_props, k_ctrl, abs_obs, n_permutations, child_seed)
    else:
        chunk_sizes = np.diff(np.linspace(0, n_permutations, abs(n_jobs) + 1, dtype=int))
        child_seeds = rng.integers(2**31, size=len(chunk_sizes)).tolist()
        counts: list[int] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
            delayed(_cluster_perm_chunk)(all_props, k_ctrl, abs_obs, int(cs), s)
            for cs, s in zip(chunk_sizes, child_seeds)
        )
        extreme_count = sum(counts)

    return float((extreme_count + 1) / (n_permutations + 1))
