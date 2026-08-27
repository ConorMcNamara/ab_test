"""Pre-analysis diagnostics for A/B tests.

Run these checks before trusting experiment results. A significant
sample ratio mismatch, for example, can indicate a data pipeline bug,
a broken randomisation layer, or bot traffic that invalidates the
entire analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as ss

__all__ = [
    "srm_test",
]


def srm_test(
    observed: np.ndarray[Any, Any] | list[int],
    expected_proportions: np.ndarray[Any, Any] | list[float] | None = None,
) -> tuple[float, float]:
    """Chi-squared goodness-of-fit test for sample ratio mismatch.

    Checks whether the observed traffic split matches the intended
    split. A significant result (small p-value) indicates that the
    randomisation did not produce the expected allocation — typically a
    sign of a data pipeline bug, broken randomiser, or differential
    attrition.

    Parameters
    ----------
    observed : array_like
        Observed number of units (e.g. users) in each group.
    expected_proportions : array_like or None, default=None
        Intended fraction of traffic in each group. Must sum to 1.
        If ``None``, an equal split is assumed.

    Returns
    -------
    statistic : float
        Chi-squared goodness-of-fit statistic.
    pvalue : float
        P-value from a chi-squared(k - 1) distribution, where *k* is
        the number of groups.

    Raises
    ------
    ValueError
        If *expected_proportions* do not sum to 1 (within tolerance) or
        if their length does not match *observed*.
    """
    observed_arr = np.asarray(observed, dtype=float)
    k = len(observed_arr)
    total = observed_arr.sum()

    if expected_proportions is None:
        proportions = np.full(k, 1.0 / k)
    else:
        proportions = np.asarray(expected_proportions, dtype=float)
        if len(proportions) != k:
            raise ValueError(f"expected_proportions has {len(proportions)} elements but observed has {k}")
        if not np.isclose(proportions.sum(), 1.0):
            raise ValueError(f"expected_proportions must sum to 1, got {proportions.sum():.6f}")

    expected = total * proportions
    chi2 = float(np.sum((observed_arr - expected) ** 2 / expected))
    pvalue = float(ss.chi2.sf(chi2, df=k - 1))

    return chi2, pvalue
