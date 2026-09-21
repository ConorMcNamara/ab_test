from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import scipy.stats as ss

from ab_test.frequentist_normal.utils import validate_two_group

__all__ = [
    "ab_test",
    "score_test",
    "likelihood_ratio_test",
    "wald_test"
    "welch_test"
]


def welch_test(
    means: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
) -> float | bool:
    """Welch's t-test for 2 experiment groups.

        Parameters
        ----------
         means: array_like
            The average for each group
         variances: array_like
            The variacnes for each group.
         trials : array_like
            Number of trials in each group.
         null_lift : float
            Lift associated with null hypothesis. Defaults to 0.0.
         lift : ["relative", "absolute"]
            Whether to interpret the null lift relative to the baseline success
            rate, or in absolute terms. See Notes in
            `maximum_likelihood_estimation`.
         crit : float, optional
            Critical value for the test statistic. If omitted, a p-value will be
            returned. If passed, a boolean will be returned corresponding to
            whether the result is statistically significant. Useful primarily for
            simulations where we will be repeatedly assessing significance, since
            calculating the critical value can be done once instead of repeatedly.
            This makes such simulations about 5x faster.

        Returns
        -------
         pval : float
            P-value. Returned if `crit` is None.
         stat_sig : boolean
            True if the result is statistically significant, i.e. if the test
            statistic is >= `crit`. Returned if `crit` is not None.

        Notes
        -----
        Only supports two experiment groups at this time.
        """
    validate_two_group(trials, trials, variances, null_lift, lift)
    mean1, mean2 = means[0], means[1]
    var1, var2 = variances[0], variances[1]
    trial1, trial2 = trials[0], trials[1]
    df = np.power((var1 / trial1) + (var2 / trial2), 2) / (
            (np.power(var1, 2) / (np.power(trial1, 2) * (trial1 - 1)))
            + (np.power(var2, 2) / (np.power(trial2, 2) * (trial2 - 1)))
        )
    standard_error_difference = math.sqrt((var1 / trial1) + (var2 / trial2))
    t_value = (mean1 - mean2) / standard_error_difference
    if crit is None:
        pval = 2 * (1.0 - ss.t.cdf(abs(t_value), df))  # type: ignore[no-untyped-call]
        return float(pval)
    return t_value >= crit