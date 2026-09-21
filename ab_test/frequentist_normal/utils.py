import math
from typing import Any

import numpy as np

__all__ = [
    "validate_two_group",
]


def validate_two_group(
    means: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
    variances: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    allow_relative_null: bool = True,
) -> None:
    """Validate the inputs shared by every significance test.

    Parameters
    ----------
    means, trials, variances : array_like
        Per-group means, trial counts and variances. At most two groups are supported.
    null_lift : float
        Lift associated with the null hypothesis.
    lift : str
        Whether ``null_lift`` is interpreted in relative or absolute terms.
    allow_relative_null : bool
        If False, a nonzero relative ``null_lift`` is rejected (only tests that
        support a nonzero relative null pass True).

    Raises
    ------
    NotImplementedError
        If more than two groups are supplied, or a nonzero relative ``null_lift``
        is given when ``allow_relative_null`` is False.
    """
    if len(trials) > 2 or len(means) > 2 or len(variances) > 2:
        raise NotImplementedError("Only supports a 2x2 continuous table")
    if not allow_relative_null and lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Only supports relative lift with a null of 0%")
