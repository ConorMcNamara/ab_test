"""Confidence intervals by inverting a significance test with binary search.

Shared by the frequentist binomial and normal confidence interval modules,
which differ only in how they call their tests and in the limits of the
lift being searched.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

__all__ = ["binary_search_interval"]


def _search_lower_bound(
    pvalue: Callable[[float], float],
    lb_lb: float,
    lb_ub: float,
    alpha: float,
    limit: float,
    tol: float,
) -> float | None:
    eps = 0.01
    while True:
        if lb_lb < limit:
            return None
        if pvalue(lb_lb) >= alpha:
            lb_ub = lb_lb
            lb_lb -= eps
            eps *= 2
        else:
            break
    while (lb_ub - lb_lb) > tol:
        lb = 0.5 * (lb_lb + lb_ub)
        if pvalue(lb) >= alpha:
            lb_ub = lb
        else:
            lb_lb = lb
    return 0.5 * (lb_lb + lb_ub)


def _search_upper_bound(
    pvalue: Callable[[float], float],
    ub_lb: float,
    ub_ub: float,
    alpha: float,
    limit: float,
    tol: float,
) -> float | None:
    eps = 0.01
    while True:
        if ub_ub > limit:
            return None
        if pvalue(ub_ub) >= alpha:
            ub_lb = ub_ub
            ub_ub += eps
            eps *= 2
        else:
            break
    while (ub_ub - ub_lb) > tol:
        ub = 0.5 * (ub_lb + ub_ub)
        if pvalue(ub) >= alpha:
            ub_lb = ub
        else:
            ub_ub = ub
    return 0.5 * (ub_lb + ub_ub)


def binary_search_interval(
    pvalue: Callable[[float], float],
    lower_start: tuple[float, float],
    upper_start: tuple[float, float],
    alpha: float = 0.05,
    lower_limit: float = float("-inf"),
    upper_limit: float = float("inf"),
    tol: float = 1e-06,
    search_upper: bool = True,
) -> tuple[float | None, float | None]:
    """Find the lifts at which a test's p-value crosses ``alpha``.

    Each bound is found by stepping outward from its starting bracket with
    doubling step sizes until the test rejects, then bisecting to ``tol``.

    Parameters
    ----------
    pvalue : callable
        Maps a null lift to the test's p-value, e.g.
        ``lambda d: score_test(trials, successes, null_lift=d, lift=lift)``.
    lower_start, upper_start : tuple of float
        Initial ``(low, high)`` brackets for the lower and upper bounds,
        typically just below and just above the observed lift.
    alpha : float, default=0.05
        Threshold for significance. The interval has level 100(1-alpha)%.
    lower_limit, upper_limit : float
        Lifts beyond which the search stops, e.g. ``-1`` for relative lift.
    tol : float, default=1e-06
        Width at which bisection stops. Lower values mean narrower intervals.
    search_upper : bool, default=True
        If False, the upper bound is not searched and is returned as None.

    Returns
    -------
    lb, ub : float or None
        The interval bounds. A bound is None when its search passed its
        limit (or was skipped), so the caller can substitute its own value.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        lb_future = executor.submit(_search_lower_bound, pvalue, *lower_start, alpha, lower_limit, tol)
        ub_future = (
            executor.submit(_search_upper_bound, pvalue, *upper_start, alpha, upper_limit, tol)
            if search_upper
            else None
        )
        lb = lb_future.result()
        ub = ub_future.result() if ub_future is not None else None
    return lb, ub
