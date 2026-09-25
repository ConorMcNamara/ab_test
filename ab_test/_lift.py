"""Shared lift conversion and computation helpers.

Every module that converts between lift types (absolute, relative,
incremental, roas, revenue, cpa) should call through here rather than
reimplementing the branching logic.
"""

from typing import Any, overload

import numpy as np

__all__ = [
    "to_absolute",
    "from_absolute",
    "scale_metric",
    "scale_bounds",
    "compute_sample_lift",
]

_SCALED_LIFTS = frozenset({"incremental", "roas", "revenue", "cpa"})


def to_absolute(
    lift_value: float,
    lift: str,
    scale: int,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Convert a lift value from the given lift type to absolute.

    Parameters
    ----------
    lift_value : float
        The lift in the units of ``lift``.
    lift : str
        One of ``"relative"``, ``"absolute"``, ``"incremental"``,
        ``"roas"``, ``"revenue"``, ``"cpa"``.
    scale : int
        Number of units used to scale incremental-family lifts (typically
        ``max(group_sizes)``).
    spend : float, optional
        Total ad spend — required for ``"roas"`` and ``"cpa"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

    Returns
    -------
    float
        The equivalent absolute lift (difference in proportions).
    """
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


def from_absolute(
    abs_value: float,
    lift: str,
    scale: int,
    spend: float | None = None,
    msrp: float | None = None,
) -> float:
    """Convert an absolute lift value to the given lift type.

    Parameters
    ----------
    abs_value : float
        The lift as an absolute difference in proportions.
    lift : str
        Target lift type — see :func:`to_absolute`.
    scale : int
        Number of units used to scale incremental-family lifts.
    spend : float, optional
        Total ad spend — required for ``"roas"`` and ``"cpa"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

    Returns
    -------
    float
        The lift in the units of ``lift``.
    """
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


@overload
def scale_metric(
    value: float | int,
    lift: str,
    spend: float | None = ...,
    msrp: float | None = ...,
) -> float: ...


@overload
def scale_metric(
    value: np.ndarray[Any, Any],
    lift: str,
    spend: float | None = ...,
    msrp: float | None = ...,
) -> np.ndarray[Any, Any]: ...


def scale_metric(
    value: float | int | np.ndarray[Any, Any],
    lift: str,
    spend: float | None = None,
    msrp: float | None = None,
) -> float | np.ndarray[Any, Any]:
    """Convert value(s) from incremental-count space to the target metric.

    For values already multiplied by ``max(trials)`` this applies the
    final metric transform: division by ``spend`` for ROAS, multiplication
    by ``msrp`` for revenue, or inversion for CPA.  Works on both scalars
    and numpy arrays.

    Parameters
    ----------
    value : float, int, or np.ndarray
        Value(s) in incremental-count space.
    lift : {"incremental", "roas", "revenue", "cpa"}
        Target metric.
    spend : float, optional
        Total ad spend — required for ``"roas"`` and ``"cpa"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

    Returns
    -------
    float or np.ndarray
        Value(s) on the target metric scale.
    """
    if lift == "incremental":
        return value
    if lift == "roas":
        if spend is None:
            raise ValueError("spend must be set for ROAS calculations")
        return value / spend
    if lift == "revenue":
        if msrp is None:
            raise ValueError("msrp must be set for revenue calculations")
        return value * msrp
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be set for CPA calculations")
        if isinstance(value, np.ndarray):
            return np.where(np.abs(value) > 1e-12, spend / value, np.inf)
        return spend / value if abs(value) > 1e-12 else np.inf
    raise ValueError(f"Unsupported lift type for scale_metric: {lift}")


def scale_bounds(
    lb: float,
    ub: float,
    lift: str,
    spend: float | None = None,
    msrp: float | None = None,
) -> tuple[float, float]:
    """Scale CI bounds from incremental-count space to the target metric.

    Handles CPA's ordering inversion: since CPA is ``spend / count``,
    a larger count yields a smaller CPA, so the lower/upper bounds swap.
    Non-positive bounds produce ``np.inf`` for CPA.

    Parameters
    ----------
    lb : float
        Lower confidence/credible interval bound in incremental-count space.
    ub : float
        Upper confidence/credible interval bound in incremental-count space.
    lift : {"incremental", "roas", "revenue", "cpa"}
        Target metric.
    spend : float, optional
        Total ad spend — required for ``"roas"`` and ``"cpa"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds on the target metric scale.
    """
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be set for CPA calculations")
        new_lb = spend / ub if ub > 0 else np.inf
        new_ub = spend / lb if lb > 0 else np.inf
        return float(new_lb), float(new_ub)
    scaled_lb = scale_metric(lb, lift, spend, msrp)
    scaled_ub = scale_metric(ub, lift, spend, msrp)
    return float(scaled_lb), float(scaled_ub)


def compute_sample_lift(
    samples_a: np.ndarray[Any, Any],
    samples_b: np.ndarray[Any, Any],
    lift: str = "absolute",
    trials: tuple[int, int] | None = None,
    spend: float | None = None,
    msrp: float | None = None,
) -> np.ndarray[Any, Any]:
    """Compute an array of lift values from paired posterior samples.

    Parameters
    ----------
    samples_a : np.ndarray
        Posterior (or simulated) samples for variant A.
    samples_b : np.ndarray
        Posterior (or simulated) samples for variant B.
    lift : str, optional
        Lift type — one of ``"relative"``, ``"absolute"``,
        ``"incremental"``, ``"roas"``, ``"revenue"``, ``"cpa"``.
        Default is ``"absolute"``.
    trials : tuple[int, int], optional
        ``(trials_a, trials_b)`` — required for incremental-family lifts.
    spend : float, optional
        Total ad spend — required for ``"roas"`` and ``"cpa"``.
    msrp : float, optional
        Average product price — required for ``"revenue"``.

    Returns
    -------
    np.ndarray
        Element-wise lift between ``samples_b`` and ``samples_a``.
    """
    if lift == "relative":
        return (samples_b - samples_a) / samples_a
    if lift == "absolute":
        return samples_b - samples_a

    if trials is None:
        raise ValueError(f"trials must be provided for lift='{lift}'")
    max_n = max(trials)
    diff = samples_b - samples_a

    if lift == "incremental":
        return diff * max_n
    if lift == "revenue":
        if msrp is None:
            raise ValueError("msrp must be provided for lift='revenue'")
        return diff * max_n * msrp
    if lift == "roas":
        if spend is None:
            raise ValueError("spend must be provided for lift='roas'")
        return diff * max_n / spend
    if lift == "cpa":
        if spend is None:
            raise ValueError("spend must be provided for lift='cpa'")
        incremental = diff * max_n
        return np.where(np.abs(incremental) > 1e-12, spend / incremental, np.inf)

    raise NotImplementedError(f"lift '{lift}' not implemented")
