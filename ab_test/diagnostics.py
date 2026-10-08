"""Pre-analysis diagnostics for A/B tests.

Run these checks before trusting experiment results. A significant
sample ratio mismatch, for example, can indicate a data pipeline bug,
a broken randomisation layer, or bot traffic that invalidates the
entire analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import scipy.stats as ss

from ab_test._display import apply_dark_mode, format_percent
from ab_test.frequentist_binomial.confidence_intervals import confidence_interval
from ab_test.frequentist_binomial.stats_tests import ab_test
from ab_test.frequentist_binomial.utils import observed_lift

__all__ = [
    "srm_test",
    "time_trend_test",
    "placebo_test",
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


def _period_lift_and_se(
    successes_a: np.ndarray[Any, Any],
    trials_a: np.ndarray[Any, Any],
    successes_b: np.ndarray[Any, Any],
    trials_b: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute per-period absolute lift and its standard error."""
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    lift = p_b - p_a
    se = np.sqrt(p_a * (1 - p_a) / trials_a + p_b * (1 - p_b) / trials_b)
    return lift, se


def time_trend_test(
    successes_a: np.ndarray[Any, Any] | list[int],
    trials_a: np.ndarray[Any, Any] | list[int],
    successes_b: np.ndarray[Any, Any] | list[int],
    trials_b: np.ndarray[Any, Any] | list[int],
    alpha: float = 0.05,
    labels: list[str] | np.ndarray[Any, Any] | None = None,
    *,
    dark_mode: bool = False,
) -> dict[str, Any]:
    """Test whether the treatment effect is stable over time.

    Computes the absolute lift (difference in proportions) for each
    time period and fits a weighted least-squares regression of lift on
    time index.  A significant slope indicates the effect is trending —
    decaying (novelty effect) or growing (primacy effect).

    Parameters
    ----------
    successes_a : array_like
        Per-period successes for variant A (control), length *T*.
    trials_a : array_like
        Per-period trials for variant A, length *T*.
    successes_b : array_like
        Per-period successes for variant B (treatment), length *T*.
    trials_b : array_like
        Per-period trials for variant B, length *T*.
    alpha : float, optional
        Significance level for the trend test, and the level of the
        per-period error bars (100(1 - alpha)% intervals).  Default is 0.05.
    labels : array_like or None, optional
        Display labels for each period (e.g. dates).  When ``None``,
        periods are numbered ``1, 2, …, T``.
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"slope"`` : float — WLS slope (change in absolute lift per
          period).
        - ``"slope_se"`` : float — standard error of the slope.
        - ``"t_stat"`` : float — t-statistic for the slope.
        - ``"p_value"`` : float — two-sided p-value (H₀: slope = 0).
        - ``"trending"`` : bool — ``True`` when p_value < alpha.
        - ``"diagnosis"`` : str — one of ``"stable"``, ``"novelty"``
          (decaying), or ``"primacy"`` (growing).
        - ``"period_lifts"`` : np.ndarray — per-period absolute lifts.
        - ``"period_se"`` : np.ndarray — per-period standard errors.
        - ``"cumulative_lift"`` : np.ndarray — running cumulative lift
          up to each period.
        - ``"figure"`` : plotly.graph_objects.Figure — diagnostic plot.

    Raises
    ------
    ValueError
        If the arrays have mismatched lengths or fewer than 3 periods.
    """
    s_a = np.asarray(successes_a, dtype=float)
    t_a = np.asarray(trials_a, dtype=float)
    s_b = np.asarray(successes_b, dtype=float)
    t_b = np.asarray(trials_b, dtype=float)

    if not (len(s_a) == len(t_a) == len(s_b) == len(t_b)):
        raise ValueError("All input arrays must have the same length")
    n_periods = len(s_a)
    if n_periods < 3:
        raise ValueError(f"Need at least 3 time periods for trend detection, got {n_periods}")

    period_lift, period_se = _period_lift_and_se(s_a, t_a, s_b, t_b)

    cum_s_a = np.cumsum(s_a)
    cum_t_a = np.cumsum(t_a)
    cum_s_b = np.cumsum(s_b)
    cum_t_b = np.cumsum(t_b)
    cumulative_lift = cum_s_b / cum_t_b - cum_s_a / cum_t_a

    x = np.arange(n_periods, dtype=float)
    weights = 1.0 / np.maximum(period_se**2, 1e-30)
    w_sum = weights.sum()
    x_bar = np.sum(weights * x) / w_sum
    y_bar = np.sum(weights * period_lift) / w_sum
    slope = float(np.sum(weights * (x - x_bar) * (period_lift - y_bar)) / np.sum(weights * (x - x_bar) ** 2))
    residuals = period_lift - (y_bar + slope * (x - x_bar))
    df = n_periods - 2
    mse = float(np.sum(weights * residuals**2) / df)
    slope_se = float(np.sqrt(mse / np.sum(weights * (x - x_bar) ** 2)))
    t_stat = slope / slope_se if slope_se > 0 else 0.0
    p_value = float(2 * ss.t.sf(abs(t_stat), df=df))

    trending = p_value < alpha
    if not trending:
        diagnosis = "stable"
    elif slope < 0:
        diagnosis = "novelty"
    else:
        diagnosis = "primacy"

    if labels is None:
        display_labels = [str(i + 1) for i in range(n_periods)]
    else:
        display_labels = [str(lbl) for lbl in labels]

    trend_line = y_bar + slope * (x - x_bar)

    fig = go.Figure()  # type: ignore[attr-defined]

    fig.add_trace(
        go.Scatter(  # type: ignore[attr-defined]
            x=display_labels,
            y=period_lift,
            error_y=dict(type="data", array=float(ss.norm.isf(alpha / 2)) * period_se, visible=True),
            mode="markers",
            name=f"Period lift ({format_percent(1 - alpha)}% CI)",
            marker=dict(size=8, color="#636EFA"),
        )
    )

    fig.add_trace(
        go.Scatter(  # type: ignore[attr-defined]
            x=display_labels,
            y=cumulative_lift,
            mode="lines+markers",
            name="Cumulative lift",
            line=dict(color="#00CC96", width=2),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(  # type: ignore[attr-defined]
            x=display_labels,
            y=trend_line,
            mode="lines",
            name=f"Trend (p={p_value:.3f})",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    title = f"Time Trend Diagnostic — {diagnosis.capitalize()}"
    if trending:
        title += f" (slope={slope:.5f}, p={p_value:.4f})"

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Absolute Lift (B − A)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    apply_dark_mode(fig, dark_mode)
    return {
        "slope": slope,
        "slope_se": slope_se,
        "t_stat": t_stat,
        "p_value": p_value,
        "trending": trending,
        "diagnosis": diagnosis,
        "period_lifts": period_lift,
        "period_se": period_se,
        "cumulative_lift": cumulative_lift,
        "figure": fig,
    }


def placebo_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    alpha: float = 0.05,
    lift: str = "absolute",
    test_method: str = "score",
) -> dict[str, Any]:
    """Check for a treatment effect where none should exist.

    Compares the two groups on data the treatment cannot have affected —
    either the pre-experiment period for the same users, or a placebo
    outcome measured before exposure. The true effect is zero by
    construction, so a significant result points to pre-existing imbalance,
    a broken randomiser, or a data pipeline bug rather than a real effect.

    Parameters
    ----------
    successes_a : int
        Placebo successes for variant A (control).
    trials_a : int
        Placebo trials for variant A.
    successes_b : int
        Placebo successes for variant B (treatment).
    trials_b : int
        Placebo trials for variant B.
    alpha : float, optional
        Significance level for the placebo check.  Default is 0.05.
    lift : {"absolute", "relative"}, optional
        Scale of the reported lift and confidence interval.  Default is
        ``"absolute"``, which stays defined when the control group has no
        placebo successes.
    test_method : str, optional
        Significance test used for the p-value; any ``method`` accepted by
        :func:`~ab_test.frequentist_binomial.stats_tests.ab_test`.  Default
        is ``"score"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"lift"`` : float — observed placebo lift (B vs. A).
        - ``"ci_lower"`` : float — lower bound of the 100(1 − alpha)%
          confidence interval, from inverting the score test.
        - ``"ci_upper"`` : float — upper bound of that interval.
        - ``"p_value"`` : float — p-value for H₀: no placebo effect.
        - ``"failed"`` : bool — ``True`` when p_value < alpha, i.e. the
          groups differ where they should not.

    Raises
    ------
    ValueError
        If any count is negative, a group has no trials, successes exceed
        trials, or ``lift="relative"`` with no control successes.

    Examples
    --------
    >>> result = placebo_test(successes_a=480, trials_a=10000, successes_b=495, trials_b=10000)
    >>> result["failed"]
    False
    """
    trials = [trials_a, trials_b]
    successes = [successes_a, successes_b]
    if min(successes) < 0 or min(trials) <= 0:
        raise ValueError("Trials must be positive and successes non-negative")
    if successes_a > trials_a or successes_b > trials_b:
        raise ValueError("Successes cannot exceed trials")
    lift = lift.casefold()
    if lift == "relative" and successes_a == 0:
        raise ValueError('Relative lift is undefined with no control successes; use lift="absolute"')

    p_value = float(ab_test(trials, successes, null_lift=0.0, lift=lift, method=test_method))
    ci_lower, ci_upper = confidence_interval(trials, successes, alpha=alpha, lift=lift)

    return {
        "lift": observed_lift(trials, successes, lift=lift),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": p_value,
        "failed": p_value < alpha,
    }
