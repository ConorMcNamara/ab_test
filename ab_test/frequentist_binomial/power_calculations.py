"""Methods to calculate the power of a test."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import scipy.stats as ss

from ab_test._display import apply_dark_mode
from ab_test._lift import _SCALED_LIFTS, from_absolute, to_absolute
from ab_test.frequentist_binomial.utils import simple_hypothesis_from_composite

# Expansion steps when searching for a minimum detectable lift; each halves the gap to the
# feasible limit, so 50 steps reach within ~1e-15 of it.
_MAX_EXPANSIONS = 50

__all__ = [
    "score_power",
    "abtest_power",
    "minimum_detectable_lift",
    "required_sample_size",
    "plot_power_curve",
    "plot_sensitivity_curve",
]


def _score_contrast(
    n: np.ndarray[Any, Any] | list[Any],
    p_null: np.ndarray[Any, Any] | list[Any],
    p_alt: np.ndarray[Any, Any] | list[Any],
) -> tuple[float, float, float]:
    """Mean of the score test's contrast under H1, and its SD under H0 and under H1.

    The score test rejects when a linear contrast of the group rates is far
    from its null value. With ``p_null`` the restricted MLE, the contrast's
    direction is ``c_i = n_i (p_alt_i - p_null_i) / (p_null_i (1 - p_null_i))``
    (the Lagrange condition), which gives ``(-1, 1)`` for a null of no
    difference. It needs no lift type, and ``(mean / sd_null)^2`` is the
    chi-squared noncentrality ``sum_i n_i (p_alt_i - p_null_i)^2 / (p_null_i (1 - p_null_i))``.
    """
    n_arr = np.asarray(n, dtype=float)
    null = np.asarray(p_null, dtype=float)
    alt = np.asarray(p_alt, dtype=float)
    weights = n_arr * (alt - null) / (null * (1 - null))
    mean = float(np.sum(weights * (alt - null)))
    sd_null = math.sqrt(float(np.sum(weights**2 * null * (1 - null) / n_arr)))
    sd_alt = math.sqrt(float(np.sum(weights**2 * alt * (1 - alt) / n_arr)))
    return mean, sd_null, sd_alt


def score_power(
    n: np.ndarray[Any, Any] | list[Any],
    p_null: np.ndarray[Any, Any] | list[Any],
    p_alt: np.ndarray[Any, Any] | list[Any],
    alpha: float = 0.05,
) -> float:
    """Power of Rao's Score Test.

    Parameters
    ----------
     n : array_like
        Number of experimental units in each group.
     p_null : array_like
        Probability of success in each group under the null
        hypothesis.
     p_alt : array_like
        Probability of success in each group under the alternative
        hypothesis.
     alpha : float
        Type-I error rate. Defaults to 0.05

    Returns
    -------
     power : float
        The power of the test.

    Notes
    -----
    The score test standardises its contrast by the variance under the null,
    but under the alternative the contrast varies with the alternative's
    variance. The power is therefore

    ``Phi((|m| - z sd_0) / sd_1) + Phi((-|m| - z sd_0) / sd_1)``

    with ``m`` the contrast's mean under H1, ``sd_0`` and ``sd_1`` its standard
    deviations under H0 and H1, and ``z`` the two-sided critical value
    (Fleiss, Tytun & Ury, 1980; Farrington & Manning, 1990). With equal group
    sizes and a null of no difference the two variances nearly agree, but
    with unequal allocation the null-only approximation was off by up to
    0.06 in power.
    """
    mean, sd_null, sd_alt = _score_contrast(n, p_null, p_alt)
    if sd_null == 0.0:
        # p_alt equals p_null: the alternative is the null.
        return alpha
    z_crit = float(ss.norm.isf(alpha / 2))
    mean = abs(mean)
    return float(ss.norm.sf((z_crit * sd_null - mean) / sd_alt) + ss.norm.cdf((-z_crit * sd_null - mean) / sd_alt))


def _weakest_comparison(
    group_sizes: np.ndarray[Any, Any] | list[Any], alpha: float, comparisons: str
) -> tuple[list[Any], float]:
    """Group sizes of the least-powered comparison, and the per-comparison alpha.

    With three or more groups, ``analyze()`` compares each variant against the
    first group (``"control"``) or every pair (``"all"``) and adjusts for the
    ``m`` comparisons. Power is computed for the smallest comparison at the
    Bonferroni level ``alpha / m``; Holm (the ``analyze()`` default) rejects at
    least as often, so this power is a lower bound.
    """
    comparisons = comparisons.casefold()
    if comparisons not in ("control", "all"):
        raise ValueError(f"comparisons must be 'control' or 'all', got {comparisons!r}")
    k = len(group_sizes)
    if k <= 2:
        return list(group_sizes), alpha
    if comparisons == "control":
        return [group_sizes[0], min(group_sizes[1:])], alpha / (k - 1)
    a, b = sorted(group_sizes)[:2]
    return [a, b], alpha / (k * (k - 1) // 2)


def abtest_power(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alt_lift: float,
    alpha: float = 0.05,
    null_lift: float = 0.0,
    power: Callable[..., float] = score_power,
    lift: str = "relative",
    spend: float | None = None,
    msrp: float | None = None,
    *,
    comparisons: str = "control",
) -> float:
    """Power associated with an A/B Test.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Lift associated with alternative hypothesis.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How to interpret the null/alternative lift. Defaults to "relative".
     spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
     msrp : float, optional
        Revenue per unit. Required for "revenue" lift.
     comparisons : {"control", "all"}, optional
        With three or more groups, which comparisons ``analyze()`` will make:
        each variant against the first group (the default), or every pair.
        Ignored with two groups.

    Returns
    -------
     power : float
        The power of the test. With three or more groups, the power of the
        least-powered comparison (the control against the smallest variant,
        or the two smallest groups for ``"all"``) at the Bonferroni level
        ``alpha / m`` for ``m`` comparisons. Holm's correction rejects at
        least as often, so this is a lower bound on its power.
    """
    group_sizes, alpha = _weakest_comparison(group_sizes, alpha, comparisons)

    if lift in _SCALED_LIFTS:
        scale = max(group_sizes)
        internal_lift = "absolute"
        alt_lift = to_absolute(alt_lift, lift, scale, spend, msrp)
        null_lift = to_absolute(null_lift, lift, scale, spend, msrp) if null_lift != 0.0 else 0.0
    else:
        internal_lift = lift

    p_null, p_alt = simple_hypothesis_from_composite(group_sizes, baseline, null_lift, alt_lift, lift=internal_lift)
    return power(group_sizes, p_null, p_alt, alpha=alpha)


def minimum_detectable_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    power: Callable[..., float] = score_power,
    drop: bool = False,
    lift: str = "relative",
    spend: float | None = None,
    msrp: float | None = None,
    *,
    comparisons: str = "control",
) -> float:
    """Minimum detectable lift.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     beta : float, optional
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     drop : boolean, optional
        If True, the minimum detectable drop will be returned.
        Defaults to False, returning the minimum detectable lift.
     lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How to interpret the null/alternative lift. Defaults to "relative".
     spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
     msrp : float, optional
        Revenue per unit. Required for "revenue" lift.

    Returns
    -------
     mdl : float
        Minimum detectable lift/drop associated with test, in the units
        specified by ``lift``.

    Notes
    -----
    Uses binary search to compute the smallest lift/drop with adequate
    power. With three or more groups, it is the minimum detectable lift of
    the least-powered comparison at ``alpha / m`` (see :func:`abtest_power`),
    and scaled lifts are expressed over that comparison's groups.
    """
    group_sizes, alpha = _weakest_comparison(group_sizes, alpha, comparisons)
    if lift in _SCALED_LIFTS:
        scale = max(group_sizes)
        internal_lift = "absolute"
        internal_null = to_absolute(null_lift, lift, scale, spend, msrp) if null_lift != 0.0 else 0.0
    else:
        internal_lift = lift
        internal_null = null_lift

    tol = 1e-6

    # The alternative rate must stay in (0, 1): a relative drop cannot pass -100%,
    # and an increase cannot push the rate above 1.
    if internal_lift == "relative":
        limit = -1.0 if drop else (1 - baseline) / baseline
    else:
        limit = -baseline if drop else 1 - baseline

    # Find an extremum bound on the MDL, approaching the limit without crossing it
    mdl_inner = 0.0
    if internal_lift == "relative":
        mdl_extremum = math.copysign(min(0.2, 0.5 * abs(limit)), limit)
    else:
        mdl_extremum = 0.99 * limit

    pwr = abtest_power(
        group_sizes,
        baseline,
        mdl_extremum,
        alpha=alpha,
        null_lift=internal_null,
        power=power,
        lift=internal_lift,
    )

    for _ in range(_MAX_EXPANSIONS):
        if pwr >= 1 - beta:
            break
        mdl_inner = mdl_extremum
        if internal_lift == "relative" and abs(2 * mdl_extremum) < abs(limit):
            mdl_extremum *= 2
        else:
            mdl_extremum = 0.5 * (limit + mdl_extremum)

        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl_extremum,
            alpha=alpha,
            null_lift=internal_null,
            power=power,
            lift=internal_lift,
        )
    else:
        direction = "drop" if drop else "lift"
        raise ValueError(
            f"No {direction} keeping the treatment rate within [0, 1] reaches power {1 - beta}. "
            "Consider larger group sizes, a smaller target power, or a larger alpha."
        )

    while abs(mdl_extremum - mdl_inner) > tol:
        mdl = 0.5 * (mdl_inner + mdl_extremum)
        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl,
            alpha=alpha,
            null_lift=internal_null,
            power=power,
            lift=internal_lift,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase mdl
            mdl_inner = mdl
        else:
            # Adequate power, decrease mdl
            mdl_extremum = mdl

    if drop:
        mdl_extremum *= -1.0

    if lift in _SCALED_LIFTS:
        return from_absolute(mdl_extremum, lift, scale, spend, msrp)
    return mdl_extremum


def required_sample_size(
    baseline: float,
    alt_lift: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    null_lift: float = 0.0,
    power: Callable[..., float] = score_power,
    lift: str = "relative",
    *,
    comparisons: str = "control",
) -> int:
    """Calculate the required sample size.

    Parameters
    ----------
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Lift associated with the alternative hypothesis, in the units
        specified by ``lift``.
     alpha : float
        Type-I error rate threshold. Defaults to 0.05.
     beta : float
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     group_proportions : array_like or None
        Fraction of experimental units in each group. If None
        (default), will use an even split.
     null_lift : float
        Lift associated with the null hypothesis. Defaults to 0.0.
     power : function
        Function that computes power, such as `score_power` (default).
     lift : {"relative", "absolute"}, optional
        How to interpret the null/alternative lift. Defaults to "relative".
        Scaled lift types (incremental, roas, revenue, cpa) are not
        supported because the effect size depends on the unknown sample size.
     comparisons : {"control", "all"}, optional
        With three or more groups (``group_proportions`` of length three or
        more), which comparisons ``analyze()`` will make; the sample size
        gives the least-powered comparison the target power at ``alpha / m``
        (see :func:`abtest_power`). Ignored with two groups.

    Returns
    -------
     sample_size : int
        Minimum sample size, across all experiment groups, required to
        have desired sensitivity.

    Notes
    -----
    Uses binary search to compute the smallest sample size with
    adequate power. Only "relative" and "absolute" lifts are supported;
    scaled lifts (incremental, roas, revenue, cpa) depend on the group
    sizes which are the unknown being solved for.
    """
    if lift in _SCALED_LIFTS:
        raise ValueError(
            f"lift={lift!r} is not supported for required_sample_size because "
            f"the absolute effect size depends on the group sizes being solved "
            f"for. Convert to 'relative' or 'absolute' lift first."
        )

    tol = 0.01

    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    def sample_size_to_group_sizes(ss: int) -> list[int]:
        return [int(ss * g) for g in group_proportions]

    # Find an upper bound on the required sample size
    ss_lower = 0
    ss_upper = 1000

    pwr = abtest_power(
        sample_size_to_group_sizes(ss_upper),
        baseline,
        alt_lift,
        alpha=alpha,
        null_lift=null_lift,
        power=power,
        lift=lift,
        comparisons=comparisons,
    )

    while pwr < 1 - beta:
        ss_lower = ss_upper
        ss_upper *= 2
        pwr = abtest_power(
            sample_size_to_group_sizes(ss_upper),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
        )

    # Stop at a gap of one: below 100 the relative tolerance is under 1, and the
    # integer midpoint would then equal ss_lower forever.
    while ss_upper - ss_lower > max(1, tol * ss_lower):
        ss = int(0.5 * (ss_lower + ss_upper))
        pwr = abtest_power(
            sample_size_to_group_sizes(ss),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase ss
            ss_lower = ss
        else:
            # Adequate power, decrease ss
            ss_upper = ss

    return ss_upper


def plot_power_curve(
    baseline: float,
    alt_lift: float,
    alpha: float = 0.05,
    null_lift: float = 0.0,
    power: Callable[..., float] = score_power,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
    spend: float | None = None,
    msrp: float | None = None,
    *,
    comparisons: str = "control",
    dark_mode: bool = False,
) -> go.Figure:
    """Plot statistical power as a function of total sample size.

    Parameters
    ----------
    baseline : float
        Baseline success rate associated with the first experiment group.
    alt_lift : float
        Lift associated with the alternative hypothesis.
    alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
    null_lift : float, optional
        Lift associated with the null hypothesis. Defaults to 0.0.
    power : function, optional
        Function that computes power, such as ``score_power`` (default).
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How to interpret the null/alternative lift. Defaults to ``"relative"``.
    sample_sizes : array_like or None, optional
        Explicit total sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    group_proportions : array_like or None, optional
        Fraction of experimental units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 100.
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.
    comparisons : {"control", "all"}, optional
        With three or more groups in ``group_proportions``, which comparisons
        ``analyze()`` will make (see :func:`abtest_power`).
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).

    Returns
    -------
    go.Figure
        An interactive Plotly figure with total sample size on the x-axis and
        power on the y-axis.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    if sample_sizes is None:
        beta = 0.2
        target_ss = required_sample_size(
            baseline,
            alt_lift,
            alpha=alpha,
            beta=beta,
            group_proportions=group_proportions,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
        )
        max_ss = int(target_ss * 2)
        sample_sizes = np.linspace(max(20, max_ss // n_points), max_ss, n_points, dtype=int)

    powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
            spend=spend,
            msrp=msrp,
        )
        for ss in sample_sizes
    ]

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

    fig.update_layout(
        title="Power Curve",
        xaxis_title="Total sample size",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    apply_dark_mode(fig, dark_mode)
    return fig


def plot_sensitivity_curve(
    baseline: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    power: Callable[..., float] = score_power,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
    spend: float | None = None,
    msrp: float | None = None,
    *,
    comparisons: str = "control",
    dark_mode: bool = False,
) -> go.Figure:
    """Plot minimum detectable lift as a function of total sample size.

    Parameters
    ----------
    baseline : float
        Baseline success rate associated with the first experiment group.
    alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
    beta : float, optional
        Type-II error rate threshold (1 - power). Defaults to 0.2.
    null_lift : float, optional
        Lift associated with the null hypothesis. Defaults to 0.0.
    power : function, optional
        Function that computes power, such as ``score_power`` (default).
    lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
        How to interpret the null/alternative lift. Defaults to ``"relative"``.
    sample_sizes : array_like or None, optional
        Explicit total sample sizes to evaluate. When ``None`` (default), an
        evenly spaced sequence of ``n_points`` values is generated automatically.
    group_proportions : array_like or None, optional
        Fraction of experimental units in each group. Defaults to ``[0.5, 0.5]``.
    n_points : int, optional
        Number of sample-size points to evaluate when ``sample_sizes`` is
        ``None``. Defaults to 100.
    spend : float, optional
        Campaign spend. Required for "roas" and "cpa" lifts.
    msrp : float, optional
        Revenue per unit. Required for "revenue" lift.
    comparisons : {"control", "all"}, optional
        With three or more groups in ``group_proportions``, which comparisons
        ``analyze()`` will make (see :func:`abtest_power`).
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).

    Returns
    -------
    go.Figure
        An interactive Plotly figure with total sample size on the x-axis and
        minimum detectable lift on the y-axis.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    if sample_sizes is None:
        if lift in _SCALED_LIFTS:
            raise ValueError(
                f"lift={lift!r} requires explicit sample_sizes because the "
                f"automatic range uses required_sample_size, which does not "
                f"support scaled lift types."
            )
        target_ss = required_sample_size(
            baseline,
            alt_lift=0.05,
            alpha=alpha,
            beta=beta,
            group_proportions=group_proportions,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
        )
        min_ss = max(20, target_ss // 10)
        max_ss = target_ss * 5
        sample_sizes = np.linspace(min_ss, max_ss, n_points, dtype=int)

    mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=power,
            lift=lift,
            comparisons=comparisons,
            spend=spend,
            msrp=msrp,
        )
        for ss in sample_sizes
    ]

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
    fig.update_layout(
        title="Sensitivity Curve",
        xaxis_title="Total sample size",
        yaxis_title=y_label,
        yaxis_tickformat=y_format,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    apply_dark_mode(fig, dark_mode)
    return fig
