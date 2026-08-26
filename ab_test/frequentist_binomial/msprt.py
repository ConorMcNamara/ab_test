"""Mixture Sequential Probability Ratio Test (mSPRT) for always-valid inference.

Provides always-valid p-values and confidence sequences that maintain
type-I error control regardless of how many times results are checked.
The test uses a Gaussian mixture on the effect size, following the
framework of Johari et al. (2017).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from ab_test.frequentist_binomial.utils import mle_under_alternative, mle_under_null, observed_lift, validate_two_group

if TYPE_CHECKING:
    from ab_test.frequentist_binomial.contingency import ContingencyTable

__all__ = [
    "msprt_test",
    "msprt_critical_value",
    "plot_msprt_over_time",
]


def _msprt_log_lambda(d: float, sigma2: float, tau2: float) -> float:
    """Log of the mSPRT likelihood ratio.

    Parameters
    ----------
    d : float
        Centered effect: observed absolute effect minus null-expected
        absolute effect.
    sigma2 : float
        Variance of the effect estimate under H0.
    tau2 : float
        Squared scale of the Gaussian mixing distribution.

    Returns
    -------
    float
        Log of the likelihood ratio Lambda_n.
    """
    return -0.5 * math.log(1 + tau2 / sigma2) + d * d * tau2 / (2 * sigma2 * (sigma2 + tau2))


def msprt_test(
    trials: np.ndarray[Any, Any] | list[Any],
    successes: np.ndarray[Any, Any] | list[Any],
    null_lift: float = 0.0,
    lift: str = "relative",
    crit: float | None = None,
    *,
    tau: float | None = None,
) -> float | bool:
    """Mixture Sequential Probability Ratio Test for a 2x2 contingency table.

    Computes an always-valid p-value that controls type-I error no matter
    how many times the data are examined. The test mixes over a Gaussian
    prior ``N(0, tau^2)`` on the effect size.

    Parameters
    ----------
    trials : array_like
        Number of trials in each group.
    successes : array_like
        Number of successes in each group.
    null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
    lift : {"relative", "absolute"}
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms.
    crit : float, optional
        Critical value for the likelihood ratio. If omitted, an always-valid
        p-value is returned. If passed, a boolean is returned indicating
        whether the likelihood ratio exceeds the critical value. Use
        :func:`msprt_critical_value` to obtain the threshold for a given
        alpha.
    tau : float or None, optional
        Scale of the Gaussian mixing distribution on the effect size. Larger
        values give more power for large effects but less for small ones.
        When ``None`` (the default), ``tau`` is set to the standard error of
        the difference under the pooled null — a unit-information prior that
        scales naturally with the data.

    Returns
    -------
    pval : float
        Always-valid p-value. Returned if ``crit`` is None.
    stat_sig : bool
        True if the likelihood ratio exceeds ``crit``. Returned if ``crit``
        is not None.

    Notes
    -----
    The always-valid p-value is ``min(1, 1 / Lambda_n)`` where ``Lambda_n``
    is the likelihood ratio. It is safe to compute at any sample size and
    reject whenever it drops below alpha, without inflating the type-I error.

    References
    ----------
    Johari, R., Pekelis, L., & Walsh, D. J. (2017). Peeking at A/B Tests:
    Why it matters, and what to do about it. *KDD '17*.
    """
    validate_two_group(trials, successes, null_lift, lift)

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    if min(p0) <= 1e-12 or max(p0) + 1e-12 >= 1.0:
        return 1.0 if crit is None else False

    sigma2 = p0[0] * (1 - p0[0]) / trials[0] + p0[1] * (1 - p0[1]) / trials[1]
    if sigma2 <= 1e-24:
        return 1.0 if crit is None else False

    d = (p1[1] - p1[0]) - (p0[1] - p0[0])

    if tau is None:
        tau = math.sqrt(sigma2)
    tau2 = tau * tau

    log_lambda = _msprt_log_lambda(d, sigma2, tau2)
    lambda_n = math.exp(min(log_lambda, 700))

    if crit is None:
        return min(1.0, 1.0 / lambda_n)
    return lambda_n >= crit


def msprt_critical_value(alpha: float = 0.05) -> float:
    """Critical value for the mSPRT likelihood ratio.

    Parameters
    ----------
    alpha : float
        Type-I error rate. Defaults to 0.05.

    Returns
    -------
    float
        The threshold ``1 / alpha``. Reject H0 when the likelihood ratio
        exceeds this value.
    """
    return 1.0 / alpha


def plot_msprt_over_time(
    tables: list[ContingencyTable],
    labels: list[str],
    lift: str = "relative",
    alpha: float = 0.05,
    null_lift: float = 0.0,
    *,
    tau: float | None = None,
) -> go.Figure:
    """Plot the mSPRT point estimate and confidence sequence over time.

    Parameters
    ----------
    tables : list of ContingencyTable
        One :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
        per checkpoint, each containing cumulative data up to that point.
    labels : list of str
        Display labels for the x-axis, one per table (e.g. dates).
    lift : {"relative", "absolute"}
        Kind of lift to plot.
    alpha : float
        Significance level for the confidence sequence.
    null_lift : float
        Lift associated with the null hypothesis.
    tau : float or None, optional
        Scale of the Gaussian mixing distribution. When ``None``, auto-derived
        at each checkpoint.

    Returns
    -------
    go.Figure
        An interactive Plotly figure showing the point estimate as a line with
        markers, a shaded confidence band, and a dashed null-lift reference.
    """
    if len(tables) != len(labels):
        raise ValueError(f"tables and labels must have the same length, got {len(tables)} and {len(labels)}")

    from ab_test.frequentist_binomial.confidence_intervals import confidence_interval

    estimates: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []

    test = functools.partial(msprt_test, tau=tau)
    functools.update_wrapper(test, msprt_test)

    for ct in tables:
        estimates.append(observed_lift(ct.trials, ct.successes, lift))
        lb, ub = confidence_interval(ct.trials, ct.successes, test=test, alpha=alpha, lift=lift)
        lbs.append(lb)
        ubs.append(ub)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=ubs,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=lbs,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
            name=f"{round((1 - alpha) * 100)}% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=estimates,
            mode="lines+markers",
            marker={"size": 8, "symbol": "diamond"},
            line={"color": "#636EFA", "width": 2},
            name="Point estimate",
        )
    )
    fig.add_hline(
        y=null_lift,
        line_dash="dash",
        line_color="gray",
        annotation_text="H₀",
        annotation_position="top left",
    )

    tick_format = ",.0%" if lift in ("relative", "absolute") else "~s"
    fig.update_layout(
        title="mSPRT Confidence Sequence Over Time",
        xaxis_title="Checkpoint",
        yaxis_title=f"{lift.capitalize()} lift",
        yaxis_tickformat=tick_format,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig
