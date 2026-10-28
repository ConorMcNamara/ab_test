"""The posterior density of each variant's lift over the control, with its interval and ROPE."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import plotly.colors
from scipy.ndimage import gaussian_filter1d

from ab_test._display import format_percent, resolve_plot_color
from ab_test._plotting.spec import AxisFormat, LineSeries, LineSpec, Segment, Span, VLine
from ab_test.bayesian_binomial.credible_intervals import calculate_hdi_from_samples

__all__ = ["lift_pdf_spec", "rope_verdict", "variant_colors"]

_BINS = 512


def _density(samples: np.ndarray[Any, Any], lo: float, hi: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Smooth density of ``samples`` on ``[lo, hi]``: a fine histogram blurred by a Gaussian kernel.

    The kernel's width follows Silverman's rule, so this matches a Gaussian
    KDE without evaluating every sample at every grid point.
    """
    counts, edges = np.histogram(samples, bins=_BINS, range=(lo, hi))
    width = edges[1] - edges[0]
    bandwidth = 1.06 * np.std(samples) * len(samples) ** -0.2
    smoothed = gaussian_filter1d(counts.astype(float), bandwidth / width, mode="constant")
    return (edges[:-1] + edges[1:]) / 2, smoothed / (len(samples) * width)


def variant_colors(color: str | dict[str, Any] | list[Any] | None, names: Sequence[str]) -> list[str | None]:
    """Each variant's color from a ``color`` argument as the ``plot`` methods take it; ``None`` where unset."""
    palette = resolve_plot_color(color)
    if isinstance(palette, list):
        return [palette[i % len(palette)] for i in range(len(names))]
    if isinstance(palette, dict):
        return [palette.get(name) for name in names]
    return [None] * len(names)


def rope_verdict(lower: float, upper: float, low_threshold: float, high_threshold: float) -> str:
    """Where an interval sits against the ROPE: wholly outside, wholly inside, or overlapping it."""
    if upper < low_threshold or lower > high_threshold:
        return "outside ROPE"
    if low_threshold <= lower and upper <= high_threshold:
        return "inside ROPE"
    return "overlaps ROPE"


def lift_pdf_spec(
    names: Sequence[str],
    draws: Sequence[np.ndarray[Any, Any]],
    lift: str,
    confidence_level: float,
    cred_int_method: Literal["credible", "hdi"],
    rope: tuple[float, float],
    colors: Sequence[str | None],
    title: str,
    *,
    dark_mode: bool = False,
) -> LineSpec:
    """Describe the posterior density of each variant's lift over the first, the control.

    Each variant's interval is a bar beneath its density, solid when it
    excludes zero and dashed when it spans it, as forest-plot markers are
    filled or hollow. A dashed line marks zero and a gray band the ROPE.
    Where each interval sits against the ROPE is added to the title with a
    single variant, which gets no legend, and to each legend entry otherwise.

    Parameters
    ----------
    names : sequence of str
        Variant names, the control first.
    draws : sequence of ndarray
        Posterior draws of each variant's rate, in the order of ``names``.
    lift : {"relative", "absolute"}
        Kind of lift.
    confidence_level : float
        Probability mass of each interval.
    cred_int_method : {"credible", "hdi"}
        Equal-tailed interval or Highest Density Interval.
    rope : tuple of float
        Lower and upper bounds of the ROPE, in the units of ``lift``.
    colors : sequence of str or None
        Color of each variant; ``None`` takes Plotly's default color at the variant's position.
    title : str
        Title of the plot, before the interval and the ROPE verdict are added.
    dark_mode : bool
        Whether to draw on a dark background.
    """
    control = draws[0]
    lifts = []
    for treatment in draws[1:]:
        diff = treatment - control
        lifts.append(diff / np.where(control == 0, 1e-9, control) if lift == "relative" else diff)
    # Wide enough to show zero and the ROPE beside the bulk of every density.
    lo = min(float(np.min([np.quantile(s, 0.0005) for s in lifts])), 0.0, rope[0])
    hi = max(float(np.max([np.quantile(s, 0.9995) for s in lifts])), 0.0, rope[1])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    palette = plotly.colors.qualitative.Plotly
    curves = []
    intervals = []
    verdicts = []
    for index, samples in enumerate(lifts, start=1):
        if cred_int_method == "hdi":
            lower, upper = calculate_hdi_from_samples(samples, confidence_level)
        else:
            tail = (1 - confidence_level) / 2 * 100
            lower, upper = np.percentile(samples, [tail, 100 - tail])
        x, y = _density(samples, lo, hi)
        color = colors[index] or palette[index % len(palette)]
        verdicts.append(rope_verdict(lower, upper, *rope))
        curves.append(LineSeries(f"{names[index]} vs {names[0]}", x.tolist(), y.tolist(), color, fill=True))
        intervals.append((float(lower), float(upper), color))

    interval = f"{format_percent(confidence_level)}% {'HDI' if cred_int_method == 'hdi' else 'credible interval'}"
    if len(curves) == 1:
        title = f"{title} · {interval} {verdicts[0]}"
    else:
        title = f"{title} (bars: {interval})"
        curves = [replace(curve, name=f"{curve.name}: {verdict}") for curve, verdict in zip(curves, verdicts)]

    # The interval bars sit just below the densities, scaled to the tallest one.
    max_y = max(max(curve.y) for curve in curves)
    segments = [
        Segment(lower, upper, -max_y * 0.05 * row, color, width=5, dash="dash" if lower <= 0 <= upper else None)
        for row, (lower, upper, color) in enumerate(intervals, start=1)
    ]
    return LineSpec(
        series=curves,
        title=title,
        x_title=f"{lift.capitalize()} lift",
        y_title="Probability Density",
        x_format=AxisFormat(tickformat=".1%"),
        segments=segments,
        vlines=[VLine(0.0)],
        spans=[Span(rope[0], rope[1], "ROPE")],
        dark_mode=dark_mode,
    )
