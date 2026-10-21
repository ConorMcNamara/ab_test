"""Specs for the power and sensitivity curves shared by the power modules."""

from collections.abc import Iterable

from ab_test._plotting.spec import AxisFormat, HLine, LineSeries, LineSpec

__all__ = ["curve", "power_curve_spec", "sensitivity_curve_spec"]

# The first two colors of Plotly's palette: the curve, and the curve it is compared with.
_CURVE_COLOR = "#636EFA"
_BASELINE_COLOR = "#EF553B"


def curve(
    name: str, x: Iterable[float], y: Iterable[float], *, baseline_dash: str | None = None, markers: bool = False
) -> LineSeries:
    """Describe one curve, or with ``baseline_dash`` a dashed baseline to compare it with.

    Parameters
    ----------
    name : str
        Legend entry of the curve.
    x, y : iterable of float
        The curve's points.
    baseline_dash : {"dash", "dot"} or None, default=None
        Draw the curve as a baseline in this dash pattern and a second color.
    markers : bool, default=False
        Whether to mark each point.
    """
    return LineSeries(
        name,
        [float(value) for value in x],
        [float(value) for value in y],
        color=_BASELINE_COLOR if baseline_dash else _CURVE_COLOR,
        dash=baseline_dash,
        markers=markers,
    )


def power_curve_spec(
    series: list[LineSeries], title: str, x_title: str, *, y_title: str = "Power", dark_mode: bool = False
) -> LineSpec:
    """Describe a power curve: power from 0% to 100% with a line at 80%."""
    return LineSpec(
        series=series,
        title=title,
        x_title=x_title,
        y_title=y_title,
        y_format=AxisFormat(tickformat=",.0%"),
        y_range=(0.0, 1.05),
        hlines=[HLine(0.8, "80% power")],
        dark_mode=dark_mode,
    )


def sensitivity_curve_spec(
    series: list[LineSeries], title: str, x_title: str, y_title: str, lift: str, *, dark_mode: bool = False
) -> LineSpec:
    """Describe a sensitivity curve, with percentage ticks for relative and absolute lift."""
    return LineSpec(
        series=series,
        title=title,
        x_title=x_title,
        y_title=y_title,
        y_format=AxisFormat(tickformat=",.0%" if lift in ("relative", "absolute") else None),
        dark_mode=dark_mode,
    )
