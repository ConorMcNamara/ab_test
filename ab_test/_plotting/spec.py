"""Backend-neutral descriptions of the package's plots.

Plot functions compute their numbers into one of these specs; a backend in
:mod:`ab_test._plotting` then draws it with Plotly or Altair. A spec holds
only data, labels and formats, never library objects, so both backends draw
the same plot from it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "AxisFormat",
    "ForestPoint",
    "VLine",
    "ForestPanel",
    "ForestSpec",
    "LineSeries",
    "Band",
    "Segment",
    "HLine",
    "LineSpec",
]


@dataclass(frozen=True)
class AxisFormat:
    """Tick labels of a numeric axis.

    Parameters
    ----------
    tickformat : str or None
        A d3-format string, such as ``",.0%"``, which both Plotly and Vega-Lite
        understand. ``None`` keeps the backend's default.
    tickprefix : str or None
        Text put before every tick label, such as ``"$"``.
    """

    tickformat: str | None = None
    tickprefix: str | None = None


@dataclass(frozen=True)
class ForestPoint:
    """One row of a forest plot: a point estimate and its interval.

    Parameters
    ----------
    label : str
        The row's name on the y-axis.
    estimate : float
        The point estimate.
    lower, upper : float or None
        Its interval; ``None`` draws the point alone, such as an observed
        cluster rate.
    color : str or None
        Color of the row; ``None`` leaves it to the backend's color scheme.
    shape : {"diamond", "circle"}
        Marker shape; pooled estimates are drawn as diamonds.
    size : float
        Marker diameter in pixels.
    """

    label: str
    estimate: float
    lower: float | None
    upper: float | None
    color: str | None = None
    shape: str = "diamond"
    size: float = 12.5


@dataclass(frozen=True)
class VLine:
    """A dashed gray vertical reference line, such as zero (no effect)."""

    x: float
    opacity: float = 1.0


@dataclass(frozen=True)
class ForestPanel:
    """One panel of a forest plot, sharing its rows with the other panels.

    Parameters
    ----------
    points : list of ForestPoint
        The rows, in order.
    x_title : str or None
        Title of the x-axis; ``None`` leaves it untitled.
    x_format : AxisFormat
        Tick labels of the x-axis.
    vlines : list of VLine
        Vertical reference lines.
    """

    points: list[ForestPoint]
    x_title: str | None = None
    x_format: AxisFormat = field(default_factory=AxisFormat)
    vlines: list[VLine] = field(default_factory=list)


@dataclass(frozen=True)
class ForestSpec:
    """A dot-and-whisker plot of one or more side-by-side panels.

    Parameters
    ----------
    panels : list of ForestPanel
        One panel per lift, drawn left to right.
    title : str
        Title of the plot.
    y_title : str
        Title of the shared y-axis.
    reverse : bool
        Whether the first row is drawn at the top rather than the bottom.
    dark_mode : bool
        Whether to draw on a dark background.
    show_legend : bool
        Whether to show a legend of the rows.
    white_background : bool
        Whether to draw on a plain white background rather than the backend's
        default light one; ignored in dark mode.
    """

    panels: list[ForestPanel]
    title: str
    y_title: str
    reverse: bool = True
    dark_mode: bool = False
    show_legend: bool = True
    white_background: bool = False


@dataclass(frozen=True)
class LineSeries:
    """One line, or set of points, of a line chart.

    Parameters
    ----------
    name : str
        Legend entry of the series.
    x : sequence of float or sequence of str
        The points' x values; strings when the x-axis is categorical.
    y : sequence of float
        The points' y values.
    color : str or None
        Color of the series; ``None`` leaves it to the backend's color scheme.
    dash : {"dash", "dot"} or None
        Dash pattern, such as for a baseline to compare against; ``None`` draws
        a solid line.
    markers : bool
        Whether to mark each point, such as when the points are few.
    lines : bool
        Whether to join the points with a line; ``False`` draws points only.
    marker_shape : {"circle", "diamond"}
        Shape of the markers.
    marker_size : float or None
        Marker diameter in pixels; ``None`` keeps the backend's default.
    fill : bool
        Whether to shade the area between the line and zero, such as under a
        density.
    errors : sequence of float or None
        Half-width of a vertical error bar at each point.
    """

    name: str
    x: Sequence[float] | Sequence[str]
    y: Sequence[float]
    color: str | None = None
    dash: str | None = None
    markers: bool = False
    lines: bool = True
    marker_shape: str = "circle"
    marker_size: float | None = None
    fill: bool = False
    errors: Sequence[float] | None = None


@dataclass(frozen=True)
class Band:
    """A shaded band between two lines, such as a confidence sequence.

    Parameters
    ----------
    name : str
        Legend entry of the band.
    x : sequence of float or sequence of str
        The band's x values.
    lower, upper : sequence of float
        The band's edges at each x.
    color : str
        Color of the band, drawn translucent.
    """

    name: str
    x: Sequence[float] | Sequence[str]
    lower: Sequence[float]
    upper: Sequence[float]
    color: str = "#636EFA"


@dataclass(frozen=True)
class Segment:
    """A thick horizontal bar from ``x0`` to ``x1`` at height ``y``, such as an interval under a density."""

    x0: float
    x1: float
    y: float
    color: str
    width: float = 4.0


@dataclass(frozen=True)
class HLine:
    """A gray horizontal reference line, labelled at its left end.

    Parameters
    ----------
    y : float
        Height of the line.
    label : str or None
        Text at its left end; ``None`` leaves it unlabelled.
    dash : {"dash", "dot"}
        Dash pattern.
    opacity : float
        Opacity of the line.
    """

    y: float
    label: str | None = None
    dash: str = "dash"
    opacity: float = 1.0


@dataclass(frozen=True)
class LineSpec:
    """A line chart, such as a power curve, a density or a value over time.

    Parameters
    ----------
    series : list of LineSeries
        The lines and points to draw.
    title, x_title, y_title : str
        Titles of the chart and its axes.
    y_format : AxisFormat
        Tick labels of the y-axis.
    y_range : tuple of float or None
        Fixed y-axis range; ``None`` fits it to the data.
    hlines : list of HLine
        Horizontal reference lines.
    dark_mode : bool
        Whether to draw on a dark background.
    x_format : AxisFormat
        Tick labels of the x-axis.
    x_categorical : bool
        Whether the x values are labels, such as checkpoint names, kept in the
        order given.
    x_tick_step : float or None
        Smallest gap between x ticks, such as 1 for whole-numbered analyses.
    bands : list of Band
        Shaded bands, drawn beneath the series.
    segments : list of Segment
        Horizontal bars, such as intervals drawn under densities.
    """

    series: list[LineSeries]
    title: str
    x_title: str
    y_title: str
    y_format: AxisFormat = field(default_factory=AxisFormat)
    y_range: tuple[float, float] | None = None
    hlines: list[HLine] = field(default_factory=list)
    dark_mode: bool = False
    x_format: AxisFormat = field(default_factory=AxisFormat)
    x_categorical: bool = False
    x_tick_step: float | None = None
    bands: list[Band] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
