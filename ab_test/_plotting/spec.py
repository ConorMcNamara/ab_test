"""Backend-neutral descriptions of the package's plots.

Plot functions compute their numbers into one of these specs; a backend in
:mod:`ab_test._plotting` then draws it with Plotly or Altair. A spec holds
only data, labels and formats, never library objects, so both backends draw
the same plot from it.
"""

from dataclasses import dataclass, field

__all__ = ["AxisFormat", "ForestPoint", "VLine", "ForestPanel", "ForestSpec", "LineSeries", "HLine", "LineSpec"]


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
    estimate, lower, upper : float
        The point estimate and its interval.
    color : str or None
        Color of the row; ``None`` leaves it to the backend's color scheme.
    shape : {"diamond", "circle"}
        Marker shape; pooled estimates are drawn as diamonds.
    size : float
        Marker diameter in pixels.
    """

    label: str
    estimate: float
    lower: float
    upper: float
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
    legend_title : str or None
        Title of this panel's legend section when there are several panels.
    vlines : list of VLine
        Vertical reference lines.
    """

    points: list[ForestPoint]
    x_title: str | None = None
    x_format: AxisFormat = field(default_factory=AxisFormat)
    legend_title: str | None = None
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
    """One line of a line chart.

    Parameters
    ----------
    name : str
        Legend entry of the line.
    x, y : list of float
        The line's points.
    color : str or None
        Color of the line; ``None`` uses the first color of Plotly's palette.
    dash : {"dash", "dot"} or None
        Dash pattern, such as for a baseline to compare against; ``None`` draws
        a solid line.
    markers : bool
        Whether to mark each point, such as when the points are few.
    """

    name: str
    x: list[float]
    y: list[float]
    color: str | None = None
    dash: str | None = None
    markers: bool = False


@dataclass(frozen=True)
class HLine:
    """A dashed horizontal reference line, labelled at its left end."""

    y: float
    label: str | None = None


@dataclass(frozen=True)
class LineSpec:
    """A line chart, such as a power or sensitivity curve.

    Parameters
    ----------
    series : list of LineSeries
        The lines to draw.
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
    """

    series: list[LineSeries]
    title: str
    x_title: str
    y_title: str
    y_format: AxisFormat = field(default_factory=AxisFormat)
    y_range: tuple[float, float] | None = None
    hlines: list[HLine] = field(default_factory=list)
    dark_mode: bool = False
