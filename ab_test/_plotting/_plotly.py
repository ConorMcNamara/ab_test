"""Plotly backend for the plot specs in :mod:`ab_test._plotting.spec`."""

from typing import Any

import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ab_test._plotting.spec import AxisFormat, ForestSpec, LineSpec

__all__ = ["forest", "line"]


def apply_dark_mode(fig: go.Figure, dark_mode: bool) -> go.Figure:
    """Switch a figure to Plotly's dark template when ``dark_mode`` is True.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to restyle in place.
    dark_mode : bool
        If True, use the ``"plotly_dark"`` template (dark background, light
        text and gridlines). If False, the figure is left unchanged.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, for chaining.
    """
    if dark_mode:
        fig.update_layout(template="plotly_dark")
    return fig


def _axis_kwargs(axis_format: AxisFormat) -> dict[str, str]:
    """Turn ``axis_format`` into Plotly axis properties, leaving unset ones out."""
    kwargs = {"tickformat": axis_format.tickformat, "tickprefix": axis_format.tickprefix}
    return {key: value for key, value in kwargs.items() if value is not None}


def forest(spec: ForestSpec) -> go.Figure:
    """Draw a forest plot, one subplot per panel sharing the y-axis."""
    multi = len(spec.panels) > 1
    # Each row keeps one color in every panel: its own, or Plotly's next default color.
    palette = plotly.colors.qualitative.Plotly
    colors: dict[str, str] = {}
    for point in (point for panel in spec.panels for point in panel.points):
        colors.setdefault(point.label, point.color or palette[len(colors) % len(palette)])
    fig = (
        make_subplots(rows=1, cols=len(spec.panels), shared_yaxes=True, horizontal_spacing=0.08)
        if multi
        else go.Figure()
    )
    for col, panel in enumerate(spec.panels, start=1):
        for point in panel.points:
            color = colors[point.label]
            marker: dict[str, Any] = {"symbol": point.shape, "size": point.size, "color": color}
            trace = go.Scatter(x=[point.estimate], y=[point.label], marker=marker, name=point.label)  # type: ignore[attr-defined]
            if point.lower is not None and point.upper is not None:
                error_x: dict[str, Any] = {
                    "type": "data",
                    "symmetric": False,
                    "array": [point.upper - point.estimate],
                    "arrayminus": [point.estimate - point.lower],
                    "visible": True,
                    "color": color,
                }
                trace.update(error_x=error_x)
            if multi:
                # One legend entry per row, which toggles that row in every panel.
                trace.update(legendgroup=point.label, showlegend=spec.show_legend and col == 1)
                fig.add_trace(trace, row=1, col=col)
            else:
                fig.add_trace(trace)
        for vline in panel.vlines:
            subplot = {"row": 1, "col": col} if multi else {}
            fig.add_vline(x=vline.x, line_dash="dash", line_color="gray", opacity=vline.opacity, **subplot)
        if multi:
            fig.update_xaxes(title_text=panel.x_title, row=1, col=col, **_axis_kwargs(panel.x_format))
        else:
            fig.update_xaxes(**_axis_kwargs(panel.x_format))
    fig.update_layout(title_text=spec.title, yaxis_title=spec.y_title)
    if not spec.show_legend:
        fig.update_layout(showlegend=False)
    if spec.white_background:
        fig.update_layout(template="plotly_white")
    # With several panels each x-axis carries its own title.
    if not multi and spec.panels[0].x_title is not None:
        fig.update_layout(xaxis_title=spec.panels[0].x_title)
    apply_dark_mode(fig, spec.dark_mode)
    if spec.reverse:
        fig.update_yaxes(autorange="reversed")
    return fig


def _rgba(color: str, opacity: float) -> str:
    """Turn a ``#rrggbb`` color into a translucent ``rgba(...)`` one."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {opacity})"


def line(spec: LineSpec) -> go.Figure:
    """Draw a line chart with its bands, bars and dashed horizontal reference lines."""
    fig = go.Figure()
    for band in spec.bands:
        # Fill from the lower edge up to the invisible upper one drawn just before it.
        fig.add_trace(
            go.Scatter(x=band.x, y=band.upper, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=band.x,
                y=band.lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(band.color, 0.2),
                name=band.name,
            )
        )
    for series in spec.series:
        trace = go.Scatter(x=series.x, y=series.y, name=series.name)
        if series.lines:
            line_style: dict[str, Any] = {"width": 2}
            if series.color is not None:
                line_style["color"] = series.color
            if series.dash is not None:
                line_style["dash"] = series.dash
            trace.update(mode="lines+markers" if series.markers else "lines", line=line_style)
        else:
            trace.update(mode="markers")
        if series.markers or not series.lines:
            marker: dict[str, Any] = {}
            if series.marker_shape != "circle":
                marker["symbol"] = series.marker_shape
            if series.marker_size is not None:
                marker["size"] = series.marker_size
            if series.color is not None and not series.lines:
                marker["color"] = series.color
            if marker:
                trace.update(marker=marker)
        if series.fill:
            trace.update(fill="tozeroy", opacity=0.4)
        if series.errors is not None:
            trace.update(error_y={"type": "data", "array": series.errors, "visible": True})
        fig.add_trace(trace)
    for segment in spec.segments:
        fig.add_shape(
            type="line",
            x0=segment.x0,
            x1=segment.x1,
            y0=segment.y,
            y1=segment.y,
            line={"color": segment.color, "width": segment.width},
        )
    for hline in spec.hlines:
        label = {"annotation_text": hline.label, "annotation_position": "top left"} if hline.label else {}
        fig.add_hline(y=hline.y, line_dash=hline.dash, line_color="gray", opacity=hline.opacity, **label)
    fig.update_layout(
        title=spec.title,
        xaxis_title=spec.x_title,
        yaxis_title=spec.y_title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(**_axis_kwargs(spec.x_format))
    if spec.x_tick_step is not None:
        fig.update_xaxes(dtick=spec.x_tick_step)
    fig.update_yaxes(**_axis_kwargs(spec.y_format))
    if spec.y_range is not None:
        fig.update_yaxes(range=list(spec.y_range))
    apply_dark_mode(fig, spec.dark_mode)
    return fig
