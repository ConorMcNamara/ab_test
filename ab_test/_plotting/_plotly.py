"""Plotly backend for the plot specs in :mod:`ab_test._plotting.spec`."""

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ab_test._display import apply_dark_mode
from ab_test._plotting.spec import AxisFormat, ForestSpec, LineSpec

__all__ = ["forest", "line"]


def _axis_kwargs(axis_format: AxisFormat) -> dict[str, str]:
    """Turn ``axis_format`` into Plotly axis properties, leaving unset ones out."""
    kwargs = {"tickformat": axis_format.tickformat, "tickprefix": axis_format.tickprefix}
    return {key: value for key, value in kwargs.items() if value is not None}


def forest(spec: ForestSpec) -> go.Figure:
    """Draw a forest plot, one subplot per panel sharing the y-axis."""
    multi = len(spec.panels) > 1
    fig = (
        make_subplots(rows=1, cols=len(spec.panels), shared_yaxes=True, horizontal_spacing=0.08)
        if multi
        else go.Figure()
    )
    for col, panel in enumerate(spec.panels, start=1):
        for point in panel.points:
            marker: dict[str, Any] = {"symbol": point.shape, "size": point.size}
            error_x: dict[str, Any] = {
                "type": "data",
                "symmetric": False,
                "array": [point.upper - point.estimate],
                "arrayminus": [point.estimate - point.lower],
                "visible": True,
            }
            if point.color is not None:
                marker["color"] = point.color
                error_x["color"] = point.color
            trace = go.Scatter(x=[point.estimate], y=[point.label], marker=marker, error_x=error_x, name=point.label)  # type: ignore[attr-defined]
            if multi:
                # Each panel gets its own legend section, titled with its lift, so every
                # point has an entry and the title toggles the whole panel.
                trace.update(legendgroup=f"panel{col}", legendgrouptitle_text=panel.legend_title)
                if spec.show_legend:
                    trace.update(showlegend=True)
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
    if multi:
        # Clicking an entry hides that point only; clicking a section title hides the panel.
        fig.update_layout(legend_groupclick="toggleitem")
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


def line(spec: LineSpec) -> go.Figure:
    """Draw a line chart with dashed horizontal reference lines."""
    fig = go.Figure()
    for series in spec.series:
        line_style = {"color": series.color or "#636EFA", "width": 2}
        if series.dash is not None:
            line_style["dash"] = series.dash
        fig.add_trace(
            go.Scatter(
                x=series.x,
                y=series.y,
                mode="lines+markers" if series.markers else "lines",
                line=line_style,
                name=series.name,
            )
        )
    for hline in spec.hlines:
        fig.add_hline(
            y=hline.y,
            line_dash="dash",
            line_color="gray",
            annotation_text=hline.label,
            annotation_position="top left",
        )
    fig.update_layout(
        title=spec.title,
        xaxis_title=spec.x_title,
        yaxis_title=spec.y_title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(**_axis_kwargs(spec.y_format))
    if spec.y_range is not None:
        fig.update_yaxes(range=list(spec.y_range))
    apply_dark_mode(fig, spec.dark_mode)
    return fig
