"""Altair backend for the plot specs in :mod:`ab_test._plotting.spec`.

Altair is an optional dependency, so this module is imported only when a plot
asks for ``backend="altair"``.
"""

from typing import Any

import altair as alt
import pandas as pd

from ab_test._plotting.spec import AxisFormat, ForestSpec, LineSpec

__all__ = ["forest", "line"]

# Vega's "dark" theme, written into the chart so it holds in static exports
# too, not only where vega-embed applies the theme.
_DARK_BACKGROUND = "#333"
_DARK_CONFIG: dict[str, Any] = {
    "view": {"stroke": "#888"},
    "title": {"color": "#fff"},
    "axis": {
        "domainColor": "#fff",
        "gridColor": "#888",
        "tickColor": "#fff",
        "labelColor": "#fff",
        "titleColor": "#fff",
    },
    "legend": {"labelColor": "#fff", "titleColor": "#fff"},
    "text": {"color": "#fff"},
}


def _axis(axis_format: AxisFormat, title: str | None) -> alt.Axis:
    """Build an Altair axis with ``axis_format``'s tick labels."""
    if axis_format.tickprefix is not None:
        # Vega has no tick prefix; prepend it in a label expression instead.
        number = f'format(datum.value, "{axis_format.tickformat}")' if axis_format.tickformat else "datum.label"
        return alt.Axis(title=title, labelExpr=f'"{axis_format.tickprefix}" + {number}')
    if axis_format.tickformat is not None:
        return alt.Axis(title=title, format=axis_format.tickformat)
    return alt.Axis(title=title)


def _tooltip_format(axis_format: AxisFormat) -> str:
    """Pick a tooltip format with more precision than the axis ticks."""
    return ".2%" if axis_format.tickformat and axis_format.tickformat.endswith("%") else ",.4~g"


def _color_scale(labels: list[str], colors: list[str | None]) -> alt.Scale:
    """Color each label explicitly when every one has a color, else use Vega's default scheme."""
    if any(color is None for color in colors):
        return alt.Scale()
    domain = list(dict.fromkeys(labels))
    by_label: dict[str, Any] = dict(zip(labels, colors, strict=True))
    return alt.Scale(domain=domain, range=[by_label[label] for label in domain])


def _finish(chart: Any, title: str, dark_mode: bool) -> Any:
    """Add the title and, in dark mode, the dark theme to a top-level chart."""
    chart = chart.properties(title=title)
    if dark_mode:
        chart = chart.properties(usermeta={"embedOptions": {"theme": "dark"}}).configure(
            background=_DARK_BACKGROUND, **_DARK_CONFIG
        )
    return chart


def forest(spec: ForestSpec) -> alt.TopLevelMixin:
    """Draw a forest plot, one chart per panel concatenated with a shared y-axis."""
    points = [point for panel in spec.panels for point in panel.points]
    labels = [point.label for point in points]
    rows = list(dict.fromkeys(labels))
    color = alt.Color(
        "label:N", scale=_color_scale(labels, [point.color for point in points]), legend=alt.Legend(title=None)
    )
    # Clicking a legend entry fades every other row, in every panel.
    picked = alt.selection_point(fields=["label"], bind="legend")
    charts = []
    for index, panel in enumerate(spec.panels):
        data = pd.DataFrame(
            {
                "label": [point.label for point in panel.points],
                "estimate": [point.estimate for point in panel.points],
                "lower": [point.lower for point in panel.points],
                "upper": [point.upper for point in panel.points],
            }
        )
        number = _tooltip_format(panel.x_format)
        base = alt.Chart(data).encode(
            y=alt.Y(
                "label:N",
                sort=rows if spec.reverse else rows[::-1],
                # Later panels share the first panel's row labels.
                axis=alt.Axis(title=spec.y_title or None) if index == 0 else None,
            ),
            color=color,
            opacity=alt.condition(picked, alt.value(1.0), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("label:N", title="Name"),
                alt.Tooltip("estimate:Q", title="Estimate", format=number),
                alt.Tooltip("lower:Q", title="Lower", format=number),
                alt.Tooltip("upper:Q", title="Upper", format=number),
            ],
        )
        x_scale = alt.Scale(zero=False)
        whisker = base.mark_rule(strokeWidth=2).encode(
            x=alt.X("lower:Q", scale=x_scale, axis=_axis(panel.x_format, panel.x_title)), x2="upper:Q"
        )
        dot = base.mark_point(shape="diamond", filled=True, size=150).encode(x=alt.X("estimate:Q", scale=x_scale))
        layer = alt.layer(whisker, dot)
        charts.append(layer.add_params(picked) if index == 0 else layer)
    chart: Any = charts[0] if len(charts) == 1 else alt.hconcat(*charts).resolve_scale(y="shared")
    return _finish(chart, spec.title, spec.dark_mode)  # type: ignore[no-any-return]


def line(spec: LineSpec) -> alt.TopLevelMixin:
    """Draw a line chart with dashed horizontal reference lines."""
    data = pd.DataFrame(
        {
            "series": [series.name for series in spec.series for _ in series.x],
            "x": [x for series in spec.series for x in series.x],
            "y": [y for series in spec.series for y in series.y],
        }
    )
    names = [series.name for series in spec.series]
    colors = [series.color or "#636EFA" for series in spec.series]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=colors),
        # Like Plotly, a lone line gets no legend.
        legend=alt.Legend(title=None, orient="top-right") if len(names) > 1 else None,
    )
    y_scale = alt.Scale(domain=list(spec.y_range), nice=False) if spec.y_range is not None else alt.Scale(zero=False)
    base = alt.Chart(data).encode(
        x=alt.X("x:Q", title=spec.x_title),
        y=alt.Y("y:Q", scale=y_scale, axis=_axis(spec.y_format, spec.y_title)),
        color=color,
    )
    # Hovering anywhere marks the nearest x and shows every line's value there.
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["x"], empty=False)
    layers: list[Any] = [
        base.mark_line(strokeWidth=2),
        base.mark_point(filled=True)
        .encode(
            opacity=alt.condition(nearest, alt.value(1.0), alt.value(0.0)),
            tooltip=[
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("x:Q", title=spec.x_title, format=","),
                alt.Tooltip("y:Q", title=spec.y_title, format=_tooltip_format(spec.y_format)),
            ],
        )
        .add_params(nearest),
    ]
    if spec.hlines:
        hlines = pd.DataFrame({"y": [h.y for h in spec.hlines], "label": [h.label or "" for h in spec.hlines]})
        rule = alt.Chart(hlines).encode(y="y:Q")
        layers.append(rule.mark_rule(color="gray", strokeDash=[6, 4]))
        layers.append(
            rule.mark_text(align="left", baseline="bottom", dx=4, dy=-2, color="gray").encode(
                x=alt.value(0), text="label:N"
            )
        )
    chart = alt.layer(*layers).properties(width=600, height=400)
    return _finish(chart, spec.title, spec.dark_mode)  # type: ignore[no-any-return]
