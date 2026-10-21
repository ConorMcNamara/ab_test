"""Altair backend for the plot specs in :mod:`ab_test._plotting.spec`.

Altair is an optional dependency, so this module is imported only when a plot
asks for ``backend="altair"``.
"""

from typing import Any

import altair as alt
import pandas as pd

from ab_test._plotting.spec import AxisFormat, ForestSpec, LineSpec

__all__ = ["forest", "line"]

# Vega stroke dashes for the spec's dash names; [1, 0] is solid.
_DASHES = {None: [1, 0], "dash": [6, 4], "dot": [2, 3]}

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


def _vega_format(tickformat: str) -> str:
    """Give a d3 format with no type a trimmed fixed-point one.

    Plotly reads ``"$,"`` as plain numbers, but Vega's default type turns such
    axes into exponent notation (``$5e+2``).
    """
    return tickformat if tickformat[-1:].isalpha() or tickformat.endswith("%") else f"{tickformat}~f"


def _axis(axis_format: AxisFormat, title: str | None, **kwargs: Any) -> alt.Axis:
    """Build an Altair axis with ``axis_format``'s tick labels and any other ``alt.Axis`` options."""
    tickformat = _vega_format(axis_format.tickformat) if axis_format.tickformat else None
    if axis_format.tickprefix is not None:
        # Vega has no tick prefix; prepend it in a label expression instead.
        number = f'format(datum.value, "{tickformat}")' if tickformat else "datum.label"
        return alt.Axis(title=title, labelExpr=f'"{axis_format.tickprefix}" + {number}', **kwargs)
    if tickformat is not None:
        return alt.Axis(title=title, format=tickformat, **kwargs)
    return alt.Axis(title=title, **kwargs)


def _tooltip_format(axis_format: AxisFormat) -> str:
    """Pick a tooltip format with more precision than the axis ticks."""
    return ".2%" if axis_format.tickformat and axis_format.tickformat.endswith("%") else ",.4~g"


def _color_scale(labels: list[str], colors: list[str | None]) -> alt.Scale:
    """Color each label explicitly when every one has a color, else use Vega's default scheme.

    The domain is explicit either way, so a legend shared with another channel,
    such as stroke dash, merges into one.
    """
    domain = list(dict.fromkeys(labels))
    if any(color is None for color in colors):
        return alt.Scale(domain=domain)
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
        "label:N",
        scale=_color_scale(labels, [point.color for point in points]),
        legend=alt.Legend(title=None) if spec.show_legend else None,
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
                "shape": [point.shape for point in panel.points],
                # Plotly sizes markers by diameter, Vega by area.
                "area": [point.size**2 for point in panel.points],
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
            tooltip=[
                alt.Tooltip("label:N", title="Name"),
                alt.Tooltip("estimate:Q", title="Estimate", format=number),
                alt.Tooltip("lower:Q", title="Lower", format=number),
                alt.Tooltip("upper:Q", title="Upper", format=number),
            ],
        )
        if spec.show_legend:
            base = base.encode(opacity=alt.condition(picked, alt.value(1.0), alt.value(0.2)))
        x_scale = alt.Scale(zero=False)
        layers: list[Any] = [
            # Rows without an interval, such as observed cluster rates, get no whisker.
            base.mark_rule(strokeWidth=2)
            .encode(x=alt.X("lower:Q", scale=x_scale, axis=_axis(panel.x_format, panel.x_title)), x2="upper:Q")
            .transform_filter("isValid(datum.lower)"),
            base.mark_point(filled=True, opacity=1.0).encode(
                x=alt.X("estimate:Q", scale=x_scale),
                shape=alt.Shape("shape:N", scale=None),
                size=alt.Size("area:Q", scale=None),
            ),
        ]
        for vline in panel.vlines:
            layers.append(
                alt.Chart(pd.DataFrame({"x": [vline.x]}))
                .mark_rule(color="gray", strokeDash=[6, 4], opacity=vline.opacity)
                .encode(x="x:Q")
            )
        layer = alt.layer(*layers)
        charts.append(layer.add_params(picked) if index == 0 and spec.show_legend else layer)
    chart: Any = charts[0] if len(charts) == 1 else alt.hconcat(*charts).resolve_scale(y="shared")
    return _finish(chart, spec.title, spec.dark_mode)  # type: ignore[no-any-return]


def _only(names: list[str]) -> alt.FieldOneOfPredicate:
    """Keep only the rows of the named series."""
    return alt.FieldOneOfPredicate(field="series", oneOf=names)


def line(spec: LineSpec) -> alt.TopLevelMixin:
    """Draw a line chart with its bands, bars and dashed horizontal reference lines."""
    data = pd.DataFrame(
        {
            "series": [series.name for series in spec.series for _ in series.x],
            "x": [x for series in spec.series for x in series.x],
            "y": [y for series in spec.series for y in series.y],
            "err": [e for series in spec.series for e in (series.errors or [float("nan")] * len(series.x))],
            "shape": [series.marker_shape for series in spec.series for _ in series.x],
            # Plotly sizes markers by diameter, Vega by area; Plotly's default diameter is 6.
            "area": [(series.marker_size or 6) ** 2 for series in spec.series for _ in series.x],
        }
    )
    # Bands share the series' color scale and legend.
    names = [band.name for band in spec.bands] + [series.name for series in spec.series]
    colors = [band.color for band in spec.bands] + [series.color for series in spec.series]
    # Like Plotly, a lone line gets no legend. Above the plot like Plotly's, with
    # line samples that show each dash pattern.
    legend = (
        alt.Legend(title=None, orient="top", direction="horizontal", symbolType="stroke", symbolStrokeWidth=2)
        if len(names) > 1
        else None
    )
    color = alt.Color("series:N", scale=_color_scale(names, colors), legend=legend)
    # Same field and legend as the color, so Vega merges them into one legend.
    dashes = [_DASHES[None]] * len(spec.bands) + [_DASHES[s.dash] for s in spec.series]
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=dashes), legend=legend)

    # Categorical x values are labels; keep the order they come in.
    x_order = list(dict.fromkeys(str(x) for series in spec.series for x in series.x))
    if spec.x_categorical:
        x_axis = alt.Axis(title=spec.x_title, labelAngle=0)
        x_tooltip = alt.Tooltip("x:N", title=spec.x_title)
    else:
        extra = {"tickMinStep": spec.x_tick_step} if spec.x_tick_step is not None else {}
        x_axis = _axis(spec.x_format, spec.x_title, **extra)
        x_tooltip = alt.Tooltip("x:Q", title=spec.x_title, format=",")

    def x_channel(field: str = "x") -> alt.X:
        if spec.x_categorical:
            return alt.X(f"{field}:N", sort=x_order, axis=x_axis)
        return alt.X(f"{field}:Q", axis=x_axis, scale=alt.Scale(zero=False))

    y_scale = alt.Scale(domain=list(spec.y_range), nice=False) if spec.y_range is not None else alt.Scale(zero=False)
    y_axis = _axis(spec.y_format, spec.y_title)
    base = alt.Chart(data).encode(x=x_channel(), y=alt.Y("y:Q", scale=y_scale, axis=y_axis), color=color)

    layers: list[Any] = []
    if spec.bands:
        bands = pd.DataFrame(
            {
                "series": [band.name for band in spec.bands for _ in band.x],
                "x": [x for band in spec.bands for x in band.x],
                "lower": [y for band in spec.bands for y in band.lower],
                "upper": [y for band in spec.bands for y in band.upper],
            }
        )
        layers.append(
            alt.Chart(bands)
            .mark_area(opacity=0.2)
            .encode(x=x_channel(), y=alt.Y("lower:Q", scale=y_scale, axis=y_axis), y2="upper:Q", color=color)
        )
    lined = [series.name for series in spec.series if series.lines]
    filled = [series.name for series in spec.series if series.fill]
    marked = [series.name for series in spec.series if series.markers or not series.lines]
    with_errors = [series.name for series in spec.series if series.errors is not None]
    if filled:
        layers.append(base.mark_area(opacity=0.4).encode(y2=alt.datum(0)).transform_filter(_only(filled)))
    line_layer = base.mark_line(strokeWidth=2).encode(strokeDash=dash)
    layers.append(line_layer if len(lined) == len(spec.series) else line_layer.transform_filter(_only(lined)))
    if marked:
        layers.append(
            base.mark_point(filled=True, opacity=1.0)
            .encode(shape=alt.Shape("shape:N", scale=None), size=alt.Size("area:Q", scale=None))
            .transform_filter(_only(marked))
        )
    if with_errors:
        layers.append(
            base.mark_rule(strokeWidth=1.5)
            .encode(y=alt.Y("lo:Q", scale=y_scale, axis=y_axis), y2="hi:Q")
            .transform_filter(_only(with_errors))
            .transform_calculate(lo="datum.y - datum.err", hi="datum.y + datum.err")
        )
    # Hovering anywhere marks the nearest x and shows every line's value there.
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["x"], empty=False)
    layers.append(
        base.mark_point(filled=True)
        .encode(
            opacity=alt.condition(nearest, alt.value(1.0), alt.value(0.0)),
            tooltip=[
                alt.Tooltip("series:N", title=""),
                x_tooltip,
                alt.Tooltip("y:Q", title=spec.y_title, format=_tooltip_format(spec.y_format)),
            ],
        )
        .add_params(nearest)
    )
    if spec.segments:
        segments = pd.DataFrame(
            {
                "x0": [segment.x0 for segment in spec.segments],
                "x1": [segment.x1 for segment in spec.segments],
                "y": [segment.y for segment in spec.segments],
                "stroke": [segment.color for segment in spec.segments],
                "width": [segment.width for segment in spec.segments],
            }
        )
        layers.append(
            alt.Chart(segments)
            .mark_rule()
            .encode(
                x=alt.X("x0:Q"),
                x2="x1:Q",
                y="y:Q",
                stroke=alt.Stroke("stroke:N", scale=None),
                strokeWidth=alt.StrokeWidth("width:Q", scale=None),
            )
        )
    for hline in spec.hlines:
        rule = alt.Chart(pd.DataFrame({"y": [hline.y], "label": [hline.label or ""]})).encode(y="y:Q")
        style: dict[str, Any] = {"color": "gray", "strokeDash": _DASHES[hline.dash]}
        if hline.opacity != 1.0:
            style["opacity"] = hline.opacity
        layers.append(rule.mark_rule(**style))
        if hline.label:
            layers.append(
                rule.mark_text(align="left", baseline="bottom", dx=4, dy=-2, color="gray").encode(
                    x=alt.value(0), text="label:N"
                )
            )
    chart = alt.layer(*layers).properties(width=600, height=400)
    return _finish(chart, spec.title, spec.dark_mode)  # type: ignore[no-any-return]
