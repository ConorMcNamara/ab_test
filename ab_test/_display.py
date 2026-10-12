"""Shared display helpers for contingency tables.

Holds the colorblind palettes, lift-value formatting, and the forest-plot
renderer used by both :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
and :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`.
"""

import math
from typing import Any, overload

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

__all__ = [
    "COLORBLIND_PALETTES",
    "resolve_plot_color",
    "convert_to_tabulate_str",
    "render_forest_plot",
    "apply_dark_mode",
    "combine_lift_panels",
    "tabulate_summary",
    "format_percent",
]

# Colorblind-friendly palettes keyed by name. "wong" and "ito" are aliases for
# the same palette.
COLORBLIND_PALETTES: dict[str, list[str]] = {
    "ibm": ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"],
    "wong": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"],
    "ito": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"],
    "tol": ["#332288", "#117733", "#44aa99", "#88ccee", "#ddcc77", "#cc6677", "#aa4499", "#882255"],
    "tol_bright": ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"],
    "tol_vibrant": ["#ee7733", "#0077bb", "#33bbee", "#ee3377", "#cc3311", "#009988"],
    "tol_muted": ["#cc6677", "#332288", "#ddcc77", "#117733", "#88ccee", "#882255", "#44aa99", "#999933", "#aa4499"],
    "tol_light": ["#77aadd", "#ee8866", "#eedd88", "#ffaabb", "#99ddff", "#44bb99", "#bbcc33", "#bbcc33"],
}


def format_percent(fraction: float) -> str:
    """Format a probability as a percentage label without rounding it to an integer.

    ``0.95`` gives ``"95"``, ``0.975`` gives ``"97.5"`` and ``0.003`` gives ``"0.3"``,
    where ``round`` would give ``"98"`` (banker's rounding to an even integer) or ``"0"``.
    """
    return f"{round(fraction * 100, 6):g}"


def _format_infinity(value: float) -> str:
    """Render an infinite bound as a compact symbol.

    Parameters
    ----------
    value : float
        An infinite value (``math.inf`` or ``-math.inf``).

    Returns
    -------
    str
        ``"∞"`` for positive infinity, ``"-∞"`` for negative infinity.
    """
    return "∞" if value > 0 else "-∞"


def resolve_plot_color(
    color: str | dict[str, Any] | list[Any] | None,
) -> list[Any] | dict[str, Any] | None:
    """Resolve a color argument into a concrete palette.

    Parameters
    ----------
    color : str, list, dict, or None
        If ``None``, defers to Plotly's default color scheme (returns ``None``).
        If a string, one of the names in :data:`COLORBLIND_PALETTES`.
        If a list or dict, returned unchanged.

    Returns
    -------
    list, dict, or None
        The resolved palette, or ``None`` when no explicit color was requested.

    Raises
    ------
    ValueError
        If ``color`` is a string that names no known palette.
    TypeError
        If ``color`` is not a string, list, dict, or ``None``.
    """
    if color is None:
        return None
    if isinstance(color, str):
        if color not in COLORBLIND_PALETTES:
            raise ValueError(f"No support for color scheme {color}")
        return COLORBLIND_PALETTES[color]
    if isinstance(color, (list, dict)):
        return color
    raise TypeError("Color can be a string, list, dict, or None")


@overload
def convert_to_tabulate_str(value: float, lift: str) -> str | float: ...


@overload
def convert_to_tabulate_str(value: list[Any], lift: str) -> list[Any]: ...


def convert_to_tabulate_str(value: float | list[Any], lift: str) -> str | list[Any] | float:
    """Convert lift values to display strings (percentages or dollar amounts).

    Parameters
    ----------
    value : float or list
        The value(s) to format.
    lift : str
        The lift type, which determines the unit: ``"revenue"``/``"cpa"`` render
        as dollar amounts, ``"roas"`` (incremental conversions per dollar) as a
        unitless number to 4 significant figures, ``"absolute"``/``"relative"``
        as percentages, and ``"incremental"`` is returned unchanged.

    Returns
    -------
    str, float, or list
        The formatted value(s). A scalar is returned as a formatted string (or
        the raw value for ``"incremental"``); a list is returned element-wise.

    Raises
    ------
    ValueError
        If ``lift`` is not a supported type.
    TypeError
        If ``value`` is neither a number nor a list.
    """

    def _format_one(val: float) -> str | float:
        if math.isinf(val):
            return _format_infinity(val)
        if lift == "roas":
            # Conversions per dollar, not dollars; often well below 0.01.
            return f"{val:,.4g}"
        if lift in ["revenue", "cpa"]:
            return f"${round(val, 2):,}"
        if lift in ["absolute", "relative"]:
            return f"{round(val * 100.0, 2)}%"
        if lift == "incremental":
            return val
        raise ValueError(f"No support for {lift}")

    if isinstance(value, (int, float)):
        return _format_one(value)
    if isinstance(value, list):
        return [_format_one(val) for val in value]
    raise TypeError(f"No support for converting {value} to string")


def tabulate_summary(row_labels: list[str], values: list[Any]) -> str:
    """Render a single result as a vertical ``Statistic | Value`` grid.

    Parameters
    ----------
    row_labels : list of str
        The name of each statistic.
    values : list
        The value of each statistic, in the same order as ``row_labels``.

    Returns
    -------
    str
        The grid-formatted table.
    """

    def _format_value(val: Any) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, int):
            return f"{val:,}"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return str(val)

    result: str = tabulate(
        [[label, _format_value(val)] for label, val in zip(row_labels, values)],
        headers=["Statistic", "Value"],
        tablefmt="grid",
        disable_numparse=True,
    )
    return result


_LIFT_LABELS = {
    "relative": "Relative Lift",
    "absolute": "Absolute Lift",
    "incremental": "Incremental Lift",
    "revenue": "Revenue Lift",
    "roas": "ROAS Lift",
    "cpa": "CPA Lift",
}


def _lift_axis_format(lift_type: str) -> dict[str, str]:
    """Tick format (and prefix) for an x-axis showing ``lift_type``."""
    if lift_type in ("relative", "absolute"):
        return {"tickformat": ",.0%"}
    if lift_type == "revenue":
        return {"tickprefix": "$", "tickformat": "~s"}
    if lift_type == "roas":
        # Conversions per dollar: unitless, with significant figures for small values.
        return {"tickformat": ".3~g"}
    if lift_type == "cpa":
        return {"tickprefix": "$", "tickformat": "0.2"}
    return {"tickformat": "~s"}


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


def combine_lift_panels(figures: list[go.Figure], title: str, panel_titles: list[str]) -> go.Figure:
    """Place single-lift forest plots side by side, sharing the y-axis.

    Each figure becomes one panel: its traces, vertical reference lines and
    x-axis tick format are copied over, and its legend entries are kept only
    for the first panel. Layout taken from the first figure: template, legend
    visibility, y-axis title and a reversed y-axis.

    Parameters
    ----------
    figures : list of plotly.graph_objects.Figure
        One finished single-lift figure per panel, with the same rows.
    title : str
        Title of the combined figure.
    panel_titles : list of str
        x-axis title of each panel.

    Returns
    -------
    plotly.graph_objects.Figure
        The combined figure.
    """
    combined = make_subplots(rows=1, cols=len(figures), shared_yaxes=True, horizontal_spacing=0.08)
    for col, (panel, panel_title) in enumerate(zip(figures, panel_titles, strict=True), start=1):
        layout: Any = panel.layout
        traces: Any = panel.data
        for trace in traces:
            if col > 1:
                trace.showlegend = False
            combined.add_trace(trace, row=1, col=col)
        for shape in layout.shapes:
            if shape.type == "line" and shape.x0 == shape.x1:
                combined.add_vline(x=shape.x0, line=shape.line.to_plotly_json(), opacity=shape.opacity, row=1, col=col)
        axis = layout.xaxis
        combined.update_xaxes(
            title_text=panel_title, tickformat=axis.tickformat, tickprefix=axis.tickprefix, row=1, col=col
        )
    first: Any = figures[0].layout
    combined.update_layout(
        title_text=title, template=first.template, showlegend=first.showlegend, yaxis_title=first.yaxis.title.text
    )
    if first.yaxis.autorange == "reversed":
        combined.update_yaxes(autorange="reversed")
    return combined


def render_forest_plot(
    names: list[str],
    individual_results: dict[str, dict[str, float]],
    incremental_results: dict[str, Any] | list[dict[str, Any]] | None,
    is_individual: bool = True,
    reverse_plot: bool = True,
    color: str | dict[str, Any] | list[Any] | None = None,
    experiment_name: str | None = None,
    metric_name: str | None = None,
    dark_mode: bool = False,
) -> None:
    """Render a dot-and-whisker (forest) plot of point estimates and intervals.

    Parameters
    ----------
    names : list of str
        Cell names, in order.
    individual_results : dict
        Per-cell results keyed by name (plus a ``"Total"`` entry), each holding
        ``"lift"``, ``"ci_lower"``, and ``"ci_upper"``. Used when
        ``is_individual`` is True. This is only populated once
        ``.analyze_individually()`` has been run; nothing will render before
        then.
    incremental_results : dict, list of dict, or None
        Comparative results holding ``"lift"``, ``"ci_lower"``, ``"ci_upper"``,
        and ``"lift_type"``. A list of results (e.g. absolute and relative lift)
        is drawn as side-by-side panels sharing the y-axis, each with its own
        interval and axis format. Used when ``is_individual`` is False. This is
        only populated once ``.analyze()`` has been run; nothing will render
        before then.
    is_individual : bool, default=True
        Whether to plot each cell's individual performance or the comparative
        performance between variants.
    reverse_plot : bool, default=True
        Whether to reverse the y-axis order.
    color : str, list, dict, or None, default=None
        Passed to :func:`resolve_plot_color`.
    experiment_name : str or None, default=None
        Name of the experiment, included in the plot title when given.
    metric_name : str or None, default=None
        Name of the metric being plotted, included in the plot title and, for
        individual plots, the x-axis label.
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template). Very dark palette colors, such as the
        first colors of ``"tol"``, can be hard to see on it.

    Raises
    ------
    ValueError
        If ``is_individual`` is False but ``incremental_results`` is None,
        i.e. ``.analyze()`` was not run first.
    KeyError
        If ``is_individual`` is True but ``individual_results`` is empty,
        i.e. ``.analyze_individually()`` was not run first.
    """
    plot_color = resolve_plot_color(color)
    fig = go.Figure()  # type: ignore[attr-defined]
    if is_individual:
        for index, name in enumerate(names):
            ind_results = individual_results[name]
            c = (
                (plot_color[index] if isinstance(plot_color, list) else plot_color[name])
                if plot_color is not None
                else None
            )
            marker: dict[str, Any] = {"symbol": "diamond", "size": 12.5}
            error_x: dict[str, Any] = {
                "type": "data",
                "symmetric": False,
                "array": [ind_results["ci_upper"] - ind_results["lift"]],
                "arrayminus": [ind_results["lift"] - ind_results["ci_lower"]],
                "visible": True,
            }
            if c is not None:
                marker["color"] = c
                error_x["color"] = c
            fig.add_trace(
                go.Scatter(  # type: ignore[attr-defined]
                    x=[ind_results["lift"]],
                    y=[name],
                    marker=marker,
                    error_x=error_x,
                    name=name,
                )
            )
        total_results = individual_results["Total"]
        c_total = (
            (plot_color[index + 1] if isinstance(plot_color, list) else plot_color["Total"])
            if plot_color is not None
            else None
        )
        marker_total: dict[str, Any] = {"symbol": "diamond", "size": 12.5}
        error_x_total: dict[str, Any] = {
            "type": "data",
            "symmetric": False,
            "array": [total_results["ci_upper"] - total_results["lift"]],
            "arrayminus": [total_results["lift"] - total_results["ci_lower"]],
            "visible": True,
        }
        if c_total is not None:
            marker_total["color"] = c_total
            error_x_total["color"] = c_total
        fig.add_trace(
            go.Scatter(  # type: ignore[attr-defined]
                x=[total_results["lift"]],
                y=["Total"],
                marker=marker_total,
                error_x=error_x_total,
                name="Total",
            )
        )
        fig.update_layout(xaxis_tickformat=",.0%")
    else:
        if incremental_results is None:
            raise ValueError("Call .analyze() before plotting incremental results.")
        panels = incremental_results if isinstance(incremental_results, list) else [incremental_results]
        if len(panels) > 1:
            fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.08)
        c_inc = (
            (plot_color[0] if isinstance(plot_color, list) else list(plot_color.values())[0])
            if plot_color is not None
            else None
        )
        for col, result in enumerate(panels, start=1):
            # Three or more variants give one row per comparison; two give a single "Total" row.
            rows = list(result["comparisons"].items()) if "comparisons" in result else [("Total", result)]
            for row_index, (row_name, row) in enumerate(rows):
                c_row = c_inc
                if plot_color is not None and len(rows) > 1:
                    c_row = (
                        plot_color[row_index % len(plot_color)]
                        if isinstance(plot_color, list)
                        else plot_color.get(row_name, c_inc)
                    )
                marker_inc: dict[str, Any] = {"symbol": "diamond", "size": 12.5}
                error_x_inc: dict[str, Any] = {
                    "type": "data",
                    "symmetric": False,
                    "array": [row["ci_upper"] - row["lift"]],
                    "arrayminus": [row["lift"] - row["ci_lower"]],
                    "visible": True,
                }
                if c_row is not None:
                    marker_inc["color"] = c_row
                    error_x_inc["color"] = c_row
                trace = go.Scatter(  # type: ignore[attr-defined]
                    x=[row["lift"]],
                    y=[row_name],
                    marker=marker_inc,
                    error_x=error_x_inc,
                    name=row_name,
                    showlegend=col == 1,
                )
                if len(panels) > 1:
                    fig.add_trace(trace, row=1, col=col)
                else:
                    fig.add_trace(trace)
            axis_format = _lift_axis_format(result["lift_type"])
            if len(panels) > 1:
                fig.update_xaxes(
                    title_text=_LIFT_LABELS.get(result["lift_type"], "Lift"), row=1, col=col, **axis_format
                )
            else:
                fig.update_xaxes(**axis_format)

    subtitle = " - ".join(part for part in (experiment_name, metric_name) if part)
    if is_individual:
        title = f"Individual Performance by Cell{f': {subtitle}' if subtitle else ''}"
        xaxis_title = metric_name or "Success Rate"
        yaxis_title = "Cell"
    else:
        panels = incremental_results if isinstance(incremental_results, list) else [incremental_results]
        labels = [_LIFT_LABELS.get(r["lift_type"], "Lift") if r is not None else "Lift" for r in panels]
        compared = f"{names[0]} vs. {names[1]}" if len(names) == 2 else "Each Comparison"
        title = f"{' and '.join(labels)} of {compared}{f': {subtitle}' if subtitle else ''}"
        # With several panels each x-axis carries its own title.
        xaxis_title = labels[0] if len(labels) == 1 else None
        yaxis_title = ""
    fig.update_layout(title_text=title, yaxis_title=yaxis_title)
    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    apply_dark_mode(fig, dark_mode)
    if reverse_plot:
        fig.update_yaxes(autorange="reversed")
    fig.show()  # type: ignore[no-untyped-call]
