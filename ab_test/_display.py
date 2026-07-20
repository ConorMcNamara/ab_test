"""Shared display helpers for contingency tables.

Holds the colorblind palettes, lift-value formatting, and the forest-plot
renderer used by both :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
and :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`.
"""

import math
from typing import Any, overload

import plotly.graph_objects as go

__all__ = [
    "COLORBLIND_PALETTES",
    "resolve_plot_color",
    "convert_to_tabulate_str",
    "render_forest_plot",
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
        The lift type, which determines the unit: ``"revenue"``/``"roas"`` render
        as dollar amounts, ``"absolute"``/``"relative"`` as percentages, and
        ``"incremental"`` is returned unchanged.

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
        if lift in ["revenue", "roas"]:
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


def render_forest_plot(
    names: list[str],
    individual_results: dict[str, dict[str, float]],
    incremental_results: dict[str, Any] | None,
    is_individual: bool = True,
    reverse_plot: bool = True,
    color: str | dict[str, Any] | list[Any] | None = None,
) -> None:
    """Render a dot-and-whisker (forest) plot of point estimates and intervals.

    Parameters
    ----------
    names : list of str
        Cell names, in order.
    individual_results : dict
        Per-cell results keyed by name (plus a ``"Total"`` entry), each holding
        ``"lift"``, ``"ci_lower"``, and ``"ci_upper"``. Used when
        ``is_individual`` is True.
    incremental_results : dict or None
        Comparative results holding ``"lift"``, ``"ci_lower"``, ``"ci_upper"``,
        and ``"lift_type"``. Used when ``is_individual`` is False.
    is_individual : bool, default=True
        Whether to plot each cell's individual performance or the comparative
        performance between variants.
    reverse_plot : bool, default=True
        Whether to reverse the y-axis order.
    color : str, list, dict, or None, default=None
        Passed to :func:`resolve_plot_color`.

    Raises
    ------
    ValueError
        If ``is_individual`` is False but ``incremental_results`` is None.
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
        c_inc = (
            (plot_color[0] if isinstance(plot_color, list) else list(plot_color.values())[0])
            if plot_color is not None
            else None
        )
        marker_inc: dict[str, Any] = {"symbol": "diamond", "size": 12.5}
        error_x_inc: dict[str, Any] = {
            "type": "data",
            "symmetric": False,
            "array": [incremental_results["ci_upper"] - incremental_results["lift"]],
            "arrayminus": [incremental_results["lift"] - incremental_results["ci_lower"]],
            "visible": True,
        }
        if c_inc is not None:
            marker_inc["color"] = c_inc
            error_x_inc["color"] = c_inc
        fig.add_trace(
            go.Scatter(  # type: ignore[attr-defined]
                x=[incremental_results["lift"]],
                y=["Total"],
                marker=marker_inc,
                error_x=error_x_inc,
                name="Total",
            )
        )
        if incremental_results["lift_type"] in ["relative", "absolute"]:
            fig.update_layout(xaxis_tickformat=",.0%")
        elif incremental_results["lift_type"] in ["revenue", "roas"]:
            if incremental_results["lift_type"] == "revenue":
                fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat="~s")
            else:
                fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat="0.2")
        else:
            fig.update_layout(xaxis_tickformat="~s")
    if reverse_plot:
        fig.update_layout(yaxis={"autorange": "reversed"})
    fig.show()  # type: ignore[no-untyped-call]
