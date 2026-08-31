"""Difference-in-differences analysis for heterogeneous treatment effects.

Compares treatment effects across independent segments to determine
whether the effect of an intervention varies by subgroup.  Accepts
multiple :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
objects (one per segment) and produces:

1. Per-segment treatment effects with Wald confidence intervals
2. A Cochran's Q omnibus test for effect heterogeneity
3. All pairwise DiD comparisons with multiplicity correction
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import scipy.stats as ss
from tabulate import tabulate

from ab_test._display import convert_to_tabulate_str, resolve_plot_color
from ab_test.corrections import adjust_pvalues
from ab_test.frequentist_binomial.contingency import ContingencyTable

__all__ = [
    "DiffInDiff",
    "cochrans_q",
]


def cochrans_q(
    effects: np.ndarray[Any, Any] | list[float],
    variances: np.ndarray[Any, Any] | list[float],
) -> tuple[float, float]:
    """Cochran's Q test for heterogeneity of treatment effects.

    Tests whether a set of effect estimates are consistent with a common
    underlying effect.  Under the null of homogeneity the statistic
    follows a chi-squared distribution with *K - 1* degrees of freedom,
    where *K* is the number of segments.

    Parameters
    ----------
    effects : array_like
        Point estimates (risk differences or log-risk-ratios) for each
        segment.
    variances : array_like
        Estimated variance of each point estimate.

    Returns
    -------
    statistic : float
        Cochran's Q statistic.
    pvalue : float
        P-value from a chi-squared(*K* - 1) distribution.

    Raises
    ------
    ValueError
        If fewer than 2 effects are supplied or any variance is
        non-positive.
    """
    effects_arr = np.asarray(effects, dtype=float)
    variances_arr = np.asarray(variances, dtype=float)
    k = len(effects_arr)
    if k < 2:
        raise ValueError(f"Cochran's Q requires at least 2 segments, got {k}")
    if np.any(variances_arr <= 0):
        raise ValueError("All variances must be positive — zero variance segments cannot be weighted")
    weights = 1.0 / variances_arr
    pooled = float(np.sum(weights * effects_arr) / np.sum(weights))
    q = float(np.sum(weights * (effects_arr - pooled) ** 2))
    pvalue = float(ss.chi2.sf(q, df=k - 1))
    return q, pvalue


_VALID_LIFTS = frozenset({"absolute", "relative", "incremental", "roas", "revenue"})


def _compute_segment_stats(
    tables: list[ContingencyTable],
    lift: str,
    alpha: float,
) -> dict[str, Any]:
    """Compute per-segment treatment effects, variances, and CIs.

    Parameters
    ----------
    tables : list[ContingencyTable]
        One table per segment, each with exactly 2 cells.
    lift : str
        One of ``"absolute"``, ``"relative"``, ``"incremental"``,
        ``"roas"``, or ``"revenue"``.
    alpha : float
        Significance level for Wald confidence intervals.

    Returns
    -------
    dict
        Keys: ``names``, ``effects`` (display scale),
        ``internal_effects`` (analysis scale), ``variances``,
        ``ci_lowers``, ``ci_uppers``, ``p_controls``, ``p_treatments``.
    """
    z_crit = ss.norm.isf(alpha / 2)
    names: list[str] = []
    effects: list[float] = []
    internal_effects: list[float] = []
    variances: list[float] = []
    ci_lowers: list[float] = []
    ci_uppers: list[float] = []
    p_controls: list[float] = []
    p_treatments: list[float] = []

    for table in tables:
        n_c, n_t = table.trials[0], table.trials[1]
        p_c = table.successes[0] / n_c
        p_t = table.successes[1] / n_t

        names.append(table.experiment_name)
        p_controls.append(p_c)
        p_treatments.append(p_t)

        if lift == "relative":
            if p_c == 0:
                raise ValueError(
                    f"Segment {table.experiment_name!r} has zero control rate — relative lift is undefined"
                )
            if p_t == 0:
                raise ValueError(
                    f"Segment {table.experiment_name!r} has zero treatment rate — log-risk-ratio is undefined"
                )
            log_rr = float(np.log(p_t / p_c))
            var_log = float((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
            se_log = np.sqrt(var_log)
            effects.append(p_t / p_c - 1)
            internal_effects.append(log_rr)
            variances.append(var_log)
            ci_lowers.append(float(np.exp(log_rr - z_crit * se_log) - 1))
            ci_uppers.append(float(np.exp(log_rr + z_crit * se_log) - 1))
        else:
            d = p_t - p_c
            var = p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c
            se = np.sqrt(var)

            if lift in ("incremental", "roas", "revenue"):
                n_max = max(n_c, n_t)
                scale = float(n_max)
                d_scaled = d * scale
                var_scaled = var * scale * scale
                se_scaled = se * scale
                ci_lo = d_scaled - z_crit * se_scaled
                ci_hi = d_scaled + z_crit * se_scaled

                if lift == "roas":
                    if table.spend is None:
                        raise ValueError(f"spend must be set on segment {table.experiment_name!r} for ROAS")
                    spend = table.spend
                    d_scaled /= spend
                    var_scaled /= spend * spend
                    ci_lo /= spend
                    ci_hi /= spend
                elif lift == "revenue":
                    if table.msrp is None:
                        raise ValueError(f"msrp must be set on segment {table.experiment_name!r} for revenue")
                    msrp = table.msrp
                    d_scaled *= msrp
                    var_scaled *= msrp * msrp
                    ci_lo *= msrp
                    ci_hi *= msrp

                effects.append(d_scaled)
                internal_effects.append(d_scaled)
                variances.append(var_scaled)
                ci_lowers.append(ci_lo)
                ci_uppers.append(ci_hi)
            else:
                effects.append(d)
                internal_effects.append(d)
                variances.append(var)
                ci_lowers.append(d - z_crit * se)
                ci_uppers.append(d + z_crit * se)

    return {
        "names": names,
        "effects": effects,
        "internal_effects": internal_effects,
        "variances": variances,
        "ci_lowers": ci_lowers,
        "ci_uppers": ci_uppers,
        "p_controls": p_controls,
        "p_treatments": p_treatments,
    }


class DiffInDiff:
    """Difference-in-differences analysis across independent segments.

    Accepts two or more
    :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
    objects, each representing an independent segment (e.g. men, women),
    and tests whether the treatment effect differs across segments.

    Parameters
    ----------
    *tables : ContingencyTable
        Two or more contingency tables, one per segment.  Each must have
        exactly 2 cells (control and treatment).  The ``experiment_name``
        of each table is used as the segment label.

    Raises
    ------
    ValueError
        If fewer than 2 tables are provided, any table does not have
        exactly 2 cells, segment names are duplicated, or metric names
        differ across tables.

    Examples
    --------
    >>> from ab_test.frequentist_binomial.contingency import ContingencyTable
    >>> from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff
    >>> men = ContingencyTable("Men", "converted")
    >>> _ = men.add("Control", successes=100, trials=1000)
    >>> _ = men.add("Treatment", successes=130, trials=1000)
    >>> women = ContingencyTable("Women", "converted")
    >>> _ = women.add("Control", successes=120, trials=1000)
    >>> _ = women.add("Treatment", successes=125, trials=1000)
    >>> test = DiffInDiff(men, women)
    >>> result = test.analyze(lift="absolute")
    """

    def __init__(self, *tables: ContingencyTable) -> None:
        if len(tables) < 2:
            raise ValueError(f"DiffInDiff requires at least 2 segments, got {len(tables)}")
        for t in tables:
            if len(t.names) != 2:
                raise ValueError(
                    f"Each segment must have exactly 2 cells (control and treatment), "
                    f"segment {t.experiment_name!r} has {len(t.names)}"
                )
        seg_names = [t.experiment_name for t in tables]
        if len(set(seg_names)) != len(seg_names):
            dupes = [n for n in seg_names if seg_names.count(n) > 1]
            raise ValueError(f"Duplicate segment name {dupes[0]!r}")
        metric_names = {t.metric_name for t in tables}
        if len(metric_names) > 1:
            raise ValueError(f"All segments must measure the same metric, got {metric_names}")

        self._tables: list[ContingencyTable] = list(tables)
        self.segment_names: list[str] = seg_names
        self.metric_name: str = tables[0].metric_name
        self.segment_results: dict[str, dict[str, Any]] | None = None
        self.heterogeneity_results: dict[str, Any] | None = None
        self.pairwise_results: list[dict[str, Any]] | None = None

    def analyze(
        self,
        lift: str = "absolute",
        alpha: float = 0.05,
        correction: str = "holm",
    ) -> str:
        """Run the difference-in-differences analysis.

        Parameters
        ----------
        lift : str, default="absolute"
            Scale for treatment effects: ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"roas"``, or ``"revenue"``.
        alpha : float, default=0.05
            Significance level for confidence intervals and tests.
        correction : str, default="holm"
            Multiplicity correction method for pairwise comparisons.
            Any method accepted by
            :func:`~ab_test.corrections.adjust_pvalues`.

        Returns
        -------
        str
            Formatted table with per-segment effects, Cochran's Q
            omnibus test, and pairwise DiD comparisons.
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        stats = _compute_segment_stats(self._tables, lift, alpha)
        z_crit = ss.norm.isf(alpha / 2)

        self.segment_results = {}
        for i, name in enumerate(stats["names"]):
            self.segment_results[name] = {
                "effect": stats["effects"][i],
                "ci_lower": stats["ci_lowers"][i],
                "ci_upper": stats["ci_uppers"][i],
                "p_control": stats["p_controls"][i],
                "p_treatment": stats["p_treatments"][i],
            }

        q_stat, q_pval = cochrans_q(stats["internal_effects"], stats["variances"])
        self.heterogeneity_results = {
            "Q_statistic": q_stat,
            "Q_pvalue": q_pval,
            "df": len(stats["names"]) - 1,
        }

        k = len(stats["names"])
        pairs = list(itertools.combinations(range(k), 2))
        raw_pvals: list[float] = []
        pairwise: list[dict[str, Any]] = []

        for i, j in pairs:
            delta_internal = stats["internal_effects"][i] - stats["internal_effects"][j]
            se = float(np.sqrt(stats["variances"][i] + stats["variances"][j]))
            z_val = delta_internal / se if se > 0 else 0.0
            raw_p = float(2 * ss.norm.sf(abs(z_val)))
            raw_pvals.append(raw_p)

            if lift == "relative":
                did_display = float(np.exp(delta_internal) - 1)
                ci_lo = float(np.exp(delta_internal - z_crit * se) - 1)
                ci_hi = float(np.exp(delta_internal + z_crit * se) - 1)
            else:
                did_display = stats["effects"][i] - stats["effects"][j]
                ci_lo = delta_internal - z_crit * se
                ci_hi = delta_internal + z_crit * se

            pairwise.append(
                {
                    "segment_i": stats["names"][i],
                    "segment_j": stats["names"][j],
                    "did_estimate": did_display,
                    "se": se,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "raw_pvalue": raw_p,
                }
            )

        adj_pvals = adjust_pvalues(raw_pvals, method=correction) if raw_pvals else []
        for idx, pw in enumerate(pairwise):
            pw["adjusted_pvalue"] = adj_pvals[idx]

        self.pairwise_results = pairwise

        return self._format_output(stats, lift, alpha, correction)

    def _format_output(
        self,
        stats: dict[str, Any],
        lift: str,
        alpha: float,
        correction: str,
    ) -> str:
        """Build the tabulate output string."""

        def fmt(v: float) -> str | float:
            return convert_to_tabulate_str(v, lift)

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        seg_headers = ["Segment", "Control", "Treatment", "Lift", "CI Lower **", "CI Upper **"]
        seg_rows = []
        for i, name in enumerate(stats["names"]):
            seg_rows.append(
                [
                    name,
                    fmt_rate(stats["p_controls"][i]),
                    fmt_rate(stats["p_treatments"][i]),
                    fmt(stats["effects"][i]),
                    fmt(stats["ci_lowers"][i]),
                    fmt(stats["ci_uppers"][i]),
                ]
            )
        seg_table = tabulate(seg_rows, headers=seg_headers, tablefmt="grid")

        assert self.heterogeneity_results is not None
        q_line = (
            f"\nCochran's Q: {self.heterogeneity_results['Q_statistic']:.4f}, "
            f"df={self.heterogeneity_results['df']}, "
            f"p={self.heterogeneity_results['Q_pvalue']:.4f}"
        )

        assert self.pairwise_results is not None
        pw_headers = ["Comparison", "DiD", "CI Lower **", "CI Upper **", "p-value", f"Adj. p ({correction})"]
        pw_rows = []
        for pw in self.pairwise_results:
            star = " *" if pw["adjusted_pvalue"] < alpha else ""
            pw_rows.append(
                [
                    f"{pw['segment_i']} vs {pw['segment_j']}",
                    fmt(pw["did_estimate"]),
                    fmt(pw["ci_lower"]),
                    fmt(pw["ci_upper"]),
                    f"{pw['raw_pvalue']:.4f}",
                    f"{pw['adjusted_pvalue']:.4f}{star}",
                ]
            )
        pw_table = tabulate(pw_rows, headers=pw_headers, tablefmt="grid")

        ci_pct = int((1 - alpha) * 100)
        footer = f"\n* significant at alpha={alpha}; ** {ci_pct}% Confidence Interval"

        return f"{seg_table}{q_line}\n\n{pw_table}{footer}"

    def plot(
        self,
        lift: str = "absolute",
        alpha: float = 0.05,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Forest plot of per-segment treatment effects.

        Parameters
        ----------
        lift : str, default="absolute"
            Scale for treatment effects: ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"roas"``, or ``"revenue"``.
        alpha : float, default=0.05
            Significance level for confidence intervals.
        reverse_plot : bool, default=True
            If ``True``, the first segment appears at the top of the
            plot.
        color : str, dict, list, or None, default=None
            Colorblind palette name, mapping of segment names to colors,
            list of colors, or ``None`` for Plotly defaults.
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        stats = _compute_segment_stats(self._tables, lift, alpha)
        resolved = resolve_plot_color(color)

        names = stats["names"]
        effects = stats["effects"]
        ci_lowers = stats["ci_lowers"]
        ci_uppers = stats["ci_uppers"]

        if reverse_plot:
            names = names[::-1]
            effects = effects[::-1]
            ci_lowers = ci_lowers[::-1]
            ci_uppers = ci_uppers[::-1]

        fig = go.Figure()

        for i, name in enumerate(names):
            marker_color = None
            if isinstance(resolved, list):
                marker_color = resolved[i % len(resolved)]
            elif isinstance(resolved, dict):
                marker_color = resolved.get(name)

            fig.add_trace(
                go.Scatter(
                    x=[effects[i]],
                    y=[name],
                    mode="markers",
                    marker={"size": 10, "symbol": "circle", "color": marker_color},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_uppers[i] - effects[i]],
                        "arrayminus": [effects[i] - ci_lowers[i]],
                    },
                    name=name,
                    showlegend=False,
                )
            )

        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        lift_labels = {
            "absolute": "Risk Difference",
            "relative": "Relative Lift",
            "incremental": "Incremental Conversions",
            "roas": "Return on Ad Spend",
            "revenue": "Revenue",
        }
        tick_formats = {
            "absolute": ",.1%",
            "relative": ",.1%",
            "incremental": ",",
            "roas": "$,",
            "revenue": "$,",
        }
        lift_label = lift_labels[lift]
        fig.update_layout(
            title=f"{self.metric_name} — Treatment Effect by Segment ({lift_label})",
            xaxis_title=lift_label,
            xaxis_tickformat=tick_formats[lift],
            yaxis_title="",
            template="plotly_white",
        )

        fig.show()
