"""Bayesian difference-in-differences analysis for heterogeneous treatment effects.

Compares treatment effects across independent segments using posterior
sampling to determine whether the effect of an intervention varies by
subgroup.  Accepts multiple
:class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`
objects (one per segment) and produces:

1. Per-segment treatment effects with credible intervals
2. A posterior estimate of between-segment heterogeneity (tau)
3. All pairwise DiD comparisons with posterior probabilities
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from tabulate import tabulate

from ab_test._display import convert_to_tabulate_str, resolve_plot_color
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.credible_intervals import calculate_hdi_from_samples
from ab_test.bayesian_binomial.utils import posterior_mean, sample_beta

__all__ = [
    "BayesianDiffInDiff",
]

_VALID_LIFTS = frozenset({"absolute", "relative", "incremental", "roas", "revenue", "cpa"})


def _credible_interval_from_samples(
    samples: np.ndarray[Any, Any],
    confidence_level: float,
    method: str,
) -> tuple[float, float]:
    if method == "credible":
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q
        lo, hi = np.percentile(samples, [lower_q * 100, upper_q * 100])
        return float(lo), float(hi)
    return calculate_hdi_from_samples(samples, confidence_level)


def _compute_segment_samples(
    tables: list[BayesianContingencyTable],
    lift: str,
    n_samples: int,
) -> dict[str, Any]:
    """Draw posterior samples and compute per-segment lift distributions.

    Parameters
    ----------
    tables : list[BayesianContingencyTable]
        One table per segment, each with exactly 2 cells.
    lift : str
        One of ``"absolute"``, ``"relative"``, ``"incremental"``,
        ``"roas"``, ``"revenue"``, or ``"cpa"``.
    n_samples : int
        Number of posterior samples to draw per variant.

    Returns
    -------
    dict
        Keys: ``names``, ``lift_samples`` (list of arrays),
        ``p_controls``, ``p_treatments``.
    """
    names: list[str] = []
    all_lift_samples: list[np.ndarray[Any, Any]] = []
    p_controls: list[float] = []
    p_treatments: list[float] = []

    for table in tables:
        s_c, s_t = table.successes[0], table.successes[1]
        n_c, n_t = table.trials[0], table.trials[1]
        alpha_c, alpha_t = table.alphas[0], table.alphas[1]
        beta_c, beta_t = table.betas[0], table.betas[1]

        samples_c = sample_beta(s_c, n_c, alpha_c, beta_c, n_samples)
        samples_t = sample_beta(s_t, n_t, alpha_t, beta_t, n_samples)

        names.append(table.experiment_name)
        p_controls.append(posterior_mean(s_c, n_c, alpha_c, beta_c))
        p_treatments.append(posterior_mean(s_t, n_t, alpha_t, beta_t))

        if lift == "relative":
            safe_c = np.where(samples_c == 0, 1e-9, samples_c)
            segment_lift = (samples_t - samples_c) / safe_c
        elif lift in ("incremental", "roas", "revenue", "cpa"):
            n_max = max(n_c, n_t)
            segment_lift = (samples_t - samples_c) * n_max
            if lift == "roas":
                if table.spend is None:
                    raise ValueError(f"spend must be set on segment {table.experiment_name!r} for ROAS")
                segment_lift = segment_lift / table.spend
            elif lift == "cpa":
                if table.spend is None:
                    raise ValueError(f"spend must be set on segment {table.experiment_name!r} for CPA")
                segment_lift = np.where(np.abs(segment_lift) > 1e-12, table.spend / segment_lift, np.inf)
            elif lift == "revenue":
                if table.msrp is None:
                    raise ValueError(f"msrp must be set on segment {table.experiment_name!r} for revenue")
                segment_lift = segment_lift * table.msrp
        else:
            segment_lift = samples_t - samples_c

        all_lift_samples.append(segment_lift)

    return {
        "names": names,
        "lift_samples": all_lift_samples,
        "p_controls": p_controls,
        "p_treatments": p_treatments,
    }


class BayesianDiffInDiff:
    """Bayesian difference-in-differences analysis across independent segments.

    Accepts two or more
    :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`
    objects, each representing an independent segment (e.g. men, women),
    and uses posterior sampling to test whether the treatment effect
    differs across segments.

    Parameters
    ----------
    *tables : BayesianContingencyTable
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
    >>> from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
    >>> from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff
    >>> men = BayesianContingencyTable("Men", "converted")
    >>> _ = men.add("Control", successes=100, trials=1000, alpha=1, beta=1)
    >>> _ = men.add("Treatment", successes=130, trials=1000, alpha=1, beta=1)
    >>> women = BayesianContingencyTable("Women", "converted")
    >>> _ = women.add("Control", successes=120, trials=1000, alpha=1, beta=1)
    >>> _ = women.add("Treatment", successes=125, trials=1000, alpha=1, beta=1)
    >>> test = BayesianDiffInDiff(men, women)
    >>> result = test.analyze(lift="absolute")
    """

    def __init__(self, *tables: BayesianContingencyTable) -> None:
        if len(tables) < 2:
            raise ValueError(f"BayesianDiffInDiff requires at least 2 segments, got {len(tables)}")
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

        self._tables: list[BayesianContingencyTable] = list(tables)
        self.segment_names: list[str] = seg_names
        self.metric_name: str = tables[0].metric_name
        self.segment_results: dict[str, dict[str, Any]] | None = None
        self.heterogeneity_results: dict[str, Any] | None = None
        self.pairwise_results: list[dict[str, Any]] | None = None

    def analyze(
        self,
        lift: str = "absolute",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
    ) -> str:
        """Run the Bayesian difference-in-differences analysis.

        Parameters
        ----------
        lift : str, default="absolute"
            Scale for treatment effects: ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"roas"``, or ``"revenue"``.
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples to draw per variant.
        cred_int_method : {"credible", "hdi"}, default="credible"
            ``"credible"`` uses equal-tailed percentiles; ``"hdi"`` uses
            the Highest Density Interval.

        Returns
        -------
        str
            Formatted table with per-segment effects, heterogeneity
            estimate, and pairwise DiD comparisons.
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        stats = _compute_segment_samples(self._tables, lift, n_samples)
        lift_samples = stats["lift_samples"]

        self.segment_results = {}
        for i, name in enumerate(stats["names"]):
            samples = lift_samples[i]
            effect = float(np.mean(samples))
            ci_lo, ci_hi = _credible_interval_from_samples(samples, confidence_level, cred_int_method)
            prob_t_gt_c = float(np.mean(samples > 0))
            self.segment_results[name] = {
                "effect": effect,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "prob_t_gt_c": prob_t_gt_c,
                "p_control": stats["p_controls"][i],
                "p_treatment": stats["p_treatments"][i],
            }

        stacked = np.vstack(lift_samples)
        tau_samples = np.std(stacked, axis=0, ddof=0)
        tau_mean = float(np.mean(tau_samples))
        tau_ci_lo, tau_ci_hi = _credible_interval_from_samples(tau_samples, confidence_level, cred_int_method)
        self.heterogeneity_results = {
            "tau_mean": tau_mean,
            "tau_ci_lower": tau_ci_lo,
            "tau_ci_upper": tau_ci_hi,
        }

        k = len(stats["names"])
        pairs = list(itertools.combinations(range(k), 2))
        pairwise: list[dict[str, Any]] = []
        for i, j in pairs:
            did_samples = lift_samples[i] - lift_samples[j]
            did_mean = float(np.mean(did_samples))
            did_ci_lo, did_ci_hi = _credible_interval_from_samples(did_samples, confidence_level, cred_int_method)
            prob_i_gt_j = float(np.mean(did_samples > 0))
            pairwise.append(
                {
                    "segment_i": stats["names"][i],
                    "segment_j": stats["names"][j],
                    "did_estimate": did_mean,
                    "ci_lower": did_ci_lo,
                    "ci_upper": did_ci_hi,
                    "prob_i_gt_j": prob_i_gt_j,
                }
            )
        self.pairwise_results = pairwise

        return self._format_output(stats, lift, confidence_level)

    def _format_output(
        self,
        stats: dict[str, Any],
        lift: str,
        confidence_level: float,
    ) -> str:
        """Build the tabulate output string."""

        def fmt(v: float) -> str | float:
            return convert_to_tabulate_str(v, lift)

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        assert self.segment_results is not None
        seg_headers = ["Segment", "Control", "Treatment", "Lift", "CI Lower **", "CI Upper **", "P(T > C)"]
        seg_rows = []
        for name in stats["names"]:
            r = self.segment_results[name]
            seg_rows.append(
                [
                    name,
                    fmt_rate(r["p_control"]),
                    fmt_rate(r["p_treatment"]),
                    fmt(r["effect"]),
                    fmt(r["ci_lower"]),
                    fmt(r["ci_upper"]),
                    f"{r['prob_t_gt_c']:.4f}",
                ]
            )
        seg_table = tabulate(seg_rows, headers=seg_headers, tablefmt="grid")

        assert self.heterogeneity_results is not None
        het = self.heterogeneity_results
        het_line = (
            f"\nBetween-segment tau: {fmt(het['tau_mean'])} ({fmt(het['tau_ci_lower'])}, {fmt(het['tau_ci_upper'])})"
        )

        assert self.pairwise_results is not None
        pw_headers = ["Comparison", "DiD", "CI Lower **", "CI Upper **", "P(i > j)"]
        pw_rows = []
        for pw in self.pairwise_results:
            pw_rows.append(
                [
                    f"{pw['segment_i']} vs {pw['segment_j']}",
                    fmt(pw["did_estimate"]),
                    fmt(pw["ci_lower"]),
                    fmt(pw["ci_upper"]),
                    f"{pw['prob_i_gt_j']:.4f}",
                ]
            )
        pw_table = tabulate(pw_rows, headers=pw_headers, tablefmt="grid")

        ci_pct = int(confidence_level * 100)
        footer = f"\n** {ci_pct}% Credible Interval"

        return f"{seg_table}{het_line}\n\n{pw_table}{footer}"

    def plot(
        self,
        lift: str = "absolute",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Forest plot of per-segment treatment effects.

        Parameters
        ----------
        lift : str, default="absolute"
            Scale for treatment effects: ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"roas"``, or ``"revenue"``.
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples to draw per variant.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Method for computing credible intervals.
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

        stats = _compute_segment_samples(self._tables, lift, n_samples)
        resolved = resolve_plot_color(color)
        lift_samples = stats["lift_samples"]

        names = stats["names"]
        effects: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []
        for samples in lift_samples:
            effects.append(float(np.mean(samples)))
            lo, hi = _credible_interval_from_samples(samples, confidence_level, cred_int_method)
            ci_lowers.append(lo)
            ci_uppers.append(hi)

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
            "cpa": "Cost Per Acquisition",
        }
        tick_formats = {
            "absolute": ",.1%",
            "relative": ",.1%",
            "incremental": ",",
            "roas": "$,",
            "revenue": "$,",
            "cpa": "$,",
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
