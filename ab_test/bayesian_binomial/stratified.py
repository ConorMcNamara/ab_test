"""Bayesian stratified analysis for binomial A/B tests.

Provides a :class:`BayesianStratifiedContingencyTable` that mirrors the
frequentist :class:`~ab_test.frequentist_binomial.stratified.StratifiedContingencyTable`
API but uses Beta-Binomial conjugate models to produce posterior
distributions, credible intervals, P(T > C), expected loss, and ROPE
probabilities instead of p-values and confidence intervals.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from tabulate import tabulate

from ab_test._display import (
    apply_dark_mode,
    combine_lift_panels,
    convert_to_tabulate_str,
    format_percent,
    resolve_plot_color,
    tabulate_summary,
)
from ab_test._lift import scale_bounds, scale_metric
from ab_test.bayesian_binomial.credible_intervals import calculate_hdi_from_samples
from ab_test.bayesian_binomial.utils import (
    _between_group_sd_samples,
    _default_rope_half_width,
    posterior_mean,
    sample_beta,
)

__all__ = [
    "BayesianStratifiedContingencyTable",
]

_VALID_LIFTS = frozenset({"absolute", "relative", "incremental", "roas", "revenue", "cpa"})


class BayesianStratifiedContingencyTable:
    """Bayesian stratified analysis of a two-group binomial A/B test.

    Collects per-stratum 2x2 tables with Beta priors via :meth:`add` and
    produces a pooled Bayesian analysis using inverse-variance weighted
    posterior samples via :meth:`analyze`.

    Parameters
    ----------
    name : str
        Experiment name.
    metric_name : str
        Metric being measured.
    spend : float or None
        Campaign spend (required for ``lift="roas"``).
    msrp : float or None
        Average product price (required for ``lift="revenue"``).

    Examples
    --------
    >>> table = BayesianStratifiedContingencyTable("Checkout test", "conversion")
    >>> table = (
    ...     table.add("Control", 120, 1000, 1, 1, stratum="desktop")
    ...     .add("Treatment", 140, 1000, 1, 1, stratum="desktop")
    ...     .add("Control", 80, 1200, 1, 1, stratum="mobile")
    ...     .add("Treatment", 95, 1150, 1, 1, stratum="mobile")
    ... )
    >>> results = table.analyze(lift="relative")
    """

    def __init__(
        self,
        name: str,
        metric_name: str,
        spend: float | None = None,
        msrp: float | None = None,
    ) -> None:
        self.experiment_name: str = name
        self.metric_name: str = metric_name
        self.spend: float | None = spend
        self.msrp: float | None = msrp
        self._strata: dict[str, dict[str, dict[str, int | float]]] = {}
        self._cell_names: list[str] = []
        self.pooled_results: dict[str, Any] | None = None
        self.stratum_results: dict[str, dict[str, Any]] | None = None
        self.heterogeneity_results: dict[str, Any] | None = None

    def add(
        self,
        cell_name: str,
        successes: int,
        trials: int,
        alpha: float,
        beta: float,
        *,
        stratum: str,
    ) -> Self:
        """Add a cell to the stratified contingency table.

        Parameters
        ----------
        cell_name : str
            Experimental group name (e.g. ``"Control"``, ``"Treatment"``).
        successes : int
            Number of successes.
        trials : int
            Number of trials.
        alpha : float
            Alpha parameter of the Beta prior.
        beta : float
            Beta parameter of the Beta prior.
        stratum : str
            Stratum this observation belongs to.

        Returns
        -------
        BayesianStratifiedContingencyTable
            Self, for method chaining.
        """
        if cell_name not in self._cell_names:
            if len(self._cell_names) >= 2:
                raise ValueError(f"Only 2 groups are supported, got third group {cell_name!r}")
            self._cell_names.append(cell_name)

        if stratum not in self._strata:
            self._strata[stratum] = {}
        if cell_name in self._strata[stratum]:
            raise ValueError(f"Stratum {stratum!r} already has data for {cell_name!r}")

        self._strata[stratum][cell_name] = {
            "successes": successes,
            "trials": trials,
            "alpha": alpha,
            "beta": beta,
        }
        return self

    def _build_arrays(
        self,
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        list[str],
    ]:
        """Return per-stratum arrays and stratum names.

        Returns
        -------
        successes : ndarray, shape (K, 2)
        trials : ndarray, shape (K, 2)
        alphas : ndarray, shape (K, 2)
        betas : ndarray, shape (K, 2)
        strata_names : list[str]
        """
        if len(self._cell_names) != 2:
            raise ValueError(f"analyze requires exactly 2 groups, got {len(self._cell_names)}")
        strata_names = list(self._strata.keys())
        K = len(strata_names)
        successes = np.empty((K, 2), dtype=float)
        trials = np.empty((K, 2), dtype=float)
        alphas = np.empty((K, 2), dtype=float)
        betas = np.empty((K, 2), dtype=float)
        for k, s_name in enumerate(strata_names):
            stratum = self._strata[s_name]
            for j, c_name in enumerate(self._cell_names):
                if c_name not in stratum:
                    raise ValueError(f"Stratum {s_name!r} is missing group {c_name!r}")
                cell = stratum[c_name]
                successes[k, j] = cell["successes"]
                trials[k, j] = cell["trials"]
                alphas[k, j] = cell["alpha"]
                betas[k, j] = cell["beta"]
        return successes, trials, alphas, betas, strata_names

    def _validate_lift(self, lift: str) -> str:
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")
        if lift in ("roas", "cpa") and self.spend is None:
            raise ValueError(f"spend must be set for {lift.upper()} calculations")
        if lift == "revenue" and self.msrp is None:
            raise ValueError("msrp must be set for revenue calculations")
        return lift

    def _draw_stratum_samples(
        self,
        successes: np.ndarray[Any, Any],
        trials: np.ndarray[Any, Any],
        alphas: np.ndarray[Any, Any],
        betas: np.ndarray[Any, Any],
        lift: str,
        n_samples: int,
    ) -> list[np.ndarray[Any, Any]]:
        """Draw posterior samples and compute per-stratum lift on the analysis scale.

        For absolute/incremental/roas/revenue: returns risk-difference samples.
        For relative: returns log-RR samples (pooling happens on this scale).
        """
        K = successes.shape[0]
        all_lift_samples: list[np.ndarray[Any, Any]] = []

        for k in range(K):
            samples_c = sample_beta(int(successes[k, 0]), int(trials[k, 0]), alphas[k, 0], betas[k, 0], n_samples)
            samples_t = sample_beta(int(successes[k, 1]), int(trials[k, 1]), alphas[k, 1], betas[k, 1], n_samples)

            if lift == "relative":
                safe_c = np.where(samples_c == 0, 1e-9, samples_c)
                safe_t = np.where(samples_t == 0, 1e-9, samples_t)
                all_lift_samples.append(np.log(safe_t / safe_c))
            else:
                all_lift_samples.append(samples_t - samples_c)

        return all_lift_samples

    def _pool_samples(
        self,
        stratum_lift_samples: list[np.ndarray[Any, Any]],
        trials: np.ndarray[Any, Any],
        lift: str,
    ) -> np.ndarray[Any, Any]:
        """Inverse-variance weighted pooling of per-stratum lift samples.

        Returns pooled samples on the display scale, except for ``"cpa"``,
        which is returned in incremental conversions (see :meth:`_summarize`).
        """
        stacked = np.vstack(stratum_lift_samples)
        variances = np.var(stacked, axis=1)
        weights = 1.0 / variances
        weights_norm = weights / weights.sum()

        pooled = np.dot(weights_norm, stacked)

        if lift == "relative":
            return np.exp(pooled) - 1

        if lift in ("incremental", "roas", "revenue", "cpa"):
            n_max = float(max(np.sum(trials[:, 0]), np.sum(trials[:, 1])))
            pooled = scale_metric(pooled * n_max, "incremental" if lift == "cpa" else lift, self.spend, self.msrp)

        return pooled

    def _stratum_display_samples(self, samples: np.ndarray[Any, Any], n_max: float, lift: str) -> np.ndarray[Any, Any]:
        """Per-stratum samples on the display scale (incremental conversions for ``"cpa"``)."""
        if lift == "relative":
            return np.exp(samples) - 1  # type: ignore[no-any-return]
        if lift in ("incremental", "roas", "revenue", "cpa"):
            return scale_metric(samples * n_max, "incremental" if lift == "cpa" else lift, self.spend, self.msrp)
        return samples

    def _summarize(
        self,
        samples: np.ndarray[Any, Any],
        lift: str,
        confidence_level: float,
        method: str,
    ) -> tuple[float, float, float, float]:
        """Point estimate, credible interval and P(T > C) on the display scale.

        For ``"cpa"`` the samples are incremental conversions. CPA is
        ``spend / incremental``, which has no posterior mean and is not
        monotone through zero, so the summary is computed on the incremental
        scale and then transformed: the point estimate is ``spend`` over the
        mean increment, and the interval bounds are the transformed quantiles.
        A mean increment of zero or less buys no conversions, so its CPA is
        infinite, matching how the interval bounds treat non-positive increments.
        """
        lo, hi = self._credible_interval(samples, confidence_level, method)
        prob_t_gt_c = float(np.mean(samples > 0))
        mean = float(np.mean(samples))
        if lift == "cpa":
            cpa_lo, cpa_hi = scale_bounds(lo, hi, "cpa", self.spend)
            cpa = float(scale_metric(mean, "cpa", self.spend)) if mean > 0 else np.inf
            return cpa, cpa_lo, cpa_hi, prob_t_gt_c
        return mean, lo, hi, prob_t_gt_c

    @staticmethod
    def _credible_interval(
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

    def analyze(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> str:
        """Analyze the stratified experiment with Bayesian pooling.

        Computes a pooled treatment effect via inverse-variance weighted
        posterior samples, along with credible intervals, P(T > C),
        expected loss, ROPE probabilities, and a heterogeneity diagnostic:
        the posterior of the between-stratum standard deviation of the true
        effects (tau) from a normal random-effects model, which does not count
        within-stratum noise as heterogeneity. For relative lift, tau is on
        the log risk-ratio scale. It is not reported for ``"cpa"``. The
        expected loss, E[max(-lift, 0)], is in the units of ``lift``; for
        ``"cpa"`` it is a difference in rates.

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
        confidence_level : float, default=0.95
            Probability mass for the credible interval.
        n_samples : int, default=100_000
            Number of posterior samples to draw per variant.
        cred_int_method : {"credible", "hdi"}, default="credible"
            ``"credible"`` uses equal-tailed percentiles; ``"hdi"`` uses
            the Highest Density Interval.
        low_threshold : float, optional
            Lower bound of the Region of Practical Equivalence (ROPE), in the
            units of ``lift``. By default the ROPE is +/-10% of the pooled
            control's posterior rate expressed in those units, as in
            :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`,
            so for relative lift it is +/-0.1. The ROPE is not computed for
            ``lift="cpa"`` (shown as n/a).
        high_threshold : float, optional
            Upper bound of the ROPE, in the units of ``lift``. Defaults as
            described for ``low_threshold``.

        Returns
        -------
        str
            Formatted results table.
        """
        lift = self._validate_lift(lift)
        successes, trials, alphas, betas, strata_names = self._build_arrays()

        stratum_samples = self._draw_stratum_samples(successes, trials, alphas, betas, lift, n_samples)
        pooled_samples = self._pool_samples(stratum_samples, trials, lift)

        pooled_mean, ci_lo, ci_hi, prob_t_gt_c = self._summarize(
            pooled_samples, lift, confidence_level, cred_int_method
        )
        p_control = float(
            np.sum(
                [
                    posterior_mean(int(successes[k, 0]), int(trials[k, 0]), alphas[k, 0], betas[k, 0]) * trials[k, 0]
                    for k in range(len(strata_names))
                ]
            )
            / np.sum(trials[:, 0])
        )
        p_treatment = float(
            np.sum(
                [
                    posterior_mean(int(successes[k, 1]), int(trials[k, 1]), alphas[k, 1], betas[k, 1]) * trials[k, 1]
                    for k in range(len(strata_names))
                ]
            )
            / np.sum(trials[:, 1])
        )

        n_max = float(max(np.sum(trials[:, 0]), np.sum(trials[:, 1])))
        if lift == "cpa":
            # Loss and ROPE are not meaningful on the CPA scale; report the loss as a rate difference.
            expected_loss = float(np.mean(np.maximum(-pooled_samples / n_max, 0)))
            prob_in_rope = float("nan")
        else:
            expected_loss = float(np.mean(np.maximum(-pooled_samples, 0)))
            if low_threshold is None or high_threshold is None:
                default_rope = _default_rope_half_width(p_control, lift, n_max, self.spend, self.msrp)
                low_threshold = -default_rope if low_threshold is None else low_threshold
                high_threshold = default_rope if high_threshold is None else high_threshold
            prob_in_rope = float(np.mean((pooled_samples >= low_threshold) & (pooled_samples <= high_threshold)))

        self.pooled_results = {
            "lift_type": lift,
            "lift": pooled_mean,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_control": p_control,
            "p_treatment": p_treatment,
            "prob_t_gt_c": prob_t_gt_c,
            "expected_loss": expected_loss,
            "prob_rope": prob_in_rope,
        }

        if lift == "cpa":
            tau_mean = tau_ci_lo = tau_ci_hi = float("nan")
        else:
            # Heterogeneity on the pooling scale (log risk ratio for relative lift), so stratum size does not
            # masquerade as a different effect; linear lifts are then rescaled like the pooled estimate.
            tau_samples = _between_group_sd_samples(
                [float(np.mean(s)) for s in stratum_samples], [float(np.var(s)) for s in stratum_samples], n_samples
            )
            if lift in ("incremental", "roas", "revenue"):
                n_max = float(max(np.sum(trials[:, 0]), np.sum(trials[:, 1])))
                tau_samples = scale_metric(tau_samples * n_max, lift, self.spend, self.msrp)
            tau_mean = float(np.mean(tau_samples))
            tau_ci_lo, tau_ci_hi = self._credible_interval(tau_samples, confidence_level, cred_int_method)
        self.heterogeneity_results = {
            "tau_mean": tau_mean,
            "tau_ci_lower": tau_ci_lo,
            "tau_ci_upper": tau_ci_hi,
        }

        return self._format_analyze(lift, confidence_level)

    def _format_analyze(self, lift: str, confidence_level: float) -> str:
        assert self.pooled_results is not None
        assert self.heterogeneity_results is not None
        r = self.pooled_results

        def fmt(v: float) -> str | float:
            return convert_to_tabulate_str(v, lift)

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        str_prob = (
            f"{convert_to_tabulate_str(r['prob_t_gt_c'], 'relative')}*"
            if r["prob_t_gt_c"] >= confidence_level
            else f"{convert_to_tabulate_str(r['prob_t_gt_c'], 'relative')}"
        )

        row_labels = (
            ["Metric", "Metric Name"]
            + self._cell_names
            + [
                "Lift",
                "Cred. Int. Lower **",
                "Cred. Int. Upper **",
                f"Prob {self._cell_names[1]} Is Best",
                f"Expected Loss of {self._cell_names[1]}" + (" (rate difference)" if lift == "cpa" else ""),
                "Probability Lift is in ROPE ***",
            ]
        )
        values = (
            [lift, self.metric_name]
            + [fmt_rate(r["p_control"]), fmt_rate(r["p_treatment"])]
            + [fmt(r["lift"]), fmt(r["ci_lower"]), fmt(r["ci_upper"])]
            + [str_prob]
            # The loss is on the lift's scale, except for CPA, where it is a rate difference.
            + [fmt_rate(r["expected_loss"]) if lift == "cpa" else fmt(r["expected_loss"])]
            + ["n/a" if np.isnan(r["prob_rope"]) else convert_to_tabulate_str(r["prob_rope"], "relative")]
        )
        return_string = tabulate_summary(row_labels, values)

        het = self.heterogeneity_results
        if np.isnan(het["tau_mean"]):
            return_string += "\nBetween-stratum tau: n/a"
        else:
            return_string += (
                f"\nBetween-stratum tau: {fmt(het['tau_mean'])} "
                f"({fmt(het['tau_ci_lower'])}, {fmt(het['tau_ci_upper'])})"
            )

        ci_pct = format_percent(confidence_level)
        return_string += f"\n* next to the prob means it exceeds our confidence level at {ci_pct}% level"
        return_string += f"\n** {ci_pct}% Credible Interval"
        return_string += "\n*** Region of Practical Equivalence"
        return return_string

    def analyze_by_stratum(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
    ) -> str:
        """Analyze each stratum individually using Bayesian posteriors.

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples to draw per variant.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Method for computing credible intervals.

        Returns
        -------
        str
            Table with per-stratum effect estimates, credible intervals,
            and P(T > C).
        """
        lift = self._validate_lift(lift)
        successes, trials, alphas, betas, strata_names = self._build_arrays()

        stratum_samples = self._draw_stratum_samples(successes, trials, alphas, betas, lift, n_samples)

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        self.stratum_results = {}
        table_list = []
        for k, s_name in enumerate(strata_names):
            display_samples = self._stratum_display_samples(stratum_samples[k], max(trials[k, 0], trials[k, 1]), lift)
            effect, ci_lo, ci_hi, prob_t_gt_c = self._summarize(
                display_samples, lift, confidence_level, cred_int_method
            )

            p_c = posterior_mean(int(successes[k, 0]), int(trials[k, 0]), alphas[k, 0], betas[k, 0])
            p_t = posterior_mean(int(successes[k, 1]), int(trials[k, 1]), alphas[k, 1], betas[k, 1])

            self.stratum_results[s_name] = {
                "effect": effect,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "prob_t_gt_c": prob_t_gt_c,
                "p_control": p_c,
                "p_treatment": p_t,
            }

            table_list.append(
                [s_name]
                + [fmt_rate(p_c), fmt_rate(p_t)]
                + convert_to_tabulate_str([effect, ci_lo, ci_hi], lift)
                + [f"{prob_t_gt_c:.4f}"]
                + [int(trials[k, 0]) + int(trials[k, 1])]
            )

        table_headers = ["Stratum"] + self._cell_names + ["Lift", "CI Lower **", "CI Upper **", "P(T > C)", "N"]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")
        return_string += f"\n** {format_percent(confidence_level)}% Credible Interval"
        return return_string

    def plot(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
        *,
        dark_mode: bool = False,
    ) -> None:
        """Forest plot of per-stratum and pooled treatment effects.

        Each stratum is shown as a circle with a credible-interval
        whisker. The pooled inverse-variance weighted estimate is shown
        as a diamond. A vertical dashed line marks zero (no effect).

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
            ``"both"`` draws absolute and relative lift side by side, sharing
            the y-axis, each with its own interval.
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples to draw per variant.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Method for computing credible intervals.
        reverse_plot : bool, default=True
            Whether to reverse the y-axis order (first stratum at top).
        color : str, list, dict, or None, default=None
            Colorblind palette name, mapping of stratum names to colors,
            list of colors, or ``None`` for Plotly defaults.
        dark_mode : bool, default=False
            Render on a dark background with light text and gridlines (Plotly's
            ``"plotly_dark"`` template).
        """
        if isinstance(lift, str) and lift.casefold() == "both":
            figures = [
                self._plot_figure(
                    lift=panel_lift,
                    confidence_level=confidence_level,
                    n_samples=n_samples,
                    cred_int_method=cred_int_method,
                    reverse_plot=reverse_plot,
                    color=color,
                )
                for panel_lift in ("absolute", "relative")
            ]
            fig = combine_lift_panels(
                figures,
                f"{self.experiment_name} — {self.metric_name} (Risk Difference and Relative Lift)",
                ["Risk Difference", "Relative Lift"],
            )
        else:
            fig = self._plot_figure(
                lift=lift,
                confidence_level=confidence_level,
                n_samples=n_samples,
                cred_int_method=cred_int_method,
                reverse_plot=reverse_plot,
                color=color,
            )
        apply_dark_mode(fig, dark_mode)
        fig.show()  # type: ignore[no-untyped-call]

    def _plot_figure(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> go.Figure:
        """Build the forest plot for a single lift (see :meth:`plot`)."""
        lift = self._validate_lift(lift)
        successes, trials, alphas, betas, strata_names = self._build_arrays()

        stratum_samples = self._draw_stratum_samples(successes, trials, alphas, betas, lift, n_samples)
        pooled_samples = self._pool_samples(stratum_samples, trials, lift)

        effects: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []
        for k, samples in enumerate(stratum_samples):
            display_samples = self._stratum_display_samples(samples, max(trials[k, 0], trials[k, 1]), lift)
            effect, lo, hi, _ = self._summarize(display_samples, lift, confidence_level, cred_int_method)
            effects.append(effect)
            ci_lowers.append(lo)
            ci_uppers.append(hi)

        pooled_est, pooled_lb, pooled_ub, _ = self._summarize(pooled_samples, lift, confidence_level, cred_int_method)

        plot_color = resolve_plot_color(color)
        fig = go.Figure()  # type: ignore[attr-defined]

        for k, s_name in enumerate(strata_names):
            c = None
            if plot_color is not None:
                if isinstance(plot_color, list):
                    c = plot_color[k % len(plot_color)]
                elif isinstance(plot_color, dict):
                    c = plot_color.get(s_name)

            marker_kw: dict[str, Any] = {"symbol": "circle", "size": 10}
            error_x_kw: dict[str, Any] = {
                "type": "data",
                "symmetric": False,
                "array": [ci_uppers[k] - effects[k]],
                "arrayminus": [effects[k] - ci_lowers[k]],
                "visible": True,
            }
            if c is not None:
                marker_kw["color"] = c
                error_x_kw["color"] = c

            fig.add_trace(
                go.Scatter(  # type: ignore[attr-defined]
                    x=[effects[k]],
                    y=[s_name],
                    marker=marker_kw,
                    error_x=error_x_kw,
                    name=s_name,
                )
            )

        c_pooled = None
        if plot_color is not None:
            if isinstance(plot_color, list):
                c_pooled = plot_color[len(strata_names) % len(plot_color)]
            elif isinstance(plot_color, dict):
                c_pooled = plot_color.get("Overall")

        marker_pooled: dict[str, Any] = {"symbol": "diamond", "size": 14}
        error_x_pooled: dict[str, Any] = {
            "type": "data",
            "symmetric": False,
            "array": [pooled_ub - pooled_est],
            "arrayminus": [pooled_est - pooled_lb],
            "visible": True,
        }
        if c_pooled is not None:
            marker_pooled["color"] = c_pooled
            error_x_pooled["color"] = c_pooled

        fig.add_trace(
            go.Scatter(  # type: ignore[attr-defined]
                x=[pooled_est],
                y=["Overall"],
                marker=marker_pooled,
                error_x=error_x_pooled,
                name="Overall (pooled)",
            )
        )

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

        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"{self.experiment_name} — {self.metric_name} ({lift_labels[lift]})",
            xaxis_tickformat=tick_formats[lift],
            showlegend=False,
        )
        if reverse_plot:
            fig.update_layout(yaxis={"autorange": "reversed"})
        return fig
