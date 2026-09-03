"""Stratified analysis for binomial A/B tests.

Provides the Cochran-Mantel-Haenszel (CMH) test for combining evidence
across strata, the Breslow-Day test for homogeneity of odds ratios, and
a :class:`StratifiedContingencyTable` that mirrors the
:class:`~ab_test.frequentist_binomial.contingency.ContingencyTable` API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as ss

import plotly.graph_objects as go  # type: ignore[import-untyped]

from ab_test._display import convert_to_tabulate_str, resolve_plot_color

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "StratifiedContingencyTable",
    "cmh_test",
    "breslow_day_test",
    "stratified_power",
]


def cmh_test(
    successes: np.ndarray[Any, Any] | list[list[int]],
    trials: np.ndarray[Any, Any] | list[list[int]],
) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel test for conditional independence.

    Tests whether the treatment effect is zero across all strata.

    Parameters
    ----------
    successes : array_like, shape (K, 2)
        ``successes[k, 0]`` and ``successes[k, 1]`` are the number of
        successes in the control and treatment groups for stratum *k*.
    trials : array_like, shape (K, 2)
        ``trials[k, 0]`` and ``trials[k, 1]`` are the number of trials
        in the control and treatment groups for stratum *k*.

    Returns
    -------
    statistic : float
        CMH chi-squared statistic.
    pvalue : float
        Two-sided p-value from a chi-squared(1) distribution.
    """
    successes_arr = np.asarray(successes, dtype=float)
    trials_arr = np.asarray(trials, dtype=float)

    a = successes_arr[:, 0]
    c = successes_arr[:, 1]
    n1 = trials_arr[:, 0]
    n2 = trials_arr[:, 1]
    m1 = a + c
    t = n1 + n2
    m0 = t - m1

    e_a = n1 * m1 / t
    var_a = n1 * n2 * m1 * m0 / (t**2 * (t - 1))

    chi2 = float((np.sum(a - e_a)) ** 2 / np.sum(var_a))
    pvalue = float(ss.chi2.sf(chi2, df=1))

    return chi2, pvalue


def _mh_odds_ratio(
    successes: np.ndarray[Any, Any],
    trials: np.ndarray[Any, Any],
) -> float:
    """Mantel-Haenszel common odds ratio estimate."""
    a = successes[:, 0]
    c = successes[:, 1]
    n1 = trials[:, 0]
    n2 = trials[:, 1]
    b = n1 - a
    d = n2 - c
    t = n1 + n2

    numerator = np.sum(a * d / t)
    denominator = np.sum(b * c / t)
    if denominator == 0:
        return float("inf")
    return float(numerator / denominator)


def breslow_day_test(
    successes: np.ndarray[Any, Any] | list[list[int]],
    trials: np.ndarray[Any, Any] | list[list[int]],
) -> tuple[float, float]:
    """Breslow-Day test for homogeneity of odds ratios across strata.

    Tests whether the stratum-specific odds ratios are all equal.
    A significant result suggests effect modification: the treatment
    effect varies meaningfully across strata.

    Parameters
    ----------
    successes : array_like, shape (K, 2)
        ``successes[k, 0]`` and ``successes[k, 1]`` are the number of
        successes in the control and treatment groups for stratum *k*.
    trials : array_like, shape (K, 2)
        ``trials[k, 0]`` and ``trials[k, 1]`` are the number of trials
        in the control and treatment groups for stratum *k*.

    Returns
    -------
    statistic : float
        Breslow-Day chi-squared statistic.
    pvalue : float
        P-value from a chi-squared(K - 1) distribution.

    Raises
    ------
    ValueError
        If fewer than 2 strata are provided.
    """
    successes_arr = np.asarray(successes, dtype=float)
    trials_arr = np.asarray(trials, dtype=float)
    K = successes_arr.shape[0]
    if K < 2:
        raise ValueError("Breslow-Day test requires at least 2 strata")

    or_mh = _mh_odds_ratio(successes_arr, trials_arr)

    a = successes_arr[:, 0]
    c = successes_arr[:, 1]
    n1 = trials_arr[:, 0]
    n2 = trials_arr[:, 1]
    m1 = a + c

    coef_a = 1.0 - or_mh
    coef_b = n2 - m1 + or_mh * (n1 + m1)
    coef_c = -or_mh * n1 * m1

    if abs(coef_a) < 1e-10:
        a_star = -coef_c / coef_b
    else:
        disc = coef_b**2 - 4 * coef_a * coef_c
        r1 = (-coef_b + np.sqrt(disc)) / (2 * coef_a)
        r2 = (-coef_b - np.sqrt(disc)) / (2 * coef_a)
        upper = np.minimum(n1, m1)
        a_star = np.where((r1 >= 0) & (r1 <= upper), r1, r2)

    b_star = n1 - a_star
    c_star = m1 - a_star
    d_star = n2 - c_star
    var_a = 1.0 / (1.0 / a_star + 1.0 / b_star + 1.0 / c_star + 1.0 / d_star)

    chi2 = float(np.sum((a - a_star) ** 2 / var_a))
    pvalue = float(ss.chi2.sf(chi2, df=K - 1))

    return chi2, pvalue


def stratified_power(
    strata_sizes: list[tuple[int, int]] | np.ndarray[Any, Any],
    baseline_rates: list[float] | float,
    alt_lift: float,
    alpha: float = 0.05,
    lift: str = "relative",
) -> float:
    """Power of the stratified test under a common treatment effect.

    Computes the probability of rejecting the null hypothesis when the
    true treatment effect is *alt_lift*, using the inverse-variance
    weighted estimator across strata.

    Parameters
    ----------
    strata_sizes : list of (int, int)
        ``(n_control, n_treatment)`` per stratum.
    baseline_rates : float or list of float
        Control-group success rate per stratum. If a single float, the
        same rate is used for all strata.
    alt_lift : float
        Assumed treatment effect (relative or absolute).
    alpha : float, default=0.05
        Significance level.
    lift : str, default='relative'
        ``'relative'`` or ``'absolute'``.

    Returns
    -------
    float
        Power (probability of rejecting H0).
    """
    strata_arr = np.asarray(strata_sizes, dtype=float)
    K = strata_arr.shape[0]

    if isinstance(baseline_rates, (int, float)):
        p1 = np.full(K, baseline_rates)
    else:
        p1 = np.asarray(baseline_rates, dtype=float)

    if lift == "relative":
        p2 = p1 * (1 + alt_lift)
    else:
        p2 = p1 + alt_lift

    n1 = strata_arr[:, 0]
    n2 = strata_arr[:, 1]

    if lift == "absolute":
        var_k = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        ncp = alt_lift**2 * float(np.sum(1 / var_k))
    else:
        var_log_rr_k = (1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)
        ncp = np.log(1 + alt_lift) ** 2 * float(np.sum(1 / var_log_rr_k))

    crit = float(ss.chi2.isf(alpha, df=1))
    return float(ss.ncx2.sf(crit, df=1, nc=ncp))


_VALID_LIFTS = frozenset({"absolute", "relative", "incremental", "roas", "revenue", "cpa"})


def _stratum_effect(
    p1: float,
    p2: float,
    n1: float,
    n2: float,
    lift: str,
    z: float,
    spend: float | None = None,
    msrp: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute a single stratum's effect, SE, and CI on the requested scale.

    Returns ``(effect, se, ci_lower, ci_upper)`` on the display scale.
    """
    if lift == "relative":
        log_rr = float(np.log(p2 / p1))
        se_log = float(np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)))
        effect = float(np.exp(log_rr) - 1)
        ci_lo = float(np.exp(log_rr - z * se_log) - 1)
        ci_hi = float(np.exp(log_rr + z * se_log) - 1)
        return effect, se_log, ci_lo, ci_hi

    d = p2 - p1
    var = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    se = float(np.sqrt(var))

    if lift in ("incremental", "roas", "revenue", "cpa"):
        n_max = max(n1, n2)
        d_scaled = d * n_max
        se_scaled = se * n_max
        ci_lo = d_scaled - z * se_scaled
        ci_hi = d_scaled + z * se_scaled
        if lift == "roas":
            assert spend is not None
            d_scaled /= spend
            se_scaled /= spend
            ci_lo /= spend
            ci_hi /= spend
        elif lift == "cpa":
            assert spend is not None
            se_scaled = np.inf
            d_scaled = spend / d_scaled if abs(d_scaled) > 1e-12 else np.inf
            ci_lo, ci_hi = (
                (spend / ci_hi if ci_hi > 0 else np.inf),
                (spend / ci_lo if ci_lo > 0 else np.inf),
            )
        elif lift == "revenue":
            assert msrp is not None
            d_scaled *= msrp
            se_scaled *= msrp
            ci_lo *= msrp
            ci_hi *= msrp
        return d_scaled, se_scaled, ci_lo, ci_hi

    return d, se, d - z * se, d + z * se


def _pooled_effect(
    p1: np.ndarray[Any, Any],
    p2: np.ndarray[Any, Any],
    trials_arr: np.ndarray[Any, Any],
    lift: str,
    z: float,
    spend: float | None = None,
    msrp: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute the inverse-variance pooled effect on the requested scale.

    Returns ``(estimate, se, ci_lower, ci_upper)`` on the display scale.
    """
    if lift == "relative":
        log_rr = np.log(p2 / p1)
        var_log_rr = (1 - p1) / (trials_arr[:, 0] * p1) + (1 - p2) / (trials_arr[:, 1] * p2)
        w = 1 / var_log_rr
        log_rr_pooled = float(np.sum(w * log_rr) / np.sum(w))
        se_log = float(1 / np.sqrt(np.sum(w)))
        estimate = float(np.exp(log_rr_pooled) - 1)
        lb = float(np.exp(log_rr_pooled - z * se_log) - 1)
        ub = float(np.exp(log_rr_pooled + z * se_log) - 1)
        return estimate, se_log, lb, ub

    rd = p2 - p1
    var_rd = p1 * (1 - p1) / trials_arr[:, 0] + p2 * (1 - p2) / trials_arr[:, 1]
    w = 1 / var_rd
    pooled_d = float(np.sum(w * rd) / np.sum(w))
    pooled_se = float(1 / np.sqrt(np.sum(w)))

    if lift in ("incremental", "roas", "revenue", "cpa"):
        n_max = float(max(np.sum(trials_arr[:, 0]), np.sum(trials_arr[:, 1])))
        est = pooled_d * n_max
        se_scaled = pooled_se * n_max
        lb = est - z * se_scaled
        ub = est + z * se_scaled
        if lift == "roas":
            assert spend is not None
            est /= spend
            se_scaled /= spend
            lb /= spend
            ub /= spend
        elif lift == "cpa":
            assert spend is not None
            se_scaled = np.inf
            est = spend / est if abs(est) > 1e-12 else np.inf
            lb, ub = (spend / ub if ub > 0 else np.inf), (spend / lb if lb > 0 else np.inf)
        elif lift == "revenue":
            assert msrp is not None
            est *= msrp
            se_scaled *= msrp
            lb *= msrp
            ub *= msrp
        return est, se_scaled, lb, ub

    lb = pooled_d - z * pooled_se
    ub = pooled_d + z * pooled_se
    return pooled_d, pooled_se, lb, ub


class StratifiedContingencyTable:
    """Stratified analysis of a two-group binomial A/B test.

    Collects per-stratum 2x2 tables via :meth:`add` and produces a
    pooled analysis using the Cochran-Mantel-Haenszel framework via
    :meth:`analyze`.

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
        self._strata: dict[str, dict[str, dict[str, int]]] = {}
        self._cell_names: list[str] = []

    def add(
        self,
        cell_name: str,
        successes: int,
        trials: int,
        *,
        stratum: str,
    ) -> StratifiedContingencyTable:
        """Add a cell to the stratified contingency table.

        Parameters
        ----------
        cell_name : str
            Experimental group name (e.g. ``"Control"``, ``"Treatment"``).
        successes : int
            Number of successes.
        trials : int
            Number of trials.
        stratum : str
            Stratum this observation belongs to.

        Returns
        -------
        StratifiedContingencyTable
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

        self._strata[stratum][cell_name] = {"successes": successes, "trials": trials}
        return self

    def _build_arrays(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
        """Return ``(successes, trials)`` arrays of shape ``(K, 2)`` and stratum names."""
        if len(self._cell_names) != 2:
            raise ValueError(f"analyze requires exactly 2 groups, got {len(self._cell_names)}")
        strata_names = list(self._strata.keys())
        K = len(strata_names)
        successes = np.empty((K, 2), dtype=float)
        trials = np.empty((K, 2), dtype=float)
        for k, s_name in enumerate(strata_names):
            stratum = self._strata[s_name]
            for j, c_name in enumerate(self._cell_names):
                if c_name not in stratum:
                    raise ValueError(f"Stratum {s_name!r} is missing group {c_name!r}")
                successes[k, j] = stratum[c_name]["successes"]
                trials[k, j] = stratum[c_name]["trials"]
        return successes, trials, strata_names

    def _validate_lift(self, lift: str) -> str:
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")
        if lift == "roas" and self.spend is None:
            raise ValueError("spend must be set for ROAS calculations")
        if lift == "cpa" and self.spend is None:
            raise ValueError("spend must be set for CPA calculations")
        if lift == "revenue" and self.msrp is None:
            raise ValueError("msrp must be set for revenue calculations")
        return lift

    def analyze(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
    ) -> str:
        """Analyze the stratified experiment.

        Computes the Cochran-Mantel-Haenszel p-value, a pooled effect
        estimate via inverse-variance weighting, and a Wald confidence
        interval. Also reports the Breslow-Day homogeneity p-value when
        there are at least two strata.

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
        alpha : float, default=0.05
            Significance level for the confidence interval.

        Returns
        -------
        str
            Formatted results table.
        """
        lift = self._validate_lift(lift)

        successes, trials_arr, strata_names = self._build_arrays()

        _, p_value = cmh_test(successes, trials_arr)

        p1 = successes[:, 0] / trials_arr[:, 0]
        p2 = successes[:, 1] / trials_arr[:, 1]
        z = float(ss.norm.ppf(1 - alpha / 2))

        estimate, _, lb, ub = _pooled_effect(p1, p2, trials_arr, lift, z, self.spend, self.msrp)

        p_control = float(np.sum(successes[:, 0]) / np.sum(trials_arr[:, 0]))
        p_treatment = float(np.sum(successes[:, 1]) / np.sum(trials_arr[:, 1]))

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        success_rate: list[str | float] = [fmt_rate(p_control), fmt_rate(p_treatment)]
        str_pvalue = f"{p_value}" if p_value >= alpha else f"{p_value}*"
        table_headers = (
            ["Metric", "Metric Name"]
            + self._cell_names
            + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value (CMH)"]
        )
        table_list = [
            [lift, self.metric_name] + success_rate + convert_to_tabulate_str([estimate, lb, ub], lift) + [str_pvalue]
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")

        if len(strata_names) >= 2:
            _, bd_pvalue = breslow_day_test(successes, trials_arr)
            return_string += f"\nBreslow-Day homogeneity p-value: {bd_pvalue:.4f}"

        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def analyze_by_stratum(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
    ) -> str:
        """Analyze each stratum individually.

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        str
            Table with per-stratum effect estimates and confidence
            intervals.
        """
        lift = self._validate_lift(lift)
        successes, trials_arr, strata_names = self._build_arrays()

        p1 = successes[:, 0] / trials_arr[:, 0]
        p2 = successes[:, 1] / trials_arr[:, 1]
        z = float(ss.norm.ppf(1 - alpha / 2))

        def fmt_rate(v: float) -> str | float:
            return convert_to_tabulate_str(v, "absolute")

        table_list = []
        for k, s_name in enumerate(strata_names):
            effect, _, lb, ub = _stratum_effect(
                p1[k], p2[k], trials_arr[k, 0], trials_arr[k, 1], lift, z, self.spend, self.msrp
            )
            table_list.append(
                [s_name]
                + [fmt_rate(p1[k]), fmt_rate(p2[k])]
                + convert_to_tabulate_str([effect, lb, ub], lift)
                + [int(trials_arr[k, 0]) + int(trials_arr[k, 1])]
            )

        table_headers = ["Stratum"] + self._cell_names + ["Lift", "CI Lower **", "CI Upper **", "N"]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def plot(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Forest plot of per-stratum and pooled treatment effects.

        Each stratum is shown as a circle with a confidence-interval
        whisker. The pooled inverse-variance weighted estimate is shown
        as a diamond. A vertical dashed line marks zero (no effect).

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"``, ``"absolute"``, ``"incremental"``,
            ``"roas"``, ``"revenue"``, or ``"cpa"``.
        alpha : float, default=0.05
            Significance level for confidence intervals.
        reverse_plot : bool, default=True
            Whether to reverse the y-axis order (first stratum at top).
        color : str, list, dict, or None, default=None
            If ``None``, uses Plotly's default color scheme.
            If a string, one of the colorblind-friendly palette names
            (see :func:`~ab_test._display.resolve_plot_color`).
            If a list, each item is a color for the corresponding
            stratum (last entry is used for the pooled row).
            If a dict, keys are stratum names (use ``"Overall"`` for
            the pooled row).
        """
        lift = self._validate_lift(lift)
        successes, trials_arr, strata_names = self._build_arrays()

        p1 = successes[:, 0] / trials_arr[:, 0]
        p2 = successes[:, 1] / trials_arr[:, 1]
        z = float(ss.norm.ppf(1 - alpha / 2))

        effects: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []
        for k in range(len(strata_names)):
            effect, _, lb, ub = _stratum_effect(
                p1[k], p2[k], trials_arr[k, 0], trials_arr[k, 1], lift, z, self.spend, self.msrp
            )
            effects.append(effect)
            ci_lowers.append(lb)
            ci_uppers.append(ub)

        pooled_est, _, pooled_lb, pooled_ub = _pooled_effect(p1, p2, trials_arr, lift, z, self.spend, self.msrp)

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
        fig.show()  # type: ignore[no-untyped-call]
