"""Cluster-randomized trial analysis and design.

Provides tools for analyzing experiments where randomization occurs at
the cluster level (e.g., stores, markets, time windows) rather than the
individual level.  Intra-cluster correlation inflates variance and must
be accounted for in both inference and sample-size planning.

The analysis uses a cluster-summary Welch t-test on per-cluster
proportions, which is the standard approach for cluster-randomized
trials with binary outcomes (Donner & Klar, 2000).  Power and
sample-size calculations use the design-effect adjustment to deflate
effective sample sizes, integrating with the existing pluggable power
framework via :func:`cluster_adjusted_power`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import scipy.stats as ss

from ab_test._display import convert_to_tabulate_str, resolve_plot_color
from ab_test.frequentist_binomial.randomization_inference import cluster_randomization_test
from ab_test.frequentist_binomial.power_calculations import (
    abtest_power,
    minimum_detectable_lift,
    required_sample_size,
    score_power,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ClusterRandomizedTrial",
    "estimate_icc",
    "design_effect",
    "cluster_adjusted_power",
    "cluster_required_sample_size",
    "cluster_required_clusters",
    "cluster_minimum_detectable_lift",
    "plot_cluster_power_curve",
    "plot_cluster_sensitivity_curve",
]

_VALID_LIFTS = frozenset({"absolute", "relative"})


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def estimate_icc(
    successes: np.ndarray[Any, Any] | list[int],
    trials: np.ndarray[Any, Any] | list[int],
) -> float:
    """Estimate intra-cluster correlation for binary outcomes.

    Parameters
    ----------
    successes : array_like
        Number of successes in each cluster.
    trials : array_like
        Number of trials in each cluster.

    Returns
    -------
    float
        Estimated ICC, clamped to [0, 1].

    Notes
    -----
    Uses the one-way ANOVA estimator for binary data (Ridout, Demetrio
    & Firth, 1999).
    """
    s = np.asarray(successes, dtype=float)
    m = np.asarray(trials, dtype=float)
    K = len(s)
    if K < 2:
        raise ValueError("estimate_icc requires at least 2 clusters")
    if np.any(m < 1):
        raise ValueError("All cluster sizes (trials) must be >= 1")
    if np.any(s < 0) or np.any(s > m):
        raise ValueError("successes must satisfy 0 <= successes <= trials for each cluster")

    N = float(np.sum(m))
    p_k = s / m
    p_bar = float(np.sum(s) / N)
    m0 = (N - float(np.sum(m**2)) / N) / (K - 1)

    MSB = float(np.sum(m * (p_k - p_bar) ** 2)) / (K - 1)
    MSW = float(np.sum(m * p_k * (1 - p_k))) / (N - K)

    denom = MSB + (m0 - 1) * MSW
    if denom == 0:
        return 0.0
    rho = (MSB - MSW) / denom
    return float(np.clip(rho, 0.0, 1.0))


def design_effect(avg_cluster_size: float, icc: float) -> float:
    """Compute the design effect for a cluster-randomized trial.

    Parameters
    ----------
    avg_cluster_size : float
        Average number of individuals per cluster.
    icc : float
        Intra-cluster correlation coefficient.

    Returns
    -------
    float
        Design effect (DEFF), always >= 1 for valid inputs.
    """
    if avg_cluster_size < 1:
        raise ValueError(f"avg_cluster_size must be >= 1, got {avg_cluster_size}")
    if not 0 <= icc <= 1:
        raise ValueError(f"icc must be between 0 and 1, got {icc}")
    return 1.0 + (avg_cluster_size - 1.0) * icc


# ---------------------------------------------------------------------------
# Power factory and wrappers
# ---------------------------------------------------------------------------


def cluster_adjusted_power(
    icc: float,
    avg_cluster_size: float,
    power_func: Callable[..., float] = score_power,
) -> Callable[..., float]:
    """Return a power function adjusted for cluster randomization.

    Parameters
    ----------
    icc : float
        Intra-cluster correlation coefficient.
    avg_cluster_size : float
        Average number of individuals per cluster.
    power_func : callable
        Underlying power function with signature ``(n, p_null, p_alt, alpha)``.
        Defaults to :func:`~ab_test.frequentist_binomial.power_calculations.score_power`.

    Returns
    -------
    callable
        A power function with the same signature as ``power_func`` but with
        effective sample sizes deflated by the design effect.
    """
    deff = design_effect(avg_cluster_size, icc)

    def adjusted(
        n: np.ndarray[Any, Any] | list[Any],
        p_null: np.ndarray[Any, Any] | list[Any],
        p_alt: np.ndarray[Any, Any] | list[Any],
        alpha: float = 0.05,
    ) -> float:
        n_eff = [ni / deff for ni in n]
        return power_func(n_eff, p_null, p_alt, alpha=alpha)

    return adjusted


def cluster_required_sample_size(
    baseline: float,
    alt_lift: float,
    icc: float,
    avg_cluster_size: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    null_lift: float = 0.0,
    lift: str = "relative",
) -> int:
    """Calculate the required sample size accounting for clustering.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    icc : float
        Intra-cluster correlation coefficient.
    avg_cluster_size : float
        Average number of individuals per cluster.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to 50/50.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : str
        ``"relative"`` or ``"absolute"``.

    Returns
    -------
    int
        Minimum total sample size across all groups.
    """
    return required_sample_size(
        baseline,
        alt_lift,
        alpha=alpha,
        beta=beta,
        group_proportions=group_proportions,
        null_lift=null_lift,
        power=cluster_adjusted_power(icc, avg_cluster_size),
        lift=lift,
    )


def cluster_required_clusters(
    baseline: float,
    alt_lift: float,
    icc: float,
    cluster_size: int,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    lift: str = "relative",
) -> int:
    """Calculate the number of clusters per arm needed for desired power.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Number of individuals per cluster (assumed equal).
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : str
        ``"relative"`` or ``"absolute"``.

    Returns
    -------
    int
        Minimum number of clusters per arm.
    """
    total_n = cluster_required_sample_size(
        baseline,
        alt_lift,
        icc,
        float(cluster_size),
        alpha=alpha,
        beta=beta,
        null_lift=null_lift,
        lift=lift,
    )
    return math.ceil(total_n / (2 * cluster_size))


def cluster_minimum_detectable_lift(
    group_sizes: np.ndarray[Any, Any] | list[Any],
    baseline: float,
    icc: float,
    avg_cluster_size: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    drop: bool = False,
    lift: str = "relative",
) -> float:
    """Minimum detectable lift accounting for clustering.

    Parameters
    ----------
    group_sizes : array_like
        Number of experimental units in each group.
    baseline : float
        Baseline success rate.
    icc : float
        Intra-cluster correlation coefficient.
    avg_cluster_size : float
        Average number of individuals per cluster.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    drop : bool
        If True, return the minimum detectable drop. Defaults to False.
    lift : str
        ``"relative"`` or ``"absolute"``.

    Returns
    -------
    float
        Minimum detectable lift (or drop).
    """
    return minimum_detectable_lift(
        group_sizes,
        baseline,
        alpha=alpha,
        beta=beta,
        null_lift=null_lift,
        power=cluster_adjusted_power(icc, avg_cluster_size),
        drop=drop,
        lift=lift,
    )


# ---------------------------------------------------------------------------
# ClusterRandomizedTrial class
# ---------------------------------------------------------------------------


class ClusterRandomizedTrial:
    """Analyze a cluster-randomized trial with binary outcomes.

    Data is entered via :meth:`add` calls (one per cluster), then
    :meth:`analyze` runs a cluster-summary Welch t-test.

    Parameters
    ----------
    name : str
        Experiment name.
    metric_name : str
        Name of the outcome metric.

    Examples
    --------
    >>> crt = ClusterRandomizedTrial("Store test", "conversion")
    >>> crt.add("store_1", 45, 500, group="control")
    >>> crt.add("store_2", 52, 480, group="control")
    >>> crt.add("store_3", 62, 490, group="treatment")
    >>> crt.add("store_4", 58, 520, group="treatment")
    """

    def __init__(self, name: str = "CRT", metric_name: str = "outcome") -> None:
        self.experiment_name: str = name
        self.metric_name: str = metric_name
        self._clusters: dict[str, dict[str, Any]] = {}
        self._group_names: list[str] = []
        self._analyzed: dict[str, Any] | None = None

    def add(
        self,
        cluster_name: str,
        successes: int,
        trials: int,
        *,
        group: str,
    ) -> ClusterRandomizedTrial:
        """Add a cluster's data.

        Parameters
        ----------
        cluster_name : str
            Unique identifier for the cluster.
        successes : int
            Number of successes in this cluster.
        trials : int
            Number of trials in this cluster.
        group : str
            Group this cluster belongs to (e.g. ``"control"``,
            ``"treatment"``).  Exactly two distinct group names are
            allowed.

        Returns
        -------
        ClusterRandomizedTrial
            Self, for method chaining.
        """
        if group not in self._group_names:
            if len(self._group_names) >= 2:
                raise ValueError(f"Only 2 groups are supported, got third group {group!r}")
            self._group_names.append(group)

        if cluster_name in self._clusters:
            raise ValueError(f"Cluster {cluster_name!r} already added")
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if successes < 0 or successes > trials:
            raise ValueError(f"successes must satisfy 0 <= successes <= trials, got {successes}/{trials}")

        self._clusters[cluster_name] = {
            "group": group,
            "successes": successes,
            "trials": trials,
        }
        self._analyzed = None
        return self

    def _build_arm_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract per-arm arrays of (successes, trials).

        Returns
        -------
        s_ctrl, m_ctrl, s_treat, m_treat : np.ndarray
            Per-cluster successes and trials for each arm.
        """
        if len(self._group_names) != 2:
            raise ValueError(f"analyze requires exactly 2 groups, got {len(self._group_names)}")

        ctrl_name, treat_name = self._group_names[0], self._group_names[1]
        ctrl_s, ctrl_m, treat_s, treat_m = [], [], [], []
        for info in self._clusters.values():
            if info["group"] == ctrl_name:
                ctrl_s.append(info["successes"])
                ctrl_m.append(info["trials"])
            else:
                treat_s.append(info["successes"])
                treat_m.append(info["trials"])

        if len(ctrl_s) < 2:
            raise ValueError(f"Group {ctrl_name!r} has {len(ctrl_s)} cluster(s); need at least 2 per arm")
        if len(treat_s) < 2:
            raise ValueError(f"Group {treat_name!r} has {len(treat_s)} cluster(s); need at least 2 per arm")

        return (
            np.array(ctrl_s, dtype=float),
            np.array(ctrl_m, dtype=float),
            np.array(treat_s, dtype=float),
            np.array(treat_m, dtype=float),
        )

    def analyze(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
        *,
        method: str = "welch",
        n_permutations: int = 10_000,
        seed: int | None = None,
        exact: bool = False,
    ) -> str:
        """Analyze the cluster-randomized trial.

        Parameters
        ----------
        lift : str
            ``"relative"`` or ``"absolute"``.
        alpha : float
            Significance level. Defaults to 0.05.
        method : str
            ``"welch"`` for a cluster-summary Welch t-test (default), or
            ``"randomization"`` for randomization inference.
        n_permutations : int
            Number of Monte Carlo permutations.  Only used when
            ``method="randomization"`` and ``exact=False``.
        seed : int or None
            Random seed.  Only used when ``method="randomization"``.
        exact : bool
            Enumerate all possible cluster assignments instead of Monte
            Carlo.  Only used when ``method="randomization"``.

        Returns
        -------
        str
            Formatted results table.
        """
        from tabulate import tabulate

        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")
        method = method.casefold()
        _valid_methods = {"welch", "randomization"}
        if method not in _valid_methods:
            raise ValueError(f"method must be one of {sorted(_valid_methods)}, got {method!r}")

        s_ctrl, m_ctrl, s_treat, m_treat = self._build_arm_data()

        p_ctrl = s_ctrl / m_ctrl
        p_treat = s_treat / m_treat
        K_ctrl = len(p_ctrl)
        K_treat = len(p_treat)

        mean_ctrl = float(np.mean(p_ctrl))
        mean_treat = float(np.mean(p_treat))
        abs_diff = mean_treat - mean_ctrl

        if method == "randomization":
            p_value = cluster_randomization_test(
                s_ctrl,
                m_ctrl,
                s_treat,
                m_treat,
                n_permutations=n_permutations,
                seed=seed,
                exact=exact,
            )
            ci_lower_abs = -math.inf
            ci_upper_abs = math.inf
            pvalue_label = "p-value (RI)"
        else:
            var_ctrl = float(np.var(p_ctrl, ddof=1))
            var_treat = float(np.var(p_treat, ddof=1))

            se_ctrl = var_ctrl / K_ctrl
            se_treat = var_treat / K_treat
            se_sum = se_ctrl + se_treat

            if se_sum == 0:
                t_stat = 0.0 if mean_treat == mean_ctrl else math.copysign(math.inf, mean_treat - mean_ctrl)
                welch_df = float(K_ctrl + K_treat - 2)
                se = 0.0
            else:
                se = math.sqrt(se_sum)
                t_stat = (mean_treat - mean_ctrl) / se
                welch_df = se_sum**2 / (se_ctrl**2 / (K_ctrl - 1) + se_treat**2 / (K_treat - 1))

            if math.isinf(t_stat):
                p_value = 0.0
            else:
                p_value = float(2 * ss.t.sf(abs(t_stat), df=welch_df))

            t_crit = float(ss.t.ppf(1 - alpha / 2, df=welch_df))
            ci_lower_abs = abs_diff - t_crit * se
            ci_upper_abs = abs_diff + t_crit * se
            pvalue_label = "p-value (Welch t)"

        if lift == "relative":
            if mean_ctrl == 0:
                test_lift = math.inf if abs_diff > 0 else (-math.inf if abs_diff < 0 else 0.0)
                ci_lower = -math.inf
                ci_upper = math.inf
            else:
                test_lift = abs_diff / mean_ctrl
                ci_lower = ci_lower_abs / mean_ctrl
                ci_upper = ci_upper_abs / mean_ctrl
        else:
            test_lift = abs_diff
            ci_lower = ci_lower_abs
            ci_upper = ci_upper_abs

        all_s = np.concatenate([s_ctrl, s_treat])
        all_m = np.concatenate([m_ctrl, m_treat])
        icc_val = estimate_icc(all_s, all_m)
        avg_m = float(np.mean(all_m))
        deff_val = design_effect(avg_m, icc_val)

        self._analyzed = {
            "method": method,
            "lift_type": lift,
            "lift": test_lift,
            "control_rate": mean_ctrl,
            "treatment_rate": mean_treat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "icc": icc_val,
            "deff": deff_val,
            "n_clusters_control": K_ctrl,
            "n_clusters_treatment": K_treat,
            "alpha": alpha,
        }
        if method == "welch":
            self._analyzed["se"] = se
            self._analyzed["t_stat"] = t_stat
            self._analyzed["welch_df"] = welch_df

        str_pvalue = f"{p_value}" if p_value >= alpha else f"{p_value}*"
        success_rate: list[str | float] = [
            convert_to_tabulate_str(mean_ctrl, "absolute"),
            convert_to_tabulate_str(mean_treat, "absolute"),
        ]
        table_headers = (
            ["Metric", "Metric Name"]
            + self._group_names
            + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", pvalue_label]
        )
        table_list = [
            [lift, self.metric_name]
            + success_rate
            + convert_to_tabulate_str([test_lift, ci_lower, ci_upper], lift)
            + [str_pvalue]
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")
        ctrl_name, treat_name = self._group_names
        footer = f"\nICC: {icc_val:.4f} | DEFF: {deff_val:.2f} | Clusters: {K_ctrl} {ctrl_name}, {K_treat} {treat_name}"
        if method == "welch":
            footer += f" | Welch df: {welch_df:.1f}"
        return_string += footer
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    @property
    def icc(self) -> float:
        """Estimated intra-cluster correlation from the trial data."""
        if self._analyzed is None:
            raise RuntimeError("Call analyze() before accessing icc")
        return self._analyzed["icc"]

    @property
    def deff(self) -> float:
        """Estimated design effect from the trial data."""
        if self._analyzed is None:
            raise RuntimeError("Call analyze() before accessing deff")
        return self._analyzed["deff"]

    def summary(self, alpha: float = 0.05) -> dict[str, Any]:
        """Return the full results as a dictionary.

        Parameters
        ----------
        alpha : float
            Significance level. Defaults to 0.05.

        Returns
        -------
        dict
            Analysis results including lift, CI, p-value, ICC, and DEFF.
        """
        if self._analyzed is None:
            self.analyze(alpha=alpha)
        return dict(self._analyzed)  # type: ignore[arg-type]

    def plot(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Forest plot of per-cluster proportions and arm estimates.

        Parameters
        ----------
        lift : str
            ``"relative"`` or ``"absolute"``.
        alpha : float
            Significance level for confidence intervals.
        reverse_plot : bool
            Whether to reverse the y-axis (first cluster at top).
        color : str, list, dict, or None
            Color specification (see
            :func:`~ab_test._display.resolve_plot_color`).
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        s_ctrl, m_ctrl, s_treat, m_treat = self._build_arm_data()
        p_ctrl = s_ctrl / m_ctrl
        p_treat = s_treat / m_treat

        ctrl_name, treat_name = self._group_names

        ctrl_clusters = [name for name, info in self._clusters.items() if info["group"] == ctrl_name]
        treat_clusters = [name for name, info in self._clusters.items() if info["group"] == treat_name]

        plot_color = resolve_plot_color(color)
        fig = go.Figure()

        all_labels = ctrl_clusters + treat_clusters
        all_proportions = list(p_ctrl) + list(p_treat)
        all_groups = [ctrl_name] * len(ctrl_clusters) + [treat_name] * len(treat_clusters)

        for i, (label, prop, grp) in enumerate(zip(all_labels, all_proportions, all_groups)):
            c = None
            if plot_color is not None:
                if isinstance(plot_color, list):
                    c = plot_color[0] if grp == ctrl_name else plot_color[min(1, len(plot_color) - 1)]
                elif isinstance(plot_color, dict):
                    c = plot_color.get(grp)

            marker_kw: dict[str, Any] = {"symbol": "circle", "size": 8}
            if c is not None:
                marker_kw["color"] = c

            fig.add_trace(
                go.Scatter(
                    x=[float(prop)],
                    y=[f"{label} ({grp})"],
                    marker=marker_kw,
                    name=label,
                    showlegend=False,
                )
            )

        mean_ctrl_val = float(np.mean(p_ctrl))
        mean_treat_val = float(np.mean(p_treat))

        for grp_name, grp_mean in [(ctrl_name, mean_ctrl_val), (treat_name, mean_treat_val)]:
            c_arm = None
            if plot_color is not None:
                if isinstance(plot_color, list):
                    c_arm = plot_color[0] if grp_name == ctrl_name else plot_color[min(1, len(plot_color) - 1)]
                elif isinstance(plot_color, dict):
                    c_arm = plot_color.get(grp_name)

            marker_kw_arm: dict[str, Any] = {"symbol": "diamond", "size": 14}
            if c_arm is not None:
                marker_kw_arm["color"] = c_arm

            fig.add_trace(
                go.Scatter(
                    x=[grp_mean],
                    y=[f"{grp_name} (mean)"],
                    marker=marker_kw_arm,
                    name=f"{grp_name} mean",
                )
            )

        fig.update_layout(
            title=f"Cluster Proportions — {self.experiment_name}",
            xaxis_title="Success Rate",
            xaxis_tickformat=",.2%",
            template="plotly_white",
            yaxis={"autorange": "reversed" if reverse_plot else True},
        )
        fig.show()


# ---------------------------------------------------------------------------
# Plotting free functions
# ---------------------------------------------------------------------------


def plot_cluster_power_curve(
    baseline: float,
    alt_lift: float,
    icc: float,
    avg_cluster_size: float,
    alpha: float = 0.05,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot statistical power with and without cluster adjustment.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    alt_lift : float
        Lift under the alternative hypothesis.
    icc : float
        Intra-cluster correlation coefficient.
    avg_cluster_size : float
        Average number of individuals per cluster.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : str
        ``"relative"`` or ``"absolute"``.
    sample_sizes : array_like or None
        Explicit sample sizes. Auto-ranged when ``None``.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to 50/50.
    n_points : int
        Number of points when auto-ranging. Defaults to 100.

    Returns
    -------
    go.Figure
        Plotly figure with cluster-adjusted and unadjusted power curves.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    adjusted_power = cluster_adjusted_power(icc, avg_cluster_size)

    if sample_sizes is None:
        target_ss = cluster_required_sample_size(
            baseline,
            alt_lift,
            icc,
            avg_cluster_size,
            alpha=alpha,
            beta=0.2,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        max_ss = int(target_ss * 2)
        sample_sizes = np.linspace(max(20, max_ss // n_points), max_ss, n_points, dtype=int)

    adjusted_powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=adjusted_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    unadjusted_powers = [
        abtest_power(
            [int(ss * g) for g in group_proportions],
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=score_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=adjusted_powers,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"Cluster-adjusted (ICC={icc:.3f}, m={avg_cluster_size:.0f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=unadjusted_powers,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dot"},
            name="No clustering",
        )
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="80% power",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Power Curve (Cluster-Adjusted)",
        xaxis_title="Total sample size",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def plot_cluster_sensitivity_curve(
    baseline: float,
    icc: float,
    avg_cluster_size: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    null_lift: float = 0.0,
    lift: str = "relative",
    sample_sizes: np.ndarray[Any, Any] | list[int] | None = None,
    group_proportions: np.ndarray[Any, Any] | list[Any] | None = None,
    n_points: int = 100,
) -> go.Figure:
    """Plot minimum detectable lift with and without cluster adjustment.

    Parameters
    ----------
    baseline : float
        Baseline success rate.
    icc : float
        Intra-cluster correlation coefficient.
    avg_cluster_size : float
        Average number of individuals per cluster.
    alpha : float
        Type-I error rate. Defaults to 0.05.
    beta : float
        Type-II error rate (1 - power). Defaults to 0.2.
    null_lift : float
        Lift under the null hypothesis. Defaults to 0.0.
    lift : str
        ``"relative"`` or ``"absolute"``.
    sample_sizes : array_like or None
        Explicit sample sizes. Auto-ranged when ``None``.
    group_proportions : array_like or None
        Fraction of units in each group. Defaults to 50/50.
    n_points : int
        Number of points when auto-ranging. Defaults to 100.

    Returns
    -------
    go.Figure
        Plotly figure with cluster-adjusted and unadjusted sensitivity
        curves.
    """
    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    adj_power = cluster_adjusted_power(icc, avg_cluster_size)

    if sample_sizes is None:
        target_ss = cluster_required_sample_size(
            baseline,
            0.05,
            icc,
            avg_cluster_size,
            alpha=alpha,
            beta=beta,
            group_proportions=group_proportions,
            null_lift=null_lift,
            lift=lift,
        )
        min_ss = max(20, target_ss // 10)
        max_ss = target_ss * 5
        sample_sizes = np.linspace(min_ss, max_ss, n_points, dtype=int)

    adjusted_mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=adj_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]
    unadjusted_mdls = [
        minimum_detectable_lift(
            [int(ss * g) for g in group_proportions],
            baseline,
            alpha=alpha,
            beta=beta,
            null_lift=null_lift,
            power=score_power,
            lift=lift,
        )
        for ss in sample_sizes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=adjusted_mdls,
            mode="lines",
            line={"color": "#636EFA", "width": 2},
            name=f"Cluster-adjusted (ICC={icc:.3f}, m={avg_cluster_size:.0f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(sample_sizes),
            y=unadjusted_mdls,
            mode="lines",
            line={"color": "#EF553B", "width": 2, "dash": "dot"},
            name="No clustering",
        )
    )

    y_label = f"Minimum detectable {lift} lift"
    fig.update_layout(
        title="Sensitivity Curve (Cluster-Adjusted)",
        xaxis_title="Total sample size",
        yaxis_title=y_label,
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
