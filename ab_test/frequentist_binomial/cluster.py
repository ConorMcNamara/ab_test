"""Cluster-randomized trial analysis and design.

Provides tools for analyzing experiments where randomization occurs at
the cluster level (e.g., stores, markets, time windows) rather than the
individual level.  Intra-cluster correlation inflates variance and must
be accounted for in both inference and sample-size planning.

The analysis uses a cluster-summary Welch t-test on per-cluster
proportions, which is the standard approach for cluster-randomized
trials with binary outcomes (Donner & Klar, 2000).  Power and
sample-size calculations use the design-effect adjustment to deflate
effective sample sizes, with a t critical value for the number of clusters,
integrating with the existing pluggable power framework via
:func:`cluster_adjusted_power`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import scipy.stats as ss
from scipy.optimize import brentq
from tabulate import tabulate

from ab_test._display import (
    apply_dark_mode,
    convert_to_tabulate_str,
    format_percent,
    resolve_plot_color,
    tabulate_summary,
)
from ab_test.corrections import adjust_pvalues
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


def _binomial_floor(successes: np.ndarray[Any, Any], trials: np.ndarray[Any, Any]) -> float:
    """Variance of cluster rates from binomial sampling alone, at the arm's pooled rate."""
    pooled = float(np.sum(successes) / np.sum(trials))
    return pooled * (1 - pooled) * float(np.mean(1.0 / np.asarray(trials, dtype=float)))


def _arm_variance(successes: np.ndarray[Any, Any], trials: np.ndarray[Any, Any]) -> float:
    """Sample variance of an arm's cluster rates, or the binomial variance when it is exactly 0."""
    return float(np.var(successes / trials, ddof=1)) or _binomial_floor(successes, trials)


def _welch_anova(groups: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]) -> tuple[float, int, float, float]:
    """Welch's heteroscedastic one-way ANOVA on cluster rates (Welch, 1951).

    Parameters
    ----------
    groups : list of (successes, trials)
        Per-cluster successes and trials for each group.

    Returns
    -------
    statistic, df1, df2, pvalue : float, int, float, float
        Welch's F statistic, its degrees of freedom and the p-value. With two
        groups ``statistic`` is the square of the Welch t statistic and ``df2``
        its Welch-Satterthwaite degrees of freedom.
    """
    k = len(groups)
    n = np.array([len(s) for s, _ in groups], dtype=float)
    means = np.array([float(np.mean(s / m)) for s, m in groups])
    variances = np.array([_arm_variance(s, m) for s, m in groups])
    if np.all(variances == 0):
        # Every cluster rate equals its group's (all 0 or all 1), as in the two-group test.
        if np.ptp(means) == 0:
            return 0.0, k - 1, float(np.sum(n) - k), 1.0
        return math.inf, k - 1, math.nan, 0.0
    # A group whose clusters are all 0 (or all 1) has no variance at all; a tiny one keeps the limit.
    weights = n / np.maximum(variances, 1e-24)
    total = float(np.sum(weights))
    grand = float(np.sum(weights * means)) / total
    spread = float(np.sum(weights * (means - grand) ** 2)) / (k - 1)
    lam = float(np.sum((1 - weights / total) ** 2 / (n - 1)))
    statistic = spread / (1 + 2 * (k - 2) * lam / (k**2 - 1))
    df2 = (k**2 - 1) / (3 * lam)
    return statistic, k - 1, df2, float(ss.f.sf(statistic, k - 1, df2))


def estimate_icc(
    successes: np.ndarray[Any, Any] | list[int],
    trials: np.ndarray[Any, Any] | list[int],
    groups: np.ndarray[Any, Any] | list[Any] | None = None,
) -> float:
    """Estimate intra-cluster correlation for binary outcomes.

    Parameters
    ----------
    successes : array_like
        Number of successes in each cluster.
    trials : array_like
        Number of trials in each cluster.
    groups : array_like, optional
        Arm label of each cluster. When given, between-cluster variation is
        measured around each arm's own rate, so a treatment effect is not
        mistaken for clustering. Pass this whenever the clusters come from
        more than one arm.

    Returns
    -------
    float
        Estimated ICC, clamped to [0, 1]. NaN when every cluster has a
        single trial: there is then no within-cluster variation, so the ICC
        cannot be estimated (and does not matter: the design effect is 1).

    Notes
    -----
    Uses the one-way ANOVA estimator for binary data (Ridout, Demetrio
    & Firth, 1999), with arm as a fixed factor when ``groups`` is given.
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
    g = np.zeros(K, dtype=int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    if len(g) != K:
        raise ValueError(f"groups must have one label per cluster, got {len(g)} for {K} clusters")
    G = int(g.max()) + 1
    if K - G < 1:
        raise ValueError("estimate_icc requires more clusters than groups")

    N = float(np.sum(m))
    if N == K:
        # Every cluster has one trial, so the within-cluster mean square is 0/0.
        return math.nan
    p_k = s / m
    s_g = np.bincount(g, weights=s)
    N_g = np.bincount(g, weights=m)
    p_g = (s_g / N_g)[g]
    m0 = (N - float(np.sum(np.bincount(g, weights=m**2) / N_g))) / (K - G)

    MSB = float(np.sum(m * (p_k - p_g) ** 2)) / (K - G)
    MSW = float(np.sum(m * p_k * (1 - p_k))) / (N - K)

    denom = MSB + (m0 - 1) * MSW
    if denom == 0:
        return 0.0
    rho = (MSB - MSW) / denom
    return float(np.clip(rho, 0.0, 1.0))


def _fieller_ratio_interval(
    mean_num: float,
    var_num: float,
    mean_den: float,
    var_den: float,
    t_crit: float,
) -> tuple[float, float]:
    """Fieller confidence interval for ``mean_num / mean_den`` from independent estimates.

    The interval is the set of ratios ``R`` with
    ``(mean_num - R * mean_den)**2 <= t_crit**2 * (var_num + R**2 * var_den)``.
    Unlike dividing a difference interval by ``mean_den``, it accounts for the
    uncertainty in the denominator. It is unbounded when ``mean_den`` is not
    clearly separated from zero.
    """
    a = mean_den**2 - t_crit**2 * var_den
    if a <= 0:
        return -math.inf, math.inf
    b = mean_num * mean_den
    c = mean_num**2 - t_crit**2 * var_num
    root = math.sqrt(max(b**2 - a * c, 0.0))
    return (b - root) / a, (b + root) / a


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
        effective sample sizes deflated by the design effect, and with the
        two-sided critical value taken from a t distribution with
        ``n_clusters - 2`` degrees of freedom to match the cluster-summary
        t-test used by :class:`ClusterRandomizedTrial`.
    """
    deff = design_effect(avg_cluster_size, icc)

    def adjusted(
        n: np.ndarray[Any, Any] | list[Any],
        p_null: np.ndarray[Any, Any] | list[Any],
        p_alt: np.ndarray[Any, Any] | list[Any],
        alpha: float = 0.05,
    ) -> float:
        n_eff = [ni / deff for ni in n]
        z_power = power_func(n_eff, p_null, p_alt, alpha=alpha)
        df = sum(n) / avg_cluster_size - 2
        if df <= 0:
            return 0.0
        return _t_test_power(z_power, alpha, df)

    return adjusted


def _t_test_power(z_power: float, alpha: float, df: float) -> float:
    """Convert the power of a two-sided z-test to that of a t-test with ``df``.

    The z-test power is inverted to its noncentrality, which is then used as
    the noncentrality of a t statistic. With few clusters the t-test's larger
    critical value and heavier tails cost noticeable power.
    """
    z_crit = float(ss.norm.isf(alpha / 2))
    if z_power <= alpha:
        return z_power
    if z_power >= 1.0:
        return 1.0

    def z_test_power(ncp: float) -> float:
        return float(ss.norm.sf(z_crit - ncp) + ss.norm.cdf(-z_crit - ncp)) - z_power

    ncp = brentq(z_test_power, 0.0, z_crit + 40.0)
    t_crit = float(ss.t.isf(alpha / 2, df))
    return float(ss.nct.sf(t_crit, df, ncp) + ss.nct.cdf(-t_crit, df, ncp))


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
    >>> crt = (
    ...     crt.add("store_1", 45, 500, group="control")
    ...     .add("store_2", 52, 480, group="control")
    ...     .add("store_3", 62, 490, group="treatment")
    ...     .add("store_4", 58, 520, group="treatment")
    ... )
    >>> results = crt.analyze(lift="relative")
    """

    def __init__(self, name: str = "CRT", metric_name: str = "outcome") -> None:
        self.experiment_name: str = name
        self.metric_name: str = metric_name
        self._clusters: dict[str, dict[str, Any]] = {}
        self._group_names: list[str] = []
        self._analyzed: dict[str, Any] | None = None
        self._analyze_kwargs: dict[str, Any] = {}

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
            ``"treatment"``).  The first group added is the control; with
            three or more groups, :meth:`analyze` compares them pairwise.

        Returns
        -------
        ClusterRandomizedTrial
            Self, for method chaining.
        """
        if group not in self._group_names:
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

    def _build_groups(self) -> list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        """Per-group arrays of cluster successes and trials, in the order groups were added."""
        if len(self._group_names) < 2:
            raise ValueError(f"analyze requires at least 2 groups, got {len(self._group_names)}")
        groups = []
        for name in self._group_names:
            s = [info["successes"] for info in self._clusters.values() if info["group"] == name]
            m = [info["trials"] for info in self._clusters.values() if info["group"] == name]
            if len(s) < 2:
                raise ValueError(f"Group {name!r} has {len(s)} cluster(s); need at least 2 per arm")
            groups.append((name, np.array(s, dtype=float), np.array(m, dtype=float)))
        return groups

    def _build_arm_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract per-arm arrays of (successes, trials) for a two-group trial.

        Returns
        -------
        s_ctrl, m_ctrl, s_treat, m_treat : np.ndarray
            Per-cluster successes and trials for each arm.
        """
        (_, s_ctrl, m_ctrl), (_, s_treat, m_treat) = self._build_groups()
        return s_ctrl, m_ctrl, s_treat, m_treat

    def analyze(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
        *,
        method: str = "welch",
        n_permutations: int = 10_000,
        seed: int | None = None,
        exact: bool = False,
        n_jobs: int = 1,
        comparisons: str = "control",
        correction: str = "holm",
    ) -> str:
        """Analyze the cluster-randomized trial.

        Parameters
        ----------
        lift : str
            ``"relative"`` or ``"absolute"``. For ``"relative"`` with the Welch
            method, the confidence interval is Fieller's interval for the ratio
            of arm means, which accounts for uncertainty in the control mean and
            is unbounded when the control mean is not clearly above zero.
        alpha : float
            Significance level. Defaults to 0.05.
        method : str
            ``"welch"`` for a cluster-summary Welch t-test (default), or
            ``"randomization"`` for randomization inference (two groups only).
        n_permutations : int
            Number of Monte Carlo permutations.  Only used when
            ``method="randomization"`` and ``exact=False``.
        seed : int or None
            Random seed.  Only used when ``method="randomization"``.
        exact : bool
            Enumerate all possible cluster assignments instead of Monte
            Carlo.  Only used when ``method="randomization"``.
        n_jobs : int
            Number of parallel jobs for Monte Carlo permutations.
            ``1`` (default) runs sequentially; ``-1`` uses all cores.
            Only used when ``method="randomization"`` and ``exact=False``.
        comparisons : {"control", "all"}, default="control"
            With three or more groups, which pairs to compare: each group
            against the first group added (the control), or every pair.
            Ignored with two groups.
        correction : str, default="holm"
            With three or more groups, how the pairwise p-values are adjusted
            for multiple comparisons; any method accepted by
            :func:`~ab_test.corrections.adjust_pvalues`. Ignored with two
            groups.

        Returns
        -------
        str
            Formatted results table.

        Notes
        -----
        With three or more groups, ``analyze()`` reports Welch's
        heteroscedastic one-way ANOVA on the cluster rates (Welch, 1951) as an
        omnibus test that every group has the same mean rate, then the chosen
        pairwise comparisons, each computed exactly as a two-group analysis
        (Welch t-test, Fieller interval for relative lift). The p-values are
        adjusted with ``correction`` and the intervals are Bonferroni
        intervals at ``1 - alpha / m`` for ``m`` comparisons. With two groups
        Welch's ANOVA reduces to the Welch t-test (F = t^2). Randomization
        inference supports two groups only. ``summary()`` then returns
        ``"omnibus"``, ``"group_rates"`` and a ``"comparisons"`` dict keyed by
        labels such as ``"B vs A"``.
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")
        method = method.casefold()
        _valid_methods = {"welch", "randomization"}
        if method not in _valid_methods:
            raise ValueError(f"method must be one of {sorted(_valid_methods)}, got {method!r}")
        comparisons = comparisons.casefold()
        if comparisons not in ("control", "all"):
            raise ValueError(f"comparisons must be 'control' or 'all', got {comparisons!r}")
        adjust_pvalues([0.5], method=correction)  # Validate the correction before any work.
        self._analyze_kwargs = {
            "lift": lift,
            "alpha": alpha,
            "method": method,
            "n_permutations": n_permutations,
            "seed": seed,
            "exact": exact,
            "n_jobs": n_jobs,
            "comparisons": comparisons,
            "correction": correction,
        }

        groups = self._build_groups()
        if len(groups) > 2:
            if method == "randomization":
                raise ValueError(
                    "method='randomization' supports two groups only; use method='welch' with three or more groups"
                )
            return self._analyze_many(groups, lift, alpha, comparisons, correction)

        (_, s_ctrl, m_ctrl), (_, s_treat, m_treat) = groups
        result = self._compare_arms(
            s_ctrl, m_ctrl, s_treat, m_treat, lift, alpha, method, n_permutations, seed, exact, n_jobs
        )
        test_lift, mean_ctrl, mean_treat = result["lift"], result["control_rate"], result["treatment_rate"]
        p_value, ci_lower, ci_upper = result["p_value"], result["ci_lower"], result["ci_upper"]
        pvalue_label = "p-value (Welch t)" if method == "welch" else "p-value (RI)"
        K_ctrl, K_treat = len(s_ctrl), len(s_treat)

        all_s = np.concatenate([s_ctrl, s_treat])
        all_m = np.concatenate([m_ctrl, m_treat])
        arms = np.repeat([0, 1], [K_ctrl, K_treat])
        icc_val, deff_val = self._icc_and_deff(all_s, all_m, arms)

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
            welch_df = result["welch_df"]
            self._analyzed["se"] = result["se"]
            self._analyzed["t_stat"] = result["t_stat"]
            self._analyzed["welch_df"] = welch_df

        str_pvalue = f"{p_value:.4f}" if p_value >= alpha else f"{p_value:.4f}*"
        success_rate: list[str | float] = [
            convert_to_tabulate_str(mean_ctrl, "absolute"),
            convert_to_tabulate_str(mean_treat, "absolute"),
        ]
        row_labels = (
            ["Metric", "Metric Name"]
            + self._group_names
            + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", pvalue_label]
        )
        values = (
            [lift, self.metric_name]
            + success_rate
            + convert_to_tabulate_str([test_lift, ci_lower, ci_upper], lift)
            + [str_pvalue]
        )
        return_string = tabulate_summary(row_labels, values)
        ctrl_name, treat_name = self._group_names
        icc_str = "n/a" if math.isnan(icc_val) else f"{icc_val:.4f}"
        footer = f"\nICC: {icc_str} | DEFF: {deff_val:.2f} | Clusters: {K_ctrl} {ctrl_name}, {K_treat} {treat_name}"
        if method == "welch":
            footer += f" | Welch df: {welch_df:.1f}"
        return_string += footer
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {format_percent(alpha)}% level"
        )
        return_string += f"\n** {format_percent(1 - alpha)}% Confidence Interval"
        return return_string

    @staticmethod
    def _icc_and_deff(
        all_s: np.ndarray[Any, Any], all_m: np.ndarray[Any, Any], arms: np.ndarray[Any, Any]
    ) -> tuple[float, float]:
        """Within-arm ICC and the design effect at the average cluster size."""
        icc_val = estimate_icc(all_s, all_m, groups=arms)
        avg_m = float(np.mean(all_m))
        # With clusters of one trial the ICC is undefined but irrelevant: the design effect is 1.
        deff_val = 1.0 if math.isnan(icc_val) else design_effect(avg_m, icc_val)
        return icc_val, deff_val

    @staticmethod
    def _compare_arms(
        s_ctrl: np.ndarray[Any, Any],
        m_ctrl: np.ndarray[Any, Any],
        s_treat: np.ndarray[Any, Any],
        m_treat: np.ndarray[Any, Any],
        lift: str,
        alpha: float,
        method: str,
        n_permutations: int = 10_000,
        seed: int | None = None,
        exact: bool = False,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Compare a treatment arm with a control arm: lift, rates, p-value and interval."""
        p_ctrl = s_ctrl / m_ctrl
        p_treat = s_treat / m_treat
        K_ctrl = len(p_ctrl)
        K_treat = len(p_treat)

        mean_ctrl = float(np.mean(p_ctrl))
        mean_treat = float(np.mean(p_treat))
        abs_diff = mean_treat - mean_ctrl
        result: dict[str, Any] = {}

        if method == "randomization":
            p_value = cluster_randomization_test(
                s_ctrl,
                m_ctrl,
                s_treat,
                m_treat,
                n_permutations=n_permutations,
                seed=seed,
                exact=exact,
                n_jobs=n_jobs,
            )
            ci_lower_abs = -math.inf
            ci_upper_abs = math.inf
        else:
            # Identical cluster rates give a sample variance of exactly 0, so the SE was 0,
            # p = 0 and the interval had zero width. Cluster rates vary at least as much as
            # binomial sampling makes them (ICC >= 0), so use that variance instead. Only the
            # degenerate case: flooring every small variance made the test overly conservative.
            var_ctrl = _arm_variance(s_ctrl, m_ctrl)
            var_treat = _arm_variance(s_treat, m_treat)

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
            result.update({"se": se, "t_stat": t_stat, "welch_df": welch_df})

        if lift == "relative":
            if mean_ctrl == 0:
                test_lift = math.inf if abs_diff > 0 else (-math.inf if abs_diff < 0 else 0.0)
                ci_lower = -math.inf
                ci_upper = math.inf
            else:
                test_lift = abs_diff / mean_ctrl
                if method == "randomization":
                    ci_lower, ci_upper = -math.inf, math.inf
                else:
                    ratio_lo, ratio_hi = _fieller_ratio_interval(mean_treat, se_treat, mean_ctrl, se_ctrl, t_crit)
                    ci_lower, ci_upper = ratio_lo - 1, ratio_hi - 1
        else:
            test_lift = abs_diff
            ci_lower = ci_lower_abs
            ci_upper = ci_upper_abs

        result.update(
            {
                "lift": test_lift,
                "control_rate": mean_ctrl,
                "treatment_rate": mean_treat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
        return result

    def _analyze_many(
        self,
        groups: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        lift: str,
        alpha: float,
        comparisons: str,
        correction: str,
    ) -> str:
        """Three or more groups: Welch's ANOVA, then corrected pairwise comparisons."""
        k = len(groups)
        pairs = [(0, j) for j in range(1, k)] if comparisons == "control" else list(itertools.combinations(range(k), 2))
        # Bonferroni intervals: simultaneous coverage of 1 - alpha across all comparisons.
        ci_alpha = alpha / len(pairs)
        results = [
            self._compare_arms(groups[i][1], groups[i][2], groups[j][1], groups[j][2], lift, ci_alpha, "welch")
            for i, j in pairs
        ]
        adjusted = adjust_pvalues([r["p_value"] for r in results], method=correction)
        statistic, df1, df2, omnibus_p = _welch_anova([(s, m) for _, s, m in groups])

        names = [name for name, _, _ in groups]
        compared: dict[str, dict[str, Any]] = {}
        for (i, j), result, adj_p in zip(pairs, results, adjusted):
            compared[f"{names[j]} vs {names[i]}"] = {
                "lift": result["lift"],
                f"{names[i]}": result["control_rate"],
                f"{names[j]}": result["treatment_rate"],
                "p_value": adj_p,
                "raw_p_value": result["p_value"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
            }
        rates = {name: float(np.mean(s / m)) for name, s, m in groups}
        all_s = np.concatenate([s for _, s, _ in groups])
        all_m = np.concatenate([m for _, _, m in groups])
        arms = np.repeat(np.arange(k), [len(s) for _, s, _ in groups])
        icc_val, deff_val = self._icc_and_deff(all_s, all_m, arms)
        self._analyzed = {
            "method": "welch",
            "lift_type": lift,
            "comparison_type": comparisons,
            "correction": correction,
            "group_rates": rates,
            "omnibus": {"test": "Welch ANOVA", "statistic": statistic, "df1": df1, "df2": df2, "p_value": omnibus_p},
            "comparisons": compared,
            "icc": icc_val,
            "deff": deff_val,
            "n_clusters": {name: len(s) for name, s, _ in groups},
            "alpha": alpha,
        }

        str_omnibus = f"{omnibus_p:.4f}*" if omnibus_p < alpha else f"{omnibus_p:.4f}"
        return_string = tabulate_summary(
            ["Metric", "Metric Name"] + names + ["Omnibus p-value ***"],
            [lift, self.metric_name] + [convert_to_tabulate_str(r, "absolute") for r in rates.values()] + [str_omnibus],
        )
        rows = []
        for label, comparison in compared.items():
            star = "*" if comparison["p_value"] < alpha else ""
            rows.append(
                [label]
                + convert_to_tabulate_str([comparison["lift"], comparison["ci_lower"], comparison["ci_upper"]], lift)
                + [f"{comparison['raw_p_value']:.4f}", f"{comparison['p_value']:.4f}{star}"]
            )
        headers = ["Comparison", "Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value (Welch t)"]
        headers.append(f"Adj. p ({correction})")
        return_string += "\n" + tabulate(rows, headers=headers, tablefmt="grid")
        icc_str = "n/a" if math.isnan(icc_val) else f"{icc_val:.4f}"
        clusters = ", ".join(f"{len(s)} {name}" for name, s, _ in groups)
        return_string += f"\nICC: {icc_str} | DEFF: {deff_val:.2f} | Clusters: {clusters}"
        return_string += (
            f"\n* next to a p-value means it's statistically significant at the {format_percent(alpha)}% level"
            f" ({correction}-adjusted for {len(pairs)} comparisons)"
        )
        return_string += (
            f"\n** {format_percent(1 - alpha)}% simultaneous Confidence Intervals"
            f" (Bonferroni: each at {round(100 * (1 - ci_alpha), 2):g}%)"
        )
        return_string += (
            f"\n*** Welch's ANOVA on cluster rates that all {k} groups share one mean rate,"
            f" F({df1}, {df2:.1f}) = {statistic:.3f}"
        )
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

    def summary(self, alpha: float | None = None, lift: str | None = None) -> dict[str, Any]:
        """Return the full results as a dictionary.

        Returns the results of the last :meth:`analyze` call. If ``alpha`` or
        ``lift`` is given and differs from that call (or nothing has been
        analyzed yet), :meth:`analyze` is re-run with the new value and the
        previous call's other settings.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to the last analysis's, or 0.05.
        lift : str, optional
            ``"relative"`` or ``"absolute"``. Defaults to the last analysis's,
            or ``"relative"``.

        Returns
        -------
        dict
            Analysis results including lift, CI, p-value, ICC, and DEFF.
        """
        given: dict[str, Any] = {"alpha": alpha, "lift": lift.casefold() if lift is not None else None}
        requested = {k: v for k, v in given.items() if v is not None}
        if self._analyzed is None or any(self._analyze_kwargs.get(k) != v for k, v in requested.items()):
            kwargs: dict[str, Any] = {**self._analyze_kwargs, **requested}
            self.analyze(**kwargs)
        assert self._analyzed is not None
        return dict(self._analyzed)

    def plot(
        self,
        lift: str = "relative",
        alpha: float = 0.05,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
        *,
        dark_mode: bool = False,
    ) -> None:
        """Forest plot of per-cluster proportions and arm estimates.

        Each cluster is a dot and each group's mean a diamond, for any number
        of groups.

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
        dark_mode : bool, default=False
            Render on a dark background with light text and gridlines (Plotly's
            ``"plotly_dark"`` template).
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        groups = self._build_groups()
        plot_color = resolve_plot_color(color)
        fig = go.Figure()

        def group_color(index: int, name: str) -> Any:
            if isinstance(plot_color, list):
                return plot_color[min(index, len(plot_color) - 1)]
            if isinstance(plot_color, dict):
                return plot_color.get(name)
            return None

        # One dot per cluster, grouped by arm in the order the groups were added.
        for index, (grp, s, m) in enumerate(groups):
            labels = [name for name, info in self._clusters.items() if info["group"] == grp]
            for label, prop in zip(labels, s / m):
                c = group_color(index, grp)
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

        for index, (grp_name, s, m) in enumerate(groups):
            c_arm = group_color(index, grp_name)
            marker_kw_arm: dict[str, Any] = {"symbol": "diamond", "size": 14}
            if c_arm is not None:
                marker_kw_arm["color"] = c_arm

            fig.add_trace(
                go.Scatter(
                    x=[float(np.mean(s / m))],
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
        apply_dark_mode(fig, dark_mode)
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
    *,
    dark_mode: bool = False,
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
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).

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
    apply_dark_mode(fig, dark_mode)
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
    *,
    dark_mode: bool = False,
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
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines (Plotly's
        ``"plotly_dark"`` template).

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
    apply_dark_mode(fig, dark_mode)
    return fig
