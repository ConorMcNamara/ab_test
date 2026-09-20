"""Bayesian cluster-randomized trial analysis for binomial outcomes.

Uses a beta-binomial hierarchical model to analyze cluster-randomized
experiments. Per-arm Beta distribution parameters are estimated via
method of moments from observed cluster proportions, giving a posterior
for the arm-level conversion rate that accounts for intra-cluster
correlation. Posterior samples drive P(T > C), expected loss, credible
intervals, and ROPE probabilities — the same decision metrics as
:class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`.

Simulation-based assurance functions provide cluster-aware power analysis,
minimum cluster counts, and minimum detectable lifts for both
P(B > A) and expected-loss decision criteria.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import scipy.stats as ss
from tabulate import tabulate

from ab_test._display import convert_to_tabulate_str, resolve_plot_color
from ab_test.bayesian_binomial.credible_intervals import calculate_hdi_from_samples

__all__ = [
    "BayesianClusterRandomizedTrial",
    "estimate_beta_binomial_params",
    "beta_binomial_icc",
    "cluster_bayes_power_lift",
    "cluster_bayes_power_loss",
    "cluster_bayes_minimum_clusters",
    "cluster_bayes_minimum_clusters_loss",
    "cluster_bayes_minimum_detectable_lift",
    "cluster_bayes_minimum_detectable_lift_loss",
    "plot_cluster_bayes_power_curve",
    "plot_cluster_bayes_sensitivity_curve",
]

_VALID_LIFTS = frozenset({"relative", "absolute"})


def _resolve_alt_rate(
    baseline: float,
    alt_lift: float | None,
    alt_rate: float | None,
    lift: Literal["relative", "absolute"],
) -> float:
    if alt_rate is not None:
        return alt_rate
    if alt_lift is None:
        raise ValueError("Provide either alt_lift or alt_rate")
    if lift == "relative":
        return baseline * (1 + alt_lift)
    if lift == "absolute":
        return baseline + alt_lift
    raise NotImplementedError(f"lift '{lift}' not implemented")


def estimate_beta_binomial_params(
    successes: np.ndarray[Any, Any] | list[Any],
    trials: np.ndarray[Any, Any] | list[Any],
) -> tuple[float, float]:
    """Estimate Beta(a, b) parameters via method of moments from cluster data.

    Fits the beta-binomial model to observed per-cluster success counts.
    The estimated Beta distribution captures both the arm-level conversion
    rate and the between-cluster variability. The within-cluster sampling
    variance is subtracted so that only true between-cluster heterogeneity
    contributes to the estimate.

    Parameters
    ----------
    successes : array-like
        Number of successes in each cluster.
    trials : array-like
        Number of trials in each cluster.

    Returns
    -------
    tuple[float, float]
        ``(a, b)`` parameters of the fitted Beta distribution.

    Raises
    ------
    ValueError
        If fewer than 2 clusters are provided, any trial count is < 1,
        or any successes exceed their trial count.
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    if len(successes) < 2:
        raise ValueError("At least 2 clusters are required for estimation")
    if np.any(trials < 1):
        raise ValueError("All trial counts must be >= 1")
    if np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("Successes must be between 0 and trials for each cluster")

    proportions = successes / trials
    mu = float(np.mean(proportions))

    mu = np.clip(mu, 1e-9, 1 - 1e-9)

    v = float(np.var(proportions, ddof=1))

    sampling_var = float(mu * (1 - mu) * np.mean(1.0 / trials))
    between_var = max(v - sampling_var, 1e-12)

    concentration = mu * (1 - mu) / between_var - 1
    concentration = max(concentration, 2.0)

    a = mu * concentration
    b = (1 - mu) * concentration
    return a, b


def beta_binomial_icc(a: float, b: float) -> float:
    """Compute the intra-cluster correlation implied by Beta(a, b).

    For a beta-binomial model where cluster proportions are drawn from
    Beta(a, b), the ICC equals ``1 / (a + b + 1)``.

    Parameters
    ----------
    a : float
        Alpha parameter of the Beta distribution.
    b : float
        Beta parameter of the Beta distribution.

    Returns
    -------
    float
        Intra-cluster correlation coefficient in (0, 1).
    """
    return 1.0 / (a + b + 1.0)


class BayesianClusterRandomizedTrial:
    """Bayesian analysis of a two-group cluster-randomized trial.

    Collects per-cluster binomial observations via :meth:`add`, fits a
    beta-binomial hierarchical model per arm using method-of-moments
    estimation, and produces posterior comparisons via :meth:`analyze`.

    Parameters
    ----------
    name : str
        Experiment name.
    metric_name : str
        Metric being measured (e.g. ``"conversion"``).
    """

    def __init__(
        self,
        name: str = "Bayesian CRT",
        metric_name: str = "outcome",
    ) -> None:
        self.experiment_name: str = name
        self.metric_name: str = metric_name
        self._clusters: dict[str, dict[str, dict[str, int]]] = {}
        self._groups: list[str] = []
        self.pooled_results: dict[str, Any] | None = None
        self.cluster_results: dict[str, dict[str, Any]] | None = None
        self.model_params: dict[str, Any] | None = None

    def add(
        self,
        cluster_name: str,
        successes: int,
        trials: int,
        *,
        group: str,
    ) -> Self:
        """Add a cluster observation.

        Parameters
        ----------
        cluster_name : str
            Unique identifier for this cluster within its group.
        successes : int
            Number of successes in the cluster.
        trials : int
            Number of trials in the cluster.
        group : str
            Group name (e.g. ``"Control"``, ``"Treatment"``). Keyword-only.

        Returns
        -------
        BayesianClusterRandomizedTrial
            Self, for method chaining.
        """
        if group not in self._groups:
            if len(self._groups) >= 2:
                raise ValueError(f"Only 2 groups are supported, got third group {group!r}")
            self._groups.append(group)

        if group not in self._clusters:
            self._clusters[group] = {}
        if cluster_name in self._clusters[group]:
            raise ValueError(f"Group {group!r} already has cluster {cluster_name!r}")

        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if successes < 0 or successes > trials:
            raise ValueError(f"successes must be between 0 and trials, got {successes}")

        self._clusters[group][cluster_name] = {
            "successes": successes,
            "trials": trials,
        }
        return self

    def _build_arm_arrays(self) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        if len(self._groups) != 2:
            raise ValueError(f"analyze requires exactly 2 groups, got {len(self._groups)}")
        result: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
        for g in self._groups:
            clusters = self._clusters[g]
            if len(clusters) < 2:
                raise ValueError(f"Group {g!r} has {len(clusters)} cluster(s), need at least 2")
            s = np.array([c["successes"] for c in clusters.values()], dtype=float)
            n = np.array([c["trials"] for c in clusters.values()], dtype=float)
            result[g] = (s, n)
        return result

    def _fit_model(self) -> dict[str, Any]:
        arms = self._build_arm_arrays()
        params: dict[str, Any] = {}
        all_s: list[np.ndarray[Any, Any]] = []
        all_n: list[np.ndarray[Any, Any]] = []

        for g in self._groups:
            s, n = arms[g]
            a, b = estimate_beta_binomial_params(s, n)
            icc = beta_binomial_icc(a, b)
            mu = float(np.mean(s / n))
            total_s = float(np.sum(s))
            total_n = float(np.sum(n))
            avg_m = float(np.mean(n))
            deff = 1.0 + (avg_m - 1.0) * icc
            n_eff = total_n / deff
            s_eff = total_s / deff
            post_a = 1.0 + s_eff
            post_b = 1.0 + n_eff - s_eff
            params[g] = {
                "a": a,
                "b": b,
                "icc": icc,
                "mean": mu,
                "deff": deff,
                "n_eff": n_eff,
                "post_a": post_a,
                "post_b": post_b,
            }
            all_s.append(s)
            all_n.append(n)

        pooled_s = np.concatenate(all_s)
        pooled_n = np.concatenate(all_n)
        pooled_a, pooled_b = estimate_beta_binomial_params(pooled_s, pooled_n)
        params["pooled_icc"] = beta_binomial_icc(pooled_a, pooled_b)

        self.model_params = params
        return params

    @property
    def icc(self) -> dict[str, float]:
        """Per-arm intra-cluster correlation coefficients."""
        if self.model_params is None:
            self._fit_model()
        assert self.model_params is not None
        return {g: self.model_params[g]["icc"] for g in self._groups}

    @property
    def pooled_icc(self) -> float:
        """Pooled intra-cluster correlation across both arms."""
        if self.model_params is None:
            self._fit_model()
        assert self.model_params is not None
        return self.model_params["pooled_icc"]

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
        low_threshold: float = -0.1,
        high_threshold: float = 0.1,
    ) -> str:
        """Analyze the cluster-randomized experiment.

        Fits a beta-binomial model per arm, draws posterior samples for the
        arm-level rates, and computes P(T > C), expected loss, credible
        intervals, and ROPE probabilities.

        Parameters
        ----------
        lift : str, default='relative'
            ``"relative"`` or ``"absolute"``.
        confidence_level : float, default=0.95
            Probability mass for the credible interval.
        n_samples : int, default=100_000
            Number of posterior samples to draw per arm.
        cred_int_method : {"credible", "hdi"}, default="credible"
            ``"credible"`` uses equal-tailed percentiles; ``"hdi"`` uses
            the Highest Density Interval.
        low_threshold : float, default=-0.1
            Lower bound of the ROPE.
        high_threshold : float, default=0.1
            Upper bound of the ROPE.

        Returns
        -------
        str
            Formatted results table.
        """
        lift = lift.casefold()
        if lift not in _VALID_LIFTS:
            raise ValueError(f"lift must be one of {sorted(_VALID_LIFTS)}, got {lift!r}")

        params = self._fit_model()
        ctrl, treat = self._groups[0], self._groups[1]

        samples_c = np.random.beta(params[ctrl]["post_a"], params[ctrl]["post_b"], n_samples)
        samples_t = np.random.beta(params[treat]["post_a"], params[treat]["post_b"], n_samples)

        if lift == "relative":
            safe_c = np.where(samples_c == 0, 1e-9, samples_c)
            lift_samples = (samples_t - samples_c) / safe_c
        else:
            lift_samples = samples_t - samples_c

        prob_t_gt_c = float(np.mean(samples_t > samples_c))
        expected_loss = float(np.mean(np.maximum(samples_c - samples_t, 0)))
        lift_mean = float(np.mean(lift_samples))
        ci_lo, ci_hi = self._credible_interval(lift_samples, confidence_level, cred_int_method)
        prob_rope = float(np.mean((lift_samples >= low_threshold) & (lift_samples <= high_threshold)))

        p_control = params[ctrl]["mean"]
        p_treatment = params[treat]["mean"]

        self.pooled_results = {
            "lift_type": lift,
            "lift": lift_mean,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_control": p_control,
            "p_treatment": p_treatment,
            "prob_t_gt_c": prob_t_gt_c,
            "expected_loss": expected_loss,
            "prob_rope": prob_rope,
        }

        return self._format_analyze(lift, confidence_level)

    def _format_analyze(self, lift: str, confidence_level: float) -> str:
        assert self.pooled_results is not None
        assert self.model_params is not None
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

        table_headers = (
            ["Metric", "Metric Name"]
            + self._groups
            + [
                "Lift",
                "Cred. Int. Lower **",
                "Cred. Int. Upper **",
                f"Prob {self._groups[1]} Is Best",
                f"Expected Loss of {self._groups[1]}",
                "Probability Lift is in ROPE ***",
            ]
        )
        table_list = [
            [lift, self.metric_name]
            + [fmt_rate(r["p_control"]), fmt_rate(r["p_treatment"])]
            + [fmt(r["lift"]), fmt(r["ci_lower"]), fmt(r["ci_upper"])]
            + [str_prob]
            + [convert_to_tabulate_str(r["expected_loss"], "relative")]
            + [convert_to_tabulate_str(r["prob_rope"], "relative")]
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")

        ctrl, treat = self._groups[0], self._groups[1]
        n_ctrl = len(self._clusters[ctrl])
        n_treat = len(self._clusters[treat])
        icc_ctrl = self.model_params[ctrl]["icc"]
        icc_treat = self.model_params[treat]["icc"]
        p_icc = self.model_params["pooled_icc"]
        return_string += (
            f"\nICC: {ctrl}={icc_ctrl:.4f}, {treat}={icc_treat:.4f} (pooled={p_icc:.4f})"
            f" | Clusters: {n_ctrl} {ctrl}, {n_treat} {treat}"
        )

        ci_pct = int(confidence_level * 100)
        return_string += f"\n* next to the prob means it exceeds our confidence level at {ci_pct}% level"
        return_string += f"\n** {ci_pct}% Credible Interval"
        return_string += "\n*** Region of Practical Equivalence"
        return return_string

    def analyze_by_cluster(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
    ) -> str:
        """Analyze each cluster individually using Beta posteriors.

        Each cluster gets a Beta(1 + s, 1 + n - s) posterior (uniform prior)
        and the resulting mean and credible interval are reported.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples per cluster.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Method for computing credible intervals.

        Returns
        -------
        str
            Table with per-cluster posterior estimates.
        """
        if len(self._groups) != 2:
            raise ValueError(f"analyze requires exactly 2 groups, got {len(self._groups)}")

        self.cluster_results = {}
        table_list = []

        for g in self._groups:
            for c_name, data in self._clusters[g].items():
                s, n = data["successes"], data["trials"]
                post_a = 1.0 + s
                post_b = 1.0 + n - s
                samples = np.random.beta(post_a, post_b, n_samples)
                mean = float(np.mean(samples))
                ci_lo, ci_hi = self._credible_interval(samples, confidence_level, cred_int_method)

                self.cluster_results[f"{g}:{c_name}"] = {
                    "group": g,
                    "cluster": c_name,
                    "mean": mean,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "trials": n,
                }

                table_list.append(
                    [
                        g,
                        c_name,
                        convert_to_tabulate_str(mean, "absolute"),
                        convert_to_tabulate_str(ci_lo, "absolute"),
                        convert_to_tabulate_str(ci_hi, "absolute"),
                        n,
                    ]
                )

        table_headers = ["Group", "Cluster", "Post. Mean", "CI Lower **", "CI Upper **", "N"]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f")
        return_string += f"\n** {int(confidence_level * 100)}% Credible Interval"
        return return_string

    def summary(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        low_threshold: float = -0.1,
        high_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Return a dict of analysis results.

        Calls :meth:`analyze` if results have not been computed yet, then
        returns a dict combining the pooled results and model parameters.

        Parameters
        ----------
        lift : str, default='relative'
            Lift type.
        confidence_level : float, default=0.95
            Credible interval probability mass.
        n_samples : int, default=100_000
            Number of posterior samples.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Credible interval method.
        low_threshold : float, default=-0.1
            Lower ROPE bound.
        high_threshold : float, default=0.1
            Upper ROPE bound.

        Returns
        -------
        dict
            Combined pooled results and model parameters.
        """
        if self.pooled_results is None:
            self.analyze(
                lift=lift,
                confidence_level=confidence_level,
                n_samples=n_samples,
                cred_int_method=cred_int_method,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
        assert self.pooled_results is not None
        assert self.model_params is not None
        return {**self.pooled_results, "model_params": self.model_params}

    def plot(
        self,
        lift: str = "relative",
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Forest plot of per-cluster proportions and pooled credible interval.

        Each cluster is shown as a small dot at its observed proportion.
        The pooled arm-level posterior mean is shown as a diamond with
        credible-interval whiskers.

        Parameters
        ----------
        lift : str, default='relative'
            Lift type for the pooled credible interval.
        confidence_level : float, default=0.95
            Probability mass for credible intervals.
        n_samples : int, default=100_000
            Number of posterior samples.
        cred_int_method : {"credible", "hdi"}, default="credible"
            Credible interval method.
        reverse_plot : bool, default=True
            Whether to reverse the y-axis order.
        color : str, list, dict, or None, default=None
            Colorblind palette name, mapping of group names to colors,
            list of colors, or ``None`` for Plotly defaults.
        """
        if self.pooled_results is None or self.model_params is None:
            self.analyze(
                lift=lift,
                confidence_level=confidence_level,
                n_samples=n_samples,
                cred_int_method=cred_int_method,
            )
        assert self.model_params is not None

        plot_color = resolve_plot_color(color)
        fig = go.Figure()  # type: ignore[attr-defined]

        y_labels: list[str] = []
        for gi, g in enumerate(self._groups):
            c = None
            if plot_color is not None:
                if isinstance(plot_color, list):
                    c = plot_color[gi % len(plot_color)]
                elif isinstance(plot_color, dict):
                    c = plot_color.get(g)

            for c_name, data in self._clusters[g].items():
                prop = data["successes"] / data["trials"]
                label = f"{g}: {c_name}"
                y_labels.append(label)
                marker_kw: dict[str, Any] = {"symbol": "circle", "size": 6}
                if c is not None:
                    marker_kw["color"] = c
                fig.add_trace(
                    go.Scatter(  # type: ignore[attr-defined]
                        x=[prop],
                        y=[label],
                        marker=marker_kw,
                        name=label,
                        showlegend=False,
                    )
                )

            params = self.model_params[g]
            samples_arm = np.random.beta(params["post_a"], params["post_b"], n_samples)
            arm_mean = float(np.mean(samples_arm))
            ci_lo, ci_hi = self._credible_interval(samples_arm, confidence_level, cred_int_method)

            label_arm = f"{g} (pooled)"
            y_labels.append(label_arm)
            marker_pooled: dict[str, Any] = {"symbol": "diamond", "size": 14}
            error_x_pooled: dict[str, Any] = {
                "type": "data",
                "symmetric": False,
                "array": [ci_hi - arm_mean],
                "arrayminus": [arm_mean - ci_lo],
                "visible": True,
            }
            if c is not None:
                marker_pooled["color"] = c
                error_x_pooled["color"] = c
            fig.add_trace(
                go.Scatter(  # type: ignore[attr-defined]
                    x=[arm_mean],
                    y=[label_arm],
                    marker=marker_pooled,
                    error_x=error_x_pooled,
                    name=label_arm,
                    showlegend=False,
                )
            )

        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"{self.experiment_name} — {self.metric_name} (Cluster Proportions)",
            xaxis_tickformat=",.1%",
            showlegend=False,
        )
        if reverse_plot:
            fig.update_layout(yaxis={"autorange": "reversed"})
        fig.show()  # type: ignore[no-untyped-call]

    def plot_pdf(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> go.Figure:
        """Plot overlapping Beta posterior PDFs for both arms.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Probability mass for HDI annotation bars.
        n_samples : int, default=100_000
            Number of posterior samples for HDI computation.
        color : str, list, dict, or None, default=None
            Colorblind palette name, list of colors, or ``None``.

        Returns
        -------
        go.Figure
            Plotly figure with overlapping PDF curves.
        """
        if self.model_params is None:
            self._fit_model()
        assert self.model_params is not None

        plot_color = resolve_plot_color(color)
        fig = go.Figure()  # type: ignore[attr-defined]

        x = np.linspace(0.001, 0.999, 1000)

        for gi, g in enumerate(self._groups):
            params = self.model_params[g]
            a, b = params["a"], params["b"]
            y = ss.beta.pdf(x, a, b)

            c = None
            if plot_color is not None:
                if isinstance(plot_color, list):
                    c = plot_color[gi % len(plot_color)]
                elif isinstance(plot_color, dict):
                    c = plot_color.get(g)

            line_kw: dict[str, Any] = {"width": 2}
            if c is not None:
                line_kw["color"] = c

            fig.add_trace(
                go.Scatter(  # type: ignore[attr-defined]
                    x=x.tolist(),
                    y=y.tolist(),
                    mode="lines",
                    line=line_kw,
                    name=f"{g} (a={a:.2f}, b={b:.2f})",
                )
            )

            samples = np.random.beta(a, b, n_samples)
            hdi_lo, hdi_hi = calculate_hdi_from_samples(samples, confidence_level)
            fig.add_shape(
                type="line",
                x0=hdi_lo,
                x1=hdi_hi,
                y0=-0.02 * max(y) * (gi + 1),
                y1=-0.02 * max(y) * (gi + 1),
                line={"color": c or "gray", "width": 4},
            )

        prob_t_gt_c = None
        if self.pooled_results is not None:
            prob_t_gt_c = self.pooled_results["prob_t_gt_c"]

        title = f"{self.experiment_name} — {self.metric_name} Posterior"
        if prob_t_gt_c is not None:
            title += f" | P({self._groups[1]} > {self._groups[0]}) = {prob_t_gt_c:.4f}"

        fig.update_layout(
            title=title,
            xaxis_title="Conversion Rate",
            yaxis_title="Density",
            xaxis_tickformat=",.1%",
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        return fig


# ---------------------------------------------------------------------------
# Power / Assurance
# ---------------------------------------------------------------------------


def _vectorized_icc(
    successes: np.ndarray[Any, Any],
    cluster_size: int,
) -> np.ndarray[Any, Any]:
    """Vectorized ICC estimation for equal-sized clusters.

    Parameters
    ----------
    successes : ndarray, shape (n_samples, n_clusters)
        Success counts per cluster per experiment.
    cluster_size : int
        Common cluster size.

    Returns
    -------
    icc : ndarray, shape (n_samples,)
        Estimated ICC, clamped to [1e-6, 1-1e-6].
    """
    proportions = successes / cluster_size
    mu = np.mean(proportions, axis=1)
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    v = np.var(proportions, axis=1, ddof=1)
    sampling_var = mu * (1 - mu) / cluster_size
    between_var = np.maximum(v - sampling_var, 1e-12)
    concentration = mu * (1 - mu) / between_var - 1
    concentration = np.maximum(concentration, 2.0)
    icc = 1.0 / (concentration + 1.0)
    return np.clip(icc, 1e-6, 1 - 1e-6)


def _simulate_crt_experiment(
    n_clusters: int,
    cluster_size: int,
    a_ctrl: float,
    b_ctrl: float,
    a_treat: float,
    b_treat: float,
    n_samples: int,
    mc_samples: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Simulate CRT experiments with design-effect-adjusted posteriors.

    For each simulated experiment, generates cluster data from the
    beta-binomial model, estimates the ICC, computes the design-effect-
    adjusted effective sample size, and draws posterior samples from
    the adjusted Beta posterior.

    Returns
    -------
    tuple of ndarray
        ``(samples_ctrl, samples_treat)``, each of shape
        ``(n_samples, mc_samples)``.
    """
    total_n = n_clusters * cluster_size

    theta_c = np.random.beta(a_ctrl, b_ctrl, (n_samples, n_clusters))
    y_c = np.random.binomial(cluster_size, theta_c)
    icc_c = _vectorized_icc(y_c, cluster_size)
    deff_c = 1.0 + (cluster_size - 1.0) * icc_c
    n_eff_c = total_n / deff_c
    s_total_c = np.sum(y_c, axis=1).astype(float)
    s_eff_c = s_total_c / deff_c
    post_a_c = 1.0 + s_eff_c
    post_b_c = 1.0 + n_eff_c - s_eff_c
    post_b_c = np.maximum(post_b_c, 0.01)
    samples_ctrl = np.random.beta(post_a_c[:, np.newaxis], post_b_c[:, np.newaxis], (n_samples, mc_samples))

    theta_t = np.random.beta(a_treat, b_treat, (n_samples, n_clusters))
    y_t = np.random.binomial(cluster_size, theta_t)
    icc_t = _vectorized_icc(y_t, cluster_size)
    deff_t = 1.0 + (cluster_size - 1.0) * icc_t
    n_eff_t = total_n / deff_t
    s_total_t = np.sum(y_t, axis=1).astype(float)
    s_eff_t = s_total_t / deff_t
    post_a_t = 1.0 + s_eff_t
    post_b_t = 1.0 + n_eff_t - s_eff_t
    post_b_t = np.maximum(post_b_t, 0.01)
    samples_treat = np.random.beta(post_a_t[:, np.newaxis], post_b_t[:, np.newaxis], (n_samples, mc_samples))

    return samples_ctrl, samples_treat


def _icc_to_beta_params(mu: float, icc: float) -> tuple[float, float]:
    """Convert a mean rate and ICC to Beta(a, b) parameters."""
    if icc <= 0 or icc >= 1:
        raise ValueError(f"icc must be in (0, 1), got {icc}")
    concentration = 1.0 / icc - 1.0
    a = mu * concentration
    b = (1 - mu) * concentration
    return a, b


def cluster_bayes_power_lift(
    n_clusters: int,
    cluster_size: int,
    icc: float,
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    n_samples: int = 10_000,
    mc_samples: int = 1_000,
    confidence_level: float = 0.95,
) -> float:
    """Estimate Bayesian power (assurance) for a cluster-randomized trial.

    Simulates ``n_samples`` CRT experiments under the alternative hypothesis.
    For each, generates cluster data from a beta-binomial model, fits
    posteriors via method of moments, and counts a "win" when
    P(T > C) >= ``confidence_level``.

    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm.
    cluster_size : int
        Number of observations per cluster (equal sizes assumed).
    icc : float
        Intra-cluster correlation coefficient in (0, 1).
    baseline : float
        Expected control conversion rate.
    alt_lift : float, optional
        Expected treatment lift over control.
    alt_rate : float, optional
        Treatment conversion rate specified directly.
    lift : {"relative", "absolute"}, default="relative"
        How ``alt_lift`` is applied to ``baseline``.
    n_samples : int, default=10_000
        Number of simulated experiments.
    mc_samples : int, default=1_000
        Posterior draws per simulated experiment.
    confidence_level : float, default=0.95
        P(T > C) threshold defining a "win".

    Returns
    -------
    float
        Estimated power in [0, 1].
    """
    alt_rate_val = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    a_ctrl, b_ctrl = _icc_to_beta_params(baseline, icc)
    a_treat, b_treat = _icc_to_beta_params(alt_rate_val, icc)

    samples_ctrl, samples_treat = _simulate_crt_experiment(
        n_clusters, cluster_size, a_ctrl, b_ctrl, a_treat, b_treat, n_samples, mc_samples
    )

    prob_b_better = np.mean(samples_treat > samples_ctrl, axis=1)
    return float(np.mean(prob_b_better >= confidence_level))


def cluster_bayes_power_loss(
    n_clusters: int,
    cluster_size: int,
    icc: float,
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    n_samples: int = 10_000,
    mc_samples: int = 1_000,
    loss_threshold: float = 0.001,
) -> float:
    """Estimate Bayesian power via expected loss for a cluster-randomized trial.

    A simulation counts as a "win" when E[max(C - T, 0)] <= ``loss_threshold``.

    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm.
    cluster_size : int
        Number of observations per cluster.
    icc : float
        Intra-cluster correlation coefficient in (0, 1).
    baseline : float
        Expected control conversion rate.
    alt_lift : float, optional
        Expected treatment lift over control.
    alt_rate : float, optional
        Treatment conversion rate specified directly.
    lift : {"relative", "absolute"}, default="relative"
        How ``alt_lift`` is applied to ``baseline``.
    n_samples : int, default=10_000
        Number of simulated experiments.
    mc_samples : int, default=1_000
        Posterior draws per simulated experiment.
    loss_threshold : float, default=0.001
        Maximum acceptable expected loss.

    Returns
    -------
    float
        Estimated power in [0, 1].
    """
    alt_rate_val = _resolve_alt_rate(baseline, alt_lift, alt_rate, lift)
    a_ctrl, b_ctrl = _icc_to_beta_params(baseline, icc)
    a_treat, b_treat = _icc_to_beta_params(alt_rate_val, icc)

    samples_ctrl, samples_treat = _simulate_crt_experiment(
        n_clusters, cluster_size, a_ctrl, b_ctrl, a_treat, b_treat, n_samples, mc_samples
    )

    expected_loss = np.mean(np.maximum(samples_ctrl - samples_treat, 0), axis=1)
    return float(np.mean(expected_loss <= loss_threshold))


def _search_min_clusters(
    power_fn: Any,
    target_power: float,
    max_clusters: int,
    error_message: str,
) -> int:
    low, high = 2, 4
    while high <= max_clusters:
        if power_fn(high) >= target_power:
            break
        low, high = high, high * 2
    else:
        raise ValueError(error_message)

    while high - low > 1:
        mid = (low + high) // 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def _search_min_lift(
    power_fn: Any,
    target_power: float,
    max_lift: float,
    tol: float,
    error_message: str,
) -> float:
    low, high = 0.0, 0.01
    while high <= max_lift:
        if power_fn(high) >= target_power:
            break
        low, high = high, high * 2
    else:
        raise ValueError(error_message)

    while high - low > tol:
        mid = (low + high) / 2
        if power_fn(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def cluster_bayes_minimum_clusters(
    icc: float,
    cluster_size: int,
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_clusters: int = 500,
) -> int:
    """Find the minimum clusters per arm for target Bayesian power via P(T > C).

    Parameters
    ----------
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Observations per cluster.
    baseline : float
        Control conversion rate.
    alt_lift : float, optional
        Treatment lift.
    alt_rate : float, optional
        Treatment rate directly.
    lift : {"relative", "absolute"}, default="relative"
        How ``alt_lift`` is applied.
    target_power : float, default=0.80
        Minimum acceptable power.
    confidence_level : float, default=0.95
        P(T > C) threshold.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    max_clusters : int, default=500
        Upper bound on cluster search.

    Returns
    -------
    int
        Minimum clusters per arm.
    """

    def _power(k: int) -> float:
        return cluster_bayes_power_lift(
            n_clusters=k,
            cluster_size=cluster_size,
            icc=icc,
            baseline=baseline,
            alt_lift=alt_lift,
            alt_rate=alt_rate,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            confidence_level=confidence_level,
        )

    return _search_min_clusters(
        _power,
        target_power,
        max_clusters,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"{max_clusters} clusters per arm. "
            "Consider a larger effect size or lower ICC."
        ),
    )


def cluster_bayes_minimum_clusters_loss(
    icc: float,
    cluster_size: int,
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    target_power: float = 0.80,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_clusters: int = 500,
) -> int:
    """Find the minimum clusters per arm for target Bayesian power via expected loss.

    Parameters
    ----------
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Observations per cluster.
    baseline : float
        Control conversion rate.
    alt_lift : float, optional
        Treatment lift.
    alt_rate : float, optional
        Treatment rate directly.
    lift : {"relative", "absolute"}, default="relative"
        How ``alt_lift`` is applied.
    target_power : float, default=0.80
        Minimum acceptable power.
    loss_threshold : float, default=0.001
        Maximum acceptable expected loss.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    max_clusters : int, default=500
        Upper bound on cluster search.

    Returns
    -------
    int
        Minimum clusters per arm.
    """

    def _power(k: int) -> float:
        return cluster_bayes_power_loss(
            n_clusters=k,
            cluster_size=cluster_size,
            icc=icc,
            baseline=baseline,
            alt_lift=alt_lift,
            alt_rate=alt_rate,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            loss_threshold=loss_threshold,
        )

    return _search_min_clusters(
        _power,
        target_power,
        max_clusters,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"{max_clusters} clusters per arm. "
            "Consider a larger effect size or lower ICC."
        ),
    )


def cluster_bayes_minimum_detectable_lift(
    n_clusters: int,
    cluster_size: int,
    icc: float,
    baseline: float,
    lift: Literal["relative", "absolute"] = "relative",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_lift: float = 10.0,
    tol: float = 0.0001,
) -> float:
    """Find the minimum detectable lift for a Bayesian CRT via P(T > C).

    Parameters
    ----------
    n_clusters : int
        Clusters per arm.
    cluster_size : int
        Observations per cluster.
    icc : float
        Intra-cluster correlation coefficient.
    baseline : float
        Control conversion rate.
    lift : {"relative", "absolute"}, default="relative"
        How the searched lift is applied to ``baseline``.
    target_power : float, default=0.80
        Minimum acceptable power.
    confidence_level : float, default=0.95
        P(T > C) threshold.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    max_lift : float, default=10.0
        Upper bound on lift search.
    tol : float, default=0.0001
        Convergence tolerance.

    Returns
    -------
    float
        Minimum detectable lift.
    """

    def _power(alt_lift_val: float) -> float:
        return cluster_bayes_power_lift(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            baseline=baseline,
            alt_lift=alt_lift_val,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            confidence_level=confidence_level,
        )

    return _search_min_lift(
        _power,
        target_power,
        max_lift,
        tol,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"a lift of {max_lift}. "
            "Consider more clusters or lower ICC."
        ),
    )


def cluster_bayes_minimum_detectable_lift_loss(
    n_clusters: int,
    cluster_size: int,
    icc: float,
    baseline: float,
    lift: Literal["relative", "absolute"] = "relative",
    target_power: float = 0.80,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    max_lift: float = 10.0,
    tol: float = 0.0001,
) -> float:
    """Find the minimum detectable lift for a Bayesian CRT via expected loss.

    Parameters
    ----------
    n_clusters : int
        Clusters per arm.
    cluster_size : int
        Observations per cluster.
    icc : float
        Intra-cluster correlation coefficient.
    baseline : float
        Control conversion rate.
    lift : {"relative", "absolute"}, default="relative"
        How the searched lift is applied to ``baseline``.
    target_power : float, default=0.80
        Minimum acceptable power.
    loss_threshold : float, default=0.001
        Maximum acceptable expected loss.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    max_lift : float, default=10.0
        Upper bound on lift search.
    tol : float, default=0.0001
        Convergence tolerance.

    Returns
    -------
    float
        Minimum detectable lift.
    """

    def _power(alt_lift_val: float) -> float:
        return cluster_bayes_power_loss(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            baseline=baseline,
            alt_lift=alt_lift_val,
            lift=lift,
            n_samples=n_samples,
            mc_samples=mc_samples,
            loss_threshold=loss_threshold,
        )

    return _search_min_lift(
        _power,
        target_power,
        max_lift,
        tol,
        error_message=(
            f"Could not reach target power of {target_power} within "
            f"a lift of {max_lift}. "
            "Consider more clusters or lower ICC."
        ),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cluster_bayes_power_curve(
    icc: float,
    cluster_size: int,
    baseline: float,
    alt_lift: float | None = None,
    alt_rate: float | None = None,
    lift: Literal["relative", "absolute"] = "relative",
    decision: Literal["lift", "loss"] = "lift",
    confidence_level: float = 0.95,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    cluster_counts: np.ndarray[Any, Any] | list[int] | None = None,
    n_points: int = 20,
) -> go.Figure:
    """Plot Bayesian CRT power as a function of clusters per arm.

    Parameters
    ----------
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Observations per cluster.
    baseline : float
        Control conversion rate.
    alt_lift : float, optional
        Treatment lift.
    alt_rate : float, optional
        Treatment rate directly.
    lift : {"relative", "absolute"}, default="relative"
        How ``alt_lift`` is applied.
    decision : {"lift", "loss"}, default="lift"
        Decision rule.
    confidence_level : float, default=0.95
        P(T > C) threshold when ``decision="lift"``.
    loss_threshold : float, default=0.001
        Expected loss threshold when ``decision="loss"``.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    cluster_counts : array-like or None, default=None
        Explicit cluster counts to evaluate. When ``None``, auto-ranges.
    n_points : int, default=20
        Number of points when ``cluster_counts`` is ``None``.

    Returns
    -------
    go.Figure
    """
    power_fn = cluster_bayes_power_lift if decision == "lift" else cluster_bayes_power_loss

    if cluster_counts is None:
        cluster_counts = np.linspace(3, 60, n_points, dtype=int).tolist()
        cluster_counts = sorted(set(cluster_counts))

    powers = []
    for k in cluster_counts:
        kwargs: dict[str, Any] = {
            "n_clusters": int(k),
            "cluster_size": cluster_size,
            "icc": icc,
            "baseline": baseline,
            "alt_lift": alt_lift,
            "alt_rate": alt_rate,
            "lift": lift,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
        }
        if decision == "lift":
            kwargs["confidence_level"] = confidence_level
        else:
            kwargs["loss_threshold"] = loss_threshold
        powers.append(power_fn(**kwargs))

    fig = go.Figure()  # type: ignore[attr-defined]
    fig.add_trace(
        go.Scatter(  # type: ignore[attr-defined]
            x=list(cluster_counts),
            y=powers,
            mode="lines+markers",
            line={"color": "#636EFA", "width": 2},
            name="Power",
        )
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="gray",
        annotation_text="80% power",
        annotation_position="top left",
    )

    rule = f"P(T>C) >= {confidence_level}" if decision == "lift" else f"E[loss] <= {loss_threshold}"
    fig.update_layout(
        title=f"Bayesian CRT Power Curve ({rule})",
        xaxis_title="Clusters per arm",
        yaxis_title="Power (Assurance)",
        yaxis_range=[0, 1.05],
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def plot_cluster_bayes_sensitivity_curve(
    icc: float,
    cluster_size: int,
    baseline: float,
    lift: Literal["relative", "absolute"] = "relative",
    decision: Literal["lift", "loss"] = "lift",
    target_power: float = 0.80,
    confidence_level: float = 0.95,
    loss_threshold: float = 0.001,
    n_samples: int = 10_000,
    mc_samples: int = 500,
    cluster_counts: np.ndarray[Any, Any] | list[int] | None = None,
    n_points: int = 20,
) -> go.Figure:
    """Plot minimum detectable lift as a function of clusters per arm.

    Parameters
    ----------
    icc : float
        Intra-cluster correlation coefficient.
    cluster_size : int
        Observations per cluster.
    baseline : float
        Control conversion rate.
    lift : {"relative", "absolute"}, default="relative"
        How the lift is applied to ``baseline``.
    decision : {"lift", "loss"}, default="lift"
        Decision rule.
    target_power : float, default=0.80
        Minimum acceptable power.
    confidence_level : float, default=0.95
        P(T > C) threshold when ``decision="lift"``.
    loss_threshold : float, default=0.001
        Expected loss threshold when ``decision="loss"``.
    n_samples : int, default=10_000
        Simulated experiments per evaluation.
    mc_samples : int, default=500
        Posterior draws per experiment.
    cluster_counts : array-like or None, default=None
        Explicit cluster counts to evaluate. When ``None``, auto-ranges.
    n_points : int, default=20
        Number of points when ``cluster_counts`` is ``None``.

    Returns
    -------
    go.Figure
    """
    mdl_fn = cluster_bayes_minimum_detectable_lift if decision == "lift" else cluster_bayes_minimum_detectable_lift_loss

    if cluster_counts is None:
        cluster_counts = np.linspace(5, 60, n_points, dtype=int).tolist()
        cluster_counts = sorted(set(cluster_counts))

    mdls = []
    for k in cluster_counts:
        kwargs: dict[str, Any] = {
            "n_clusters": int(k),
            "cluster_size": cluster_size,
            "icc": icc,
            "baseline": baseline,
            "lift": lift,
            "target_power": target_power,
            "n_samples": n_samples,
            "mc_samples": mc_samples,
        }
        if decision == "lift":
            kwargs["confidence_level"] = confidence_level
        else:
            kwargs["loss_threshold"] = loss_threshold
        mdls.append(mdl_fn(**kwargs))

    fig = go.Figure()  # type: ignore[attr-defined]
    fig.add_trace(
        go.Scatter(  # type: ignore[attr-defined]
            x=list(cluster_counts),
            y=mdls,
            mode="lines+markers",
            line={"color": "#636EFA", "width": 2},
            name="MDL",
        )
    )

    y_label = f"Minimum detectable {lift} lift"
    rule = f"P(T>C) >= {confidence_level}" if decision == "lift" else f"E[loss] <= {loss_threshold}"
    fig.update_layout(
        title=f"Bayesian CRT Sensitivity Curve ({rule})",
        xaxis_title="Clusters per arm",
        yaxis_title=y_label,
        yaxis_tickformat=",.0%",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig
