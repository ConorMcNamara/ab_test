"""Our wrapper for analyzing experiment results."""

import itertools
from typing import Any, ClassVar, Literal, cast

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta
from tabulate import tabulate

from ab_test._contingency import BaseContingencyTable
from ab_test._display import (
    apply_dark_mode,
    convert_to_tabulate_str,
    format_percent,
    resolve_plot_color,
    tabulate_summary,
)
from ab_test._lift import scale_bounds, scale_metric
from ab_test.bayesian_binomial.credible_intervals import credible_interval, individual_credible_interval
from ab_test.bayesian_binomial.stats_tests import calculate_metrics, prob_lift_exceeds
from ab_test.bayesian_binomial.utils import _default_rope_half_width, posterior_mean, sample_beta

__all__ = [
    "BayesianContingencyTable",
]


class BayesianContingencyTable(BaseContingencyTable):
    """A class for analyzing experiment results using Bayesian approaches.

    Examples
    --------
    >>> table = BayesianContingencyTable("Checkout redesign", "conversion")
    >>> table = table.add("Control", 1000, 10000, alpha=1, beta=1).add("Treatment", 1100, 10000, alpha=1, beta=1)
    >>> print(table.analyze(lift="relative"))  # doctest: +SKIP
    >>> print(table.analyze_individually())  # doctest: +SKIP
    >>> table.plot(is_individual=False)  # doctest: +SKIP
    """

    _columns: ClassVar[list[str]] = ["cell_name", "successes", "trials", "alpha", "beta"]
    _pyspark_types: ClassVar[dict[str, str]] = {
        "cell_name": "StringType",
        "successes": "IntegerType",
        "trials": "IntegerType",
        "alpha": "DoubleType",
        "beta": "DoubleType",
    }

    def __init__(self, name: str, metric_name: str, spend: float | None = None, msrp: float | None = None) -> None:
        """BayesianContingencyTable is our class for creating and analyzing experiment results.

        Parameters
        ----------
        name : str
            The name of our experiment associated with our Contingency Table
        metric_name : str
            The name of our metric
        spend : float
            The amount we spent for this campaign. Used to calculate the ROAS of our campaign
        msrp : float
            The average msrp of our product. Used to calculate the revenue return of our campaign
        """
        super().__init__(name, metric_name, spend, msrp)
        self.alphas: list[float] = []
        self.betas: list[float] = []

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        return ["Total", np.sum(self.successes), np.sum(self.trials), np.nan, np.nan]

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        return {
            "successes": int(np.sum(self.successes)),
            "trials": int(np.sum(self.trials)),
            "alpha": np.nan,
            "beta": np.nan,
        }

    def _deserialize_extra(self, serial: dict[str, Any]) -> None:
        """Restore per-cell Beta prior parameters during :meth:`deserialize`."""
        self.alphas = [v["alpha"] for v in serial["table"].values()]
        self.betas = [v["beta"] for v in serial["table"].values()]

    def add(self, cell_name: str, successes: int, trials: int, alpha: float, beta: float) -> "BayesianContingencyTable":
        """Add cells to our contingency table.

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        successes : int
            The number of successes in our cell_name
        trials : int
            The number of trials in our cell_name
        alpha : float
            Alpha parameter of the Beta prior for this cell.
        beta : float
            Beta parameter of the Beta prior for this cell.

        Returns
        -------
        BayesianContingencyTable, to be chained with other methods
        """
        cell_dict = {"successes": successes, "trials": trials, "alpha": alpha, "beta": beta}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.successes.append(successes)
        self.trials.append(trials)
        self.alphas.append(alpha)
        self.betas.append(beta)
        return self

    def analyze(
        self,
        lift: str = "relative",
        cred_int_method: Literal["credible", "hdi"] = "credible",
        confidence_level: float = 0.95,
        is_sample: bool = False,
        n_samples: int = 100_000,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        comparisons: str = "control",
    ) -> str:
        """Analyze the experiment and return a formatted summary table.

        Computes Bayesian metrics for the two variants — posterior means, credible
        interval, probability B exceeds A, expected loss, and ROPE probabilities —
        then formats them into a grid table string. Results are also stored on
        ``self.incremental_results`` for programmatic access.

        Parameters
        ----------
        lift : {"relative", "absolute", "incremental", "roas", "revenue", "cpa"}, optional
            Type of lift to compute, by default ``"relative"``.
        cred_int_method : {"credible", "hdi"}, optional
            Method used to compute the credible interval, by default ``"credible"``.
        confidence_level : float, optional
            Probability mass for the credible interval and significance threshold,
            by default 0.95.
        is_sample : bool, optional
            Whether to use Monte Carlo sampling for the credible interval. If
            ``False``, uses the normal approximation, by default ``False``.
        n_samples : int, optional
            Number of posterior samples to draw, by default 100_000.
        low_threshold : float, optional
            Lower bound of the Region of Practical Equivalence (ROPE), in the
            units of ``lift`` (a fraction for relative, a rate difference for
            absolute, conversions for incremental, conversions per dollar for
            roas, currency for revenue). By default the ROPE is +/-10% of the
            control's posterior rate expressed in those units, so for relative
            lift it is +/-0.1. The ROPE is not computed for ``lift="cpa"``
            (shown as n/a), since CPA is not monotone in the lift.
        high_threshold : float, optional
            Upper bound of the ROPE, in the units of ``lift``. Defaults as
            described for ``low_threshold``.
        comparisons : {"control", "all"}, optional
            With three or more variants, which pairs to compare: each variant
            against the first cell added (the control, the default), or every
            pair. Ignored with two variants.

        Returns
        -------
        str
            A grid-formatted table summarising the lift, credible interval,
            probability B is best, expected loss, and ROPE probability, with
            footnotes explaining annotated values. The expected loss,
            E[max(-lift, 0)], is in the units of ``lift``; for ``"cpa"`` it is
            a difference in rates.

        Raises
        ------
        ValueError
            If ``lift="roas"`` or ``lift="cpa"`` and ``spend`` was not set on the table.
        ValueError
            If ``lift="revenue"`` and ``msrp`` was not set on the table.
        ValueError
            If ``lift`` is not one of the supported types.

        Notes
        -----
        With three or more variants, the summary reports each variant's
        probability of being best and its expected loss, E[best rate - its
        rate], both from one joint posterior draw across all variants, and the
        expected loss is a difference in rates whatever ``lift`` is. Each
        pairwise comparison is reported as a two-variant analysis would be,
        with its ROPE scaled to that comparison's reference variant.
        ``incremental_results`` then holds ``"prob_best"``,
        ``"expected_loss"`` (both keyed by variant) and a ``"comparisons"``
        dict keyed by labels such as ``"B vs A"``. Posterior probabilities
        need no multiple-comparison correction, but stopping as soon as one
        crosses a threshold still inflates false wins.
        """
        k = len(self.names)
        if k < 2:
            raise ValueError(f"analyze requires at least 2 variants, got {k}")
        comparisons = comparisons.casefold()
        if comparisons not in ("control", "all"):
            raise ValueError(f"comparisons must be 'control' or 'all', got {comparisons!r}")
        self._analyze_settings = {
            "cred_int_method": cred_int_method,
            "confidence_level": confidence_level,
            "is_sample": is_sample,
            "n_samples": n_samples,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "comparisons": comparisons,
        }
        lift = lift.casefold()
        settings: dict[str, Any] = {
            "lift": lift,
            "cred_int_method": cred_int_method,
            "confidence_level": confidence_level,
            "is_sample": is_sample,
            "n_samples": n_samples,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }
        if k == 2:
            return self._analyze_pair(settings)
        return self._analyze_many(comparisons, settings)

    def _compare(self, i: int, j: int, settings: dict[str, Any]) -> dict[str, Any]:
        """Compare cell ``j`` against cell ``i``: posterior lift, interval, P(j > i), loss and ROPE."""
        lift = settings["lift"]
        cred_int_method = settings["cred_int_method"]
        confidence_level = settings["confidence_level"]
        is_sample = settings["is_sample"]
        n_samples = settings["n_samples"]
        low_threshold = settings["low_threshold"]
        high_threshold = settings["high_threshold"]
        successes = [self.successes[i], self.successes[j]]
        trials = [self.trials[i], self.trials[j]]
        alphas = [self.alphas[i], self.alphas[j]]
        betas = [self.betas[i], self.betas[j]]
        if low_threshold is None or high_threshold is None:
            control_rate = posterior_mean(successes[0], trials[0], alphas[0], betas[0])
            default_rope = _default_rope_half_width(control_rate, lift, max(trials), self.spend, self.msrp)
            low_threshold = -default_rope if low_threshold is None else low_threshold
            high_threshold = default_rope if high_threshold is None else high_threshold
        if lift in ["relative", "absolute"]:
            results = calculate_metrics(
                successes,
                trials,
                alphas,
                betas,
                n_samples,
                lift,
                low_threshold,
                high_threshold,
                loss_in_lift_units=True,
            )
            lb, ub = credible_interval(
                successes,
                trials,
                alphas,
                betas,
                confidence_level,
                cast(Literal["relative", "absolute"], lift),
                is_sample,
                n_samples,
                cred_int_method,
            )
        elif lift in ["incremental", "roas", "revenue", "cpa"]:
            results = calculate_metrics(
                successes,
                trials,
                alphas,
                betas,
                n_samples,
                lift,
                low_threshold,
                high_threshold,
                spend=self.spend,
                msrp=self.msrp,
                loss_in_lift_units=True,
            )
            lb, ub = credible_interval(
                successes,
                trials,
                alphas,
                betas,
                confidence_level,
                "absolute",
                is_sample,
                n_samples,
                cred_int_method,
            )
        else:
            raise ValueError(f"No support for lift type {lift}")
        pa = posterior_mean(successes[0], trials[0], alphas[0], betas[0])
        pb = posterior_mean(successes[1], trials[1], alphas[1], betas[1])
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            # Scale unrounded values; rounding each bound separately biased the interval.
            n_scale = max(trials)
            pa, pb, lb, ub = pa * n_scale, pb * n_scale, lb * n_scale, ub * n_scale
            test_lift = scale_metric(pb - pa, lift, self.spend, self.msrp)
            pa = scale_metric(pa, lift, self.spend, self.msrp)
            pb = scale_metric(pb, lift, self.spend, self.msrp)
            lb, ub = scale_bounds(lb, ub, lift, self.spend, self.msrp)
        elif lift == "relative":
            test_lift = (pb - pa) / pa
        elif lift == "absolute":
            test_lift = pb - pa
        else:
            raise ValueError(f"lift type {lift} not supported")
        return {
            "lift": test_lift,
            "rates": [pa, pb],
            "prob_greater": results["Proportion of samples where B exceeds A"],
            "ci_lower": lb,
            "ci_upper": ub,
            "expected_loss": results["Expected loss"],
            "prob_rope": results["Probability of ROPE"],
            "prob_lift_exceeds_threshold": results[f"Probability {lift} exceeds {high_threshold}"],
            "prob_lift_below_threshold": results[f"Probability {lift} is below {low_threshold}"],
        }

    def _analyze_pair(self, settings: dict[str, Any]) -> str:
        """Two variants: one comparison, reported as before multi-arm support."""
        lift = settings["lift"]
        confidence_level = settings["confidence_level"]
        result = self._compare(0, 1, settings)
        test_lift, success_rate, lb, ub = result["lift"], result["rates"], result["ci_lower"], result["ci_upper"]
        self.incremental_results = {
            "lift_type": lift,
            "lift": test_lift,
            f"{self.names[0]}": success_rate[0],
            f"{self.names[1]}": success_rate[1],
            "prob_b_greater_a": result["prob_greater"],
            "ci_lower": lb,
            "ci_upper": ub,
            "expected_loss": result["expected_loss"],
            "prob_rope": result["prob_rope"],
            "prob_lift_exceeds_threshold": result["prob_lift_exceeds_threshold"],
            "prob_lift_below_threshold": result["prob_lift_below_threshold"],
        }
        prob_b_exceeds_a = result["prob_greater"]
        str_pvalue = (
            f"{convert_to_tabulate_str(prob_b_exceeds_a, 'relative')}*"
            if prob_b_exceeds_a >= confidence_level
            else f"{convert_to_tabulate_str(prob_b_exceeds_a, 'relative')}"
        )
        row_labels = (
            ["Metric", "Metric Name"]
            + self.names
            + [
                "Lift",
                "Cred. Int. Lower **",
                "Cred. Int. Upper **",
                f"Prob {self.names[1]} Is Best",
                # The loss is in the lift's units, except for CPA (see calculate_metrics).
                f"Expected Loss of {self.names[1]}" + (" (rate difference)" if lift == "cpa" else ""),
                "Probability Lift is in ROPE ***",
            ]
        )
        values = (
            [lift]
            + [self.metric_name]
            + convert_to_tabulate_str(success_rate, lift)
            + convert_to_tabulate_str([test_lift, lb, ub], lift)
            + [str_pvalue]
            + [convert_to_tabulate_str(result["expected_loss"], "absolute" if lift == "cpa" else lift)]
            + ["n/a" if np.isnan(result["prob_rope"]) else convert_to_tabulate_str(result["prob_rope"], "relative")]
        )
        return_string = tabulate_summary(row_labels, values)
        return_string += (
            f"\n* next to the prob means it exceeds our confidence level at {format_percent(confidence_level)}% level"
        )
        return_string += f"\n** {format_percent(confidence_level)}% Credible Interval"
        return_string += "\n*** Region of Practical Equivalence"
        return return_string

    def _analyze_many(self, comparisons: str, settings: dict[str, Any]) -> str:
        """Three or more variants: P(each arm is best), expected loss per arm, and pairwise comparisons."""
        lift = settings["lift"]
        confidence_level = settings["confidence_level"]
        k = len(self.names)
        pairs = [(0, j) for j in range(1, k)] if comparisons == "control" else list(itertools.combinations(range(k), 2))
        compared: dict[str, dict[str, Any]] = {}
        for i, j in pairs:
            result = self._compare(i, j, settings)
            compared[f"{self.names[j]} vs {self.names[i]}"] = {
                "lift": result["lift"],
                f"{self.names[i]}": result["rates"][0],
                f"{self.names[j]}": result["rates"][1],
                "prob_greater": result["prob_greater"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "expected_loss": result["expected_loss"],
                "prob_rope": result["prob_rope"],
            }
        # One joint draw across every arm for "which arm is best" and its loss.
        draws = np.column_stack(
            [
                sample_beta(s, n, a, b, settings["n_samples"])
                for s, n, a, b in zip(self.successes, self.trials, self.alphas, self.betas)
            ]
        )
        best = np.argmax(draws, axis=1)
        prob_best = {name: float(np.mean(best == index)) for index, name in enumerate(self.names)}
        regret = draws.max(axis=1, keepdims=True) - draws
        expected_loss = {name: float(regret[:, index].mean()) for index, name in enumerate(self.names)}
        means = [posterior_mean(s, n, a, b) for s, n, a, b in zip(self.successes, self.trials, self.alphas, self.betas)]
        self.incremental_results = {
            "lift_type": lift,
            "comparison_type": comparisons,
            "prob_best": prob_best,
            "expected_loss": expected_loss,
            "comparisons": compared,
        }

        arm_rows = []
        for name, mean in zip(self.names, means):
            star = "*" if prob_best[name] >= confidence_level else ""
            arm_rows.append(
                [
                    name,
                    convert_to_tabulate_str(mean, "absolute"),
                    f"{convert_to_tabulate_str(prob_best[name], 'relative')}{star}",
                    convert_to_tabulate_str(expected_loss[name], "absolute"),
                ]
            )
        return_string = tabulate_summary(["Metric", "Metric Name"], [lift, self.metric_name])
        return_string += "\n" + tabulate(
            arm_rows,
            headers=["Variant", "Posterior Mean", "Prob Is Best *", "Expected Loss (rate difference) ****"],
            tablefmt="grid",
        )
        comparison_rows = []
        for label, comparison in compared.items():
            star = "*" if comparison["prob_greater"] >= confidence_level else ""
            comparison_rows.append(
                [label]
                + convert_to_tabulate_str([comparison["lift"], comparison["ci_lower"], comparison["ci_upper"]], lift)
                + [
                    f"{convert_to_tabulate_str(comparison['prob_greater'], 'relative')}{star}",
                    convert_to_tabulate_str(comparison["expected_loss"], "absolute" if lift == "cpa" else lift),
                    "n/a"
                    if np.isnan(comparison["prob_rope"])
                    else convert_to_tabulate_str(comparison["prob_rope"], "relative"),
                ]
            )
        return_string += "\n" + tabulate(
            comparison_rows,
            headers=[
                "Comparison",
                "Lift",
                "Cred. Int. Lower **",
                "Cred. Int. Upper **",
                "Prob Greater *",
                "Expected Loss" + (" (rate difference)" if lift == "cpa" else ""),
                "Probability Lift is in ROPE ***",
            ],
            tablefmt="grid",
        )
        level = format_percent(confidence_level)
        return_string += f"\n* next to a probability means it exceeds our confidence level at {level}% level"
        return_string += f"\n** {level}% Credible Interval"
        return_string += "\n*** Region of Practical Equivalence"
        return_string += "\n**** E[best rate - this variant's rate], from one joint posterior draw across all variants"
        return return_string

    def analyze_individually(
        self,
        cred_int_method: Literal["credible", "hdi"] = "credible",
        confidence_level: float = 0.95,
    ) -> str:
        """Analyzes the individual cells using Bayesian credible intervals.

        Parameters
        ----------
        cred_int_method : {"credible", "hdi"}
            Method for calculating individual credible intervals.
        confidence_level : float
            Probability mass for the credible interval. Defaults to 0.95.

        Returns
        -------
        The results (posterior mean and credible intervals) of each cell in string format.

        Notes
        -----
        The Total row uses a pooled Beta prior — Beta(Σα_i, Σβ_i) — which aggregates
        the individual cell priors. This assumes all observations come from the same
        underlying process; if cells have meaningfully different true conversion rates,
        the true aggregate is a mixture of Beta distributions, not a single Beta.
        """
        table_list: list[list] = []
        for name_i, s_i, n_i, alpha_i, beta_i in zip(self.names, self.successes, self.trials, self.alphas, self.betas):
            success_rate = posterior_mean(s_i, n_i, alpha_i, beta_i)
            lb, ub = individual_credible_interval(s_i, n_i, confidence_level, alpha_i, beta_i, method=cred_int_method)
            name_list = [name_i, s_i, n_i, alpha_i, beta_i] + convert_to_tabulate_str(
                [success_rate, lb, ub], "absolute"
            )
            self.individual_results[name_i] = {"lift": success_rate, "ci_lower": lb, "ci_upper": ub}
            table_list.append(name_list)
        total_success, total_trials = int(np.sum(self.successes)), int(np.sum(self.trials))
        total_alpha, total_beta = float(np.sum(self.alphas)), float(np.sum(self.betas))
        total_success_rate = posterior_mean(total_success, total_trials, total_alpha, total_beta)
        lb_total, ub_total = individual_credible_interval(
            total_success, total_trials, confidence_level, total_alpha, total_beta, method=cred_int_method
        )
        total_list = ["Total", total_success, total_trials, total_alpha, total_beta] + convert_to_tabulate_str(
            [total_success_rate, lb_total, ub_total], "absolute"
        )
        self.individual_results["Total"] = {"lift": total_success_rate, "ci_lower": lb_total, "ci_upper": ub_total}
        table_list.append(total_list)
        table_headers = [
            "Cell Name",
            "Successes",
            "Trials",
            "Prior Alpha",
            "Prior Beta",
            "Posterior Mean",
            "Cred. Int. Lower**",
            "Cred. Int. Upper**",
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid")
        return_string += f"\n** {format_percent(confidence_level)}% Credible Interval"
        return return_string

    def plot_pdf(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        color: str | dict[str, Any] | list[Any] | None = None,
        *,
        dark_mode: bool = False,
    ) -> go.Figure:
        """Plot the posterior Beta distributions for each variant as an interactive figure.

        Renders overlapping PDF curves for both variants, annotates each with its
        Highest Density Interval, and titles the chart with the probability that
        variant B's conversion rate exceeds variant A's.

        Parameters
        ----------
        confidence_level : float, optional
            Probability mass used to compute each variant's HDI, by default 0.95.
        n_samples : int, optional
            Number of posterior samples drawn to estimate the win probability,
            by default 100_000.
        color : str, list, dict, or None, optional
            Controls the colors used for each variant. If None, uses Plotly's
            default color scheme. If a string, one of the colorblind-friendly
            palette names: ``"ibm"``, ``"wong"``, ``"ito"``, ``"tol"``,
            ``"tol_bright"``, ``"tol_vibrant"``, ``"tol_muted"``, ``"tol_light"``.
            If a list, each item corresponds to a variant in order. If a dict,
            keys are variant names and values are colors.
        dark_mode : bool, default=False
            Render on a dark background with light text and gridlines (Plotly's
            ``"plotly_dark"`` template).

        Returns
        -------
        go.Figure
            An interactive Plotly figure showing the posterior PDFs, HDI bars,
            and a title containing P(B > A).
        """
        # plot_pdf needs two concrete colors; fall back to Plotly's defaults when
        # no explicit color was requested.
        plot_color = resolve_plot_color(color) or ["#636EFA", "#EF553B"]
        color_a = plot_color[0] if isinstance(plot_color, list) else plot_color[self.names[0]]
        color_b = plot_color[1] if isinstance(plot_color, list) else plot_color[self.names[1]]
        # Posterior parameters: Beta(alpha + successes, beta + trials - successes)
        post_alpha_a = self.alphas[0] + self.successes[0]
        post_beta_a = self.betas[0] + self.trials[0] - self.successes[0]
        post_alpha_b = self.alphas[1] + self.successes[1]
        post_beta_b = self.betas[1] + self.trials[1] - self.successes[1]

        # 1. Define X-axis range (0 to 1, but zoomed to relevant area)
        ppf_lo = min(beta.ppf(0.001, post_alpha_a, post_beta_a), beta.ppf(0.001, post_alpha_b, post_beta_b))
        ppf_hi = max(beta.ppf(0.999, post_alpha_a, post_beta_a), beta.ppf(0.999, post_alpha_b, post_beta_b))
        x_min = max(0, ppf_lo * 0.8)
        x_max = min(1, ppf_hi * 1.2)
        x = np.linspace(x_min, x_max, 500)

        # 2. PDF Curves
        pdf_a = beta.pdf(x, post_alpha_a, post_beta_a)
        pdf_b = beta.pdf(x, post_alpha_b, post_beta_b)

        # 3. HDI Lines
        hdi_a = individual_credible_interval(
            self.successes[0], self.trials[0], confidence_level, self.alphas[0], self.betas[0], method="hdi"
        )
        hdi_b = individual_credible_interval(
            self.successes[1], self.trials[1], confidence_level, self.alphas[1], self.betas[1], method="hdi"
        )

        # 4. Win Probability for Title
        _sample_a = sample_beta(self.successes[0], self.trials[0], self.alphas[0], self.betas[0], n_samples)
        _sample_b = sample_beta(self.successes[1], self.trials[1], self.alphas[1], self.betas[1], n_samples)
        p_b_better = prob_lift_exceeds(_sample_a, _sample_b, threshold=0.0)

        fig = go.Figure()

        # Trace A (Control)
        fig.add_trace(
            go.Scatter(
                x=x, y=pdf_a, name=f"{self.names[0]}", fill="tozeroy", line=dict(color=color_a, width=2), opacity=0.4
            )
        )
        # Trace B (Variant)
        fig.add_trace(
            go.Scatter(
                x=x, y=pdf_b, name=f"{self.names[1]}", fill="tozeroy", line=dict(color=color_b, width=2), opacity=0.4
            )
        )

        # Add HDI indicators as horizontal bars at the bottom
        # We calculate the max height to position the bars relatively
        max_y = max(np.max(pdf_a), np.max(pdf_b))
        y_pos_a = -max_y * 0.05
        y_pos_b = -max_y * 0.10

        for hdi, hdi_color, y_pos in [(hdi_a, color_a, y_pos_a), (hdi_b, color_b, y_pos_b)]:
            fig.add_shape(type="line", x0=hdi[0], y0=y_pos, x1=hdi[1], y1=y_pos, line=dict(color=hdi_color, width=5))
        fig.update_layout(
            title=f"Bayesian Binary Test: P({self.names[1]} > {self.names[0]}) = {p_b_better:.1%}",
            xaxis_title="Conversion Rate (%)",
            xaxis_tickformat=".1%",
            yaxis_title="Probability Density",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        apply_dark_mode(fig, dark_mode)
        return fig
