"""Our wrapper for analyzing experiment results."""

import math
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from scipy.stats import beta
from tabulate import tabulate

from ab_test._display import convert_to_tabulate_str, render_forest_plot, resolve_plot_color
from ab_test.bayesian_binomial.credible_intervals import credible_interval, individual_credible_interval
from ab_test.bayesian_binomial.stats_tests import calculate_metrics, prob_lift_exceeds
from ab_test.bayesian_binomial.utils import posterior_mean, sample_beta


class BayesianContingencyTable:
    """A class for analyzing experiment results using Bayesian approaches."""

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
        self.experiment_name: str = name
        self.names: list[str] = []
        self.metric_name: str = metric_name
        self.spend: float | None = spend
        self.msrp: float | None = msrp
        self.cells: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": {},
        }
        self.successes: list[int] = []
        self.trials: list[int] = []
        self.alphas: list[float] = []
        self.betas: list[float] = []
        self.incremental_results: dict[str, Any] | None = None
        self.individual_results: dict[str, dict[str, float]] = {}

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

    def to_df(
        self,
        method: str = "pandas",
        include_total: bool = False,
        spark_session: Any | None = None,
        ibis_backend: Any | None = None,
    ) -> pd.DataFrame | pl.DataFrame | Any:
        """Return our BayesianContingencyTable as a DataFrame.

        Parameters
        ----------
        method : {"pandas", "polars", "pyspark", "modin", "ibis", "narwhals"}
            Whether we want our DataFrame as a pandas, polars, pyspark, modin, ibis, or narwhals DataFrame
        include_total : bool, default=False
            Whether we want to include another section with the total amount
        spark_session : SparkSession, optional
            An active SparkSession, required when method="pyspark"
        ibis_backend : ibis backend, optional
            An Ibis backend connection. When provided, it is set as the active backend before
            creating the memtable. When omitted, the existing default backend is used.

        Returns
        -------
        Our BayesianContingencyTable as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(
                self.to_list(include_total),
                columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
        elif method == "polars":
            return_df = pl.DataFrame(
                self.to_list(include_total), schema=["cell_name", "successes", "trials", "alpha", "beta"], orient="row"
            )
        elif method == "pyspark":
            from pyspark.sql import SparkSession
            from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

            if spark_session is None:
                spark_session = SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No active SparkSession found. Please provide a spark_session argument.")
            schema = StructType(
                [
                    StructField("cell_name", StringType(), True),
                    StructField("successes", IntegerType(), True),
                    StructField("trials", IntegerType(), True),
                    StructField("alpha", DoubleType(), True),
                    StructField("beta", DoubleType(), True),
                ]
            )
            return_df = spark_session.createDataFrame(self.to_list(include_total), schema=schema)
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            import modin.pandas as mpd  # type: ignore[import-not-found]

            return_df = mpd.DataFrame(
                self.to_list(include_total),
                columns=mpd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
        elif method == "ibis":
            import ibis  # type: ignore[import-not-found]

            if ibis_backend is not None:
                ibis.set_backend(ibis_backend)
            pandas_df = pd.DataFrame(
                self.to_list(include_total),
                columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
            return_df = ibis.memtable(pandas_df)
        elif method == "narwhals":
            import narwhals as nw

            pandas_df = pd.DataFrame(
                self.to_list(include_total),
                columns=pd.Index(["cell_name", "successes", "trials", "alpha", "beta"]),
            )
            return_df = nw.from_native(pandas_df)
        else:
            raise ValueError(f"Method {method} not supported for creating DataFrames")
        return return_df

    def to_list(self, include_total: bool = False) -> list[Any]:
        """Return our BayesianContingencyTable as a list.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our BayesianContingencyTable as a list
        """
        return_list = []
        for name in self.names:
            return_list.append([name] + list(self.cells["table"][name].values()))
        if include_total:
            return_list.append(["Total", np.sum(self.successes), np.sum(self.trials), np.nan, np.nan])
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray[Any, Any]:
        """Return our BayesianContingencyTable as a numpy array.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our BayesianContingencyTable as a numpy array
        """
        return np.array(self.to_list(include_total))

    def serialize(self, include_total: bool = False) -> dict[str, Any]:
        """Return our BayesianContingencyTable as a JSON, with all information.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our BayesianContingencyTable as a JSON
        """
        table = dict(self.cells["table"])
        if include_total:
            table["Total"] = {
                "successes": int(np.sum(self.successes)),
                "trials": int(np.sum(self.trials)),
                "alpha": np.nan,
                "beta": np.nan,
            }
        return {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": table,
        }

    def deserialize(self, serial: dict[str, Any]) -> "BayesianContingencyTable":
        """Populate our BayesianContingencyTable from a serialized version.

        Used when we want to populate our BayesianContingencyTable with results from a prior campaign.

        Parameters
        ----------
        serial : dict
            A serialized version of our ContingencyTable

        Returns
        -------
        Itself, to be chained with other methods
        """
        self.experiment_name = serial["experiment_name"]
        self.metric_name = serial["metric_name"]
        self.spend = serial["spend"]
        self.msrp = serial["msrp"]
        self.cells = serial
        self.names = list(serial["table"].keys())
        self.successes = [v["successes"] for v in serial["table"].values()]
        self.trials = [v["trials"] for v in serial["table"].values()]
        self.alphas = [v["alpha"] for v in serial["table"].values()]
        self.betas = [v["beta"] for v in serial["table"].values()]
        return self

    def analyze(
        self,
        lift: str = "relative",
        cred_int_method: Literal["credible", "hdi"] = "credible",
        confidence_level: float = 0.95,
        is_sample: bool = False,
        n_samples: int = 100_000,
        low_threshold: float = -0.1,
        high_threshold: float = 0.1,
    ) -> str:
        """Analyze the experiment and return a formatted summary table.

        Computes Bayesian metrics for the two variants — posterior means, credible
        interval, probability B exceeds A, expected loss, and ROPE probabilities —
        then formats them into a grid table string. Results are also stored on
        ``self.incremental_results`` for programmatic access.

        Parameters
        ----------
        lift : {"relative", "absolute", "incremental", "roas", "revenue"}, optional
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
            Lower bound of the Region of Practical Equivalence (ROPE),
            by default -0.1.
        high_threshold : float, optional
            Upper bound of the Region of Practical Equivalence (ROPE),
            by default 0.1.

        Returns
        -------
        str
            A grid-formatted table summarising the lift, credible interval,
            probability B is best, expected loss, and ROPE probability, with
            footnotes explaining annotated values.

        Raises
        ------
        ValueError
            If ``lift="roas"`` and ``spend`` was not set on the table.
        ValueError
            If ``lift="revenue"`` and ``msrp`` was not set on the table.
        ValueError
            If ``lift`` is not one of the supported types.
        """
        if len(self.names) != 2:
            raise ValueError(f"analyze requires exactly 2 variants, got {len(self.names)}")
        lift = lift.casefold()
        if lift in ["relative", "absolute"]:
            results = calculate_metrics(
                self.successes, self.trials, self.alphas, self.betas, n_samples, lift, low_threshold, high_threshold
            )
            lb, ub = credible_interval(
                self.successes,
                self.trials,
                self.alphas,
                self.betas,
                confidence_level,
                cast(Literal["relative", "absolute"], lift),
                is_sample,
                n_samples,
                cred_int_method,
            )
        elif lift in ["incremental", "roas", "revenue"]:
            results = calculate_metrics(
                self.successes,
                self.trials,
                self.alphas,
                self.betas,
                n_samples,
                lift,
                low_threshold,
                high_threshold,
                spend=self.spend,
                msrp=self.msrp,
            )
            lb, ub = credible_interval(
                self.successes,
                self.trials,
                self.alphas,
                self.betas,
                confidence_level,
                "absolute",
                is_sample,
                n_samples,
                cred_int_method,
            )
        else:
            raise ValueError(f"No support for lift type {lift}")
        pa = posterior_mean(self.successes[0], self.trials[0], self.alphas[0], self.betas[0])
        pb = posterior_mean(self.successes[1], self.trials[1], self.alphas[1], self.betas[1])
        success_rate: list[int | float]
        if lift in ["incremental", "roas", "revenue"]:
            if self.trials[0] > self.trials[1]:
                pb = math.ceil(pb * self.trials[0])
                pa = math.ceil(pa * self.trials[0])
                lb = math.ceil(lb * self.trials[0])
                ub = math.ceil(ub * self.trials[0])
            else:
                pa = math.ceil(pa * self.trials[1])
                pb = math.ceil(pb * self.trials[1])
                lb = math.ceil(lb * self.trials[1])
                ub = math.ceil(ub * self.trials[1])
            test_lift = pb - pa
            if lift == "roas":
                if self.spend is None:
                    raise ValueError("spend must be set for ROAS calculations")
                test_lift = self.spend / test_lift
                pa = self.spend / pa if pa > 0 else np.inf
                pb = self.spend / pb if pb > 0 else np.inf
                lb = self.spend / lb if lb > 0 else np.inf
                ub = self.spend / ub if ub > 0 else np.inf
            if lift == "revenue":
                if self.msrp is None:
                    raise ValueError("msrp must be set for revenue calculations")
                test_lift *= self.msrp
                pa *= self.msrp
                pb *= self.msrp
                lb *= self.msrp
                ub *= self.msrp
        elif lift == "relative":
            test_lift = (pb - pa) / pa
        elif lift == "absolute":
            test_lift = pb - pa
        else:
            raise ValueError(f"lift type {lift} not supported")
        success_rate = [pa, pb]
        if lift in ["relative", "absolute"]:
            self.incremental_results = {
                "lift_type": lift,
                "lift": test_lift,
                f"{self.names[0]}": success_rate[0],
                f"{self.names[1]}": success_rate[1],
                "prob_b_greater_a": results["Proportion of samples where B exceeds A"],
                "ci_lower": lb,
                "ci_upper": ub,
                "expected_loss": results["Expected loss"],
                "prob_rope": results["Probability of ROPE"],
                "prob_lift_exceeds_threshold": results[f"Probability {lift} exceeds {high_threshold}"],
                "prob_lift_below_threshold": results[f"Probability {lift} is below {low_threshold}"],
            }
        else:
            self.incremental_results = {
                "lift_type": lift,
                "lift": test_lift,
                f"{self.names[0]}": success_rate[0],
                f"{self.names[1]}": success_rate[1],
                "prob_b_greater_a": results["Proportion of samples where B exceeds A"],
                "ci_lower": lb,
                "ci_upper": ub,
                "expected_loss": results["Expected loss"],
                "prob_rope": results["Probability of ROPE"],
                "prob_lift_exceeds_threshold": results[f"Probability {lift} exceeds {high_threshold}"],
                "prob_lift_below_threshold": results[f"Probability {lift} is below {low_threshold}"],
            }
        prob_b_exceeds_a = results["Proportion of samples where B exceeds A"]
        str_pvalue = (
            f"{convert_to_tabulate_str(prob_b_exceeds_a, 'relative')}*"
            if prob_b_exceeds_a >= confidence_level
            else f"{convert_to_tabulate_str(prob_b_exceeds_a, 'relative')}"
        )
        table_headers = (
            ["Metric", "Metric Name"]
            + self.names
            + [
                "Lift",
                "Cred. Int. Lower **",
                "Cred. Int. Upper **",
                f"Prob {self.names[1]} Is Best",
                f"Expected Loss of {self.names[1]}",
                "Probability Lift is in ROPE ***",
            ]
        )
        table_list = [
            [lift]
            + [self.metric_name]
            + convert_to_tabulate_str(success_rate, lift)
            + convert_to_tabulate_str([test_lift, lb, ub], lift)
            + [str_pvalue]
            + [convert_to_tabulate_str(results["Expected loss"], "relative")]
            + [convert_to_tabulate_str(results["Probability of ROPE"], "relative")]
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f", intfmt=",")
        return_string += (
            f"\n* next to the prob means it exceeds our confidence level at {round(confidence_level * 100)}% level"
        )
        return_string += f"\n** {round(confidence_level * 100)}% Confidence Interval"
        return_string += "\n*** Region of Practical Equivalence"
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
        return_string += f"\n** {round(confidence_level * 100)}% Credible Interval"
        return return_string

    def plot(
        self,
        is_individual: bool = True,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Plot the posterior means and credible intervals as a forest plot.

        Parameters
        ----------
        is_individual : bool, default=True
            Whether we are looking at the individual performance of each cell or
            the comparative performance between variants.
        reverse_plot : bool, default=True
            Whether we are reversing the order of our plot or not.
        color : str, list, dict, or None, default=None
            If None, uses Plotly's default color scheme.
            If a string, one of the colorblind-friendly palette names: ``"ibm"``,
            ``"wong"``, ``"ito"``, ``"tol"``, ``"tol_bright"``, ``"tol_vibrant"``,
            ``"tol_muted"``, ``"tol_light"``.
            If a list, each item corresponds to a color for the relevant group.
            If a dict, keys are group names and values are colors.

        Notes
        -----
        This function is intended to be run after either .analyze() or
        .analyze_individually().
        """
        render_forest_plot(
            self.names,
            self.individual_results,
            self.incremental_results,
            is_individual=is_individual,
            reverse_plot=reverse_plot,
            color=color,
        )

    def plot_pdf(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 100_000,
        color: str | dict[str, Any] | list[Any] | None = None,
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
        # 1. Define X-axis range (0 to 1, but zoomed to relevant area)
        x_min = max(
            0, min(beta.ppf(0.001, self.alphas[0], self.betas[0]), beta.ppf(0.001, self.alphas[1], self.betas[1])) * 0.8
        )
        x_max = min(
            1, max(beta.ppf(0.999, self.alphas[0], self.betas[0]), beta.ppf(0.999, self.alphas[1], self.betas[1])) * 1.2
        )
        x = np.linspace(x_min, x_max, 500)

        # 2. PDF Curves
        pdf_a = beta.pdf(x, self.alphas[0], self.betas[0])
        pdf_b = beta.pdf(x, self.alphas[1], self.betas[1])

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

        return fig

    def __str__(self) -> str:
        """Return a grid-formatted string representation of the contingency table.

        Returns
        -------
        str
            A tabulated grid showing each cell's name, successes, trials, alpha,
            and beta, plus a totals row.
        """
        result: str = tabulate(
            self.to_list(include_total=True),
            headers=["cell_name", "successes", "trials", "alpha", "beta"],
            tablefmt="grid",
        )
        return result
