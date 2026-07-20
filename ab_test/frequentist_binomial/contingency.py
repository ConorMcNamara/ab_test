"""Our wrapper for analyzing experiment results."""

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate

from ab_test._display import convert_to_tabulate_str, render_forest_plot
from ab_test.frequentist_binomial.confidence_intervals import confidence_interval, individual_confidence_interval
from ab_test.frequentist_binomial.stats_tests import (
    ab_test,
    score_test,
    likelihood_ratio_test,
    z_test,
    cressie_read_test,
)
from ab_test.frequentist_binomial.utils import observed_lift


class ContingencyTable:
    """A class for analyzing experiment results."""

    def __init__(self, name: str, metric_name: str, spend: float | None = None, msrp: float | None = None) -> None:
        """ContingencyTable is our class for creating and analyzing experiment results.

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
        self.incremental_results: dict[str, Any] | None = None
        self.individual_results: dict[str, dict[str, float]] = {}

    def add(self, cell_name: str, successes: int, trials: int) -> "ContingencyTable":
        """Add cells to our contingency table.

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        successes : int
            The number of successes in our cell_name
        trials : int
            The number of trials in our cell_name

        Returns
        -------
        ContingencyTable, to be chained with other methods
        """
        cell_dict = {"successes": successes, "trials": trials}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.successes.append(successes)
        self.trials.append(trials)
        return self

    def to_df(
        self,
        method: str = "pandas",
        include_total: bool = False,
        spark_session: Any | None = None,
        ibis_backend: Any | None = None,
    ) -> pd.DataFrame | pl.DataFrame | Any:
        """Return our ContingencyTable as a DataFrame.

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
        Our ContingencyTable as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(
                self.to_list(include_total), columns=pd.Index(["cell_name", "successes", "trials"])
            )
        elif method == "polars":
            return_df = pl.DataFrame(
                self.to_list(include_total), schema=["cell_name", "successes", "trials"], orient="row"
            )
        elif method == "pyspark":
            from pyspark.sql import SparkSession
            from pyspark.sql.types import IntegerType, StringType, StructField, StructType

            if spark_session is None:
                spark_session = SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No active SparkSession found. Please provide a spark_session argument.")
            schema = StructType(
                [
                    StructField("cell_name", StringType(), True),
                    StructField("successes", IntegerType(), True),
                    StructField("trials", IntegerType(), True),
                ]
            )
            return_df = spark_session.createDataFrame(self.to_list(include_total), schema=schema)
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            import modin.pandas as mpd  # type: ignore[import-not-found]

            return_df = mpd.DataFrame(
                self.to_list(include_total),
                columns=mpd.Index(["cell_name", "successes", "trials"]),
            )
        elif method == "ibis":
            import ibis  # type: ignore[import-not-found]

            if ibis_backend is not None:
                ibis.set_backend(ibis_backend)
            pandas_df = pd.DataFrame(
                self.to_list(include_total),
                columns=pd.Index(["cell_name", "successes", "trials"]),
            )
            return_df = ibis.memtable(pandas_df)
        elif method == "narwhals":
            import narwhals as nw

            pandas_df = pd.DataFrame(
                self.to_list(include_total),
                columns=pd.Index(["cell_name", "successes", "trials"]),
            )
            return_df = nw.from_native(pandas_df)
        else:
            raise ValueError(f"Method {method} not supported for creating DataFrames")
        return return_df

    def to_list(self, include_total: bool = False) -> list[Any]:
        """Return our ContingencyTable as a list.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a list
        """
        return_list = []
        for name in self.names:
            return_list.append([name] + list(self.cells["table"][name].values()))
        if include_total:
            return_list.append(["Total", np.sum(self.successes), np.sum(self.trials)])
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray[Any, Any]:
        """Return our ContingencyTable as a numpy array.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a numpy array
        """
        return np.array(self.to_list(include_total))

    def serialize(self, include_total: bool = False) -> dict[str, Any]:
        """Return our ContingencyTable as a JSON, with all information.

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a JSON
        """
        table = dict(self.cells["table"])
        if include_total:
            table["Total"] = {"successes": int(np.sum(self.successes)), "trials": int(np.sum(self.trials))}
        return {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": table,
        }

    def deserialize(self, serial: dict[str, Any]) -> "ContingencyTable":
        """Populate our ContingencyTable from a serialized version.

        Used when we want to populate our ContingencyTable with results from a prior campaign.

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
        return self

    def analyze(
        self,
        lift: str = "relative",
        test_method: str = "score",
        conf_int_method: str = "binary_search",
        alpha: float = 0.05,
        null_lift: float = 0.0,
    ) -> str:
        """Analyzes the effect of our experiments through the ContingencyTable.

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue'}
            The kind of lift we are measuring for our campaign
        test_method : {'score', 'likelihood', 'z', 'fisher', 'barnard', 'boschloo',
                       'modified_likelihood', 'freeman-tukey', 'neyman', 'cressie-read'}
            The method we plan to use to assess whether our result is statistically significant
        conf_int_method : {'binary_search', "wilson", "jeffrey", "agresti-coull", "clopper-pearson", 'wald'}
            The method we plan to use to craft confidence intervals of our lift
        alpha : float, default = 0.05
            The alpha level of our experiment, to be used to craft confidence intervals.
        null_lift : float
            Lift associated with null hypothesis. Defaults to 0.0.

        Returns
        -------
        The results (lift as well as confidence intervals) of our experiment in string format, to be printed
        """
        if len(self.names) != 2:
            raise ValueError(f"analyze requires exactly 2 variants, got {len(self.names)}")
        lift = lift.casefold()
        p_value = ab_test(self.trials, self.successes, null_lift, lift, method=test_method)
        test_lift = observed_lift(self.trials, self.successes, lift)
        if test_method == "score":
            test = score_test
        elif test_method == "likelihood":
            test = likelihood_ratio_test
        elif test_method == "z":
            test = z_test
        else:
            test = cressie_read_test
        if lift in ["incremental", "roas", "revenue"]:
            ci_lift = "absolute"
        else:
            ci_lift = lift
        lb, ub = confidence_interval(
            self.trials, self.successes, test=test, alpha=alpha, lift=ci_lift, method=conf_int_method
        )
        success_rate: list[int | float]
        if lift in ["incremental", "roas", "revenue"]:
            pa: int | float
            pb: int | float
            if self.trials[0] > self.trials[1]:
                pb = math.ceil(self.successes[1] * (self.trials[0] / self.trials[1]))
                pa = math.ceil(self.successes[0])
                lb = math.ceil(lb * self.trials[0])
                ub = math.ceil(ub * self.trials[0])
            else:
                pa = math.ceil(self.successes[0] * (self.trials[1] / self.trials[0]))
                pb = math.ceil(self.successes[1])
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
            success_rate = [pa, pb]
        else:
            success_rate = [si / ti for ti, si in zip(self.trials, self.successes)]
        self.incremental_results = {
            "lift_type": lift,
            "lift": test_lift,
            f"{self.names[0]}": success_rate[0],
            f"{self.names[1]}": success_rate[1],
            "p_value": p_value,
            "ci_lower": lb,
            "ci_upper": ub,
        }
        table_headers = (
            ["Metric", "Metric Name"] + self.names + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value"]
        )
        str_pvalue = f"{p_value}" if p_value >= alpha else f"{p_value}*"
        table_list = [
            [lift, self.metric_name]
            + convert_to_tabulate_str(success_rate, lift)
            + convert_to_tabulate_str([test_lift, lb, ub], lift)
            + [str_pvalue]
        ]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f", intfmt=",")
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def analyze_individually(
        self,
        conf_int_method: str = "wilson",
        alpha: float = 0.05,
    ) -> str:
        """Analyzes the individual cells.

        Parameters
        ----------
        conf_int_method : {"wilson", "agresti-coull", "jeffrey", "clopper-pearson", "wald"}
            The method for calculating individual confidence intervals
        alpha : float
            The significance level. Defaults to 0.05, corresponding to a 95%
            confidence interval.

        Returns
        -------
        The results (success as well as confidence intervals) of our individual cells in string format, to be printed
        """
        table_list = []
        for name_i, s_i, n_i in zip(self.names, self.successes, self.trials):
            success_rate = s_i / n_i
            lb, ub = individual_confidence_interval(s_i, n_i, alpha, conf_int_method)
            name_list = [name_i, s_i, n_i] + convert_to_tabulate_str([success_rate, lb, ub], "absolute")
            self.individual_results[name_i] = {"lift": success_rate, "ci_lower": lb, "ci_upper": ub}
            table_list.append(name_list)
        total_success, total_trials = np.sum(self.successes), np.sum(self.trials)
        total_success_rate = total_success / total_trials
        lb_total, ub_total = individual_confidence_interval(total_success, total_trials, alpha, conf_int_method)
        total_list = ["Total", total_success, total_trials] + convert_to_tabulate_str(
            [total_success_rate, lb_total, ub_total], "absolute"
        )
        self.individual_results["Total"] = {"lift": total_success_rate, "ci_lower": lb_total, "ci_upper": ub_total}
        table_list.append(total_list)
        table_headers = ["Cell Name", "Successes", "Trials", "Success Rate", "Conf. Int. Lower**", "Conf. Int. Upper**"]
        return_string: str = tabulate(table_list, headers=table_headers, tablefmt="grid")
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def plot(
        self,
        is_individual: bool = True,
        reverse_plot: bool = True,
        color: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        """Plot the point estimates as well as confidence intervals.

        Parameters
        ----------
        is_individual : bool, default=True
            Whether we are looking at the individual performance of each cell or the comparative performance
        reverse_plot : bool, default=True
            Whether we are reversing the order of our plot or not.
        color : str or list or dict, default=None
            If None, then uses plotly's default color scheme.
            If a string, then one of the available colorblind options
            If a list, then each item in the list corresponds to a color for the relevant group
            If a dictionary, then each key in the dictionary corresponds to a group with the
            value pertaining to its color

        Returns
        -------
        A plot of our point estimates as well as confidence intervals

        Notes
        -----
        This function is intended to be run _after_ either .analyze() or .analyze_individually()
        """
        render_forest_plot(
            self.names,
            self.individual_results,
            self.incremental_results,
            is_individual=is_individual,
            reverse_plot=reverse_plot,
            color=color,
        )

    def __str__(self) -> str:
        """Return a tabulated string representation of the ContingencyTable."""
        result: str = tabulate(
            self.to_list(include_total=True),
            headers=["cell_name", "successes", "trials"],
            tablefmt="grid",
        )
        return result
