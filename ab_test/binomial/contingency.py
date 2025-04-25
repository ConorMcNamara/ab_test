"""Our wrapper for analyzing experiment results"""

from typing import Optional, Union
import math

import numpy as np
import pandas as pd
import polars as pl
from tabulate import tabulate

from ab_test.binomial.confidence_intervals import confidence_interval, individual_confidence_interval
from ab_test.binomial.stats_tests import ab_test, score_test, likelihood_ratio_test, z_test, cressie_read_test
from ab_test.binomial.utils import observed_lift


class ContingencyTable:
    """A class for analyzing experiment results"""

    def __init__(
        self, name: str, metric_name: str, spend: Optional[float] = None, msrp: Optional[float] = None
    ) -> None:
        """ContingencyTable is our class for creating and analyzing experiment results

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
        self.experiment_name = name
        self.names = []
        self.metric_name = metric_name
        self.spend = spend
        self.msrp = msrp
        self.cells = {
            "experiment_name": self.experiment_name,
            "metric_name": self.metric_name,
            "spend": self.spend,
            "msrp": self.msrp,
            "table": {},
        }
        self.successes = []
        self.trials = []
        self.results = None

    def add(self, cell_name: str, successes: int, trials: int) -> "ContingencyTable":
        """A method to add cells to our contingency table

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

    def to_df(self, method: str = "pandas", include_total: bool = False) -> Union[pd.DataFrame, pl.DataFrame]:
        """Returns our ContingencyTable as a DataFrame

        Parameters
        ----------
        method : {"pandas", "polars"}
            Whether we want our DataFrame as a pandas or polars DataFrame
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a DataFrame
        """
        method = method.casefold()
        if method == "pandas":
            return_df = pd.DataFrame(self.to_list(include_total), columns=["cell_name", "successes", "trials"])
        elif method == "polars":
            return_df = pl.DataFrame(
                self.to_list(include_total), schema=["cell_name", "successes", "trials"], orient="row"
            )
        elif method == "pyspark":
            raise NotImplementedError("Have not implemented Pyspark yet")
        elif method == "data.table":
            raise NotImplementedError("Have not implemented data.table yet")
        elif method == "modin":
            raise NotImplementedError("Have not implemented modin yet")
        else:
            raise ValueError(f"Methdod {method} not supported for creating DataFrames")
        return return_df

    def to_list(self, include_total: bool = False) -> list:
        """Returns our ContingencyTable as a list

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a list
        """
        return_list = []
        for key, value in self.cells.items():
            if key == "table":
                for name in self.names:
                    loop_list = [name] + list(value[f"{name}"].values())
                    return_list.append(loop_list)
                if include_total:
                    total_list = ["Total", np.sum(self.successes), np.sum(self.trials)]
                    return_list.append(total_list)
            else:
                pass
        return return_list

    def to_numpy(self, include_total: bool = False) -> np.ndarray:
        """Returns our ContingencyTable as a numpy array

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a numpy array
        """
        return np.array(self.to_list(include_total))

    def serialize(self, include_total: bool = False):
        """Returns our ContingencyTable as a JSON, with all information

        Parameters
        ----------
        include_total : bool, default=False
            Whether we want to include another section with the total amount

        Returns
        -------
        Our ContingencyTable as a JSON
        """
        if include_total:
            total_dict = {"successes": self.successes, "trials": self.trials}
            self.cells["table"]["Total"] = total_dict
        return self.cells

    def deserialize(self, serial: dict) -> "ContingencyTable":
        """Takes in a serialized version of our ContingencyTable. Used when we want to populate our
        ContingencyTable with results from a prior campaign.

        Parameters
        ----------
        serial : dict
            A serialized version of our ContingencyTable

        Returns
        -------
        Itself, to be chained with other methods
        """
        self.experiment_name = serial["experiment_name"]
        self.spend = serial["spend"]
        self.msrp = serial["msrp"]
        self.cells = serial
        self.metric_name = serial["metric_name"]
        return self

    def analyze(
        self,
        lift: str = "relative",
        test_method: str = "score",
        conf_int_method: str = "binary_search",
        alpha: float = 0.05,
        null_lift: float = 0.0,
    ) -> str:
        """Analyzes the effect of our experiments through the ContingencyTable

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue'}
            The kind of lift we are measuring for our campaign
        test_method : {'score', 'likelihood', 'z', 'fisher', 'barnard', 'boschloo', 'modified_likelihood', 'freeman-tukey', 'neyman', 'cressie-read'}
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
        if lift in ["incremental", "roas", "revenue"]:
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
                test_lift = self.spend / test_lift
                pa = self.spend / pa if pa > 0 else np.inf
                pb = self.spend / pb if pb > 0 else np.inf
                lb = self.spend / lb if lb > 0 else np.inf
                ub = self.spend / ub if ub > 0 else np.inf
            if lift == "revenue":
                test_lift *= self.msrp
                pa *= self.msrp
                pb *= self.msrp
                lb *= self.msrp
                ub *= self.msrp
            success_rate = [pa, pb]
        else:
            success_rate = [si / ti for ti, si in zip(self.trials, self.successes)]
        self.results = {
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
            + self._convert_to_tabulate_str(success_rate, lift)
            + self._convert_to_tabulate_str([test_lift, lb, ub], lift)
            + [str_pvalue]
        ]
        return_string = tabulate(table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f", intfmt=",")
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
        """Analyzes the individual cells

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
            name_list = [name_i, s_i, n_i] + self._convert_to_tabulate_str([success_rate, lb, ub], "absolute")
            table_list.append(name_list)
        total_success, total_trials = np.sum(self.successes), np.sum(self.trials)
        total_success_rate = total_success / total_trials
        lb_total, ub_total = individual_confidence_interval(total_success, total_trials, alpha, conf_int_method)
        total_list = ["Total", total_success, total_trials] + self._convert_to_tabulate_str(
            [total_success_rate, lb_total, ub_total], "absolute"
        )
        table_list.append(total_list)
        table_headers = ["Cell Name", "Successes", "Trials", "Success Rate", "Conf. Int. Lower**", "Conf. Int. Upper**"]
        return_string = tabulate(table_list, headers=table_headers, tablefmt="grid", intfmt=",")
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    @staticmethod
    def _convert_to_tabulate_str(value: Union[float, list], lift: str) -> Union[str, list, float]:
        """Converts our lift values to either percentages or dollar signs

        Parameters
        ----------
        value : float or list
            The value we are changing
        lift : str
            Depending on the lift type, whether we are adding percentages or dollar signs

        Returns
        -------
        Our new str_value, as either a percentage or dollar sign
        """
        if isinstance(value, float):
            if lift in ["revenue", "roas"]:
                str_value = f"${round(value, 2):,}"
            elif lift in ["absolute", "relative"]:
                str_value = f"{round(value * 100.0, 2)}%"
            elif lift in ["incremental"]:
                str_value = value
            else:
                raise ValueError(f"No support for {lift}")
        elif isinstance(value, list):
            if lift in ["revenue", "roas"]:
                str_value = [f"${round(val, 2):,}" for val in value]
            elif lift in ["absolute", "relative"]:
                str_value = [f"{round(val * 100.0, 2)}%" for val in value]
            elif lift == "incremental":
                str_value = value
            else:
                raise ValueError(f"No support for {lift}")
        else:
            raise TypeError(f"No support for converting {value} to string")
        return str_value

    def __str__(self):
        return tabulate(
            self.to_list(include_total=True),
            headers=["cell_name", "successes", "trials", "90% CI Lower", "90% CI Upper"],
            tablefmt="grid",
            intfmt=",",
        )
