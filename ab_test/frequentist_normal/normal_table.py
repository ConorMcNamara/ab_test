"""Our wrapper for analyzing experiment results."""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from tabulate import tabulate

from ab_test._continuous_table import BaseContinuousTable
from ab_test._display import convert_to_tabulate_str
from ab_test.frequentist_normal.stats_tests import welch_test
from ab_test.frequentist_normal.confidence_intervals import confidence_interval
from ab_test.frequentist_normal.utils import observed_lift


class NormalTable(BaseContinuousTable):
    _columns: ClassVar[list[str]] = ["cell_name", "means", "variances", "trials"]
    _pyspark_types: ClassVar[dict[str, str]] = {
        "cell_name": "StringType",
        "means": "DoubleType",
        "variances": "DoubleType",
        "trials": "IntegerType",
    }

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        return [
            "Total",
            np.sum(np.array(self.means) * np.array(self.trials)) / np.sum(self.trials),
            np.nan,
            np.sum(self.trials),
        ]

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        return {
            "means": float(np.sum(np.array(self.means) * np.array(self.trials))) / np.sum(self.trials),
            "variances": np.nan,
            "trials": int(np.sum(self.trials)),
        }

    def add(self, cell_name: str, means: float, variances: float, trials: int) -> "NormalTable":
        """Add cells to our contingency table.

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        means : float
            The mean / average in our cell_name
        variances : float
            The variances in our cell_name
        trials : int
            The number of trials in our cell_name

        Returns
        -------
        NormalTable, to be chained with other methods
        """
        cell_dict = {"means": means, "variances": variances, "trials": trials}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.means.append(means)
        self.variances.append(variances)
        self.trials.append(trials)
        return self

    def add_data(self, cell_name: str, data: np.generic | np.ndarray | list | tuple) -> "NormalTable":
        """Add a cell from raw data, computing summary statistics automatically.

        Parameters
        ----------
        cell_name : str
            The name of our cell.
        data : numpy array, list, or tuple
            The dataset of outcomes for our cell_name

        Returns
        -------
        NormalTable, to be chained with other methods
        """
        means = float(np.mean(data))
        variances = float(np.var(data, ddof=1))
        trials = int(len(data))
        cell_dict = {"means": means, "variances": variances, "trials": trials}
        self.cells["table"][cell_name] = cell_dict
        self.names.append(cell_name)
        self.means.append(means)
        self.variances.append(variances)
        self.trials.append(trials)
        return self

    def analyze(
        self,
        lift: str = "relative",
        test_method: str = "welch",
        alpha: float = 0.05,
        null_lift: float = 0.0,
    ) -> str:
        """Analyzes the effect of our experiments through the NormalTable.

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue', 'cpa'}
            The kind of lift we are measuring for our campaign
        test_method : str
            The method we plan to use to assess whether our result is
            statistically significant. Currently only ``'welch'`` is supported.
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
        mean_a, mean_b = self.means[0], self.means[1]
        n_a, n_b = self.trials[0], self.trials[1]
        p_value = welch_test(self.means, self.variances, self.trials, null_lift, lift)
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            ci_lift = "absolute"
        else:
            ci_lift = lift
        lb, ub = confidence_interval(self.means, self.variances, self.trials, test_method, alpha, ci_lift)
        test_lift = observed_lift(self.means, self.trials, lift=ci_lift)
        cell_values: list[float]
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            scale = max(n_a, n_b)
            test_lift *= scale
            lb *= scale
            ub *= scale
            total_a = mean_a * scale
            total_b = mean_b * scale
            if lift == "roas":
                if self.spend is None:
                    raise ValueError("spend must be set for ROAS calculations")
                test_lift /= self.spend
                total_a /= self.spend
                total_b /= self.spend
                lb /= self.spend
                ub /= self.spend
            elif lift == "cpa":
                if self.spend is None:
                    raise ValueError("spend must be set for CPA calculations")
                test_lift = self.spend / test_lift if test_lift != 0 else np.inf
                total_a = self.spend / total_a if total_a > 0 else np.inf
                total_b = self.spend / total_b if total_b > 0 else np.inf
                lb, ub = (
                    self.spend / ub if ub > 0 else np.inf,
                    self.spend / lb if lb > 0 else np.inf,
                )
            if lift == "revenue":
                if self.msrp is None:
                    raise ValueError("msrp must be set for revenue calculations")
                test_lift *= self.msrp
                total_a *= self.msrp
                total_b *= self.msrp
                lb *= self.msrp
                ub *= self.msrp
            cell_values = [total_a, total_b]
        else:
            cell_values = [mean_a, mean_b]
        self.incremental_results = {
            "lift_type": lift,
            "lift": test_lift,
            f"{self.names[0]}": cell_values[0],
            f"{self.names[1]}": cell_values[1],
            "p_value": p_value,
            "ci_lower": lb,
            "ci_upper": ub,
        }
        table_headers = (
            ["Metric", "Metric Name"]
            + self.names
            + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value"]
        )
        str_pvalue = f"{p_value}" if p_value >= alpha else f"{p_value}*"
        if lift == "relative":
            table_list = [
                [lift, self.metric_name]
                + cell_values
                + convert_to_tabulate_str([test_lift, lb, ub], "relative")
                + [str_pvalue]
            ]
        elif lift in ["revenue", "roas", "cpa"]:
            table_list = [
                [lift, self.metric_name]
                + convert_to_tabulate_str(cell_values, lift)
                + convert_to_tabulate_str([test_lift, lb, ub], lift)
                + [str_pvalue]
            ]
        else:
            table_list = [
                [lift, self.metric_name]
                + cell_values
                + [test_lift, lb, ub]
                + [str_pvalue]
            ]
        return_string: str = tabulate(
            table_list, headers=table_headers, tablefmt="grid", floatfmt=".2f", intfmt=","
        )
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string
