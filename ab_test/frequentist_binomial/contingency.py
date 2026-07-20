"""Our wrapper for analyzing experiment results."""

import math
from typing import Any, ClassVar

import numpy as np
from tabulate import tabulate

from ab_test._contingency import BaseContingencyTable
from ab_test._display import convert_to_tabulate_str
from ab_test.frequentist_binomial.confidence_intervals import confidence_interval, individual_confidence_interval
from ab_test.frequentist_binomial.stats_tests import (
    ab_test,
    score_test,
    likelihood_ratio_test,
    z_test,
    cressie_read_test,
)
from ab_test.frequentist_binomial.utils import observed_lift


def _scale_bound(bound: float, factor: float) -> float:
    """Scale a lift bound to a count, leaving unbounded (infinite) bounds intact.

    ``confidence_interval`` returns ``±math.inf`` when a bound does not exist.
    ``math.ceil`` cannot convert an infinite float to an int, so such bounds are
    passed through unchanged.

    Parameters
    ----------
    bound : float
        The confidence-interval bound to scale.
    factor : float
        The multiplier used to convert the bound to a count.

    Returns
    -------
    float
        ``bound`` unchanged if it is infinite, otherwise ``ceil(bound * factor)``.
    """
    if math.isinf(bound):
        return bound
    return math.ceil(bound * factor)


class ContingencyTable(BaseContingencyTable):
    """A class for analyzing experiment results."""

    _columns: ClassVar[list[str]] = ["cell_name", "successes", "trials"]
    _pyspark_types: ClassVar[dict[str, str]] = {
        "cell_name": "StringType",
        "successes": "IntegerType",
        "trials": "IntegerType",
    }

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        return ["Total", np.sum(self.successes), np.sum(self.trials)]

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        return {"successes": int(np.sum(self.successes)), "trials": int(np.sum(self.trials))}

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
                lb = _scale_bound(lb, self.trials[0])
                ub = _scale_bound(ub, self.trials[0])
            else:
                pa = math.ceil(self.successes[0] * (self.trials[1] / self.trials[0]))
                pb = math.ceil(self.successes[1])
                lb = _scale_bound(lb, self.trials[1])
                ub = _scale_bound(ub, self.trials[1])
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
