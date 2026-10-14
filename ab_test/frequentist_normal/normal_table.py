"""Our wrapper for analyzing experiment results."""

from __future__ import annotations

import functools
import itertools
from typing import Any, ClassVar

import numpy as np
from tabulate import tabulate

from ab_test._continuous_table import BaseContinuousTable
from ab_test._display import convert_to_tabulate_str, format_percent, tabulate_summary
from ab_test._lift import to_absolute
from ab_test.corrections import adjust_pvalues
from ab_test.frequentist_normal.stats_tests import score_omnibus_test, score_test, welch_anova, welch_test
from ab_test.frequentist_normal.confidence_intervals import confidence_interval
from ab_test.frequentist_normal.utils import observed_lift


class NormalTable(BaseContinuousTable):
    """A class for analyzing experiment results on continuous, normally distributed metrics.

    Examples
    --------
    >>> table = NormalTable("Checkout redesign", "order value")
    >>> table = table.add("Control", 52.0, 144.0, 1000).add("Treatment", 54.5, 150.0, 1000)
    >>> print(table.analyze(lift="absolute"))  # doctest: +SKIP
    >>> print(table.analyze(lift="absolute", test_method="score", equal_var=False))  # doctest: +SKIP
    >>> table = table.add("Treatment 2", 55.0, 160.0, 1000)
    >>> print(table.analyze(lift="absolute", comparisons="all"))  # doctest: +SKIP
    """

    _columns: ClassVar[list[str]] = ["cell_name", "means", "variances", "trials"]
    _pyspark_types: ClassVar[dict[str, str]] = {
        "cell_name": "StringType",
        "means": "DoubleType",
        "variances": "DoubleType",
        "trials": "IntegerType",
    }

    def _total_variance(self) -> float:
        """Compute the combined variance using the law of total variance."""
        means = np.array(self.means)
        variances = np.array(self.variances)
        trials = np.array(self.trials)
        total_n = np.sum(trials)
        total_mean = np.sum(means * trials) / total_n
        within = np.sum(trials * variances) / total_n
        between = np.sum(trials * (means - total_mean) ** 2) / total_n
        return float(within + between)

    def _total_row(self) -> list[Any]:
        """Return the ``"Total"`` row appended to :meth:`to_list`."""
        total_n = np.sum(self.trials)
        total_mean = np.sum(np.array(self.means) * np.array(self.trials)) / total_n
        return [
            "Total",
            total_mean,
            self._total_variance(),
            total_n,
        ]

    def _total_cell(self) -> dict[str, Any]:
        """Return the ``"Total"`` cell dict appended to :meth:`serialize`."""
        total_n = int(np.sum(self.trials))
        total_mean = float(np.sum(np.array(self.means) * np.array(self.trials)) / total_n)
        return {
            "means": total_mean,
            "variances": self._total_variance(),
            "trials": total_n,
        }

    def add(self, cell_name: str, means: float, variances: float, trials: int) -> NormalTable:
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

    def add_data(self, cell_name: str, data: np.ndarray[Any, Any] | list[float] | tuple[float, ...]) -> NormalTable:
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
        values = np.asarray(data, dtype=float)
        means = float(np.mean(values))
        variances = float(np.var(values, ddof=1))
        trials = int(values.size)
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
        conf_int_method: str = "welch",
        alpha: float = 0.05,
        null_lift: float = 0.0,
        equal_var: bool = True,
        *,
        comparisons: str = "control",
        correction: str = "holm",
    ) -> str:
        """Analyzes the effect of our experiments through the NormalTable.

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue', 'cpa'}
            The kind of lift we are measuring for our campaign
        test_method : str
            The method we plan to use to assess whether our result is
            statistically significant. One of ``'welch'`` or ``'score'``.
        conf_int_method : str
            The method we plan to use to craft confidence intervals of our lift.
            One of ``'welch'``, ``'z'``, ``'binary_search'`` (which inverts the
            test chosen by ``test_method``), or ``'delta'``.
        alpha : float, default = 0.05
            The alpha level of our experiment, to be used to craft confidence intervals.
        null_lift : float
            Lift associated with null hypothesis, in the units of ``lift``. Defaults to 0.0.
        equal_var : bool, default=True
            Whether the score test assumes every group shares a common variance.
            Only used when ``test_method='score'``; Welch's test always allows
            unequal variances. See :func:`~ab_test.frequentist_normal.stats_tests.score_test`.
        comparisons : {'control', 'all'}, default='control'
            With three or more variants, which pairs to compare: each variant
            against the first cell added (the control), or every pair. Ignored
            with two variants.
        correction : str, default='holm'
            With three or more variants, how the pairwise p-values are adjusted
            for multiple comparisons; any method accepted by
            :func:`~ab_test.corrections.adjust_pvalues`. Ignored with two
            variants.

        Returns
        -------
        The results (lift as well as confidence intervals) of our experiment in string format, to be printed

        Notes
        -----
        With three or more variants, ``analyze()`` reports an omnibus test that
        every variant shares one mean, then the chosen pairwise comparisons.
        Each comparison uses ``test_method`` and ``conf_int_method`` exactly as
        a two-variant analysis would. The p-values are adjusted with
        ``correction``, and the intervals are Bonferroni intervals (each at
        level ``1 - alpha / m`` for ``m`` comparisons), so all of them hold
        simultaneously with probability ``1 - alpha``. The omnibus test is
        Welch's ANOVA for ``'welch'`` and the k-group score test for
        ``'score'`` (see :func:`~ab_test.frequentist_normal.stats_tests.welch_anova`
        and :func:`~ab_test.frequentist_normal.stats_tests.score_omnibus_test`);
        it tests equal means, whatever ``null_lift`` is. ``incremental_results``
        then holds ``"omnibus"`` and a ``"comparisons"`` dict keyed by labels
        such as ``"B vs A"``, each with the adjusted ``"p_value"`` and the
        ``"raw_p_value"``.
        """
        k = len(self.names)
        if k < 2:
            raise ValueError(f"analyze requires at least 2 variants, got {k}")
        comparisons = comparisons.casefold()
        if comparisons not in ("control", "all"):
            raise ValueError(f"comparisons must be 'control' or 'all', got {comparisons!r}")
        adjust_pvalues([0.5], method=correction)  # Validate the correction before any work.
        if test_method == "welch":
            test = welch_test
        elif test_method == "score":
            test = functools.partial(score_test, equal_var=equal_var)
        else:
            raise NotImplementedError(f"No support for {test_method} test method")
        settings: dict[str, Any] = {
            "lift": lift.casefold(),
            "test_method": test_method,
            "test": test,
            "conf_int_method": conf_int_method,
            "null_lift": null_lift,
            "equal_var": equal_var,
        }
        if k == 2:
            return self._analyze_pair(alpha, settings)
        return self._analyze_many(alpha, comparisons, correction, settings)

    def _compare(self, i: int, j: int, alpha: float, settings: dict[str, Any]) -> dict[str, Any]:
        """Compare cell ``j`` against cell ``i``: lift, the two cell values, p-value and interval."""
        lift = settings["lift"]
        test = settings["test"]
        null_lift = settings["null_lift"]
        means = [self.means[i], self.means[j]]
        variances = [self.variances[i], self.variances[j]]
        trials = [self.trials[i], self.trials[j]]
        mean_a, mean_b = means
        if lift == "relative" and mean_a == 0:
            raise ValueError(
                f"Relative lift of {self.names[j]} vs {self.names[i]} is undefined: {self.names[i]} has a mean "
                'of 0; use lift="absolute"'
            )
        # Every comparison is expressed over the table's largest arm, so with three or
        # more arms identical differences in means give identical scaled lifts. With two
        # arms that is the larger of the pair, as before.
        scale = max(self.trials)
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            # The tests work on the per-unit difference in means, so test the null on
            # the same absolute scale that the interval is built on.
            ci_lift = "absolute"
            if lift == "cpa" and null_lift == 0:
                test_null = 0.0
            else:
                test_null = to_absolute(null_lift, lift, scale, self.spend, self.msrp)
        else:
            ci_lift = lift
            test_null = null_lift
        p_value = test(means, variances, trials, test_null, ci_lift)
        lb, ub = confidence_interval(
            means, variances, trials, test=test, method=settings["conf_int_method"], alpha=alpha, lift=ci_lift
        )
        test_lift = observed_lift(means, trials, lift=ci_lift)
        cell_values: list[float]
        if lift in ["incremental", "roas", "revenue", "cpa"]:
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
        return {"lift": test_lift, "values": cell_values, "p_value": p_value, "ci_lower": lb, "ci_upper": ub}

    def _analyze_pair(self, alpha: float, settings: dict[str, Any]) -> str:
        """Two variants: one comparison, reported as before multi-arm support."""
        lift = settings["lift"]
        result = self._compare(0, 1, alpha, settings)
        test_lift, cell_values, p_value = result["lift"], result["values"], result["p_value"]
        lb, ub = result["ci_lower"], result["ci_upper"]
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
            ["Metric", "Metric Name"] + self.names + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value"]
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
            table_list = [[lift, self.metric_name] + cell_values + [test_lift, lb, ub] + [str_pvalue]]
        # ROAS is formatted as a bare number; reparsing a string like "2" as an int
        # would apply intfmt to it.
        return_string: str = tabulate(
            table_list,
            headers=table_headers,
            tablefmt="grid",
            floatfmt=".2f",
            intfmt=",",
            disable_numparse=lift == "roas",
        )
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {round(alpha * 100)}% level"
        )
        return_string += f"\n** {round((1 - alpha) * 100)}% Confidence Interval"
        return return_string

    def _omnibus_test(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Test that every cell shares one mean, with the omnibus form of ``test_method``."""
        if settings["test_method"] == "welch":
            statistic, df_num, df_denom, pval = welch_anova(self.means, self.variances, self.trials)
            return {
                "test": "Welch's ANOVA",
                "statistic": statistic,
                "df": int(df_num),
                "df_denom": df_denom,
                "p_value": pval,
            }
        statistic, df, pval = score_omnibus_test(
            self.means, self.variances, self.trials, equal_var=settings["equal_var"]
        )
        return {"test": "Score", "statistic": statistic, "df": df, "p_value": pval}

    def _analyze_many(self, alpha: float, comparisons: str, correction: str, settings: dict[str, Any]) -> str:
        """Three or more variants: an omnibus test plus corrected pairwise comparisons."""
        lift = settings["lift"]
        k = len(self.names)
        pairs = [(0, j) for j in range(1, k)] if comparisons == "control" else list(itertools.combinations(range(k), 2))
        # Bonferroni intervals: simultaneous coverage of 1 - alpha across all comparisons.
        ci_alpha = alpha / len(pairs)
        results = [self._compare(i, j, ci_alpha, settings) for i, j in pairs]
        adjusted = adjust_pvalues([r["p_value"] for r in results], method=correction)
        omnibus = self._omnibus_test(settings)

        compared: dict[str, dict[str, Any]] = {}
        for (i, j), result, adj_p in zip(pairs, results, adjusted):
            compared[f"{self.names[j]} vs {self.names[i]}"] = {
                "lift": result["lift"],
                f"{self.names[i]}": result["values"][0],
                f"{self.names[j]}": result["values"][1],
                "p_value": adj_p,
                "raw_p_value": result["p_value"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
            }
        self.incremental_results = {
            "lift_type": lift,
            "comparison_type": comparisons,
            "correction": correction,
            "omnibus": omnibus,
            "comparisons": compared,
        }

        omnibus_p = omnibus["p_value"]
        # NaN fails every comparison, so test for significance explicitly rather than with >= alpha.
        str_omnibus = f"{omnibus_p:.4f}*" if omnibus_p < alpha else f"{omnibus_p:.4f}"
        return_string = tabulate_summary(
            ["Metric", "Metric Name"] + self.names + ["Omnibus p-value ***"],
            [lift, self.metric_name] + [float(m) for m in self.means] + [str_omnibus],
        )

        def _format_lifts(values: list[float]) -> list[Any]:
            if lift in ["relative", "revenue", "roas", "cpa"]:
                return convert_to_tabulate_str(values, lift)
            return [f"{v:,.2f}" for v in values]

        rows = []
        for label, comparison in compared.items():
            star = "*" if comparison["p_value"] < alpha else ""
            rows.append(
                [label]
                + _format_lifts([comparison["lift"], comparison["ci_lower"], comparison["ci_upper"]])
                + [f"{comparison['raw_p_value']:.4f}", f"{comparison['p_value']:.4f}{star}"]
            )
        headers = [
            "Comparison",
            "Lift",
            "Conf. Int. Lower **",
            "Conf. Int. Upper **",
            "p-value",
            f"Adj. p ({correction})",
        ]
        return_string += "\n" + tabulate(rows, headers=headers, tablefmt="grid", disable_numparse=True)
        return_string += (
            f"\n* next to a p-value means it's statistically significant at the {format_percent(alpha)}% level"
            f" ({correction}-adjusted for {len(pairs)} comparisons)"
        )
        return_string += (
            f"\n** {format_percent(1 - alpha)}% simultaneous Confidence Intervals"
            f" (Bonferroni: each at {round(100 * (1 - ci_alpha), 2):g}%)"
        )
        df = f"({omnibus['df']}, {omnibus['df_denom']:.1f})" if "df_denom" in omnibus else f"{omnibus['df']}"
        return_string += f"\n*** {omnibus['test']} test that all {k} variants share one mean, df={df}"
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            return_string += f"\nScaled lifts are per {max(self.trials):,} units (the largest arm) for every comparison"
        return return_string
