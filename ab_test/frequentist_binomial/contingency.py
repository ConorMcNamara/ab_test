"""Our wrapper for analyzing experiment results."""

import functools
import itertools
import math
from typing import Any, ClassVar, Literal

import numpy as np
import scipy.stats as ss
from tabulate import tabulate

from ab_test._contingency import BaseContingencyTable
from ab_test._display import convert_to_tabulate_str, format_percent, tabulate_summary
from ab_test._lift import scale_bounds, scale_metric, to_absolute
from ab_test.corrections import adjust_pvalues
from ab_test.frequentist_binomial.confidence_intervals import confidence_interval, individual_confidence_interval
from ab_test.frequentist_binomial.msprt import msprt_test
from ab_test.frequentist_binomial.randomization_inference import randomization_test
from ab_test.frequentist_binomial.stats_tests import (
    ab_test,
    score_test,
    likelihood_ratio_test,
    z_test,
    wald_test,
)
from ab_test.frequentist_binomial.utils import observed_lift

_INVERTIBLE_TESTS = {
    "score": score_test,
    "likelihood": likelihood_ratio_test,
    "z": z_test,
    "wald": wald_test,
}

_Divergence = Literal["pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"]

# Power-divergence statistic for the omnibus test, matching test_method where one exists.
_OMNIBUS_STATISTICS: dict[str, tuple[_Divergence, str]] = {
    "likelihood": ("log-likelihood", "Likelihood-ratio (G)"),
    "modified_likelihood": ("mod-log-likelihood", "Modified log-likelihood"),
    "freeman-tukey": ("freeman-tukey", "Freeman-Tukey"),
    "neyman": ("neyman", "Neyman"),
    "cressie-read": ("cressie-read", "Cressie-Read"),
}


def _omnibus_test(trials: list[Any], successes: list[Any], test_method: str) -> tuple[float, int, float, str]:
    """Test that every cell shares one success rate (a k x 2 power-divergence test).

    Uses the statistic matching ``test_method`` where one exists, and Pearson's
    chi-squared otherwise (the score test's k-group form). Returns the
    statistic, degrees of freedom, p-value and the test's name.
    """
    lambda_, name = _OMNIBUS_STATISTICS.get(test_method, ("pearson", "Pearson chi-squared"))
    df = len(trials) - 1
    total_successes, total_trials = sum(successes), sum(trials)
    if total_successes == 0 or total_successes == total_trials:
        # Every cell has the same (degenerate) rate, so there is nothing to test.
        return 0.0, df, 1.0, name
    table = np.array([[s, t - s] for s, t in zip(successes, trials)], dtype=float)
    if np.any(table == 0):
        # As in the two-group tests (_power_divergence_test).
        if lambda_ in ("neyman", "mod-log-likelihood"):
            raise ValueError(
                f"The {name} statistic is undefined when a cell has zero observed count "
                "(it divides by, or takes the log of, the observed count). "
                "Use the score test or Fisher's exact test instead."
            )
        if lambda_ == "freeman-tukey":
            # scipy evaluates 0 * inf here; the statistic's limit is 4 * sum((sqrt(O) - sqrt(E))**2).
            expected = ss.contingency.expected_freq(table)
            statistic = float(4 * np.sum((np.sqrt(table) - np.sqrt(expected)) ** 2))
            return statistic, df, float(ss.chi2.sf(statistic, df=df)), name
    result = ss.chi2_contingency(table, correction=False, lambda_=lambda_)
    return float(result.statistic), df, float(result.pvalue), name


__all__ = [
    "ContingencyTable",
]


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
    """A class for analyzing experiment results.

    Examples
    --------
    >>> table = ContingencyTable("Checkout redesign", "conversion")
    >>> table = table.add("Control", 1000, 10000).add("Treatment", 1100, 10000)
    >>> print(table.analyze(lift="relative"))  # doctest: +SKIP
    >>> print(table.analyze_individually())  # doctest: +SKIP
    >>> table.plot(is_individual=False)  # doctest: +SKIP
    """

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
        *,
        tau: float | None = None,
        n_permutations: int = 10_000,
        seed: int | None = None,
        comparisons: str = "control",
        correction: str = "holm",
    ) -> str:
        """Analyzes the effect of our experiments through the ContingencyTable.

        Parameters
        ----------
        lift : {'relative', 'absolute', 'incremental', 'roas', 'revenue', 'cpa'}
            The kind of lift we are measuring for our campaign
        test_method : str
            The method we plan to use to assess whether our result is
            statistically significant.  One of ``'score'``, ``'likelihood'``,
            ``'z'``, ``'wald'``, ``'fisher'``, ``'barnard'``, ``'boschloo'``,
            ``'modified_likelihood'``, ``'freeman-tukey'``, ``'neyman'``,
            ``'cressie-read'``, or ``'msprt'``.
        conf_int_method : str
            The method we plan to use to craft confidence intervals of our lift.
            ``'binary_search'`` inverts the test chosen by ``test_method``, so it
            requires ``'score'``, ``'likelihood'``, ``'z'``, ``'wald'`` or
            ``'msprt'``; the other tests only support a null lift of 0, so use a
            method such as ``'wilson'`` with them.
        alpha : float, default = 0.05
            The alpha level of our experiment, to be used to craft confidence intervals.
        null_lift : float
            Lift associated with null hypothesis, in the units of ``lift``.
            Defaults to 0.0. Only ``'score'``, ``'likelihood'``, ``'z'``,
            ``'wald'``, ``'msprt'`` and ``'randomization'`` support a nonzero
            value. Scaled lifts (``'incremental'``, ``'roas'``, ``'revenue'``,
            ``'cpa'``) are converted to a difference in proportions before
            testing. For ``'cpa'``, ``null_lift=0`` means no incremental
            conversions (an unbounded CPA); any other value is a target CPA.
        tau : float or None, optional
            Scale of the Gaussian mixing distribution for the mSPRT test.
            Only used when ``test_method="msprt"``. When ``None``, the scale
            is the larger of ``0.1`` times the pooled success rate and the
            absolute null effect. See
            :func:`~ab_test.frequentist_binomial.msprt.msprt_test`.
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
        every variant shares one success rate, then the chosen pairwise
        comparisons. Each comparison uses ``test_method`` and
        ``conf_int_method`` exactly as a two-variant analysis would. The
        p-values are adjusted with ``correction``, and the intervals are
        Bonferroni intervals (each at level ``1 - alpha / m`` for ``m``
        comparisons), so all of them hold simultaneously with probability
        ``1 - alpha``. The omnibus test uses the power-divergence statistic
        matching ``test_method`` (for example the G-test for
        ``'likelihood'``) and Pearson's chi-squared otherwise; it tests equal
        rates, whatever ``null_lift`` is. ``incremental_results`` then holds
        ``"omnibus"`` and a ``"comparisons"`` dict keyed by labels such as
        ``"B vs A"``, each with the adjusted ``"p_value"`` and the
        ``"raw_p_value"``.
        """
        k = len(self.names)
        if k < 2:
            raise ValueError(f"analyze requires at least 2 variants, got {k}")
        comparisons = comparisons.casefold()
        if comparisons not in ("control", "all"):
            raise ValueError(f"comparisons must be 'control' or 'all', got {comparisons!r}")
        adjust_pvalues([0.5], method=correction)  # Validate the correction before any work.
        # The null lift only affects the p-value, so it is not needed to redraw intervals.
        self._analyze_settings = {
            "test_method": test_method,
            "conf_int_method": conf_int_method,
            "alpha": alpha,
            "tau": tau,
            "n_permutations": n_permutations,
            "seed": seed,
            "comparisons": comparisons,
            "correction": correction,
        }
        lift = lift.casefold()
        if conf_int_method == "binary_search" and test_method not in {*_INVERTIBLE_TESTS, "msprt", "randomization"}:
            raise ValueError(
                f"conf_int_method='binary_search' inverts the significance test, but test_method={test_method!r} "
                "cannot be inverted. Use test_method 'score', 'likelihood', 'z', 'wald' or 'msprt', or a "
                "conf_int_method such as 'wilson'."
            )
        settings: dict[str, Any] = {
            "lift": lift,
            "test_method": test_method,
            "conf_int_method": conf_int_method,
            "null_lift": null_lift,
            "tau": tau,
            "n_permutations": n_permutations,
            "seed": seed,
        }
        if k == 2:
            return self._analyze_pair(alpha, settings)
        return self._analyze_many(alpha, comparisons, correction, settings)

    def _compare(self, i: int, j: int, alpha: float, settings: dict[str, Any]) -> dict[str, Any]:
        """Compare cell ``j`` against cell ``i``: lift, the two rates, p-value and interval."""
        lift = settings["lift"]
        test_method = settings["test_method"]
        null_lift = settings["null_lift"]
        tau = settings["tau"]
        trials = [self.trials[i], self.trials[j]]
        successes = [self.successes[i], self.successes[j]]
        if lift == "relative" and successes[0] == 0:
            if len(self.names) == 2:
                raise ValueError('Relative lift is undefined with no control successes; use lift="absolute"')
            raise ValueError(
                f"Relative lift of {self.names[j]} vs {self.names[i]} is undefined: {self.names[i]} has no "
                'successes; use lift="absolute"'
            )
        test_lift = observed_lift(trials, successes, lift)
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            # The tests work on proportions, so test the null on the same absolute
            # scale that the interval is built on (and that _scale_bound converts back).
            ci_lift = "absolute"
            if lift == "cpa" and null_lift == 0:
                test_null = 0.0
            else:
                test_null = to_absolute(null_lift, lift, max(self.trials), self.spend, self.msrp)
        else:
            ci_lift = lift
            test_null = null_lift
        if test_method == "randomization":
            test_fn = functools.partial(
                randomization_test, n_permutations=settings["n_permutations"], seed=settings["seed"]
            )
            functools.update_wrapper(test_fn, randomization_test)
            p_value = test_fn(trials, successes, test_null, ci_lift)
        elif test_method == "msprt":
            p_value = msprt_test(trials, successes, test_null, ci_lift, tau=tau)
            test_fn = functools.partial(msprt_test, tau=tau)
            functools.update_wrapper(test_fn, msprt_test)
        else:
            p_value = ab_test(trials, successes, test_null, ci_lift, method=test_method)
            # Only used by binary_search, which analyze() limits to invertible tests.
            test_fn = _INVERTIBLE_TESTS.get(test_method, score_test)
        if test_method == "randomization":
            lb, ub = -math.inf, math.inf
        else:
            lb, ub = confidence_interval(
                trials, successes, test=test_fn, alpha=alpha, lift=ci_lift, method=settings["conf_int_method"]
            )
        success_rate: list[int | float]
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            # Every comparison is expressed over the table's largest arm, so with three or
            # more arms identical rate differences give identical scaled lifts. With two
            # arms that is the larger of the pair, as before.
            n_scale = max(self.trials)
            pa: int | float = math.ceil(successes[0] * (n_scale / trials[0]))
            pb: int | float = math.ceil(successes[1] * (n_scale / trials[1]))
            lb = _scale_bound(lb, n_scale)
            ub = _scale_bound(ub, n_scale)
            test_lift = scale_metric(pb - pa, lift, self.spend, self.msrp)
            pa = scale_metric(pa, lift, self.spend, self.msrp)
            pb = scale_metric(pb, lift, self.spend, self.msrp)
            lb, ub = scale_bounds(lb, ub, lift, self.spend, self.msrp)
            success_rate = [pa, pb]
        else:
            success_rate = [si / ti for ti, si in zip(trials, successes)]
        return {"lift": test_lift, "rates": success_rate, "p_value": p_value, "ci_lower": lb, "ci_upper": ub}

    def _analyze_pair(self, alpha: float, settings: dict[str, Any]) -> str:
        """Two variants: one comparison, reported as before multi-arm support."""
        lift = settings["lift"]
        result = self._compare(0, 1, alpha, settings)
        test_lift, success_rate, p_value = result["lift"], result["rates"], result["p_value"]
        lb, ub = result["ci_lower"], result["ci_upper"]
        self.incremental_results = {
            "lift_type": lift,
            "lift": test_lift,
            f"{self.names[0]}": success_rate[0],
            f"{self.names[1]}": success_rate[1],
            "p_value": p_value,
            "ci_lower": lb,
            "ci_upper": ub,
        }
        row_labels = (
            ["Metric", "Metric Name"] + self.names + ["Lift", "Conf. Int. Lower **", "Conf. Int. Upper **", "p-value"]
        )
        str_pvalue = f"{p_value:.4f}" if p_value >= alpha else f"{p_value:.4f}*"
        values = (
            [lift, self.metric_name]
            + convert_to_tabulate_str(success_rate, lift)
            + convert_to_tabulate_str([test_lift, lb, ub], lift)
            + [str_pvalue]
        )
        return_string = tabulate_summary(row_labels, values)
        return_string += (
            f"\n* next to the p-value means it's statistically significant at the {format_percent(alpha)}% level"
        )
        return_string += f"\n** {format_percent(1 - alpha)}% Confidence Interval"
        return return_string

    def _analyze_many(self, alpha: float, comparisons: str, correction: str, settings: dict[str, Any]) -> str:
        """Three or more variants: an omnibus test plus corrected pairwise comparisons."""
        lift = settings["lift"]
        k = len(self.names)
        pairs = [(0, j) for j in range(1, k)] if comparisons == "control" else list(itertools.combinations(range(k), 2))
        # Bonferroni intervals: simultaneous coverage of 1 - alpha across all comparisons.
        ci_alpha = alpha / len(pairs)
        results = [self._compare(i, j, ci_alpha, settings) for i, j in pairs]
        adjusted = adjust_pvalues([r["p_value"] for r in results], method=correction)
        statistic, df, omnibus_p, omnibus_name = _omnibus_test(self.trials, self.successes, settings["test_method"])

        compared: dict[str, dict[str, Any]] = {}
        for (i, j), result, adj_p in zip(pairs, results, adjusted):
            compared[f"{self.names[j]} vs {self.names[i]}"] = {
                "lift": result["lift"],
                f"{self.names[i]}": result["rates"][0],
                f"{self.names[j]}": result["rates"][1],
                "p_value": adj_p,
                "raw_p_value": result["p_value"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
            }
        self.incremental_results = {
            "lift_type": lift,
            "comparison_type": comparisons,
            "correction": correction,
            "omnibus": {"test": omnibus_name, "statistic": statistic, "df": df, "p_value": omnibus_p},
            "comparisons": compared,
        }

        rates = [si / ti for ti, si in zip(self.trials, self.successes)]
        # NaN fails every comparison, so test for significance explicitly rather than with >= alpha.
        str_omnibus = f"{omnibus_p:.4f}*" if omnibus_p < alpha else f"{omnibus_p:.4f}"
        return_string = tabulate_summary(
            ["Metric", "Metric Name"] + self.names + ["Omnibus p-value ***"],
            [lift, self.metric_name] + convert_to_tabulate_str(rates, "absolute") + [str_omnibus],
        )
        rows = []
        for label, comparison in compared.items():
            star = "*" if comparison["p_value"] < alpha else ""
            rows.append(
                [label]
                + convert_to_tabulate_str([comparison["lift"], comparison["ci_lower"], comparison["ci_upper"]], lift)
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
        return_string += "\n" + tabulate(rows, headers=headers, tablefmt="grid")
        return_string += (
            f"\n* next to a p-value means it's statistically significant at the {format_percent(alpha)}% level"
            f" ({correction}-adjusted for {len(pairs)} comparisons)"
        )
        return_string += (
            f"\n** {format_percent(1 - alpha)}% simultaneous Confidence Intervals"
            f" (Bonferroni: each at {round(100 * (1 - ci_alpha), 2):g}%)"
        )
        return_string += f"\n*** {omnibus_name} test that all {k} variants share one rate, df={df}"
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            return_string += f"\nScaled lifts are per {max(self.trials):,} units (the largest arm) for every comparison"
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
        return_string += f"\n** {format_percent(1 - alpha)}% Confidence Interval"
        return return_string
