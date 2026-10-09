"""Testing our Contingency Tables"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.stats as ss
from polars.testing import assert_frame_equal

from ab_test.frequentist_binomial.contingency import ContingencyTable, _omnibus_test
from ab_test.frequentist_binomial.confidence_intervals import confidence_interval
from ab_test.frequentist_binomial.stats_tests import ab_test, likelihood_ratio_test, score_test, wald_test, z_test


class TestContingencyTable:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                [
                    [
                        "Holdout",
                        100,
                        1_000,
                    ],
                    ["Test", 110, 1_000],
                ],
            ),
            (True, [["Holdout", 100, 1_000], ["Test", 110, 1_000], ["Total", 210, 2_000]]),
        ],
    )
    def test_contingency_to_list(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_list = ct.to_list(include_total=include_total)
        assert ct_list == expected

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pd.DataFrame({"cell_name": ["Holdout", "Test"], "successes": [100, 110], "trials": [1_000, 1_000]}),
            ),
            (
                True,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_pandas(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_df = ct.to_df(include_total=include_total)
        pd.testing.assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pl.DataFrame({"cell_name": ["Holdout", "Test"], "successes": [100, 110], "trials": [1_000, 1_000]}),
            ),
            (
                True,
                pl.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_polars(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_df = ct.to_df(method="polars", include_total=include_total)
        assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (False, np.array([["Holdout", 100, 1_000], ["Test", 110, 1_000]])),
            (True, np.array([["Holdout", 100, 1_000], ["Test", 110, 1_000], ["Total", 210, 2_000]])),
        ],
    )
    def test_contingency_to_numpy(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_array = ct.to_numpy(include_total=include_total)
        np.testing.assert_array_equal(ct_array, expected)

    @staticmethod
    def test_contingency_serialize():
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        serial = ct.serialize()
        expected = {
            "experiment_name": "Initial AB Test",
            "metric_name": "sales",
            "spend": None,
            "msrp": None,
            "table": {"Holdout": {"successes": 100, "trials": 1_000}, "Test": {"successes": 110, "trials": 1_000}},
        }
        assert serial == expected

    @staticmethod
    def test_contingency_deserialize():
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        serial = ct.serialize()
        ct_deserialized = ct.deserialize(serial)
        assert ct.cells == ct_deserialized.cells
        assert ct.experiment_name == ct_deserialized.experiment_name
        assert ct.spend == ct_deserialized.spend
        assert ct.msrp == ct_deserialized.msrp

    @staticmethod
    def test_contingency_print():
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        expected = "\n".join(
            [
                "+-------------+-------------+----------+",
                "| cell_name   |   successes |   trials |",
                "+=============+=============+==========+",
                "| Holdout     |         100 |     1000 |",
                "+-------------+-------------+----------+",
                "| Test        |         110 |     1000 |",
                "+-------------+-------------+----------+",
                "| Total       |         210 |     2000 |",
                "+-------------+-------------+----------+",
            ]
        )
        assert expected == str(ct)

    @pytest.mark.parametrize(
        "name, trials, success, lift, expected",
        [
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                "absolute",
                {
                    "lift_type": "absolute",
                    "lift": 0.1,
                    "Holdout": 0.10,
                    "Test": 0.11,
                    "p_value": 0.4657435879336349,
                    "ci_lower": -0.016966857910156258,
                    "ci_upper": 0.037053527832031245,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                "relative",
                {
                    "lift_type": "relative",
                    "lift": 1.0,
                    "Holdout": 0.10,
                    "Test": 0.11,
                    "p_value": 0.4657435879336349,
                    "ci_lower": -0.14798553466796882,
                    "ci_upper": 0.4204476928710939,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                "incremental",
                {
                    "lift_type": "incremental",
                    "lift": 10,
                    "Holdout": 100,
                    "Test": 110,
                    "p_value": 0.4657435879336349,
                    "ci_lower": -16,
                    "ci_upper": 38,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                "roas",
                {
                    "lift_type": "roas",
                    "lift": 0.1,
                    "Holdout": 1.0,
                    "Test": 1.1,
                    "p_value": 0.4657435879336349,
                    "ci_lower": -0.16,
                    "ci_upper": 0.38,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                "revenue",
                {
                    "lift_type": "revenue",
                    "lift": 20,
                    "Holdout": 200,
                    "Test": 220,
                    "p_value": 0.4657435879336349,
                    "ci_lower": -32,
                    "ci_upper": 76,
                },
            ),
        ],
    )
    def test_contingency_results(self, name, trials, success, lift, expected):
        ct = ContingencyTable(name="Initial AB Test", spend=100, msrp=2, metric_name="sales")
        ct.add(name[0], success[0], trials[0])
        ct.add(name[1], success[1], trials[1])
        print(ct.analyze(lift=lift))
        assert ct.incremental_results["lift_type"] == expected["lift_type"]
        assert expected["lift"] == pytest.approx(ct.incremental_results["lift"], abs=1)
        assert expected[f"{name[0]}"] == pytest.approx(ct.incremental_results[f"{name[0]}"])
        assert expected[f"{name[1]}"] == pytest.approx(ct.incremental_results[f"{name[1]}"])
        assert expected["p_value"] == pytest.approx(ct.incremental_results["p_value"])
        assert expected["ci_lower"] == pytest.approx(ct.incremental_results["ci_lower"])
        assert expected["ci_upper"] == pytest.approx(ct.incremental_results["ci_upper"])

    @staticmethod
    def test_contingency_cpa():
        ct = ContingencyTable(name="CPA Test", spend=100, metric_name="conversions")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct.analyze(lift="cpa")
        r = ct.incremental_results
        assert r["lift_type"] == "cpa"
        assert r["lift"] == pytest.approx(10.0)
        assert r["Holdout"] == pytest.approx(1.0)
        assert r["Test"] == pytest.approx(100 / 110)
        assert r["ci_lower"] < r["lift"]
        assert r["ci_upper"] > r["lift"]

    @staticmethod
    def test_contingency_cpa_requires_spend():
        ct = ContingencyTable(name="No Spend", metric_name="conversions")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        with pytest.raises(ValueError, match="spend must be set"):
            ct.analyze(lift="cpa")

    @staticmethod
    def test_contingency_analyze_individual_results():
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        expected = "\n".join(
            [
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",  # noqa: E501
                "| Cell Name   |   Successes |   Trials | Success Rate   | Conf. Int. Lower**   | Conf. Int. Upper**   |",  # noqa: E501
                "+=============+=============+==========+================+======================+======================+",  # noqa: E501
                "| Holdout     |         100 |     1000 | 10.0%          | 8.29%                | 12.02%               |",  # noqa: E501
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",  # noqa: E501
                "| Test        |         110 |     1000 | 11.0%          | 9.21%                | 13.09%               |",  # noqa: E501
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",  # noqa: E501
                "| Total       |         210 |     2000 | 10.5%          | 9.23%                | 11.92%               |",  # noqa: E501
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",  # noqa: E501
                "** 95% Confidence Interval",
            ]
        )
        assert expected == ct.analyze_individually()


class TestContingencyTableWald:
    @staticmethod
    def _table():
        return ContingencyTable("Wald test", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)

    @staticmethod
    def test_analyze_uses_wald_p_value():
        table = TestContingencyTableWald._table()
        table.analyze(lift="absolute", test_method="wald")
        expected = wald_test([1000, 1000], [100, 130], null_lift=0.0, lift="absolute")
        assert table.incremental_results["p_value"] == pytest.approx(expected)

    @staticmethod
    def test_analyze_binary_search_inverts_wald_test():
        table = TestContingencyTableWald._table()
        table.analyze(lift="absolute", test_method="wald", conf_int_method="binary_search")
        lb, ub = confidence_interval([1000, 1000], [100, 130], test=wald_test, lift="absolute")
        assert table.incremental_results["ci_lower"] == pytest.approx(lb)
        assert table.incremental_results["ci_upper"] == pytest.approx(ub)

    @staticmethod
    def test_analyze_relative_lift_not_supported():
        with pytest.raises(NotImplementedError):
            TestContingencyTableWald._table().analyze(lift="relative", test_method="wald")


class TestAnalyzeScaledNullLift:
    @staticmethod
    def _p_value(lift, null_lift):
        table = ContingencyTable("Scaled", "conversion", spend=5000, msrp=50)
        table.add("Control", 100, 1000).add("Treatment", 140, 1000)
        table.analyze(lift=lift, null_lift=null_lift)
        return table.incremental_results["p_value"]

    @pytest.mark.parametrize(
        "lift, null_lift",
        # Each is a 3-point difference in conversion rate at 1,000 per arm.
        [("incremental", 30), ("roas", 0.006), ("revenue", 1500), ("cpa", 5000 / 30)],
    )
    def test_null_is_converted_to_the_absolute_scale(self, lift, null_lift):
        # The null used to be passed through unconverted, e.g. incremental 30 as a
        # difference in proportions of 30, giving p = 1.0.
        expected = self._p_value("absolute", 0.03)
        assert expected == pytest.approx(0.490, abs=1e-3)
        assert self._p_value(lift, null_lift) == pytest.approx(expected)

    @pytest.mark.parametrize("lift", ["incremental", "roas", "revenue", "cpa"])
    def test_zero_null_unchanged(self, lift):
        assert self._p_value(lift, 0.0) == pytest.approx(self._p_value("absolute", 0.0))


class TestAnalyzeZeroControl:
    @staticmethod
    def _table():
        return ContingencyTable("Zero", "conversion").add("Control", 0, 100).add("Treatment", 5, 100)

    def test_relative_lift_raises_clear_error(self):
        # Used to surface a bare ZeroDivisionError from observed_lift.
        with pytest.raises(ValueError, match='use lift="absolute"'):
            self._table().analyze(lift="relative")

    def test_absolute_lift_still_works(self):
        table = self._table()
        table.analyze(lift="absolute")
        assert table.incremental_results["ci_lower"] < 0.05 < table.incremental_results["ci_upper"]


class TestAnalyzeTestMapping:
    @staticmethod
    def _table():
        return ContingencyTable("Mapping", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)

    @pytest.mark.parametrize(
        "method, test",
        [("score", score_test), ("likelihood", likelihood_ratio_test), ("z", z_test), ("wald", wald_test)],
    )
    def test_binary_search_inverts_the_chosen_test(self, method, test):
        table = self._table()
        table.analyze(lift="absolute", test_method=method, conf_int_method="binary_search")
        lb, ub = confidence_interval([1000, 1000], [100, 130], test=test, lift="absolute")
        assert table.incremental_results["ci_lower"] == pytest.approx(lb)
        assert table.incremental_results["ci_upper"] == pytest.approx(ub)

    @pytest.mark.parametrize(
        "method", ["fisher", "barnard", "boschloo", "modified_likelihood", "freeman-tukey", "neyman", "cressie-read"]
    )
    def test_binary_search_with_non_invertible_test_raises(self, method):
        with pytest.raises(ValueError, match=f"test_method={method!r} cannot be inverted"):
            self._table().analyze(lift="absolute", test_method=method, conf_int_method="binary_search")

    @pytest.mark.parametrize("method", ["fisher", "neyman", "cressie-read"])
    def test_non_invertible_test_works_with_non_search_interval(self, method):
        table = self._table()
        table.analyze(lift="absolute", test_method=method, conf_int_method="wilson")
        expected = ab_test([1000, 1000], [100, 130], null_lift=0.0, lift="absolute", method=method)
        assert table.incremental_results["p_value"] == pytest.approx(expected)
        lb, ub = confidence_interval([1000, 1000], [100, 130], lift="absolute", method="wilson")
        assert table.incremental_results["ci_lower"] == pytest.approx(lb)
        assert table.incremental_results["ci_upper"] == pytest.approx(ub)

    @staticmethod
    def test_non_invertible_test_rejects_nonzero_null():
        with pytest.raises(NotImplementedError):
            TestAnalyzeTestMapping._table().analyze(
                lift="absolute", test_method="fisher", conf_int_method="wilson", null_lift=0.02
            )

    @staticmethod
    def test_unknown_test_method_raises():
        with pytest.raises(ValueError):
            TestAnalyzeTestMapping._table().analyze(lift="absolute", test_method="not-a-test", conf_int_method="wilson")


def _three_arms():
    return ContingencyTable("Checkout", "conversion").add("A", 100, 1000).add("B", 120, 1000).add("C", 140, 1000)


class TestMultiArm:
    def test_control_comparisons_match_two_arm_analyses(self):
        table = _three_arms()
        table.analyze()
        results = table.incremental_results
        assert list(results["comparisons"]) == ["B vs A", "C vs A"]
        raw = []
        for label, successes in [("B vs A", [100, 120]), ("C vs A", [100, 140])]:
            comparison = results["comparisons"][label]
            raw_p = ab_test([1000, 1000], successes, method="score")
            # Bonferroni intervals: alpha / 2 each for two comparisons.
            lb, ub = confidence_interval([1000, 1000], successes, test=score_test, alpha=0.025, lift="relative")
            assert comparison["raw_p_value"] == pytest.approx(raw_p)
            assert comparison["ci_lower"] == pytest.approx(lb)
            assert comparison["ci_upper"] == pytest.approx(ub)
            assert comparison["lift"] == pytest.approx(successes[1] / successes[0] - 1)
            raw.append(raw_p)
        # Holm: the smaller p-value is doubled, the larger kept (and never below the first).
        small, large = sorted(raw)
        adjusted = sorted(c["p_value"] for c in results["comparisons"].values())
        assert adjusted == pytest.approx([2 * small, max(large, 2 * small)])
        assert results["correction"] == "holm" and results["comparison_type"] == "control"

    def test_all_pairs(self):
        table = _three_arms()
        table.analyze(lift="absolute", comparisons="all")
        comparisons = table.incremental_results["comparisons"]
        assert list(comparisons) == ["B vs A", "C vs A", "C vs B"]
        assert comparisons["C vs B"]["lift"] == pytest.approx(0.02)
        lb, ub = confidence_interval([1000, 1000], [120, 140], test=score_test, alpha=0.05 / 3, lift="absolute")
        assert (comparisons["C vs B"]["ci_lower"], comparisons["C vs B"]["ci_upper"]) == pytest.approx((lb, ub))

    def test_correction_parameter(self):
        table = _three_arms()
        table.analyze(comparisons="all", correction="bonferroni")
        for comparison in table.incremental_results["comparisons"].values():
            assert comparison["p_value"] == pytest.approx(min(1.0, 3 * comparison["raw_p_value"]))

    @pytest.mark.parametrize(
        "test_method, lambda_, name",
        [("score", "pearson", "Pearson chi-squared"), ("likelihood", "log-likelihood", "Likelihood-ratio (G)")],
    )
    def test_omnibus(self, test_method, lambda_, name):
        table = _three_arms()
        output = table.analyze(test_method=test_method)
        observed = np.array([[100, 900], [120, 880], [140, 860]])
        expected = ss.chi2_contingency(observed, correction=False, lambda_=lambda_)
        omnibus = table.incremental_results["omnibus"]
        assert omnibus["statistic"] == pytest.approx(expected.statistic)
        assert omnibus["p_value"] == pytest.approx(expected.pvalue)
        assert omnibus["df"] == 2 and omnibus["test"] == name
        assert f"{name} test that all 3 variants share one rate, df=2" in output

    def test_omnibus_freeman_tukey_with_a_zero_cell(self):
        # scipy returns NaN here, which the table printed as "nan*".
        table = ContingencyTable("x", "c").add("A", 0, 1000).add("B", 10, 1000).add("C", 12, 1000)
        output = table.analyze(lift="absolute", test_method="freeman-tukey", conf_int_method="wilson")
        observed = np.array([[0, 1000], [10, 990], [12, 988]], dtype=float)
        expected = ss.contingency.expected_freq(observed)
        statistic = 4 * np.sum((np.sqrt(observed) - np.sqrt(expected)) ** 2)
        omnibus = table.incremental_results["omnibus"]
        assert omnibus["statistic"] == pytest.approx(statistic)
        assert omnibus["statistic"] == pytest.approx(32.5286, abs=1e-4)
        assert omnibus["p_value"] == pytest.approx(ss.chi2.sf(statistic, 2))
        assert "nan" not in output

    @pytest.mark.parametrize("test_method", ["neyman", "modified_likelihood"])
    def test_omnibus_undefined_with_a_zero_cell(self, test_method):
        # analyze() would raise first from the pairwise test on the zero cell, so call the omnibus directly.
        # scipy gives NaN (Neyman) or inf (modified log-likelihood) here.
        with pytest.raises(ValueError, match="undefined when a cell has zero observed count"):
            _omnibus_test([1000, 1000, 1000], [0, 10, 12], test_method)

    def test_omnibus_with_no_successes_anywhere(self):
        table = ContingencyTable("x", "c").add("A", 0, 100).add("B", 0, 100).add("C", 0, 100)
        table.analyze(lift="absolute")
        assert table.incremental_results["omnibus"]["p_value"] == 1.0

    def test_scaled_lift_comparisons_match_two_arm_analyses(self):
        table = ContingencyTable("x", "c", spend=5000).add("A", 100, 1000).add("B", 120, 1000).add("C", 140, 1000)
        table.analyze(lift="incremental")
        pair = ContingencyTable("x", "c", spend=5000).add("A", 100, 1000).add("C", 140, 1000)
        pair.analyze(lift="incremental", alpha=0.025)
        comparison = table.incremental_results["comparisons"]["C vs A"]
        for key in ("lift", "ci_lower", "ci_upper"):
            assert comparison[key] == pytest.approx(pair.incremental_results[key])

    def test_output_labels(self):
        output = _three_arms().analyze()
        assert "Adj. p (holm)" in output
        assert "holm-adjusted for 2 comparisons" in output
        assert "95% simultaneous Confidence Intervals (Bonferroni: each at 97.5%)" in output
        assert "Bonferroni: each at 98.33%" in _three_arms().analyze(comparisons="all")

    def test_two_variants_ignore_the_new_options(self):
        table = ContingencyTable("x", "c").add("A", 100, 1000).add("B", 130, 1000)
        default = table.analyze()
        assert table.analyze(comparisons="all", correction="bonferroni") == default
        assert "comparisons" not in table.incremental_results

    @pytest.mark.parametrize(
        "kwargs, match",
        [({"comparisons": "pairs"}, "comparisons must be"), ({"correction": "nope"}, "Unknown method")],
    )
    def test_invalid_options(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _three_arms().analyze(**kwargs)

    def test_single_variant_raises(self):
        with pytest.raises(ValueError, match="at least 2 variants"):
            ContingencyTable("x", "c").add("A", 10, 100).analyze()

    def test_zero_successes_in_reference_arm(self):
        table = ContingencyTable("x", "c").add("A", 0, 1000).add("B", 5, 1000).add("C", 7, 1000)
        with pytest.raises(ValueError, match="B vs A is undefined: A has no successes"):
            table.analyze()
        table.analyze(lift="absolute")
        assert table.incremental_results["comparisons"]["C vs A"]["lift"] == pytest.approx(0.007)


if __name__ == "__main__":
    pytest.main()
