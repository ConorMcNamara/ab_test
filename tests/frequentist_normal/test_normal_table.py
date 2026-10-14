"""Testing our Normal Tables."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ab_test.corrections import adjust_pvalues
from ab_test.frequentist_normal.normal_table import NormalTable
from ab_test.frequentist_normal.stats_tests import score_omnibus_test, score_test, welch_anova


class TestNormalTable:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                [
                    ["Holdout", 10.0, 4.0, 1000],
                    ["Test", 11.0, 5.0, 1000],
                ],
            ),
            (
                True,
                [
                    ["Holdout", 10.0, 4.0, 1000],
                    ["Test", 11.0, 5.0, 1000],
                    ["Total", 10.5, 4.75, 2000],
                ],
            ),
        ],
    )
    def test_to_list(self, include_total, expected):
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        result = nt.to_list(include_total=include_total)
        if include_total:
            for i in range(len(expected)):
                assert result[i][0] == expected[i][0]
                assert result[i][1] == pytest.approx(expected[i][1])
                assert result[i][2] == pytest.approx(expected[i][2])
                assert result[i][3] == expected[i][3]
        else:
            assert result == expected

    def test_to_df_pandas(self):
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        expected = pd.DataFrame(
            {
                "cell_name": ["Holdout", "Test"],
                "means": [10.0, 11.0],
                "variances": [4.0, 5.0],
                "trials": [1000, 1000],
            }
        )
        pd.testing.assert_frame_equal(nt.to_df(), expected)

    def test_to_df_polars(self):
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        expected = pl.DataFrame(
            {
                "cell_name": ["Holdout", "Test"],
                "means": [10.0, 11.0],
                "variances": [4.0, 5.0],
                "trials": [1000, 1000],
            }
        )
        assert_frame_equal(nt.to_df(method="polars"), expected)

    @pytest.mark.parametrize(
        "include_total",
        [False, True],
    )
    def test_to_numpy(self, include_total):
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        result = nt.to_numpy(include_total=include_total)
        assert isinstance(result, np.ndarray)
        if include_total:
            assert result.shape[0] == 3
        else:
            assert result.shape[0] == 2

    @staticmethod
    def test_serialize():
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        serial = nt.serialize()
        expected = {
            "experiment_name": "Test Experiment",
            "metric_name": "revenue",
            "spend": None,
            "msrp": None,
            "table": {
                "Holdout": {"means": 10.0, "variances": 4.0, "trials": 1000},
                "Test": {"means": 11.0, "variances": 5.0, "trials": 1000},
            },
        }
        assert serial == expected

    @staticmethod
    def test_deserialize():
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        serial = nt.serialize()
        nt_deserialized = NormalTable(name="", metric_name="").deserialize(serial)
        assert nt.cells == nt_deserialized.cells
        assert nt.experiment_name == nt_deserialized.experiment_name
        assert nt.means == nt_deserialized.means
        assert nt.variances == nt_deserialized.variances
        assert nt.trials == nt_deserialized.trials

    @staticmethod
    def test_print():
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        result = str(nt)
        assert "Holdout" in result
        assert "Test" in result
        assert "Total" in result
        assert "cell_name" in result
        assert "means" in result
        assert "variances" in result
        assert "trials" in result

    @staticmethod
    def test_add_data():
        rng = np.random.default_rng(42)
        data_a = rng.normal(10, 2, 500)
        data_b = rng.normal(11, 2, 500)
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add_data("A", data_a)
        nt.add_data("B", data_b)
        assert nt.means[0] == pytest.approx(float(np.mean(data_a)))
        assert nt.means[1] == pytest.approx(float(np.mean(data_b)))
        assert nt.variances[0] == pytest.approx(float(np.var(data_a, ddof=1)))
        assert nt.variances[1] == pytest.approx(float(np.var(data_b, ddof=1)))
        assert nt.trials == [500, 500]

    @staticmethod
    def test_add_data_accepts_list_tuple_and_array():
        values = [9.5, 10.0, 11.5, 12.0, 8.0]
        tables = [
            NormalTable(name="Test", metric_name="metric").add_data("A", data)
            for data in (values, tuple(values), np.array(values))
        ]
        for table in tables:
            assert table.means == pytest.approx([float(np.mean(values))])
            assert table.variances == pytest.approx([float(np.var(values, ddof=1))])
            assert table.trials == [5]

    @staticmethod
    def test_add_data_counts_every_observation_in_2d_input():
        data = np.arange(12, dtype=float).reshape(3, 4)
        nt = NormalTable(name="Test", metric_name="metric").add_data("A", data)
        assert nt.trials == [12]
        assert nt.means == pytest.approx([float(np.mean(data))])
        assert nt.variances == pytest.approx([float(np.var(data, ddof=1))])

    @staticmethod
    def test_add_chaining():
        nt = NormalTable(name="Test", metric_name="metric")
        result = nt.add("A", 10.0, 4.0, 1000).add("B", 11.0, 5.0, 1000)
        assert result is nt
        assert len(nt.names) == 2

    @staticmethod
    def test_analyze_requires_two_variants():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        with pytest.raises(ValueError, match="analyze requires at least 2 variants"):
            nt.analyze()

    @pytest.mark.parametrize(
        "lift, expected",
        [
            (
                "relative",
                {
                    "lift_type": "relative",
                    "lift": 0.1,
                    "Holdout": 10.0,
                    "Test": 11.0,
                    "ci_lower": 0.08054600153509416,
                    "ci_upper": 0.11945399846490586,
                },
            ),
            (
                "absolute",
                {
                    "lift_type": "absolute",
                    "lift": 1.0,
                    "Holdout": 10.0,
                    "Test": 11.0,
                    "ci_lower": 0.8139473915818949,
                    "ci_upper": 1.1860526084181051,
                },
            ),
            (
                "incremental",
                {
                    "lift_type": "incremental",
                    "lift": 1000.0,
                    "Holdout": 10000.0,
                    "Test": 11000.0,
                    "ci_lower": 813.9473915818949,
                    "ci_upper": 1186.0526084181051,
                },
            ),
            (
                "roas",
                {
                    "lift_type": "roas",
                    "lift": 0.2,
                    "Holdout": 2.0,
                    "Test": 2.2,
                    "ci_lower": 0.16278947831637897,
                    "ci_upper": 0.23721052168362103,
                },
            ),
            (
                "revenue",
                {
                    "lift_type": "revenue",
                    "lift": 50000.0,
                    "Holdout": 500000.0,
                    "Test": 550000.0,
                    "ci_lower": 40697.369579094746,
                    "ci_upper": 59302.630420905254,
                },
            ),
        ],
    )
    def test_analyze_results(self, lift, expected):
        nt = NormalTable(name="Test Experiment", spend=5000.0, msrp=50.0, metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        nt.analyze(lift=lift)
        r = nt.incremental_results
        assert r["lift_type"] == expected["lift_type"]
        assert r["lift"] == pytest.approx(expected["lift"])
        assert r["Holdout"] == pytest.approx(expected["Holdout"])
        assert r["Test"] == pytest.approx(expected["Test"])
        assert r["ci_lower"] == pytest.approx(expected["ci_lower"])
        assert r["ci_upper"] == pytest.approx(expected["ci_upper"])

    @staticmethod
    def test_analyze_cpa():
        nt = NormalTable(name="CPA Test", spend=5000.0, metric_name="conversions")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        nt.analyze(lift="cpa")
        r = nt.incremental_results
        assert r["lift_type"] == "cpa"
        assert r["lift"] == pytest.approx(5.0)
        assert r["ci_lower"] < r["lift"]
        assert r["ci_upper"] > r["lift"]

    @staticmethod
    def test_analyze_roas_requires_spend():
        nt = NormalTable(name="No Spend", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        with pytest.raises(ValueError, match="spend must be set"):
            nt.analyze(lift="roas")

    @staticmethod
    def test_analyze_cpa_requires_spend():
        nt = NormalTable(name="No Spend", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        with pytest.raises(ValueError, match="spend must be set"):
            nt.analyze(lift="cpa")

    @staticmethod
    def test_analyze_revenue_requires_msrp():
        nt = NormalTable(name="No MSRP", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        with pytest.raises(ValueError, match="msrp must be set"):
            nt.analyze(lift="revenue")

    @staticmethod
    @pytest.mark.parametrize("test_method", ["welch", "score"])
    @pytest.mark.parametrize(
        "lift, null_lift",
        # Each null is an absolute difference in means of 0.1, over 1,000 units,
        # with spend=2000 and msrp=50.
        [("incremental", 100.0), ("roas", 0.05), ("revenue", 5000.0), ("cpa", 20.0)],
    )
    def test_analyze_scaled_null_lift_matches_absolute(test_method, lift, null_lift):
        def p_value(lift, null_lift):
            nt = NormalTable(name="Scaled Null", metric_name="revenue", spend=2000.0, msrp=50.0)
            nt.add("Holdout", 2.0, 4.0, 1000)
            nt.add("Test", 2.2, 5.0, 1000)
            nt.analyze(lift=lift, test_method=test_method, null_lift=null_lift)
            return nt.incremental_results["p_value"]

        expected = p_value("absolute", 0.1)
        assert 0.01 < expected < 0.99
        assert p_value(lift, null_lift) == pytest.approx(expected)

    @staticmethod
    def test_analyze_returns_string():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        nt.add("B", 11.0, 5.0, 1000)
        result = nt.analyze()
        assert isinstance(result, str)
        assert "Conf. Int." in result
        assert "p-value" in result

    @staticmethod
    def test_analyze_pvalue_star():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        nt.add("B", 11.0, 5.0, 1000)
        result = nt.analyze()
        assert "*" in result

    @staticmethod
    def test_analyze_not_significant_no_star():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 100)
        nt.add("B", 10.05, 5.0, 100)
        result = nt.analyze()
        assert "statistically significant" in result

    @staticmethod
    def test_analyze_score_test():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        nt.add("B", 11.0, 5.0, 1000)
        result = nt.analyze(test_method="score", conf_int_method="binary_search")
        assert "p-value" in result
        assert "*" in result

    @staticmethod
    def test_analyze_score_equal_var_changes_result():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 1.0, 300)
        nt.add("B", 10.3, 9.0, 150)
        pooled = nt.analyze(lift="absolute", test_method="score", conf_int_method="binary_search")
        separate = nt.analyze(lift="absolute", test_method="score", conf_int_method="binary_search", equal_var=False)
        assert pooled != separate

    @staticmethod
    def test_analyze_score_unequal_var_uses_unequal_score_test(monkeypatch):
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs.get("equal_var"))
            return score_test(*args, **kwargs)

        monkeypatch.setattr("ab_test.frequentist_normal.normal_table.score_test", spy)
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 1.0, 300)
        nt.add("B", 10.3, 9.0, 150)
        nt.analyze(lift="absolute", test_method="score", conf_int_method="binary_search", equal_var=False)
        assert calls
        assert all(c is False for c in calls)

    @staticmethod
    def test_analyze_welch_ignores_equal_var():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 1.0, 300)
        nt.add("B", 10.3, 9.0, 150)
        assert nt.analyze(equal_var=True) == nt.analyze(equal_var=False)

    @staticmethod
    def test_analyze_unknown_test_method_raises():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        nt.add("B", 11.0, 5.0, 1000)
        with pytest.raises(NotImplementedError):
            nt.analyze(test_method="fisher")


def _three_arm_table(**kwargs):
    nt = NormalTable(name="Three Arms", metric_name="order value", **kwargs)
    nt.add("Control", 52.0, 144.0, 1000).add("T1", 54.5, 150.0, 1000).add("T2", 53.0, 160.0, 900)
    return nt


def _pair_table(names, means, variances, trials, **kwargs):
    nt = NormalTable(name="Pair", metric_name="order value", **kwargs)
    for name, mean, variance, n in zip(names, means, variances, trials):
        nt.add(name, mean, variance, n)
    return nt


class TestNormalTableMultiGroup:
    @staticmethod
    @pytest.mark.parametrize(
        "comparisons, expected",
        [("control", ["T1 vs Control", "T2 vs Control"]), ("all", ["T1 vs Control", "T2 vs Control", "T2 vs T1"])],
    )
    def test_comparison_labels(comparisons, expected):
        nt = _three_arm_table()
        nt.analyze(lift="absolute", comparisons=comparisons)
        assert list(nt.incremental_results["comparisons"]) == expected
        assert nt.incremental_results["comparison_type"] == comparisons

    @staticmethod
    @pytest.mark.parametrize("test_method", ["welch", "score"])
    @pytest.mark.parametrize("lift", ["absolute", "relative"])
    def test_comparisons_match_two_variant_tables(test_method, lift):
        # Each comparison is a two-variant analysis at the Bonferroni level alpha / m.
        nt = _three_arm_table()
        nt.analyze(lift=lift, test_method=test_method, comparisons="all", alpha=0.06)
        for label, (i, j) in {"T1 vs Control": (0, 1), "T2 vs Control": (0, 2), "T2 vs T1": (1, 2)}.items():
            pair = _pair_table(
                [nt.names[i], nt.names[j]],
                [nt.means[i], nt.means[j]],
                [nt.variances[i], nt.variances[j]],
                [nt.trials[i], nt.trials[j]],
            )
            pair.analyze(lift=lift, test_method=test_method, alpha=0.02)
            comparison = nt.incremental_results["comparisons"][label]
            for key in ["lift", "ci_lower", "ci_upper"]:
                assert comparison[key] == pytest.approx(pair.incremental_results[key])
            assert comparison["raw_p_value"] == pytest.approx(pair.incremental_results["p_value"])

    @staticmethod
    @pytest.mark.parametrize("correction", ["holm", "bonferroni", "bh"])
    def test_p_values_are_adjusted(correction):
        nt = _three_arm_table()
        nt.analyze(lift="absolute", comparisons="all", correction=correction)
        comparisons = nt.incremental_results["comparisons"].values()
        raw = [c["raw_p_value"] for c in comparisons]
        assert [c["p_value"] for c in comparisons] == pytest.approx(adjust_pvalues(raw, method=correction))
        assert nt.incremental_results["correction"] == correction

    @staticmethod
    def test_welch_omnibus():
        nt = _three_arm_table()
        nt.analyze(lift="absolute")
        statistic, df_num, df_denom, pval = welch_anova(nt.means, nt.variances, nt.trials)
        assert nt.incremental_results["omnibus"] == {
            "test": "Welch's ANOVA",
            "statistic": pytest.approx(statistic),
            "df": df_num,
            "df_denom": pytest.approx(df_denom),
            "p_value": pytest.approx(pval),
        }

    @staticmethod
    @pytest.mark.parametrize("equal_var", [True, False])
    def test_score_omnibus(equal_var):
        nt = _three_arm_table()
        nt.analyze(lift="absolute", test_method="score", equal_var=equal_var)
        statistic, df, pval = score_omnibus_test(nt.means, nt.variances, nt.trials, equal_var=equal_var)
        assert nt.incremental_results["omnibus"] == {
            "test": "Score",
            "statistic": pytest.approx(statistic),
            "df": df,
            "p_value": pytest.approx(pval),
        }

    @staticmethod
    def test_omnibus_ignores_null_lift():
        nt = _three_arm_table()
        nt.analyze(lift="absolute")
        omnibus = nt.incremental_results["omnibus"]
        nt.analyze(lift="absolute", null_lift=1.0)
        assert nt.incremental_results["omnibus"] == omnibus

    @staticmethod
    def test_scaled_lifts_use_the_largest_arm():
        # T2 vs T1 has at most 900 trials, but is scaled to the table's 1,000 like the other comparisons.
        def table():
            nt = NormalTable(name="Uneven Arms", metric_name="order value")
            return nt.add("Control", 52.0, 144.0, 1000).add("T1", 54.5, 150.0, 800).add("T2", 53.0, 160.0, 900)

        nt = table()
        nt.analyze(lift="incremental", comparisons="all")
        nt_abs = table()
        nt_abs.analyze(lift="absolute", comparisons="all")
        for label, comparison in nt.incremental_results["comparisons"].items():
            expected = nt_abs.incremental_results["comparisons"][label]
            for key in ["lift", "ci_lower", "ci_upper"]:
                assert comparison[key] == pytest.approx(expected[key] * 1000)
        assert nt.incremental_results["comparisons"]["T2 vs T1"]["T1"] == pytest.approx(54.5 * 1000)

    @staticmethod
    def test_scaled_null_lift_uses_the_largest_arm():
        def table():
            nt = NormalTable(name="Uneven Arms", metric_name="order value")
            return nt.add("Control", 52.0, 144.0, 1000).add("T1", 54.5, 150.0, 800).add("T2", 53.0, 160.0, 900)

        nt = table()
        nt.analyze(lift="incremental", null_lift=1000.0, comparisons="all")
        nt_abs = table()
        nt_abs.analyze(lift="absolute", null_lift=1.0, comparisons="all")
        for label, comparison in nt.incremental_results["comparisons"].items():
            expected = nt_abs.incremental_results["comparisons"][label]["raw_p_value"]
            assert comparison["raw_p_value"] == pytest.approx(expected)

    @staticmethod
    def test_relative_lift_with_zero_base_mean_raises():
        nt = NormalTable(name="Zero", metric_name="metric")
        nt.add("A", 1.0, 1.0, 100).add("B", 0.0, 1.0, 100).add("C", 2.0, 1.0, 100)
        with pytest.raises(ValueError, match="Relative lift of C vs B is undefined"):
            nt.analyze(lift="relative", comparisons="all")

    @staticmethod
    def test_invalid_comparisons_raises():
        with pytest.raises(ValueError, match="comparisons must be"):
            _three_arm_table().analyze(comparisons="pairs")

    @staticmethod
    def test_invalid_correction_raises():
        with pytest.raises(ValueError):
            _three_arm_table().analyze(correction="not-a-correction")

    @staticmethod
    def test_output():
        result = _three_arm_table().analyze(lift="absolute", comparisons="all")
        assert "Omnibus p-value ***" in result
        assert "T2 vs T1" in result
        assert "Adj. p (holm)" in result
        assert "(holm-adjusted for 3 comparisons)" in result
        assert "Bonferroni: each at 98.33%" in result
        assert "*** Welch's ANOVA test that all 3 variants share one mean, df=(2, " in result

    @staticmethod
    @pytest.mark.parametrize("lift", ["absolute", "incremental", "relative", "roas", "revenue", "cpa"])
    def test_output_formats_every_lift(lift):
        result = _three_arm_table(spend=5000.0, msrp=2.0).analyze(lift=lift)
        assert "T2 vs Control" in result
        if lift in ["incremental", "roas", "revenue", "cpa"]:
            assert "Scaled lifts are per 1,000 units" in result

    @staticmethod
    def test_plot_has_one_row_per_comparison(monkeypatch):
        figures = []
        monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: figures.append(self))
        nt = _three_arm_table()
        nt.analyze(lift="absolute", comparisons="all")
        nt.plot(is_individual=False)
        assert [trace.name for trace in figures[0].data] == ["T1 vs Control", "T2 vs Control", "T2 vs T1"]


if __name__ == "__main__":
    pytest.main()
