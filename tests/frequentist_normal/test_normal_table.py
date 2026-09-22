"""Testing our Normal Tables."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ab_test.frequentist_normal.normal_table import NormalTable


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
        expected = pd.DataFrame({
            "cell_name": ["Holdout", "Test"],
            "means": [10.0, 11.0],
            "variances": [4.0, 5.0],
            "trials": [1000, 1000],
        })
        pd.testing.assert_frame_equal(nt.to_df(), expected)

    def test_to_df_polars(self):
        nt = NormalTable(name="Test Experiment", metric_name="revenue")
        nt.add("Holdout", 10.0, 4.0, 1000)
        nt.add("Test", 11.0, 5.0, 1000)
        expected = pl.DataFrame({
            "cell_name": ["Holdout", "Test"],
            "means": [10.0, 11.0],
            "variances": [4.0, 5.0],
            "trials": [1000, 1000],
        })
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
    def test_add_chaining():
        nt = NormalTable(name="Test", metric_name="metric")
        result = nt.add("A", 10.0, 4.0, 1000).add("B", 11.0, 5.0, 1000)
        assert result is nt
        assert len(nt.names) == 2

    @staticmethod
    def test_analyze_requires_two_variants():
        nt = NormalTable(name="Test", metric_name="metric")
        nt.add("A", 10.0, 4.0, 1000)
        with pytest.raises(ValueError, match="analyze requires exactly 2 variants"):
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
                    "ci_lower": 0.0805342057292233,
                    "ci_upper": 0.1194657942707767,
                },
            ),
            (
                "absolute",
                {
                    "lift_type": "absolute",
                    "lift": 1.0,
                    "Holdout": 10.0,
                    "Test": 11.0,
                    "ci_lower": 0.8138359430750653,
                    "ci_upper": 1.1861640569249348,
                },
            ),
            (
                "incremental",
                {
                    "lift_type": "incremental",
                    "lift": 1000.0,
                    "Holdout": 10000.0,
                    "Test": 11000.0,
                    "ci_lower": 813.8359430750653,
                    "ci_upper": 1186.1640569249348,
                },
            ),
            (
                "roas",
                {
                    "lift_type": "roas",
                    "lift": 0.2,
                    "Holdout": 2.0,
                    "Test": 2.2,
                    "ci_lower": 0.16276718861501305,
                    "ci_upper": 0.23723281138498697,
                },
            ),
            (
                "revenue",
                {
                    "lift_type": "revenue",
                    "lift": 50000.0,
                    "Holdout": 500000.0,
                    "Test": 550000.0,
                    "ci_lower": 40691.797153753265,
                    "ci_upper": 59308.20284624674,
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


if __name__ == "__main__":
    pytest.main()
