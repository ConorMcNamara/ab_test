"""Testing our Contingency Tables"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ab_test.bayesian_binomial.contingency import BayesianContingencyTable

_pyspark_available = False
try:
    from pyspark.sql import SparkSession as _SparkSession  # noqa: F401

    _pyspark_available = True
except Exception:
    pass


@pytest.fixture(scope="session")
def spark_session():
    if not _pyspark_available:
        pytest.skip("pyspark not available or incompatible with current Python version")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local").appName("ab_test_tests").getOrCreate()
    yield spark
    spark.stop()


class TestBayesianContingencyTable:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                [
                    ["Holdout", 100, 1_000, 1, 1],
                    ["Test", 110, 1_000, 1, 1],
                ],
            ),
            (True, [["Holdout", 100, 1_000, 1, 1], ["Test", 110, 1_000, 1, 1], ["Total", 210, 2_000, np.nan, np.nan]]),
        ],
    )
    def test_contingency_to_list(self, include_total, expected):
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_list = ct.to_list(include_total=include_total)
        assert ct_list == expected

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test"],
                        "successes": [100, 110],
                        "trials": [1_000, 1_000],
                        "alpha": [1, 1],
                        "beta": [1, 1],
                    }
                ),
            ),
            (
                True,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                        "alpha": [1, 1, np.nan],
                        "beta": [1, 1, np.nan],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_pandas(self, include_total, expected):
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_df = ct.to_df(include_total=include_total)
        pd.testing.assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pl.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test"],
                        "successes": [100, 110],
                        "trials": [1_000, 1_000],
                        "alpha": [1, 1],
                        "beta": [1, 1],
                    }
                ),
            ),
            (
                True,
                pl.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                        "alpha": [1.0, 1.0, np.nan],
                        "beta": [1.0, 1.0, np.nan],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_polars(self, include_total, expected):
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_df = ct.to_df(method="polars", include_total=include_total)
        assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (False, np.array([["Holdout", 100, 1_000, 1, 1], ["Test", 110, 1_000, 1, 1]])),
            (
                True,
                np.array(
                    [["Holdout", 100, 1_000, 1, 1], ["Test", 110, 1_000, 1, 1], ["Total", 210, 2_000, np.nan, np.nan]]
                ),
            ),
        ],
    )
    def test_contingency_to_numpy(self, include_total, expected):
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_array = ct.to_numpy(include_total=include_total)
        np.testing.assert_array_equal(ct_array, expected)

    @staticmethod
    def test_contingency_serialize():
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        serial = ct.serialize()
        expected = {
            "experiment_name": "Initial AB Test",
            "metric_name": "sales",
            "spend": None,
            "msrp": None,
            "table": {
                "Holdout": {"successes": 100, "trials": 1_000, "alpha": 1, "beta": 1},
                "Test": {"successes": 110, "trials": 1_000, "alpha": 1, "beta": 1},
            },
        }
        assert serial == expected

    @staticmethod
    def test_contingency_print():
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, alpha=1, beta=1)
        ct.add("Test", 110, 1_000, 1, 1)
        expected = "\n".join(
            [
                "+-------------+-------------+----------+---------+--------+",
                "| cell_name   |   successes |   trials |   alpha |   beta |",
                "+=============+=============+==========+=========+========+",
                "| Holdout     |         100 |     1000 |       1 |      1 |",
                "+-------------+-------------+----------+---------+--------+",
                "| Test        |         110 |     1000 |       1 |      1 |",
                "+-------------+-------------+----------+---------+--------+",
                "| Total       |         210 |     2000 |     nan |    nan |",
                "+-------------+-------------+----------+---------+--------+",
            ]
        )
        assert expected == str(ct)

    @pytest.mark.parametrize(
        "name, trials, success, alpha, beta, lift, expected",
        [
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                [0, 0],
                [0, 0],
                "absolute",
                {
                    "lift_type": "absolute",
                    "lift": 0.1,
                    "Holdout": 0.10,
                    "Test": 0.11,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": -0.016966857910156258,
                    "ci_upper": 0.037053527832031245,
                    "expected_loss": 0.0018725,
                    "prob_rope": 1,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                [0, 0],
                [0, 0],
                "relative",
                {
                    "lift_type": "relative",
                    "lift": 1.0,
                    "Holdout": 0.10,
                    "Test": 0.11,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": -0.14798553466796882,
                    "ci_upper": 0.4204476928710939,
                    "expected_loss": 0.0018725,
                    "prob_rope": 0.4369,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                [0, 0],
                [0, 0],
                "incremental",
                {
                    "lift_type": "incremental",
                    "lift": 10,
                    "Holdout": 100,
                    "Test": 110,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": -16,
                    "ci_upper": 38,
                    "expected_loss": 0.0018725,
                    "prob_rope": 0.0044,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                [0, 0],
                [0, 0],
                "roas",
                {
                    "lift_type": "roas",
                    "lift": 10,
                    "Holdout": 1.00,
                    "Test": 0.909090909090909,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": np.inf,
                    "ci_upper": 2.63157894737,
                    "expected_loss": 0.0018725,
                    "prob_rope": 0.4651,
                },
            ),
            (
                ["Holdout", "Test"],
                [1_000, 1_000],
                [100, 110],
                [0, 0],
                [0, 0],
                "revenue",
                {
                    "lift_type": "revenue",
                    "lift": 20,
                    "Holdout": 200,
                    "Test": 220,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": -32,
                    "ci_upper": 76,
                    "expected_loss": 0.0018725,
                    "prob_rope": 0.0021,
                },
            ),
        ],
    )
    def test_contingency_results(self, name, trials, success, alpha, beta, lift, expected):
        bct = BayesianContingencyTable(name="Initial AB Test", spend=100, msrp=2, metric_name="sales")
        bct.add(name[0], success[0], trials[0], alpha[0], beta[0])
        bct.add(name[1], success[1], trials[1], alpha[1], beta[1])
        print(bct.analyze(lift=lift))
        assert bct.incremental_results["lift_type"] == expected["lift_type"]
        assert expected["lift"] == pytest.approx(bct.incremental_results["lift"], abs=1)
        assert expected[f"{name[0]}"] == pytest.approx(bct.incremental_results[f"{name[0]}"])
        assert expected[f"{name[1]}"] == pytest.approx(bct.incremental_results[f"{name[1]}"])
        assert expected["prob_b_greater_a"] == pytest.approx(bct.incremental_results["prob_b_greater_a"], abs=1e-02)
        if lift == "incremental":
            assert expected["ci_lower"] == pytest.approx(bct.incremental_results["ci_lower"], abs=1)
            assert expected["ci_upper"] == pytest.approx(bct.incremental_results["ci_upper"], abs=1)
        elif lift in ["roas", "relative"]:
            assert expected["ci_lower"] == pytest.approx(bct.incremental_results["ci_lower"], abs=1e-01)
            assert expected["ci_upper"] == pytest.approx(bct.incremental_results["ci_upper"], abs=1e-01)
        elif lift == "revenue":
            assert expected["ci_lower"] == pytest.approx(bct.incremental_results["ci_lower"], abs=2)
            assert expected["ci_upper"] == pytest.approx(bct.incremental_results["ci_upper"], abs=2)
        else:
            assert expected["ci_lower"] == pytest.approx(bct.incremental_results["ci_lower"], abs=1e-02)
            assert expected["ci_upper"] == pytest.approx(bct.incremental_results["ci_upper"], abs=1e-02)
        assert expected["expected_loss"] == pytest.approx(bct.incremental_results["expected_loss"], abs=1e-03)
        assert expected["prob_rope"] == pytest.approx(bct.incremental_results["prob_rope"], abs=1e-02)


class TestBayesianContingencyTableModin:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test"],
                        "successes": [100, 110],
                        "trials": [1_000, 1_000],
                        "alpha": [1, 1],
                        "beta": [1, 1],
                    }
                ),
            ),
            (
                True,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                        "alpha": [1, 1, np.nan],
                        "beta": [1, 1, np.nan],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_modin(self, include_total, expected):
        mpd = pytest.importorskip("modin.pandas")
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_df = ct.to_df(method="modin", include_total=include_total)
        assert isinstance(ct_df, mpd.DataFrame)
        pd.testing.assert_frame_equal(ct_df.to_pandas(), expected)


@pytest.mark.skipif(not _pyspark_available, reason="pyspark not available or incompatible with current Python version")
class TestBayesianContingencyTablePySpark:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test"],
                        "successes": [100, 110],
                        "trials": [1_000, 1_000],
                        "alpha": [1.0, 1.0],
                        "beta": [1.0, 1.0],
                    }
                ),
            ),
            (
                True,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                        "alpha": [1.0, 1.0, np.nan],
                        "beta": [1.0, 1.0, np.nan],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_pyspark(self, spark_session, include_total, expected):
        from pyspark.sql import DataFrame as SparkDataFrame

        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_df = ct.to_df(method="pyspark", include_total=include_total, spark_session=spark_session)
        assert isinstance(ct_df, SparkDataFrame)
        pd.testing.assert_frame_equal(ct_df.toPandas(), expected, check_dtype=False)


class TestBayesianContingencyTableNarwhals:
    @pytest.mark.parametrize(
        "include_total, expected",
        [
            (
                False,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test"],
                        "successes": [100, 110],
                        "trials": [1_000, 1_000],
                        "alpha": [1, 1],
                        "beta": [1, 1],
                    }
                ),
            ),
            (
                True,
                pd.DataFrame(
                    {
                        "cell_name": ["Holdout", "Test", "Total"],
                        "successes": [100, 110, 210],
                        "trials": [1_000, 1_000, 2_000],
                        "alpha": [1, 1, np.nan],
                        "beta": [1, 1, np.nan],
                    }
                ),
            ),
        ],
    )
    def test_contingency_to_df_narwhals(self, include_total, expected):
        nw = pytest.importorskip("narwhals")
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        ct_df = ct.to_df(method="narwhals", include_total=include_total)
        assert isinstance(ct_df, nw.DataFrame)
        pd.testing.assert_frame_equal(nw.to_native(ct_df), expected)


if __name__ == "__main__":
    pytest.main()
