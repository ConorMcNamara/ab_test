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
        pd.testing.assert_frame_equal(ct_df.modin.to_pandas(), expected)

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
                    "prob_rope": 0.43,  # default ROPE is +/-10% of the control rate
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
                    "expected_loss": 0.0166,  # E[max(-relative lift, 0)]
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
                    "ci_lower": -16.85,  # unrounded (was ceil-rounded to -16)
                    "ci_upper": 36.85,  # unrounded (was ceil-rounded to 38)
                    "expected_loss": 1.86,  # conversions
                    "prob_rope": 0.43,  # default ROPE is +/-10% of the control rate
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
                    "lift": 0.1,
                    "Holdout": 1.0,
                    "Test": 1.1,
                    "prob_b_greater_a": 0.76625,
                    "ci_lower": -0.16,
                    "ci_upper": 0.38,
                    "expected_loss": 0.0186,  # conversions per dollar
                    "prob_rope": 0.43,
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
                    "ci_lower": -33.71,
                    "ci_upper": 73.71,
                    "expected_loss": 3.72,  # dollars
                    "prob_rope": 0.43,
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
        # In the units of the lift, so the tolerance is relative.
        assert expected["expected_loss"] == pytest.approx(bct.incremental_results["expected_loss"], rel=0.1)
        assert expected["prob_rope"] == pytest.approx(bct.incremental_results["prob_rope"], abs=1e-02)

    @staticmethod
    def test_contingency_cpa():
        bct = BayesianContingencyTable(name="CPA Test", spend=100, metric_name="conversions")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        bct.analyze(lift="cpa")
        r = bct.incremental_results
        assert r["lift_type"] == "cpa"
        assert r["lift"] == pytest.approx(10.0, abs=2)
        assert r["ci_lower"] < r["lift"]
        assert r["ci_upper"] > r["lift"]
        assert np.isnan(r["prob_rope"])

    @staticmethod
    def test_contingency_cpa_shows_rope_as_na():
        bct = BayesianContingencyTable(name="CPA Test", spend=100, metric_name="conversions")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        rope_row = next(line for line in bct.analyze(lift="cpa").splitlines() if "ROPE" in line)
        assert "n/a" in rope_row

    @staticmethod
    @pytest.mark.parametrize(
        "lift, expected",
        [("absolute", "| 0.1"), ("relative", "| 1."), ("incremental", "| 1."), ("revenue", "| $3.")],
    )
    def test_expected_loss_shown_in_lift_units(lift, expected):
        # The loss was a rate difference shown as a percent whatever the lift: "0.19%" next to
        # a relative lift, or next to an incremental lift in conversions.
        np.random.seed(0)
        bct = BayesianContingencyTable(name="Loss", msrp=2, metric_name="sales")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        loss_row = next(line for line in bct.analyze(lift=lift).splitlines() if "Expected Loss" in line)
        assert expected in loss_row
        assert ("%" in loss_row) == (lift in ("absolute", "relative"))

    @staticmethod
    def test_contingency_cpa_loss_labelled_rate_difference():
        bct = BayesianContingencyTable(name="CPA Test", spend=100, metric_name="conversions")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        loss_row = next(line for line in bct.analyze(lift="cpa").splitlines() if "Expected Loss" in line)
        assert "(rate difference)" in loss_row and "%" in loss_row

    @staticmethod
    def test_contingency_cpa_requires_spend():
        bct = BayesianContingencyTable(name="No Spend", metric_name="conversions")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        with pytest.raises(ValueError, match="spend"):
            bct.analyze(lift="cpa")

    @staticmethod
    @pytest.mark.parametrize("confidence_level, label", [(0.95, "95"), (0.9, "90")])
    def test_analyze_footnote_says_credible_interval(confidence_level, label):
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        result = ct.analyze(confidence_level=confidence_level)
        assert f"** {label}% Credible Interval" in result
        assert "Confidence Interval" not in result

    @staticmethod
    def test_contingency_analyze_individual_results():
        ct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000, 1, 1)
        ct.add("Test", 110, 1_000, 1, 1)
        expected = "\n".join(
            [
                "+-------------+-------------+----------+---------------+--------------+------------------+----------------------+----------------------+",  # noqa: E501
                "| Cell Name   |   Successes |   Trials |   Prior Alpha |   Prior Beta | Posterior Mean   | Cred. Int. Lower**   | Cred. Int. Upper**   |",  # noqa: E501
                "+=============+=============+==========+===============+==============+==================+======================+======================+",  # noqa: E501
                "| Holdout     |         100 |     1000 |             1 |            1 | 10.08%           | 8.29%                | 12.02%               |",  # noqa: E501
                "+-------------+-------------+----------+---------------+--------------+------------------+----------------------+----------------------+",  # noqa: E501
                "| Test        |         110 |     1000 |             1 |            1 | 11.08%           | 9.21%                | 13.09%               |",  # noqa: E501
                "+-------------+-------------+----------+---------------+--------------+------------------+----------------------+----------------------+",  # noqa: E501
                "| Total       |         210 |     2000 |             2 |            2 | 10.58%           | 9.27%                | 11.96%               |",  # noqa: E501
                "+-------------+-------------+----------+---------------+--------------+------------------+----------------------+----------------------+",  # noqa: E501
                "** 95% Credible Interval",
            ]
        )
        assert expected == ct.analyze_individually()


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


if __name__ == "__main__":
    pytest.main()


def _rope_table(**kwargs):
    table = BayesianContingencyTable(name="ROPE", metric_name="conversions", **kwargs)
    table.add("Holdout", 100, 1_000, 1, 1)
    table.add("Test", 104, 1_000, 1, 1)
    return table


class TestScaledLiftRopeAndRounding:
    @staticmethod
    def test_default_rope_is_consistent_across_lift_units():
        # Was 1.000 (absolute), 0.006 (incremental), 0.999 (roas) and 0.000 (revenue) for the same data.
        probs = {}
        for lift, kwargs in [
            ("relative", {}),
            ("absolute", {}),
            ("incremental", {}),
            ("roas", {"spend": 500}),
            ("revenue", {"msrp": 20}),
        ]:
            np.random.seed(0)
            table = _rope_table(**kwargs)
            table.analyze(lift=lift)
            probs[lift] = table.incremental_results["prob_rope"]
        assert max(probs.values()) - min(probs.values()) < 0.03

    @staticmethod
    def test_explicit_thresholds_are_in_lift_units():
        np.random.seed(0)
        table = _rope_table()
        table.analyze(lift="incremental", low_threshold=-200, high_threshold=200)
        assert table.incremental_results["prob_rope"] > 0.999

    @staticmethod
    def test_incremental_bounds_are_not_rounded():
        absolute, incremental = _rope_table(), _rope_table()
        absolute.analyze(lift="absolute")
        incremental.analyze(lift="incremental")
        assert incremental.incremental_results["ci_lower"] == pytest.approx(
            1_000 * absolute.incremental_results["ci_lower"]
        )
        assert incremental.incremental_results["ci_upper"] == pytest.approx(
            1_000 * absolute.incremental_results["ci_upper"]
        )
