"""Testing our Contingency Tables"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ab_test.binomial.contingency import ContingencyTable


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
                    "lift": 10,
                    "Holdout": 1.00,
                    "Test": 0.909090909090909,
                    "p_value": 0.4657435879336349,
                    "ci_lower": np.inf,
                    "ci_upper": 2.63157894737,
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
        assert ct.results["lift_type"] == expected["lift_type"]
        assert expected["lift"] == pytest.approx(ct.results["lift"], abs=1)
        assert expected[f"{name[0]}"] == pytest.approx(ct.results[f"{name[0]}"])
        assert expected[f"{name[1]}"] == pytest.approx(ct.results[f"{name[1]}"])
        assert expected["p_value"] == pytest.approx(ct.results["p_value"])
        assert expected["ci_lower"] == pytest.approx(ct.results["ci_lower"])
        assert expected["ci_upper"] == pytest.approx(ct.results["ci_upper"])

    @staticmethod
    def test_contingency_analyze_individual_results():
        ct = ContingencyTable(name="Initial AB Test", metric_name="sales")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        expected = "\n".join(
            [
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",
                "| Cell Name   |   Successes |   Trials | Success Rate   | Conf. Int. Lower**   | Conf. Int. Upper**   |",
                "+=============+=============+==========+================+======================+======================+",
                "| Holdout     |         100 |     1000 | 10.0%          | 8.29%                | 12.02%               |",
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",
                "| Test        |         110 |     1000 | 11.0%          | 9.21%                | 13.09%               |",
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",
                "| Total       |         210 |     2000 | 10.5%          | 9.23%                | 11.92%               |",
                "+-------------+-------------+----------+----------------+----------------------+----------------------+",
                "** 95% Confidence Interval",
            ]
        )
        assert expected == ct.analyze_individually()


if __name__ == "__main__":
    pytest.main()
