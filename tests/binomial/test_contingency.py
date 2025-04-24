"""Testing our Contingency Tables"""
import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from ab_test.binomial.contingency import ContingencyTable


class TestContingencyTable:

    @pytest.mark.parametrize("include_total, expected",
                             [
                                 (False, [["Holdout", 100, 1_000], ["Test", 110, 1_000]]),
                                 (True, [["Holdout", 100, 1_000], ["Test", 110, 1_000], ["Total", 210, 2_000]])
                             ])
    def test_contingency_to_list(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_list = ct.to_list(include_total=include_total)
        assert ct_list == expected

    @pytest.mark.parametrize("include_total, expected",
                             [
                                 (False, pd.DataFrame({"cell_name": ["Holdout", "Test"],
                                                       "successes": [100, 110],
                                                       "trials": [1_000, 1_000]})),
                                 (True, pd.DataFrame({"cell_name": ["Holdout", "Test", "Total"],
                                                      "successes": [100, 110, 210],
                                                      "trials": [1_000, 1_000, 2_000], }))
                             ])
    def test_contingency_to_df_pandas(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_df = ct.to_df(include_total=include_total)
        pd.testing.assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize("include_total, expected",
                             [
                                 (False, pl.DataFrame({"cell_name": ["Holdout", "Test"],
                                                       "successes": [100, 110],
                                                       "trials": [1_000, 1_000]})),
                                 (True, pl.DataFrame({"cell_name": ["Holdout", "Test", "Total"],
                                                      "successes": [100, 110, 210],
                                                      "trials": [1_000, 1_000, 2_000], }))
                             ])
    def test_contingency_to_df_polars(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_df = ct.to_df(method="polars", include_total=include_total)
        assert_frame_equal(ct_df, expected)

    @pytest.mark.parametrize("include_total, expected",
                             [
                                 (False, np.array([["Holdout", 100, 1_000], ["Test", 110, 1_000]])),
                                 (True, np.array([["Holdout", 100, 1_000], ["Test", 110, 1_000], ["Total", 210, 2_000]])),
                             ])
    def test_contingency_to_numpy(self, include_total, expected):
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        ct_array = ct.to_numpy(include_total=include_total)
        np.testing.assert_array_equal(ct_array, expected)

    @staticmethod
    def test_contingency_serialize():
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        serial = ct.serialize()
        expected = {
            "experiment_name": "Initial AB Test",
            "spend": None,
            "msrp": None,
            "table": {
                "Holdout": {
                    "successes": 100,
                    "trials": 1_000
                },
                "Test": {
                    "successes": 110,
                    "trials": 1_000
                }
            }
        }
        assert serial == expected

    @staticmethod
    def test_contingency_deserialize():
        ct = ContingencyTable(name="Initial AB Test")
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
        ct = ContingencyTable(name="Initial AB Test")
        ct.add("Holdout", 100, 1_000)
        ct.add("Test", 110, 1_000)
        expected = '\n'.join(['+-------------+-------------+----------+',
                              '| cell_name   |   successes |   trials |',
                              '+=============+=============+==========+',
                              '| Holdout     |         100 |     1000 |',
                              '+-------------+-------------+----------+',
                              '| Test        |         110 |     1000 |',
                              '+-------------+-------------+----------+',
                              '| Total       |         210 |     2000 |',
                              '+-------------+-------------+----------+', ])
        assert expected == str(ct)


if __name__ == '__main__':
    pytest.main()
