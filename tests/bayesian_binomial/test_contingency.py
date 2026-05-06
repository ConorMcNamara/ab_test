"""Testing our Contingency Tables"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ab_test.bayesian_binomial.contingency import BayesianContingencyTable

class TestBayesianContingencyTable:

    @staticmethod
    def test_contingency_results():
        bct = BayesianContingencyTable(name="Initial AB Test", metric_name="sales")
        bct.add("Holdout", 100, 1_000, 1, 1)
        bct.add("Test", 110, 1_000, 1, 1)
        print(bct.analyze(lift="incremental"))
