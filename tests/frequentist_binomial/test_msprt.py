"""Tests for the mSPRT (Mixture Sequential Probability Ratio Test)."""

import numpy as np
import pytest

from ab_test.frequentist_binomial.confidence_intervals import confidence_interval
from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.msprt import msprt_critical_value, msprt_test, plot_msprt_over_time
from ab_test.frequentist_binomial.stats_tests import ab_test


class TestMsprtTest:

    @staticmethod
    def test_msprt_returns_float():
        trials = [1000, 1000]
        successes = [100, 110]
        p_value = msprt_test(trials, successes)
        assert isinstance(p_value, float)
        assert 0 < p_value <= 1

    @staticmethod
    def test_msprt_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        p1 = msprt_test(trials, successes, lift="absolute")
        p2 = msprt_test(list(reversed(trials)), list(reversed(successes)), lift="absolute")
        assert p1 == pytest.approx(p2)

    @staticmethod
    def test_msprt_strong_effect():
        trials = [1000, 1000]
        successes = [100, 200]
        p_value = msprt_test(trials, successes)
        assert p_value < 0.01

    @staticmethod
    def test_msprt_no_effect():
        trials = [1000, 1000]
        successes = [100, 100]
        p_value = msprt_test(trials, successes)
        assert p_value == pytest.approx(1.0)

    @staticmethod
    def test_msprt_absolute_lift():
        trials = [5000, 5000]
        successes = [500, 600]
        p_value = msprt_test(trials, successes, lift="absolute")
        assert isinstance(p_value, float)
        assert 0 < p_value < 1

    @staticmethod
    def test_msprt_relative_lift():
        trials = [5000, 5000]
        successes = [500, 600]
        p_value = msprt_test(trials, successes, lift="relative")
        assert isinstance(p_value, float)
        assert 0 < p_value < 1

    @staticmethod
    def test_msprt_nonzero_null_absolute():
        trials = [5000, 5000]
        successes = [500, 600]
        p_at_zero = msprt_test(trials, successes, null_lift=0.0, lift="absolute")
        p_at_true = msprt_test(trials, successes, null_lift=0.02, lift="absolute")
        assert p_at_true > p_at_zero

    @staticmethod
    def test_msprt_nonzero_null_relative():
        trials = [5000, 5000]
        successes = [500, 600]
        p_at_zero = msprt_test(trials, successes, null_lift=0.0, lift="relative")
        p_at_true = msprt_test(trials, successes, null_lift=0.2, lift="relative")
        assert p_at_true > p_at_zero

    @staticmethod
    def test_msprt_custom_tau():
        trials = [5000, 5000]
        successes = [500, 600]
        p_default = msprt_test(trials, successes)
        p_custom = msprt_test(trials, successes, tau=0.05)
        assert p_default != pytest.approx(p_custom)

    @staticmethod
    def test_msprt_crit():
        trials = [1000, 1000]
        successes = [100, 200]
        crit = msprt_critical_value(0.05)
        result = msprt_test(trials, successes, crit=crit)
        assert isinstance(result, bool)
        assert result is True

    @staticmethod
    def test_msprt_crit_not_significant():
        trials = [1000, 1000]
        successes = [100, 100]
        crit = msprt_critical_value(0.05)
        result = msprt_test(trials, successes, crit=crit)
        assert result is False

    @staticmethod
    def test_msprt_critical_value():
        assert msprt_critical_value(0.05) == pytest.approx(20.0)
        assert msprt_critical_value(0.01) == pytest.approx(100.0)

    @staticmethod
    def test_msprt_edge_zero_successes():
        trials = [100, 100]
        successes = [0, 0]
        p_value = msprt_test(trials, successes)
        assert isinstance(p_value, float)
        assert p_value == pytest.approx(1.0)

    @staticmethod
    def test_msprt_edge_one_success():
        trials = [2, 2]
        successes = [0, 1]
        p_value = msprt_test(trials, successes)
        assert isinstance(p_value, float)
        assert 0 < p_value <= 1

    @staticmethod
    def test_msprt_dispatcher():
        trials = [1000, 1000]
        successes = [100, 110]
        p_direct = msprt_test(trials, successes)
        p_dispatch = ab_test(trials, successes, method="msprt")
        assert p_direct == pytest.approx(p_dispatch)

    @staticmethod
    def test_msprt_confidence_interval():
        trials = [5000, 5000]
        successes = [500, 600]
        lb, ub = confidence_interval(trials, successes, test=msprt_test, alpha=0.05, lift="relative")
        assert lb < ub
        observed_relative = (600 / 5000 - 500 / 5000) / (500 / 5000)
        assert lb < observed_relative < ub

    @staticmethod
    def test_msprt_confidence_interval_absolute():
        trials = [5000, 5000]
        successes = [500, 600]
        lb, ub = confidence_interval(trials, successes, test=msprt_test, alpha=0.05, lift="absolute")
        assert lb < ub
        observed_absolute = 600 / 5000 - 500 / 5000
        assert lb < observed_absolute < ub

    @staticmethod
    def test_msprt_more_conservative_than_fixed():
        trials = [1000, 1000]
        successes = [100, 120]
        p_msprt = msprt_test(trials, successes)
        p_fixed = ab_test(trials, successes, method="score")
        assert p_msprt >= p_fixed


class TestMsprtContingencyTable:

    @staticmethod
    def test_analyze_msprt():
        ct = ContingencyTable("test_exp", "conversion")
        ct.add("control", 500, 5000)
        ct.add("variant", 600, 5000)
        result = ct.analyze(test_method="msprt")
        assert "conversion" in result
        assert "control" in result
        assert "variant" in result

    @staticmethod
    def test_analyze_msprt_absolute():
        ct = ContingencyTable("test_exp", "conversion")
        ct.add("control", 500, 5000)
        ct.add("variant", 600, 5000)
        result = ct.analyze(lift="absolute", test_method="msprt")
        assert "conversion" in result

    @staticmethod
    def test_analyze_msprt_with_tau():
        ct = ContingencyTable("test_exp", "conversion")
        ct.add("control", 500, 5000)
        ct.add("variant", 600, 5000)
        result = ct.analyze(test_method="msprt", tau=0.05)
        assert "conversion" in result

    @staticmethod
    def test_analyze_msprt_incremental_results():
        ct = ContingencyTable("test_exp", "conversion")
        ct.add("control", 500, 5000)
        ct.add("variant", 600, 5000)
        ct.analyze(test_method="msprt")
        assert ct.incremental_results is not None
        assert "p_value" in ct.incremental_results
        assert "ci_lower" in ct.incremental_results
        assert "ci_upper" in ct.incremental_results
        assert ct.incremental_results["ci_lower"] < ct.incremental_results["ci_upper"]


class TestMsprtAlwaysValidProperty:

    @staticmethod
    def test_type_i_error_control():
        """Under the null, the mSPRT rejects at most alpha across multiple peeks."""
        np.random.seed(42)
        alpha = 0.05
        n_simulations = 5000
        peek_sizes = [100, 200, 500, 1000]
        true_p = 0.1
        rejections = 0

        for _ in range(n_simulations):
            all_a = np.random.binomial(1, true_p, peek_sizes[-1])
            all_b = np.random.binomial(1, true_p, peek_sizes[-1])
            rejected = False
            for n in peek_sizes:
                s_a = int(all_a[:n].sum())
                s_b = int(all_b[:n].sum())
                p_value = msprt_test([n, n], [s_a, s_b], tau=0.02)
                if p_value < alpha:
                    rejected = True
                    break
            if rejected:
                rejections += 1

        rejection_rate = rejections / n_simulations
        assert rejection_rate < alpha + 0.02, (
            f"Rejection rate {rejection_rate:.3f} exceeds alpha={alpha} + margin"
        )


class TestPlotMsprtOverTime:

    @staticmethod
    def test_returns_figure():
        tables = []
        for n in [500, 1000, 2000]:
            ct = ContingencyTable("test", "conv")
            ct.add("Control", successes=int(n * 0.10), trials=n)
            ct.add("Treatment", successes=int(n * 0.12), trials=n)
            tables.append(ct)
        labels = ["Day 1", "Day 2", "Day 3"]
        import plotly.graph_objects as go

        fig = plot_msprt_over_time(tables, labels)
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_trace_count():
        tables = []
        for n in [500, 1000]:
            ct = ContingencyTable("test", "conv")
            ct.add("Control", successes=int(n * 0.10), trials=n)
            ct.add("Treatment", successes=int(n * 0.12), trials=n)
            tables.append(ct)
        fig = plot_msprt_over_time(tables, ["Day 1", "Day 2"])
        assert len(fig.data) == 3

    @staticmethod
    def test_absolute_lift():
        tables = []
        for n in [500, 1000]:
            ct = ContingencyTable("test", "conv")
            ct.add("Control", successes=int(n * 0.10), trials=n)
            ct.add("Treatment", successes=int(n * 0.12), trials=n)
            tables.append(ct)
        fig = plot_msprt_over_time(tables, ["Day 1", "Day 2"], lift="absolute")
        assert isinstance(fig.layout.yaxis.title.text, str)
        assert "absolute" in fig.layout.yaxis.title.text.lower()

    @staticmethod
    def test_custom_tau():
        tables = []
        for n in [500, 1000]:
            ct = ContingencyTable("test", "conv")
            ct.add("Control", successes=int(n * 0.10), trials=n)
            ct.add("Treatment", successes=int(n * 0.12), trials=n)
            tables.append(ct)
        fig = plot_msprt_over_time(tables, ["Day 1", "Day 2"], tau=0.05)
        assert len(fig.data) == 3

    @staticmethod
    def test_mismatched_lengths():
        ct = ContingencyTable("test", "conv")
        ct.add("Control", successes=50, trials=500)
        ct.add("Treatment", successes=60, trials=500)
        with pytest.raises(ValueError, match="same length"):
            plot_msprt_over_time([ct], ["Day 1", "Day 2"])
