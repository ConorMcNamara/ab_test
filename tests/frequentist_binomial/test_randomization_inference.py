"""Tests for randomization inference module."""

import math

import pytest

from ab_test.frequentist_binomial.randomization_inference import (
    cluster_randomization_test,
    randomization_test,
)
from ab_test.frequentist_binomial.stats_tests import ab_test, fisher_test


class TestRandomizationTest:
    @staticmethod
    def test_returns_float():
        p = randomization_test([1000, 1000], [100, 110], seed=0)
        assert isinstance(p, float)
        assert 0 < p <= 1

    @staticmethod
    def test_close_to_fisher():
        trials = [1000, 1000]
        successes = [100, 130]
        fisher_p = fisher_test(trials, successes)
        ri_p = randomization_test(trials, successes, n_permutations=100_000, seed=42)
        assert ri_p == pytest.approx(fisher_p, abs=0.01)

    @staticmethod
    def test_seed_reproducibility():
        args = ([1000, 1000], [100, 120])
        p1 = randomization_test(*args, seed=42)
        p2 = randomization_test(*args, seed=42)
        p3 = randomization_test(*args, seed=99)
        assert p1 == p2
        assert p1 != p3

    @staticmethod
    def test_strong_effect():
        p = randomization_test([1000, 1000], [100, 200], seed=0)
        assert p < 0.01

    @staticmethod
    def test_no_effect():
        p = randomization_test([1000, 1000], [100, 100], seed=0)
        assert p > 0.5

    @staticmethod
    def test_crit_significant():
        result = randomization_test([1000, 1000], [100, 200], crit=0.05, seed=0)
        assert result is True

    @staticmethod
    def test_crit_not_significant():
        result = randomization_test([1000, 1000], [100, 103], crit=0.05, seed=0)
        assert result is False

    @staticmethod
    def test_crit_returns_bool():
        result = randomization_test([1000, 1000], [100, 120], crit=0.05, seed=0)
        assert isinstance(result, bool)

    @staticmethod
    def test_nonzero_null_lift_raises():
        with pytest.raises(ValueError, match="null_lift=0"):
            randomization_test([1000, 1000], [100, 120], null_lift=0.1, lift="absolute")

    @staticmethod
    def test_zero_null_relative_ok():
        p = randomization_test([1000, 1000], [100, 120], null_lift=0.0, lift="relative", seed=0)
        assert isinstance(p, float)

    @staticmethod
    def test_zero_successes_both():
        p = randomization_test([1000, 1000], [0, 0], seed=0)
        assert p == pytest.approx(1.0, abs=0.01)

    @staticmethod
    def test_dispatcher():
        p = ab_test([1000, 1000], [100, 130], method="randomization")
        assert isinstance(p, float)
        assert 0 < p < 1

    @staticmethod
    def test_pvalue_never_zero():
        p = randomization_test([100, 100], [0, 100], seed=0, n_permutations=1000)
        assert p > 0


class TestClusterRandomizationTest:
    @staticmethod
    def test_returns_float():
        p = cluster_randomization_test(
            [10, 8, 12], [100, 100, 100],
            [15, 18, 20], [100, 100, 100],
            seed=0,
        )
        assert isinstance(p, float)
        assert 0 < p <= 1

    @staticmethod
    def test_exact_matches_monte_carlo():
        s_c = [10, 8, 12]
        m_c = [100, 100, 100]
        s_t = [15, 18, 20]
        m_t = [100, 100, 100]
        exact_p = cluster_randomization_test(s_c, m_c, s_t, m_t, exact=True)
        mc_p = cluster_randomization_test(s_c, m_c, s_t, m_t, n_permutations=100_000, seed=42)
        assert mc_p == pytest.approx(exact_p, abs=0.02)

    @staticmethod
    def test_seed_reproducibility():
        args = ([10, 8, 12], [100, 100, 100], [15, 18, 20], [100, 100, 100])
        p1 = cluster_randomization_test(*args, seed=42)
        p2 = cluster_randomization_test(*args, seed=42)
        p3 = cluster_randomization_test(*args, seed=99)
        assert p1 == p2
        assert p1 != p3

    @staticmethod
    def test_clear_signal():
        s_c = [5, 6, 4, 7, 5]
        m_c = [100, 100, 100, 100, 100]
        s_t = [30, 35, 28, 32, 33]
        m_t = [100, 100, 100, 100, 100]
        p = cluster_randomization_test(s_c, m_c, s_t, m_t, seed=0)
        assert p < 0.05

    @staticmethod
    def test_no_signal():
        s_c = [10, 11, 9, 12, 10]
        m_c = [100, 100, 100, 100, 100]
        s_t = [10, 12, 11, 9, 10]
        m_t = [100, 100, 100, 100, 100]
        p = cluster_randomization_test(s_c, m_c, s_t, m_t, seed=0)
        assert p > 0.5

    @staticmethod
    def test_exact_known_outcome():
        s_c = [0, 0]
        m_c = [100, 100]
        s_t = [50, 50]
        m_t = [100, 100]
        p = cluster_randomization_test(s_c, m_c, s_t, m_t, exact=True)
        assert p == pytest.approx(1 / 3, abs=1e-10)

    @staticmethod
    def test_exact_too_many_clusters_raises():
        s = list(range(1, 16))
        m = [100] * 15
        with pytest.raises(ValueError, match="Monte Carlo"):
            cluster_randomization_test(s, m, s, m, exact=True)

    @staticmethod
    def test_insufficient_clusters_ctrl():
        with pytest.raises(ValueError, match="Control arm has 1"):
            cluster_randomization_test([10], [100], [10, 12], [100, 100])

    @staticmethod
    def test_insufficient_clusters_treat():
        with pytest.raises(ValueError, match="Treatment arm has 1"):
            cluster_randomization_test([10, 12], [100, 100], [10], [100])


class TestContingencyTableRandomization:
    @staticmethod
    def _make_ct():
        from ab_test.frequentist_binomial.contingency import ContingencyTable

        ct = ContingencyTable("test_exp", "conversions")
        ct.add("control", 100, 1000).add("treatment", 130, 1000)
        return ct

    def test_analyze_returns_string(self):
        ct = self._make_ct()
        result = ct.analyze(test_method="randomization", seed=42)
        assert isinstance(result, str)

    def test_analyze_ci_is_inf(self):
        ct = self._make_ct()
        ct.analyze(test_method="randomization", seed=42)
        assert ct.incremental_results["ci_lower"] == -math.inf
        assert ct.incremental_results["ci_upper"] == math.inf

    def test_analyze_pvalue_is_float(self):
        ct = self._make_ct()
        ct.analyze(test_method="randomization", seed=42)
        p = ct.incremental_results["p_value"]
        assert isinstance(p, float)
        assert 0 < p < 1

    def test_analyze_seed_reproducibility(self):
        ct1 = self._make_ct()
        ct1.analyze(test_method="randomization", seed=42)

        ct2 = self._make_ct()
        ct2.analyze(test_method="randomization", seed=42)

        assert ct1.incremental_results["p_value"] == ct2.incremental_results["p_value"]


class TestClusterRandomizedTrialRandomization:
    @staticmethod
    def _build_crt():
        from ab_test.frequentist_binomial.cluster import ClusterRandomizedTrial

        crt = ClusterRandomizedTrial(name="Test CRT")
        for i, s in enumerate([10, 8, 12, 9, 11]):
            crt.add(f"ctrl_{i}", s, 100, group="control")
        for i, s in enumerate([20, 22, 18, 25, 19]):
            crt.add(f"treat_{i}", s, 100, group="treatment")
        return crt

    def test_analyze_returns_string(self):
        crt = self._build_crt()
        result = crt.analyze(method="randomization", seed=42)
        assert isinstance(result, str)
        assert "RI" in result

    def test_summary_has_method(self):
        crt = self._build_crt()
        crt.analyze(method="randomization", seed=42)
        s = crt.summary()
        assert s["method"] == "randomization"

    def test_welch_summary_has_method(self):
        crt = self._build_crt()
        crt.analyze(method="welch")
        s = crt.summary()
        assert s["method"] == "welch"
        assert "t_stat" in s
        assert "welch_df" in s

    def test_ri_summary_no_welch_fields(self):
        crt = self._build_crt()
        crt.analyze(method="randomization", seed=42)
        s = crt.summary()
        assert "t_stat" not in s
        assert "welch_df" not in s

    def test_analyze_exact(self):
        from ab_test.frequentist_binomial.cluster import ClusterRandomizedTrial

        crt = ClusterRandomizedTrial(name="Small CRT")
        for i, s in enumerate([10, 8, 12]):
            crt.add(f"ctrl_{i}", s, 100, group="control")
        for i, s in enumerate([20, 22, 18]):
            crt.add(f"treat_{i}", s, 100, group="treatment")
        result = crt.analyze(method="randomization", exact=True)
        assert isinstance(result, str)

    def test_default_is_welch(self):
        crt = self._build_crt()
        result = crt.analyze()
        assert "Welch" in result

    def test_invalid_method_raises(self):
        crt = self._build_crt()
        with pytest.raises(ValueError, match="method must be one of"):
            crt.analyze(method="unknown")


if __name__ == "__main__":
    pytest.main()
