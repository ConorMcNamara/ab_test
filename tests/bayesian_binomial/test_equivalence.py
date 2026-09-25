"""Tests for Bayesian equivalence testing (ROPE)."""

from __future__ import annotations

import pytest

from ab_test.bayesian_binomial.equivalence import bayes_equivalence_test


class TestBayesEquivalenceBasic:
    @staticmethod
    def test_returns_dict() -> None:
        result = bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.05, seed=42)
        assert isinstance(result, dict)
        assert set(result) == {"prob_equivalent", "equivalent", "prob_superior", "prob_inferior"}

    @staticmethod
    def test_equivalent_when_rates_close() -> None:
        result = bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.05, n_samples=50_000, seed=42)
        assert result["equivalent"] is True
        assert result["prob_equivalent"] > 0.95

    @staticmethod
    def test_not_equivalent_when_rates_differ() -> None:
        result = bayes_equivalence_test([100, 200], [1000, 1000], [1, 1], [1, 1], delta=0.02, n_samples=50_000, seed=42)
        assert result["equivalent"] is False
        assert result["prob_equivalent"] < 0.05

    @staticmethod
    def test_identical_rates() -> None:
        result = bayes_equivalence_test([100, 100], [1000, 1000], [1, 1], [1, 1], delta=0.05, n_samples=50_000, seed=42)
        assert result["equivalent"] is True
        assert result["prob_equivalent"] > 0.99


class TestBayesEquivalenceProbabilities:
    @staticmethod
    def test_probabilities_sum_to_one() -> None:
        result = bayes_equivalence_test([100, 120], [1000, 1000], [1, 1], [1, 1], delta=0.05, n_samples=50_000, seed=42)
        total = result["prob_equivalent"] + result["prob_superior"] + result["prob_inferior"]
        assert total == pytest.approx(1.0, abs=1e-6)

    @staticmethod
    def test_superior_when_b_much_better() -> None:
        result = bayes_equivalence_test([100, 200], [1000, 1000], [1, 1], [1, 1], delta=0.02, n_samples=50_000, seed=42)
        assert result["prob_superior"] > 0.95
        assert result["prob_inferior"] < 0.01

    @staticmethod
    def test_inferior_when_b_much_worse() -> None:
        result = bayes_equivalence_test([200, 100], [1000, 1000], [1, 1], [1, 1], delta=0.02, n_samples=50_000, seed=42)
        assert result["prob_inferior"] > 0.95
        assert result["prob_superior"] < 0.01


class TestBayesEquivalenceSeed:
    @staticmethod
    def test_seed_reproducibility() -> None:
        args = ([100, 120], [1000, 1000], [1, 1], [1, 1])
        r1 = bayes_equivalence_test(*args, delta=0.03, seed=42)
        r2 = bayes_equivalence_test(*args, delta=0.03, seed=42)
        r3 = bayes_equivalence_test(*args, delta=0.03, seed=99)
        assert r1["prob_equivalent"] == r2["prob_equivalent"]
        assert r1["prob_equivalent"] != r3["prob_equivalent"]


class TestBayesEquivalenceThreshold:
    @staticmethod
    def test_lower_threshold_easier_to_pass() -> None:
        args = ([100, 110], [1000, 1000], [1, 1], [1, 1])
        strict = bayes_equivalence_test(*args, delta=0.05, threshold=0.99, n_samples=50_000, seed=42)
        lenient = bayes_equivalence_test(*args, delta=0.05, threshold=0.50, n_samples=50_000, seed=42)
        assert strict["prob_equivalent"] == lenient["prob_equivalent"]
        if not strict["equivalent"]:
            assert lenient["equivalent"] is True


class TestBayesEquivalenceLiftTypes:
    @staticmethod
    def test_relative_lift() -> None:
        result = bayes_equivalence_test(
            [100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.30, lift="relative", n_samples=50_000, seed=42
        )
        assert result["equivalent"] is True

    @staticmethod
    def test_incremental_lift() -> None:
        result = bayes_equivalence_test(
            [100, 102], [1000, 1000], [1, 1], [1, 1], delta=50, lift="incremental", n_samples=50_000, seed=42
        )
        assert result["equivalent"] is True

    @staticmethod
    def test_incremental_requires_trials() -> None:
        result = bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=50, lift="incremental", seed=42)
        assert isinstance(result["prob_equivalent"], float)


class TestBayesEquivalenceValidation:
    @staticmethod
    def test_negative_delta_raises() -> None:
        with pytest.raises(ValueError, match="positive"):
            bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=-0.05)

    @staticmethod
    def test_zero_delta_raises() -> None:
        with pytest.raises(ValueError, match="positive"):
            bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.0)

    @staticmethod
    def test_roas_requires_spend() -> None:
        with pytest.raises(ValueError, match="spend"):
            bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.05, lift="roas", seed=42)

    @staticmethod
    def test_revenue_requires_msrp() -> None:
        with pytest.raises(ValueError, match="msrp"):
            bayes_equivalence_test([100, 102], [1000, 1000], [1, 1], [1, 1], delta=0.05, lift="revenue", seed=42)
