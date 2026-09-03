"""Validation of multiple testing corrections via Monte Carlo simulation.

Tests the theoretical guarantees of each correction method:

1. FWER control — Bonferroni, Sidak, and Holm each limit the probability
   of at least one false rejection to ≤ alpha under the global null.
2. FDR control — Benjamini-Hochberg controls the expected false discovery
   proportion at ≤ alpha.
3. Power ordering — Holm rejects at least as many hypotheses as Bonferroni.
4. Conservativeness ordering — Sidak-adjusted p-values ≤ Bonferroni.

All tests are marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.corrections import benjamini_hochberg, bonferroni, holm, sidak


@pytest.mark.slow
class TestFWERControl:
    """Under the global null, FWER methods reject at most alpha of the time."""

    @staticmethod
    def _fwer_rate(correction_fn, n_sims=5000, m=10, alpha=0.05, seed=42):
        rng = np.random.default_rng(seed)
        false_rejections = 0
        for _ in range(n_sims):
            pvalues = rng.uniform(0, 1, m).tolist()
            adjusted = correction_fn(pvalues)
            if any(p < alpha for p in adjusted):
                false_rejections += 1
        return false_rejections / n_sims

    def test_bonferroni_fwer(self):
        rate = self._fwer_rate(bonferroni)
        assert rate <= 0.05 + 0.015, f"Bonferroni FWER {rate:.3f} exceeds threshold"

    def test_sidak_fwer(self):
        rate = self._fwer_rate(sidak)
        assert rate <= 0.05 + 0.015, f"Sidak FWER {rate:.3f} exceeds threshold"

    def test_holm_fwer(self):
        rate = self._fwer_rate(holm)
        assert rate <= 0.05 + 0.015, f"Holm FWER {rate:.3f} exceeds threshold"


@pytest.mark.slow
class TestFDRControl:
    """Benjamini-Hochberg controls the expected false discovery proportion."""

    @staticmethod
    def test_bh_fdr():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 5000
        m0 = 8  # true nulls
        m1 = 2  # true alternatives

        fdps = []
        for _ in range(n_sims):
            null_pvals = rng.uniform(0, 1, m0).tolist()
            alt_pvals = rng.beta(1, 20, m1).tolist()
            pvalues = null_pvals + alt_pvals

            adjusted = benjamini_hochberg(pvalues)
            rejections = [i for i, p in enumerate(adjusted) if p < alpha]

            if rejections:
                false_rejections = sum(1 for i in rejections if i < m0)
                fdps.append(false_rejections / len(rejections))
            else:
                fdps.append(0.0)

        mean_fdp = np.mean(fdps)
        assert mean_fdp <= alpha + 0.02, f"BH mean FDP {mean_fdp:.3f} exceeds {alpha + 0.02}"


@pytest.mark.slow
class TestHolmMorePowerfulThanBonferroni:
    """Under a mixture of nulls and alternatives, Holm rejects at least as many hypotheses as Bonferroni."""

    @staticmethod
    def test_holm_rejects_at_least_as_many():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 2000
        m0 = 7
        m1 = 3

        bonf_rejections = []
        holm_rejections = []

        for _ in range(n_sims):
            null_pvals = rng.uniform(0, 1, m0).tolist()
            alt_pvals = rng.beta(1, 30, m1).tolist()
            pvalues = null_pvals + alt_pvals

            bonf_adj = bonferroni(pvalues)
            holm_adj = holm(pvalues)

            bonf_rejections.append(sum(1 for p in bonf_adj if p < alpha))
            holm_rejections.append(sum(1 for p in holm_adj if p < alpha))

        assert np.mean(holm_rejections) >= np.mean(bonf_rejections), (
            f"Holm mean rejections ({np.mean(holm_rejections):.2f}) < Bonferroni ({np.mean(bonf_rejections):.2f})"
        )


@pytest.mark.slow
class TestSidakLessConservativeThanBonferroni:
    """Sidak-adjusted p-values should be ≤ Bonferroni-adjusted p-values for independent tests."""

    @staticmethod
    def test_sidak_pvalues_leq_bonferroni():
        rng = np.random.default_rng(42)
        pvalues = rng.uniform(0, 1, 20).tolist()
        adj_bonf = bonferroni(pvalues)
        adj_sidak = sidak(pvalues)
        for i, (b, s) in enumerate(zip(adj_bonf, adj_sidak)):
            assert s <= b + 1e-12, f"Sidak ({s}) > Bonferroni ({b}) at index {i}"
