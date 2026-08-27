"""Tests for multiple hypothesis testing corrections.

Reference values are from R's p.adjust() unless noted otherwise.
"""

import numpy as np
import pytest

from ab_test.corrections import (
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    holm,
    sidak,
)


PVALUES = [0.01, 0.04, 0.03, 0.005]


class TestBonferroni:
    @staticmethod
    def test_known_values():
        # R: p.adjust(c(0.01, 0.04, 0.03, 0.005), method="bonferroni")
        # [1] 0.04 0.16 0.12 0.02
        result = bonferroni(PVALUES)
        expected = [0.04, 0.16, 0.12, 0.02]
        np.testing.assert_allclose(result, expected)

    @staticmethod
    def test_capped_at_one():
        result = bonferroni([0.5, 0.6])
        assert all(p <= 1.0 for p in result)

    @staticmethod
    def test_single_pvalue():
        result = bonferroni([0.03])
        np.testing.assert_allclose(result, [0.03])

    @staticmethod
    def test_preserves_order():
        result = bonferroni([0.05, 0.01, 0.10])
        assert result[1] < result[0] < result[2]


class TestSidak:
    @staticmethod
    def test_known_values():
        # 1 - (1 - p)^m for m=4
        result = sidak(PVALUES)
        expected = [1 - (1 - p) ** 4 for p in PVALUES]
        np.testing.assert_allclose(result, expected)

    @staticmethod
    def test_less_conservative_than_bonferroni():
        adj_bonf = bonferroni(PVALUES)
        adj_sidak = sidak(PVALUES)
        for b, s in zip(adj_bonf, adj_sidak):
            assert s <= b

    @staticmethod
    def test_capped_at_one():
        result = sidak([0.5, 0.6, 0.7])
        assert all(p <= 1.0 for p in result)


class TestHolm:
    @staticmethod
    def test_known_values():
        # R: p.adjust(c(0.01, 0.04, 0.03, 0.005), method="holm")
        # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        # Cummax: 0.02, 0.03, 0.06, 0.06
        result = holm(PVALUES)
        expected = [0.03, 0.06, 0.06, 0.02]
        np.testing.assert_allclose(result, expected)

    @staticmethod
    def test_more_powerful_than_bonferroni():
        adj_bonf = bonferroni(PVALUES)
        adj_holm = holm(PVALUES)
        for b, h in zip(adj_bonf, adj_holm):
            assert h <= b

    @staticmethod
    def test_monotonicity():
        result = holm([0.001, 0.002, 0.003, 0.004])
        sorted_result = sorted(result)
        assert result == sorted_result or all(
            result[i] <= result[j]
            for i, j in zip(
                sorted(range(4), key=lambda k: [0.001, 0.002, 0.003, 0.004][k]),
                sorted(range(4), key=lambda k: [0.001, 0.002, 0.003, 0.004][k])[1:],
            )
        )

    @staticmethod
    def test_single_pvalue():
        result = holm([0.03])
        np.testing.assert_allclose(result, [0.03])


class TestBenjaminiHochberg:
    @staticmethod
    def test_known_values():
        # R: p.adjust(c(0.01, 0.04, 0.03, 0.005), method="BH")
        # [1] 0.02 0.04 0.04 0.02
        result = benjamini_hochberg(PVALUES)
        expected = [0.02, 0.04, 0.04, 0.02]
        np.testing.assert_allclose(result, expected)

    @staticmethod
    def test_less_conservative_than_holm():
        adj_holm = holm(PVALUES)
        adj_bh = benjamini_hochberg(PVALUES)
        for h, b in zip(adj_holm, adj_bh):
            assert b <= h + 1e-10

    @staticmethod
    def test_capped_at_one():
        result = benjamini_hochberg([0.5, 0.6, 0.7, 0.8])
        assert all(p <= 1.0 for p in result)

    @staticmethod
    def test_single_pvalue():
        result = benjamini_hochberg([0.03])
        np.testing.assert_allclose(result, [0.03])

    @staticmethod
    def test_many_pvalues():
        # R: p.adjust(c(0.001, 0.01, 0.05, 0.10, 0.50), method="BH")
        # [1] 0.005 0.025 0.08333 0.125 0.500
        pvals = [0.001, 0.01, 0.05, 0.10, 0.50]
        result = benjamini_hochberg(pvals)
        expected = [0.005, 0.025, 5 / 60, 0.125, 0.50]
        np.testing.assert_allclose(result, expected)


class TestAdjustPvalues:
    @staticmethod
    def test_dispatcher_bonferroni():
        result = adjust_pvalues(PVALUES, method="bonferroni")
        assert result == bonferroni(PVALUES)

    @staticmethod
    def test_dispatcher_holm():
        result = adjust_pvalues(PVALUES, method="holm")
        assert result == holm(PVALUES)

    @staticmethod
    def test_dispatcher_bh_alias():
        result_bh = adjust_pvalues(PVALUES, method="bh")
        result_fdr = adjust_pvalues(PVALUES, method="fdr")
        result_full = adjust_pvalues(PVALUES, method="benjamini_hochberg")
        assert result_bh == result_full
        assert result_fdr == result_full

    @staticmethod
    def test_dispatcher_sidak():
        result = adjust_pvalues(PVALUES, method="sidak")
        assert result == sidak(PVALUES)

    @staticmethod
    def test_default_is_holm():
        result = adjust_pvalues(PVALUES)
        assert result == holm(PVALUES)

    @staticmethod
    def test_unknown_method_raises():
        with pytest.raises(ValueError, match="Unknown method"):
            adjust_pvalues(PVALUES, method="unknown")

    @staticmethod
    def test_case_insensitive():
        result = adjust_pvalues(PVALUES, method="Bonferroni")
        assert result == bonferroni(PVALUES)

    @staticmethod
    def test_hyphen_alias():
        result = adjust_pvalues(PVALUES, method="benjamini-hochberg")
        assert result == benjamini_hochberg(PVALUES)


class TestEdgeCases:
    @staticmethod
    def test_all_significant():
        result = holm([0.001, 0.001, 0.001])
        assert all(p < 0.01 for p in result)

    @staticmethod
    def test_none_significant():
        result = bonferroni([0.5, 0.6, 0.7])
        assert all(p >= 0.5 for p in result)

    @staticmethod
    def test_identical_pvalues():
        result = holm([0.05, 0.05, 0.05])
        np.testing.assert_allclose(result, [0.15, 0.15, 0.15])
        result_bh = benjamini_hochberg([0.05, 0.05, 0.05])
        np.testing.assert_allclose(result_bh, [0.05, 0.05, 0.05])

    @staticmethod
    def test_two_pvalues():
        result = bonferroni([0.03, 0.04])
        np.testing.assert_allclose(result, [0.06, 0.08])
