"""Testing the shared binary-search confidence interval helper."""

import pytest
import scipy.stats as ss

from ab_test._binary_search import binary_search_interval


def _z_pvalue(estimate: float, se: float):
    return lambda d: float(2 * ss.norm.sf(abs(estimate - d) / se))


class TestBinarySearchInterval:
    @staticmethod
    def test_recovers_wald_interval():
        pvalue = _z_pvalue(estimate=2.0, se=0.5)
        lb, ub = binary_search_interval(pvalue, (1.99, 2.0), (2.0, 2.01), alpha=0.05)
        z = ss.norm.isf(0.025)
        assert lb == pytest.approx(2.0 - z * 0.5, abs=1e-5)
        assert ub == pytest.approx(2.0 + z * 0.5, abs=1e-5)

    @staticmethod
    def test_bound_beyond_limit_returns_none():
        pvalue = _z_pvalue(estimate=0.0, se=10.0)
        lb, ub = binary_search_interval(
            pvalue, (-0.01, 0.0), (0.0, 0.01), alpha=0.05, lower_limit=-1.0, upper_limit=1.0
        )
        assert lb is None
        assert ub is None

    @staticmethod
    def test_skip_upper_search():
        pvalue = _z_pvalue(estimate=2.0, se=0.5)
        lb, ub = binary_search_interval(pvalue, (1.99, 2.0), (2.0, 2.01), search_upper=False)
        assert lb is not None
        assert ub is None

    @staticmethod
    def test_unbounded_search_finds_large_bounds():
        pvalue = _z_pvalue(estimate=150.0, se=5.0)
        lb, ub = binary_search_interval(pvalue, (149.99, 150.0), (150.0, 150.01))
        z = ss.norm.isf(0.025)
        assert lb == pytest.approx(150.0 - z * 5.0, abs=1e-4)
        assert ub == pytest.approx(150.0 + z * 5.0, abs=1e-4)

    @staticmethod
    def test_narrower_with_larger_alpha():
        pvalue = _z_pvalue(estimate=2.0, se=0.5)
        lb_95, ub_95 = binary_search_interval(pvalue, (1.99, 2.0), (2.0, 2.01), alpha=0.05)
        lb_90, ub_90 = binary_search_interval(pvalue, (1.99, 2.0), (2.0, 2.01), alpha=0.10)
        assert lb_90 > lb_95
        assert ub_90 < ub_95


if __name__ == "__main__":
    pytest.main()
