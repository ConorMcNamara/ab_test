"""Validation of DiffInDiff and Cochran's Q against theoretical properties.

Tests verify:
1. Cochran's Q Type I error under homogeneity
2. Cochran's Q power under heterogeneity
3. Pairwise DiD confidence interval coverage
4. Holm correction controls FWER for pairwise comparisons

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff


def _make_table(name, s_c, n_c, s_t, n_t):
    ct = ContingencyTable(name, "converted")
    ct.add("Control", successes=s_c, trials=n_c)
    ct.add("Treatment", successes=s_t, trials=n_t)
    return ct


def _simulate_segment(rng, n_per_group, p_control, treatment_effect):
    """Generate one segment's data: control and treatment binomial draws."""
    s_c = int(rng.binomial(n_per_group, p_control))
    p_treat = min(p_control + treatment_effect, 0.99)
    s_t = int(rng.binomial(n_per_group, p_treat))
    # Clamp to avoid zero successes for relative lift
    s_c = max(s_c, 1)
    s_t = max(s_t, 1)
    return s_c, n_per_group, s_t, n_per_group


@pytest.mark.slow
class TestCochransQTypeIError:
    """Under homogeneous treatment effects, Q should reject at approximately alpha."""

    @staticmethod
    def test_type_i_error_three_segments():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 1000
        n_per_group = 500
        baseline = 0.15
        true_effect = 0.03  # same for all segments
        rejections = 0

        for _ in range(n_sims):
            tables = []
            for i in range(3):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, n_per_group, baseline, true_effect)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = DiffInDiff(*tables)
            did.analyze(lift="absolute", alpha=alpha)
            if did.heterogeneity_results["Q_pvalue"] < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert rate < alpha + 0.02, f"Type I error {rate:.3f} exceeds {alpha} + 0.02"

    @staticmethod
    def test_type_i_error_five_segments():
        """More segments means more df for Q, but Type I error should still hold."""
        rng = np.random.default_rng(99)
        alpha = 0.05
        n_sims = 1000
        n_per_group = 500
        baseline = 0.10
        true_effect = 0.02
        rejections = 0

        for _ in range(n_sims):
            tables = []
            for i in range(5):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, n_per_group, baseline, true_effect)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = DiffInDiff(*tables)
            did.analyze(lift="absolute", alpha=alpha)
            if did.heterogeneity_results["Q_pvalue"] < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert rate < alpha + 0.02, f"Type I error {rate:.3f} exceeds {alpha} + 0.02"


@pytest.mark.slow
class TestCochransQPower:
    """Under true heterogeneity, Q should reject with high probability."""

    @staticmethod
    def test_power_increases_with_heterogeneity():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 500
        n_per_group = 500
        baseline = 0.15
        # Three segments with genuinely different effects
        effects = [0.00, 0.05, 0.10]
        rejections = 0

        for _ in range(n_sims):
            tables = []
            for i, eff in enumerate(effects):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, n_per_group, baseline, eff)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = DiffInDiff(*tables)
            did.analyze(lift="absolute", alpha=alpha)
            if did.heterogeneity_results["Q_pvalue"] < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert rate > 0.50, f"Power {rate:.3f} too low — expected > 0.50 with heterogeneous effects"


@pytest.mark.slow
class TestPairwiseDiDCoverage:
    """Pairwise DiD CIs should cover the true difference-in-differences."""

    @staticmethod
    def test_absolute_lift_coverage():
        """CI for the difference in absolute treatment effects between two segments."""
        rng = np.random.default_rng(42)
        n_sims = 500
        n_per_group = 500
        baseline = 0.15
        alpha = 0.05
        # Two segments with different true effects
        effect_a = 0.05
        effect_b = 0.02
        true_did = effect_a - effect_b  # 0.03
        covers = 0

        for _ in range(n_sims):
            s_c_a, n_c_a, s_t_a, n_t_a = _simulate_segment(rng, n_per_group, baseline, effect_a)
            s_c_b, n_c_b, s_t_b, n_t_b = _simulate_segment(rng, n_per_group, baseline, effect_b)

            t_a = _make_table("SegA", s_c_a, n_c_a, s_t_a, n_t_a)
            t_b = _make_table("SegB", s_c_b, n_c_b, s_t_b, n_t_b)

            did = DiffInDiff(t_a, t_b)
            did.analyze(lift="absolute", alpha=alpha)

            pw = did.pairwise_results[0]
            if pw["ci_lower"] <= true_did <= pw["ci_upper"]:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.92, f"Coverage {coverage:.3f} below 0.92"

    @staticmethod
    def test_relative_lift_coverage():
        """CI for the difference in relative treatment effects (log-RR scale)."""
        rng = np.random.default_rng(123)
        n_sims = 500
        n_per_group = 600
        baseline = 0.15
        alpha = 0.05
        effect_a = 0.06
        effect_b = 0.02
        # True relative lifts
        rr_a = (baseline + effect_a) / baseline
        rr_b = (baseline + effect_b) / baseline
        # True DiD on relative scale: exp(log_rr_a - log_rr_b) - 1
        true_did_relative = rr_a / rr_b - 1
        covers = 0

        for _ in range(n_sims):
            s_c_a, n_c_a, s_t_a, n_t_a = _simulate_segment(rng, n_per_group, baseline, effect_a)
            s_c_b, n_c_b, s_t_b, n_t_b = _simulate_segment(rng, n_per_group, baseline, effect_b)

            t_a = _make_table("SegA", s_c_a, n_c_a, s_t_a, n_t_a)
            t_b = _make_table("SegB", s_c_b, n_c_b, s_t_b, n_t_b)

            did = DiffInDiff(t_a, t_b)
            did.analyze(lift="relative", alpha=alpha)

            pw = did.pairwise_results[0]
            if pw["ci_lower"] <= true_did_relative <= pw["ci_upper"]:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.92, f"Coverage {coverage:.3f} below 0.92"


@pytest.mark.slow
class TestHolmFWERControl:
    """Under homogeneity, the rate of any Holm-corrected rejection should be ≤ alpha."""

    @staticmethod
    def test_fwer_under_null():
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_sims = 1000
        n_per_group = 500
        baseline = 0.15
        true_effect = 0.03  # same for all 4 segments
        any_rejection = 0

        for _ in range(n_sims):
            tables = []
            for i in range(4):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, n_per_group, baseline, true_effect)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = DiffInDiff(*tables)
            did.analyze(lift="absolute", alpha=alpha, correction="holm")

            if any(pw["adjusted_pvalue"] < alpha for pw in did.pairwise_results):
                any_rejection += 1

        fwer = any_rejection / n_sims
        assert fwer < alpha + 0.02, f"FWER {fwer:.3f} exceeds {alpha} + 0.02"
