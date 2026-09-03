"""Validation of Bayesian stratified analysis against theoretical properties.

Tests that the inverse-variance weighted posterior pooling in
:class:`BayesianStratifiedContingencyTable` produces calibrated
estimates: unbiased pooled means, correct credible interval coverage,
and meaningful heterogeneity detection via posterior tau.

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable


def _build_table(strata_data):
    """Build a BayesianStratifiedContingencyTable from generated data.

    Parameters
    ----------
    strata_data : list of dict
        Each dict has keys: stratum, s_ctrl, n_ctrl, s_treat, n_treat
    """
    table = BayesianStratifiedContingencyTable(name="sim", metric_name="cvr")
    for s in strata_data:
        table.add("Control", successes=s["s_ctrl"], trials=s["n_ctrl"], alpha=1.0, beta=1.0, stratum=s["stratum"])
        table.add("Treatment", successes=s["s_treat"], trials=s["n_treat"], alpha=1.0, beta=1.0, stratum=s["stratum"])
    return table


def _simulate_strata(rng, n_per_group, baseline_rates, treatment_effect):
    """Generate binomial data for multiple strata with a common additive effect."""
    strata_data = []
    for k, p_ctrl in enumerate(baseline_rates):
        p_treat = min(p_ctrl + treatment_effect, 0.99)
        s_ctrl = int(rng.binomial(n_per_group, p_ctrl))
        s_treat = int(rng.binomial(n_per_group, p_treat))
        strata_data.append(
            {
                "stratum": f"stratum_{k}",
                "s_ctrl": s_ctrl,
                "n_ctrl": n_per_group,
                "s_treat": s_treat,
                "n_treat": n_per_group,
            }
        )
    return strata_data


@pytest.mark.slow
class TestPooledPosteriorUnbiased:
    """Pooled posterior mean should be approximately equal to the true effect."""

    @staticmethod
    def test_absolute_lift_unbiased():
        true_effect = 0.04
        baseline_rates = [0.10, 0.15, 0.20]
        n_per_group = 500
        n_sims = 300
        lifts = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            strata_data = _simulate_strata(rng, n_per_group, baseline_rates, true_effect)
            table = _build_table(strata_data)
            table.analyze(lift="absolute", n_samples=20_000)
            lifts.append(table.pooled_results["lift"])

        mean_lift = np.mean(lifts)
        assert mean_lift == pytest.approx(true_effect, abs=0.008), (
            f"Mean pooled lift {mean_lift:.4f} too far from true effect {true_effect}"
        )


@pytest.mark.slow
class TestCredibleIntervalCoverage:
    """95% credible intervals should contain the true effect at approximately the nominal rate."""

    @staticmethod
    def test_coverage_absolute_lift():
        true_effect = 0.03
        baseline_rates = [0.10, 0.15, 0.20]
        n_per_group = 500
        n_sims = 500
        covers = 0

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            strata_data = _simulate_strata(rng, n_per_group, baseline_rates, true_effect)
            table = _build_table(strata_data)
            table.analyze(lift="absolute", confidence_level=0.95, n_samples=20_000)
            ci_lo = table.pooled_results["ci_lower"]
            ci_hi = table.pooled_results["ci_upper"]
            if ci_lo <= true_effect <= ci_hi:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.90, f"Coverage {coverage:.3f} below 0.90 threshold"
        assert coverage <= 0.99, f"Coverage {coverage:.3f} suspiciously high"


@pytest.mark.slow
class TestHeterogeneityDetection:
    """Posterior tau should distinguish homogeneous from heterogeneous strata."""

    @staticmethod
    def test_tau_small_under_homogeneity():
        """When all strata share the same effect, tau should be small."""
        true_effect = 0.03
        baseline_rates = [0.10, 0.15, 0.20]
        n_per_group = 500
        n_sims = 200
        taus = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            strata_data = _simulate_strata(rng, n_per_group, baseline_rates, true_effect)
            table = _build_table(strata_data)
            table.analyze(lift="absolute", n_samples=20_000)
            taus.append(table.heterogeneity_results["tau_mean"])

        mean_tau = np.mean(taus)
        assert mean_tau < 0.03, f"Mean tau {mean_tau:.4f} too large under homogeneity"

    @staticmethod
    def test_tau_larger_under_heterogeneity():
        """When strata have different effects, tau should be noticeably larger."""
        baseline_rates = [0.10, 0.15, 0.20]
        effects = [0.01, 0.05, 0.10]
        n_per_group = 500
        n_sims = 200

        taus_homo = []
        taus_hetero = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)

            # Homogeneous: same effect for all strata
            strata_homo = _simulate_strata(rng, n_per_group, baseline_rates, 0.04)
            table_homo = _build_table(strata_homo)
            table_homo.analyze(lift="absolute", n_samples=20_000)
            taus_homo.append(table_homo.heterogeneity_results["tau_mean"])

            # Heterogeneous: different effects per stratum
            strata_hetero = []
            for k, (p_ctrl, eff) in enumerate(zip(baseline_rates, effects)):
                p_treat = min(p_ctrl + eff, 0.99)
                s_ctrl = int(rng.binomial(n_per_group, p_ctrl))
                s_treat = int(rng.binomial(n_per_group, p_treat))
                strata_hetero.append(
                    {
                        "stratum": f"stratum_{k}",
                        "s_ctrl": s_ctrl,
                        "n_ctrl": n_per_group,
                        "s_treat": s_treat,
                        "n_treat": n_per_group,
                    }
                )
            table_hetero = _build_table(strata_hetero)
            table_hetero.analyze(lift="absolute", n_samples=20_000)
            taus_hetero.append(table_hetero.heterogeneity_results["tau_mean"])

        assert np.mean(taus_hetero) > np.mean(taus_homo), (
            f"Heterogeneous tau ({np.mean(taus_hetero):.4f}) should exceed homogeneous tau ({np.mean(taus_homo):.4f})"
        )


@pytest.mark.slow
class TestPooledPrecision:
    """Pooled credible interval should be narrower than any single-stratum interval."""

    @staticmethod
    def test_pooled_ci_narrower_than_strata():
        baseline_rates = [0.10, 0.15, 0.20]
        n_per_group = 500
        n_sims = 100
        pooled_narrower_count = 0

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            strata_data = _simulate_strata(rng, n_per_group, baseline_rates, 0.03)
            table = _build_table(strata_data)
            table.analyze(lift="absolute", n_samples=20_000)
            table.analyze_by_stratum(lift="absolute", n_samples=20_000)

            pooled_width = table.pooled_results["ci_upper"] - table.pooled_results["ci_lower"]

            stratum_widths = []
            for s_name, s_res in table.stratum_results.items():
                stratum_widths.append(s_res["ci_upper"] - s_res["ci_lower"])

            if pooled_width < min(stratum_widths):
                pooled_narrower_count += 1

        fraction = pooled_narrower_count / n_sims
        assert fraction >= 0.80, f"Pooled CI narrower than all strata in only {fraction:.1%} of sims (expected ≥ 80%)"
